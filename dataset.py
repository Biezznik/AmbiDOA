import os
import json
import random
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from AmbiDOA import log_print
from datetime import datetime


class AmbiDOADataset(Dataset):
    """
    Dataset for multi-source DOA localization based on a JSON manifest:
    [
      {
        "audio_filepath": "...",
        "n_sources": 1,
        "regions": [...],
        "azimuths": [...],      # [max_sources] (float or None)
        "elevations": [...],    # [max_sources] (float or None)
        "mask": [...]           # [max_sources] (0/1)
      },
      ...
    ]
    """

    def __init__(
        self,
        manifest_path,
        audio_dir,
        sr=11025,
        window_len_ms=250,
        dbfs_threshold=-80.0,
        max_sources=12,
        subsample_pct=None,
        subsample_seed=42,
        label_mode="auto",
        window_active_frac=0.5,
        skip_inactive_windows=True,
    ):
        # label_mode: how per-window labels are derived.
        #   "auto" -> per entry: time-aware if the entry has a "sources" key
        #             (time-indexed (t, az, el, active) tracks), else file-level.
        #   "file" -> always the legacy file-level path (azimuths/elevations/mask).
        #   "time" -> always time-aware (errors if an entry lacks "sources").
        # window_active_frac: in time-aware mode a source counts as active in a
        #   window if the mean of its in-window track activity flags >= this.
        # skip_inactive_windows: in time-aware mode, do not build windows with zero
        #   active sources (the model was never trained on count=0). No effect on
        #   file-level entries (their count is fixed and > 0).
        if label_mode not in ("auto", "file", "time"):
            raise ValueError("label_mode must be 'auto', 'file', or 'time'.")
        self.label_mode = label_mode
        self.window_active_frac = float(window_active_frac)
        self.skip_inactive_windows = bool(skip_inactive_windows)
        with open(manifest_path, "r") as f:
            self.entries = json.load(f)
        # Optional manifest-level subsampling for fast debug iterations. Cuts
        # both manifest scan time (torchaudio.info per file) and dataset size
        # proportionally. Subsamples files (not windows), so split-by-file
        # logic downstream remains valid.
        self._full_manifest_size = len(self.entries)
        self.subsample_pct = None
        if subsample_pct is not None and 0 < float(subsample_pct) < 100:
            rng = random.Random(int(subsample_seed))
            n_keep = max(1, int(self._full_manifest_size * float(subsample_pct) / 100.0))
            self.entries = rng.sample(self.entries, n_keep)
            self.subsample_pct = float(subsample_pct)
        self.audio_dir = audio_dir
        self.sr = sr
        self.window_len_ms = window_len_ms
        # Canonical OUTPUT length (samples) after resampling to `sr`. Every window is
        # resampled then padded/trimmed to exactly this, so T is constant across files.
        self.target_win = int(round(window_len_ms * sr / 1000))
        self.dbfs_threshold = dbfs_threshold
        self.max_sources = max_sources

        self.windows = []
        for item in self.entries:
            audio_path = os.path.join(audio_dir, item["audio_filepath"])
            info = torchaudio.info(audio_path)
            n_samples = info.num_frames
            # Window by DURATION at the file's NATIVE sample rate, so window_len_ms is
            # the true window length regardless of the file's sr. Those native frames
            # are resampled to `sr` (-> self.target_win samples) in __getitem__.
            # (Fixes the prior bug: a target-sr frame count was read at the native
            # rate, giving an effective window of window_len_ms * sr/native_sr.)
            native_win = int(round(self.window_len_ms * info.sample_rate / 1000))
            n_full_windows = n_samples // native_win

            # Decide per-entry label source.
            if self.label_mode == "file":
                time_aware = False
            elif self.label_mode == "time":
                if "sources" not in item:
                    raise ValueError("label_mode='time' but an entry has no 'sources' field.")
                time_aware = True
            else:  # auto
                time_aware = "sources" in item

            if time_aware:
                # Parse the time-indexed tracks once per entry, then attach a compact
                # per-window label (active slots + unit DOAs) computed from the track
                # span. Windows with no active source are skipped when requested.
                tracks = _parse_source_tracks(item["sources"], max_sources)
                for w in range(n_full_windows):
                    start = w * native_win
                    t_start = start / info.sample_rate
                    t_end = (start + native_win) / info.sample_rate
                    active = _window_active_sources(
                        tracks, t_start, t_end, self.window_active_frac)
                    if self.skip_inactive_windows and not active:
                        continue
                    self.windows.append({
                        "audio_path": audio_path,
                        "start": start,
                        "native_win": native_win,
                        "active": active,   # list of (slot, [x, y, z])
                    })
                continue

            # --- File-level (legacy) entry ---
            # Ensure fixed-length manifest fields (mask/az/el) equal to max_sources.
            azimuths = _pad_or_trim(item.get("azimuths", []), max_sources, None)
            elevations = _pad_or_trim(item.get("elevations", []), max_sources, None)
            mask_list = _pad_or_trim(item.get("mask", []), max_sources, 0)

            # Store per-window source metadata.
            for w in range(n_full_windows):
                start = w * native_win
                self.windows.append(
                    {
                        "audio_path": audio_path,
                        "start": start,
                        "native_win": native_win,
                        "azimuths": azimuths,
                        "elevations": elevations,
                        "mask": mask_list,
                        "n_sources": item.get("n_sources", 0),
                    }
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Scan forward for the next non-silent window. Iterative on purpose (not
        # recursive): real recordings contain long digital-silence runs (pre/post
        # silence, gaps between events), and recursing once per skipped window
        # overflows Python's stack. Bounded to a single full pass; if every window
        # is silent we fall back to the last one loaded so a sample is still returned.
        n = len(self.windows)
        win = self.windows[idx]
        waveform = None
        for offset in range(n):
            win = self.windows[(idx + offset) % n]
            waveform, fs = torchaudio.load(win["audio_path"], frame_offset=win["start"], num_frames=win["native_win"])
            if fs != self.sr:
                waveform = torchaudio.functional.resample(waveform, fs, self.sr)
            # Resampling rounding can be +-1 sample; pad/trim to the canonical length
            # so every window has exactly target_win samples (constant T).
            L = waveform.shape[-1]
            if L < self.target_win:
                waveform = torch.nn.functional.pad(waveform, (0, self.target_win - L))
            elif L > self.target_win:
                waveform = waveform[..., : self.target_win]
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

            # Check dBFS; keep the first window at/above threshold.
            if dbfs(waveform.clone().detach()) >= self.dbfs_threshold:
                break

        # Build position labels with fixed shape: [max_sources, 3].
        mask = torch.zeros(self.max_sources, dtype=torch.float32)
        pos_labels = torch.zeros((self.max_sources, 3), dtype=torch.float32)

        if "active" in win:
            # Time-aware: window-level active sources + unit DOAs, precomputed at init.
            for slot, vec in win["active"]:
                mask[slot] = 1.0
                pos_labels[slot] = torch.tensor(vec, dtype=torch.float32)
        else:
            # File-level (legacy): fill positions for the manifest's active sources.
            az_list = win["azimuths"]  # e.g. [0, None, ...]
            el_list = win["elevations"]
            file_mask = win["mask"]    # e.g. [1, 0, 0, ...], length=max_sources
            for i, (az, el, m) in enumerate(zip(az_list, el_list, file_mask)):
                if m > 0.5 and az is not None and el is not None:
                    mask[i] = 1.0
                    pos_labels[i] = torch.tensor(angles_to_unitvec(az, el), dtype=torch.float32)
                # Padding slots (mask==0 or missing angle) remain [0, 0, 0].

        # Number of sources (scalar float tensor).
        n_sources = torch.tensor(float(mask.sum().item()), dtype=torch.float32)

        # Return: waveform, source count, activity mask, position labels.
        return waveform.clone().detach(), n_sources, mask, pos_labels


# ---------- Helper functions ----------


def angles_to_unitvec(azimuth, elevation):
    """
    Convert [azimuth, elevation] in degrees to a unit vector [x, y, z].
    """
    if azimuth is None or elevation is None:
        return [0.0, 0.0, 0.0]
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return [x, y, z]


def _parse_source_tracks(sources, max_sources):
    """Parse time-aware manifest `sources` into per-source numpy arrays.

    Each source has track rows [t_sec, az_deg, el_deg, active]. Returns a list of
    (t[N], vec[N,3], active[N]) up to max_sources, with (az, el) pre-converted to
    unit vectors so later per-window averaging is free of azimuth wrap-around.
    """
    tracks = []
    for s in sources[:max_sources]:
        tr = np.asarray(s["track"], dtype=float)
        if tr.ndim != 2 or tr.shape[1] < 4:
            raise ValueError("each source 'track' must be rows of [t, az, el, active].")
        t, az, el, active = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
        azr, elr = np.deg2rad(az), np.deg2rad(el)
        vec = np.stack([np.cos(elr) * np.cos(azr),
                        np.cos(elr) * np.sin(azr),
                        np.sin(elr)], axis=1)
        tracks.append((t, vec, active))
    return tracks


def _window_active_sources(tracks, t_start, t_end, window_active_frac):
    """Active sources for a window [t_start, t_end): list of (slot, [x, y, z]).

    A source is active if the mean of its in-window activity flags >= frac; its DOA
    is the normalized mean of its in-window active-frame unit vectors (constant for
    static sources, mean heading for moving ones). Empty windows yield [].
    """
    active = []
    for j, (t, vec, act) in enumerate(tracks):
        m = (t >= t_start) & (t < t_end)
        if not m.any():
            # Window falls between track samples: use the single nearest frame.
            k = int(np.argmin(np.abs(t - 0.5 * (t_start + t_end))))
            m = np.zeros(len(t), dtype=bool)
            m[k] = True
        if act[m].mean() < window_active_frac:
            continue
        sel = vec[m][act[m] > 0.5]
        if len(sel) == 0:
            sel = vec[m]
        mv = sel.mean(axis=0)
        mv = mv / (np.linalg.norm(mv) + 1e-9)
        active.append((j, mv.tolist()))
    return active


def unitvec_to_angles(vec):
    """
    Convert tensor [N, 3] to angles (azimuth, elevation) in degrees.
    """
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    az = torch.atan2(y, x)
    el = torch.asin(z / torch.sqrt(x**2 + y**2 + z**2))
    return torch.rad2deg(az), torch.rad2deg(el)


def dbfs(waveform):
    """
    Compute signal level in dBFS (assuming waveform range [-1, 1]).
    """
    rms = (waveform**2).mean()
    if rms == 0:
        return -120.0
    return 10 * torch.log10(rms + 1e-10)


def _pad_or_trim(seq, length, pad_value):
    """
    Pad or trim a list to a fixed length (used for mask/az/el manifest fields).
    """
    seq = list(seq)
    if len(seq) < length:
        seq = seq + [pad_value] * (length - len(seq))
    elif len(seq) > length:
        seq = seq[:length]
    return seq
