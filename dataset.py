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
    ):
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
        self.window_len = int(window_len_ms * sr / 1000)
        self.dbfs_threshold = dbfs_threshold
        self.max_sources = max_sources

        self.windows = []
        for item in self.entries:
            audio_path = os.path.join(audio_dir, item["audio_filepath"])
            info = torchaudio.info(audio_path)
            n_samples = info.num_frames
            n_full_windows = n_samples // self.window_len

            # Ensure fixed-length manifest fields (mask/az/el) equal to max_sources.
            azimuths = _pad_or_trim(item.get("azimuths", []), max_sources, None)
            elevations = _pad_or_trim(item.get("elevations", []), max_sources, None)
            mask_list = _pad_or_trim(item.get("mask", []), max_sources, 0)

            # Store per-window source metadata.
            for w in range(n_full_windows):
                start = w * self.window_len
                self.windows.append(
                    {
                        "audio_path": audio_path,
                        "start": start,
                        "azimuths": azimuths,
                        "elevations": elevations,
                        "mask": mask_list,
                        "n_sources": item.get("n_sources", 0),
                    }
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        win = self.windows[idx]
        audio_path = win["audio_path"]
        start = win["start"]

        # Load one time window from audio.
        waveform, fs = torchaudio.load(audio_path, frame_offset=start, num_frames=self.window_len)
        if fs != self.sr:
            waveform = torchaudio.functional.resample(waveform, fs, self.sr)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

        # Check dBFS; if below threshold, move to the next non-silent window.
        window_dbfs = dbfs(waveform.clone().detach())
        if window_dbfs < self.dbfs_threshold:
            return self.__getitem__((idx + 1) % len(self.windows))

        # Build position labels with fixed shape: [max_sources, 3].
        az_list = win["azimuths"]  # e.g. [0, None, ...]
        el_list = win["elevations"]
        mask = torch.tensor(win["mask"], dtype=torch.float32)  # e.g. [1, 0, 0, ...], length=max_sources
        pos_labels = torch.zeros((self.max_sources, 3), dtype=torch.float32)

        # Fill positions for active sources.
        for i, (az, el, m) in enumerate(zip(az_list, el_list, mask)):
            if m > 0.5 and az is not None and el is not None:
                pos_labels[i] = torch.tensor(angles_to_unitvec(az, el), dtype=torch.float32)
            # Padding slots (mask==0 or missing angle) remain [0, 0, 0].

        # Number of sources (scalar float tensor).
        n_sources = torch.tensor(float(mask.sum().item()), dtype=torch.float32)
        # Alternative: n_sources = torch.tensor(win["n_sources"], dtype=torch.float32)

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
