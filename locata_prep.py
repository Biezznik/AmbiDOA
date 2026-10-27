"""LOCATA -> AmbiDOA preprocessor (manifest builder + convention calibration).

Turns the LOCATA dev/eval recordings into a TIME-AWARE manifest our dataloader can
consume. Encoding is NOT done here: the 32-ch em32 wavs are encoded to 16-ch 3rd-order
ACN/SN3D ambiX externally with the official mh acoustics tool (EigenStudio/EigenUnits,
ordering=ACN, normalization=SN3D, order=3 -> 16 channels). This script reads those
already-encoded ambiX wavs plus the LOCATA geometry/VAD and emits, per recording:

    {
      "audio_filepath": "dev/task1/recording1/eigenmike/audio_array_eigenmike_ambix.wav",
      "split": "dev", "task": 1, "recording": "recording1", "array": "eigenmike",
      "sr": 48000, "num_frames": 155072, "duration_s": 3.23,
      "sources": [
        {"id": "loudspeaker1",
         "track": [[t_sec, az_deg, el_deg, active], ...]}   # the (time, position) tuples
      ]
    }

DOA is computed from geometry (angles are NOT stored in LOCATA):
    d_global = p_src(t) - p_arr(t)
    d_body   = R_arr(t)^T @ d_global          # global -> array body frame
    d_ac     = R_cal @ d_body                 # body -> em32 acoustic/ambiX frame (calib)
    (az, el) = atan2(y, x), asin(z)           # our convention (see dataset.angles_to_unitvec)

The em32 ambiX convention (EigenStudio manual R02D): azimuth CCW, 0 deg at the "mh
acoustics" logo; elevation is a COLATITUDE theta (0=top, 90=horizontal, 180=shaft).
We already emit our-convention (az CCW from +x, el from horizontal, +z up), so the
only unknown is the fixed body<->acoustic rotation R_cal between LOCATA's OptiTrack
body frame and the em32 acoustic frame. Default R_cal = I; verify/replace it with
`--calib` (intensity-vector DOA from the encoded ambiX vs geometric DOA) BEFORE
trusting any model run.

Usage:
  # Build the manifest (skips recordings whose ambiX wav isn't encoded yet, and
  # reports them so you can see encoding progress):
  python locata_prep.py --root "D:/pm/bb/Validation_Dataset" \
      --out "D:/pm/bb/Validation_Dataset/locata_manifest.json" \
      --splits dev eval --tasks 1 2 3 4

  # Just list which recordings still need encoding:
  python locata_prep.py --root "D:/pm/bb/Validation_Dataset" --list-pending

  # Convention calibration on one encoded static recording (task1):
  python locata_prep.py --root "D:/pm/bb/Validation_Dataset" \
      --calib "dev/task1/recording1/eigenmike"
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

# ACN channel indices for a 3rd-order ambiX file (we only need first order for the
# intensity check). ACN: 0=W, 1=Y, 2=Z, 3=X.
ACN_W, ACN_Y, ACN_Z, ACN_X = 0, 1, 2, 3

EXPECTED_AMBIX_CHANNELS = 16  # 3rd order -> (3+1)^2
DEFAULT_AMBIX_NAME = "audio_array_eigenmike_ambix.wav"
# Raw/source wavs we must never mistake for the encoded ambiX file.
_NON_AMBIX_WAV_SUBSTR = ("audio_source", "audio_array_eigenmike.wav")


# --------------------------------------------------------------------------- #
# LOCATA file parsing
# --------------------------------------------------------------------------- #
def _abs_seconds(day, hour, minute, second):
    """Monotonic seconds within a recording from day/hour/minute/second columns.

    year/month are ignored (a recording never spans a month boundary); day is kept
    so the value stays monotonic across a midnight rollover. Relative differences
    (what we actually use) are unaffected by the arbitrary origin.
    """
    return (((day * 24.0 + hour) * 60.0 + minute) * 60.0) + second


def _load_locata_txt(path):
    """Load a tab-separated LOCATA txt (one header row of names) as a float2d array."""
    return np.genfromtxt(path, delimiter="\t", skip_header=1, dtype=float)


def _pose_times(arr):
    """Column 0..5 = year, month, day, hour, minute, second -> seconds vector."""
    arr = np.atleast_2d(arr)
    return _abs_seconds(arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5])


def parse_source_pose(path):
    """position_source_*.txt -> (t[N], pos[N,3]).  Columns 6,7,8 = x,y,z."""
    a = np.atleast_2d(_load_locata_txt(path))
    return _pose_times(a), a[:, 6:9]


def parse_array_pose(path):
    """position_array_*.txt -> (t[N], pos[N,3], R[N,3,3]).

    Columns 6,7,8 = x,y,z; columns 12..20 = rotation_11..rotation_33 (row-major),
    the body->global rotation (its columns are the body axes in global coordinates).
    """
    a = np.atleast_2d(_load_locata_txt(path))
    t = _pose_times(a)
    pos = a[:, 6:9]
    R = a[:, 12:21].reshape(-1, 3, 3)
    return t, pos, R


def parse_audio_start(path):
    """First data row of audio_array_*_timestamps.txt -> start time in seconds."""
    a = np.atleast_2d(_load_locata_txt(path))
    return float(_pose_times(a[:1])[0])


def load_vad(path):
    """VAD_*.txt (header 'VAD', one 0/1 per audio sample) -> int8 array."""
    return np.genfromtxt(path, skip_header=1, dtype=np.int8)


# --------------------------------------------------------------------------- #
# Geometry -> DOA
# --------------------------------------------------------------------------- #
def geom_doa(p_src, p_arr, R_arr, R_cal=None):
    """Per-frame (az, el) in our convention (degrees).

    p_src, p_arr: [N,3] global positions on a shared time grid.
    R_arr:        [N,3,3] body->global rotation.
    R_cal:        optional [3,3] fixed body->acoustic calibration (default identity).
    """
    d = p_src - p_arr                          # [N,3] global
    # global -> body:  d_body = R^T d   (batched)
    d_body = np.einsum("nji,nj->ni", R_arr, d)  # R^T via swapped indices
    if R_cal is not None:
        d_body = d_body @ R_cal.T
    n = np.linalg.norm(d_body, axis=1, keepdims=True)
    d_body = d_body / np.clip(n, 1e-9, None)
    x, y, z = d_body[:, 0], d_body[:, 1], d_body[:, 2]
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return az, el


def _interp_pos(t_query, t_src, pos_src):
    """Linear-interpolate source positions onto the array time grid (per axis)."""
    out = np.empty((len(t_query), 3), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(t_query, t_src, pos_src[:, k])
    return out


# --------------------------------------------------------------------------- #
# ambiX wav resolution / validation
# --------------------------------------------------------------------------- #
def _audio_info(path):
    import torchaudio
    i = torchaudio.info(path)
    return i.num_channels, i.sample_rate, i.num_frames


def resolve_ambix(folder, ambix_name):
    """Return the encoded ambiX wav path in `folder`, or None if not present yet.

    Prefers the configured name; otherwise searches for a 16-channel wav that is
    not a raw-array or clean-source file.
    """
    cand = os.path.join(folder, ambix_name)
    if os.path.isfile(cand):
        return cand
    for w in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        base = os.path.basename(w).lower()
        if any(s in base for s in _NON_AMBIX_WAV_SUBSTR):
            continue
        try:
            ch, _, _ = _audio_info(w)
        except Exception:
            continue
        if ch == EXPECTED_AMBIX_CHANNELS:
            return w
    return None


# --------------------------------------------------------------------------- #
# Manifest building
# --------------------------------------------------------------------------- #
def _discover_sources(folder):
    """List source ids from position_source_<id>.txt files in the folder."""
    ids = []
    for p in sorted(glob.glob(os.path.join(folder, "position_source_*.txt"))):
        name = os.path.basename(p)[len("position_source_"):-len(".txt")]
        ids.append(name)
    return ids


def _vad_frame_activity(vad, t_rel, sr, vad_active_frac):
    """Per track-frame activity (0/1): VAD fraction over each frame's sample span >= frac.

    Frame span = midpoints between consecutive track-time sample centers.
    """
    idx = np.clip(np.round(t_rel * sr).astype(int), 0, len(vad) - 1)
    edges = np.empty(len(idx) + 1, dtype=int)
    if len(idx) == 1:
        edges[0], edges[1] = 0, len(vad)
    else:
        edges[1:-1] = (idx[:-1] + idx[1:]) // 2
        edges[0], edges[-1] = 0, len(vad)
    edges = np.clip(edges, 0, len(vad))
    active = np.empty(len(t_rel), dtype=np.int8)
    for k in range(len(t_rel)):
        lo, hi = edges[k], max(edges[k] + 1, edges[k + 1])
        seg = vad[lo:hi]
        active[k] = 1 if seg.size and (seg.mean() >= vad_active_frac) else 0
    return active


def _build_sources(t_arr_abs, p_arr, R_arr, t_rel, sr, source_specs, vad_active_frac, R_cal):
    """source_specs: list of (sid, src_pose_path, vad_path) -> list of {id, track}."""
    sources = []
    for sid, sp, vad_p in source_specs:
        t_src_abs, p_src = parse_source_pose(sp)
        p_src_i = _interp_pos(t_arr_abs, t_src_abs, p_src)
        az, el = geom_doa(p_src_i, p_arr, R_arr, R_cal=R_cal)
        vad = load_vad(vad_p)                  # [num_audio_samples] at audio sr
        active = _vad_frame_activity(vad, t_rel, sr, vad_active_frac)
        track = [[round(float(t_rel[k]), 6), round(float(az[k]), 3),
                  round(float(el[k]), 3), int(active[k])] for k in range(len(t_rel))]
        sources.append({"id": sid, "track": track})
    return sources


def build_entry(spec, audio_filepath, vad_active_frac, R_cal=None):
    """Build one manifest entry from a resolved recording `spec`.

    spec keys: ambix, arr_pose, ts_file, sources (list of (sid, src_pose, vad)),
    split, task, recording, array. `audio_filepath` is stored verbatim (relative
    to the dataloader's audio_dir). Returns (entry_dict, status) where status is
    'ok' or a short 'skip:<reason>'.
    """
    ch, sr, num_frames = _audio_info(spec["ambix"])
    if ch != EXPECTED_AMBIX_CHANNELS:
        return None, f"skip:ambix_has_{ch}ch_not_{EXPECTED_AMBIX_CHANNELS}"
    if not (os.path.isfile(spec["arr_pose"]) and os.path.isfile(spec["ts_file"])):
        return None, "skip:missing_array_pose_or_timestamps"
    if not spec["sources"]:
        return None, "skip:no_sources"
    for sid, sp, vad_p in spec["sources"]:
        if not (os.path.isfile(sp) and os.path.isfile(vad_p)):
            return None, f"skip:missing_src_files_for_{sid}"

    t_arr_abs, p_arr, R_arr = parse_array_pose(spec["arr_pose"])
    audio_start = parse_audio_start(spec["ts_file"])
    t_rel = t_arr_abs - audio_start            # track time, seconds from audio sample 0
    sources = _build_sources(t_arr_abs, p_arr, R_arr, t_rel, sr,
                             spec["sources"], vad_active_frac, R_cal)
    entry = {
        "audio_filepath": audio_filepath,
        "split": spec["split"], "task": spec["task"], "recording": spec["recording"],
        "array": spec["array"],
        "sr": int(sr), "num_frames": int(num_frames),
        "duration_s": round(num_frames / sr, 3),
        "n_sources_file": len(spec["sources"]),   # nominal; per-window count comes from VAD
        "sources": sources,
    }
    return entry, "ok"


def _nested_spec(folder, split, task, rec, array, ambix_name):
    """Resolve a recording spec from a nested .../recordingN/<array>/ folder.

    Returns None if the ambiX wav isn't encoded yet (pending).
    """
    ambix = resolve_ambix(folder, ambix_name)
    if ambix is None:
        return None
    sources = [(sid, os.path.join(folder, f"position_source_{sid}.txt"),
                os.path.join(folder, f"VAD_{array}_{sid}.txt"))
               for sid in _discover_sources(folder)]
    return {
        "ambix": ambix,
        "arr_pose": os.path.join(folder, f"position_array_{array}.txt"),
        "ts_file": os.path.join(folder, f"audio_array_timestamps_{array}.txt"),
        "sources": sources, "split": split, "task": task, "recording": rec, "array": array,
    }


# Flat-batch filename prefix: "{split}_task{n}_{recordingM}_"
_PREFIX_RE = re.compile(r"^(dev|eval)_task(\d+)_(recording\w+)_$", re.IGNORECASE)


def _flat_specs(flat_dir, array):
    """Resolve recording specs from a FLAT batch dir of prefixed files.

    Each ambiX wav "{prefix}audio_array_{array}_AmbiX.wav" is grouped with its
    "{prefix}position_array_*", "{prefix}audio_array_timestamps_*", and per-source
    "{prefix}position_source_<sid>" / "{prefix}VAD_{array}_<sid>" siblings.
    `audio_filepath` is stored as the bare wav basename (flat dir == audio_dir).
    """
    specs = []
    pat = os.path.join(flat_dir, f"*audio_array_{array}_AmbiX.wav")
    for w in sorted(glob.glob(pat), key=str.lower):
        base = os.path.basename(w)
        prefix = base.split(f"audio_array_{array}")[0]      # "{split}_task{n}_{rec}_"
        m = _PREFIX_RE.match(prefix)
        if m:
            split, task, rec = m.group(1).lower(), int(m.group(2)), m.group(3)
        else:
            split, task, rec = "?", -1, prefix.rstrip("_")
        src_ids = []
        for sp in sorted(glob.glob(os.path.join(flat_dir, f"{prefix}position_source_*.txt"))):
            sid = os.path.basename(sp)[len(prefix) + len("position_source_"):-len(".txt")]
            src_ids.append(sid)
        sources = [(sid, os.path.join(flat_dir, f"{prefix}position_source_{sid}.txt"),
                    os.path.join(flat_dir, f"{prefix}VAD_{array}_{sid}.txt")) for sid in src_ids]
        specs.append({
            "ambix": w,
            "arr_pose": os.path.join(flat_dir, f"{prefix}position_array_{array}.txt"),
            "ts_file": os.path.join(flat_dir, f"{prefix}audio_array_timestamps_{array}.txt"),
            "sources": sources, "split": split, "task": task, "recording": rec,
            "array": array, "prefix": prefix,
        })
    return specs


def walk_recordings(root, splits, tasks, array):
    """Yield (split, task, recording, folder) for each requested eigenmike folder."""
    for split in splits:
        for task in tasks:
            tdir = os.path.join(root, split, f"task{task}")
            if not os.path.isdir(tdir):
                continue
            for rec in sorted(os.listdir(tdir)):
                folder = os.path.join(tdir, rec, array)
                if os.path.isdir(folder):
                    yield split, task, rec, folder


def cmd_build(args, R_cal):
    """Nested mode: build from the original .../recordingN/<array>/ layout."""
    root = str(Path(args.root))
    entries, pending, skipped = [], [], []
    for split, task, rec, folder in walk_recordings(root, args.splits, args.tasks, args.array):
        tag = f"{split}/task{task}/{rec}"
        spec = _nested_spec(folder, split, task, rec, args.array, args.ambix_name)
        if spec is None:
            pending.append(tag)
            print(f"[pending] {tag}  (ambiX not encoded yet)")
            continue
        entry, status = build_entry(
            spec, os.path.relpath(spec["ambix"], root).replace(os.sep, "/"),
            args.vad_active_frac, R_cal=R_cal)
        if status == "ok":
            entries.append(entry)
            print(f"[ok]      {tag}  srcs={entry['n_sources_file']}  dur={entry['duration_s']}s")
        else:
            skipped.append((tag, status))
            print(f"[skip]    {tag}  {status}")

    if not args.list_pending:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        print(f"\nwrote {len(entries)} entries -> {args.out}")
    print(f"summary: {len(entries)} ok | {len(pending)} pending encode | {len(skipped)} skipped")
    if pending:
        print("pending encode:")
        for t in pending:
            print("   " + t)


def cmd_build_flat(args, R_cal):
    """Flat mode: build from a self-contained batch dir of prefixed wavs + txts
    (e.g. Dev-Static-Batch). audio_filepath is the bare wav basename, so point the
    dataloader's audio_dir at this same folder."""
    flat = str(Path(args.flat_dir))
    tasks = set(args.tasks) if args.tasks else None
    entries, skipped = [], []
    for spec in _flat_specs(flat, args.array):
        if tasks and spec["task"] not in tasks:
            continue
        tag = f"{spec['split']}/task{spec['task']}/{spec['recording']}"
        entry, status = build_entry(spec, os.path.basename(spec["ambix"]),
                                    args.vad_active_frac, R_cal=R_cal)
        if status == "ok":
            entries.append(entry)
            print(f"[ok]   {tag}  srcs={entry['n_sources_file']}  dur={entry['duration_s']}s")
        else:
            skipped.append((tag, status))
            print(f"[skip] {tag}  {status}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"\nwrote {len(entries)} entries -> {args.out}")
    print(f"summary: {len(entries)} ok | {len(skipped)} skipped")


# --------------------------------------------------------------------------- #
# Convention calibration: intensity-vector DOA vs geometric DOA
# --------------------------------------------------------------------------- #
def _calib_report(spec, R_cal):
    """Intensity-vector DOA (from encoded ambiX) vs geometric DOA, per source."""
    import torchaudio

    wav, sr = torchaudio.load(spec["ambix"])   # [16, T]
    wav = wav.numpy()
    W, X, Y, Z = wav[ACN_W], wav[ACN_X], wav[ACN_Y], wav[ACN_Z]

    t_arr_abs, p_arr, R_arr = parse_array_pose(spec["arr_pose"])
    audio_start = parse_audio_start(spec["ts_file"])
    t_rel = t_arr_abs - audio_start

    print(f"ambiX: {os.path.basename(spec['ambix'])}  sr={sr}  T={wav.shape[1]}  "
          f"sources={[s[0] for s in spec['sources']]}\n")

    for sid, sp, vad_p in spec["sources"]:
        t_src_abs, p_src = parse_source_pose(sp)
        p_src_i = _interp_pos(t_arr_abs, t_src_abs, p_src)
        az_g, el_g = geom_doa(p_src_i, p_arr, R_arr, R_cal=R_cal)

        vad = load_vad(vad_p)
        act = vad[:wav.shape[1]].astype(bool)
        if act.sum() < sr // 10:               # need some active audio
            print(f"[{sid}] too little active audio ({int(act.sum())} samples) — skipping")
            continue

        # Active-intensity vector over active samples -> source direction in ambiX frame.
        ix = float(np.mean(W[act] * X[act]))
        iy = float(np.mean(W[act] * Y[act]))
        iz = float(np.mean(W[act] * Z[act]))
        v = np.array([ix, iy, iz])
        v /= np.clip(np.linalg.norm(v), 1e-9, None)
        az_i = np.degrees(np.arctan2(v[1], v[0]))
        el_i = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))

        # Geometric DOA averaged over track frames spanning the active audio.
        lo, hi = act_idx_bounds(act, sr, t_rel)
        sel = slice(lo, hi + 1)
        g = _unit(np.nanmean(_angles_to_vec(az_g[sel], el_g[sel]), axis=0))
        az_gm = np.degrees(np.arctan2(g[1], g[0]))
        el_gm = np.degrees(np.arcsin(np.clip(g[2], -1, 1)))
        ang = np.degrees(np.arccos(np.clip(np.dot(v, g), -1, 1)))

        print(f"[{sid}]")
        print(f"   geometric DOA : az={az_gm:7.2f}  el={el_gm:7.2f}   vec={g.round(3)}")
        print(f"   intensity DOA : az={az_i:7.2f}  el={el_i:7.2f}   vec={v.round(3)}")
        print(f"   angular diff  : {ang:5.2f} deg   "
              f"{'OK (convention aligned)' if ang < 15 else 'MISALIGNED -> fix R_cal / axis map'}\n")


def cmd_calib(args, R_cal):
    """Nested-mode calibration: --calib is a recording folder relative to --root."""
    folder = os.path.join(str(Path(args.root)), args.calib)
    rec = os.path.basename(os.path.dirname(folder))
    spec = _nested_spec(folder, "?", -1, rec, args.array, args.ambix_name)
    if spec is None:
        raise SystemExit(f"No encoded ambiX wav found in {folder} (encode it first).")
    _calib_report(spec, R_cal)


def cmd_calib_flat(args, R_cal):
    """Flat-mode calibration: --calib-rec is a prefix like 'dev_task1_recording1'."""
    specs = {s["prefix"].rstrip("_"): s for s in _flat_specs(str(Path(args.flat_dir)), args.array)}
    spec = specs.get(args.calib_rec.rstrip("_"))
    if spec is None:
        raise SystemExit(f"'{args.calib_rec}' not found in {args.flat_dir}. "
                         f"Available: {sorted(specs)[:6]}")
    _calib_report(spec, R_cal)


def act_idx_bounds(act, sr, t_rel):
    """First/last track-frame indices overlapping the active audio span."""
    on = np.nonzero(act)[0]
    t_lo, t_hi = on[0] / sr, on[-1] / sr
    lo = int(np.clip(np.searchsorted(t_rel, t_lo), 0, len(t_rel) - 1))
    hi = int(np.clip(np.searchsorted(t_rel, t_hi), 0, len(t_rel) - 1))
    return lo, hi


def _angles_to_vec(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def _unit(v):
    return v / np.clip(np.linalg.norm(v), 1e-9, None)


# --------------------------------------------------------------------------- #
def _load_calib_matrix(path):
    if path is None:
        return None
    R = np.array(json.load(open(path)), dtype=float).reshape(3, 3)
    return R


def main():
    p = argparse.ArgumentParser(description="LOCATA -> AmbiDOA manifest builder + calibration.")
    p.add_argument("--root", default=None,
                   help="LOCATA root (contains dev/ and eval/). Required for nested modes.")
    p.add_argument("--flat-dir", default=None,
                   help="Self-contained batch dir of prefixed wavs+txts (e.g. Dev-Static-Batch). "
                        "Flat mode: builds the manifest from this dir (audio_dir == this dir).")
    p.add_argument("--calib-rec", default=None,
                   help="With --flat-dir, run the convention check on this recording prefix "
                        "(e.g. dev_task1_recording1) instead of building.")
    p.add_argument("--out", default=None, help="Output manifest JSON path (build mode).")
    p.add_argument("--splits", nargs="+", default=["dev", "eval"])
    p.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="Tasks to include (moving-array tasks 5,6 are excluded by default).")
    p.add_argument("--array", default="eigenmike")
    p.add_argument("--ambix-name", default=DEFAULT_AMBIX_NAME,
                   help="Expected encoded ambiX filename inside each eigenmike folder.")
    p.add_argument("--vad-active-frac", type=float, default=0.5,
                   help="A source is 'active' in a track frame if its VAD fraction >= this.")
    p.add_argument("--calib-matrix", default=None,
                   help="JSON file with a 3x3 body->acoustic rotation; default identity.")
    p.add_argument("--list-pending", action="store_true",
                   help="Only report which recordings still need encoding; do not write.")
    p.add_argument("--calib", default=None,
                   help="Recording folder (relative to --root) to run the intensity-vs-geometry "
                        "convention check on instead of building.")
    args = p.parse_args()

    R_cal = _load_calib_matrix(args.calib_matrix)

    # Flat mode (self-contained batch dir): build or calibrate.
    if args.flat_dir:
        if args.calib_rec:
            cmd_calib_flat(args, R_cal)
        else:
            if args.out is None:
                p.error("--out is required for flat build.")
            cmd_build_flat(args, R_cal)
        return

    # Nested mode: needs the LOCATA root.
    if args.root is None:
        p.error("--root is required (nested mode), or pass --flat-dir for a batch dir.")
    if args.calib:
        cmd_calib(args, R_cal)
        return
    if args.out is None and not args.list_pending:
        p.error("--out is required in build mode (or pass --list-pending).")
    cmd_build(args, R_cal)


if __name__ == "__main__":
    main()
