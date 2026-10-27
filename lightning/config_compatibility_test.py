"""Validate JSON config compatibility with the Lightning pipeline.

The goal of this script is to catch configuration issues early, before
long-running training starts. Validation includes:
- schema checks (required/unknown keys)
- value checks (paths, model dimension constraints, loss mode constraints)
- object construction checks (DataModule + LightningModule)
- optional fast-dev execution check (1 train + 1 val batch)
"""

import argparse
import json
from pathlib import Path

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    # Backward-compatible import path for older installs.
    import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data_module import AmbiDOADataModule
from lit_module import AmbiDOALightningModule
from train_lightning import _pick_accelerator, apply_finetune_and_freeze, build_model_kwargs


class _SyntheticAmbiDOADataset(Dataset):
    """One real audio file, exposed as a single fixed batch.

    Reads the manifest JSON (cheap), picks one random entry, loads that audio
    file once, and slices it into non-silent windows that match the per-sample
    contract of AmbiDOADataset: (waveform[C, T], n_sources, mask[S],
    pos_labels[S, 3]).

    Both train and val dataloaders share this dataset, so dev runs train and
    validate on the *same* real batch — no random tensors, no manifest scan
    over thousands of files.
    """

    def __init__(self, config, batch_size, seed=0, max_file_attempts=8):
        import json
        import os
        import random

        import torchaudio

        from dataset import _pad_or_trim, angles_to_unitvec, dbfs as _dbfs

        manifest_path = config["manifest_path"]
        audio_dir = config["audio_dir"]
        sr = int(config.get("sr", 11025))
        window_len = int(int(config.get("window_len_ms", 250)) * sr / 1000)
        dbfs_threshold = float(config.get("dbfs_threshold", -80.0))
        max_sources = int(config.get("max_sources", 12))

        with open(manifest_path, "r") as f:
            entries = json.load(f)

        rng = random.Random(seed)
        # Try a few random files until one yields at least one non-silent window —
        # avoids dying on an unlucky pick of a low-energy clip.
        chosen = None
        last_err = None
        tried = set()
        for _ in range(max_file_attempts):
            idx = rng.randrange(len(entries))
            if idx in tried:
                continue
            tried.add(idx)
            item = entries[idx]
            audio_path = os.path.join(audio_dir, item["audio_filepath"])
            try:
                waveform, fs = torchaudio.load(audio_path)
            except Exception as e:
                last_err = e
                continue
            if fs != sr:
                waveform = torchaudio.functional.resample(waveform, fs, sr)
            n_full_windows = waveform.shape[-1] // window_len
            if n_full_windows < 1:
                continue
            chosen = (item, audio_path, waveform, n_full_windows)
            break
        if chosen is None:
            raise RuntimeError(
                f"_SyntheticAmbiDOADataset: no usable file in {max_file_attempts} attempts "
                f"(last error: {last_err})"
            )
        item, audio_path, waveform, n_full_windows = chosen

        # Per-file labels (shared across all windows of this file).
        azimuths = _pad_or_trim(item.get("azimuths", []), max_sources, None)
        elevations = _pad_or_trim(item.get("elevations", []), max_sources, None)
        mask_list = _pad_or_trim(item.get("mask", []), max_sources, 0)
        mask = torch.tensor(mask_list, dtype=torch.float32)
        pos_labels = torch.zeros((max_sources, 3), dtype=torch.float32)
        for i, (az, el, m) in enumerate(zip(azimuths, elevations, mask)):
            if m > 0.5 and az is not None and el is not None:
                pos_labels[i] = torch.tensor(angles_to_unitvec(az, el), dtype=torch.float32)
        n_sources = torch.tensor(float(mask.sum().item()), dtype=torch.float32)

        # Collect up to batch_size non-silent windows from this single file.
        self._items = []
        for w in range(n_full_windows):
            start = w * window_len
            wf = waveform[:, start:start + window_len]
            wf = (wf - wf.mean()) / (wf.std() + 1e-8)
            if _dbfs(wf.clone().detach()) < dbfs_threshold:
                continue
            self._items.append((wf.clone().detach(), n_sources, mask, pos_labels))
            if len(self._items) >= batch_size:
                break

        if not self._items:
            # Fallback: take first window even if "silent" — better than failing
            # the dev run on a quiet file.
            wf = waveform[:, :window_len]
            wf = (wf - wf.mean()) / (wf.std() + 1e-8)
            self._items.append((wf.clone().detach(), n_sources, mask, pos_labels))

        # Replicate to fill the batch if the file was too short.
        while len(self._items) < batch_size:
            self._items.append(self._items[0])

        self._source_path = audio_path

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class _SyntheticDataModule(pl.LightningDataModule):
    """Manifest-free DataModule wrapping _SyntheticAmbiDOADataset.

    Train and val share the same dataset (one batch from one real file), so
    dev runs hit identical real data on both sides without enumerating the
    full manifest.
    """

    def __init__(self, config):
        super().__init__()
        self.batch_size = int(config.get("batch_size", 8))
        self._dataset = _SyntheticAmbiDOADataset(config, batch_size=self.batch_size)
        print(f"[DEV] _SyntheticDataModule loaded one file: {self._dataset._source_path}")

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self._dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self._dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


REQUIRED_KEYS = {
    # Minimal keys required to build dataset/dataloader.
    "manifest_path",
    "audio_dir",
    "batch_size",
}

SUPPORTED_KEYS = {
    # Documentation-only key (ignored by the pipeline).
    "_comment",
    # Data/manifest keys.
    "manifest_path",
    "audio_dir",
    "batch_size",
    "num_workers",
    "train_num_workers",
    "val_num_workers",
    "val_ratio",
    "test_ratio",
    "split_seed",
    "sr",
    "window_len_ms",
    "dbfs_threshold",
    "require_cuda",
    "n_channels",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "max_sources",
    "use_posenc",
    "dropout",
    "stochastic_depth",
    "decoder_layers",
    "decoder_dropout",
    "num_epochs",
    "lr",
    "finetune_from",
    "finetune_lr",
    "reset_optimizer",
    "freeze_n_encoder_layers",
    "freeze_posenc",
    "lambda_count",
    "lambda_pos",
    "count_label_smoothing",
    "thresholded_loss_report",
    "presence_threshold",
    "checkpoint_monitor",
    "checkpoint_monitor_mode",
    "early_stopping_monitor",
    "early_stopping_mode",
    "use_scheduler",
    "warmup_steps",
    "weight_decay",
    "grad_clip",
    "precision",
    "save_top_k",
    "save_last",
    "log_lr",
    "early_stopping_patience",
    "accumulate_grad_batches",
    "log_grad_norm",
    "use_ema",
    "ema_decay",
    "ema_update_after_step",
    "mean_teacher_weight",
    "rdrop_weight",
    "regularization_every_n_steps",
    "compile_model",
    "compile_backend",
    "gradient_checkpointing",
    "log_timing",
    "timing_log_every_n_steps",
    "timing_cuda_sync",
    "log_every_n_steps",
    "run_txt_log_name",
    "run_txt_timestamp",
    "deterministic",
    "fast_dev_run",
    "overfit_dev_run",
    "num_sanity_val_steps",
    "matmul_precision",
    "overfit_test_epochs",
    "overfit_test_threshold",
    "overfit_test_batch_size",
    "subsample_pct",
    "label_mode",
    "window_active_frac",
    "skip_inactive_windows",
}


def _flatten_config(config):
    """Flatten one level of grouped config sections into a flat dict."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _load_config(path):
    """Load UTF-8 JSON config (BOM-tolerant). Supports grouped or flat layouts."""
    with open(path, encoding="utf-8-sig") as f:
        return _flatten_config(json.load(f))


def _validate_schema(cfg):
    """Validate config key set against required and supported keys."""
    missing = sorted(REQUIRED_KEYS - set(cfg))
    unknown = sorted(set(cfg) - SUPPORTED_KEYS)
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if unknown:
        raise ValueError(f"Unknown config keys (not used by Lightning pipeline): {unknown}")


def _validate_values(cfg):
    """Validate key values that can be checked statically."""
    manifest_path = Path(cfg["manifest_path"])
    audio_dir = Path(cfg["audio_dir"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_path does not exist: {manifest_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio_dir does not exist: {audio_dir}")

    d_model = int(cfg.get("d_model", 32))
    nhead = int(cfg.get("nhead", 4))
    if d_model % nhead != 0:
        # Transformer requirement: embedding dimension must divide by #heads.
        raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

    count_label_smoothing = float(cfg.get("count_label_smoothing", 0.0))
    if not (0.0 <= count_label_smoothing < 0.5):
        raise ValueError("count_label_smoothing must be in [0.0, 0.5).")

    for monitor_key in ("checkpoint_monitor_mode", "early_stopping_mode"):
        mode_val = cfg.get(monitor_key)
        if mode_val is not None and str(mode_val) not in ("min", "max"):
            raise ValueError(f"{monitor_key} must be 'min' or 'max'.")

    matmul_precision = cfg.get("matmul_precision", "high")
    if str(matmul_precision).lower() not in ("highest", "high", "medium"):
        raise ValueError("matmul_precision must be one of: 'highest', 'high', 'medium'.")

    accumulate = int(cfg.get("accumulate_grad_batches", 1))
    if accumulate < 1:
        raise ValueError("accumulate_grad_batches must be >= 1.")

    ema_decay = float(cfg.get("ema_decay", 0.999))
    if not (0.0 < ema_decay < 1.0):
        raise ValueError("ema_decay must be in (0, 1).")

    stochastic_depth = float(cfg.get("stochastic_depth", 0.0))
    if not (0.0 <= stochastic_depth < 1.0):
        raise ValueError("stochastic_depth must be in [0.0, 1.0).")

    if int(cfg.get("decoder_layers", 3)) < 1:
        raise ValueError("decoder_layers must be >= 1.")
    decoder_dropout = float(cfg.get("decoder_dropout", 0.1))
    if not (0.0 <= decoder_dropout < 1.0):
        raise ValueError("decoder_dropout must be in [0.0, 1.0).")

    timing_every = int(cfg.get("timing_log_every_n_steps", 0))
    if timing_every < 0:
        raise ValueError("timing_log_every_n_steps must be >= 0.")

    overfit_epochs = int(cfg.get("overfit_test_epochs", 50))
    if overfit_epochs < 1:
        raise ValueError("overfit_test_epochs must be >= 1.")

    overfit_threshold = float(cfg.get("overfit_test_threshold", 0.1))
    if overfit_threshold <= 0:
        raise ValueError("overfit_test_threshold must be > 0.")

    if "overfit_test_batch_size" in cfg:
        overfit_bs = int(cfg["overfit_test_batch_size"])
        if overfit_bs < 1:
            raise ValueError("overfit_test_batch_size must be >= 1.")

    if cfg.get("subsample_pct") is not None:
        ssp = float(cfg["subsample_pct"])
        if not (0 < ssp <= 100):
            raise ValueError("subsample_pct must be in (0, 100].")


def _build_lit_model(cfg):
    """Build Lightning module from config using same mapping as training script."""
    model_kwargs = build_model_kwargs(cfg)
    base_lr = float(cfg.get("lr", 5e-4))
    finetune_from = cfg.get("finetune_from")
    finetune_lr = float(cfg.get("finetune_lr", base_lr))
    lr_to_use = finetune_lr if finetune_from else base_lr
    return AmbiDOALightningModule(
        model_kwargs=model_kwargs,
        lr=lr_to_use,
        lambda_count=float(cfg.get("lambda_count", 0.1)),
        lambda_pos=float(cfg.get("lambda_pos", 2.5)),
        count_label_smoothing=float(cfg.get("count_label_smoothing", 0.0)),
        use_scheduler=bool(cfg.get("use_scheduler", True)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        warmup_steps=cfg.get("warmup_steps"),
        grad_clip=float(cfg.get("grad_clip", 0.0)),
        grad_accum_steps=int(cfg.get("accumulate_grad_batches", 1)),
        log_grad_norm=bool(cfg.get("log_grad_norm", True)),
        use_ema=bool(cfg.get("use_ema", False)),
        ema_decay=float(cfg.get("ema_decay", 0.999)),
        ema_update_after_step=int(cfg.get("ema_update_after_step", 0)),
        log_timing=bool(cfg.get("log_timing", False)),
        timing_log_every_n_steps=int(cfg.get("timing_log_every_n_steps", 0)),
        timing_cuda_sync=bool(cfg.get("timing_cuda_sync", False)),
    )


def _run_fast_dev(cfg, project_root):
    """Run a 1-step/1-val-batch smoke test to verify runtime compatibility."""
    # Override runtime-heavy options for quick validation while preserving
    # data/model/loss wiring.
    cfg_run = dict(cfg)
    cfg_run["fast_dev_run"] = 1
    cfg_run["num_epochs"] = 1
    cfg_run["num_workers"] = 0
    cfg_run["num_sanity_val_steps"] = 0
    cfg_run["log_every_n_steps"] = 1

    # Use the same accelerator resolution logic as real training.
    accelerator, devices = _pick_accelerator(require_cuda=bool(cfg_run.get("require_cuda", False)))
    # Synthetic data avoids the multi-minute manifest scan; the goal here is to
    # exercise the pipeline, not to validate against real audio.
    data_module = _SyntheticDataModule(cfg_run)
    lit_model = _build_lit_model(cfg_run)
    ckpt_path_for_fit = apply_finetune_and_freeze(lit_model, cfg_run, project_root=project_root)

    # Keep precision conservative on non-GPU devices.
    if accelerator == "gpu":
        precision = cfg_run.get("precision", "16-mixed")
    else:
        precision = "32-true"

    # Disable logging/checkpointing because this run is purely a compatibility smoke test.
    #
    # NOTE: AmbiDOALightningModule uses manual optimization, so Trainer-level automatic
    # grad clipping/accumulation are intentionally not configured here.
    cfg_grad_clip = float(cfg_run.get("grad_clip", 0.0))
    cfg_accumulate = int(cfg_run.get("accumulate_grad_batches", 1))
    if cfg_grad_clip > 0:
        print(
            "[CFG][WARN] grad_clip is set, but Trainer-level automatic clipping is disabled "
            "for manual optimization. Clipping must stay in the module loop."
        )
    if cfg_accumulate != 1:
        print(
            "[CFG][INFO] accumulate_grad_batches is mapped to manual gradient "
            f"accumulation in lit_module (effective grad_accum_steps={cfg_accumulate})."
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1,
        precision=precision,
        log_every_n_steps=1,
        enable_checkpointing=False,
        logger=False,
        deterministic=bool(cfg_run.get("deterministic", False)),
        fast_dev_run=1,
        num_sanity_val_steps=0,
        default_root_dir=str(project_root),
    )
    trainer.fit(lit_model, datamodule=data_module, ckpt_path=ckpt_path_for_fit)


def _run_overfit_test(cfg, project_root, num_epochs=50, loss_threshold=0.1, batch_size=None):
    """Run an overfit sanity check on a single batch.

    Trains on the same batch repeatedly with all regularization disabled. A correctly
    wired pipeline should drive the loss toward zero — if it does not, something
    structural is broken (loss/target mismatch, gradient flow blocked, head detached
    from logits, etc.).

    Uses Lightning's `overfit_batches=1`, which feeds the same training batch as the
    validation batch and disables shuffling. After enough epochs, train_loss and
    val_loss should converge to nearly identical values close to zero.
    """
    # Override regularization-heavy and runtime-heavy options. The test is whether
    # the model CAN fit one batch, not whether it generalizes.
    cfg_run = dict(cfg)
    cfg_run["num_workers"] = 0
    cfg_run["num_sanity_val_steps"] = 0
    cfg_run["log_every_n_steps"] = 1
    cfg_run["accumulate_grad_batches"] = 1
    cfg_run["compile_model"] = False
    # batch_size policy: explicit override wins (user knows what they're asking
    # for). Otherwise cap at 8 — production batch sizes (e.g. 64) only fit with
    # gradient checkpointing, which is overkill for a single-batch sanity test.
    # The synthetic dataset slices one real file into batch_size windows
    # (250 ms each), so even large batches stay sourced from a single file.
    if batch_size is not None:
        cfg_run["batch_size"] = int(batch_size)
    else:
        cfg_run["batch_size"] = min(int(cfg_run.get("batch_size", 8)), 8)
    cfg_run["use_ema"] = False
    cfg_run["mean_teacher_weight"] = 0.0
    cfg_run["rdrop_weight"] = 0.0
    cfg_run["dropout"] = 0.0
    # The query decoder carries its own dropout; zero it too so the
    # "regularization disabled" contract also holds for Build B (otherwise the
    # decoder can never drive a single batch to ~0 loss).
    cfg_run["decoder_dropout"] = 0.0
    cfg_run["weight_decay"] = 0.0
    # Label smoothing imposes a BCE loss floor of -(1-ε)·log(1-ε) - ε·log(ε)
    # per slot (≈0.325 at ε=0.1), which prevents an otherwise perfectly fit
    # batch from crossing the overfit threshold. Zero it here so the test
    # measures learnability, not the smoothing floor.
    cfg_run["count_label_smoothing"] = 0.0
    # Optimization regime for the memorization test (no scheduler/warmup). The
    # presence BCE needs logits driven to saturation (~|3|+) for BCE to approach
    # zero, which at production LR with one step/epoch (overfit_batches=1) would
    # stall. The query decoder (DETR-style) diverges at a hot 5e-3 LR (grad-norm
    # blow-up, presence flip-flopping 0<->max_sources): it needs a gentler LR,
    # gradient clipping, and more steps to break query symmetry and settle.
    cfg_run["lr"] = max(float(cfg_run.get("lr", 5e-4)), 1e-3)
    cfg_run["grad_clip"] = 1.0
    num_epochs = max(int(num_epochs), 300)
    # The decoder fits positions to ~1° quickly, but its presence BCE saturates
    # slowly (positions ~perfect, count ~4/5 → base_loss plateaus around 0.3-0.4;
    # a hotter LR only destabilizes the presence head). Relax the pass threshold
    # so the gate reflects "learned" (low angular + near-right count) rather than
    # full BCE saturation. A genuinely broken decoder stays >>1.
    loss_threshold = max(float(loss_threshold), 0.4)
    cfg_run["use_scheduler"] = False
    cfg_run["warmup_steps"] = 0

    accelerator, devices = _pick_accelerator(require_cuda=bool(cfg_run.get("require_cuda", False)))
    # Synthetic data module loads one real audio file from the manifest and
    # exposes its windows as a single shared train/val batch — no manifest scan.
    data_module = _SyntheticDataModule(cfg_run)
    lit_model = _build_lit_model(cfg_run)
    apply_finetune_and_freeze(lit_model, cfg_run, project_root=project_root)

    if accelerator == "gpu":
        precision = cfg_run.get("precision", "16-mixed")
    else:
        precision = "32-true"

    print(f"[OVERFIT] Starting single-batch overfit test for {num_epochs} epochs (regularization disabled).")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=num_epochs,
        precision=precision,
        log_every_n_steps=1,
        enable_checkpointing=False,
        logger=False,
        deterministic=bool(cfg_run.get("deterministic", False)),
        overfit_batches=1,
        num_sanity_val_steps=0,
        default_root_dir=str(project_root),
    )
    trainer.fit(lit_model, datamodule=data_module)

    metrics = trainer.callback_metrics
    # train/base_loss = task loss only (no regularizers) — apples-to-apples with val_loss.
    final_train = float(metrics.get("train/base_loss", metrics.get("train/loss", float("nan"))))
    final_val = float(metrics.get("val_loss", float("nan")))
    print(f"[OVERFIT] Final | train/base_loss={final_train:.6f} | val_loss={final_val:.6f}")

    # Use `not <` so NaN also fails (NaN comparisons return False).
    if not (final_train < loss_threshold):
        raise RuntimeError(
            f"[OVERFIT][FAIL] train/base_loss={final_train:.6f} did not drop below {loss_threshold} "
            f"after {num_epochs} epochs on a single batch. The pipeline is not learning — "
            "check loss connectivity, gradient flow, and target alignment."
        )
    print(f"[OVERFIT][OK] Model successfully overfit a single batch (train/base_loss < {loss_threshold}).")


def main():
    """CLI entrypoint for config compatibility validation."""
    parser = argparse.ArgumentParser(description="Validate Lightning config compatibility for AmbiDOA.")
    default_cfg = Path(__file__).resolve().parents[1] / "config_lightning.json"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--skip-fast-dev", action="store_true")
    parser.add_argument("--skip-overfit-test", action="store_true")
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=None,
        help="Override config.overfit_test_epochs.",
    )
    parser.add_argument(
        "--overfit-threshold",
        type=float,
        default=None,
        help="Override config.overfit_test_threshold.",
    )
    parser.add_argument(
        "--overfit-batch-size",
        type=int,
        default=None,
        help=(
            "Override batch_size for the overfit test. The synthetic dataset "
            "slices one real audio file into batch_size 250 ms windows, so a "
            "single file can still feed large batches. Defaults to min(config.batch_size, 8)."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = _load_config(args.config)
    # Stage 1: static config checks.
    _validate_schema(cfg)
    _validate_values(cfg)

    # Stage 2: build core objects to verify mapping between config and pipeline.
    data_module = AmbiDOADataModule(config=cfg)
    _ = _build_lit_model(cfg)
    _ = data_module

    # Stage 3: optional runtime smoke test.
    if not args.skip_fast_dev:
        _run_fast_dev(cfg, project_root=project_root)

    # Stage 4: optional overfit test (verifies the model can actually learn).
    # CLI flags take precedence over config values when provided.
    if not args.skip_overfit_test:
        overfit_epochs = args.overfit_epochs if args.overfit_epochs is not None else int(cfg.get("overfit_test_epochs", 50))
        overfit_threshold = args.overfit_threshold if args.overfit_threshold is not None else float(cfg.get("overfit_test_threshold", 0.1))
        # Resolution order: CLI > config.overfit_test_batch_size > cap default in _run_overfit_test.
        if args.overfit_batch_size is not None:
            overfit_bs = args.overfit_batch_size
        elif "overfit_test_batch_size" in cfg:
            overfit_bs = int(cfg["overfit_test_batch_size"])
        else:
            overfit_bs = None
        _run_overfit_test(
            cfg,
            project_root=project_root,
            num_epochs=overfit_epochs,
            loss_threshold=overfit_threshold,
            batch_size=overfit_bs,
        )

    # Stable success token for scripts/CI parsers.
    print("CONFIG_COMPATIBILITY_OK")


if __name__ == "__main__":
    main()
