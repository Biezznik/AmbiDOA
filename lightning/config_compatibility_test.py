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

from data_module import AmbiDOADataModule
from lit_module import AmbiDOALightningModule
from train_lightning import _pick_accelerator, apply_finetune_and_freeze, build_model_kwargs


REQUIRED_KEYS = {
    # Minimal keys required to build dataset/dataloader.
    "manifest_path",
    "audio_dir",
    "batch_size",
}

SUPPORTED_KEYS = {
    # Data/manifest keys.
    "manifest_path",
    "audio_dir",
    "batch_size",
    "num_workers",
    "val_ratio",
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
    "num_epochs",
    "lr",
    "finetune_from",
    "finetune_lr",
    "reset_optimizer",
    "freeze_n_encoder_layers",
    "freeze_posenc",
    "count_loss_type",
    "lambda_count",
    "lambda_pos",
    "pos_loss_type",
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
    "log_timing",
    "timing_log_every_n_steps",
    "timing_cuda_sync",
    "log_every_n_steps",
    "deterministic",
    "fast_dev_run",
    "num_sanity_val_steps",
    "matmul_precision",
}


def _load_config(path):
    """Load UTF-8 JSON config (BOM-tolerant)."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


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

    count_loss_type = cfg.get("count_loss_type", "regression")
    if count_loss_type != "regression":
        raise ValueError(
            "count_loss_type must be 'regression'. "
            "Current AmbiDOA count head predicts scalar counts only."
        )

    matmul_precision = cfg.get("matmul_precision", "high")
    if str(matmul_precision).lower() not in ("highest", "high", "medium"):
        raise ValueError("matmul_precision must be one of: 'highest', 'high', 'medium'.")

    accumulate = int(cfg.get("accumulate_grad_batches", 1))
    if accumulate < 1:
        raise ValueError("accumulate_grad_batches must be >= 1.")

    ema_decay = float(cfg.get("ema_decay", 0.999))
    if not (0.0 < ema_decay < 1.0):
        raise ValueError("ema_decay must be in (0, 1).")

    timing_every = int(cfg.get("timing_log_every_n_steps", 0))
    if timing_every < 0:
        raise ValueError("timing_log_every_n_steps must be >= 0.")


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
        count_loss_type=cfg.get("count_loss_type", "regression"),
        lambda_count=float(cfg.get("lambda_count", 0.1)),
        lambda_pos=float(cfg.get("lambda_pos", 2.5)),
        pos_loss_type=cfg.get("pos_loss_type", "cosine"),
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
    data_module = AmbiDOADataModule(config=cfg_run)
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


def main():
    """CLI entrypoint for config compatibility validation."""
    parser = argparse.ArgumentParser(description="Validate Lightning config compatibility for AmbiDOA.")
    default_cfg = Path(__file__).resolve().parents[1] / "config_lightning.json"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--skip-fast-dev", action="store_true")
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

    # Stable success token for scripts/CI parsers.
    print("CONFIG_COMPATIBILITY_OK")


if __name__ == "__main__":
    main()
