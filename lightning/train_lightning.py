"""Main Lightning training entrypoint for AmbiDOA.

Responsibilities of this script:
- Load JSON config and map it to DataModule + LightningModule arguments.
- Resolve hardware backend (GPU/MPS/CPU).
- Apply optional fine-tuning initialization and selective layer freezing.
- Configure callbacks (checkpointing, LR logging, early stopping).
- Start Trainer.fit (train mode) or Trainer.validate (validate-only mode).
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:
    # Backward-compatible import path for older environments.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

# Lightning run uses its own run-scoped TXT logger; disable legacy global TXT writes.
os.environ.setdefault("AMBIDOA_DISABLE_LEGACY_LOG", "1")

from data_module import AmbiDOADataModule
from finetune_utils import (
    freeze_first_n_encoder_layers,
    load_pretrained_into,
    maybe_freeze_posenc,
    resolve_ckpt_path,
)
from lit_module import AmbiDOALightningModule


def _load_config(config_path):
    """Load UTF-8 JSON config (BOM-tolerant)."""
    with open(config_path, encoding="utf-8-sig") as f:
        return json.load(f)


def _pick_accelerator(require_cuda=False):
    """Pick accelerator/devices with explicit fallback order.

    Preference order:
    1. CUDA GPU
    2. MPS (Apple Silicon)
    3. CPU
    """
    if torch.cuda.is_available():
        return "gpu", 1
    # Hard guard for configs that require CUDA-only execution.
    if require_cuda:
        raise RuntimeError("config.require_cuda=true but CUDA is not available on this machine.")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def _configure_matmul_precision(config, accelerator):
    """Configure float32 matmul precision for Tensor Core utilization on CUDA."""
    if accelerator != "gpu":
        return

    precision = str(config.get("matmul_precision", "high")).lower()
    if precision not in {"highest", "high", "medium"}:
        raise ValueError("matmul_precision must be one of: 'highest', 'high', 'medium'.")

    torch.set_float32_matmul_precision(precision)
    print(f"[Perf] torch.set_float32_matmul_precision('{precision}')")


def build_model_kwargs(config):
    """Map config keys to AmbiTransformerDOA constructor kwargs."""
    return {
        "n_channels": int(config.get("n_channels", 16)),
        "d_model": int(config.get("d_model", 32)),
        "nhead": int(config.get("nhead", 4)),
        "num_layers": int(config.get("num_layers", 4)),
        "dim_feedforward": int(config.get("dim_feedforward", 512)),
        "max_sources": int(config.get("max_sources", 12)),
        "use_posenc": bool(config.get("use_posenc", True)),
        "dropout": float(config.get("dropout", 0.003)),
    }


def apply_finetune_and_freeze(lit_model, config, project_root):
    """Apply optional pretrained loading and freezing policy.

    Returns:
    - `ckpt_path_for_fit`: Lightning checkpoint path when full-state resume is
      requested and possible, else None.
    """
    finetune_from = config.get("finetune_from")
    reset_optimizer = bool(config.get("reset_optimizer", True))
    freeze_n_encoder_layers = int(config.get("freeze_n_encoder_layers", 0))
    freeze_posenc = bool(config.get("freeze_posenc", False))
    ckpt_path_for_fit = None

    if finetune_from:
        # Resolve checkpoint path from absolute or project-relative config value.
        ckpt_path = resolve_ckpt_path(finetune_from, project_root=project_root)
        is_lightning_ckpt = ckpt_path.suffix.lower() == ".ckpt"

        # Full-state resume is only supported when passing .ckpt directly to fit().
        if (not reset_optimizer) and is_lightning_ckpt:
            ckpt_path_for_fit = str(ckpt_path)
            print(f"[FT] Resuming full training state from Lightning checkpoint: {ckpt_path}")
        else:
            # For plain .pt/.pth or reset_optimizer=True, load model weights only.
            load_pretrained_into(lit_model.model, ckpt_path, strict=False, map_location="cpu")
            if not reset_optimizer:
                print(
                    "[FT][WARN] reset_optimizer=false requested, but full optimizer restore is "
                    "only available with Lightning .ckpt checkpoints."
                )

    # Optional staged fine-tuning controls.
    if freeze_n_encoder_layers > 0:
        freeze_first_n_encoder_layers(lit_model.model, freeze_n_encoder_layers)
    if freeze_posenc:
        maybe_freeze_posenc(lit_model.model)

    return ckpt_path_for_fit


def main():
    """CLI entrypoint for training or validation-only runs."""
    parser = argparse.ArgumentParser(description="Train/validate AmbiDOA using PyTorch Lightning.")
    default_config = Path(__file__).resolve().parents[1] / "config.json"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run only validation on the dataset from --config (no training).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint path for --validate-only. If omitted, uses config.finetune_from. "
            "Supports .ckpt/.pth/.pt formats."
        ),
    )
    args = parser.parse_args()

    # 1) Load config and resolve runtime device.
    config = _load_config(args.config)
    run_started_at = datetime.now()
    run_txt_log_name = config.get("run_txt_log_name")
    if not run_txt_log_name:
        run_txt_log_name = f"log_AmbiDOA_{run_started_at.strftime('%Y-%m-%d')}.txt"
    run_txt_timestamp = bool(config.get("run_txt_timestamp", True))
    require_cuda = bool(config.get("require_cuda", False))
    accelerator, devices = _pick_accelerator(require_cuda=require_cuda)
    project_root = Path(__file__).resolve().parents[1]
    _configure_matmul_precision(config, accelerator)

    # 2) Build data and model modules from config.
    data_module = AmbiDOADataModule(
        config_path=args.config,
        config=config,
        run_txt_log_name=run_txt_log_name,
        run_txt_timestamp=run_txt_timestamp,
    )
    model_kwargs = build_model_kwargs(config)
    base_lr = float(config.get("lr", 5e-4))
    finetune_from = config.get("finetune_from")
    finetune_lr = float(config.get("finetune_lr", base_lr))
    lr_to_use = finetune_lr if finetune_from else base_lr
    lit_model = AmbiDOALightningModule(
        model_kwargs=model_kwargs,
        lr=lr_to_use,
        count_loss_type=config.get("count_loss_type", "regression"),
        lambda_count=float(config.get("lambda_count", 0.1)),
        lambda_pos=float(config.get("lambda_pos", 2.5)),
        pos_loss_type=config.get("pos_loss_type", "cosine"),
        use_scheduler=bool(config.get("use_scheduler", True)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        warmup_steps=config.get("warmup_steps"),
        grad_clip=float(config.get("grad_clip", 0.0)),
        grad_accum_steps=int(config.get("accumulate_grad_batches", 1)),
        log_grad_norm=bool(config.get("log_grad_norm", True)),
        use_ema=bool(config.get("use_ema", False)),
        ema_decay=float(config.get("ema_decay", 0.999)),
        ema_update_after_step=int(config.get("ema_update_after_step", 0)),
        log_timing=bool(config.get("log_timing", False)),
        timing_log_every_n_steps=int(config.get("timing_log_every_n_steps", 0)),
        timing_cuda_sync=bool(config.get("timing_cuda_sync", False)),
        run_txt_log_name=run_txt_log_name,
        run_txt_timestamp=run_txt_timestamp,
    )
    # 3) Load checkpoint/freeze policy depending on run mode.
    ckpt_path_for_fit = None
    if args.validate_only:
        ckpt_ref = str(args.checkpoint) if args.checkpoint is not None else config.get("finetune_from")
        if not ckpt_ref:
            raise RuntimeError(
                "--validate-only requires checkpoint weights. "
                "Provide --checkpoint or set finetune_from in config."
            )
        ckpt_path = resolve_ckpt_path(ckpt_ref, project_root=project_root)
        load_pretrained_into(lit_model.model, ckpt_path, strict=False, map_location="cpu")
        print(f"[VAL] Validation-only mode with checkpoint: {ckpt_path}")
    else:
        ckpt_path_for_fit = apply_finetune_and_freeze(lit_model, config, project_root=project_root)

    # 4) Prepare deterministic daily output directories.
    date_dir = run_started_at.strftime("%Y_%m_%d")
    save_dir = project_root / "models" / date_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = project_root / "Logs" / date_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_txt_path = logs_dir / run_txt_log_name
    run_started_line = f"[{run_started_at.strftime('%Y-%m-%d %H:%M:%S')}] Run started"
    with open(run_txt_path, "a", encoding="utf-8") as f:
        f.write(run_started_line + "\n")
    print(run_started_line)

    # CSV logger stores step/epoch metrics under Logs/YYYY_MM_DD/lightning_logs.
    logger = CSVLogger(save_dir=str(logs_dir), name="lightning_logs")
    callbacks = []
    if not args.validate_only:
        # 5) Configure checkpointing (best + optional last).
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(save_dir),
            filename="best_model_{epoch:02d}_{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=int(config.get("save_top_k", 1)),
            save_last=bool(config.get("save_last", True)),
            save_weights_only=True,
        )
        callbacks.append(checkpoint_cb)

        # Optional LR tracking per step for scheduler diagnostics.
        if bool(config.get("log_lr", True)):
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        # Optional early stopping based on epoch validation loss.
        early_stopping_patience = int(config.get("early_stopping_patience", 0))
        if early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=early_stopping_patience,
                )
            )

    # Keep mixed precision default only on GPU; use full precision elsewhere.
    if accelerator == "gpu":
        precision = config.get("precision", "16-mixed")
    else:
        precision = config.get("precision", "32-true")

    # 6) Create Lightning Trainer with config-driven knobs.
    #
    # NOTE: AmbiDOALightningModule uses manual optimization.
    # Lightning does not support automatic gradient accumulation/clipping in this mode.
    # Both are handled explicitly in lit_module.training_step.
    if not args.validate_only:
        cfg_grad_clip = float(config.get("grad_clip", 0.0))
        cfg_accumulate = int(config.get("accumulate_grad_batches", 1))
        if cfg_grad_clip > 0:
            print(
                "[CFG][WARN] grad_clip is configured in JSON, but Trainer-level automatic "
                "gradient clipping is disabled for manual optimization. "
                "Clipping is handled in lit_module.training_step."
            )
        if cfg_accumulate != 1:
            print(
                "[CFG][INFO] accumulate_grad_batches is mapped to manual gradient "
                f"accumulation in lit_module (effective grad_accum_steps={cfg_accumulate})."
            )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(config.get("num_epochs", 5)),
        precision=precision,
        log_every_n_steps=int(config.get("log_every_n_steps", 10)),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(project_root),
        deterministic=bool(config.get("deterministic", False)),
        fast_dev_run=config.get("fast_dev_run", False),
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 2)),
    )
    # 7) Execute selected mode.
    if args.validate_only:
        trainer.validate(lit_model, datamodule=data_module)
    else:
        # `ckpt_path` is non-None only for full-state resume in training mode.
        trainer.fit(lit_model, datamodule=data_module, ckpt_path=ckpt_path_for_fit)


if __name__ == "__main__":
    main()
