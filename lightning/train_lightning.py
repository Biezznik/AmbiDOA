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


def _flatten_config(config):
    """Flatten one level of grouped config sections into a flat dict."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _load_config(config_path):
    """Load UTF-8 JSON config (BOM-tolerant). Supports grouped or flat layouts."""
    with open(config_path, encoding="utf-8-sig") as f:
        return _flatten_config(json.load(f))


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
        # Positional-encoding length must cover the window's token count
        # T = round(window_len_ms * sr / 1000). Derive it so longer windows
        # (e.g. 400 ms @11025 -> T=4410) don't exceed the pos-enc table and crash.
        "input_len": int(round(float(config.get("window_len_ms", 250)) * float(config.get("sr", 11025)) / 1000)),
        "dropout": float(config.get("dropout", 0.003)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
        "stochastic_depth": float(config.get("stochastic_depth", 0.0)),
        "decoder_layers": int(config.get("decoder_layers", 3)),
        "decoder_dropout": float(config.get("decoder_dropout", 0.1)),
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
        "--test-only",
        action="store_true",
        help="Run only evaluation on the held-out TEST split (requires test_ratio>0). No training.",
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
    parser.add_argument(
        "--overfit-batch-size",
        type=int,
        default=None,
        help=(
            "Override batch_size for the overfit_dev_run path. The synthetic "
            "dataset slices one real audio file into batch_size 250 ms windows, "
            "so a single file can still feed large batches. Defaults to "
            "min(config.batch_size, 8)."
        ),
    )
    parser.add_argument(
        "--subsample-pct",
        type=float,
        default=None,
        help=(
            "Debug flag: run regular train/validation on a random subset of "
            "the manifest (percentage 0-100). Logging and checkpoints stay "
            "active, but the run is flagged as subsampled in the run TXT log. "
            "Overrides config.subsample_pct."
        ),
    )
    args = parser.parse_args()

    # 1) Load config and resolve runtime device.
    config = _load_config(args.config)
    # CLI subsample flag overrides config.subsample_pct so it can be toggled
    # per-invocation without editing the JSON.
    if args.subsample_pct is not None:
        config["subsample_pct"] = float(args.subsample_pct)
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
    # In any dev mode (fast_dev_run OR overfit_dev_run), swap in the synthetic
    # data module so the manifest scan (torchaudio.info on every file in the
    # manifest) is skipped — dev iterations should reach the model in seconds,
    # not minutes. Also override heavy options that aren't what dev mode is
    # testing and that don't fit on a single GPU at the production batch size:
    #   - batch_size=8 : production batch_size (64) only fits with gradient
    #     checkpointing AND no second forward — too tight for a sanity check.
    #     Smaller batch keeps activations under ~3 GB so it works even with
    #     another process holding part of the GPU.
    #   - compile_model=False : avoid 30s+ compilation cost; the OptimizedModule
    #     wrapper has been observed to leave inner submodules in eval mode at
    #     training start.
    #   - use_ema=False, mean_teacher_weight=0 : no second forward pass through
    #     a teacher copy; halves activation memory and removes a moving target
    #     from the dev sanity test.
    _fast_dev_run_cfg = bool(config.get("fast_dev_run", False))
    _overfit_dev_run_cfg = bool(config.get("overfit_dev_run", False))
    if _fast_dev_run_cfg or _overfit_dev_run_cfg:
        from config_compatibility_test import _SyntheticDataModule
        config["batch_size"] = min(int(config.get("batch_size", 8)), 8)
        config["compile_model"] = False
        config["use_ema"] = False
        config["mean_teacher_weight"] = 0.0
        config["stochastic_depth"] = 0.0
        config["decoder_dropout"] = 0.0
        data_module = _SyntheticDataModule(config)
    else:
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
        lambda_count=float(config.get("lambda_count", 0.1)),
        lambda_pos=float(config.get("lambda_pos", 2.5)),
        count_label_smoothing=float(config.get("count_label_smoothing", 0.0)),
        thresholded_loss_report=bool(config.get("thresholded_loss_report", False)),
        presence_threshold=float(config.get("presence_threshold", 0.5)),
        use_scheduler=bool(config.get("use_scheduler", True)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        warmup_steps=config.get("warmup_steps"),
        grad_clip=float(config.get("grad_clip", 0.0)),
        grad_accum_steps=int(config.get("accumulate_grad_batches", 1)),
        log_grad_norm=bool(config.get("log_grad_norm", True)),
        use_ema=bool(config.get("use_ema", False)),
        ema_decay=float(config.get("ema_decay", 0.999)),
        ema_update_after_step=int(config.get("ema_update_after_step", 0)),
        mean_teacher_weight=float(config.get("mean_teacher_weight", 0.0)),
        rdrop_weight=float(config.get("rdrop_weight", 0.0)),
        regularization_every_n_steps=int(config.get("regularization_every_n_steps", 1)),
        log_timing=bool(config.get("log_timing", False)),
        timing_log_every_n_steps=int(config.get("timing_log_every_n_steps", 0)),
        timing_cuda_sync=bool(config.get("timing_cuda_sync", False)),
        run_txt_log_name=run_txt_log_name,
        run_txt_timestamp=run_txt_timestamp,
    )
    # 3) Optionally compile the student model for faster kernel execution.
    if bool(config.get("compile_model", False)) and accelerator == "gpu":
        compile_backend = str(config.get("compile_backend", "cudagraphs"))
        try:
            lit_model.model = torch.compile(lit_model.model, backend=compile_backend)
            print(f"[Perf] torch.compile applied to student model (backend='{compile_backend}').")
        except Exception as e:
            print(f"[Perf][WARN] torch.compile(backend='{compile_backend}') failed, falling back to eager: {e}")

    # 4) Load checkpoint/freeze policy depending on run mode.
    ckpt_path_for_fit = None
    if args.validate_only or args.test_only:
        mode = "validate" if args.validate_only else "test"
        ckpt_ref = str(args.checkpoint) if args.checkpoint is not None else config.get("finetune_from")
        if not ckpt_ref:
            raise RuntimeError(
                f"--{mode}-only requires checkpoint weights. "
                "Provide --checkpoint or set finetune_from in config."
            )
        ckpt_path = resolve_ckpt_path(ckpt_ref, project_root=project_root)
        load_pretrained_into(lit_model.model, ckpt_path, strict=False, map_location="cpu")
        print(f"[{mode.upper()}] {mode}-only mode with checkpoint: {ckpt_path}")
    else:
        ckpt_path_for_fit = apply_finetune_and_freeze(lit_model, config, project_root=project_root)

    # 5) Prepare deterministic daily output directories.
    # Two independent dev-mode flags suppress all filesystem side effects:
    #   fast_dev_run    — runs Lightning's 1-batch smoke test
    #   overfit_dev_run — runs the single-batch overfit sanity check
    # Either one (or both) suppresses models/, Logs/, run-started txt,
    # CSVLogger and ModelCheckpoint.
    fast_dev_run_value = config.get("fast_dev_run", False)
    fast_dev_run_active = bool(fast_dev_run_value)
    overfit_dev_run_active = bool(config.get("overfit_dev_run", False))
    dev_mode_active = fast_dev_run_active or overfit_dev_run_active
    save_dir = None
    logs_dir = None
    if not dev_mode_active:
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
    else:
        active = []
        if fast_dev_run_active:
            active.append("fast_dev_run")
        if overfit_dev_run_active:
            active.append("overfit_dev_run")
        print(
            f"[{run_started_at.strftime('%Y-%m-%d %H:%M:%S')}] Run started "
            f"(dev mode: {' + '.join(active)} — filesystem logging suppressed)"
        )

    # CSV logger stores step/epoch metrics under Logs/YYYY_MM_DD/lightning_logs.
    if dev_mode_active:
        logger = False
    else:
        logger = CSVLogger(save_dir=str(logs_dir), name="lightning_logs")
    callbacks = []
    if not args.validate_only and not args.test_only and not dev_mode_active:
        # 6) Configure checkpointing (best + optional last).
        # Default monitor is val_angular_error_deg (slash-free alias for
        # val/angular_error_deg). The angular error tracks the actual operational
        # metric and avoids ordinal BCE's confidence-overfit drift in val_loss.
        # Filename templates cannot contain '/' (it produces subdirectories), so
        # the alias is what gets interpolated.
        monitor_key = str(config.get("checkpoint_monitor", "val_angular_error_deg"))
        monitor_mode = str(config.get("checkpoint_monitor_mode", "min"))
        if monitor_mode not in ("min", "max"):
            raise ValueError("checkpoint_monitor_mode must be 'min' or 'max'.")
        filename_template = "best_model_{epoch:02d}_{" + monitor_key + ":.4f}"
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(save_dir),
            filename=filename_template,
            monitor=monitor_key,
            mode=monitor_mode,
            save_top_k=int(config.get("save_top_k", 1)),
            save_last=bool(config.get("save_last", True)),
            save_weights_only=True,
        )
        callbacks.append(checkpoint_cb)

        # Optional LR tracking per step for scheduler diagnostics.
        if bool(config.get("log_lr", True)):
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        # Optional early stopping — uses the same monitor as the checkpoint by
        # default, but can be overridden explicitly.
        early_stopping_patience = int(config.get("early_stopping_patience", 0))
        if early_stopping_patience > 0:
            es_monitor = str(config.get("early_stopping_monitor", monitor_key))
            es_mode = str(config.get("early_stopping_mode", monitor_mode))
            if es_mode not in ("min", "max"):
                raise ValueError("early_stopping_mode must be 'min' or 'max'.")
            callbacks.append(
                EarlyStopping(
                    monitor=es_monitor,
                    mode=es_mode,
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
    if not args.validate_only and not args.test_only:
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
    # Three execution paths:
    #   --validate-only          → trainer.validate, skip fit.
    #   overfit_dev_run only     → skip fit, run only the overfit sanity check.
    #   fast_dev_run / normal    → trainer.fit (fast_dev_run still routes through
    #                              Lightning's 1-batch smoke test via the
    #                              fast_dev_run kwarg passed to Trainer above).
    if args.validate_only:
        trainer.validate(lit_model, datamodule=data_module)
    elif args.test_only:
        trainer.test(lit_model, datamodule=data_module)
    elif overfit_dev_run_active and not fast_dev_run_active:
        print("[OVERFIT] overfit_dev_run is active without fast_dev_run — skipping trainer.fit().")
    else:
        # `ckpt_path` is non-None only for full-state resume in training mode.
        trainer.fit(lit_model, datamodule=data_module, ckpt_path=ckpt_path_for_fit)

        # Held-out test evaluation on the best checkpoint — unbiased final metrics
        # on files disjoint from train/val. Runs only when a test split was
        # configured (test_ratio > 0) and a checkpoint exists (non-dev run).
        if not dev_mode_active and float(config.get("test_ratio", 0.0)) > 0:
            try:
                trainer.test(lit_model, datamodule=data_module, ckpt_path="best")
            except Exception as exc:
                print(f"[TEST][WARN] Held-out test evaluation skipped: {exc}")

    # 8) When overfit_dev_run is active, run the single-batch overfit sanity check.
    # This verifies the pipeline can drive loss to zero on a fixed batch — a
    # learning-correctness probe independent of the smoke test that fast_dev_run
    # performs. Late import avoids circular dependency: config_compatibility_test
    # imports _pick_accelerator/build_model_kwargs/apply_finetune_and_freeze
    # from this module.
    if overfit_dev_run_active and not args.validate_only:
        from config_compatibility_test import _run_overfit_test
        overfit_epochs = int(config.get("overfit_test_epochs", 50))
        overfit_threshold = float(config.get("overfit_test_threshold", 0.1))
        # Resolution order: CLI > config.overfit_test_batch_size > cap default.
        if args.overfit_batch_size is not None:
            overfit_bs = args.overfit_batch_size
        elif "overfit_test_batch_size" in config:
            overfit_bs = int(config["overfit_test_batch_size"])
        else:
            overfit_bs = None
        _run_overfit_test(
            config,
            project_root=project_root,
            num_epochs=overfit_epochs,
            loss_threshold=overfit_threshold,
            batch_size=overfit_bs,
        )


if __name__ == "__main__":
    main()
