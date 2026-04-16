"""LightningModule that keeps AmbiDOA training behavior close to legacy training.

This module intentionally preserves the original optimization order and loss
composition from the non-Lightning pipeline:
1. Forward pass.
2. Hungarian/PI matching + count/position losses.
3. Backpropagation.
4. Optional gradient clipping.
5. Optimizer step.
6. Optional scheduler step (per iteration).

Keeping this sequence explicit makes migration safer and easier to audit.
"""

import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    # Backward-compatible import path for environments with legacy package name.
    import pytorch_lightning as pl

# Ensure project root is importable when this file is launched as a script/module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import AmbiTransformerDOA
from train_utils import angular_error_stats_PI, build_warmup_cosine_scheduler, multi_source_loss


class AmbiDOALightningModule(pl.LightningModule):
    """AmbiDOA model wrapped as LightningModule.

    The class keeps the model interface unchanged:
    - Input: ambisonic frame tensor.
    - Outputs: source count prediction and source direction predictions.

    Validation metrics are accumulated manually to preserve the same epoch-level
    reports used in the original training utility.
    """

    def __init__(
        self,
        model_kwargs=None,
        lr=5e-4,
        count_loss_type="regression",
        lambda_count=0.1,
        lambda_pos=2.5,
        pos_loss_type="cosine",
        use_scheduler=True,
        weight_decay=0.0,
        warmup_steps=None,
        grad_clip=0.0,
        grad_accum_steps=1,
        log_grad_norm=True,
        use_ema=False,
        ema_decay=0.999,
        ema_update_after_step=0,
        log_timing=False,
        timing_log_every_n_steps=0,
        timing_cuda_sync=False,
        run_txt_log_name=None,
        run_txt_timestamp=True,
    ):
        super().__init__()
        # Default architecture mirrors existing AmbiDOA settings so that
        # config files can override only what they need.
        model_defaults = dict(
            n_channels=16,
            d_model=32,
            nhead=4,
            num_layers=4,
            dim_feedforward=512,
            max_sources=12,
            use_posenc=True,
            dropout=0.003,
        )
        if model_kwargs:
            model_defaults.update(model_kwargs)

        # Core AmbiDOA transformer model.
        self.model = AmbiTransformerDOA(**model_defaults)

        # Loss and optimization hyperparameters.
        self.lr = float(lr)
        self.count_loss_type = str(count_loss_type)
        self.lambda_count = float(lambda_count)
        self.lambda_pos = float(lambda_pos)
        self.pos_loss_type = pos_loss_type
        self.use_scheduler = bool(use_scheduler)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = warmup_steps
        self.grad_clip = float(grad_clip)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.log_grad_norm = bool(log_grad_norm)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_update_after_step = max(0, int(ema_update_after_step))
        self.log_timing = bool(log_timing)
        self.timing_log_every_n_steps = max(0, int(timing_log_every_n_steps))
        self.timing_cuda_sync = bool(timing_cuda_sync)
        self.run_txt_log_name = run_txt_log_name
        self.run_txt_timestamp = bool(run_txt_timestamp)
        self._run_txt_log_path = None

        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1).")
        if self.count_loss_type != "regression":
            raise ValueError(
                f"Unsupported count_loss_type='{self.count_loss_type}'. "
                "Current AmbiDOA count head supports only 'regression'."
            )

        # Use manual optimization to mirror the original training loop order:
        # zero_grad -> forward -> PI/Hungarian loss -> backward -> clip -> step -> scheduler.step
        self.automatic_optimization = False

        # Per-epoch validation accumulators (streaming; no full-epoch tensor caching).
        self._val_ang_err_sum = 0.0
        self._val_ang_cnt_sum = 0.0
        self._val_count_err_sum = 0.0
        self._val_count_sample_sum = 0
        self._val_pred_count_hist = None
        self._val_per_k_samples = None
        self._val_per_k_count_err_sum = None
        self._val_per_k_ang_err_sum = None
        self._val_per_k_ang_cnt_sum = None

        # Manual optimization/telemetry state.
        self._optimizer_step_count_total = 0
        self._train_optimizer_steps_epoch = 0
        self._train_grad_norm_sum = 0.0
        self._train_grad_norm_count = 0
        self._train_clip_events = 0
        self._train_step_time_sum = 0.0
        self._train_step_time_count = 0
        self._val_step_time_sum = 0.0
        self._val_step_time_count = 0
        self._val_step_ms_epoch = None

        # EMA shadow and swap state (optional).
        self._ema_state = {}
        self._ema_backup = None
        self._using_ema_weights = False

        # Save hyperparameters for checkpoint reproducibility and experiment traceability.
        self.save_hyperparameters()

    def forward(self, x):
        """Forward proxy to keep Lightning and plain PyTorch usage aligned."""
        return self.model(x)

    def _resolve_run_txt_log_path(self):
        if self._run_txt_log_path is not None:
            return self._run_txt_log_path
        logger = self.trainer.logger if self.trainer is not None else None
        if logger is None:
            return None
        save_dir = getattr(logger, "save_dir", None)
        if not save_dir:
            return None

        run_logs_dir = Path(save_dir)
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        if self.run_txt_log_name:
            log_name = self.run_txt_log_name
        else:
            log_name = f"log_AmbiDOA_{datetime.now().strftime('%Y-%m-%d')}.txt"
            self.run_txt_log_name = log_name
        self._run_txt_log_path = run_logs_dir / log_name
        return self._run_txt_log_path

    def _print_and_append_txt(self, text):
        """Emit a line to terminal and append it to the single run-scoped TXT log."""
        prefix = ""
        if self.run_txt_timestamp:
            prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        line = f"{prefix}{text}"
        self.print(line)
        if self.trainer is not None and not self.trainer.is_global_zero:
            return

        log_path = self._resolve_run_txt_log_path()
        if log_path is None:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _metric_to_float(value):
        """Convert Lightning metric values (tensor/number) to float when possible."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _global_grad_norm(parameters):
        """Compute global L2 norm of all available gradients."""
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        total = grads[0].new_tensor(0.0)
        for g in grads:
            total += g.pow(2).sum()
        return float(torch.sqrt(total).detach().cpu().item())

    def _init_ema_state(self):
        """Initialize EMA shadow with current model parameters."""
        self._ema_state = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
        }

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA shadow after optimizer step."""
        if not self.use_ema or not self._ema_state:
            return
        decay = self.ema_decay
        one_minus = 1.0 - decay
        for name, p in self.model.named_parameters():
            ema_t = self._ema_state.get(name)
            if ema_t is None:
                self._ema_state[name] = p.detach().clone()
                continue
            ema_t.mul_(decay).add_(p.detach(), alpha=one_minus)

    @torch.no_grad()
    def _swap_model_to_ema(self):
        """Swap model parameters to EMA weights for validation/checkpointing."""
        if (
            not self.use_ema
            or self._using_ema_weights
            or not self._ema_state
        ):
            return
        self._ema_backup = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
        }
        for name, p in self.model.named_parameters():
            p.data.copy_(self._ema_state[name])
        self._using_ema_weights = True

    @torch.no_grad()
    def _restore_model_from_ema(self):
        """Restore non-EMA training weights after validation phase."""
        if not self._using_ema_weights or self._ema_backup is None:
            return
        for name, p in self.model.named_parameters():
            p.data.copy_(self._ema_backup[name])
        self._ema_backup = None
        self._using_ema_weights = False

    def on_fit_start(self):
        """Initialize EMA shadow at fit start if requested."""
        if self.use_ema:
            self._init_ema_state()

    def on_train_epoch_start(self):
        """Reset manual-optimization telemetry and restore train weights if needed."""
        self._train_optimizer_steps_epoch = 0
        self._train_grad_norm_sum = 0.0
        self._train_grad_norm_count = 0
        self._train_clip_events = 0
        self._train_step_time_sum = 0.0
        self._train_step_time_count = 0
        self._restore_model_from_ema()

    def _reset_validation_accumulators(self):
        """Prepare lightweight running statistics for epoch-level validation metrics."""
        max_sources = int(self.model.max_sources)
        size = max_sources + 1
        self._val_ang_err_sum = 0.0
        self._val_ang_cnt_sum = 0.0
        self._val_count_err_sum = 0.0
        self._val_count_sample_sum = 0
        self._val_pred_count_hist = torch.zeros(size, dtype=torch.long)
        self._val_per_k_samples = torch.zeros(size, dtype=torch.long)
        self._val_per_k_count_err_sum = torch.zeros(size, dtype=torch.float64)
        self._val_per_k_ang_err_sum = torch.zeros(size, dtype=torch.float64)
        self._val_per_k_ang_cnt_sum = torch.zeros(size, dtype=torch.float64)

    def training_step(self, batch, batch_idx):
        """One optimization step with manual loop control.

        Manual optimization is used on purpose to keep scheduler stepping and
        gradient flow identical to the previous custom training utility.
        """
        # Retrieve optimizer from Lightning (single optimizer expected).
        t0 = perf_counter() if self.log_timing else None
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]

        x_batch, y_count, mask, y_pos = batch
        # Zero gradients only at accumulation boundaries.
        if batch_idx % self.grad_accum_steps == 0:
            opt.zero_grad(set_to_none=True)

        # Forward pass: predict source count and source positions.
        pred_count, pred_pos = self.model(x_batch)

        # Compute combined multi-source loss:
        # - count component
        # - position component with permutation-invariant Hungarian matching.
        total_loss, loss_cnt, loss_pos = multi_source_loss(
            pred_count,
            y_count,
            pred_pos,
            y_pos,
            mask,
            count_loss_type=self.count_loss_type,
            lambda_count=self.lambda_count,
            lambda_pos=self.lambda_pos,
            pos_loss_type=self.pos_loss_type,
            return_components=True,
        )

        # Backpropagate scaled loss so effective gradient matches larger batch.
        scaled_loss = total_loss / float(self.grad_accum_steps)
        self.manual_backward(scaled_loss)

        # Flush optimizer step on accumulation boundary or last batch in epoch.
        num_batches = getattr(self.trainer, "num_training_batches", None) if self.trainer is not None else None
        is_last_batch = isinstance(num_batches, int) and (batch_idx + 1 >= num_batches)
        should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or is_last_batch

        if should_step:
            grad_norm = None
            if self.log_grad_norm or self.grad_clip > 0:
                grad_norm = self._global_grad_norm(self.model.parameters())
                self._train_grad_norm_sum += grad_norm
                self._train_grad_norm_count += 1

            # Optional gradient clipping in norm mode.
            if self.grad_clip > 0:
                if grad_norm is not None and grad_norm > self.grad_clip:
                    self._train_clip_events += 1
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.grad_clip,
                    gradient_clip_algorithm="norm",
                )

            # Parameter update.
            opt.step()
            self._train_optimizer_steps_epoch += 1
            self._optimizer_step_count_total += 1

            # Explicit scheduler step (per optimizer step).
            if self.use_scheduler:
                sch = self.lr_schedulers()
                if isinstance(sch, (list, tuple)):
                    for s in sch:
                        s.step()
                elif sch is not None:
                    sch.step()

            # EMA update after optimizer step.
            if self.use_ema and self._optimizer_step_count_total >= self.ema_update_after_step:
                self._update_ema()

        # Training logs are emitted per-step and aggregated per-epoch.
        # This keeps live monitoring responsive while preserving epoch summaries.
        batch_size = x_batch.size(0)
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/loss_count", loss_cnt, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_pos", loss_pos, on_step=True, on_epoch=True, batch_size=batch_size)

        if self.log_timing and t0 is not None:
            if self.timing_cuda_sync and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dt_ms = (perf_counter() - t0) * 1000.0
            self._train_step_time_sum += dt_ms
            self._train_step_time_count += 1
            if self.timing_log_every_n_steps > 0 and (batch_idx + 1) % self.timing_log_every_n_steps == 0:
                self._print_and_append_txt(
                    f"[Timing] train_step batch={batch_idx + 1} | approx_step_ms={dt_ms:.2f}"
                )
        return total_loss

    def on_validation_epoch_start(self):
        """Reset validation accumulators and swap to EMA weights if enabled."""
        if self.use_ema:
            self._swap_model_to_ema()
        self._reset_validation_accumulators()
        self._val_step_time_sum = 0.0
        self._val_step_time_count = 0
        self._val_step_ms_epoch = None

    def validation_step(self, batch, batch_idx):
        """Validation batch pass without optimization."""
        t0 = perf_counter() if self.log_timing else None
        x_val, y_count, mask, y_pos = batch
        pred_count, pred_pos = self.model(x_val)

        # Keep loss function identical to training for consistent model selection.
        loss, loss_cnt, loss_pos = multi_source_loss(
            pred_count,
            y_count,
            pred_pos,
            y_pos,
            mask,
            count_loss_type=self.count_loss_type,
            lambda_count=self.lambda_count,
            lambda_pos=self.lambda_pos,
            pos_loss_type=self.pos_loss_type,
            return_components=True,
        )

        # Log validation losses on epoch boundaries (less noisy than per-step logs).
        # Keep a single loss key (`val_loss`) to avoid duplicate CSV columns.
        batch_size = x_val.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("val/loss_count", loss_cnt, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/loss_pos", loss_pos, on_step=False, on_epoch=True, batch_size=batch_size)

        max_sources = int(self.model.max_sources)

        # Count metrics and histogram (streaming, CPU side).
        pred_count_cpu = pred_count.detach().reshape(-1).cpu()
        true_count_cpu = y_count.detach().reshape(-1).cpu()
        count_abs_err_cpu = (pred_count_cpu - true_count_cpu).abs()
        self._val_count_err_sum += float(count_abs_err_cpu.sum().item())
        self._val_count_sample_sum += int(count_abs_err_cpu.numel())

        pred_counts_rounded = pred_count_cpu.clamp(min=0, max=max_sources).round().long()
        self._val_pred_count_hist += torch.bincount(
            pred_counts_rounded, minlength=max_sources + 1
        )

        k_true = true_count_cpu.round().clamp(0, max_sources).long()
        self._val_per_k_samples += torch.bincount(k_true, minlength=max_sources + 1)
        per_k_count_err = torch.zeros(max_sources + 1, dtype=torch.float64)
        per_k_count_err.scatter_add_(0, k_true, count_abs_err_cpu.to(torch.float64))
        self._val_per_k_count_err_sum += per_k_count_err

        # PI angular metrics (streaming sums; each sample matched once).
        ang_sum, ang_cnt, per_k_ang_sum, per_k_ang_cnt = angular_error_stats_PI(
            pred_pos.detach(),
            y_pos.detach(),
            mask.detach(),
            true_count=y_count.detach(),
            max_sources=max_sources,
        )
        self._val_ang_err_sum += float(ang_sum.detach().cpu().item())
        self._val_ang_cnt_sum += float(ang_cnt.detach().cpu().item())
        if per_k_ang_sum is not None and per_k_ang_cnt is not None:
            self._val_per_k_ang_err_sum += per_k_ang_sum.detach().cpu().to(torch.float64)
            self._val_per_k_ang_cnt_sum += per_k_ang_cnt.detach().cpu().to(torch.float64)

        if self.log_timing and t0 is not None:
            if self.timing_cuda_sync and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dt_ms = (perf_counter() - t0) * 1000.0
            self._val_step_time_sum += dt_ms
            self._val_step_time_count += 1

        return loss

    def on_validation_epoch_end(self):
        """Compute full-epoch validation metrics and print concise diagnostics."""
        if self._val_count_sample_sum <= 0:
            return

        # Main aggregate metrics from running sums.
        mean_angular_error = self._val_ang_err_sum / max(1e-8, self._val_ang_cnt_sum)
        mean_count_error = self._val_count_err_sum / max(1, self._val_count_sample_sum)

        # Histogram of rounded predicted source counts helps detect count-head bias.
        max_sources = int(self.model.max_sources)
        hist = self._val_pred_count_hist.tolist()

        # Surface key metrics to progress bar and logger.
        self.log("val/angular_error_deg", mean_angular_error, prog_bar=True)
        self.log("val/count_mae", mean_count_error, prog_bar=True)

        if self.log_timing and self._val_step_time_count > 0:
            self._val_step_ms_epoch = self._val_step_time_sum / float(self._val_step_time_count)
            self.log("perf/val_step_ms", self._val_step_ms_epoch, prog_bar=False)

        # Build per-true-count diagnostics (k=number of active sources).
        per_k_lines = []
        for k in range(max_sources + 1):
            n_k = int(self._val_per_k_samples[k].item())
            if n_k <= 0:
                continue
            cnt_mae_k = float(self._val_per_k_count_err_sum[k].item()) / max(1, n_k)
            ang_cnt_k = float(self._val_per_k_ang_cnt_sum[k].item())
            if ang_cnt_k > 0:
                ang_k = float(self._val_per_k_ang_err_sum[k].item()) / ang_cnt_k
            else:
                ang_k = float("nan")
            per_k_lines.append(f"k={k}: N={n_k}, ang={ang_k:.1f}, cntMAE={cnt_mae_k:.2f}")

        # Keep human-readable validation summary in terminal output.
        summary = (
            f"Validation | angular_error={mean_angular_error:.2f} deg | "
            f"count_mae={mean_count_error:.3f} | pred_count_hist={hist}"
        )
        self._print_and_append_txt(summary)

        if per_k_lines:
            self._print_and_append_txt("Per-k:")
            for line in per_k_lines:
                self._print_and_append_txt(f"   {line}")

    def on_train_epoch_end(self):
        """Write concise epoch loss summary to TXT after epoch metrics are finalized."""
        metrics = self.trainer.callback_metrics if self.trainer is not None else {}

        # Prefer epoch-aggregated key; keep fallback for compatibility.
        train_loss_epoch = self._metric_to_float(metrics.get("train/loss_epoch"))
        if train_loss_epoch is None:
            train_loss_epoch = self._metric_to_float(metrics.get("train/loss"))

        val_loss_epoch = self._metric_to_float(metrics.get("val_loss"))

        if train_loss_epoch is None and val_loss_epoch is None:
            return

        mean_grad_norm = None
        if self._train_grad_norm_count > 0:
            mean_grad_norm = self._train_grad_norm_sum / float(self._train_grad_norm_count)
            self.log("train/grad_norm", mean_grad_norm, on_epoch=True, prog_bar=False)
        clip_ratio = None
        if self.grad_clip > 0 and self._train_optimizer_steps_epoch > 0:
            clip_ratio = self._train_clip_events / float(self._train_optimizer_steps_epoch)
            self.log("train/clip_ratio", clip_ratio, on_epoch=True, prog_bar=False)

        train_txt = f"{train_loss_epoch:.6f}" if train_loss_epoch is not None else "n/a"
        val_txt = f"{val_loss_epoch:.6f}" if val_loss_epoch is not None else "n/a"
        grad_txt = f"{mean_grad_norm:.4f}" if mean_grad_norm is not None else "n/a"
        if clip_ratio is None:
            clip_txt = "n/a"
        else:
            clip_txt = f"{clip_ratio:.3f}"
        train_step_ms_txt = "n/a"
        if self.log_timing and self._train_step_time_count > 0:
            train_step_ms = self._train_step_time_sum / float(self._train_step_time_count)
            self.log("perf/train_step_ms", train_step_ms, on_epoch=True, prog_bar=False)
            train_step_ms_txt = f"{train_step_ms:.2f}"
        val_step_ms_txt = "n/a"
        if self.log_timing and self._val_step_ms_epoch is not None:
            val_step_ms_txt = f"{self._val_step_ms_epoch:.2f}"
        epoch_line = (
            f"Epoch {int(self.current_epoch) + 1} | "
            f"train_loss={train_txt} | val_loss={val_txt} | "
            f"grad_norm={grad_txt} | clip_ratio={clip_txt} | "
            f"opt_steps={self._train_optimizer_steps_epoch} | "
            f"train_step_ms={train_step_ms_txt} | val_step_ms={val_step_ms_txt}"
        )
        self._print_and_append_txt(epoch_line)

    def configure_optimizers(self):
        """Create optimizer and optional warmup+cosine scheduler."""
        # AdamW with no-decay exclusions for bias and normalization parameters.
        decay_params = []
        no_decay_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if p.ndim <= 1 or name.endswith(".bias") or ("norm" in lname):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": self.weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)
        if not self.use_scheduler:
            return optimizer

        # Lightning computes estimated stepping batches after dataloader setup.
        # For manual gradient accumulation, convert batch steps -> optimizer steps.
        # Use a safe default when estimate is unavailable (e.g., dry runs).
        total_steps = 1
        if self.trainer is not None:
            steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if isinstance(steps, int) and steps > 0:
                total_steps = max(1, (steps + self.grad_accum_steps - 1) // self.grad_accum_steps)

        # Match legacy warmup heuristic unless warmup_steps is explicitly provided.
        warmup_steps = (
            int(self.warmup_steps)
            if self.warmup_steps is not None
            else min(2000, max(1, total_steps // 10))
        )

        # Scheduler is returned in Lightning format with step-level interval.
        scheduler = build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_fit_end(self):
        """Release backup tensor storage if fit ends while EMA weights are active."""
        if self._using_ema_weights:
            # Keep EMA weights on the model at fit end; just drop raw backup copy.
            self._ema_backup = None
