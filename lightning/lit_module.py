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

Additional regularization supported:
- Mean Teacher: EMA model acts as a teacher during training. A position-only
  consistency loss between student and teacher predictions is added to the
  task loss.
- R-Drop: Two student forward passes with different dropout masks are compared.
  A position-only consistency loss between the two outputs is added to the
  task loss.

Both consistency losses are restricted to the position head — applying MSE
consistency on the scalar count head collapses it toward the dataset-mean count.
"""

import copy
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
from train_utils import (
    angular_error_stats_PI,
    build_warmup_cosine_scheduler,
    multi_source_loss,
)


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
        lambda_count=0.1,
        lambda_pos=2.5,
        count_label_smoothing=0.0,
        thresholded_loss_report=False,
        presence_threshold=0.5,
        use_scheduler=True,
        weight_decay=0.0,
        warmup_steps=None,
        grad_clip=0.0,
        grad_accum_steps=1,
        log_grad_norm=True,
        use_ema=False,
        ema_decay=0.999,
        ema_update_after_step=0,
        mean_teacher_weight=0.0,
        rdrop_weight=0.0,
        regularization_every_n_steps=1,
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
            stochastic_depth=0.0,
        )
        if model_kwargs:
            model_defaults.update(model_kwargs)

        # Core AmbiDOA transformer model (query decoder, per-slot presence).
        self.model = AmbiTransformerDOA(**model_defaults)

        # Loss and optimization hyperparameters.
        self.lr = float(lr)
        self.lambda_count = float(lambda_count)
        self.lambda_pos = float(lambda_pos)
        self.count_label_smoothing = float(count_label_smoothing)
        # Diagnostic-only: report train/base_loss and val_loss with a
        # confidence-free ("thresholded") count term so they stay on the same BCE
        # scale (comparable to each other) but do not exhibit the count-BCE
        # confidence up-drift. Affects logging only — not the gradient target and
        # not the checkpoint/early-stop monitor (val_angular_error_deg).
        self.thresholded_loss_report = bool(thresholded_loss_report)
        # Inference-only decision threshold for deriving the integer source count
        # from presence/ordinal logits: count = sum_j 1{sigmoid(logit_j) > thr}.
        # Affects count_mae / pred_count_hist only — NOT the loss, gradient, or
        # weights, so it can be swept post-hoc on any trained checkpoint.
        self.presence_threshold = float(presence_threshold)
        if not (0.0 < self.presence_threshold < 1.0):
            raise ValueError("presence_threshold must be in (0, 1).")

        self.use_scheduler = bool(use_scheduler)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = warmup_steps
        self.grad_clip = float(grad_clip)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.log_grad_norm = bool(log_grad_norm)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_update_after_step = max(0, int(ema_update_after_step))
        self.mean_teacher_weight = float(mean_teacher_weight)
        self.rdrop_weight = float(rdrop_weight)
        self.regularization_every_n_steps = max(1, int(regularization_every_n_steps))
        self.log_timing = bool(log_timing)
        self.timing_log_every_n_steps = max(0, int(timing_log_every_n_steps))
        self.timing_cuda_sync = bool(timing_cuda_sync)
        self.run_txt_log_name = run_txt_log_name
        self.run_txt_timestamp = bool(run_txt_timestamp)
        self._run_txt_log_path = None

        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1).")
        if not (0.0 <= self.count_label_smoothing < 0.5):
            raise ValueError("count_label_smoothing must be in [0.0, 0.5).")
        if self.mean_teacher_weight > 0 and not self.use_ema:
            raise ValueError("mean_teacher_weight > 0 requires use_ema=True.")

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

        # EMA teacher model — created in on_fit_start so it inherits pretrained weights.
        # Auto-registered as a child nn.Module by __setattr__; train() override below
        # keeps it permanently in eval so dropout/etc. don't activate in the teacher.
        self._ema_model = None

        # Save hyperparameters for checkpoint reproducibility and experiment traceability.
        self.save_hyperparameters()

    def forward(self, x):
        """Forward proxy to keep Lightning and plain PyTorch usage aligned."""
        return self.model(x)

    def train(self, mode=True):
        """Force EMA teacher to remain in eval mode whenever the module is toggled.

        Lightning calls `.train(True)` between validation and training epochs;
        without this override the recursion would flip the EMA submodule into
        train mode, activating dropout in the teacher and corrupting the
        mean-teacher consistency target.
        """
        super().train(mode)
        if self._ema_model is not None:
            self._ema_model.eval()
        return self

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

    # -------------------------------------------------------------------------
    # EMA model management
    # -------------------------------------------------------------------------

    def _init_ema_model(self):
        """Create EMA teacher as a detached copy of the current student weights.

        Called in on_fit_start so the teacher starts from the same weights as
        the student — including any pretrained weights loaded before training.
        """
        self._ema_model = copy.deepcopy(self.model)
        for p in self._ema_model.parameters():
            p.requires_grad_(False)
        self._ema_model.eval()
        self._ema_model.to(self.device)

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA teacher parameters after each optimizer step."""
        if not self.use_ema or self._ema_model is None:
            return
        decay = self.ema_decay
        one_minus = 1.0 - decay
        for ema_p, model_p in zip(
            self._ema_model.parameters(), self.model.parameters()
        ):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=one_minus)

    # -------------------------------------------------------------------------
    # Consistency losses
    # -------------------------------------------------------------------------

    def _mean_teacher_loss(self, pred_count, pred_pos, x_batch):
        """Consistency loss between student and EMA teacher predictions.

        Applied to the position head only. Count is intentionally excluded:
        MSE consistency on a scalar regression drives the prediction toward
        its mean under perturbation, which collapses the count head onto the
        dataset-mean count. Position is a direction vector where consistency
        is informative.
        """
        with torch.no_grad():
            _, t_pos = self._ema_model(x_batch)
        return (1.0 - (pred_pos * t_pos).sum(dim=-1)).mean()

    def _rdrop_loss(self, pred_count, pred_pos, x_batch):
        """Consistency loss between two student forward passes with different dropout masks.

        Runs a second forward pass and penalizes deviation from the first
        position output. Gradients flow through both passes — neither output
        is detached. Count is excluded for the same reason as in
        _mean_teacher_loss: MSE consistency on a scalar regresses toward the
        mean instead of sharpening the prediction.
        """
        _, pred_pos2 = self.model(x_batch)
        return (1.0 - (pred_pos * pred_pos2).sum(dim=-1)).mean()

    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------

    def on_fit_start(self):
        """Create EMA teacher at fit start, after pretrained weights are in place."""
        if self.use_ema:
            self._init_ema_model()

    def on_train_epoch_start(self):
        """Reset manual-optimization telemetry at the start of each training epoch."""
        self._train_optimizer_steps_epoch = 0
        self._train_grad_norm_sum = 0.0
        self._train_grad_norm_count = 0
        self._train_clip_events = 0
        self._train_step_time_sum = 0.0
        self._train_step_time_count = 0

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

        Loss composition (when regularizers are enabled):
          combined_loss = task_loss
                        + mean_teacher_weight * mt_loss
                        + rdrop_weight * rdrop_loss

        train/loss logs the combined loss (the actual gradient target).
        train/base_loss logs the task loss only — directly comparable to val_loss.
        """
        t0 = perf_counter() if self.log_timing else None
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]

        x_batch, y_count, mask, y_pos = batch
        # Zero gradients only at accumulation boundaries.
        if batch_idx % self.grad_accum_steps == 0:
            opt.zero_grad(set_to_none=True)

        # --- Student forward pass 1 ---
        pred_count, pred_pos = self.model(x_batch)

        # --- Task loss (per-slot presence count + PI position) ---
        _loss_out = multi_source_loss(
            pred_count,
            y_count,
            pred_pos,
            y_pos,
            mask,
            lambda_count=self.lambda_count,
            lambda_pos=self.lambda_pos,
            count_label_smoothing=self.count_label_smoothing,
            return_components=True,
            return_thresholded_count=self.thresholded_loss_report,
        )
        if self.thresholded_loss_report:
            total_loss, loss_cnt, loss_pos, loss_cnt_thresh = _loss_out
        else:
            total_loss, loss_cnt, loss_pos = _loss_out
            loss_cnt_thresh = None

        # --- Mean Teacher and R-Drop (applied every N steps to reduce compute) ---
        apply_regularization = (batch_idx % self.regularization_every_n_steps == 0)

        mt_loss = pred_count.new_tensor(0.0)
        if apply_regularization and self.mean_teacher_weight > 0 and self._ema_model is not None:
            mt_loss = self._mean_teacher_loss(pred_count, pred_pos, x_batch)

        rdrop_loss = pred_count.new_tensor(0.0)
        if apply_regularization and self.rdrop_weight > 0:
            rdrop_loss = self._rdrop_loss(pred_count, pred_pos, x_batch)

        # --- Combined loss for backward ---
        combined_loss = (
            total_loss
            + self.mean_teacher_weight * mt_loss
            + self.rdrop_weight * rdrop_loss
        )

        # Backpropagate scaled loss so effective gradient matches larger batch.
        scaled_loss = combined_loss / float(self.grad_accum_steps)
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

        # --- Logging ---
        # train/loss = combined loss (actual gradient target, includes mt + rdrop).
        # train/base_loss = task loss only — apples-to-apples with val_loss.
        batch_size = x_batch.size(0)
        self.log("train/loss", combined_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        # train/base_loss uses the confidence-free count term when enabled so it
        # stays comparable to val_loss and does not drift; the true gradient
        # magnitude is train/loss. Falls back to the exact task loss otherwise.
        base_loss_report = (
            self.lambda_count * loss_cnt_thresh + self.lambda_pos * loss_pos
            if self.thresholded_loss_report
            else total_loss
        )
        self.log("train/base_loss", base_loss_report, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train/loss_count", loss_cnt, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_pos", loss_pos, on_step=True, on_epoch=True, batch_size=batch_size)
        if self.mean_teacher_weight > 0:
            self.log("train/loss_mt", mt_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        if self.rdrop_weight > 0:
            self.log("train/loss_rdrop", rdrop_loss, on_step=True, on_epoch=True, batch_size=batch_size)

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
        """Reset validation accumulators at the start of each validation epoch."""
        self._reset_validation_accumulators()
        self._val_step_time_sum = 0.0
        self._val_step_time_count = 0
        self._val_step_ms_epoch = None

    def on_test_epoch_start(self):
        """Reset the (shared) eval accumulators before the held-out test pass."""
        self._reset_validation_accumulators()
        self._val_step_time_sum = 0.0
        self._val_step_time_count = 0
        self._val_step_ms_epoch = None

    def _eval_step(self, batch, stage):
        """Shared validation/test batch pass (no optimization).

        `stage` is "val" or "test": it only selects the log-key prefix and, for
        val, the slash-free checkpoint-monitor aliases. Forward, loss, and metric
        accumulation are identical, so val and test report the same numbers on
        their respective splits. Uses the EMA teacher when available.
        """
        t0 = perf_counter() if self.log_timing else None
        x_val, y_count, mask, y_pos = batch

        # Evaluate with EMA teacher when available; fall back to student model.
        eval_model = self._ema_model if (self.use_ema and self._ema_model is not None) else self.model
        pred_count, pred_pos = eval_model(x_val)

        # Keep loss function identical to training for consistent model selection.
        _val_out = multi_source_loss(
            pred_count,
            y_count,
            pred_pos,
            y_pos,
            mask,
            lambda_count=self.lambda_count,
            lambda_pos=self.lambda_pos,
            count_label_smoothing=self.count_label_smoothing,
            return_components=True,
            return_thresholded_count=self.thresholded_loss_report,
        )
        if self.thresholded_loss_report:
            loss, loss_cnt, loss_pos, loss_cnt_thresh = _val_out
        else:
            loss, loss_cnt, loss_pos = _val_out
            loss_cnt_thresh = None

        batch_size = x_val.size(0)
        max_sources = int(self.model.max_sources)

        # Collapse per-slot presence logits to a scalar predicted count before
        # metrics: k_pred = sum_j 1{ sigmoid(slot_logit_j) > presence_threshold }.
        pred_count_scalar = (
            torch.sigmoid(pred_count.detach()) > self.presence_threshold
        ).sum(dim=-1).float()

        # Count metrics and histogram (streaming, CPU side).
        pred_count_cpu = pred_count_scalar.reshape(-1).cpu()
        true_count_cpu = y_count.detach().reshape(-1).cpu()
        count_abs_err_cpu = (pred_count_cpu - true_count_cpu).abs()

        # --- Reported val_loss (diagnostic only) ---
        # val_loss is NOT the gradient target and NOT the checkpoint/early-stop
        # monitor (that is val_angular_error_deg), so its definition is free to
        # optimize for readability without touching model performance.
        #   * default: the exact validation task loss (count BCE + position cosine).
        #     The count BCE drifts UP under confidence miscalibration even while
        #     count_mae / angular improve.
        #   * thresholded: swap the confidence-sensitive count BCE for the
        #     confidence-free thresholded BCE (same targets, predictions snapped to
        #     the label-smoothing confidence). Same BCE scale — comparable to
        #     train/base_loss — but it moves only with the hard decisions, so no
        #     up-drift. The raw BCE count term is still logged as val/loss_count.
        if self.thresholded_loss_report:
            val_loss_value = self.lambda_pos * loss_pos + self.lambda_count * loss_cnt_thresh
        else:
            val_loss_value = loss

        # Log validation losses on epoch boundaries (less noisy than per-step logs).
        # Keep a single loss key (`val_loss`) to avoid duplicate CSV columns.
        self.log(f"{stage}_loss", val_loss_value, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}/loss_count", loss_cnt, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_pos", loss_pos, on_step=False, on_epoch=True, batch_size=batch_size)

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

    def validation_step(self, batch, batch_idx):
        """Validation batch pass (EMA teacher when available)."""
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Held-out test batch pass — same metrics as validation, on the test split."""
        return self._eval_step(batch, "test")

    def _finalize_eval(self, stage):
        """Aggregate the streaming eval metrics and print a concise summary.

        `stage` in {"val", "test"} selects the log-key prefix, the val-only
        checkpoint-monitor aliases / timing key, and the printed label.
        """
        if self._val_count_sample_sum <= 0:
            return

        # Main aggregate metrics from running sums.
        mean_angular_error = self._val_ang_err_sum / max(1e-8, self._val_ang_cnt_sum)
        mean_count_error = self._val_count_err_sum / max(1, self._val_count_sample_sum)

        # Histogram of rounded predicted source counts helps detect count-head bias.
        max_sources = int(self.model.max_sources)
        hist = self._val_pred_count_hist.tolist()

        # Surface key metrics to progress bar and logger.
        self.log(f"{stage}/angular_error_deg", mean_angular_error, prog_bar=True)
        self.log(f"{stage}/count_mae", mean_count_error, prog_bar=True)
        if stage == "val":
            # Slash-free aliases so ModelCheckpoint.filename can interpolate the
            # monitor value without producing nested subdirectories. Keys in
            # ModelCheckpoint filename templates must not contain '/'.
            self.log("val_angular_error_deg", mean_angular_error)
            self.log("val_count_mae", mean_count_error)

        if stage == "val" and self.log_timing and self._val_step_time_count > 0:
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

        # Keep human-readable summary in terminal output.
        label = "Validation" if stage == "val" else "Test"
        summary = (
            f"{label} | angular_error={mean_angular_error:.2f} deg | "
            f"count_mae={mean_count_error:.3f} | pred_count_hist={hist}"
        )
        self._print_and_append_txt(summary)

        if per_k_lines:
            self._print_and_append_txt("Per-k:")
            for line in per_k_lines:
                self._print_and_append_txt(f"   {line}")

    def on_validation_epoch_end(self):
        """Compute full-epoch validation metrics and print concise diagnostics."""
        self._finalize_eval("val")

    def on_test_epoch_end(self):
        """Compute full-epoch metrics on the held-out test split."""
        self._finalize_eval("test")

    def on_train_epoch_end(self):
        """Write concise epoch loss summary to TXT after epoch metrics are finalized."""
        metrics = self.trainer.callback_metrics if self.trainer is not None else {}

        # Prefer epoch-aggregated key; keep fallback for compatibility.
        train_loss_epoch = self._metric_to_float(metrics.get("train/loss_epoch"))
        if train_loss_epoch is None:
            train_loss_epoch = self._metric_to_float(metrics.get("train/loss"))

        train_base_loss_epoch = self._metric_to_float(metrics.get("train/base_loss_epoch"))
        if train_base_loss_epoch is None:
            train_base_loss_epoch = self._metric_to_float(metrics.get("train/base_loss"))

        val_loss_epoch = self._metric_to_float(metrics.get("val_loss"))
        # Slash-free aliases logged in on_validation_epoch_end so all key val
        # metrics can appear side-by-side in the txt epoch summary. val_loss
        # rises via ordinal BCE confidence drift even while these operational
        # metrics still improve — showing them together prevents killing runs
        # on the wrong signal.
        val_ang_err_epoch = self._metric_to_float(metrics.get("val_angular_error_deg"))
        val_count_mae_epoch = self._metric_to_float(metrics.get("val_count_mae"))

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
        base_txt = f"{train_base_loss_epoch:.6f}" if train_base_loss_epoch is not None else "n/a"
        val_txt = f"{val_loss_epoch:.6f}" if val_loss_epoch is not None else "n/a"
        val_ang_txt = f"{val_ang_err_epoch:.2f}" if val_ang_err_epoch is not None else "n/a"
        val_cmae_txt = f"{val_count_mae_epoch:.3f}" if val_count_mae_epoch is not None else "n/a"
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
            f"train_loss={train_txt} | base_loss={base_txt} | val_loss={val_txt} | "
            f"val_ang_err={val_ang_txt} | val_count_mae={val_cmae_txt} | "
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
