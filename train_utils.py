import contextlib
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from AmbiDOA import log_print
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import Hungarian assignment for optimal permutation-invariant matching.
try:
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ------------------------- Position losses -------------------------


def _pos_loss_cosine(pred_m, true_m):
    """
    Cosine position loss:
    - normalize vectors to unit length,
    - compute 1 - cos(theta) per matched source pair,
    - return mean over active pairs.
    """
    pred_n = pred_m / (pred_m.norm(dim=1, keepdim=True) + 1e-8)
    true_n = true_m / (true_m.norm(dim=1, keepdim=True) + 1e-8)
    cos = (pred_n * true_n).sum(dim=1).clamp(-1, 1)
    return (1.0 - cos).mean()


def _pos_loss_mse(pred_m, true_m):
    """
    MSE position loss in Cartesian (x, y, z) space.
    """
    return ((pred_m - true_m) ** 2).sum(dim=1).mean()


# ---------------------- Permutation-invariant matching ----------------------


def _solve_assignment_from_cost(cost):
    """
    Solve assignment for cost matrix [S, k].
    Returns predicted slot indices aligned to GT column order (0..k-1).
    """
    if isinstance(cost, torch.Tensor):
        S, k = int(cost.size(0)), int(cost.size(1))
    else:
        S, k = int(cost.shape[0]), int(cost.shape[1])

    if k == 0:
        return []
    if k == 1:
        # Fast path for single active source.
        if isinstance(cost, torch.Tensor):
            return [int(torch.argmin(cost[:, 0]).item())]
        return [int(np.argmin(cost[:, 0]))]

    if _HAVE_SCIPY:
        if isinstance(cost, torch.Tensor):
            if cost.device.type != "cpu":
                cost = cost.cpu()
            C = cost.numpy()
        else:
            C = cost
        r, c = linear_sum_assignment(C)
        order = np.argsort(c)
        return [int(r[i]) for i in order]

    # Greedy fallback when SciPy is unavailable.
    if isinstance(cost, torch.Tensor):
        if cost.device.type != "cpu":
            cost = cost.cpu()
        C = cost.numpy()
    else:
        C = cost

    used = np.zeros(S, dtype=np.bool_)
    pred_idx = []
    for j in range(k):
        for ri in np.argsort(C[:, j]).tolist():
            if not used[ri]:
                used[ri] = True
                pred_idx.append(int(ri))
                break
    return pred_idx


def _match_batch(pred_pos, true_pos, mask):
    """
    Compute PI matching for a full batch.

    Optimization notes:
    - matching does not require gradients, so detached tensors are used,
    - when SciPy is enabled, do GPU->CPU transfer once per batch.
    """
    B, S, _ = pred_pos.shape
    with torch.no_grad():
        pred_det = pred_pos.detach()
        true_det = true_pos.detach()
        mask_det = mask.detach()

        if _HAVE_SCIPY:
            if pred_det.device.type != "cpu":
                pred_det = pred_det.cpu()
            if true_det.device.type != "cpu":
                true_det = true_det.cpu()
            if mask_det.device.type != "cpu":
                mask_det = mask_det.cpu()

        pred_n = pred_det / (pred_det.norm(dim=2, keepdim=True) + 1e-8)
        true_n = true_det / (true_det.norm(dim=2, keepdim=True) + 1e-8)

        out = []
        for b in range(B):
            active = mask_det[b] > 0.5
            k = int(active.sum().item())
            if k == 0:
                out.append([])
                continue
            true_active = true_n[b, active, :]  # [k, 3]
            cost = 1.0 - pred_n[b] @ true_active.T  # [S, k]
            out.append(_solve_assignment_from_cost(cost))
    return out


def _match_one_sample(pred_S3, true_S3, mask_S):
    """
    Single-sample wrapper around batch matching.
    """
    return _match_batch(
        pred_S3.unsqueeze(0),
        true_S3.unsqueeze(0),
        mask_S.unsqueeze(0),
    )[0]


# --------------------------- Main multi-source loss ---------------------------


def _thresholded_count_bce(logits, targets, count_label_smoothing, weight=None):
    """BCE of the count/presence targets against THRESHOLDED predictions.

    Snaps every predicted probability to a fixed confidence level (the label-
    smoothing level, or 0.05 when smoothing is off) according to the >0.5
    decision, then computes BCE against the same targets. This removes the
    confidence degree of freedom from the reported count loss: it moves only with
    the hard presence/ordinal decisions, not with calibration — so it does not
    exhibit the BCE confidence up-drift, while staying on the same scale and units
    as the real count BCE. Detached — intended for logging only, never gradient.
    """
    snap_eps = max(float(count_label_smoothing), 0.05)
    with torch.no_grad():
        p = torch.sigmoid(logits.detach())
        p_snap = torch.where(
            p > 0.5,
            p.new_full((), 1.0 - snap_eps),
            p.new_full((), snap_eps),
        )
        return F.binary_cross_entropy(p_snap, targets, weight=weight)


def multi_source_loss(
    pred_count,
    true_count,
    pred_pos,
    true_pos,
    mask,
    count_loss_type="regression",
    lambda_count=0.5,
    lambda_pos=1.5,
    pos_loss_type="cosine",
    count_label_smoothing: float = 0.0,
    count_per_k_weights: Optional[torch.Tensor] = None,
    return_components: bool = False,
    return_matches: bool = False,
    return_thresholded_count: bool = False,
):
    """
    Weighted multi-task loss:
      total = lambda_count * count_loss + lambda_pos * position_loss_PI

    Inputs:
      pred_count: [B,1] in "regression" mode, [B, max_sources] of logits in
                  "ordinal" mode, [B, max_sources] per-slot presence logits in
                  "presence" mode.
      true_count: [B]
      pred_pos:   [B,S,3]
      true_pos:   [B,S,3]
      mask:       [B,S] (1 for active GT slots)

    count_label_smoothing: applied in "ordinal" and "presence" modes. Hard
      {0, 1} BCE targets become {ε, 1-ε}. Prevents the train-time confidence
      runaway that drives val BCE up while val/count_mae stays stable. Optimal
      BCE floor becomes -(1-ε)·log(1-ε) - ε·log(ε) (≈ 0.20 at ε=0.05) instead of 0.

    count_per_k_weights: only applied in "ordinal" mode. Tensor of length
      max_sources+1 giving a per-sample weight based on the sample's true k.
      Alternative to a class-balanced sampler: reweights the BCE contribution
      of each sample by w_k so under-represented counts get amplified gradient.
      Suggested formula: w_k = 1 / sqrt(freq_k). None disables (baseline).
      Undefined and rejected in "presence" mode (presence BCE is naturally
      balanced by the mask ratio k/S vs (S-k)/S per sample).
    """
    # PI matching is needed once — for the position loss always, and for the
    # presence-mode head-1 loss as well. Compute up-front and reuse.
    B, S, _ = pred_pos.shape
    match_idx_batch = _match_batch(pred_pos, true_pos, mask)

    # Optional confidence-free ("thresholded") count loss for reporting. Filled in
    # per mode below only when requested; stays None otherwise.
    loss_count_thresh = None

    if count_loss_type == "regression":
        # SmoothL1 is more robust than plain MSE on the scalar target.
        loss_count = nn.SmoothL1Loss(beta=1.0)(pred_count.squeeze(), true_count.squeeze().float())
        if return_thresholded_count:
            # SmoothL1 is not confidence-sensitive, so it does not drift; the
            # thresholded report is the same value.
            loss_count_thresh = loss_count
    elif count_loss_type == "ordinal":
        # Ordinal cumulative encoding: head j predicts P(k >= j+1) via BCE.
        # Targets shape [B, max_sources]: targets[b, j] = 1 iff true_count[b] >= j+1.
        # This sidesteps the SmoothL1 mean-regression collapse on uncertain k.
        max_sources = pred_count.size(-1)
        thresholds = torch.arange(1, max_sources + 1, device=pred_count.device, dtype=pred_count.dtype)
        targets = (true_count.reshape(-1, 1).float() >= thresholds.reshape(1, -1)).to(pred_count.dtype)
        if count_label_smoothing > 0.0:
            eps = float(count_label_smoothing)
            targets = targets * (1.0 - 2.0 * eps) + eps
        if count_per_k_weights is not None:
            # Per-sample weight w_k[b], broadcast [B, 1] over BCE's [B, max_sources].
            # F.BCE(..., weight=w, reduction='mean') = mean(w * bce_per_elem), so
            # w == 1 everywhere reproduces the un-weighted baseline exactly.
            weights_dev = count_per_k_weights.to(device=pred_count.device, dtype=pred_count.dtype)
            k_idx = true_count.reshape(-1).round().clamp(0, max_sources).long()
            sample_weight = weights_dev.index_select(0, k_idx).reshape(-1, 1)
            loss_count = F.binary_cross_entropy_with_logits(
                pred_count, targets, weight=sample_weight
            )
            if return_thresholded_count:
                loss_count_thresh = _thresholded_count_bce(
                    pred_count, targets, count_label_smoothing, weight=sample_weight
                )
        else:
            loss_count = F.binary_cross_entropy_with_logits(pred_count, targets)
            if return_thresholded_count:
                loss_count_thresh = _thresholded_count_bce(
                    pred_count, targets, count_label_smoothing
                )
    elif count_loss_type == "presence":
        # Per-slot presence logits. Target[b, s] = 1 iff slot s was PI-matched to
        # a real GT source in sample b, else 0. Structurally couples the count
        # signal to the position head's slot usage: a "used" slot must both hold
        # a source (BCE=1) AND localize it well (cosine loss on the matched pair).
        if count_per_k_weights is not None:
            raise ValueError(
                "count_per_k_weights is undefined in 'presence' mode; presence BCE is "
                "naturally balanced by the mask. Set count_per_k_weights=None."
            )
        presence_target = pred_count.new_zeros((B, S))
        for b in range(B):
            active = mask[b] > 0.5
            k = int(active.sum().item())
            if k == 0:
                continue
            idx_pred = match_idx_batch[b]
            if len(idx_pred) != k:
                continue
            idx_pred_t = torch.as_tensor(idx_pred, dtype=torch.long, device=pred_pos.device)
            presence_target[b].index_fill_(0, idx_pred_t, 1.0)
        if count_label_smoothing > 0.0:
            eps = float(count_label_smoothing)
            presence_target = presence_target * (1.0 - 2.0 * eps) + eps
        loss_count = F.binary_cross_entropy_with_logits(pred_count, presence_target)
        if return_thresholded_count:
            loss_count_thresh = _thresholded_count_bce(
                pred_count, presence_target, count_label_smoothing
            )
    else:
        raise ValueError(
            f"Unsupported count_loss_type='{count_loss_type}'. "
            "Expected 'regression', 'ordinal', or 'presence'."
        )

    # PI position loss: compute only over active matched sources. Reuses the
    # match_idx_batch computed above.
    loss_pos_sum = pred_pos.new_tensor(0.0)
    active_sum = 0

    for b in range(B):
        active = mask[b] > 0.5
        k = int(active.sum().item())
        if k == 0:
            continue
        idx_pred = match_idx_batch[b]
        if len(idx_pred) != k:
            continue

        idx_pred_t = torch.as_tensor(idx_pred, dtype=torch.long, device=pred_pos.device)
        pred_m = pred_pos[b].index_select(0, idx_pred_t)
        true_m = true_pos[b, active, :]

        if pos_loss_type == "cosine":
            lp = _pos_loss_cosine(pred_m, true_m)
        else:
            lp = _pos_loss_mse(pred_m, true_m)

        loss_pos_sum += lp * k
        active_sum += k

    loss_pos = loss_pos_sum / (float(active_sum) + 1e-8)
    total_loss = lambda_count * loss_count + lambda_pos * loss_pos

    if return_thresholded_count:
        # Confidence-free count loss appended for drift-free base_loss/val_loss
        # reporting. Used together with return_components.
        if return_matches:
            return (total_loss, loss_count.detach(), loss_pos.detach(),
                    match_idx_batch, loss_count_thresh.detach())
        return total_loss, loss_count.detach(), loss_pos.detach(), loss_count_thresh.detach()
    if return_components and return_matches:
        return total_loss, loss_count.detach(), loss_pos.detach(), match_idx_batch
    if return_components:
        return total_loss, loss_count.detach(), loss_pos.detach()
    if return_matches:
        return total_loss, match_idx_batch
    return total_loss


# ----- Auxiliary count-consistency loss (Strategy #3) -----


def position_consistency_count_loss(
    pred_count,
    pred_pos,
    true_pos,
    mask,
    count_loss_type: str,
    match_idx_batch=None,
    cos_threshold: float = 0.5,
):
    """Auxiliary loss tying the count head to position-quality evidence.

    Forces the count head's prediction to agree with the number of position
    slots the model has *actually localized* (PI-matched and within cos_threshold
    of GT). Early in training, n_good < k_gt, pushing the count head not to
    claim sources it cannot place. Late in training, n_good → k_gt and the term
    vanishes naturally — no scheduling needed.

    Differs from the primary count loss: primary targets k_gt regardless of
    position quality, this targets the number of *well-localized* matched
    slots. Provides a second gradient path to the count head grounded in the
    position output rather than only the [CLS] global pool.

    match_idx_batch: optional pre-computed PI matches from multi_source_loss to
      avoid running Hungarian twice per training step. When None, computed here.
    """
    B, S, _ = pred_pos.shape

    # Differentiable scalar prediction from the count head.
    if count_loss_type == "regression":
        pred_count_scalar = pred_count.reshape(B, -1).squeeze(-1).float()
    elif count_loss_type == "ordinal":
        # Expected count = Σ_j P(k>=j+1); differentiable, no thresholding.
        pred_count_scalar = torch.sigmoid(pred_count).sum(dim=-1)
    else:
        raise ValueError(
            f"Unsupported count_loss_type='{count_loss_type}'. Expected 'regression' or 'ordinal'."
        )

    if match_idx_batch is None:
        match_idx_batch = _match_batch(pred_pos, true_pos, mask)

    # Non-differentiable target: number of slots with cos(pred, gt) > τ after matching.
    with torch.no_grad():
        pred_n = pred_pos / (pred_pos.norm(dim=2, keepdim=True) + 1e-8)
        true_n = true_pos / (true_pos.norm(dim=2, keepdim=True) + 1e-8)
        n_good = pred_pos.new_zeros(B)
        for b in range(B):
            active = mask[b] > 0.5
            k = int(active.sum().item())
            if k == 0:
                continue
            idx_pred = match_idx_batch[b]
            if len(idx_pred) != k:
                continue
            idx_pred_t = torch.as_tensor(idx_pred, dtype=torch.long, device=pred_pos.device)
            cos_per_pair = (pred_n[b].index_select(0, idx_pred_t) * true_n[b, active, :]).sum(dim=-1)
            n_good[b] = (cos_per_pair > cos_threshold).float().sum()

    return F.smooth_l1_loss(pred_count_scalar, n_good)


# --------------------------- PI angular metrics ---------------------------


def angular_error_deg_PI(pred_pos, true_pos, mask):
    """
    Mean angular error in degrees after PI matching.
    """
    errs_sum, cnt_sum, _, _ = angular_error_stats_PI(
        pred_pos, true_pos, mask, true_count=None, max_sources=None
    )
    return (errs_sum / (cnt_sum + 1e-8)).item()


def angular_error_stats_PI(pred_pos, true_pos, mask, true_count=None, max_sources=None):
    """
    Return PI angular statistics as sums:
      - errs_sum: sum of angular errors [deg]
      - cnt_sum:  number of active sources

    Optional per-k outputs (when true_count and max_sources are provided):
      - per_k_err_sum[k]
      - per_k_cnt_sum[k]
    """
    B, S, _ = pred_pos.shape
    errs_sum = pred_pos.new_tensor(0.0)
    cnt_sum = pred_pos.new_tensor(0.0)

    per_k_err_sum = None
    per_k_cnt_sum = None
    k_true = None
    if true_count is not None:
        if max_sources is None:
            max_sources = S
        per_k_err_sum = pred_pos.new_zeros(int(max_sources) + 1)
        per_k_cnt_sum = pred_pos.new_zeros(int(max_sources) + 1)
        k_true = true_count.reshape(-1).round().clamp(0, int(max_sources)).long()

    match_idx_batch = _match_batch(pred_pos, true_pos, mask)

    for b in range(B):
        active = mask[b] > 0.5
        k = int(active.sum().item())
        if k == 0:
            continue
        idx_pred = match_idx_batch[b]
        if len(idx_pred) != k:
            continue

        idx_pred_t = torch.as_tensor(idx_pred, dtype=torch.long, device=pred_pos.device)
        pred_m = pred_pos[b].index_select(0, idx_pred_t)
        true_m = true_pos[b, active, :]

        pred_n = pred_m / (pred_m.norm(dim=1, keepdim=True) + 1e-8)
        true_n = true_m / (true_m.norm(dim=1, keepdim=True) + 1e-8)
        dot = (pred_n * true_n).sum(dim=1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(dot))
        ang_sum = ang.sum()
        errs_sum += ang_sum
        cnt_sum += k

        if per_k_err_sum is not None and per_k_cnt_sum is not None and k_true is not None:
            kk = int(k_true[b].item())
            per_k_err_sum[kk] += ang_sum
            per_k_cnt_sum[kk] += k

    return errs_sum, cnt_sum, per_k_err_sum, per_k_cnt_sum


# --------------------------- One-batch overfit debug ---------------------------


def debug_overfit_one_batch(model, loader, device="cuda", steps=300, lr=1e-3, count_loss_type="regression"):
    """
    Fast sanity check: can the model overfit a single batch?
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if count_loss_type != "regression":
        raise ValueError(
            f"Unsupported count_loss_type='{count_loss_type}' in debug_overfit_one_batch. "
            "Current AmbiDOA count head supports only regression."
        )

    # Fetch first batch once.
    X, y_count, mask, y_pos = next(iter(loader))
    X, y_count, mask, y_pos = X.to(device), y_count.to(device), mask.to(device), y_pos.to(device)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        pred_count, pred_pos = model(X)

        # Simpler debug loss (without PI matching) for quick diagnostics.
        loss_count = nn.MSELoss()(pred_count.squeeze(), y_count.squeeze().float())
        diff = (pred_pos - y_pos).pow(2).sum(dim=-1) * mask
        loss_pos = diff.sum() / (mask.sum() + 1e-8)

        loss = loss_count + loss_pos
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1 or step == steps:
            # Approximate angular error without matching, for trend tracking only.
            with torch.no_grad():
                ang_err = vector_angular_error_deg(pred_pos, y_pos, mask)
            log_print(
                f"[Overfit] step {step:4d} | loss={loss.item():.4f} "
                f"| loss_cnt={loss_count.item():.4f} | loss_pos={loss_pos.item():.4f} "
                f"| ang_err={ang_err:.2f}"
            )


# --------------------------- Dataset splitting ---------------------------


def split_dataset(dataset, val_ratio=0.2, seed=42):
    """
    Random window-level split.
    Note: if one file yields many windows, this split can leak file content
    between train/val. Prefer split_by_file when possible.
    """
    n = len(dataset)
    log_print(f"Total frames in dataset: {n}")
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    from torch.utils.data import Subset

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_by_file(dataset, val_ratio=0.2, seed=42, log_fn=log_print):
    """
    File-level split: all windows from one audio file stay in one split.
    Requires dataset.windows[i]["audio_path"] to be available.
    """
    file_ids = [w["audio_path"] for w in dataset.windows]
    files = np.array(file_ids)
    indices = np.arange(len(dataset))
    unique_files = np.unique(files)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_files)
    n_val_files = int(len(unique_files) * val_ratio)
    val_files = set(unique_files[:n_val_files])

    train_idx = indices[[f not in val_files for f in files]]
    val_idx = indices[[f in val_files for f in files]]

    from torch.utils.data import Subset

    if log_fn is not None:
        log_fn(
            f"Frames (train/val): {len(train_idx)} / {len(val_idx)} | files: "
            f"{len(unique_files) - n_val_files} / {n_val_files}"
        )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_by_file_train_val_test(dataset, val_ratio=0.2, test_ratio=0.1, seed=42, log_fn=log_print):
    """File-level 3-way split: train / val / test, with no window leakage.

    All windows from one audio file stay within a single split. The test files
    are carved out first, then val, then the remainder is train — so a held-out
    test set (files disjoint from both train and val) is available for unbiased
    final evaluation. Deterministic under `seed`.
    """
    file_ids = [w["audio_path"] for w in dataset.windows]
    files = np.array(file_ids)
    indices = np.arange(len(dataset))
    unique_files = np.unique(files)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_files)
    n_files = len(unique_files)
    n_test_files = int(n_files * test_ratio)
    n_val_files = int(n_files * val_ratio)
    test_files = set(unique_files[:n_test_files])
    val_files = set(unique_files[n_test_files:n_test_files + n_val_files])

    in_test = np.array([f in test_files for f in files])
    in_val = np.array([f in val_files for f in files])
    test_idx = indices[in_test]
    val_idx = indices[in_val]
    train_idx = indices[~(in_test | in_val)]

    from torch.utils.data import Subset

    if log_fn is not None:
        log_fn(
            f"Frames (train/val/test): {len(train_idx)} / {len(val_idx)} / {len(test_idx)} | "
            f"files: {n_files - n_val_files - n_test_files} / {n_val_files} / {n_test_files}"
        )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# --------------------------- Non-PI angular error ---------------------------


def vector_angular_error_deg(pred_pos, true_pos, mask):
    """
    Mean angular error in degrees without matching (slot-to-slot).
    Useful for quick diagnostics; PI metrics are preferred for evaluation.
    """
    pred_norm = pred_pos / (pred_pos.norm(dim=2, keepdim=True) + 1e-8)
    true_norm = true_pos / (true_pos.norm(dim=2, keepdim=True) + 1e-8)
    dot = (pred_norm * true_norm).sum(dim=2)  # [B, S]
    dot = torch.clamp(dot, -1.0, 1.0)
    ang = torch.acos(dot)
    deg = torch.rad2deg(ang)
    masked_deg = deg * mask  # evaluate only active slots
    mean_deg = masked_deg.sum() / (mask.sum() + 1e-8)
    return mean_deg.item()


# --------------------------- Scheduler: warmup + cosine decay ---------------------------


def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=5000):
    """
    Learning-rate schedule:
      - linear warmup from ~0 to base LR,
      - cosine decay until the end of training.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return max(1e-8, float(current_step + 1) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------- Legacy training loop ---------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=30,
    lr=5e-4,
    device: torch.device = None,
    use_amp: bool = False,  # mixed precision (CUDA)
    count_loss_type="regression",
    lambda_count=0.1,
    lambda_pos=0.9,
    pos_loss_type="cosine",
    grad_clip: float = 5,  # set 0.0 to disable clipping
    use_scheduler: bool = True,
    weight_decay: float = 0.0,  # keep configurable for future use
    reset_optimizer: bool = True,
    optimizer_state_dict: Optional[dict] = None,
):
    # 0) Device selection fallback: cuda > mps > cpu.
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if not reset_optimizer:
        if optimizer_state_dict is None:
            log_print("[FT][WARN] reset_optimizer=False but optimizer_state_dict is missing. Using fresh optimizer.")
        else:
            optimizer.load_state_dict(optimizer_state_dict)
            # Ensure optimizer tensors are on the same device as model parameters.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            log_print("[FT] Optimizer state restored.")

    # 1) Optional LR scheduler: warmup + cosine.
    if use_scheduler:
        total_steps = num_epochs * max(1, len(train_loader))
        scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=min(2000, total_steps // 10))
    else:
        scheduler = None

    # 2) AMP enabled only on CUDA.
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    def autocast_ctx():
        if amp_enabled:
            return torch.amp.autocast("cuda")
        return contextlib.nullcontext()

    best_val_loss = float("inf")

    # Epoch loop.
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")

        # Training batches.
        for X_batch, y_count, mask, y_pos in pbar:
            X_batch = X_batch.to(device)
            y_count = y_count.to(device)
            mask = mask.to(device)
            y_pos = y_pos.to(device)

            optimizer.zero_grad(set_to_none=True)
            with (torch.amp.autocast("cuda") if amp_enabled else contextlib.nullcontext()):
                pred_count, pred_pos = model(X_batch)
                total_loss, loss_cnt, loss_pos = multi_source_loss(
                    pred_count,
                    y_count,
                    pred_pos,
                    y_pos,
                    mask,
                    count_loss_type=count_loss_type,
                    lambda_count=lambda_count,
                    lambda_pos=lambda_pos,
                    pos_loss_type=pos_loss_type,
                    return_components=True,
                )

            # Optimizer step with/without AMP.
            if amp_enabled:
                scaler.scale(total_loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += total_loss.item() * X_batch.size(0)
            pbar.set_postfix(
                loss=f"{total_loss.item():.3f}",
                loss_cnt=f"{loss_cnt.item():.3f}",
                loss_pos=f"{loss_pos.item():.3f}",
            )

        train_loss /= len(train_loader.dataset)

        # Validation loop.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            all_pred_pos, all_true_pos, all_mask = [], [], []
            all_count_errors = []
            pred_counts_raw, true_counts_raw = [], []

            for X_val, y_count, mask, y_pos in val_loader:
                X_val = X_val.to(device)
                y_count = y_count.to(device)
                mask = mask.to(device)
                y_pos = y_pos.to(device)

                pred_count, pred_pos = model(X_val)
                loss = multi_source_loss(
                    pred_count,
                    y_count,
                    pred_pos,
                    y_pos,
                    mask,
                    count_loss_type=count_loss_type,
                    lambda_count=lambda_count,
                    lambda_pos=lambda_pos,
                    pos_loss_type=pos_loss_type,
                    return_components=False,
                )
                val_loss += loss.item() * X_val.size(0)

                # Cache tensors for full-epoch metrics.
                all_pred_pos.append(pred_pos.cpu())
                all_true_pos.append(y_pos.cpu())
                all_mask.append(mask.cpu())
                pred_counts_raw.append(pred_count.squeeze().cpu())
                true_counts_raw.append(y_count.squeeze().cpu())
                all_count_errors.append((pred_count.squeeze() - y_count.squeeze()).abs().cpu())

            # Concatenate epoch tensors.
            all_pred_pos = torch.cat(all_pred_pos)
            all_true_pos = torch.cat(all_true_pos)
            all_mask = torch.cat(all_mask)

            # PI angular error metric.
            mean_angular_error = angular_error_deg_PI(all_pred_pos, all_true_pos, all_mask)

            # Count MAE and rounded-count histogram.
            pred_counts_raw = torch.cat(pred_counts_raw)
            true_counts_raw = torch.cat(true_counts_raw)
            mean_count_error = torch.cat(all_count_errors).mean().item()
            pred_counts_rounded = pred_counts_raw.clamp(min=0, max=model.max_sources).round().int()
            hist = torch.bincount(pred_counts_rounded, minlength=model.max_sources + 1).tolist()

            # Per-k diagnostics (k = true source count).
            k_true = true_counts_raw.round().clamp(0, model.max_sources).int()
            per_k_lines = []
            for k in range(model.max_sources + 1):
                sel = k_true == k
                if sel.any():
                    ang_k = angular_error_deg_PI(all_pred_pos[sel], all_true_pos[sel], all_mask[sel])
                    cnt_mae_k = (pred_counts_raw[sel] - true_counts_raw[sel]).abs().mean().item()
                    per_k_lines.append(f"k={k}: N={int(sel.sum())}, ang={ang_k:.1f}, cntMAE={cnt_mae_k:.2f}")
            if per_k_lines:
                log_print("Per-k:")
                log_print(*["   " + l for l in per_k_lines], sep="\n")

        val_loss /= len(val_loader.dataset)

        # Epoch summary.
        log_print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Mean angular error: {mean_angular_error:.2f} deg | "
            f"Mean source-count error: {mean_count_error:.2f} | "
            f"Predicted-count histogram: {hist}"
        )

        # Save best checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_str2 = datetime.now().strftime("%Y_%m_%d")
            save_dir = Path(__file__).resolve().parent / "models" / date_str2
            save_dir.mkdir(parents=True, exist_ok=True)
            model_filename = save_dir / f"best_model_{date_str}.pth"

            torch.save(model.state_dict(), str(model_filename))
            log_print(f">>> Model saved: {model_filename}")
