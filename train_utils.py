import math
import numpy as np
import torch
import torch.nn.functional as F
from AmbiDOA import log_print

# Try to import Hungarian assignment for optimal permutation-invariant matching.
try:
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ------------------------- Position loss -------------------------


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


# --------------------------- Main multi-source loss ---------------------------


def _thresholded_count_bce(logits, targets, count_label_smoothing, weight=None):
    """BCE of the presence targets against THRESHOLDED predictions.

    Snaps every predicted probability to a fixed confidence level (the label-
    smoothing level, or 0.05 when smoothing is off) according to the >0.5
    decision, then computes BCE against the same targets. This removes the
    confidence degree of freedom from the reported count loss: it moves only with
    the hard presence decisions, not with calibration — so it does not exhibit the
    BCE confidence up-drift, while staying on the same scale and units as the real
    count BCE. Detached — intended for logging only, never gradient.
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
    lambda_count=0.5,
    lambda_pos=1.5,
    count_label_smoothing: float = 0.0,
    return_components: bool = False,
    return_matches: bool = False,
    return_thresholded_count: bool = False,
):
    """
    Weighted multi-task loss (per-slot presence count + PI cosine position):
        total = lambda_count * presence_bce + lambda_pos * position_loss_PI

    Inputs:
      pred_count: [B, max_sources] per-slot presence logits.
      true_count: [B] (unused for the loss itself; count supervision comes from the
                  PI-matched presence targets, kept in the signature for callers).
      pred_pos:   [B, S, 3]
      true_pos:   [B, S, 3]
      mask:       [B, S] (1 for active GT slots)

    Presence target[b, s] = 1 iff slot s was PI-matched to a real GT source in
    sample b, else 0. This structurally couples the count signal to the position
    head's slot usage: a "used" slot must both hold a source (BCE=1) and localize
    it well (cosine loss on the matched pair). Count is derived downstream as
    sum_j 1{sigmoid(presence_j) > threshold}.

    count_label_smoothing: {0, 1} BCE targets become {eps, 1-eps}, curbing the
    train-time confidence runaway that drives val BCE up while val/count_mae stays
    stable. return_thresholded_count additionally returns a confidence-free BCE for
    drift-free base_loss/val_loss reporting (see _thresholded_count_bce).
    """
    B, S, _ = pred_pos.shape
    # PI matching is needed once — for the position loss and the presence targets.
    match_idx_batch = _match_batch(pred_pos, true_pos, mask)

    # Head 1: per-slot presence BCE against the PI-matched assignment.
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
    loss_count_thresh = (
        _thresholded_count_bce(pred_count, presence_target, count_label_smoothing)
        if return_thresholded_count else None
    )

    # Head 2: PI position loss (cosine), computed only over active matched sources.
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
        loss_pos_sum += _pos_loss_cosine(pred_m, true_m) * k
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


# --------------------------- PI angular metrics ---------------------------


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


# --------------------------- Dataset splitting ---------------------------


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
