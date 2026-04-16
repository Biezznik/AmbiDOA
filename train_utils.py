import contextlib
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from AmbiDOA import log_print            # Twoje adne logowanie z timestampem
from datetime import datetime            # do ewentualnego stempla w nazwie pliku
from pathlib import Path
from typing import Optional

# Prbujemy wczy Hungarian algorithm (optymalne parowanie predykcji z etykietami),
# eby liczy strat i bd ktowy Permutation Invariant (kolejno rde nie ma znaczenia).
try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ------------------------- Straty pozycji (dwch typw) -------------------------

def _pos_loss_cosine(pred_m, true_m):
    """
    Strata kosinusowa dla pozycji:
      - liczymy podobiestwo cosinusowe (czyli kt) midzy wektorami (x,y,z)
      - wynik to 1 - cos(theta) (bliej 0 -> lepiej)
    pred_m, true_m: [k, 3] (k = liczba aktywnych rde w danym przykadzie)
    """
    pred_n = pred_m / (pred_m.norm(dim=1, keepdim=True) + 1e-8)
    true_n = true_m / (true_m.norm(dim=1, keepdim=True) + 1e-8)
    cos = (pred_n * true_n).sum(dim=1).clamp(-1, 1)      # cosinus midzy parami wektorw
    return (1.0 - cos).mean()

def _pos_loss_mse(pred_m, true_m):
    """
    Zwyka MSE (bd redniokwadratowy) w przestrzeni (x,y,z).
    Dziaa, ale przy wektorach na sferze czsto cosine lepiej koresponduje z ktem.
    """
    return ((pred_m - true_m) ** 2).sum(dim=1).mean()    


# ---------------------- Parowanie predykcji do prawdy (Permutation Invariant) ----------------------

def _solve_assignment_from_cost(cost):
    """
    Rozwiązuje problem przypisania dla macierzy kosztów [S, k].
    Zwraca listę indeksów predykcji w kolejności kolumn GT (0..k-1).
    """
    if isinstance(cost, torch.Tensor):
        S, k = int(cost.size(0)), int(cost.size(1))
    else:
        S, k = int(cost.shape[0]), int(cost.shape[1])

    if k == 0:
        return []
    if k == 1:
        # Fast-path: bez Hungariana, wystarczy minimum po jedynej kolumnie GT.
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

    # Fallback zachłanny (bez SciPy).
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
    Batchowe wyznaczanie dopasowań PI.

    Optymalizacja:
      - matching nie potrzebuje gradientów -> liczymy na odłączonych tensorach,
      - przy SciPy robimy transfer GPU->CPU raz per batch (zamiast raz per próbkę).
    """
    B, S, _ = pred_pos.shape
    with torch.no_grad():
        pred_det = pred_pos.detach()
        true_det = true_pos.detach()
        mask_det = mask.detach()

        if _HAVE_SCIPY:
            # SciPy działa na CPU: pojedynczy transfer na batch redukuje overhead.
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
            true_active = true_n[b, active, :]           # [k,3]
            cost = 1.0 - pred_n[b] @ true_active.T       # [S,k]
            out.append(_solve_assignment_from_cost(cost))
    return out


def _match_one_sample(pred_S3, true_S3, mask_S):
    """
    Dla JEDNEGO przykadu (pojedynczy element batcha) czymy przewidywane sloty [S,3]
    z aktywnymi prawdziwymi rdami (tam, gdzie maska == 1).
    Cel: dopasowa ktry predykowany slot odpowiada ktremu rdu
         tak, eby bd by minimalny (niezaleno od kolejnoci).

    Wejcie:
      pred_S3:  [S,3]  przewidywane pozycje (wszystkie sloty)
      true_S3:  [S,3]  pozycje GT w ukadzie slotw (tylko cz slotw aktywna)
      mask_S:   [S]    1 tam, gdzie slot jest aktywny w GT, 0 inaczej

    Dziaanie:
      1) Wycigamy aktywne GT.
      2) Budujemy macierz kosztw: cost = 1 - cos(pred, GT)  (im mniejszy, tym lepiej).
      3) Uywamy algorytmu wgierskiego (Hungarian), eby znale pary z minimalnym kosztem,
         albo zachann alternatyw, jeli scipy nie jest dostpne.
      4) Zwracamy list indeksw predykowanych slotw w kolejnoci rde GT (dla k aktywnych).
    """
    return _match_batch(
        pred_S3.unsqueeze(0),
        true_S3.unsqueeze(0),
        mask_S.unsqueeze(0),
    )[0]


# --------------------------- Gwna funkcja straty dla multi-source ---------------------------

def multi_source_loss(
    pred_count, true_count,
    pred_pos, true_pos, mask,
    count_loss_type='regression',
    lambda_count=0.5, lambda_pos=1.5,      # wagi skadowych: liczba vs pozycje
    pos_loss_type='cosine',                # 'cosine' (polecane) albo 'mse'
    return_components: bool = False,       # czy zwrci te skadowe do logowania
):
    """
    czna strata:
      = (waga) * bd liczby rde
      + (waga) * bd pozycji (liczony tylko na aktywnych slotach i z dopasowaniem par).

    Wejcia:
      pred_count: [B,1] (regresja liczby)
      true_count: [B]   (rzeczywista liczba rde)
      pred_pos:   [B,S,3]  przewidywane wektory (sloty 0..S-1)
      true_pos:   [B,S,3]  prawda (aktywnych jest tylko k z S)
      mask:       [B,S]    1 dla aktywnych slotw GT

    Zwraca:
      total_loss (i opcjonalnie: loss_count, loss_pos jako liczby do logowania)
    """
    if count_loss_type != "regression":
        raise ValueError(
            f"Unsupported count_loss_type='{count_loss_type}'. "
            "Current AmbiDOA head predicts scalar counts, so only 'regression' is supported."
        )

    # (1) Strata dla liczby rde (regresja):
    # Huber (SmoothL1) jest odporniejszy na odlege wartoci ni MSE.
    loss_count = nn.SmoothL1Loss(beta=1.0)(pred_count.squeeze(), true_count.squeeze().float())

    # (2) Strata pozycji  P.I. (po dopasowaniu par) i tylko na aktywnych:
    B, S, _ = pred_pos.shape
    loss_pos_sum = pred_pos.new_tensor(0.0)   # sumujemy strat waon k
    active_sum   = 0                           # sumujemy ile aktywnych (dla redniej)
    match_idx_batch = _match_batch(pred_pos, true_pos, mask)

    for b in range(B):
        active = mask[b] > 0.5
        k = int(active.sum().item())          # ile rde w tym przykadzie
        if k == 0:
            continue
        idx_pred = match_idx_batch[b]
        if len(idx_pred) != k:
            continue  # (awaryjnie) jeli nie udao si sparowa do k

        idx_pred_t = torch.as_tensor(idx_pred, dtype=torch.long, device=pred_pos.device)
        pred_m = pred_pos[b].index_select(0, idx_pred_t)   # [k,3] dopasowane predykcje
        true_m = true_pos[b, active, :]                    # [k,3] aktywne GT

        # Wybr funkcji bdu (ktowy = cosine, ewentualnie MSE):
        if pos_loss_type == 'cosine':
            lp = _pos_loss_cosine(pred_m, true_m)
        else:
            lp = _pos_loss_mse(pred_m, true_m)

        loss_pos_sum += lp * k
        active_sum   += k

    loss_pos = loss_pos_sum / (float(active_sum) + 1e-8)    # rednia po aktywnych rdach

    # (3) Suma waona skadowych
    total_loss = lambda_count * loss_count + lambda_pos * loss_pos

    if return_components:
        return total_loss, loss_count.detach(), loss_pos.detach()
    else:
        return total_loss


# --------------------------- Bd ktowy P.I. (tylko do ewaluacji) ---------------------------

def angular_error_deg_PI(pred_pos, true_pos, mask):
    """
    redni bd ktowy w STOPNIACH, *po dopasowaniu par* (Permutation Invariant).
    Czyli mierzymy prawdziwe podobiestwo kierunkw, niezalenie od kolejnoci slotw.
    """
    errs_sum, cnt_sum, _, _ = angular_error_stats_PI(
        pred_pos, true_pos, mask, true_count=None, max_sources=None
    )
    return (errs_sum / (cnt_sum + 1e-8)).item()


def angular_error_stats_PI(pred_pos, true_pos, mask, true_count=None, max_sources=None):
    """
    Zwraca statystyki bdu ktowego PI w postaci sum (bez utraty precyzji na rednich):
      - errs_sum: suma bdw ktowych [deg] po wszystkich aktywnych rdach
      - cnt_sum:  liczba aktywnych rde (mianownik do redniej)

    Opcjonalnie (gdy podasz true_count i max_sources) zwraca te sumy per-k:
      - per_k_err_sum[k]: suma bdw ktowych dla prbek o prawdziwej liczbie rde = k
      - per_k_cnt_sum[k]: liczba aktywnych rde dla prbek o k

    Dzięki temu można agregować metryki strumieniowo batch-po-batch, bez trzymania
    całej walidacji w pamięci.
    """
    B, S, _ = pred_pos.shape
    errs_sum = pred_pos.new_tensor(0.0)
    cnt_sum  = pred_pos.new_tensor(0.0)

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
        ang = torch.rad2deg(torch.acos(dot))           # [k] w stopniach
        ang_sum = ang.sum()
        errs_sum += ang_sum
        cnt_sum  += k

        if per_k_err_sum is not None and per_k_cnt_sum is not None and k_true is not None:
            kk = int(k_true[b].item())
            per_k_err_sum[kk] += ang_sum
            per_k_cnt_sum[kk] += k

    return errs_sum, cnt_sum, per_k_err_sum, per_k_cnt_sum


# --------------------------- Szybki overfit jednego batcha (debug) ---------------------------

def debug_overfit_one_batch(model, loader, device='cuda', steps=300, lr=1e-3, count_loss_type='regression'):
    """
    Minimalny test: czy model potrafi nauczy si na pami jednego batcha?
    Jeli nie  zwykle problem tkwi w danych/etykietach/stracie, a nie w pojemnoci modelu.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if count_loss_type != "regression":
        raise ValueError(
            f"Unsupported count_loss_type='{count_loss_type}' in debug_overfit_one_batch. "
            "Current AmbiDOA count head supports only regression."
        )

    # Pobieramy pierwszy batch (X: audio, y_count: liczba, mask: aktywne sloty, y_pos: pozycje)
    X, y_count, mask, y_pos = next(iter(loader))
    X, y_count, mask, y_pos = X.to(device), y_count.to(device), mask.to(device), y_pos.to(device)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        pred_count, pred_pos = model(X)

        # Strata  prosta wersja (bez P.I.), tylko do szybkiej diagnostyki:
        loss_count = nn.MSELoss()(pred_count.squeeze(), y_count.squeeze().float())

        diff = (pred_pos - y_pos).pow(2).sum(dim=-1) * mask   # MSE po aktywnych
        loss_pos = diff.sum() / (mask.sum() + 1e-8)

        loss = loss_count + loss_pos
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1 or step == steps:
            # Orientacyjny bd ktowy (bez parowania)  tylko jako sygna postpu
            with torch.no_grad():
                ang_err = vector_angular_error_deg(pred_pos, y_pos, mask)
            log_print(f"[Overfit] step {step:4d} | loss={loss.item():.4f} "
                      f"| loss_cnt={loss_count.item():.4f} | loss_pos={loss_pos.item():.4f} "
                      f"| ang_err={ang_err:.2f}")


# --------------------------- Podzia danych na train/val ---------------------------

def split_dataset(dataset, val_ratio=0.2, seed=42):
    """
    Najprostszy split losowo po oknach.
    Uwaga: jeli ten sam plik audio jest pocity na wiele okien, to ich losowy podzia
           moe woy jedne okna do train, a inne do val  lekkie przecieki.
    Zwykle lepiej uy split_by_file (poniej), eby rozdziela cae pliki.
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
    Split po plikach (zalecany): wszystkie okna z danego pliku trafiaj do tego samego zbioru.
    Zmniejsza ryzyko uciekania informacji midzy train i val.
    Wymaga, by dataset.windows[i]['audio_path'] istniao.
    """
    file_ids = [w['audio_path'] for w in dataset.windows]
    files = np.array(file_ids)
    indices = np.arange(len(dataset))
    unique_files = np.unique(files)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_files)
    n_val_files = int(len(unique_files) * val_ratio)
    val_files = set(unique_files[:n_val_files])

    train_idx = indices[[f not in val_files for f in files]]
    val_idx   = indices[[f in  val_files for f in files]]

    from torch.utils.data import Subset
    if log_fn is not None:
        log_fn(
            f"Frames (train/val): {len(train_idx)} / {len(val_idx)} | files: "
            f"{len(unique_files)-n_val_files} / {n_val_files}"
        )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# --------------------------- Prosty bd ktowy (bez parowania) ---------------------------

def vector_angular_error_deg(pred_pos, true_pos, mask):
    """
    redni bd ktowy midzy predykcj a GT slot-po-slots (bez dopasowania par).
    Uyteczne do szybkiej diagnostyki; do waciwej ewaluacji lepszy jest angular_error_deg_PI.
    """
    pred_norm = pred_pos / (pred_pos.norm(dim=2, keepdim=True) + 1e-8)
    true_norm = true_pos / (true_pos.norm(dim=2, keepdim=True) + 1e-8)
    dot = (pred_norm * true_norm).sum(dim=2)  # [B, S]
    dot = torch.clamp(dot, -1.0, 1.0)
    ang = torch.acos(dot)                     # radiany
    deg = torch.rad2deg(ang)
    masked_deg = deg * mask                   # liczymy tylko tam, gdzie maska==1
    mean_deg = masked_deg.sum() / (mask.sum() + 1e-8)
    return mean_deg.item()


# --------------------------- Scheduler: warmup + cosine decay ---------------------------

def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=5000):
    """
    Klasyczny harmonogram LR:
      - najpierw warmup: LR ronie linowo od ~0 do 1x (stabilizuje start),
      - potem cosine decay: LR agodnie maleje do koca treningu.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return max(1e-8, float(current_step + 1) / float(max(1, warmup_steps)))
        # Cosine decay od koca warm-upu do koca treningu:
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------- Gwna ptla treningowa ---------------------------

def train_model(model, train_loader, val_loader,
                num_epochs=30, lr=5e-4,
                device: torch.device = None,
                use_amp: bool = False,                # Mixed Precision (CUDA) = szybciej, mniej RAM
                count_loss_type='regression',
                lambda_count=0.1, lambda_pos=0.9,     # wagi skadowych w total loss
                pos_loss_type='cosine',
                grad_clip: float = 5,               # przycinanie gradientw (0.0 = wyczone)
                use_scheduler: bool = True,
                weight_decay: float = 0.0,           # L2 regularization (kept configurable)
                reset_optimizer: bool = True,
                optimizer_state_dict: Optional[dict] = None,
                ):
    # 0) Wybr urzdzenia (gdy nie podano): cuda > mps > cpu
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
                              else 'cpu')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if not reset_optimizer:
        if optimizer_state_dict is None:
            log_print("[FT][WARN] reset_optimizer=False but optimizer_state_dict is missing. Using fresh optimizer.")
        else:
            optimizer.load_state_dict(optimizer_state_dict)
            # Ensure optimizer states are on the same device as the model parameters.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            log_print("[FT] Optimizer state restored.")

    # 1) Scheduler LR (opcjonalny): warmup + cosine
    if use_scheduler:
        total_steps = num_epochs * max(1, len(train_loader))
        scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=min(2000, total_steps//10))
    else:
        scheduler = None

    # 2) AMP (CUDA)  przyspieszenie i mniejsze uycie VRAM. Na MPS/CPU w praktyce wyczamy.
    amp_enabled = (use_amp and device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)

    def autocast_ctx():
        # Kontekst automatycznej precyzji (float16/bfloat16) tylko dla CUDA
        if amp_enabled:
            return torch.amp.autocast('cuda')
        return contextlib.nullcontext()

    best_val_loss = float('inf')    # bdziemy zapisywa najlepszy model (najnisza Val Loss)

    # --- ptla po epokach ---
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

        # --- trening po batchach ---
        for X_batch, y_count, mask, y_pos in pbar:
            X_batch = X_batch.to(device); y_count = y_count.to(device)
            mask    = mask.to(device);    y_pos   = y_pos.to(device)

            optimizer.zero_grad(set_to_none=True)
            with (torch.amp.autocast('cuda') if amp_enabled else contextlib.nullcontext()):
                pred_count, pred_pos = model(X_batch)
                # czny loss + skadowe do logw:
                total_loss, loss_cnt, loss_pos = multi_source_loss(
                    pred_count, y_count, pred_pos, y_pos, mask,
                    count_loss_type=count_loss_type,
                    lambda_count=lambda_count, lambda_pos=lambda_pos,
                    pos_loss_type=pos_loss_type,
                    return_components=True
                )

            # Krok optymalizacji (AMP/bez AMP):
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
            pbar.set_postfix(loss=f"{total_loss.item():.3f}",
                             loss_cnt=f"{loss_cnt.item():.3f}",
                             loss_pos=f"{loss_pos.item():.3f}")
            
        train_loss /= len(train_loader.dataset)

        # --- walidacja (bez gradientw) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            all_pred_pos, all_true_pos, all_mask = [], [], []
            all_count_errors = []
            pred_counts_raw, true_counts_raw = [], []

            for X_val, y_count, mask, y_pos in val_loader:
                X_val = X_val.to(device); y_count = y_count.to(device)
                mask  = mask.to(device);  y_pos   = y_pos.to(device)

                pred_count, pred_pos = model(X_val)
                loss = multi_source_loss(
                    pred_count, y_count, pred_pos, y_pos, mask,
                    count_loss_type=count_loss_type,
                    lambda_count=lambda_count, lambda_pos=lambda_pos,
                    pos_loss_type=pos_loss_type,
                    return_components=False
                )
                val_loss += loss.item() * X_val.size(0)

                # Zbieramy rzeczy do metryk po caej walidacji:
                all_pred_pos.append(pred_pos.cpu()); all_true_pos.append(y_pos.cpu()); all_mask.append(mask.cpu())
                pred_counts_raw.append(pred_count.squeeze().cpu()); true_counts_raw.append(y_count.squeeze().cpu())
                all_count_errors.append((pred_count.squeeze()-y_count.squeeze()).abs().cpu())

            # Sklejamy wektory z caej walidacji:
            all_pred_pos = torch.cat(all_pred_pos)
            all_true_pos = torch.cat(all_true_pos)
            all_mask     = torch.cat(all_mask)

            # Bd ktowy z dopasowaniem par (Permutation Invariant)
            mean_angular_error = angular_error_deg_PI(all_pred_pos, all_true_pos, all_mask)

            # Bd liczby rde (MAE) + rozkad przewidzianej liczby
            pred_counts_raw  = torch.cat(pred_counts_raw)
            true_counts_raw  = torch.cat(true_counts_raw)
            mean_count_error = torch.cat(all_count_errors).mean().item()

            # Histogram zaokrglonej liczby rde (eby wiedzie, do czego przykleja si model)
            pred_counts_rounded = pred_counts_raw.clamp(min=0, max=model.max_sources).round().int()
            hist = torch.bincount(pred_counts_rounded, minlength=model.max_sources+1).tolist()

            # ----- per-k metryki (diagnostyka trudnoci dla rnych liczebnoci rde) -----
            k_true = true_counts_raw.round().clamp(0, model.max_sources).int()
            per_k_lines = []
            for k in range(model.max_sources + 1):
                sel = (k_true == k)
                if sel.any():
                    ang_k = angular_error_deg_PI(all_pred_pos[sel], all_true_pos[sel], all_mask[sel])
                    cnt_mae_k = (pred_counts_raw[sel] - true_counts_raw[sel]).abs().mean().item()
                    per_k_lines.append(f"k={k}: N={int(sel.sum())}, ang={ang_k:.1f}, cntMAE={cnt_mae_k:.2f}")
            if per_k_lines:
                log_print("Per-k:")
                log_print(*["   " + l for l in per_k_lines], sep='\n')


        val_loss /= len(val_loader.dataset)

        # Epoch summary:
        log_print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Mean angular error: {mean_angular_error:.2f} deg | "
            f"Mean source-count error: {mean_count_error:.2f} | "
            f"Predicted-count histogram: {hist}"
        )

        # Zapis najlepszego modelu:
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_str2 = datetime.now().strftime("%Y_%m_%d")
            save_dir = Path(__file__).resolve().parent / "models" / date_str2
            save_dir.mkdir(parents=True, exist_ok=True)
            model_filename = save_dir / f"best_model_{date_str}.pth"

            torch.save(model.state_dict(), str(model_filename))
            log_print(f">>> Model saved: {model_filename}")
