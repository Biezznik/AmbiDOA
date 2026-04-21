import torch
from datetime import datetime
import json
import os
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from utils_device import dataloader_kwargs_for, pick_device

def pick_device_and_amp(require_cuda: bool = False):
    device, use_amp = pick_device(prefer_cuda=True)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("config.require_cuda=true but CUDA is not available on this machine.")
    return device, use_amp

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "Logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_filename = LOGS_DIR / f"log_AmbiDOA_{datetime.now().strftime('%Y-%m-%d')}.txt"
def log_print(*args, sep=' ', end='\n', file=None):
    text = sep.join(str(a) for a in args) + end
    print(text, end='')
    # Allow Lightning pipeline to fully own run-scoped logging without creating
    # legacy daily files in AmbiDOA/Logs.
    if os.environ.get("AMBIDOA_DISABLE_LEGACY_LOG", "0") == "1":
        return
    try:
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(text)
    except PermissionError as e:
        print(f"[WARN] Legacy log write skipped ({e}).", end='\n')

# --- NEW: safe checkpoint loading + optional model freezing ---
def _extract_model_state(d):
    # Accept common checkpoint key names
    for k in ['state_dict', 'model_state_dict', 'model', 'net', 'weights']:
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    # Fallback when checkpoint stores a raw state_dict directly.
    if isinstance(d, dict):
        return d
    raise ValueError("Unsupported checkpoint format (missing state_dict).")

def _extract_optimizer_state(d):
    for k in ["optimizer_state_dict", "optimizer", "optim_state_dict", "optimizer_state"]:
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    return None

def _strip_prefixes(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        if k.startswith('model.'):
            k = k[len('model.'):]
        out[k] = v
    return out

def load_pretrained_into(model, ckpt_path, device, strict=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = _extract_model_state(ckpt)
    sd = _strip_prefixes(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    log_print(f"[FT] Loaded: {ckpt_path}")
    log_print(f"[FT] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log_print(f"[FT] missing keys (truncated): {missing[:10]}")
    if unexpected:
        log_print(f"[FT] unexpected keys (truncated): {unexpected[:10]}")
    return ckpt

def freeze_params(module):
    for p in module.parameters():
        p.requires_grad = False

def freeze_first_n_encoder_layers(model, n):
    # Support both model.transformer.layers (current) and model.encoder.layers (legacy)
    layers = None
    layers_path = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        layers = model.transformer.layers
        layers_path = "transformer.layers"
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        layers = model.encoder.layers
        layers_path = "encoder.layers"

    if layers is None:
        log_print("[FT][WARN] No encoder layers found (checked transformer.layers, encoder.layers).")
        return

    n = max(0, min(n, len(layers)))
    for i in range(n):
        freeze_params(layers[i])
    if n > 0:
        log_print(f"[FT] Frozen first {n}/{len(layers)} layers from {layers_path}.")

def maybe_freeze_posenc(model):
    # Support current + legacy positional encoding attribute names
    for attr in ['pos_enc', 'posenc', 'pe', 'positional_encoding']:
        if hasattr(model, attr):
            freeze_params(getattr(model, attr))
            log_print(f"[FT] Frozen positional encoding: {attr}")
            return
    log_print("[FT] Positional encoding not found - skip.")

def main():
    from dataset import AmbiDOADataset
    from model import AmbiTransformerDOA
    from train_utils import split_dataset, train_model, debug_overfit_one_batch, split_by_file

    DEBUG_OVERFIT = False
    DEBUG_N = 64

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"--- Log started: {datetime.now()} ---\n\n")

    config_path = Path(__file__).with_name('config.json')
    with open(config_path, encoding='utf-8-sig') as f:
        config = json.load(f)

    manifest_path = config["manifest_path"]
    audio_dir = Path(config["audio_dir"])
    batch_size = config["batch_size"]
    require_cuda = bool(config.get("require_cuda", False))

    # --- NEW: fine-tune / resume parameters ---
    finetune_from = config.get("finetune_from", None)
    reset_optimizer = bool(config.get("reset_optimizer", True))
    freeze_n_encoder_layers = int(config.get("freeze_n_encoder_layers", 0))
    freeze_posenc_flag = bool(config.get("freeze_posenc", False))
    finetune_lr = float(config.get("finetune_lr", 5e-4))

    num_epochs = 5 #20

    device, use_amp = pick_device_and_amp(require_cuda=require_cuda)
    log_print((f"Using device: {device.type} | AMP: {use_amp}"))

    dataset = AmbiDOADataset(manifest_path, audio_dir)
    train_set, val_set = split_by_file(dataset, val_ratio=0.2)

    dl_kwargs = dataloader_kwargs_for(device, num_workers=8)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, **dl_kwargs)

    if DEBUG_OVERFIT:
        idx = list(range(min(DEBUG_N, len(train_set))))
        train_set = Subset(train_set, idx)
        val_set   = Subset(val_set, idx)
        # In debug mode keep simple dataloaders
        train_loader = DataLoader(train_set, batch_size=min(DEBUG_N, 64), shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_set, batch_size=min(DEBUG_N, 64), shuffle=False, num_workers=0)

    model = AmbiTransformerDOA(
        n_channels=16,
        d_model=32, #256
        nhead=4, #8
        num_layers=4, #8
        dim_feedforward=512, #1024
        max_sources=12,
        use_posenc=True,
        dropout=0.003, #0.1
    ).to(device)

    # --- NEW: load checkpoint and prepare fine-tuning ---
    optimizer_state_dict = None
    if finetune_from:
        try:
            ckpt = load_pretrained_into(model, finetune_from, device, strict=False)
            if not reset_optimizer:
                optimizer_state_dict = _extract_optimizer_state(ckpt)
                if optimizer_state_dict is None:
                    log_print("[FT][WARN] reset_optimizer=False but no optimizer state found in checkpoint.")
                else:
                    log_print("[FT] Restoring optimizer state from checkpoint.")
        except Exception as e:
            log_print(f"[FT][ERROR] Could not load {finetune_from}: {repr(e)}")

    if freeze_n_encoder_layers > 0:
        freeze_first_n_encoder_layers(model, freeze_n_encoder_layers)
    if freeze_posenc_flag:
        maybe_freeze_posenc(model)

    if DEBUG_OVERFIT:
        debug_overfit_one_batch(model, train_loader, device=device, steps=300, lr=1e-3)
        # short training on the same subset
        train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-3,
                    device=device, use_amp=use_amp,
                    count_loss_type='regression',
                    lambda_count=0.25, lambda_pos=1.0,
                    pos_loss_type='cosine',
                    grad_clip=0.1,
                    use_scheduler=True,
                    reset_optimizer=reset_optimizer,
                    optimizer_state_dict=optimizer_state_dict,
                    weight_decay=0.0)
        return

    # --- NEW: fine-tune with separate LR, fresh optimizer (default) ---
    # If train_model supports passing optimizer/scheduler/start_epoch, extend it here.
    lr_to_use = finetune_lr if finetune_from else 5e-4

    train_model(model, train_loader, val_loader,
                num_epochs=num_epochs, lr=lr_to_use,
                device=device, use_amp=use_amp,
                count_loss_type='regression',
                lambda_count=0.1, lambda_pos=2.5,
                pos_loss_type='cosine',
                grad_clip=0,
                use_scheduler=True,
                reset_optimizer=reset_optimizer,
                optimizer_state_dict=optimizer_state_dict,
                weight_decay=0.0)

if __name__ == "__main__":
    main()
