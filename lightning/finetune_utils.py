"""Utilities for checkpoint loading and selective freezing during fine-tuning.

These helpers keep all fine-tune-specific logic outside of the training script,
making checkpoint format handling and layer-freezing behavior easier to test and
document independently.
"""

from pathlib import Path

import torch


def _extract_model_state(d):
    """Extract model state dict from common checkpoint layouts.

    Supports direct state_dict checkpoints and nested forms used by training
    frameworks (`state_dict`, `model_state_dict`, etc.).
    """
    for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]
    if isinstance(d, dict):
        return d
    raise ValueError("Unsupported checkpoint format: missing model state dict.")


def _strip_prefixes(state_dict):
    """Normalize common key prefixes from DataParallel/Lightning checkpoints."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        out[k] = v
    return out


def resolve_ckpt_path(path_like, project_root):
    """Resolve checkpoint path from absolute or project-relative input."""
    path = Path(path_like)
    if path.exists():
        return path.resolve()
    rel_path = (Path(project_root) / path).resolve()
    if rel_path.exists():
        return rel_path
    raise FileNotFoundError(f"Checkpoint not found: {path_like}")


def load_pretrained_into(model, ckpt_path, strict=False, map_location="cpu"):
    """Load pretrained weights into model with robust key normalization.

    Returns loaded checkpoint object for optional downstream inspection.
    """
    # `weights_only=False` is intentionally defaulted here because checkpoint
    # contents may include metadata required by _extract_model_state.
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    sd = _strip_prefixes(_extract_model_state(ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[FT] Loaded: {ckpt_path}")
    print(f"[FT] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[FT] missing keys (truncated): {missing[:10]}")
    if unexpected:
        print(f"[FT] unexpected keys (truncated): {unexpected[:10]}")
    return ckpt


def freeze_params(module):
    """Disable gradients for all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False


def freeze_first_n_encoder_layers(model, n):
    """Freeze first `n` encoder layers for staged fine-tuning.

    Supports both:
    - model.transformer.layers  (standard independent-layer encoder)
    - model.encoder.layers

    Shared-weight encoder (_SharedEncoder):
      All passes through the stack use the same single layer, so partial
      freezing is not meaningful. Any n > 0 freezes the shared layer entirely.
      n == 0 leaves it trainable.
    """
    n = int(n)

    # --- Shared-weight encoder: single .layer attribute, no .layers list ---
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "layer") and not hasattr(transformer, "layers"):
        if n > 0:
            freeze_params(transformer.layer)
            print(
                f"[FT] Shared encoder: frozen entirely "
                f"(freeze_n_encoder_layers={n} with shared weights freezes the whole encoder)."
            )
        return

    # --- Standard independent-layer encoder ---
    layers = None
    layer_path = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        layers = model.transformer.layers
        layer_path = "transformer.layers"
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        layers = model.encoder.layers
        layer_path = "encoder.layers"

    if layers is None:
        print("[FT][WARN] No encoder layers found (checked transformer.layers, encoder.layers).")
        return

    # Clamp requested number of frozen layers to valid range.
    n = max(0, min(n, len(layers)))
    for i in range(n):
        freeze_params(layers[i])
    if n > 0:
        print(f"[FT] Frozen first {n}/{len(layers)} layers from {layer_path}.")


def maybe_freeze_posenc(model):
    """Freeze positional encoding module when present under known attribute names."""
    for attr in ["pos_enc", "posenc", "pe", "positional_encoding"]:
        if hasattr(model, attr):
            freeze_params(getattr(model, attr))
            print(f"[FT] Frozen positional encoding: {attr}")
            return
    print("[FT] Positional encoding not found - skip.")
