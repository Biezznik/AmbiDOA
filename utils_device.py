import torch


def pick_device(prefer_cuda: bool = True):
    """
    Return (torch.device, use_amp) with practical backend defaults.
    - CUDA: enable TF32 and cuDNN benchmark; AMP enabled (float16).
    - MPS: AMP disabled (typically limited benefit/stability), fp32 path.
    - CPU: AMP disabled.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True  # mixed precision for NVIDIA GPUs
        # Small performance tuning defaults.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # best when input shape is stable
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False  # keep fp32 on MPS by default
    else:
        device = torch.device("cpu")
        use_amp = False
    return device, use_amp


def dataloader_kwargs_for(device, num_workers: int = 8):
    """
    Build DataLoader kwargs per backend.
    - pin_memory: only for CUDA (faster host->GPU transfer).
    - persistent_workers: True when num_workers>0 (faster between epochs).
    """
    # Accept both torch.device and string forms ("cuda", "cuda:0", "mps", "cpu").
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]

    kwargs = dict(num_workers=num_workers, persistent_workers=(num_workers > 0))
    if device_type == "cuda":
        kwargs.update(pin_memory=True)
    else:
        kwargs.update(pin_memory=False)
    return kwargs
