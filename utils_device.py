import torch

def pick_device(prefer_cuda: bool = True):
    """
    Zwraca (torch.device, use_amp) z sensownymi ustawieniami backendu.
    - CUDA: włącza TF32 i benchmark dla cuDNN; AMP -> True (float16).
    - MPS: AMP wyłączone (zwykle brak korzyści/stabilności), wszystko w fp32.
    - CPU: AMP wyłączone.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True  # mixed precision dla GPU NVIDII
        # Drobne tuningi wydajności
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # jeśli stała długość wejścia/batch
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False  # na MPS zwykle trzymaj fp32
    else:
        device = torch.device("cpu")
        use_amp = False
    return device, use_amp


def dataloader_kwargs_for(device, num_workers: int = 8):
    """
    Sensowne ustawienia DataLoadera per backend.
    - pin_memory: tylko dla CUDA (szybsze kopiowanie host->GPU).
    - persistent_workers: True, jeśli num_workers>0 (szybsze powtórne epoki).
    """
    # Accept both torch.device and string forms ("cuda", "cuda:0", "mps", "cpu").
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]

    kwargs = dict(num_workers=num_workers, persistent_workers=(num_workers > 0))
    if device_type == "cuda":
        kwargs.update(pin_memory=True)
    else:
        kwargs.update(pin_memory=False)
    return kwargs
