"""Выбор устройства PyTorch для обучения и проверки модели точек."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    """В auto используем доступный GPU, иначе CPU; явный выбор проверяем."""
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "xpu", "cpu")
                         if available[name])
    elif requested not in available:
        raise ValueError(f"Неизвестное устройство {requested!r}; выберите auto, cuda, mps, xpu или cpu")
    if not available[requested]:
        raise RuntimeError(f"Устройство {requested!r} недоступно в текущем PyTorch")
    return torch.device(requested)
