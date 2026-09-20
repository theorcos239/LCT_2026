"""Отрисовка гауссиан и декодирование пика в координату."""
from __future__ import annotations

import numpy as np
import torch


def render(coords: np.ndarray, visible: np.ndarray, shape: tuple[int, int],
           sigma: float) -> np.ndarray:
    """Гауссианы в каналах с visible=True; остальные каналы — нули."""
    h, w = shape
    out = np.zeros((len(coords), h, w), dtype=np.float32)
    r = int(np.ceil(3 * sigma))
    for i, ((x, y), vis) in enumerate(zip(coords, visible)):
        if not vis:
            continue
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
        gx = np.arange(x0, x1, dtype=np.float32)[None, :] - x
        out[i, y0:y1, x0:x1] = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
        # Точка лежит между пикселями, и максимум гауссианы тогда меньше 1.
        # Явная единица в ближайшем пикселе нужна, чтобы у канала был положительный пиксель.
        yi, xi = int(round(float(y))), int(round(float(x)))
        if 0 <= yi < h and 0 <= xi < w:
            out[i, yi, xi] = 1.0
    return out


def decode(heatmaps: torch.Tensor, window: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    """Пик + soft-argmax в окне вокруг него.

    Возвращает координаты (B, C, 2) в порядке (x, y) и уверенность (B, C) — значение пика.
    """
    b, c, h, w = heatmaps.shape
    flat = heatmaps.flatten(2)
    conf, idx = flat.max(dim=2)
    py, px = torch.div(idx, w, rounding_mode="floor"), idx % w

    r = window // 2
    ys = torch.arange(-r, r + 1, device=heatmaps.device)
    dy, dx = torch.meshgrid(ys, ys, indexing="ij")
    gy = (py[..., None, None] + dy).clamp(0, h - 1)
    gx = (px[..., None, None] + dx).clamp(0, w - 1)

    patch = flat.gather(2, (gy * w + gx).flatten(2)).view(b, c, window, window)
    weights = patch.clamp_min(0)
    total = weights.sum((-2, -1), keepdim=True).clamp_min(1e-6)
    weights = weights / total
    x = (weights.sum(-2) * gx[..., 0, :].float()).sum(-1)
    y = (weights.sum(-1) * gy[..., :, 0].float()).sum(-1)
    return torch.stack([x, y], dim=-1), conf
