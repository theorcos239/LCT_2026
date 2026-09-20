"""Penalty-reduced focal loss (CenterNet) с весами каналов.

Sigmoid, а не MSE: значение пика читается как вероятность «точка здесь», и именно
по нему калибруется порог видимости. Вес канала задаёт мягкое маскирование:
0 — точка не размечена, 0.15 — чужая область, 1 — своя.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor, alpha: float = 2.0, beta: float = 4.0,
                       eps: float = 1e-6) -> torch.Tensor:
    """logits, target: (B, C, H, W); weight: (B, C)."""
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    pos = target >= 1.0 - eps                               # пик гауссианы

    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * ~pos

    per_channel = (pos_loss + neg_loss).sum((-2, -1))       # (B, C)
    n_pos = pos.sum((-2, -1)).clamp_min(1)                  # нормировка как в CenterNet
    loss = (per_channel / n_pos * weight).sum()
    return loss / weight.sum().clamp_min(eps)
