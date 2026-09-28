# -*- coding: utf-8 -*-
"""Кадр -> вход сети. Одна функция и для обучения, и для инференса.

Нарочно без torch: в рабочем образе сеть исполняет onnxruntime, и если бы
предобработка в обучении шла через `F.interpolate`, а в сервисе через PIL,
модель видела бы в сервисе чуть другие пиксели, чем в обучении.

    1. яркость -> [0, 1] по перцентилям 0.5 и 99.5: обучающий экспорт Lunar
       восьмибитный, но другой аппарат отдаст 12 бит — делить на 255 нельзя;
    2. бедро приводится к левому (правое отражается): сеть видит одну сторону;
    3. длинная сторона -> SIZE px (билинейно), остаток холста SIZE x SIZE — нули.
       Кадры DXA 260-405 px, масштаб меняется не больше чем в 1.25 раза.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = 320
MEAN, STD = 0.449, 0.226          # ImageNet, серый канал: энкодер предобучен на нём


def normalize(px: np.ndarray) -> np.ndarray:
    px = np.asarray(px, dtype=np.float32)
    lo, hi = np.percentile(px, (0.5, 99.5))
    if hi - lo < 1e-6:
        lo, hi = float(px.min()), float(max(px.max(), px.min() + 1e-6))
    return np.clip((px - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_input(px: np.ndarray, region: str, size: int = SIZE) -> np.ndarray:
    """Пиксели кадра -> массив (size, size) float32 в [0, 1], без стандартизации."""
    x = normalize(px)
    if region == 'rh':
        x = x[:, ::-1]
    h, w = x.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    x = np.asarray(Image.fromarray(np.ascontiguousarray(x), mode='F')
                   .resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    out = np.zeros((size, size), np.float32)
    out[:nh, :nw] = x
    return out


def valid_box(shape: tuple[int, int], size: int = SIZE) -> tuple[int, int]:
    """Сколько строк и столбцов холста занимает кадр (остальное — дополнение)."""
    h, w = shape
    s = size / max(h, w)
    return max(1, round(h * s)), max(1, round(w * s))
