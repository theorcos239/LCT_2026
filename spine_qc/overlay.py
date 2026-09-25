# -*- coding: utf-8 -*-
"""Визуализация найденного на кадре позвоночника.

ТЗ 2.6: «дополнительная серия с визуализацией обнаруженного нарушения на
изображении: маска, контур, ключевые точки или тепловая карта». Рисуем то, на
чём построен вердикт, а не абстрактную тепловую карту: ось с измеренным углом,
зоны гребней подвздошных костей, контуры найденных посторонних структур.
Оператор должен видеть, что именно посчитала программа, и за секунду решить,
согласен ли он.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .criteria import analyze
from .geometry import mm2px, spine_column, theil_sen

OK = (80, 210, 120)
BAD = (240, 90, 90)
HINT = (90, 170, 255)


def render_overlay(img: np.ndarray, result: dict | None = None, scale: int = 2) -> Image.Image:
    """Кадр + разметка вердикта. `result` — из criteria.analyze, иначе считается."""
    col = spine_column(img)
    res = result or analyze(img)
    c = res['criteria']

    base = Image.fromarray(col.img).convert('RGB')
    base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(base)
    h, w = col.shape

    # ось: измеренная прямая по центрам тел
    b = col.valid
    if b.sum() >= 20:
        y = np.arange(h, dtype=float)
        a, q = theil_sen(y[b], col.center[b])
        y0, y1 = float(y[b].min()), float(y[b].max())
        colr = BAD if c['axis']['violated'] else OK
        d.line([((a * y0 + q) * scale, y0 * scale), ((a * y1 + q) * scale, y1 * scale)],
               fill=colr, width=2)
        d.line([(col.center_x * scale, y0 * scale), (col.center_x * scale, y1 * scale)],
               fill=HINT, width=1)
        ang = c['axis']['angle_deg']
        if ang is not None:
            d.text((4, 4), f'ось {ang:+.1f}°', fill=colr)

    # зоны гребней подвздошных костей
    gap = mm2px(35.0)
    colr = BAD if c['position']['violated'] else OK
    for x0, x1 in ((0, max(0, col.center_x - gap)), (min(w, col.center_x + gap), w)):
        if x1 - x0 > 2:
            d.rectangle([x0 * scale, int(h * 0.5) * scale, x1 * scale - 1, h * scale - 1],
                        outline=colr, width=1)
    d.text((4, 16), f"гребни {c['position']['iliac_area']:.3f}", fill=colr)

    # посторонние структуры
    art = c['artifacts']
    if art['violated']:
        d.text((4, 28), f"артефакт {art['length_mm']:.0f} мм", fill=BAD)

    if res['violations']:
        d.text((4, h * scale - 14), ' | '.join(res['violations'])[:120], fill=BAD)
    return base
