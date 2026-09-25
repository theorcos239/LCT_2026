# -*- coding: utf-8 -*-
"""Геометрия кадра поясничного отдела: колонна позвоночника и её окружение.

Все три критерия позвоночника (ось, укладка, артефакты) считаются по одному и
тому же разбору кадра, поэтому он вынесен сюда и делается один раз:

    col = spine_column(img)     # маска, коридор, края колонны по строкам
    col.angle_deg               # дальше — criteria.py

Ничего не обучается и не случайно: одинаковый вход даёт побитово одинаковый
выход. Общие примитивы изображения (Оцу, маска кости, скользящее среднее)
переиспользуются из `hip_roi.geometry` — они не специфичны для бедра, а
дублировать проверенный код ради красивого дерева импортов смысла нет.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from hip_roi.geometry import bone_mask, bone_runs, moving_average, to_uint8

# --------------------------------------------------------------------------- #
#  Масштаб и пороги ТЗ
# --------------------------------------------------------------------------- #
# Кадр позвоночника у GE Lunar Prodigy Advance — 300 px в ширину. По тегу
# ExposedArea на 74 кадрах, где он согласован с размером, 180 мм / 300 px:
# медиана 0.600 по ширине и 0.606 по высоте. Совпадает с 0.607 у кадра бедра
# (hip_roi), то есть это масштаб аппарата, а не режима.
MM_PER_PX = 0.600

# «правильно выровненная ось позвоночника (допустимый наклон до 5 градусов)»
AXIS_LIMIT_DEG = 5.0

# Полоса строк, по которой оценивается колонна: края кадра отрезаются, потому
# что сверху в кадр попадают рёбра, снизу — крестец и таз, и ни то ни другое к
# оси поясничного отдела отношения не имеет.
BAND = (0.10, 0.90)


def mm2px(mm: float) -> int:
    return max(1, int(round(mm / MM_PER_PX)))


def px2mm(px: float) -> float:
    return float(px) * MM_PER_PX


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (y * y + x * x) <= radius * radius


# --------------------------------------------------------------------------- #
#  Разбор кадра
# --------------------------------------------------------------------------- #
@dataclass
class SpineColumn:
    """Результат разбора кадра позвоночника (координаты — в пикселях кадра)."""
    img: np.ndarray                 # uint8-кадр, кость светлая
    mask: np.ndarray                # маска кости
    left: np.ndarray                # x левого края колонны по строкам (NaN — нет)
    right: np.ndarray               # x правого края колонны
    center: np.ndarray              # взвешенный по яркости центр колонны
    valid: np.ndarray               # строки, где колонна похожа на позвоночную
    center_x: float                 # медиана center по valid
    half_width: float               # половина ширины колонны, px
    flags: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.img.shape


def spine_column(img: np.ndarray) -> SpineColumn:
    """Кадр -> колонна позвоночника: края и центр тел по каждой строке.

    Центр строки берётся не по всему кадру, а внутри КОСТНОГО ОТРЕЗКА,
    содержащего коридор позвоночника. Разница принципиальная: взвешенный
    центр по окну фиксированной ширины утягивают рёбра, гребни подвздошных
    костей и яркие мягкие ткани, и трек уходит от позвоночника (на выборке
    это давало AUC оси 0.57 вместо 0.83).

    Строки, где отрезок заметно шире или уже типичного (таз, крестец, разрыв
    маски), помечаются недостоверными и в оценку оси не идут.
    """
    a = to_uint8(img)
    h, w = a.shape
    m = bone_mask(a)

    # коридор: столбец с наибольшей костной массой
    corridor = int(np.argmax(moving_average(m.sum(axis=0).astype(float), 15))) if m.any() else w // 2

    reach = mm2px(27.0)          # насколько далеко от коридора ещё «тот самый» отрезок
    af = a.astype(float)
    left = np.full(h, np.nan)
    right = np.full(h, np.nan)
    center = np.full(h, np.nan)
    min_run = mm2px(6.0)
    for y in range(h):
        runs = [r for r in bone_runs(m[y]) if r[0] - reach <= corridor <= r[1] + reach]
        runs = [r for r in runs if r[1] - r[0] >= min_run]
        if not runs:
            continue
        r = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - corridor))
        left[y], right[y] = r[0], r[1] - 1
        seg = af[y, r[0]:r[1]]
        # квадрат превышения над самым тёмным пикселем отрезка: тела позвонков
        # плотнее отростков, и центр садится на тело, а не на середину силуэта
        wgt = np.clip(seg - seg.min(), 0.0, None) ** 2
        s = wgt.sum()
        center[y] = r[0] + float((wgt * np.arange(len(seg))).sum() / s) if s > 0 else (r[0] + r[1]) / 2

    width = right - left
    med = float(np.nanmedian(width)) if np.isfinite(width).any() else float(mm2px(45.0))
    y = np.arange(h, dtype=float)
    valid = (np.isfinite(center) & (width < med * 1.45) & (width > med * 0.55)
             & (y >= h * BAND[0]) & (y <= h * BAND[1]))

    flags: list[str] = []
    if valid.sum() < 20:
        flags.append('column:short')
    cx = float(np.nanmedian(center[valid])) if valid.any() else float(corridor)
    return SpineColumn(img=a, mask=m, left=left, right=right, center=center,
                       valid=valid, center_x=cx, half_width=med / 2.0, flags=flags)


# --------------------------------------------------------------------------- #
#  Устойчивая прямая
# --------------------------------------------------------------------------- #
def theil_sen(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Наклон и сдвиг прямой x = a*y + b по медиане попарных наклонов.

    Медиана вместо наименьших квадратов: одна строка, где маска слилась с
    ребром, смещает МНК на градус, а медиану — нет.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    if n < 3:
        return 0.0, float(x.mean()) if n else 0.0
    i, j = np.triu_indices(n, 1)
    dy = y[j] - y[i]
    ok = dy != 0
    if not ok.any():
        return 0.0, float(np.median(x))
    a = float(np.median((x[j] - x[i])[ok] / dy[ok]))
    return a, float(np.median(x - a * y))


def white_tophat(a: np.ndarray, radius_px: int) -> np.ndarray:
    """a - морфологическое раскрытие: отклик тонких ЯРКИХ структур.

    Структурный элемент шире косточки белья (2-4 px) и уже тел позвонков,
    поэтому металл остаётся, а кость и мягкие ткани уходят в фон. Это
    единственный способ поймать металл на этих кадрах: экспорт уже насыщен
    (max = 255 у всех 99 кадров), и абсолютная яркость не различает металл и
    кортикальную кость.
    """
    return a.astype(float) - ndi.grey_opening(a.astype(float), footprint=disk(radius_px))
