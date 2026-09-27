# -*- coding: utf-8 -*-
"""Геометрические примитивы: масштаб, маска кости, профили, контур.

Всё, что здесь есть, общее для всех вариантов алгоритма. Ничего не обучается,
случайности нет: одинаковый вход -> побитово одинаковый выход.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------- #
#  Масштаб и пороги ТЗ
# --------------------------------------------------------------------------- #
# GE Lunar Prodigy Advance, экспорт кадра бедра: 280 px = 170 мм по тегу
# ExposedArea на кадрах, где он согласован с размером кадра (160/263, 143/235,
# 159/261, 177/291 ...: разброс 0.605-0.609). Сам тег ненадёжен (в половине
# исследований одно значение на все кадры, бывает [0,0]) -> константа.
MM_PER_PX = 0.607

TOP_MM = 30.0       # «3 см сверху от области интереса»
BOTTOM_MM = 30.0    # «3 см снизу»
LAT_MM = 20.0       # «2 см от края ... в зависимости от бедра»

# Анатомический прибор: расстояние верхушка большого вертела -> нижняя граница
# малого вертела. 50 мм — по четырём кадрам, размеченным вручную по анатомии
# (35, 41, 45, 60 мм), и по анатомической норме 4-6 см. До ответа организаторов
# (27.09.2026) это был низ ROI; теперь низ — седалищная кость (D_TI_MM ниже), а
# уровень малого вертела остался только у вариантов B1–B4 для `custom`.
D_TB_MM = 50.0

# Низ ROI по рисунку 6 ТЗ — нижняя точка седалищной кости (ответ организаторов,
# 27.09.2026). Расстояние «верхушка большого вертела -> нижняя точка седалищной
# кости» по ручной разметке 149 кадров: медиана 69 мм, 1–99 перцентиль 52–83 мм.
# Диапазон — защита от маски, ушедшей по мягким тканям; медиана — запас, когда
# седалищную кость найти не удалось.
D_TI_MM = 69.0
D_TI_RANGE_MM = (40.0, 90.0)
# Край кончика седалищной кости бледный: маска обрывается выше, чем кончик
# виден глазом (в среднем на 2.4 мм). Край ищется по уровню 35 % между тоном
# кости и фоном — на ручной разметке это даёт медиану ошибки 0.9 мм.
ISCHIUM_EDGE_FRAC = 0.35

EIGHT = np.ones((3, 3), bool)


def mm2px(mm: float) -> int:
    return int(round(mm / MM_PER_PX))


def px2mm(px: float) -> float:
    return float(px) * MM_PER_PX


# --------------------------------------------------------------------------- #
#  Маска кости
# --------------------------------------------------------------------------- #
def otsu_threshold(values: np.ndarray) -> int:
    """Порог Оцу по 256-битной гистограмме. Детерминирован."""
    h, _ = np.histogram(values, bins=256, range=(0, 256))
    p = h.astype(np.float64) / max(h.sum(), 1)
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m = np.cumsum(p * np.arange(256))
    mt = m[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        var = (mt * w0 - m) ** 2 / (w0 * w1)
    var[~np.isfinite(var)] = -1.0
    return int(np.argmax(var))


def multi_otsu3(values: np.ndarray) -> tuple[int, int]:
    """Трёхклассовый Оцу: два порога (t1, t2), максимизирующие межклассовую дисперсию.

    Нужен именно он, а не обычный Оцу: на bone-map кость сама двугорбая
    (трабекулярная 60-120, кортикал 120-250), и двухклассовый Оцу режет
    внутри кости (порог ~110), выбрасывая трабекулярную часть вертела.
    Нижний порог t1 ложится в провал «гало/мягкие ткани ↔ кость»:
    44-76 на bone-map, 62-78 на полутоновых кадрах выборки.
    """
    h = np.bincount(np.asarray(values, dtype=np.int64).ravel(), minlength=256)[:256].astype(np.float64)
    p = h / max(h.sum(), 1.0)
    i = np.arange(256, dtype=np.float64)
    P = np.concatenate(([0.0], np.cumsum(p)))          # P[k] = сумма p[0:k]
    S = np.concatenate(([0.0], np.cumsum(p * i)))
    mt = S[-1]
    best, bt = -1.0, (1, 2)
    t2 = np.arange(2, 256)
    for t1 in range(1, 255):
        w0, s0 = P[t1], S[t1]
        if w0 <= 0:
            continue
        sel = t2 > t1
        tt = t2[sel]
        w1, s1 = P[tt] - w0, S[tt] - s0
        w2, s2 = 1.0 - P[tt], mt - S[tt]
        ok = (w1 > 0) & (w2 > 0)
        if not ok.any():
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            var = (w0 * (s0 / w0 - mt) ** 2 + w1 * (s1 / w1 - mt) ** 2 + w2 * (s2 / w2 - mt) ** 2)
        var[~ok] = -1.0
        j = int(np.argmax(var))
        if var[j] > best:
            best, bt = float(var[j]), (t1, int(tt[j]))
    return bt


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return np.round((img - lo) / (hi - lo) * 255).astype(np.uint8)


def bone_threshold(img: np.ndarray) -> int:
    """Нижний порог трёхклассового Оцу по ненулевым пикселям."""
    img = to_uint8(img)
    v = img[img > 0]
    return multi_otsu3(v)[0] if v.size else 255


def bone_mask(img: np.ndarray, min_component_px: int = 300, weak_frac: float = 0.55) -> np.ndarray:
    """Кость с гистерезисным порогом: сильный порог задаёт зёрна, слабый —
    границу. Opening 3x3, заливка дырок, компоненты >= 300 px.

    Сильный порог — нижний порог трёхклассового Оцу (см. multi_otsu3), слабый —
    weak_frac от него. Одним порогом обойтись нельзя: он подобран по всему
    кадру, а тон кости внутри бедра падает до 0.5-0.7 от него (разрежённая
    трабекулярная кость вертельной области, порозная кость). На таких кадрах
    одиночный порог терял вертельную массу целиком — вместе с верхушкой
    большого вертела, от которой отсчитывается верх ROI.

    Слабый порог не «раздувает» наружную границу: снаружи яркость падает от
    порога до нуля за 3-5 px, поэтому граница сдвигается на 1-2 px (0.6-1.2 мм),
    тогда как внутри кости возвращаются целые области.
    """
    img = to_uint8(img)
    v = img[img > 0]
    if v.size == 0:
        return np.zeros(img.shape, bool)
    strong = multi_otsu3(v)[0]
    weak = max(int(round(strong * weak_frac)), 5)
    seeds = ndi.binary_opening(img > strong, structure=EIGHT, iterations=1)
    m = ndi.binary_opening(img > weak, structure=EIGHT, iterations=1)
    lbl, n = ndi.label(m, structure=EIGHT)
    if n:
        m = np.isin(lbl, np.unique(lbl[seeds & (lbl > 0)]))     # только с зерном
    # дырки внутри кости (тёмная трабекулярная зона вертела) — не разрывы:
    # без заливки правило «первый разрыв = медиальный край» упрётся в дырку
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m, structure=EIGHT)
    if n == 0:
        return m
    sizes = ndi.sum(m, lbl, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= min_component_px) + 1
    return np.isin(lbl, keep)


def unscanned_region(img: np.ndarray, min_px: int = 200) -> np.ndarray:
    """Незасканированные вырезы, примыкающие к нижним углам кадра.

    В экспорте Lunar нижний медиальный угол часто «срезан» прямоугольником
    точных нулей (поле сканирования короче кадра). Строку, задетую таким
    вырезом, нельзя использовать для измерений по тону: сегмент кости в ней
    обрезан не анатомией, а границей поля.

    Берётся компонента точных нулей, примыкающая к нижнему углу кадра. Обычно
    это весь фон, но на кость она выходит только там, где тон обрывается в
    ноль скачком, то есть на границе поля: у настоящего края кости яркость
    спадает до нуля за 3-5 px, и эти промежуточные значения фоном не являются.
    """
    img = to_uint8(img)
    z = img == 0
    lbl, n = ndi.label(z, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    h, w = img.shape
    out = np.zeros(img.shape, bool)
    if n == 0:
        return out
    for corner in ((h - 1, 0), (h - 1, w - 1)):
        k = int(lbl[corner])
        if k and int((lbl == k).sum()) >= min_px:
            out |= lbl == k
    return out


def component_containing(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """8-связная компонента маски, содержащая точку (y, x)."""
    lbl, _ = ndi.label(mask, structure=EIGHT)
    k = lbl[point[0], point[1]]
    return lbl == k if k else np.zeros_like(mask)


# --------------------------------------------------------------------------- #
#  Нормализация стороны: латераль всегда СЛЕВА (x = 0)
# --------------------------------------------------------------------------- #
def to_lateral_left(arr: np.ndarray, side: str) -> np.ndarray:
    """Правое бедро (rh) в кадре лежит слева -> как есть; левое (lh) зеркалим."""
    if side not in ('lh', 'rh'):
        raise ValueError(f"side должен быть 'lh' или 'rh', получено {side!r}")
    return arr[:, ::-1] if side == 'lh' else arr


def x_to_original(x: int, width: int, side: str) -> int:
    return width - 1 - x if side == 'lh' else x


# --------------------------------------------------------------------------- #
#  Профили (в нормализованной ориентации)
# --------------------------------------------------------------------------- #
def lateral_profile(mask: np.ndarray) -> np.ndarray:
    """x_lat(y): первый пиксель кости в строке; W, если кости в строке нет."""
    h, w = mask.shape
    xl = np.full(h, w, dtype=int)
    has = mask.any(axis=1)
    xl[has] = mask[has].argmax(axis=1)
    return xl


def medial_profile(mask: np.ndarray, xl: np.ndarray) -> np.ndarray:
    """x_med(y): первый НЕ-костный пиксель медиальнее x_lat(y) (первый разрыв маски).

    Так седалищная кость, лежащая медиальнее через мягкотканный промежуток,
    в ширину бедра не попадает. Если разрыва нет до края кадра -> W.
    """
    h, w = mask.shape
    xm = np.full(h, w, dtype=int)
    for y in range(h):
        if xl[y] >= w:
            continue
        gap = np.flatnonzero(~mask[y, xl[y]:])
        xm[y] = xl[y] + gap[0] if gap.size else w
    return xm


def width_profile(xl: np.ndarray, xm: np.ndarray, width: int) -> np.ndarray:
    return np.where(xl < width, xm - xl, 0)


def top_profile(mask: np.ndarray) -> np.ndarray:
    """y_top(x): первый пиксель кости в столбце; H, если кости в столбце нет."""
    h, w = mask.shape
    yt = np.full(w, h, dtype=int)
    has = mask.any(axis=0)
    yt[has] = mask[:, has].argmax(axis=0)
    return yt


def bone_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Непрерывные отрезки True в строке: [(start, stop_exclusive), ...]."""
    if not row.any():
        return []
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


# --------------------------------------------------------------------------- #
#  Трассировка внешнего контура (Moore neighbour tracing, 8-связность)
# --------------------------------------------------------------------------- #
# По часовой стрелке, начиная с «запада»: W, NW, N, NE, E, SE, S, SW
_DIRS = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}


def trace_contour(mask: np.ndarray, start: tuple[int, int], max_len: int | None = None) -> np.ndarray:
    """Замкнутый внешний контур компоненты, содержащей граничный пиксель start.

    Возвращает массив (N, 2) координат (y, x) по часовой стрелке, начиная со
    start. start обязан быть граничным пикселем (хотя бы один фоновый сосед).
    Остановка по критерию Джейкоба: вернулись в start и следующий шаг совпадает
    со вторым пикселем контура.
    """
    pad = np.pad(mask, 1)
    y0, x0 = start[0] + 1, start[1] + 1
    if not pad[y0, x0]:
        raise ValueError('start не принадлежит маске')
    back = None
    for k, (dy, dx) in enumerate(_DIRS):
        if not pad[y0 + dy, x0 + dx]:
            back = k
            break
    if back is None:
        raise ValueError('start не граничный пиксель')
    if max_len is None:
        max_len = 16 * (mask.shape[0] + mask.shape[1]) + 16

    def next_pixel(cy, cx, b):
        for i in range(1, 9):
            k = (b + i) % 8
            ny, nx = cy + _DIRS[k][0], cx + _DIRS[k][1]
            if pad[ny, nx]:
                py, px = cy + _DIRS[(k - 1) % 8][0], cx + _DIRS[(k - 1) % 8][1]
                return (ny, nx), _DIR_INDEX[(py - ny, px - nx)]
        return None, None

    contour = [(y0, x0)]
    cy, cx, b = y0, x0, back
    while len(contour) < max_len:
        nxt, nb = next_pixel(cy, cx, b)
        if nxt is None:
            break
        if (cy, cx) == (y0, x0) and len(contour) > 1 and nxt == contour[1]:
            break
        contour.append(nxt)
        cy, cx = nxt
        b = nb
    return np.array(contour, dtype=int) - 1


def moving_average(v: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win) | 1)
    v = np.asarray(v, dtype=float)
    if win == 1 or v.size < win:
        return v
    k = np.ones(win) / win
    padded = np.concatenate((np.full(win // 2, v[0]), v, np.full(win // 2, v[-1])))
    return np.convolve(padded, k, mode='valid')


# --------------------------------------------------------------------------- #
#  Две ноги в одном кадре (DualFemur-экспорт): масштаб неизвестен
# --------------------------------------------------------------------------- #
def detect_dual_femur(mask: np.ndarray, bottom_frac: float = 0.15,
                      min_run_mm: float = 12.0, min_gap_mm: float = 50.0) -> bool:
    """В нижних 15 % строк стабильно два широких костных отрезка далеко друг от друга."""
    h = mask.shape[0]
    y0 = int(h * (1.0 - bottom_frac))
    min_run, min_gap = mm2px(min_run_mm), mm2px(min_gap_mm)
    hits = total = 0
    for y in range(y0, h):
        runs = [r for r in bone_runs(mask[y]) if r[1] - r[0] >= min_run]
        total += 1
        if len(runs) >= 2 and (runs[-1][0] - runs[0][1]) >= min_gap:
            hits += 1
    return total > 0 and hits / total >= 0.7
