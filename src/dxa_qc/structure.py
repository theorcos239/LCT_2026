"""Согласованная расшифровка позвоночника: точки не прыгают на соседний позвонок.

Каждый канал расшифровывается своим максимумом независимо, а тела L1–L4 на
снимке похожи друг на друга. На OOF в 28 кадрах позвоночника из 91 хотя бы
один центр стоял на соседнем позвонке, и почти всегда это был частичный сдвиг:
L1 на Th12 при верных L2–L4 (шаг цепочки 2-1-1 позвонка) или L4 на L5. Цепочка
с таким шагом анатомически невозможна, и это проверяемо без разметки.

Расшифровка по структуре:

1. Кандидаты — локальные максимумы четырёх каналов центров (не только
   главные): тело каждого позвонка светится хотя бы в одном из них.
2. Центрам L1–L4 (тем, что модель считает видимыми) назначаются кандидаты
   сверху вниз так, чтобы максимизировать сумму log-вероятностей каналов
   минус штраф за неровный шаг: векторы между соседними центрами должны быть
   почти одинаковы (позвонки одного размера, ось почти прямая).
3. Зависимые точки — края тел, диски, Th12 — должны лежать в полосе своего
   уровня вдоль оси цепочки. Если максимум канала вне полосы, канал
   расшифровывается заново только внутри неё.

Уверенность (а значит, и видимость) не меняется: она отвечает на вопрос «есть
ли структура в кадре», а структура меняет только её место.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.ndimage import maximum_filter

CENTERS = [f"l{i}_center" for i in range(1, 5)]

# Полосы вдоль оси цепочки в долях шага, от центра L{k} (k — индекс 0..3).
# Th12: центр её тела на шаг выше L1, высота тела ~0.8 шага.
def _bands(names: list[str]) -> dict[int, tuple[int, float, float]]:
    """channel -> (опорный центр 0..3, t_min, t_max)."""
    out = {}
    idx = {n: i for i, n in enumerate(names)}
    for k in range(4):
        for side in ("right", "left"):
            n = f"l{k + 1}_edge_{side}"
            if n in idx:
                out[idx[n]] = (k, -0.5, 0.5)
    discs = {"disc_th12_l1": (0, -1.0, 0.0), "disc_l1_l2": (0, 0.0, 1.0),
             "disc_l2_l3": (1, 0.0, 1.0), "disc_l3_l4": (2, 0.0, 1.0),
             "disc_l4_l5": (3, 0.0, 1.0),
             "th12_top": (0, -2.0, -0.9), "th12_bottom": (0, -1.1, -0.2)}
    for n, band in discs.items():
        if n in idx:
            out[idx[n]] = band
    return out


def _local_maxima(h: np.ndarray, min_value: float, radius: int, top_k: int):
    mx = maximum_filter(h, size=2 * radius + 1, mode="constant")
    ys, xs = np.nonzero((h == mx) & (h >= min_value))
    order = np.argsort(-h[ys, xs])[:top_k]
    return np.stack([xs[order], ys[order]], 1).astype(float), h[ys[order], xs[order]]


def _near_max(h: np.ndarray, x: float, y: float, r: int = 4) -> tuple[float, float, float]:
    """Максимум канала в окрестности точки: (x, y, значение)."""
    hh, ww = h.shape
    x0, x1 = max(0, int(round(x)) - r), min(ww, int(round(x)) + r + 1)
    y0, y1 = max(0, int(round(y)) - r), min(hh, int(round(y)) + r + 1)
    patch = h[y0:y1, x0:x1]
    if patch.size == 0:
        return x, y, 0.0
    py, px = np.unravel_index(patch.argmax(), patch.shape)
    return float(x0 + px), float(y0 + py), float(patch.max())


def _soft_argmax(h: np.ndarray, x: float, y: float, window: int) -> np.ndarray:
    hh, ww = h.shape
    r = window // 2
    xi, yi = int(round(x)), int(round(y))
    ys = np.clip(np.arange(yi - r, yi + r + 1), 0, hh - 1)
    xs = np.clip(np.arange(xi - r, xi + r + 1), 0, ww - 1)
    p = np.clip(h[np.ix_(ys, xs)], 0, None)
    s = p.sum()
    if s <= 1e-6:
        return np.array([x, y])
    return np.array([(p.sum(0) * xs).sum() / s, (p.sum(1) * ys).sum() / s])


def chain_decode(prob: np.ndarray, names: list[str], coords: np.ndarray, active: np.ndarray,
                 window: int = 11, min_value: float = 0.02, top_k: int = 6,
                 merge_px: float = 12.0, spacing_weight: float = 10.0,
                 min_step_px: float = 25.0, max_step_px: float = 110.0) -> np.ndarray:
    """prob: (C, H, W) вероятности (заполнение уже обнулено); coords: (C, 2)
    независимая расшифровка; active: (C,) каналы, которые модель считает видимыми.

    Возвращает исправленные координаты (C, 2). Если видимых центров меньше двух,
    опереться не на что — координаты возвращаются как есть.
    """
    idx = {n: i for i, n in enumerate(names)}
    if not all(n in idx for n in CENTERS):
        return coords
    cen = [idx[n] for n in CENTERS]
    levels = [k for k in range(4) if active[cen[k]]]
    if len(levels) < 2:
        return coords
    out = coords.copy()

    # 1. кандидаты — пики всех каналов центров, близкие склеены
    pts, vals = [], []
    for c in cen:
        p, v = _local_maxima(prob[c], min_value, radius=4, top_k=top_k)
        pts += list(p)
        vals += list(v)
    order = np.argsort(-np.array(vals))
    cand: list[np.ndarray] = []
    for i in order:
        if all(np.linalg.norm(pts[i] - q) > merge_px for q in cand):
            cand.append(pts[i])
    cand = sorted(cand, key=lambda q: q[1])               # сверху вниз
    if len(cand) < len(levels):
        return coords

    # 2. назначение: log-вероятность каналов минус неровность шага
    score_of = np.array([[np.log(max(_near_max(prob[cen[k]], *q)[2], 1e-4)) for q in cand]
                         for k in range(4)])
    best, best_s = None, -np.inf
    for combo in combinations(range(len(cand)), len(levels)):
        p = np.array([cand[i] for i in combo])
        # шаг на один позвонок: центры видимых уровней могут идти через пропуск
        steps = np.diff(p, axis=0) / np.diff(levels)[:, None]
        mean = steps.mean(0)
        norm = np.linalg.norm(mean)
        if not (min_step_px <= norm <= max_step_px) or mean[1] <= 0:
            continue
        pen = (np.linalg.norm(steps - mean, axis=1) ** 2).sum() / norm ** 2
        s = sum(score_of[k, i] for k, i in zip(levels, combo)) - spacing_weight * pen
        if s > best_s:
            best, best_s = combo, s
    if best is None:
        return coords
    chain = np.full((4, 2), np.nan)
    for k, i in zip(levels, best):
        x, y, _ = _near_max(prob[cen[k]], *cand[i])
        chain[k] = _soft_argmax(prob[cen[k]], x, y, window)
        out[cen[k]] = chain[k]

    # недостающие уровни — экстраполяция по среднему шагу (только как опора полос)
    known = np.array(levels)
    step = (chain[known[-1]] - chain[known[0]]) / (known[-1] - known[0])
    for k in range(4):
        if np.isnan(chain[k, 0]):
            chain[k] = chain[known[0]] + (k - known[0]) * step

    # 3. зависимые точки — в полосе своего уровня
    s2 = float(step @ step)
    hh, ww = prob.shape[1:]
    yy, xx = np.mgrid[0:hh, 0:ww]
    for c, (k, t0, t1) in _bands(names).items():
        if not active[c]:
            continue
        t = float((out[c] - chain[k]) @ step) / s2
        if t0 <= t <= t1:
            continue
        tmap = ((xx - chain[k][0]) * step[0] + (yy - chain[k][1]) * step[1]) / s2
        h = np.where((tmap >= t0) & (tmap <= t1), prob[c], 0.0)
        if h.max() < min_value:
            continue
        y, x = np.unravel_index(h.argmax(), h.shape)
        out[c] = _soft_argmax(h, float(x), float(y), window)
    return out
