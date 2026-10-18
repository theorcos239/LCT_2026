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

from .points import SPINE_CORE, SPINE_EDGES

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


def _candidate_peaks(h: np.ndarray, top_k: int = 6) -> list[tuple[np.ndarray, float]]:
    """Сильные локальные пики; относительный пол нужен для слабых TTA-карт."""
    floor = min(0.02, max(0.001, float(h.max()) * 0.1))
    points, values = _local_maxima(h, floor, radius=4, top_k=30)
    result = []
    for point, value in zip(points, values):
        if all(np.linalg.norm(point - other) > 8 for other, _ in result):
            result.append((point, float(value)))
            if len(result) == top_k:
                break
    return result


def _disc_evidence(chain: np.ndarray, peaks: list[list[tuple[np.ndarray, float]]]) -> float:
    """Поддержка пяти дисков в полосах между центрами цепочки."""
    bands = ((0, -1., 0.), (0, 0., 1.), (1, 0., 1.),
             (2, 0., 1.), (3, 0., 1.))
    step = (chain[-1] - chain[0]) / 3
    length = float(np.linalg.norm(step))
    if length < 1:
        return -100.0
    score = 0.0
    for (k, low, high), choices in zip(bands, peaks):
        best = 1e-4
        for point, value in choices:
            d = point - chain[k]
            along = float(d @ step) / length**2
            across = abs(float(d[0] * step[1] - d[1] * step[0])) / length
            if low <= along <= high and across <= length:
                best = max(best, value)
        score += np.log(best)
    return float(score)


def chain_hypotheses(prob: np.ndarray, names: list[str], top_k: int = 5,
                     spacing_weight: float = 10.0, disc_weight: float = 0.5
                     ) -> list[tuple[float, np.ndarray]]:
    """Лучшие цепочки L1–L4: 21 отклик в каждом кандидате + геометрия + диски.

    Баллы относительны только внутри одного снимка. Варианты рассматриваются
    при полной видимости L1–L4; проверка видимости остаётся у вызывающего кода.
    """
    index = {name: i for i, name in enumerate(names)}
    discs = ("disc_th12_l1", "disc_l1_l2", "disc_l2_l3",
             "disc_l3_l4", "disc_l4_l5")
    # У набора core тоже 21 выход, но восемь последних относятся к бедру.
    # Нормировать по ним отклики позвонков нельзя.
    if (prob.ndim != 3 or prob.shape[0] < 21 or
            names[:21] != SPINE_CORE + SPINE_EDGES or
            not all(n in index for n in (*CENTERS, *discs))):
        return []
    centers = [index[n] for n in CENTERS]
    hits = sorted((hit for c in centers for hit in _candidate_peaks(prob[c])),
                  key=lambda hit: hit[1], reverse=True)
    pool = []
    for point, _ in hits:
        if all(np.linalg.norm(point - other) > 8 for other in pool):
            pool.append(point)
    pool = np.array(sorted(pool, key=lambda p: p[1])).reshape(-1, 2)
    if len(pool) < 4:
        return []

    # Каждый кандидат имеет 21 отклик; нормировка эквивалентна softmax(log H).
    support = np.empty((len(pool), 21), dtype=float)
    height, width = prob.shape[1:]
    for i, (x, y) in enumerate(pool.astype(int)):
        support[i] = prob[:21, max(0, y - 2):min(height, y + 3),
                          max(0, x - 2):min(width, x + 3)].max(axis=(1, 2))
    support = np.maximum(support, 1e-4)
    raw = np.log(support[:, centers])
    normalized = raw - np.log(support.sum(axis=1, keepdims=True))

    proposals = []
    for combo in combinations(range(len(pool)), 4):
        chain = pool[list(combo)]
        steps = np.diff(chain, axis=0)
        mean = steps.mean(axis=0)
        length = float(np.linalg.norm(mean))
        if not (25 <= length <= 110 and mean[1] > 0):
            continue
        uneven = float(np.square(steps - mean).sum() / length**2)
        raw_score = float(sum(raw[i, k] for k, i in enumerate(combo))
                          - spacing_weight * uneven)
        normalized_score = float(sum(normalized[i, k] for k, i in enumerate(combo))
                                 - spacing_weight * uneven)
        proposals.append((combo, raw_score, normalized_score))
    if not proposals:
        return []

    # Диски оцениваем на объединении 100 лучших сырых и нормированных цепочек.
    shortlist = set()
    for field in (1, 2):
        shortlist.update(row[0] for row in sorted(proposals,
                                                  key=lambda row: row[field],
                                                  reverse=True)[:100])
    disc_peaks = [_candidate_peaks(prob[index[name]]) for name in discs]
    scored = [(row[2] + disc_weight * _disc_evidence(pool[list(row[0])], disc_peaks),
               pool[list(row[0])]) for row in proposals if row[0] in shortlist]
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored[:top_k]


def _align_dependent(prob: np.ndarray, names: list[str], out: np.ndarray,
                     chain: np.ndarray, active: np.ndarray, window: int,
                     min_value: float) -> np.ndarray:
    """Вернуть диски и края в полосы выбранной цепочки."""
    step = (chain[-1] - chain[0]) / 3
    s2 = float(step @ step)
    if s2 < 1:
        return out
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

    return _align_dependent(prob, names, out, chain, active, window, min_value)


def chain_decode_hypotheses(prob: np.ndarray, names: list[str], coords: np.ndarray,
                            active: np.ndarray, window: int = 11,
                            average: bool = False, temperature: float = 1.0
                            ) -> tuple[np.ndarray, dict]:
    """Согласовать L1–L4 по нескольким гипотезам; вернуть относительный вес лидера.

    При частично видимой цепочке используется прежний декодер, чтобы не
    выдумывать отсутствующие в обрезанном кадре позвонки. Усреднение является
    исследовательской опцией: оно может поставить точку между двумя уровнями.
    """
    index = {name: i for i, name in enumerate(names)}
    if not all(name in index for name in CENTERS):
        return coords, {"used": False, "reason": "missing_channels"}
    center_cols = [index[name] for name in CENTERS]
    if not np.asarray(active)[center_cols].all():
        return chain_decode(prob, names, coords, active, window), {
            "used": False, "reason": "partial_visibility"}
    ranked = chain_hypotheses(prob, names)
    if not ranked:
        return chain_decode(prob, names, coords, active, window), {
            "used": False, "reason": "no_chain"}

    scores = np.array([score for score, _ in ranked])
    temperature = max(float(temperature), 1e-6)
    weights = np.exp((scores - scores.max()) / temperature)
    weights /= weights.sum()
    if average:
        centers = sum(w * chain for w, (_, chain) in zip(weights, ranked))
    else:
        centers = np.array([_soft_argmax(prob[c], *point, window)
                            for c, point in zip(center_cols, ranked[0][1])])
    out = coords.copy()
    for c, p in zip(center_cols, centers):
        out[c] = p
    out = _align_dependent(prob, names, out, centers, active, window,
                           min_value=min(.02, max(.001, float(prob[center_cols].max()) * .1)))
    return out, {
        "used": True,
        "n_hypotheses": len(ranked),
        "top1_relative_weight": float(weights[0]),
        "score_gap_top2": float(scores[0] - scores[1]) if len(scores) > 1 else None,
        "averaged": bool(average),
    }
