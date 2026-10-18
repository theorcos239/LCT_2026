"""Проверка нескольких вариантов разметки на СОХРАНЁННЫХ OOF heatmap.

Ожидает NPZ с heatmaps (N, 29, H, W), index (N,) и image_hw (N, 2).
Каждая карта должна быть получена своим fold-чекпоинтом на отложенном кадре
тем же TTA, что Predictor. index — индекс кадра в data/keypoints.npz и
runs/keypoints/oof.npz. Сравнение идёт только по снимкам позвоночника.

Пример: PYTHONPATH=src python tools/evaluate_trajectory_hypotheses.py \\
  --heatmaps Processing/oof_heatmaps.npz --oof runs/keypoints/oof.npz

Скрипт НЕ принимает координаты вместо карт: вторые пики по ним не восстановить.
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from dxa_qc.metrics import axis_angle
from dxa_qc.structure import _bands


CENTERS = [f"l{i}_center" for i in range(1, 5)]


def peaks(h: np.ndarray, top_k: int = 6, floor: float = .02) -> list[tuple[float, float]]:
    """Локальные максимумы одной карты; плато оставляет одну точку."""
    maxima = maximum_filter(h, size=9, mode="constant")
    ys, xs = np.nonzero((h == maxima) & (h >= floor))
    out = []
    for i in np.argsort(-h[ys, xs]):
        p = (float(xs[i]), float(ys[i]))
        if all(np.linalg.norm(np.subtract(p, q)) > 8 for q in out):
            out.append(p)
            if len(out) == top_k:
                break
    return out


def candidates(h: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Все пики 21 карты; каждому кандидату соответствует 21 отклик."""
    pool: list[tuple[float, float]] = []
    center_ids = set()
    for c, name in enumerate(names[:21]):
        for p in peaks(h[c]):
            near = next((i for i, q in enumerate(pool)
                         if np.linalg.norm(np.subtract(p, q)) <= 8), None)
            if near is None:
                near = len(pool)
                pool.append(p)
            if name in CENTERS:
                center_ids.add(near)
    if not pool:
        return np.empty((0, 2)), np.empty((21, 0)), np.empty(0, dtype=int)
    pos = np.array(pool)
    # Считываем ВСЕ 21 карты в КАЖДОМ кандидате, независимо от его источника.
    support = h[:21, pos[:, 1].astype(int), pos[:, 0].astype(int)]
    centers = np.array(sorted(center_ids, key=lambda i: pos[i, 1]), dtype=int)
    return pos, support, centers


def center_proposals(pos: np.ndarray, support: np.ndarray, center_ids: np.ndarray,
                     names: list[str], limit: int = 100) -> list[tuple[float, tuple[int, ...]]]:
    """Сначала дешёвый перебор центров; затем дорогая оценка всех 21 карт."""
    if len(center_ids) < 4:
        return []
    idx = {n: i for i, n in enumerate(names)}
    rows = [idx[n] for n in CENTERS]
    proposals = []
    for combo in combinations(center_ids, 4):
        chain = pos[list(combo)]
        steps = np.diff(chain, axis=0)
        mean = steps.mean(0)
        length = np.linalg.norm(mean)
        if not (25 <= length <= 110 and mean[1] > 0):
            continue
        uneven = (np.linalg.norm(steps - mean, axis=1) ** 2).sum() / length ** 2
        value = sum(np.log(max(float(support[c, i]), 1e-4))
                    for c, i in zip(rows, combo)) - 10.0 * uneven
        proposals.append((float(value), combo))
    proposals.sort(key=lambda p: p[0], reverse=True)
    return proposals[:limit]


def full_score(center_score: float, combo: tuple[int, ...], pos: np.ndarray,
               support: np.ndarray, names: list[str], active: np.ndarray) -> float:
    """Оценка остальных видимых точек с уникальным кандидатом или пропуском."""
    chain = pos[list(combo)]
    step = (chain[-1] - chain[0]) / 3.0
    s2 = float(step @ step)
    if s2 < 1:
        return -np.inf
    bands = _bands(names)
    rows = [c for c in range(21) if names[c] not in CENTERS and active[c]]
    if not rows:
        return center_score
    available = [i for i in range(len(pos)) if i not in combo]
    # Отдельный dummy для каждой строки разрешает «не нашлась точка».
    scores = np.full((len(rows), len(available) + len(rows)), np.log(1e-4))
    for r, c in enumerate(rows):
        for col, i in enumerate(available):
            if c in bands:
                k, lo, hi = bands[c]
                t = float((pos[i] - chain[k]) @ step / s2)
                if not (lo <= t <= hi):
                    scores[r, col] = -1e6
                    continue
            scores[r, col] = np.log(max(float(support[c, i]), 1e-4))
    rr, cc = linear_sum_assignment(-scores)
    return float(center_score + scores[rr, cc].sum())


def classify(chain: np.ndarray, truth: np.ndarray) -> tuple[bool, float]:
    """Тот же смысл ошибки уровня, что в evaluate.py: Th12/L5 экстраполированы."""
    reference = np.vstack([truth[0] - (truth[1] - truth[0]), truth,
                           truth[3] + (truth[3] - truth[2])])
    nearest = np.linalg.norm(chain[:, None] - reference[None], axis=2).argmin(axis=1)
    return bool(np.array_equal(nearest, np.arange(1, 5))), abs(axis_angle(chain) - axis_angle(truth))


def evaluate(heatmaps: Path, oof_path: Path, out: Path, top_k: int = 5) -> list[dict]:
    with np.load(heatmaps) as z:
        needed = {"heatmaps", "index", "image_hw"}
        if not needed.issubset(z.files):
            raise ValueError(f"{heatmaps}: нужны массивы {sorted(needed)}")
        maps, indexes, hw = z["heatmaps"], z["index"], z["image_hw"]
        map_names = list(z["names"]) if "names" in z.files else None
    with np.load(oof_path) as z:
        names = list(z["names"])
        order = {int(j): i for i, j in enumerate(z["index"])}
        regions = z["regions"]
        true, visible, old_pred = z["true"], z["visible"], z["pred"]
        thresholds = z["thresholds"]
    if len(names) != 29 or names[:21].count("l1_center") != 1:
        raise ValueError("Ожидались 29 карт текущей модели, первые 21 — позвоночник")
    if map_names is not None and map_names != names:
        raise ValueError("Порядок каналов в heatmaps и OOF различается")
    if maps.ndim != 4 or maps.shape[1] != len(names) or len(maps) != len(indexes) or hw.shape != (len(maps), 2):
        raise ValueError("Размерности heatmaps/index/image_hw не совпадают")
    if len(set(map(int, indexes))) != len(indexes):
        raise ValueError("В heatmaps повторяются индексы снимков")
    center_cols = [names.index(n) for n in CENTERS]
    rows = []
    for n, j in enumerate(indexes):
        j = int(j)
        if j not in order:
            raise ValueError(f"Индекс {j} отсутствует в OOF")
        o = order[j]
        if regions[o] != "spine" or not visible[o, center_cols].all():
            continue
        height, width = map(int, hw[n])
        if not (0 < height <= maps.shape[2] and 0 < width <= maps.shape[3]):
            raise ValueError(f"Некорректный размер снимка у index={j}")
        h = np.array(maps[n, :, :height, :width], dtype=float)
        active = h.max(axis=(1, 2)) >= thresholds
        pos, support, center_ids = candidates(h, names)
        proposals = center_proposals(pos, support, center_ids, names)
        scored = sorted(((full_score(s, combo, pos, support, names, active), combo)
                         for s, combo in proposals), reverse=True)
        truth = true[o, center_cols]
        baseline_ok, baseline_angle = classify(old_pred[o, center_cols], truth)
        best = scored[:top_k]
        verdicts = [classify(pos[list(combo)], truth) for _, combo in best]
        rows.append({
            "index": j, "n_center_candidates": len(center_ids),
            "n_full_candidates": len(pos), "n_chains": len(scored),
            "baseline_correct_levels": int(baseline_ok),
            "baseline_axis_error_deg": round(baseline_angle, 4),
            "top1_correct_levels": int(verdicts[0][0]) if verdicts else "",
            "top1_axis_error_deg": round(verdicts[0][1], 4) if verdicts else "",
            f"top{top_k}_has_correct_levels": int(any(v[0] for v in verdicts)),
            f"top{top_k}_min_axis_error_deg": round(min((v[1] for v in verdicts), default=float("nan")), 4),
            "score_gap_top2": round(best[0][0] - best[1][0], 4) if len(best) > 1 else "",
        })
    if not rows:
        raise ValueError("Нет снимков позвоночника с размеченными L1–L4")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heatmaps", type=Path, required=True)
    p.add_argument("--oof", type=Path, default=Path("runs/keypoints/oof.npz"))
    p.add_argument("--out", type=Path, default=Path("runs/keypoints/trajectory_hypotheses.csv"))
    a = p.parse_args()
    rows = evaluate(a.heatmaps, a.oof, a.out)
    for key in ("baseline_correct_levels", "top1_correct_levels", "top5_has_correct_levels"):
        usable = [int(r[key]) for r in rows if r[key] != ""]
        print(f"{key}: {sum(usable)}/{len(usable)}")
    print(f"Отчёт: {a.out}")


if __name__ == "__main__":
    main()
