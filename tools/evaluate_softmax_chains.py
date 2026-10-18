"""Исследовательская проверка выбора цепочки по нескольким пикам.

Считает карты только на отложенных снимках каждого доступного фолда. Сравнивает
главный пик, геометрический перебор, нормировку откликов по 21 карте в каждой
кандидатной позиции и поддержку дисков. Ничего в рабочем инференсе не меняет.

PYTHONPATH=src python tools/evaluate_softmax_chains.py \\
    --run Processing/drive_models --device mps
"""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from serve_keypoint_heatmaps import Viewer, find_peaks
from dxa_qc.structure import chain_decode_hypotheses


ROOT = Path(__file__).resolve().parents[1]
CENTERS = [f"l{i}_center" for i in range(1, 5)]
DISCS = ["disc_th12_l1", "disc_l1_l2", "disc_l2_l3", "disc_l3_l4", "disc_l4_l5"]
DISC_BANDS = [(0, -1.0, 0.0), (0, 0.0, 1.0), (1, 0.0, 1.0),
              (2, 0.0, 1.0), (3, 0.0, 1.0)]


def top_peaks(h: np.ndarray, limit: int = 6) -> list[dict]:
    floor = min(.02, max(.001, float(h.max()) * .1))
    return find_peaks(h, floor=floor, top_k=limit)


def candidates(h: np.ndarray, center_cols: list[int]) -> np.ndarray:
    hits = sorted((p for c in center_cols for p in top_peaks(h[c])),
                  key=lambda p: p["value"], reverse=True)
    pool = []
    for p in hits:
        xy = np.array([p["x"], p["y"]], dtype=float)
        if all(np.linalg.norm(xy - q) > 8 for q in pool):
            pool.append(xy)
    return np.array(sorted(pool, key=lambda p: p[1])).reshape(-1, 2)


def support_at(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """21 отклик для каждой позиции с допуском в два пикселя."""
    value = np.empty((len(xy), 21), dtype=float)
    height, width = h.shape[1:]
    for i, (x, y) in enumerate(xy.astype(int)):
        value[i] = h[:21, max(0, y - 2):min(height, y + 3),
                     max(0, x - 2):min(width, x + 3)].max(axis=(1, 2))
    return value


def disc_score(chain: np.ndarray, disc_peaks: list[list[dict]]) -> float:
    step = (chain[-1] - chain[0]) / 3
    size = float(np.linalg.norm(step))
    if size < 1:
        return -100.0
    values = []
    for (k, low, high), peaks in zip(DISC_BANDS, disc_peaks):
        best = 1e-4
        for p in peaks:
            d = np.array([p["x"], p["y"]]) - chain[k]
            along = float(d @ step) / size**2
            across = abs(float(d[0] * step[1] - d[1] * step[0])) / size
            if low <= along <= high and across <= size:
                best = max(best, p["value"])
        values.append(np.log(best))
    return float(sum(values))


def label(chain: np.ndarray, truth: np.ndarray, th12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.vstack([th12, truth, truth[-1] + (truth[-1] - truth[-2])])
    closest = np.linalg.norm(chain[:, None] - reference[None], axis=-1).argmin(axis=1)
    good = closest == np.arange(1, 5)
    near = np.linalg.norm(chain - truth, axis=1) <= 25
    return good, near


def one_image(h: np.ndarray, names: list[str], truth: np.ndarray,
              th12: np.ndarray) -> dict:
    center_cols = [names.index(n) for n in CENTERS]
    disc_cols = [names.index(n) for n in DISCS]
    base = np.array([np.unravel_index(h[c].argmax(), h[c].shape)[::-1]
                     for c in center_cols], dtype=float)
    base_good, base_near = label(base, truth, th12)
    pool = candidates(h, center_cols)
    result = {"n_candidates": len(pool), "baseline_good": int(base_good.sum()),
              "baseline_all": int(base_good.all()),
              "baseline_near": int(base_near.sum())}
    # Та же функция, которую вызывает рабочий Predictor после проверки видимости.
    initial = np.zeros((len(names), 2), dtype=float)
    initial[center_cols] = base
    actual, diagnostic = chain_decode_hypotheses(
        h, names, initial, np.ones(len(names), dtype=bool))
    production_good, production_near = label(actual[center_cols], truth, th12)
    result.update(production_good=int(production_good.sum()),
                  production_all=int(production_good.all()),
                  production_near=int(production_near.sum()),
                  production_used=int(diagnostic["used"]),
                  production_top1_weight=diagnostic.get("top1_relative_weight"))
    if len(pool) < 4:
        return result
    support = support_at(h, pool).clip(1e-4)
    log_raw = np.log(support[:, center_cols])
    log_normalized = log_raw - np.log(support.sum(axis=1, keepdims=True))
    scored = []
    for combo in combinations(range(len(pool)), 4):
        chain = pool[list(combo)]
        steps = np.diff(chain, axis=0)
        mean = steps.mean(axis=0)
        length = float(np.linalg.norm(mean))
        if not (25 <= length <= 110 and mean[1] > 0):
            continue
        uneven = float(np.square(steps - mean).sum() / length**2)
        raw = float(sum(log_raw[i, k] for k, i in enumerate(combo)) - 10 * uneven)
        norm = float(sum(log_normalized[i, k] for k, i in enumerate(combo)) - 10 * uneven)
        scored.append((combo, raw, norm))
    result["n_chains"] = len(scored)
    if not scored:
        return result
    disc_peaks = [top_peaks(h[c]) for c in disc_cols]
    # Дорогую оценку по дискам делаем только для верхних 100 цепочек по каждому
    # из двух основных баллов. Это фиксированный до просмотра разметки отбор.
    shortlist = set()
    for field in (1, 2):
        shortlist.update(row[0] for row in sorted(scored, key=lambda x: x[field],
                                                  reverse=True)[:100])
    disc = {combo: disc_score(pool[list(combo)], disc_peaks) for combo in shortlist}
    score_funcs = {
        "raw": lambda x: x[1],
        "normalized": lambda x: x[2],
        "raw_discs": lambda x: x[1] + .5 * disc[x[0]],
        "normalized_discs": lambda x: x[2] + .5 * disc[x[0]],
    }
    variants = {
        name: sorted((x for x in scored if x[0] in shortlist)
                     if "discs" in name else scored, key=score_func, reverse=True)
        for name, score_func in score_funcs.items()
    }
    for name, rows in variants.items():
        chain = pool[list(rows[0][0])]
        good, near = label(chain, truth, th12)
        result[f"{name}_good"] = int(good.sum())
        result[f"{name}_all"] = int(good.all())
        result[f"{name}_near"] = int(near.sum())
        result[f"{name}_oracle5"] = int(any(label(pool[list(r[0])], truth, th12)[0].all()
                                              for r in rows[:5]))
        best = rows[:5]
        scores = np.array([score_funcs[name](r) for r in best])
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        result[f"{name}_softmax_top1"] = float(weights[0])
        averaged = sum(w * pool[list(r[0])] for w, r in zip(weights, best))
        avg_good, avg_near = label(averaged, truth, th12)
        result[f"{name}_average_all"] = int(avg_good.all())
        result[f"{name}_average_near"] = int(avg_near.sum())
    return result


def evaluate(run: Path, pack: Path, oof: Path, device: str, output: Path) -> list[dict]:
    with np.load(oof, allow_pickle=False) as z:
        order = {int(j): i for i, j in enumerate(z["index"])}
        truth, visible = z["true"], z["visible"]
        names = list(z["names"])
        regions = z["regions"]
        center_cols = [names.index(n) for n in CENTERS]
        th12_cols = [names.index(n) for n in ("th12_top", "th12_bottom")]
        rows = []
        for checkpoint in sorted(run.glob("fold[0-9].pt")):
            viewer = Viewer(checkpoint, pack, device)
            for j in viewer.indices:
                if str(viewer.data["regions"][j]) != "spine" or j not in order:
                    continue
                o = order[j]
                if regions[o] != "spine" or not visible[o, center_cols].all():
                    continue
                true = truth[o, center_cols]
                th12 = (truth[o, th12_cols].mean(axis=0) if visible[o, th12_cols].all()
                        else true[0] - (true[1] - true[0]))
                result = one_image(viewer.heatmaps(j), names, true, th12)
                rows.append({"fold": checkpoint.stem, "index": j,
                             "id": str(viewer.data["ids"][j]), **result})
                print(f"{checkpoint.stem}: {len(rows)} снимков", end="\r", flush=True)
    if not rows:
        raise RuntimeError("Нет отложенных снимков позвоночника")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print()
    for name in ("baseline", "production", "raw", "normalized", "raw_discs", "normalized_discs"):
        usable = [r for r in rows if f"{name}_all" in r]
        print(f"{name}: {sum(r[f'{name}_all'] for r in usable)}/{len(usable)} "
              f"снимков; {sum(r[f'{name}_good'] for r in usable)}/{4 * len(usable)} "
              "центров", end="")
        if name not in ("baseline", "production"):
            print(f"; верная цепочка в top-5: "
                  f"{sum(r[f'{name}_oracle5'] for r in usable)}/{len(usable)}; "
                  f"softmax-среднее: "
                  f"{sum(r[f'{name}_average_all'] for r in usable)}/{len(usable)}")
        else:
            print()
    print(f"Отчёт: {output}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "Processing/drive_models")
    p.add_argument("--pack", type=Path, default=ROOT / "data/keypoints.npz")
    p.add_argument("--oof", type=Path, default=ROOT / "runs/keypoints/oof.npz")
    p.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    p.add_argument("--out", type=Path,
                   default=ROOT / "Processing/drive_models/softmax_chain_eval.csv")
    a = p.parse_args()
    evaluate(a.run, a.pack, a.oof, a.device, a.out)


if __name__ == "__main__":
    main()
