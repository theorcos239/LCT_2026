"""Метрики по каждой точке отдельно: усреднение по каналам скрывает провалы.

Ошибка в мм, PCK и F1 видимости; отдельно — сквозные величины (угол оси, отступы),
потому что 2 мм на l1_center критичны, а на neck_center нет.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MM_PER_PX = 0.6
PCK_THRESHOLDS_MM = (1.0, 3.0, 5.0)


def per_point(names: list[str], pred: np.ndarray, conf: np.ndarray, true: np.ndarray,
              visible: np.ndarray, labeled: np.ndarray,
              threshold: float | np.ndarray = 0.3,
              mm_per_px: float = MM_PER_PX) -> pd.DataFrame:
    """pred/true: (N, C, 2); conf, visible, labeled: (N, C); threshold — число или вектор по точкам."""
    err = np.linalg.norm(pred - true, axis=-1) * mm_per_px
    found = conf >= np.asarray(threshold)
    rows = []
    for c, name in enumerate(names):
        e = err[visible[:, c], c]
        lab = labeled[:, c]
        tp = (found[:, c] & visible[:, c] & lab).sum()
        fp = (found[:, c] & ~visible[:, c] & lab).sum()
        fn = (~found[:, c] & visible[:, c] & lab).sum()
        tn = (~found[:, c] & ~visible[:, c] & lab).sum()
        row = dict(
            point=name, n=len(e),
            median_mm=np.median(e) if len(e) else np.nan,
            p90_mm=np.percentile(e, 90) if len(e) else np.nan,
            f1_visible=2 * tp / max(2 * tp + fp + fn, 1),
            recall_absent=tn / max(tn + fp, 1),
        )
        for t in PCK_THRESHOLDS_MM:
            row[f"pck@{t:g}mm"] = (e <= t).mean() if len(e) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_thresholds(names: list[str], conf: np.ndarray, visible: np.ndarray,
                         labeled: np.ndarray, grid: np.ndarray | None = None) -> pd.Series:
    """Порог видимости по каждой точке отдельно: у th12_top и trochanter_major разная сложность."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    out = {}
    for c, name in enumerate(names):
        lab, vis, cf = labeled[:, c], visible[:, c], conf[:, c]
        best, best_f1 = 0.5, -1.0
        for t in grid:
            found = cf >= t
            tp = (found & vis & lab).sum()
            fp = (found & ~vis & lab).sum()
            fn = (~found & vis & lab).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best, best_f1 = float(t), f1
        out[name] = best
    return pd.Series(out, name="threshold")


def axis_angle(centers: np.ndarray) -> float:
    """Наклон прямой через центры тел позвонков к вертикали, градусы."""
    p = centers - centers.mean(0)
    v = np.linalg.svd(p, full_matrices=False)[2][0]
    if v[1] < 0:
        v = -v
    return float(np.degrees(np.arctan2(v[0], v[1])))


def axis_curvature(centers: np.ndarray, mm_per_px: float = MM_PER_PX) -> float:
    """Максимальное отклонение центра от прямой, мм: отличает сколиоз от наклона укладки."""
    p = centers - centers.mean(0)
    v = np.linalg.svd(p, full_matrices=False)[2][0]
    n = np.array([-v[1], v[0]])
    return float(np.abs(p @ n).max() * mm_per_px)


def end_to_end(names: list[str], pred: np.ndarray, true: np.ndarray, visible: np.ndarray,
               regions: np.ndarray) -> pd.DataFrame:
    """Ошибка производных величин: угол оси и отступы, предсказание против разметки."""
    idx = {n: i for i, n in enumerate(names)}
    spine_axis = [idx[f"l{i}_center"] for i in range(1, 5)]
    margins = {"top": idx["trochanter_major"], "bottom": idx["ischium_bottom"],
               "lateral": idx["trochanter_lateral"]}
    rows = []
    for i, region in enumerate(regions):
        if region == "spine" and visible[i, spine_axis].all():
            rows.append(dict(kind="axis_deg", error=abs(axis_angle(pred[i, spine_axis])
                                                        - axis_angle(true[i, spine_axis]))))
        elif region == "hip":
            for name, c in margins.items():
                if visible[i, c]:
                    d = np.abs(pred[i, c] - true[i, c]) * MM_PER_PX
                    rows.append(dict(kind=f"margin_{name}_mm",
                                     error=d[1] if name != "lateral" else d[0]))
    if not rows:
        return pd.DataFrame(columns=["kind", "median", "p90", "n"])
    df = pd.DataFrame(rows)
    return (df.groupby("kind").error.agg(median="median",
                                         p90=lambda s: np.percentile(s, 90), n="size")
              .reset_index())
