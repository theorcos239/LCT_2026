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
            point=name, n=len(e), n_absent=int(tn + fp),
            median_mm=np.median(e) if len(e) else np.nan,
            p90_mm=np.percentile(e, 90) if len(e) else np.nan,
            f1_visible=2 * tp / max(2 * tp + fp + fn, 1),
            # неприменимо, если отсутствующих примеров не было вовсе
            recall_absent=tn / (tn + fp) if tn + fp else np.nan,
        )
        for t in PCK_THRESHOLDS_MM:
            row[f"pck@{t:g}mm"] = (e <= t).mean() if len(e) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_thresholds(names: list[str], conf: np.ndarray, visible: np.ndarray,
                         labeled: np.ndarray, grid: np.ndarray | None = None,
                         objective: str = "balanced", default: float = 0.5,
                         min_negatives: int = 10, fallback_miss: float = 0.02) -> pd.Series:
    """Порог видимости по каждой точке отдельно: у th12_top и trochanter_major разная сложность.

    objective="f1" — прежнее правило: максимум F1 класса «видна». Отсутствующих
    примеров мало, и F1 почти не штрафует за ложную «видимость», поэтому пороги
    уезжали на нижний край сетки (0.05) — модель переставала говорить «точки
    нет» и выдумывала её. objective="balanced" — максимум (TPR + TNR) / 2: пропуск
    видимой и выдумывание отсутствующей точки стоят одинаково.

    Если отсутствующих примеров меньше min_negatives (середина позвоночника:
    центры L2–L3 в кадре почти всегда, даже с обрезанным краем), балансировать
    не с чем. Тогда порог — квантиль fallback_miss уверенности видимых: теряется
    не больше 2 % видимых. Фиксированный default здесь не годится: у разных
    каналов пик разной высоты, и порог 0.5 терял до 95 % видимых L2 и L3.
    default — только если нет и видимых примеров.
    """
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    out = {}
    for c, name in enumerate(names):
        lab, vis, cf = labeled[:, c], visible[:, c], conf[:, c]
        pos, neg = (vis & lab), (~vis & lab)
        if objective == "balanced" and neg.sum() < min_negatives:
            out[name] = (float(np.clip(np.quantile(cf[pos], fallback_miss), grid[0], grid[-1]))
                         if pos.any() else default)
            continue
        best, best_score = default, -1.0
        for t in grid:
            found = cf >= t
            tp, fn = (found & pos).sum(), (~found & pos).sum()
            fp, tn = (found & neg).sum(), (~found & neg).sum()
            if objective == "f1":
                score = 2 * tp / max(2 * tp + fp + fn, 1)
            else:
                score = tp / max(tp + fn, 1) + tn / max(tn + fp, 1)
            if score > best_score + 1e-12:
                best, best_score = float(t), score
        out[name] = best
    return pd.Series(out, name="threshold")


def visibility_table(names: list[str], conf: np.ndarray, visible: np.ndarray,
                     labeled: np.ndarray, threshold) -> pd.DataFrame:
    """Сколько раз модель выдумывает точку и сколько раз теряет видимую.

    hallucination_rate — доля отсутствующих в кадре точек, которые модель
    объявила видимыми; miss_rate — доля видимых, объявленных отсутствующими.
    """
    found = conf >= np.asarray(threshold)
    rows = []
    for c, name in enumerate(names):
        lab = labeled[:, c]
        pos, neg = visible[:, c] & lab, ~visible[:, c] & lab
        tp, fn = (found[:, c] & pos).sum(), (~found[:, c] & pos).sum()
        fp, tn = (found[:, c] & neg).sum(), (~found[:, c] & neg).sum()
        rows.append(dict(point=name, n_visible=int(pos.sum()), n_absent=int(neg.sum()),
                         hallucination_rate=fp / neg.sum() if neg.sum() else np.nan,
                         miss_rate=fn / pos.sum() if pos.sum() else np.nan,
                         balanced_acc=0.5 * (tp / max(pos.sum(), 1) + tn / max(neg.sum(), 1))
                         if neg.sum() and pos.sum() else np.nan))
    return pd.DataFrame(rows)


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
    """Ошибка производных величин: угол оси, отступы ROI и выступ малого вертела.

    Выступ — shaft_medial.x − trochanter_minor.x (в кадре модели медиальная
    сторона слева), та же конструкция, что `bulge_from_points` в сервисе.
    """
    idx = {n: i for i, n in enumerate(names)}
    bulge = (idx["trochanter_minor"], idx["shaft_medial"])
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
            if visible[i, list(bulge)].all():
                b_pred = pred[i, bulge[1], 0] - pred[i, bulge[0], 0]
                b_true = true[i, bulge[1], 0] - true[i, bulge[0], 0]
                rows.append(dict(kind="bulge_mm", error=abs(b_pred - b_true) * MM_PER_PX))
    if not rows:
        return pd.DataFrame(columns=["kind", "median", "p90", "n"])
    df = pd.DataFrame(rows)
    return (df.groupby("kind").error.agg(median="median",
                                         p90=lambda s: np.percentile(s, 90), n="size")
              .reset_index())
