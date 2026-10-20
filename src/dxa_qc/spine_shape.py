"""Геометрия наклона позвоночника, устойчивая к сколиотической дуге.

Угол положителен, когда низ оси уходит вправо. Все координаты могут быть в
пикселях: ``pixel_mm`` переводит длины в мм, а углы от масштаба не зависят.
"""
from __future__ import annotations

import numpy as np

CENTERS = tuple(f"l{i}_center" for i in range(1, 5))
EDGE_PAIRS = tuple((f"l{i}_edge_left", f"l{i}_edge_right") for i in range(1, 5))


def _angle_vertical(points: np.ndarray, weights: np.ndarray | None = None) -> float:
    p = np.asarray(points, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    mean = np.average(p, axis=0, weights=w)
    q = p - mean
    cov = (q * w[:, None]).T @ q / max(float(w.sum()), 1e-9)
    vec = np.linalg.eigh(cov)[1][:, -1]
    if vec[1] < 0:
        vec = -vec
    return float(np.degrees(np.arctan2(vec[0], vec[1])))


def _circular_mean_deg(values: list[float], weights: list[float] | None = None) -> float:
    a = np.deg2rad(np.asarray(values, float))
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    return float(np.degrees(np.arctan2(np.sum(w * np.sin(a)), np.sum(w * np.cos(a)))))


def tilt_sources(points: np.ndarray, visible: np.ndarray, names: list[str],
                 pixel_mm: float = .6, contour: dict | None = None,
                 confidence: np.ndarray | None = None) -> dict:
    """Вернуть независимые оценки угла: центры, пластинки и контур."""
    del pixel_mm  # масштаб нужен дескрипторам, не углам
    idx = {n: i for i, n in enumerate(names)}
    conf = np.ones(len(names)) if confidence is None else np.asarray(confidence, float)
    center_cols = [idx[n] for n in CENTERS if n in idx and visible[idx[n]]]
    theta_c = (_angle_vertical(np.asarray(points)[center_cols], conf[center_cols])
               if len(center_cols) >= 3 else None)
    alphas, widths, alpha_conf = [], [], []
    mids = []
    for left, right in EDGE_PAIRS:
        if left not in idx or right not in idx:
            continue
        li, ri = idx[left], idx[right]
        if not (visible[li] and visible[ri]):
            continue
        a, b = np.asarray(points[li], float), np.asarray(points[ri], float)
        d = a - b
        if d[0] < 0:
            d = -d
        # Горизонтальная пластинка на вертикальном кадре имеет угол 0.
        alphas.append(float(-np.degrees(np.arctan2(d[1], d[0]))))
        widths.append(float(np.linalg.norm(d)))
        alpha_conf.append(float(np.sqrt(max(conf[li], 1e-6) * max(conf[ri], 1e-6))))
        mids.append((a + b) / 2)
    theta_e = _circular_mean_deg(alphas, alpha_conf) if len(alphas) >= 2 else None
    theta_m = _angle_vertical(np.asarray(mids)) if len(mids) >= 3 else None
    theta_k = None
    if contour:
        theta_k = contour.get("angle_deg")
        theta_k = None if theta_k is None else float(theta_k)
    return {
        "theta_centers_deg": theta_c,
        "theta_endplates_deg": theta_e,
        "theta_edge_mids_deg": theta_m,
        "theta_contour_deg": theta_k,
        "endplate_angles_deg": alphas,
        "endplate_widths_px": widths,
        "n_endplates": len(alphas),
    }


# Высокоточное правило сколиоза (pack/fold2, приоритет — не ставить там,
# где его нет): P=1.0, FP=0, TP=5 из 12.
#   • средняя дуга: segment_range ∈ [12°, 16°]
#   • экстремум: seg≥20° и только с lateral_dev≥4 мм (один seg без латерали
#     часто сбой точек: 16–26° без метки сколиоза)
#   • зазор 16–20° намеренно пуст
# Сильный режим (гашение оси): mid-band ≥14° или экстремум. Max seg среди
# y_axis=1 ≈8.4° — ни одно срабатывание туда не попадает.
SCOLIOSIS_SEG_LO_DEG = 12.0
SCOLIOSIS_SEG_HI_DEG = 16.0
SCOLIOSIS_SEG_EXTREME_DEG = 20.0
SCOLIOSIS_LAT_EXTREME_MM = 4.0
SCOLIOSIS_STRONG_SEG_DEG = 14.0
# Обратная совместимость имён (раньше были простые пороги ≥).
SCOLIOSIS_REPORT_DEG = SCOLIOSIS_SEG_LO_DEG
SCOLIOSIS_STRONG_DEG = SCOLIOSIS_STRONG_SEG_DEG


def curve_descriptors(points: np.ndarray, visible: np.ndarray, names: list[str],
                      pixel_mm: float = .6) -> dict:
    """Кривизна и положение вершины дуги в нормированных шагах L1-L4."""
    idx = {n: i for i, n in enumerate(names)}
    if not all(n in idx and visible[idx[n]] for n in CENTERS):
        return {"available": False}
    centers = np.asarray([points[idx[n]] for n in CENTERS], float)
    chord = centers[-1] - centers[0]
    step_px = float(np.linalg.norm(chord) / 3)
    if step_px < 1e-6:
        return {"available": False}
    u = chord / np.linalg.norm(chord)
    normal = np.array([u[1], -u[0]])
    origin = centers.mean(0)
    t = (centers - origin) @ u / step_px
    lateral = (centers - origin) @ normal
    q2, q1, q0 = np.polyfit(t, lateral, 2)
    fitted = q2 * t ** 2 + q1 * t + q0
    residual = lateral - fitted
    # В системе хорды линейная часть дуги поглощена самой хордой. Поэтому
    # положение вершины по одним центрам неидентифицируемо; q1 сохраняем
    # только как диагностику численной асимметрии.
    vertex_chord = float(-q1 / (2 * q2)) if abs(q2) > 1e-6 else None
    delta = np.diff(centers, axis=0)
    segment_angles = [
        float(np.degrees(np.arctan2(delta[i, 0], delta[i, 1])))
        for i in range(3)
    ]
    # Выражаем кривизну как безразмерный поворот дуги на один шаг.
    kappa_l = float(2 * q2 / step_px)
    return {
        "available": True,
        "step_px": step_px,
        "step_mm": step_px * pixel_mm,
        "kappa_l": kappa_l,
        "vertex_t_chord": vertex_chord,
        "residual_rms_mm": float(np.sqrt(np.mean(residual ** 2)) * pixel_mm),
        "lateral_dev_mm": float(np.max(np.abs(lateral)) * pixel_mm),
        "segment_angles_deg": segment_angles,
        "segment_range_deg": float(np.ptp(segment_angles)),
    }


def assess_scoliosis(curve: dict | None,
                     report_thr: float | None = None,
                     strong_thr: float | None = None) -> dict:
    """Сколиоз по размаху сегментов L1–L4 с упором на precision.

    Не тип нарушения ТЗ: поле отчёта + осторожная политика оси.
    ``strength``: ``none`` / ``weak`` (есть, ось только с corroboration) /
    ``strong`` (можно гасить ложный наклон хорды).

    ``report_thr`` / ``strong_thr`` — опциональные переопределения нижней
    границы mid-band и порога strong внутри неё (тесты/калибровка).
    """
    seg_lo = float(report_thr if report_thr is not None else SCOLIOSIS_SEG_LO_DEG)
    strong_lo = float(strong_thr if strong_thr is not None else SCOLIOSIS_STRONG_SEG_DEG)
    meta = {
        "report_thr_deg": seg_lo, "strong_thr_deg": strong_lo,
        "seg_hi_deg": SCOLIOSIS_SEG_HI_DEG,
        "seg_extreme_deg": SCOLIOSIS_SEG_EXTREME_DEG,
        "lat_extreme_mm": SCOLIOSIS_LAT_EXTREME_MM,
    }
    if not curve or not curve.get("available"):
        return {
            "available": False, "present": False, "strength": "none",
            "segment_range_deg": None, "lateral_dev_mm": None, **meta,
        }
    seg = float(curve["segment_range_deg"])
    lat = float(curve.get("lateral_dev_mm") or 0.0)
    mid = seg_lo <= seg <= SCOLIOSIS_SEG_HI_DEG
    extreme = (seg >= SCOLIOSIS_SEG_EXTREME_DEG
               and lat >= SCOLIOSIS_LAT_EXTREME_MM)
    present = bool(mid or extreme)
    if not present:
        strength = "none"
    elif extreme or seg >= strong_lo:
        strength = "strong"
    else:
        strength = "weak"
    return {
        "available": True, "present": present, "strength": strength,
        "segment_range_deg": round(seg, 2),
        "lateral_dev_mm": round(lat, 2),
        **meta,
    }


def apply_scoliosis_to_axis(axis: dict | None, scoliosis: dict | None) -> dict | None:
    """Смягчить вердикт оси при сколиозе. Никогда не создаёт новое нарушение.

    Политика (pack/fold2: флаг сколиоза только при FP=0; expert редко
    ставит y_axis при сколиозе):

    * ``strong`` + ось «брак»/review → как ``curved_suppressed``:
      violated=False, флаг ``axis:scoliosis_suppressed``.
    * ``weak`` + ось «брак» + подтверждение изгиба/расхождения →
      undetermined / review, не авто-OK.
    * ``weak`` без подтверждения → ось не трогаем.
    * чистая ось → не трогаем.
    """
    if not axis or not scoliosis or not scoliosis.get("available"):
        return axis
    out = dict(axis)
    out["scoliosis"] = scoliosis
    flags = list(out.get("flags") or [])
    reasons = list(out.get("reasons") or [])
    strength = scoliosis.get("strength") or "none"
    if strength == "none":
        out["flags"] = flags
        return out

    state = out.get("state")
    violated = out.get("violated")
    # Уже чистое OK не трогаем. Недоступные источники — тоже (не маскируем
    # «точек нет» под сколиоз).
    if violated is False and state in (None, "suitable"):
        out["flags"] = flags
        return out
    if "sources_missing" in reasons or "axis:points_missing" in flags:
        out["flags"] = flags
        return out

    corroboration = any(f in flags for f in (
        "axis:estimates_disagree", "axis:curved_suppressed", "axis:curved",
    )) or any(r in reasons for r in (
        "scoliosis_like_curve", "scoliosis_like_contour", "sources_disagree",
    ))
    is_hard = (violated is True) or (state == "unsuitable")

    if strength == "strong" and (is_hard or violated is None or state == "undetermined"):
        # Сильный размах сегментов: и жёсткий брак, и «дуга → review» гасим
        # до OK — эксперт на сколиозе y_axis почти не ставит.
        out["violated"] = False
        out["text"] = ""
        if state is not None or "state" in out:
            out["state"] = "suitable"
        flags = [f for f in flags if f != "axis:needs_review"]
        if "axis:scoliosis_suppressed" not in flags:
            flags.append("axis:scoliosis_suppressed")
        if "scoliosis_segment_range" not in reasons:
            reasons.append("scoliosis_segment_range")
        out["reasons"] = reasons
        out["flags"] = flags
        return out

    # weak: только review при подтверждении изгиба/расхождения
    if strength == "weak" and is_hard and corroboration:
        out["violated"] = None
        out["text"] = ""
        if "state" in out or state is not None:
            out["state"] = "undetermined"
        if "axis:needs_review" not in flags:
            flags.append("axis:needs_review")
        if "axis:scoliosis_review" not in flags:
            flags.append("axis:scoliosis_review")
        if "scoliosis_borderline" not in reasons:
            reasons.append("scoliosis_borderline")
        out["reasons"] = reasons
    out["flags"] = flags
    return out


def source_variances(src: dict, curve: dict, params: dict | None = None,
                     point_sigma_px: float | None = None) -> dict:
    """Дисперсии источников: дуга уменьшает вес линии центров и контура."""
    p = {
        "sigma_c0": 1.25, "sigma_e0": 2.0, "sigma_k0": 1.5,
        "b_c": 1.0, "b_e": 1.0, "b_k": 1.0, "b_r": 1.0, "tau": 1.5,
        **(params or {}),
    }
    kappa = abs(float(curve.get("kappa_l") or 0.0))
    # По центрам отдельно вершина неидентифицируема. Независимые пластинки
    # дают её эффективное плечо: θC-θE ≈ atan(κL·ta).
    theta_c, theta_e = src.get("theta_centers_deg"), src.get("theta_endplates_deg")
    if theta_c is not None and theta_e is not None and kappa > 1e-4:
        lever = min(abs(np.tan(np.deg2rad(theta_c - theta_e))) / kappa, 3.0)
    else:
        lever = p["tau"]
    lever = max(float(lever), 0.25)
    residual = float(curve.get("residual_rms_mm") or 0.0)
    step_mm = max(float(curve.get("step_mm") or 1.0), 1e-6)
    sigma_c = np.hypot(p["sigma_c0"], p["b_c"] * kappa * lever * 180 / np.pi)
    sigma_c = np.hypot(sigma_c, p["b_r"] * residual / step_mm * 180 / np.pi)
    alphas = np.asarray(src.get("endplate_angles_deg") or [], float)
    spread = float(np.std(alphas)) if len(alphas) >= 2 else 99.0
    n_e = max(int(src.get("n_endplates") or 0), 1)
    sigma_e = np.hypot(p["sigma_e0"] / np.sqrt(n_e),
                       p["b_e"] * spread / np.sqrt(n_e))
    if point_sigma_px is not None and src.get("endplate_widths_px"):
        width = max(float(np.median(src["endplate_widths_px"])), 1.0)
        sigma_e = np.hypot(sigma_e, np.degrees(np.sqrt(2) * point_sigma_px / width)
                           / np.sqrt(n_e))
    curvature_mm = float((params or {}).get("contour_curvature_mm", 0.0))
    sigma_k = np.hypot(p["sigma_k0"], p["b_k"] * curvature_mm / step_mm * 180 / np.pi)
    return {
        "theta_centers_deg": float(sigma_c),
        "theta_endplates_deg": float(sigma_e),
        "theta_contour_deg": float(sigma_k),
    }


def fused_tilt(src: dict, sigmas: dict) -> dict:
    """Объединить углы по обратным дисперсиям и проверить χ²-согласованность."""
    keys = ("theta_centers_deg", "theta_endplates_deg", "theta_contour_deg")
    active = [(k, float(src[k]), max(float(sigmas[k]), 1e-3))
              for k in keys if src.get(k) is not None and sigmas.get(k) is not None]
    if not active:
        return {"theta_fused_deg": None, "sigma_deg": None, "Q": None,
                "weights": {}, "n_sources": 0}
    raw = np.asarray([1 / s ** 2 for _, _, s in active])
    weights = raw / raw.sum()
    theta = float(sum(w * value for w, (_, value, _) in zip(weights, active)))
    sigma = float(raw.sum() ** -.5)
    q = float(sum((value - theta) ** 2 / s ** 2 for _, value, s in active))
    return {
        "theta_fused_deg": theta, "sigma_deg": sigma, "Q": q,
        "weights": {key: float(w) for w, (key, _, _) in zip(weights, active)},
        "n_sources": len(active),
    }
