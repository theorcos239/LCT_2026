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
