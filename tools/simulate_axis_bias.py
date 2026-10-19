"""Синтетическая проверка смещения наклона линией центров на C-дуге."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dxa_qc.spine_shape import (_angle_vertical, curve_descriptors, fused_tilt,
                                source_variances, tilt_sources)  # noqa: E402

NAMES = ([f"l{i}_center" for i in range(1, 5)]
         + [name for i in range(1, 5)
            for name in (f"l{i}_edge_left", f"l{i}_edge_right")])


def curve(theta_deg: float, kappa_l: float, vertex_t: float,
          step: float = 50.0) -> np.ndarray:
    t = np.array([-1.5, -.5, .5, 1.5])
    lateral = -.5 * kappa_l * step * (t - vertex_t) ** 2
    base = np.stack([lateral, t * step], axis=1)
    a = np.deg2rad(theta_deg)
    # Поворот, сохраняющий конвенцию: положительный угол = dx/dy > 0.
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return base @ rot.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/keypoints_crops/metrics_axis_synthetic.csv")
    args = parser.parse_args()
    rows = []
    for theta in np.linspace(-8, 8, 9):
        for kappa_l in np.linspace(0, .3, 7):
            for vertex in np.linspace(-3, 3, 13):
                measured = _angle_vertical(curve(theta, kappa_l, vertex))
                # Точная угловая форма; κL·ta — её малоугловое приближение.
                predicted_bias = np.degrees(np.arctan(kappa_l * vertex))
                for gamma_fraction in (0.0, .5, 1.0):
                    centers = curve(theta, kappa_l, vertex)
                    # Собственный наклон тела в дуге проверяется тремя
                    # сценариями; точки краёв создаются из этой пластинки.
                    positions = np.arange(4, dtype=float) - 1.5
                    local_angles = theta + gamma_fraction * np.degrees(
                        np.arctan(kappa_l * (vertex - positions)))
                    edges = []
                    for center, local_angle in zip(centers, local_angles):
                        alpha = np.deg2rad(local_angle)
                        edge_vector = np.array([20 * np.cos(alpha), -20 * np.sin(alpha)])
                        edges.extend((center + edge_vector, center - edge_vector))
                    points = np.vstack([centers, edges])
                    src = tilt_sources(points, np.ones(len(points), bool), NAMES,
                                       contour={"angle_deg": theta + .7 * (measured - theta)})
                    descriptors = curve_descriptors(points, np.ones(len(points), bool), NAMES)
                    sig = source_variances(src, descriptors,
                                           {"contour_curvature_mm": abs(kappa_l) * 30},
                                           point_sigma_px=1.5)
                    fused = fused_tilt(src, sig)
                    solo = [abs(src[key] - theta) for key in
                            ("theta_centers_deg", "theta_endplates_deg", "theta_contour_deg")]
                    rows.append({
                        "theta_true_deg": theta, "kappa_l": kappa_l,
                        "vertex_t": vertex, "gamma_fraction": gamma_fraction,
                        "theta_centers_deg": measured, "bias_deg": measured - theta,
                        "predicted_bias_deg": predicted_bias,
                        "fused_error_deg": abs(fused["theta_fused_deg"] - theta),
                        "best_source_error_deg": min(solo),
                        "excess_error_deg": abs(fused["theta_fused_deg"] - theta) - min(solo),
                    })
    df = pd.DataFrame(rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    mae = np.mean(np.abs(df.bias_deg - df.predicted_bias_deg))
    max_excess = float(df.excess_error_deg.max())
    summary = {
        "n": len(df), "formula_mae_deg": float(mae),
        "fused_max_excess_deg": max_excess,
        "fused_gate_passed": bool(max_excess <= .3),
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{path}: {len(df)} случаев, MAE формулы {mae:.3f}°, "
          f"максимальный проигрыш слияния {max_excess:.3f}°")


if __name__ == "__main__":
    main()
