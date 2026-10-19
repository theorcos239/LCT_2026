"""Синтетическая проверка смещения наклона линией центров на C-дуге."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dxa_qc.spine_shape import _angle_vertical  # noqa: E402


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
    parser.add_argument("--out", default="runs/keypoints_crops/axis_bias_simulation.csv")
    args = parser.parse_args()
    rows = []
    for theta in np.linspace(-8, 8, 9):
        for kappa_l in np.linspace(0, .3, 7):
            for vertex in np.linspace(-3, 3, 13):
                measured = _angle_vertical(curve(theta, kappa_l, vertex))
                # Точная угловая форма; κL·ta — её малоугловое приближение.
                predicted_bias = np.degrees(np.arctan(kappa_l * vertex))
                rows.append({
                    "theta_true_deg": theta, "kappa_l": kappa_l,
                    "vertex_t": vertex, "theta_centers_deg": measured,
                    "bias_deg": measured - theta,
                    "predicted_bias_deg": predicted_bias,
                })
    df = pd.DataFrame(rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    mae = np.mean(np.abs(df.bias_deg - df.predicted_bias_deg))
    print(f"{path}: {len(df)} случаев, MAE формулы {mae:.3f}°")


if __name__ == "__main__":
    main()
