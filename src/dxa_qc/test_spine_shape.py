from __future__ import annotations

import numpy as np

from .spine_shape import (_angle_vertical, curve_descriptors, fused_tilt, source_variances,
                          tilt_sources)


NAMES = ([f"l{i}_center" for i in range(1, 5)]
         + [n for i in range(1, 5)
            for n in (f"l{i}_edge_left", f"l{i}_edge_right")])


def _points(theta=0.0, curve=0.0):
    t = np.arange(4, dtype=float) - 1.5
    centers = np.stack([curve * t ** 2, 50 * t], axis=1)
    a = np.deg2rad(theta)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    centers = centers @ rot.T + [100, 100]
    points = list(centers)
    # Горизонтальные пластинки поворачиваются вместе с кадром.
    horizontal = np.array([20 * np.cos(a), -20 * np.sin(a)])
    for center in centers:
        points.extend([center + horizontal, center - horizontal])
    return np.asarray(points)


def test_rotated_straight_spine_has_same_center_and_endplate_angle():
    points = _points(theta=6)
    src = tilt_sources(points, np.ones(len(points), bool), NAMES)
    assert abs(src["theta_centers_deg"] - 6) < .05
    assert abs(src["theta_endplates_deg"] - 6) < .05


def test_curve_downweights_centers():
    points = _points(curve=8)
    visible = np.ones(len(points), bool)
    src = tilt_sources(points, visible, NAMES)
    curve = curve_descriptors(points, visible, NAMES)
    sig = source_variances(src, curve)
    assert sig["theta_centers_deg"] > 1.25
    fused = fused_tilt(src, sig)
    assert abs(fused["theta_fused_deg"]) < abs(src["theta_centers_deg"]) + 1e-6


def test_missing_edges_increases_uncertainty():
    points = _points(theta=3)
    full = np.ones(len(points), bool)
    sparse = full.copy()
    sparse[4:] = False
    src_full = tilt_sources(points, full, NAMES)
    src_sparse = tilt_sources(points, sparse, NAMES)
    curve = curve_descriptors(points, full, NAMES)
    a = fused_tilt(src_full, source_variances(src_full, curve))
    b = fused_tilt(src_sparse, source_variances(src_sparse, curve))
    assert b["n_sources"] < a["n_sources"]
    assert b["sigma_deg"] > a["sigma_deg"]


def test_c_curve_center_bias_matches_formula():
    t = np.array([-1.5, -.5, .5, 1.5])
    kappa_l, vertex, step = .1, 1.0, 50.0
    x = -.5 * kappa_l * step * (t - vertex) ** 2
    points = np.stack([x, t * step], axis=1)
    measured = _angle_vertical(points)
    expected = np.degrees(np.arctan(kappa_l * vertex))
    assert abs(measured - expected) < .1
