"""Быстрые тесты чистой геометрии multi-crop (без загрузки CNN)."""
from __future__ import annotations

import numpy as np

from .multicrop import (CropBox, anchor_quality, chain_conflicts, decode_allowed,
                        map_prediction, merge_predictions, spine_crop_boxes)


NAMES = [f"l{i}_center" for i in range(1, 5)]


def _prediction(ys, confidence=0.8):
    return {
        "names": NAMES,
        "coords": np.array([[100.0, y] for y in ys]),
        "confidence": np.full(4, confidence),
        "visible": np.ones(4, dtype=bool),
        "region": "spine",
    }


def test_spine_crop_boxes_cover_expected_bands():
    boxes = spine_crop_boxes(300, 200)
    assert [(b.y0, b.y1) for b in boxes] == [
        (0, 114), (39, 150), (81, 207), (135, 300), (0, 300)
    ]
    assert boxes[-1].full_frame


def test_merge_averages_consistent_sources():
    a = _prediction([50, 100, 150, 200])
    b = _prediction([52, 102, 152, 202])
    merged = merge_predictions([a, b], np.full(4, 0.2), disagree_px=10)
    np.testing.assert_allclose(merged["coords"][:, 1], [51, 101, 151, 201])
    assert merged["visible"].all()
    assert not merged["multicrop"]["abstain"]
    assert merged["multicrop"]["per_point"]["l2_center"]["n_used"] == 2


def test_crop_coordinates_are_mapped_to_full_frame():
    local = _prediction([10, 20, 30, 40])
    local["coords"][:, 0] = 15
    mapped = map_prediction(
        local, CropBox(5, 50, 105, 150), image_shape=(200, 120),
        edge_margin_px=0,
    )
    np.testing.assert_allclose(mapped["coords"][:, 0], 20)
    np.testing.assert_allclose(mapped["coords"][:, 1], [60, 70, 80, 90])


def test_shifted_chain_forces_abstain():
    a = _prediction([50, 100, 150, 200])
    # Метка L1 во втором источнике попала на физический L2 первого и т.д.
    b = _prediction([100, 150, 200, 250])
    conflicts = chain_conflicts([a, b], disagree_px=10)
    assert conflicts
    assert abs(conflicts[0]["best_shift"]) == 1
    merged = merge_predictions([a, b], np.full(4, 0.2), disagree_px=10)
    assert merged["multicrop"]["abstain"]


def test_forbidden_channel_is_never_decoded():
    heat = np.zeros((2, 40, 40), dtype=float)
    heat[0, 20, 20] = .8
    heat[1, 20, 20] = 1.0
    out = decode_allowed(
        heat, ["allowed", "forbidden"], {"allowed"},
        {"allowed": (np.array([20, 20]), 8), "forbidden": (np.array([20, 20]), 8)},
    )
    assert out["confidence"][0] == .8
    assert out["confidence"][1] == 0
    assert not out["in_prior"][1]


def test_prior_does_not_jump_to_neighbouring_level():
    heat = np.zeros((1, 80, 30), dtype=float)
    heat[0, 20, 15] = .7
    heat[0, 50, 15] = 1.0  # более сильный соседний позвонок
    out = decode_allowed(
        heat, ["l2_center"], {"l2_center"},
        {"l2_center": (np.array([15, 20]), 10)},
    )
    assert abs(out["coords"][0, 1] - 20) < 1


def test_unreliable_anchor_reports_numbering_doubt():
    pred = _prediction([50, 100, 150, 200])
    pred["visible"][2] = False
    pred["chain_hypotheses"] = {"used": True, "score_gap_top2": 2.0}
    q = anchor_quality(pred, {"anchor": {"step_ratio_max": 1.6, "chain_gap_min": 0}})
    assert not q["reliable"]
    assert "centers_missing" in q["reasons"]
