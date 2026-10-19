"""Проверки преобразования координат и итеративного согласования центров."""
import numpy as np

from .crop_consensus import (Candidate, adaptive_boxes, fixed_boxes, merge_close,
                             predict_consensus, resolve_centers)
from .multicrop import CropBox, map_prediction


def c(y):
    return Candidate(np.array([0., float(y)]))


def test_merge_repeats_with_original_observation_weights():
    got = merge_close([c(20), c(30), c(32)], 10)
    assert len(got) == 1
    assert got[0].count == 3
    np.testing.assert_allclose(got[0].xy, [0, 82 / 3])
    assert len(merge_close([c(20), c(30), c(42)], 10)) == 2


def test_exclusion_uses_only_verified_anatomical_neighbors():
    result = resolve_centers({"l1_center": [c(20)], "l2_center": [c(24), c(55)],
                              "l3_center": [c(90), c(160)], "l4_center": [c(200)]},
                             mm_per_px=1)
    assert result["verified"]["l2_center"].xy[1] == 55
    assert len(result["clusters"]["l3_center"]) == 2


def test_exclusion_threshold_depends_on_neighbor_pair():
    # 10.6 мм: дальше порога L1–L2 (10.39), но ближе порога L3–L4 (11.28).
    result = resolve_centers({"l1_center": [c(0)], "l2_center": [c(10.6), c(40)]}, 1)
    assert len(result["clusters"]["l2_center"]) == 2
    result = resolve_centers({"l3_center": [c(29.4), c(-10)], "l4_center": [c(40)]}, 1)
    assert result["verified"]["l3_center"].xy[1] == -10
    assert result["removed_near_neighbor"]["l3_center"] == 1


def test_wrong_side_candidate_is_removed_and_verification_cascades():
    # L2: кандидат выше L1 удаляется -> L2 верифицирован -> он отсекает
    # кандидата L3 выше себя -> L3 верифицирован уже по новой опоре.
    result = resolve_centers({"l1_center": [c(100)], "l2_center": [c(60), c(130)],
                              "l3_center": [c(110), c(165)]}, 1)
    assert result["removed_wrong_side"]["l2_center"] == 1
    assert result["verified"]["l2_center"].xy[1] == 130
    assert result["removed_wrong_side"]["l3_center"] == 1
    assert result["verified"]["l3_center"].xy[1] == 165


def test_last_candidate_on_wrong_side_is_not_verified():
    result = resolve_centers({"l1_center": [c(0)], "l2_center": [c(-50), c(-80)],
                              "l3_center": [c(70)]}, 1)
    assert result["clusters"]["l2_center"] == []
    assert "l2_center" not in result["verified"]
    assert result["interpolated"]["l2_center"].xy[1] == 35


def test_only_isolated_internal_gap_is_interpolated():
    result = resolve_centers({"l1_center": [c(0)], "l2_center": [],
                              "l3_center": [c(80)], "l4_center": [c(120)]}, 1)
    assert result["interpolated"]["l2_center"].xy[1] == 40
    result = resolve_centers({"l1_center": [c(0)], "l2_center": [],
                              "l3_center": [], "l4_center": [c(120)]}, 1)
    assert result["interpolated"] == {}


def test_distances_are_in_millimetres():
    # 15 px = 9 mm при масштабе .6 мм/px.
    result = resolve_centers({"l1_center": [c(0), c(15)]}, .6)
    assert result["verified"]["l1_center"].count == 2
    result = resolve_centers({"l1_center": [c(0), c(18)]}, .6)
    assert "l1_center" not in result["verified"]


def test_fixed_crop_masks_and_global_offset():
    cfg = {"specs": {"l2_l3": {"fixed_band": [.19, .79]}}}
    box = fixed_boxes((300, 200), cfg)["l2_l3"]
    assert box == CropBox(0, 57, 200, 237)
    mapped = map_prediction({"names": ["l2_center"],
                             "coords": np.array([[35., 80.]]),
                             "visible": np.array([True]),
                             "confidence": np.array([.8])}, box, (300, 200), 10)
    np.testing.assert_allclose(mapped["coords"][0], [35, 137])


def test_adaptive_box_uses_median_doctor_window_and_falls_back():
    names = [f"l{i}_center" for i in range(1, 5)]
    cfg = {"min_crop_px": 50, "specs": {"l1_l2": {
        "fixed_band": [0, .59], "window_t": {"p50": [-.5, 1.5]}}}}
    pred = {"names": names, "coords": np.array([[50., 60.], [50., 110.],
                                                 [50., 160.], [50., 210.]]),
            "visible": np.ones(4, bool)}
    box = adaptive_boxes((300, 200), cfg, pred)["l1_l2"]
    assert box == CropBox(0, 35, 200, 135)
    pred["visible"][1] = False
    assert adaptive_boxes((300, 200), cfg, pred)["l1_l2"] == CropBox(0, 0, 200, 177)


def test_full_pipeline_masks_absent_channels_and_restores_global_coordinates():
    names = [f"l{i}_center" for i in range(1, 5)] + ["l1_edge_left"]
    bands = ((0, .44), (0, .59), (.19, .79), (.37, 1))
    expected = ((names[0],), names[:2], names[1:3], names[2:4])
    cfg = {"specs": {f"crop{k}": {"fixed_band": bands[k],
                                    "expected_centers": expected[k],
                                    "output": names}
                     for k in range(4)}, "edge_margin_px": 10}
    image = np.zeros((300, 200), np.uint8)
    boxes = list(fixed_boxes(image.shape, cfg).values())
    global_y = np.array([60., 110., 170., 230., 60.])

    class FakePredictor:
        cfg = {"data": {"mm_per_px": .6}}
        thresholds = np.full(5, .2)
        crop_thresholds = thresholds.copy()

        def __init__(self):
            self.calls = 0

        def predict(self, crop, chain=None):
            k = self.calls
            assert (chain is None) if k == 0 else (chain is False)
            self.calls += 1
            y = global_y.copy() - (0 if k == 0 else boxes[k - 1].y0)
            # Канал отсутствующего L4 на верхнем кропе даёт ложный пик.
            if k == 1:
                y[3] = 25
                y[4] = 100  # край кропа конфликтует с точкой полного кадра
            visible = np.ones(5, bool)
            confidence = np.full(5, .8)
            if k == 0:
                visible[3] = False  # всё равно даёт обязательный кандидат L4
            if k == 1:
                confidence[0] = .01  # слабый пик не исчезает из списка
            return {"names": names, "coords": np.column_stack((np.full(5, 100.), y)),
                    "visible": visible, "confidence": confidence,
                    "region": "spine"}

    predictor = FakePredictor()
    out = predict_consensus(predictor, image, cfg)
    assert predictor.calls == 5
    np.testing.assert_allclose(out["coords"][:, 1], global_y)
    assert [out["multicrop"]["per_point"][n]["n_sources"] for n in names[:4]] == [3, 3, 3, 2]
    assert out["multicrop"]["per_point"]["l1_edge_left"]["status"] == "full_frame"
    assert names[3] not in out["multicrop"]["sources"][0]["points"]
    assert out["visible"].all()
