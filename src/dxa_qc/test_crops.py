"""Проверка совместимости врачебного JSON с pack, включая правое бедро."""
from __future__ import annotations

import json

import numpy as np

from .crops import load
from .pack import ABSENT, LABELED, OTHER, UNLABELED


def test_crop_inherits_parent_and_mirrors_right_hip(tmp_path):
    pixels = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    source = {
        "names": ["trochanter_major", "trochanter_minor", "neck_center", "l1_center"],
        "images": [pixels],
        # В pack правое бедро уже отражено. Исходные x=1 и x=3 стали x=6, x=4.
        "coords": np.array([[[6., 2.], [4., 3.], [0., 0.], [0., 0.]]]),
        "state": np.array([[LABELED, LABELED, UNLABELED, OTHER]], dtype=np.uint8),
        "ids": np.array(["image-1"]),
        "studies": np.array(["study-7"]),
        "regions": np.array(["hip"]),
        "sides": np.array(["rhip"]),
    }
    export = {
        "format": "dxa-crops-v1", "schema_version": 1,
        "crops": [
            {"id": "confirmed", "image_id": "image-1", "region": "hip",
             "status": "confirmed", "image_size": {"w": 8, "h": 6},
             "box": {"x0": 0, "y0": 1, "x1": 4, "y1": 5},
             "included_points": ["trochanter_major"],
             "points_crop": {"trochanter_major": {"x": 1., "y": 1.}}},
            {"id": "draft", "status": "draft"},
        ],
    }
    path = tmp_path / "crops.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    result = load(path, source)
    assert result["ids"].tolist() == ["confirmed"]
    assert result["parent_ids"].tolist() == ["image-1"]
    assert result["studies"].tolist() == ["study-7"]
    assert np.array_equal(result["images"][0], pixels[1:5, 4:8])
    assert np.array_equal(result["coords"][0, 0], [2., 1.])
    assert result["state"][0].tolist() == [LABELED, UNLABELED, UNLABELED, OTHER]


def test_selected_point_must_be_inside_crop(tmp_path):
    source = {
        "names": ["p"], "images": [np.zeros((10, 10), np.uint8)],
        "coords": np.array([[[8., 8.]]]), "state": np.array([[LABELED]], np.uint8),
        "ids": np.array(["id"]), "studies": np.array(["study"]),
        "regions": np.array(["spine"]), "sides": np.array(["spine"]),
    }
    export = {"format": "dxa-crops-v1", "schema_version": 1,
              "crops": [{"id": "bad", "image_id": "id", "region": "spine",
                         "status": "confirmed", "image_size": {"w": 10, "h": 10},
                         "box": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
                         "included_points": ["p"], "points_crop": {"p": {"x": 8, "y": 8}}}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    try:
        load(path, source)
    except ValueError as e:
        assert "вне рамки" in str(e)
    else:
        raise AssertionError("Кроп с точкой за рамкой был принят")


def test_unselected_point_inside_is_unlabeled_not_absent(tmp_path):
    source = {
        "names": ["a", "b", "c"],
        "images": [np.zeros((12, 12), np.uint8)],
        "coords": np.array([[[3., 3.], [5., 5.], [10., 10.]]]),
        "state": np.array([[LABELED, LABELED, LABELED]], np.uint8),
        "ids": np.array(["id"]), "studies": np.array(["study"]),
        "regions": np.array(["spine"]), "sides": np.array(["spine"]),
    }
    export = {"format": "dxa-crops-v1", "schema_version": 1,
              "crops": [{"id": "one", "image_id": "id", "region": "spine",
                         "status": "confirmed", "image_size": {"w": 12, "h": 12},
                         "box": {"x0": 0, "y0": 0, "x1": 8, "y1": 8},
                         "included_points": ["a"],
                         "points_crop": {"a": {"x": 3., "y": 3.}}}]}
    path = tmp_path / "crops.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    result = load(path, source)
    assert result["state"][0].tolist() == [LABELED, UNLABELED, ABSENT]
