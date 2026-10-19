"""Проверки методики G1 без загрузки нейросети."""
from __future__ import annotations

import numpy as np

from tools.report_semantic_gates import compare, metrics, pair_oof


def _sample(abstain: bool) -> dict:
    names = (["th12_top", "th12_bottom"] + [f"l{i}_center" for i in range(1, 5)]
             + [n for i in range(1, 5)
                for n in (f"l{i}_edge_left", f"l{i}_edge_right")])
    true = np.zeros((1, len(names), 2), float)
    true[0, 0:2] = [[50, 0], [50, 25]]
    for k in range(4):
        y = 50 + k * 50
        true[0, 2 + k] = [50, y]
        true[0, 6 + 2 * k] = [70, y]
        true[0, 7 + 2 * k] = [30, y]
    pred = true.copy()
    pred[0, 2:6, 1] += 50  # все четыре имени сдвинуты на позвонок
    conf = np.ones((1, len(names)))
    conf[0, 6:8] = .1  # одна пара краёв предсказана невидимой
    return {
        "names": np.asarray(names), "ids": np.asarray(["image"]),
        "studies": np.asarray(["study"]), "regions": np.asarray(["spine"]),
        "folds": np.asarray([0]), "pred": pred, "true": true,
        "visible": np.ones((1, len(names)), bool), "conf": conf,
        "thresholds": np.full((1, len(names)), .5),
        "abstain": np.asarray([abstain]),
    }


def test_abstention_does_not_hide_raw_level_error():
    result, _ = metrics(_sample(True))
    assert result["level_error_frames_raw"] == 1
    assert result["level_error_frames_accepted"] == 0
    assert result["three_edge_pairs_rate"] == 1.0


def test_pairing_rejects_different_frames():
    baseline, other = _sample(False), _sample(False)
    other["ids"] = np.asarray(["another"])
    try:
        pair_oof(baseline, other)
    except ValueError:
        return
    raise AssertionError("разные OOF-кадры были приняты")


def test_raw_shift_blocks_gate_even_after_abstention():
    def repeat(z):
        out = {key: np.repeat(value, 9, axis=0)
               if key != "names" and value.ndim and len(value) == 1 else value
               for key, value in z.items()}
        out["ids"] = np.asarray([f"image-{i}" for i in range(9)])
        return out
    data = repeat(_sample(True))
    baseline = repeat(_sample(False))
    manifest = {"multicrop": "off", "pack_sha256": "x",
                "checkpoint_sha256": {"0": "x"}, "folds": [0],
                "threshold_protocol": "crossfit", "crops_per_image": 6}
    report, _ = compare({
        "full_frame": (baseline, manifest),
        "semantic_oracle": (data, {**manifest, "multicrop": "oracle"}),
        "semantic_auto": (data, {**manifest, "multicrop": "auto",
                                 "anchor_thresholds_sha256": "x"}),
    })
    assert not report[-1]["checks"]["levels_raw"]
    assert not report[-1]["gate_passed"]


def test_four_fold_result_cannot_enable_production():
    data = _sample(False)
    data["pred"] = data["true"].copy()
    data["conf"][:] = 1
    manifest = {"multicrop": "off", "pack_sha256": "x",
                "checkpoint_sha256": {"0": "x"}, "folds": [0],
                "threshold_protocol": "crossfit", "crops_per_image": 6}
    report, _ = compare({
        "full_frame": (data, manifest),
        "semantic_oracle": (data, {**manifest, "multicrop": "oracle"}),
        "semantic_auto": (data, {**manifest, "multicrop": "auto",
                                 "anchor_thresholds_sha256": "x"}),
    })
    assert not report[-1]["checks"]["five_fold_oof"]
    assert not report[-1]["gate_passed"]
