"""Проверки согласования уровней по нескольким пикам."""
from __future__ import annotations

import numpy as np

from .points import HIP, SPINE_CORE, SPINE_EDGES
from .structure import chain_decode_hypotheses


def _blob(h: np.ndarray, x: int, y: int, value: float) -> None:
    yy, xx = np.ogrid[:h.shape[0], :h.shape[1]]
    h[:] = np.maximum(h, value * np.exp(-((xx - x)**2 + (yy - y)**2) / 8))


def test_disc_evidence_resolves_whole_vertebra_shift():
    names = SPINE_CORE + SPINE_EDGES
    h = np.zeros((21, 352, 120), dtype=np.float32)
    for k in range(4):
        _blob(h[2 + k], 60, 100 + 60*k, .8)  # правильная цепочка
        _blob(h[2 + k], 60, 40 + 60*k, .9)   # более сильный сдвиг на уровень
    for k, name in enumerate(("disc_th12_l1", "disc_l1_l2", "disc_l2_l3",
                              "disc_l3_l4", "disc_l4_l5")):
        _blob(h[names.index(name)], 60, 70 + 60*k, .8)
    coords = np.zeros((21, 2), dtype=float)
    for c in range(21):
        y, x = np.unravel_index(h[c].argmax(), h[c].shape)
        coords[c] = x, y
    assert np.all(coords[2:6, 1] == [40, 100, 160, 220])
    fixed, info = chain_decode_hypotheses(h, names, coords,
                                           np.ones(21, dtype=bool))
    assert info["used"] and info["n_hypotheses"] >= 2
    assert 0 < info["top1_relative_weight"] <= 1
    assert np.allclose(fixed[2:6, 1], [100, 160, 220, 280], atol=2)


def test_missing_center_is_not_hallucinated():
    names = SPINE_CORE + SPINE_EDGES
    h = np.zeros((21, 352, 120), dtype=np.float32)
    coords = np.zeros((21, 2), dtype=float)
    active = np.zeros(21, dtype=bool)
    active[[2, 3, 4]] = True  # L4 вне обрезанного кадра
    for k in range(3):
        _blob(h[2 + k], 60, 100 + 60*k, .8)
        coords[2 + k] = 60, 100 + 60*k
    coords[5] = 13, 14
    fixed, info = chain_decode_hypotheses(h, names, coords, active)
    assert not info["used"] and info["reason"] == "partial_visibility"
    assert np.array_equal(fixed[5], coords[5])


def test_core_model_uses_legacy_decoder():
    # У core-модели 21 канал всего, из них восемь принадлежат бедру.
    names = SPINE_CORE + HIP
    h = np.zeros((21, 352, 120), dtype=np.float32)
    coords = np.zeros((21, 2), dtype=float)
    for k in range(4):
        _blob(h[2 + k], 60, 80 + 60*k, .8)
        coords[2 + k] = 60, 80 + 60*k
    _, info = chain_decode_hypotheses(h, names, coords,
                                       np.ones(21, dtype=bool))
    assert not info["used"] and info["reason"] == "no_chain"
