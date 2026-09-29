"""Тесты защиты от выдуманных точек. Запуск: PYTHONPATH=src python -m dxa_qc.test_dxa_qc (или pytest).

1. Видимость после аугментации решается по пикселям снимка, а не по холсту:
   точка, уехавшая в заполнение, молчит.
2. Обрезание края действительно уменьшает кадр и сдвигает координаты.
3. Декод не ищет пик в заполнении до холста.
4. Порог «balanced» не уезжает на нижний край, когда отсутствующих примеров мало
   по сравнению с видимыми, — в отличие от прежнего F1; а когда их нет совсем,
   теряет не больше 2 % видимых.
5. Структурная расшифровка возвращает центр, прыгнувший на соседний позвонок.
"""
from __future__ import annotations

import numpy as np
import torch

from .heatmaps import decode
from .metrics import calibrate_thresholds
from .structure import chain_decode
from .transforms import AugmentConfig, _affine, crop_edge, inside_frame


def test_visibility_follows_frame_not_canvas():
    image = np.full((100, 80), 100.0)
    coords = np.array([[75.0, 50.0], [40.0, 50.0]])
    # сдвиг вправо на 20 px: первая точка уходит за правый край снимка,
    # но остаётся внутри холста 352×320 — раньше она считалась видимой
    _, moved, valid = _affine(image, coords, angle=0.0, scale=1.0, shift=(20.0, 0.0))
    assert moved[0, 0] > 80
    assert list(inside_frame(moved, valid)) == [False, True]


def test_crop_edge_really_crops():
    image = np.arange(100 * 80, dtype=float).reshape(100, 80)
    coords = np.array([[10.0, 5.0], [40.0, 60.0]])
    cfg = AugmentConfig(crop_max_frac=0.35, crop_targeted=1.0, crop_jitter_px=0.0)
    seen = set()
    for seed in range(40):
        rng = np.random.default_rng(seed)
        img, valid, c = crop_edge(image, np.ones(image.shape, bool), coords, rng, cfg,
                                  targetable=np.array([True, True]))
        assert img.shape == valid.shape and img.size < image.size
        # пиксель под точкой после обрезания — тот же пиксель исходного снимка
        vis = inside_frame(c, valid)
        for (x, y), (x0, y0), v in zip(c, coords, vis):
            if v:
                assert img[int(round(y)), int(round(x))] == image[int(y0), int(x0)]
        seen.add(img.shape)
    assert len(seen) > 5


def test_decode_ignores_padding():
    h = torch.zeros(1, 1, 64, 64)
    h[0, 0, 60, 60] = 0.9           # шум в заполнении
    h[0, 0, 20, 30] = 0.4           # настоящий пик внутри кадра 50×50
    coords, conf = decode(h, 5, hw=(50, 50))
    assert abs(conf[0, 0].item() - 0.4) < 1e-6
    assert torch.allclose(coords[0, 0], torch.tensor([30.0, 20.0]))


def test_balanced_threshold_says_absent():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.02, 1.0, 400)            # видимые: уверенность размазана до нуля
    neg = rng.uniform(0.0, 0.5, 20)              # отсутствующие: модель их слегка «видит»
    conf = np.r_[pos, neg][:, None]
    visible = np.r_[np.ones(400, bool), np.zeros(20, bool)][:, None]
    labeled = np.ones_like(visible)
    f1 = calibrate_thresholds(["p"], conf, visible, labeled, objective="f1")["p"]
    bal = calibrate_thresholds(["p"], conf, visible, labeled, objective="balanced")["p"]
    assert f1 < 0.1 < 0.25 < bal
    # отсутствующих меньше min_negatives: порог теряет не больше 2 % видимых,
    # а не фиксированные 0.5 (те теряли бы половину)
    few = calibrate_thresholds(["p"], conf, visible, labeled,
                               objective="balanced", min_negatives=50)["p"]
    assert abs((pos < few).mean() - 0.02) < 0.01


def _blob(h, x, y, v, s=2.0):
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    h[:] = np.maximum(h, v * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)))


def test_chain_decode_fixes_level_jump():
    names = ["th12_top", "th12_bottom", "l1_center", "l2_center", "l3_center", "l4_center",
             "disc_l1_l2"]
    ys = {0: 40, 1: 100, 2: 160, 3: 220, 4: 280}          # Th12, L1..L4, шаг 60 px
    prob = np.zeros((len(names), 320, 120))
    for k in range(4):
        _blob(prob[2 + k], 60, ys[k + 1], 0.8)             # верный уровень
    _blob(prob[2], 60, ys[0], 0.9)                         # L1 сильнее светится на Th12
    _blob(prob[6], 60, 130, 0.7)                           # диск L1/L2
    _blob(prob[6], 60, 70, 0.8)                            # и ложный — над L1
    coords = np.zeros((len(names), 2))
    for c in range(len(names)):
        y, x = np.unravel_index(prob[c].argmax(), prob[c].shape)
        coords[c] = (x, y)
    assert coords[2, 1] == ys[0]                           # независимый декод ошибся
    active = np.array([False, False, True, True, True, True, True])
    fixed = chain_decode(prob, names, coords, active)
    assert abs(fixed[2, 1] - ys[1]) < 2
    assert abs(fixed[6, 1] - 130) < 2                      # диск вернулся между L1 и L2
    assert np.allclose(fixed[3:6], coords[3:6], atol=1.0)  # верные центры не тронуты


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
