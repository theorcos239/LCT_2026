"""Torch-датасет: холст, целевые карты и веса каналов.

Дополнение нулями вместо ресайза — масштаб 0.6 мм/px должен сохраниться, иначе
расстояния в сантиметрах посчитать нельзя.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import points as P
from .heatmaps import render
from .pack import ABSENT, LABELED, OTHER, UNLABELED
from .transforms import AugmentConfig, augment, clahe

MEAN, STD = 0.449, 0.226      # серый эквивалент нормировки ImageNet


def pad_to(image: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    ch, cw = canvas
    if h > ch or w > cw:
        raise ValueError(f"снимок {h}×{w} больше холста {ch}×{cw}")
    out = np.zeros((ch, cw), dtype=np.float32)
    out[:h, :w] = image
    return out


def pad_to_multiple(image: np.ndarray, k: int = 32) -> np.ndarray:
    h, w = image.shape
    return pad_to(image, (int(np.ceil(h / k)) * k, int(np.ceil(w / k)) * k))


def normalize(image: np.ndarray) -> np.ndarray:
    return (image / 255.0 - MEAN) / STD


class KeypointDataset(Dataset):
    def __init__(self, pack: dict, indices, canvas=(352, 320), sigma=2.0,
                 soft_mask_weight=0.15, use_clahe=False,
                 aug: AugmentConfig | None = None, seed: int = 0):
        self.pack, self.indices = pack, list(indices)
        self.canvas, self.sigma = tuple(canvas), sigma
        self.soft_mask_weight, self.use_clahe = soft_mask_weight, use_clahe
        self.aug = aug or AugmentConfig(enabled=False)
        self.flip_perm = P.flip_permutation(pack["names"])
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        rng = np.random.default_rng((self.seed, self.epoch, j))
        image = self.pack["images"][j].astype(np.float32)
        coords = self.pack["coords"][j].copy()
        state = self.pack["state"][j].copy()
        region = str(self.pack["regions"][j])

        if self.use_clahe:
            image = clahe(image)
        image, coords, perm = augment(image, coords, region, self.aug, rng, self.flip_perm)
        if perm is not None:
            coords, state = coords[perm], state[perm]

        image = pad_to(image, self.canvas)
        h, w = self.canvas
        inside = (coords[:, 0] >= 0) & (coords[:, 0] < w) & (coords[:, 1] >= 0) & (coords[:, 1] < h)
        visible = (state == LABELED) & inside          # уехала за холст — учим молчанию
        target = render(coords, visible, self.canvas, self.sigma)

        weight = np.where(state == UNLABELED, 0.0,
                          np.where(state == OTHER, self.soft_mask_weight, 1.0)).astype(np.float32)
        return {
            "image": torch.from_numpy(normalize(image))[None],
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
            "coords": torch.from_numpy(coords),
            "visible": torch.from_numpy(visible),
            "labeled": torch.from_numpy((state == LABELED) | (state == ABSENT)),
            "index": j,
        }
