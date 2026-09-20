"""Аугментации, согласованные с координатами точек.

Эластичные деформации и cutout не используются: первые ломают геометрию, которую мы
потом измеряем, второй может стереть ориентир, помеченный в разметке как видимый.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter


@dataclass
class AugmentConfig:
    rotation_deg: float = 15.0
    shift_frac: float = 0.10
    scale: float = 0.10
    brightness: float = 0.20
    contrast: float = 0.20
    gamma: tuple[float, float] = (0.8, 1.25)
    noise_std: float = 4.0
    blur_sigma: float = 0.6
    p_geometric: float = 0.9
    p_intensity: float = 0.7
    p_hflip_spine: float = 0.5   # бедро не отражаем: оно уже приведено к левому
    enabled: bool = True


def clahe(image: np.ndarray, tiles: int = 8, clip: float = 2.0) -> np.ndarray:
    """Локальное выравнивание гистограммы. Проверяется аблацией, а не берётся на веру."""
    h, w = image.shape
    th, tw = max(1, h // tiles), max(1, w // tiles)
    maps = np.zeros((tiles, tiles, 256), dtype=np.float32)
    for i in range(tiles):
        for j in range(tiles):
            tile = image[i * th:(i + 1) * th if i < tiles - 1 else h,
                         j * tw:(j + 1) * tw if j < tiles - 1 else w]
            hist = np.bincount(tile.astype(np.uint8).ravel(), minlength=256).astype(np.float32)
            limit = clip * hist.sum() / 256
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256
            maps[i, j] = 255 * np.cumsum(hist) / max(hist.sum(), 1)
    yi = np.clip((np.arange(h) / th - 0.5), 0, tiles - 1)
    xi = np.clip((np.arange(w) / tw - 0.5), 0, tiles - 1)
    y0, x0 = yi.astype(int), xi.astype(int)
    y1, x1 = np.minimum(y0 + 1, tiles - 1), np.minimum(x0 + 1, tiles - 1)
    wy, wx = (yi - y0)[:, None], (xi - x0)[None, :]
    v = image.astype(np.uint8)
    lut = lambda a, b: maps[a[:, None], b[None, :], v]        # билинейно между тайлами
    out = ((1 - wy) * ((1 - wx) * lut(y0, x0) + wx * lut(y0, x1))
           + wy * ((1 - wx) * lut(y1, x0) + wx * lut(y1, x1)))
    return out.astype(np.float32)


def _affine(image: np.ndarray, coords: np.ndarray, angle: float, scale: float,
            shift: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    c = np.array([w / 2, h / 2])
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scale
    t = c + np.array(shift) - R @ c                      # x' = R·x + t, в порядке (x, y)

    Rinv = np.linalg.inv(R)
    m = Rinv[::-1, ::-1]                                 # scipy работает в порядке (y, x)
    offset = (Rinv @ -t)[::-1]
    warped = affine_transform(image.astype(np.float32), m, offset=offset,
                              order=1, mode="constant", cval=0.0)
    return warped, coords @ R.T + t


def _intensity(image: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    img = image.astype(np.float32)
    img = (img - img.mean()) * (1 + rng.uniform(-cfg.contrast, cfg.contrast)) + img.mean()
    img = img + rng.uniform(-cfg.brightness, cfg.brightness) * 255
    img = np.clip(img, 0, 255)
    g = rng.uniform(*cfg.gamma)
    img = 255.0 * (img / 255.0) ** g
    if cfg.blur_sigma > 0 and rng.random() < 0.3:
        img = gaussian_filter(img, rng.uniform(0, cfg.blur_sigma))
    if cfg.noise_std > 0:
        img = img + rng.normal(0, cfg.noise_std, img.shape)
    return np.clip(img, 0, 255)


def augment(image: np.ndarray, coords: np.ndarray, region: str, cfg: AugmentConfig,
            rng: np.random.Generator, flip_perm: list[int] | None = None):
    """Возвращает (image, coords, channel_permutation)."""
    perm = None
    if not cfg.enabled:
        return image.astype(np.float32), coords, perm

    if rng.random() < cfg.p_geometric:
        h, w = image.shape
        image, coords = _affine(
            image, coords,
            angle=rng.uniform(-cfg.rotation_deg, cfg.rotation_deg),
            scale=1 + rng.uniform(-cfg.scale, cfg.scale),
            shift=(rng.uniform(-cfg.shift_frac, cfg.shift_frac) * w,
                   rng.uniform(-cfg.shift_frac, cfg.shift_frac) * h),
        )
    if region == "spine" and rng.random() < cfg.p_hflip_spine and flip_perm is not None:
        image = image[:, ::-1].copy()
        coords = coords.copy()
        coords[:, 0] = image.shape[1] - 1 - coords[:, 0]
        perm = flip_perm                      # право и лево меняются местами
    if rng.random() < cfg.p_intensity:
        image = _intensity(image, rng, cfg)
    return image.astype(np.float32), coords, perm
