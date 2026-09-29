"""Аугментации, согласованные с координатами точек.

Эластичные деформации и cutout не используются: первые ломают геометрию, которую мы
потом измеряем, второй может стереть ориентир, помеченный в разметке как видимый.

Каждая геометрическая операция возвращает и маску кадра: какие пиксели — снимок,
а какие — заполнение. Точка, уехавшая в заполнение, должна стать невидимой. Раньше
видимость проверялась по холсту 352×320, и точка, выехавшая за правый или нижний
край снимка (у бедра холст на 40 px шире кадра), оставалась «видимой» в чёрном поле
— модель учили ставить точки там, где кости нет.

Обрезание края (`p_crop`) — главный источник примеров «структуры нет в кадре»:
в реальной выборке их единицы (отсутствующих гребней подвздошных костей 11,
седалищной кости 2), а почти все критерии ТЗ сводятся именно к вопросу «попал ли
ориентир в поле сканирования». Часть обрезаний проходит вплотную к размеченной
точке, по обе стороны от неё: так модель учится различать «у самой кромки» и
«уже за ней», а не только очевидные случаи.
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
    p_crop: float = 0.0          # обрезать один край кадра (0 — как в прежних весах)
    crop_max_frac: float = 0.35  # не больше этой доли высоты/ширины
    crop_targeted: float = 0.6   # доля обрезаний, проходящих рядом с размеченной точкой
    crop_jitter_px: float = 10.0 # разброс кромки вокруг точки, по обе стороны
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
            shift: tuple[float, float],
            valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Поворот, масштаб, сдвиг. Возвращает (кадр, точки, маска кадра после warp)."""
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
    src_valid = np.ones((h, w), np.float32) if valid is None else valid.astype(np.float32)
    new_valid = affine_transform(src_valid, m, offset=offset, order=0,
                                 mode="constant", cval=0.0) > 0.5
    return warped, coords @ R.T + t, new_valid


def crop_edge(image: np.ndarray, valid: np.ndarray, coords: np.ndarray,
              rng: np.random.Generator, cfg: AugmentConfig,
              targetable: np.ndarray | None = None):
    """Отрезать полосу у одного края кадра: кадр действительно становится меньше.

    Отрезается сам массив, а не зануляется полоса: у настоящего снимка за краем
    поля сканирования нет ничего, и после дополнения до холста нулями обрезанный
    кадр выглядит ровно так же. Верх и лево сдвигают координаты.
    Возвращает (кадр, маска, точки) либо исходные, если обрезать нечего.
    """
    h, w = image.shape
    edge = int(rng.integers(4))                          # 0 верх, 1 низ, 2 лево, 3 право
    size = h if edge < 2 else w
    k_max = int(cfg.crop_max_frac * size)
    if k_max < 2:
        return image, valid, coords
    k = None
    cand = np.flatnonzero(targetable) if targetable is not None else np.array([], int)
    if cand.size and rng.random() < cfg.crop_targeted:
        x, y = coords[int(rng.choice(cand))]
        pos = (y if edge < 2 else x) + rng.uniform(-cfg.crop_jitter_px, cfg.crop_jitter_px)
        k = int(round(pos)) if edge in (0, 2) else int(round(size - pos))
        if not 1 <= k <= k_max:
            k = None                                     # точка слишком глубоко — случайный срез
    if k is None:
        k = int(rng.integers(max(1, int(0.03 * size)), k_max + 1))
    coords = coords.copy()
    if edge == 0:
        image, valid = image[k:], valid[k:]
        coords[:, 1] -= k
    elif edge == 1:
        image, valid = image[:h - k], valid[:h - k]
    elif edge == 2:
        image, valid = image[:, k:], valid[:, k:]
        coords[:, 0] -= k
    else:
        image, valid = image[:, :w - k], valid[:, :w - k]
    return np.ascontiguousarray(image), np.ascontiguousarray(valid), coords


def inside_frame(coords: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Точка видима, только если попадает в пиксель снимка, а не в заполнение."""
    h, w = valid.shape
    xi, yi = np.round(coords[:, 0]).astype(int), np.round(coords[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(len(coords), bool)
    out[ok] = valid[yi[ok], xi[ok]]
    return out


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
            rng: np.random.Generator, flip_perm: list[int] | None = None,
            targetable: np.ndarray | None = None):
    """Возвращает (image, coords, channel_permutation, valid).

    valid — маска пикселей снимка того же размера, что image: по ней решается,
    осталась ли точка в кадре. targetable — каналы с размеченной координатой,
    рядом с которыми можно проводить прицельное обрезание.
    """
    perm = None
    valid = np.ones(image.shape, bool)
    if not cfg.enabled:
        return image.astype(np.float32), coords, perm, valid

    if rng.random() < cfg.p_geometric:
        h, w = image.shape
        image, coords, valid = _affine(
            image, coords,
            angle=rng.uniform(-cfg.rotation_deg, cfg.rotation_deg),
            scale=1 + rng.uniform(-cfg.scale, cfg.scale),
            shift=(rng.uniform(-cfg.shift_frac, cfg.shift_frac) * w,
                   rng.uniform(-cfg.shift_frac, cfg.shift_frac) * h),
            valid=valid,
        )
    if cfg.p_crop > 0 and rng.random() < cfg.p_crop:
        image, valid, coords = crop_edge(image, valid, coords, rng, cfg, targetable)
    if region == "spine" and rng.random() < cfg.p_hflip_spine and flip_perm is not None:
        image = image[:, ::-1].copy()
        valid = valid[:, ::-1].copy()
        coords = coords.copy()
        coords[:, 0] = image.shape[1] - 1 - coords[:, 0]
        perm = flip_perm                      # право и лево меняются местами
    if rng.random() < cfg.p_intensity:
        image = _intensity(image, rng, cfg)
    image = np.where(valid, image, 0.0)       # шум и яркость не должны «зажечь» заполнение
    return image.astype(np.float32), coords, perm, valid
