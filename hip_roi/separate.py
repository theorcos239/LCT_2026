# -*- coding: utf-8 -*-
"""Комплект 3: отделение бедра от таза и система координат диафиза.

Два шага перед поиском ориентиров:
1. separate_femur  — водораздел по дистанционному преобразованию с двумя
   маркерами (низ диафиза / верх таза). Граница ложится по самому узкому месту
   соединения — по шейке; головка при этом отходит к тазу, что для ориентиров
   T/B/L не мешает (головка в ROI не входит).
2. shaft_axis + rotate — ось диафиза по PCA нижней части бедра, поворот маски
   так, чтобы ось стала вертикальной. Ориентиры ищутся в повёрнутой системе и
   отображаются обратно в исходную (отступы всегда считаются к краям исходного
   кадра).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .geometry import component_containing


def minmax_flood(cost: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Водораздел по минимаксному пути (IFT с функцией стоимости f_max).

    Пиксель получает метку того семени, до которого путь с наименьшей
    максимальной стоимостью. Реализовано корзинной очередью по 256 уровням
    (cost — uint8), обход 8-связный, порядок внутри уровня — FIFO, поэтому
    результат детерминирован. scipy.ndimage.watershed_ift для семян не в
    локальных минимумах даёт неверный результат (проверено на игрушечной
    фигуре «два блоба + перемычка»), поэтому своя реализация.
    """
    from collections import deque
    h, w = cost.shape
    labels = np.zeros((h, w), dtype=np.int32)
    best = np.full((h, w), 256, dtype=np.int16)
    buckets = [deque() for _ in range(256)]
    ys, xs = np.nonzero(markers)
    for y, x in zip(ys.tolist(), xs.tolist()):
        best[y, x] = 0                      # семя — источник: его собственная толщина не в счёт
        buckets[0].append((y, x, int(markers[y, x])))
    m = mask
    for level in range(256):
        q = buckets[level]
        while q:
            y, x, lab = q.popleft()
            if labels[y, x]:
                continue
            labels[y, x] = lab
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if nx < 0 or nx >= w or labels[ny, nx] or not m[ny, nx]:
                        continue
                    nc = int(cost[ny, nx])
                    if nc < level:
                        nc = level
                    if nc < best[ny, nx]:
                        best[ny, nx] = nc
                        buckets[nc].append((ny, nx, lab))
    return labels


def _argmax_in(D: np.ndarray, region: np.ndarray):
    """Пиксель с максимумом D внутри булевой области (None, если область пуста)."""
    if not region.any():
        return None
    Dm = np.where(region, D, -1.0)
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(Dm)), D.shape))


def separate_femur(mask: np.ndarray, min_frac: float = 0.15, bottom_frac: float = 0.45,
                   top_frac: float = 0.30):
    """Маска бедра (без головки) + info. При неудаче — вся маска и флаг.

    Семена ставятся в максимумы дистанционного преобразования (самые «толстые»
    места): бедро — в латеральной половине нижних (1 - bottom_frac) строк
    (вертельная масса или диафиз — оба соединены с шейкой толстой костью);
    таз — в медиальной половине верхних top_frac строк. Граница минимаксной
    заливки ложится по самому тонкому месту между семенами — по шейке.
    """
    info = {'flags': []}
    if not mask.any():
        return mask.copy(), {'flags': ['sep:empty_mask']}
    h, w = mask.shape
    D = ndi.distance_transform_edt(mask)
    yy, xx = np.mgrid[0:h, 0:w]
    fem_region = mask & (yy >= int(bottom_frac * h)) & (xx < w // 2)
    femur_seed = _argmax_in(D, fem_region)
    pel_region = mask & (yy < max(1, int(top_frac * h))) & (xx >= w // 2)
    pelvis_seed = _argmax_in(D, pel_region)
    if femur_seed is None or pelvis_seed is None:
        info['flags'].append('sep:no_seed')
        return mask.copy(), info
    info.update(femur_seed=femur_seed, pelvis_seed=pelvis_seed)
    if femur_seed == pelvis_seed:
        info['flags'].append('sep:seeds_coincide')
        return mask.copy(), info

    cost = np.round(254.0 * (1.0 - D / D.max())).astype(np.uint8)
    cost[~mask] = 255
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[femur_seed] = 1
    markers[pelvis_seed] = 2
    out = minmax_flood(cost, markers, mask)
    femur = (out == 1) & mask
    if femur[femur_seed] and not femur[pelvis_seed]:
        femur = component_containing(femur, femur_seed)
    if femur.sum() < min_frac * mask.sum() or femur[pelvis_seed]:
        info['flags'].append('sep:failed')
        return mask.copy(), info
    info['femur_frac'] = round(float(femur.sum() / mask.sum()), 3)
    return femur, info


def shaft_axis(femur: np.ndarray, lower_frac: float = 0.40, max_deg: float = 25.0):
    """Угол оси диафиза от вертикали (градусы, + = низ уходит медиально/вправо)."""
    ys, xs = np.nonzero(femur)
    if ys.size < 10:
        return 0.0, {'flags': ['axis:no_femur']}
    y_min, y_max = int(ys.min()), int(ys.max())
    y0 = y_max - lower_frac * (y_max - y_min)
    sel = ys >= y0
    pts = np.stack((ys[sel], xs[sel]), axis=1).astype(float)
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]          # (vy, vx)
    if v[0] < 0:
        v = -v
    phi = float(np.degrees(np.arctan2(v[1], v[0])))
    flags = []
    if abs(phi) > max_deg:
        flags.append('axis:tilt_too_large')
        phi = 0.0
    return phi, {'flags': flags, 'phi_raw_deg': round(float(np.degrees(np.arctan2(v[1], v[0]))), 2)}


class Rotation:
    """Поворот маски вокруг центра кадра на phi градусов с расширением холста.

    rot_to_orig(y, x) отображает точку повёрнутого кадра в исходный. Матрица
    выписана явно, чтобы отображение обратно было точным по построению.
    """

    def __init__(self, shape: tuple[int, int], phi_deg: float, pad: int | None = None):
        self.h, self.w = shape
        self.phi = float(phi_deg)
        if pad is None:
            pad = int(0.3 * max(shape)) + 2
        self.pad = pad
        self.hn, self.wn = self.h + 2 * pad, self.w + 2 * pad
        self.c = ((self.h - 1) / 2.0, (self.w - 1) / 2.0)
        self.cn = ((self.hn - 1) / 2.0, (self.wn - 1) / 2.0)
        t = np.radians(self.phi)
        self.cos, self.sin = float(np.cos(t)), float(np.sin(t))

    def rot_to_orig(self, y, x):
        dy, dx = np.asarray(y, float) - self.cn[0], np.asarray(x, float) - self.cn[1]
        ys = self.c[0] + self.cos * dy - self.sin * dx
        xs = self.c[1] + self.sin * dy + self.cos * dx
        return ys, xs

    def apply(self, mask: np.ndarray) -> np.ndarray:
        yy, xx = np.mgrid[0:self.hn, 0:self.wn]
        ys, xs = self.rot_to_orig(yy, xx)
        out = ndi.map_coordinates(mask.astype(np.uint8), [ys, xs], order=0, cval=0)
        return out.astype(bool)

    def point_to_orig(self, y: float, x: float) -> tuple[int, int]:
        ys, xs = self.rot_to_orig(y, x)
        return int(round(float(ys))), int(round(float(xs)))
