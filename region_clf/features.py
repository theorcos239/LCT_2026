# -*- coding: utf-8 -*-
"""Загрузка DXA-снимка и извлечение признаков для классификации области.

Признаки строятся ТОЛЬКО по содержимому изображения (без размера кадра, тегов
DICOM и имён файлов): исходный размер кадра (300 px — позвоночник, 280 px —
бедро) жёстко привязан к конкретной модели денситометра и на других аппаратах
работать перестанет.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

IMG_SIZE = 64          # сторона нормализованного изображения
CLASSES = ('lh', 'rh', 'spine')
CLASS_RU = {'lh': 'левое бедро', 'rh': 'правое бедро', 'spine': 'позвоночник'}


# --------------------------------------------------------------------------- #
#  Чтение
# --------------------------------------------------------------------------- #
def read_dicom(path: str | Path) -> np.ndarray:
    """DICOM -> float-массив, яркость «кость = светлая», диапазон произвольный."""
    import pydicom
    from pydicom.pixels import apply_voi_lut

    warnings.filterwarnings('ignore')
    logging.getLogger('pydicom').setLevel(logging.ERROR)

    ds = pydicom.dcmread(str(path))
    px = ds.pixel_array
    try:
        px = apply_voi_lut(px, ds)
    except Exception:
        pass
    px = px.astype(np.float32)
    if str(ds.get('PhotometricInterpretation', 'MONOCHROME2')) == 'MONOCHROME1':
        px = px.max() - px          # в MONOCHROME1 кость тёмная — инвертируем
    return px


def read_image(path: str | Path) -> np.ndarray:
    """Любой поддерживаемый формат: .dcm или обычная картинка (png/jpg/tif)."""
    p = Path(path)
    if p.suffix.lower() in ('.dcm', '.dicom', ''):
        return read_dicom(p)
    from PIL import Image
    return np.asarray(Image.open(p).convert('L'), dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Нормализация
# --------------------------------------------------------------------------- #
def _robust_scale(px: np.ndarray) -> np.ndarray:
    """Растяжка контраста по перцентилям -> [0, 1]. Гасит разброс доз и LUT."""
    lo, hi = np.percentile(px, (1.0, 99.0))
    if hi - lo < 1e-6:
        lo, hi = float(px.min()), float(max(px.max(), px.min() + 1e-6))
    return np.clip((px - lo) / (hi - lo), 0.0, 1.0)


def _crop_content(px01: np.ndarray, thr: float = 0.35, margin: float = 0.02) -> np.ndarray:
    """Обрезка чёрного поля вокруг анатомии по профилям строк/столбцов.

    Кадр DXA часто смещён и добит нулями (см. «ступеньки» по краям снимков).
    По умолчанию ВЫКЛЮЧЕНА: в стресс-тесте (см. README) обрезка не улучшила ни
    чистую точность, ни устойчивость к сдвигам — порог по профилю сам сбивается
    на ярких артефактах. Оставлена как опция для данных с большим полем.
    """
    h, w = px01.shape
    row, col = px01.mean(axis=1), px01.mean(axis=0)

    def span(prof: np.ndarray) -> tuple[int, int]:
        lo, hi = prof.min(), prof.max()
        if hi - lo < 1e-3:
            return 0, len(prof)
        keep = np.flatnonzero(prof >= lo + thr * (hi - lo))
        if keep.size < 4:
            return 0, len(prof)
        return int(keep[0]), int(keep[-1]) + 1

    r0, r1 = span(row)
    c0, c1 = span(col)
    dr, dc = int(margin * h), int(margin * w)
    r0, r1 = max(0, r0 - dr), min(h, r1 + dr)
    c0, c1 = max(0, c0 - dc), min(w, c1 + dc)
    out = px01[r0:r1, c0:c1]
    return out if min(out.shape) >= 16 else px01


def preprocess(px: np.ndarray, crop: bool = False, size: int = IMG_SIZE) -> np.ndarray:
    """Сырые пиксели -> квадрат size×size в [0, 1] со стандартизованным контрастом."""
    from PIL import Image

    img = _robust_scale(px)
    if crop:
        img = _crop_content(img)
        img = _robust_scale(img)
    small = Image.fromarray((img * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
#  Признаки
# --------------------------------------------------------------------------- #
def _grad_hist(img: np.ndarray, cells: int = 8, bins: int = 6) -> np.ndarray:
    """Упрощённый HOG: ориентации градиента по сетке cells×cells, bins корзин.

    Даёт направление хода диафиза бедра (вниз-влево против вниз-вправо) —
    главный признак, отличающий правое бедро от левого.
    """
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) % np.pi                      # 0..pi, знак не важен
    b = np.minimum((ang / np.pi * bins).astype(int), bins - 1)

    n = img.shape[0] // cells
    out = np.zeros((cells, cells, bins), dtype=np.float32)
    for i in range(cells):
        for j in range(cells):
            sb = b[i * n:(i + 1) * n, j * n:(j + 1) * n].ravel()
            sm = mag[i * n:(i + 1) * n, j * n:(j + 1) * n].ravel()
            out[i, j] = np.bincount(sb, weights=sm, minlength=bins)
    out /= np.linalg.norm(out, axis=2, keepdims=True) + 1e-6   # L2 по ячейке
    return out.ravel()


def featurize(img: np.ndarray) -> np.ndarray:
    """Нормализованное изображение -> вектор признаков (пиксели + градиенты)."""
    small = img[::2, ::2]                                  # 32×32 = 1024 значения
    return np.concatenate([small.ravel(), _grad_hist(img)]).astype(np.float32)


def contrast(img: np.ndarray) -> float:
    """Разброс яркости нормализованного кадра.

    У пустого или равномерно засвеченного кадра он близок к нулю. Отдельная
    проверка нужна потому, что детектор новизны такой кадр пропускает: после
    нормализации константа лежит близко к среднему обучающей выборки.
    """
    return float(img.std())


def flip_lr(img: np.ndarray) -> np.ndarray:
    """Зеркалирование кадра. Меняет lh <-> rh, spine оставляет spine."""
    return np.ascontiguousarray(img[:, ::-1])


FLIP_LABEL = {'lh': 'rh', 'rh': 'lh', 'spine': 'spine'}


def flip_ud(img: np.ndarray) -> np.ndarray:
    """Переворот кадра сверху вниз.

    Физически невозможная для денситометра ориентация: таз всегда сверху.
    Используется как отрицательный пример для проверки ориентации — такой кадр
    детектор новизны не ловит, а сторону бедра модель на нём определяет неверно.
    """
    return np.ascontiguousarray(img[::-1, :])


def pipeline(path: str | Path, crop: bool = False) -> np.ndarray:
    """Файл -> вектор признаков.

    Значение crop по умолчанию обязано совпадать с обучением; в инференсе оно
    берётся не отсюда, а из сохранённой модели (см. RegionClassifier).
    """
    return featurize(preprocess(read_image(path), crop=crop))
