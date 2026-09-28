# -*- coding: utf-8 -*-
"""Чтение DICOM, обход исследований и дедупликация кадров.

Три вещи, без которых пакетная обработка этих данных даёт неверный результат:

1. **Дедупликация по пикселям.** Lunar выгружает один и тот же снимок по
   несколько раз (переанализ, печать отчёта). В 499 файлах обучающего набора
   всего 252 уникальных кадра. У копий РАЗНЫЕ SOPInstanceUID и InstanceNumber,
   поэтому ловятся они только по хешу пикселей. Без дедупликации одно
   исследование даёт до 29 строк отчёта вместо трёх.
2. **Ключ исследования — папка, а не StudyInstanceUID.** В обучающем наборе
   тег анонимизирован и с именем папки не совпадает. В отчёт пишется тег (так
   требует ТЗ), но группировка и разбиение выборки идут по папке.
3. **Ни одно исключение не должно ронять пачку.** Битый файл становится
   строкой отчёта со статусом Failure — так требует ТЗ 2.7.
"""
from __future__ import annotations

import hashlib
import logging
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUFFIXES = {'.dcm', '.dicom', ''}
_SILENCED = False

# Служебные файлы, которые лежат рядом со снимками в выгрузках из PACS и в
# архивах, собранных на macOS. Снимками они не являются, и строка Failure на
# каждый из них испортила бы и отчёт («одна строка — одно изображение»), и
# долю успешно обработанных файлов.
_SERVICE_NAMES = {'dicomdir', 'thumbs.db', 'desktop.ini'}


class NotAnImage(ValueError):
    """Валидный DICOM без пикселей: DICOMDIR, SR, отчёт PDF, KO, PR.

    Это не снимок, поэтому строки отчёта он не даёт — в отличие от битого
    файла, который снимком быть должен и уходит в Failure.
    """


def _is_service_file(p: Path) -> bool:
    return (p.name.lower() in _SERVICE_NAMES or p.name.startswith('.')
            or '__MACOSX' in p.parts)


def _is_candidate(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUFFIXES and not _is_service_file(p)


def _silence() -> None:
    """pydicom шумит на невалидных UID (1.2.643…) и длинных LO-полях."""
    global _SILENCED
    if not _SILENCED:
        warnings.filterwarnings('ignore')
        logging.getLogger('pydicom').setLevel(logging.ERROR)
        _SILENCED = True


@dataclass
class Frame:
    """Один уникальный кадр исследования."""
    path: Path
    study_key: str                  # имя папки исследования — ключ группировки
    study_uid: str                  # StudyInstanceUID из тегов (в отчёт)
    image_uid: str                  # SOPInstanceUID из тегов (в отчёт)
    pixels: np.ndarray
    px_hash: str
    duplicates: list[Path] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def normalize_orientation(px: np.ndarray, tag) -> tuple[np.ndarray, str]:
    """Кадр -> стандартный AP-вид (PatientOrientation = L, F).

    Все модели учились на выгрузке Lunar, где вправо по снимку — к левому боку
    пациента, вниз — к ногам (а у четверти кадров тег пуст). Зеркальная
    выгрузка другого аппарата (R, F) без нормализации поменяла бы местами
    левое и правое бедро. Пустой или нечитаемый тег кадр не меняет.

    Возвращает пиксели и то, что сделано: 'L\\F' (как есть), 'mirrored_lr',
    'mirrored_hf', 'rotated_180' или '' (тега нет).
    """
    try:
        row, col = (str(v).upper()[:1] for v in list(tag)[:2]) if tag else ('', '')
    except Exception:
        row, col = '', ''
    if not row or not col:
        return px, ''
    flip_lr, flip_ud = row == 'R', col == 'H'
    if row not in ('L', 'R') or col not in ('F', 'H'):
        return px, f'{row}\\{col}'            # поперечная ориентация: не трогаем, флаг в отчёте
    if flip_lr:
        px = px[:, ::-1]
    if flip_ud:
        px = px[::-1, :]
    done = {(False, False): 'L\\F', (True, False): 'mirrored_lr',
            (False, True): 'mirrored_hf', (True, True): 'rotated_180'}[(flip_lr, flip_ud)]
    return np.ascontiguousarray(px), done


def read_frame(path: str | Path, study_key: str | None = None) -> Frame:
    """DICOM -> Frame. Яркость приведена к «кость светлая»."""
    import pydicom
    from pydicom.pixels import apply_voi_lut

    _silence()
    p = Path(path)
    ds = pydicom.dcmread(str(p))
    if not any(k in ds for k in ('PixelData', 'FloatPixelData', 'DoubleFloatPixelData')):
        sop = ds.get('SOPClassUID')
        raise NotAnImage(f'нет пиксельных данных: {getattr(sop, "name", sop) or "без SOPClassUID"}')
    px = ds.pixel_array
    try:
        px = apply_voi_lut(px, ds)
    except Exception:
        pass                                # LUT необязателен, кадр читается и без него
    px = px.astype(np.float32)
    if str(ds.get('PhotometricInterpretation', 'MONOCHROME2')) == 'MONOCHROME1':
        px = px.max() - px
    if px.ndim != 2:
        raise ValueError(f'ожидался одноканальный кадр, получено {px.shape}')
    px, orientation = normalize_orientation(px, ds.get('PatientOrientation'))

    return Frame(
        path=p,
        study_key=study_key or p.parent.name,
        study_uid=str(ds.get('StudyInstanceUID', '') or ''),
        image_uid=str(ds.get('SOPInstanceUID', '') or ''),
        pixels=px,
        px_hash=hashlib.md5(np.ascontiguousarray(px).tobytes()).hexdigest(),
        meta={'rows': int(px.shape[0]), 'cols': int(px.shape[1]),
              'orientation': orientation,
              'instance': ds.get('InstanceNumber', None),
              'manufacturer': str(ds.get('Manufacturer', '') or ''),
              'modality': str(ds.get('Modality', '') or '')},
    )


def dicom_files(root: str | Path) -> list[Path]:
    """Все файлы, похожие на DICOM, рекурсивно и в детерминированном порядке.

    Служебные файлы (DICOMDIR, .DS_Store, __MACOSX/) отсеиваются по имени.
    Явно указанный файл берётся как есть.
    """
    p = Path(root)
    if p.is_file():
        return [p]
    return [f for f in sorted(p.rglob('*')) if _is_candidate(f)]


def study_dirs(root: str | Path) -> list[Path]:
    """Папки исследований.

    Исследование — это папка, внутри которой (на любой глубине) лежат DICOM.
    Если `root` сам содержит файлы, он и есть единственное исследование:
    так работает и «архив из 100 папок», и «одно исследование», и «один файл».

    Служебные файлы в корне не считаются: экспорт из PACS — это DICOMDIR в
    корне плюс папки исследований, и без этого фильтра вся выгрузка стала бы
    одним исследованием с общей дедупликацией через границы пациентов.
    """
    p = Path(root)
    if p.is_file():
        return [p.parent]
    if any(_is_candidate(f) for f in p.iterdir()):
        return [p]
    subs = _dicom_subdirs(p)
    # Архив, сжатый из папки, — это одна папка-обёртка, а в ней исследования.
    # Без этой проверки обёртка стала бы одним исследованием на всех пациентов.
    if len(subs) == 1:
        inner = _through_single_dirs(subs[0])
        if _distinct_studies(_dicom_subdirs(inner)) >= 2:
            return study_dirs(inner)
    return subs or ([p] if dicom_files(p) else [])


def _dicom_subdirs(p: Path) -> list[Path]:
    return [d for d in sorted(p.iterdir())
            if d.is_dir() and not _is_service_file(d) and dicom_files(d)]


def _through_single_dirs(d: Path) -> Path:
    """Спуск по цепочке папок, в каждой из которых только одна подпапка с DICOM."""
    while not any(_is_candidate(f) for f in d.iterdir()):
        subs = _dicom_subdirs(d)
        if len(subs) != 1:
            break
        d = subs[0]
    return d


def _distinct_studies(dirs: list[Path]) -> int:
    """Сколько разных StudyInstanceUID среди папок (по первому читаемому файлу).

    Серии одного исследования делят UID и остаются одним исследованием. Если UID
    не прочитать или он пуст — считаем, что различить нельзя (0): тогда папка
    остаётся одним исследованием, как было, и ошибиться в худшую сторону нельзя.
    """
    import pydicom

    _silence()
    uids = set()
    for d in dirs:
        for f in dicom_files(d):
            try:
                uid = str(pydicom.dcmread(str(f), stop_before_pixels=True,
                                          specific_tags=['StudyInstanceUID'])
                          .get('StudyInstanceUID', '') or '')
            except Exception:
                continue
            if not uid:
                return 0
            uids.add(uid)
            break
    return len(uids)


def unique_frames(study: str | Path, study_key: str | None = None
                  ) -> tuple[list[Frame], list[tuple[Path, Exception]]]:
    """Уникальные кадры исследования и список непрочитанных файлов.

    Дедупликация идёт ВНУТРИ исследования: одинаковых кадров в разных
    исследованиях в обучающем наборе нет, а схлопывать их между пациентами
    было бы неверно в принципе.

    Не снимки в `failed` не попадают и строк отчёта не дают: DICOM без
    пикселей (DICOMDIR, SR, PDF) и файл без расширения, который вообще не
    DICOM. Битый файл `.dcm` — попадает: он обязан был быть снимком.
    """
    from pydicom.errors import InvalidDicomError

    key = study_key or Path(study).name
    seen: dict[str, Frame] = {}
    failed: list[tuple[Path, Exception]] = []
    for f in dicom_files(study):
        try:
            fr = read_frame(f, key)
        except NotAnImage:
            continue
        except InvalidDicomError as e:
            if f.suffix == '':
                continue                            # посторонний файл без расширения
            failed.append((f, e))
            continue
        except Exception as e:                       # битый файл не роняет пачку
            failed.append((f, e))
            continue
        if fr.px_hash in seen:
            seen[fr.px_hash].duplicates.append(f)
        else:
            seen[fr.px_hash] = fr
    return list(seen.values()), failed


def extract_archive(archive: str | Path, dest: str | Path) -> Path:
    """Распаковка zip с исследованиями. Пути внутри архива проверяются.

    Защита от path traversal обязательна: архив приходит извне, а `extractall`
    сам по себе позволяет файлу с именем `../../x` уехать за пределы каталога.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f'недопустимый путь в архиве: {member}')
        z.extractall(dest)
    return dest
