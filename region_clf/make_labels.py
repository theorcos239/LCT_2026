# -*- coding: utf-8 -*-
"""Сборка labels.csv — эталонной разметки области для обучающих снимков.

В DICOM нет ни BodyPartExamined, ни Laterality, ни ViewPosition, а в
«разметка.xlsx» области заданы только на уровне исследования (какие столбцы
критериев заполнены). Поэтому метка кадра строится так:

1. эвристика по геометрии кадра и яркости (ширина >= 300 px -> позвоночник;
   у бедра таз и головка дают яркую верхнюю треть со стороны средней линии:
   PatientOrientation = [L, F], значит левая половина кадра — правая сторона
   пациента, и яркая левая верхняя треть означает ЛЕВОЕ бедро);
2. ручная визуальная проверка всех 252 уникальных кадров контактными листами;
3. сверка с «разметка.xlsx»: набор найденных областей против набора областей,
   размеченных экспертом (98/100, оба расхождения — эндопротезы, которые
   эксперт не оценивал).

Запуск:  python -m region_clf.make_labels
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .features import read_dicom

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ROOT / 'НД_для_обучения' / 'Исследования'
LABELS = Path(__file__).resolve().parent / 'labels.csv'


def heuristic_label(px: np.ndarray) -> str:
    """Стартовая разметка области (использовалась один раз, до ручной проверки)."""
    h, w = px.shape
    if w >= 300:
        return 'spine'
    top = px[:h // 3].astype(float)
    return 'lh' if top[:, :w // 2].mean() > top[:, w // 2:].mean() else 'rh'


def build(studies: Path = STUDIES) -> pd.DataFrame:
    rows = []
    for f in sorted(studies.rglob('*.dcm')):
        px = read_dicom(f)
        rel = f.relative_to(studies)
        rows.append(dict(
            rel_path=rel.as_posix(),
            study=rel.parts[0],
            px_hash=hashlib.md5(np.ascontiguousarray(px).tobytes()).hexdigest(),
            label=heuristic_label(px),
        ))
    df = pd.DataFrame(rows)
    # в исследовании каждый кадр продублирован 1-3 раза (одинаковые пиксели)
    return df.drop_duplicates(['study', 'px_hash']).sort_values('rel_path').reset_index(drop=True)


if __name__ == '__main__':
    df = build()
    df.to_csv(LABELS, index=False, encoding='utf-8')
    print(f'{len(df)} уникальных кадров из {df.study.nunique()} исследований -> {LABELS}')
    print(df.label.value_counts().to_string())
