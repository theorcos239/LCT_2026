# -*- coding: utf-8 -*-
"""Индекс обучающей выборки: уникальный кадр -> область, метки эксперта, фолд.

Одна таблица на весь проект, чтобы обучение и оценка всех критериев смотрели
на одни и те же 252 кадра и одно и то же разбиение. Собирается из трёх
источников:

    region_clf/labels.csv        область кадра (lh / rh / spine), дедуплицировано
    НД_для_обучения/разметка.xlsx  метки эксперта, заданы на исследование
    folds.csv                    5 фолдов, группа = исследование

Метки эксперта заданы по исследованию и по области сразу («правое бедро,
ротация»), поэтому здесь они раскладываются на кадр: снимок левого бедра
получает `lhip_rotation` своего исследования, снимок позвоночника — три
колонки позвоночника, чужие колонки становятся NaN.

    import trainset
    df = trainset.frames()                  # 252 строки
    df = trainset.frames('spine')           # 99
    img = trainset.read(df.rel_path[0])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'НД_для_обучения' / 'Исследования'
XLSX = ROOT / 'НД_для_обучения' / 'разметка.xlsx'
LABELS = ROOT / 'region_clf' / 'labels.csv'
FOLDS = ROOT / 'folds.csv'

# Заголовок xlsx занимает две строки с объединёнными ячейками -> header=None
# и имена задаём сами. Колонки N-S (сводка эксперта) не используются.
EXPERT_COLS = ['n', 'study', 'spine_position', 'spine_axis', 'spine_artifacts',
               'rhip_rotation', 'rhip_roi', 'lhip_rotation', 'lhip_roi',
               'spine_bad', 'rhip_bad', 'lhip_bad', 'comment']

# Критерий -> имя колонки эксперта для кадра этой области.
TARGETS = {
    'spine': {'position': 'spine_position', 'axis': 'spine_axis', 'artifacts': 'spine_artifacts'},
    'lh': {'rotation': 'lhip_rotation', 'roi': 'lhip_roi'},
    'rh': {'rotation': 'rhip_rotation', 'roi': 'rhip_roi'},
}
CRITERIA = ('position', 'axis', 'artifacts', 'rotation', 'roi')


def load_expert() -> pd.DataFrame:
    """разметка.xlsx -> DataFrame, индекс = ключ исследования (имя папки)."""
    df = pd.read_excel(XLSX, header=None).iloc[2:, :13]
    df.columns = EXPERT_COLS
    return df.set_index('study')


def load_folds() -> pd.DataFrame:
    return pd.read_csv(FOLDS).set_index('study')


def frames(region: str | None = None) -> pd.DataFrame:
    """Уникальные кадры с метками эксперта и номером фолда.

    Колонки: rel_path, study, px_hash, label, fold, stratum, y_position,
    y_axis, y_artifacts, y_rotation, y_roi, comment. Целевые колонки — 0/1 или
    NaN («эксперт эту область не оценивал»: нет снимка либо эндопротез).
    """
    lab = pd.read_csv(LABELS)
    ex, fl = load_expert(), load_folds()

    out = []
    for r in lab.itertuples(index=False):
        row = {'rel_path': r.rel_path, 'study': r.study, 'px_hash': r.px_hash, 'label': r.label}
        row['fold'] = int(fl.fold[r.study]) if r.study in fl.index else -1
        row['stratum'] = fl.stratum[r.study] if r.study in fl.index else ''
        e = ex.loc[r.study] if r.study in ex.index else None
        for crit in CRITERIA:
            col = TARGETS[r.label].get(crit)
            v = e[col] if (e is not None and col) else np.nan
            row[f'y_{crit}'] = np.nan if pd.isna(v) else int(v)
        row['comment'] = '' if e is None or pd.isna(e.comment) else str(e.comment)
        out.append(row)

    df = pd.DataFrame(out)
    if region == 'hip':
        df = df[df.label.isin(('lh', 'rh'))]
    elif region is not None:
        df = df[df.label == region]
    return df.sort_values('rel_path').reset_index(drop=True)


def read(rel_path: str) -> np.ndarray:
    """Кадр по пути относительно НД_для_обучения/Исследования."""
    from region_clf.features import read_image
    return read_image(DATA / rel_path)


if __name__ == '__main__':
    df = frames()
    print(f'{len(df)} кадров, {df.study.nunique()} исследований, {df.fold.nunique()} фолдов')
    print(df.label.value_counts().to_string())
    print('\nположительных / размечено:')
    for c in CRITERIA:
        col = df[f'y_{c}']
        print(f'  {c:10} {int(col.sum()):3d} / {int(col.notna().sum()):3d}')
