"""Разбиение на фолды: группа — исследование, страта — самое редкое нарушение.

Файл folds.csv лежит в репозитории и только читается. Пересчёт — отдельной командой,
иначе сравнение моделей между участниками теряет смысл.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# От самого редкого к частому: так редкие классы гарантированно расходятся по фолдам.
RARITY = [
    ("roi", ["rhip_roi", "lhip_roi"]),
    ("spine_position", ["spine_position"]),
    ("axis", ["spine_axis"]),
    ("artifacts", ["spine_artifacts"]),
    ("rotation", ["rhip_rotation", "lhip_rotation"]),
]


def stratum(row: pd.Series) -> str:
    for name, cols in RARITY:
        if (row[cols] == 1).any():
            return name
    return "norm"


def build(labels: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    df = labels.copy()
    df["stratum"] = df.apply(stratum, axis=1)
    df["fold"] = -1
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val) in enumerate(cv.split(df, df.stratum)):
        df.iloc[val, df.columns.get_loc("fold")] = fold
    return df[["study", "stratum", "fold"]].sort_values("study").reset_index(drop=True)


def load(path: str | Path, require: set[str] | None = None) -> pd.DataFrame:
    """Читает folds.csv. Все размеченные исследования обязаны в нём быть.

    Лишние строки допустимы: разбиение считается по всем 100 исследованиям сразу,
    а размечены пока не все.
    """
    df = pd.read_csv(path, dtype={"study": str})
    missing = set(require or ()) - set(df.study)
    if missing:
        raise ValueError(
            f"{path}: нет исследований {sorted(missing)[:3]}. Пересчитывать разбиение "
            "нельзя — результаты станут несравнимыми с прежними прогонами."
        )
    return df
