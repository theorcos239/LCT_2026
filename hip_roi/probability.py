# -*- coding: utf-8 -*-
"""Калиброванная вероятность нарушения отступов ROI (колонка p_hip_roi).

    python -m hip_roi.probability     # по eval/margins_kit2.csv -> probability.json

Вердикт по отступам выносят пороги ТЗ (3 / 3 / 2 см) — они не подбираются и
здесь не меняются. Вероятность нужна отчёту: ROC-AUC по бинарному «да/нет»
вырождается в сбалансированную точность, а у кадра бедра без непрерывного
скора и общий `quality_probability` оставался пустым.

Скор — наибольший недобор отступа до нормы ТЗ, мм: 30 − сверху, 30 − снизу,
20 − латерально. Больше нуля — хотя бы один отступ меньше нормы. Калибровка —
Платт (spine_qc/probability.py) по кадрам с экспертной меткой; коэффициентов
два, положительных примеров семь, поэтому OOF-оценка Brier кладётся рядом —
по ней видно, насколько вероятность переносится на отложенный фолд.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spine_qc import probability as prob

from .geometry import BOTTOM_MM, LAT_MM, TOP_MM

HERE = Path(__file__).resolve().parent
PARAMS_PATH = HERE / 'probability.json'
MARGINS = HERE / 'eval' / 'margins_kit2.csv'


def shortfall(roi: dict) -> float | None:
    """Наибольший недобор отступа до нормы ТЗ, мм. None — ни один не измерен."""
    gaps = [norm - roi[k] for k, norm in (('m_top_mm', TOP_MM), ('m_bottom_mm', BOTTOM_MM),
                                          ('m_lat_mm', LAT_MM))
            if roi.get(k) is not None and np.isfinite(roi[k])]
    return float(max(gaps)) if gaps else None


def load() -> dict | None:
    if not PARAMS_PATH.exists():
        return None
    return json.loads(PARAMS_PATH.read_text(encoding='utf-8')).get('params')


PARAMS = load()


def apply(roi: dict) -> float | None:
    return prob.apply(PARAMS, shortfall(roi))


def main() -> int:
    import stats
    import trainset

    d = pd.read_csv(MARGINS, encoding='utf-8-sig')
    d = d[d.expert_roi.notna()].copy()
    d['score'] = [shortfall(r) for r in d[['m_top_mm', 'm_bottom_mm', 'm_lat_mm']]
                  .to_dict('records')]
    folds = trainset.load_folds().fold
    d['fold'] = d.study.map(folds)
    y, s = d.expert_roi.values.astype(float), d.score.values.astype(float)

    oof = np.full(len(d), np.nan)
    for k in sorted(d.fold.dropna().unique()):
        te = (d.fold == k).values
        pl = prob.fit(y[~te], s[~te])
        oof[te] = [prob.apply(pl, x) for x in s[te]]
    params = prob.fit(y, s)
    ok = np.isfinite(oof)
    out = {
        'params': params,
        'score': 'max(30 - m_top_mm, 30 - m_bottom_mm, 20 - m_lat_mm), мм',
        'n': int(len(d)), 'positives': int(y.sum()),
        'roc_auc': round(float(stats.roc_auc(y.astype(int), s)), 4),
        'brier_oof': round(prob.brier(y[ok], oof[ok]), 4),
        'base_rate': round(float(y.mean()), 4),
    }
    PARAMS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
