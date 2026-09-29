# -*- coding: utf-8 -*-
"""Калиброванная вероятность нарушения отступов ROI (колонка p_hip_roi).

    python -m hip_roi.probability     # по eval/margins_kit2.csv -> probability.json

Две калибровки — по двум правилам вердикта (service.pipeline, roi_rule):

* scan_length (по умолчанию) — недобор длины поля сканирования до 129 мм
  (3 см + вертел — седалищная кость + 3 см, kits.scan_field), мм;
* margins — наибольший недобор отступа до нормы по рисунку 6 ТЗ:
  30 − сверху, 30 − снизу, 20 − латерально, мм.

Калибровка — Платт (spine_qc/probability.py) по кадрам с меткой эксперта;
коэффициентов два, положительных примеров семь, поэтому рядом кладётся OOF
Brier — по нему видно, насколько вероятность переносится на отложенный фолд.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spine_qc import probability as prob

from .geometry import BOTTOM_MM, D_TI_MM, LAT_MM, MM_PER_PX, TOP_MM

HERE = Path(__file__).resolve().parent
PARAMS_PATH = HERE / 'probability.json'
MARGINS = HERE / 'eval' / 'margins_kit2.csv'
RULES = ('scan_length', 'margins')


def shortfall(roi: dict) -> float | None:
    """Наибольший недобор отступа до нормы ТЗ, мм. None — ни один не измерен."""
    gaps = [norm - roi[k] for k, norm in (('m_top_mm', TOP_MM), ('m_bottom_mm', BOTTOM_MM),
                                          ('m_lat_mm', LAT_MM))
            if roi.get(k) is not None and np.isfinite(roi[k])]
    return float(max(gaps)) if gaps else None


def length_shortfall(roi: dict) -> float | None:
    """Недобор длины поля сканирования до нужной, мм (больше нуля — короче нормы)."""
    if roi.get('scan_length_mm') is None:
        return None
    need = roi.get('min_length_mm') or (TOP_MM + D_TI_MM + BOTTOM_MM)
    return float(need - roi['scan_length_mm'])


def score(roi: dict, rule: str | None = None) -> float | None:
    rule = rule or roi.get('roi_rule') or 'margins'
    return length_shortfall(roi) if rule == 'scan_length' else shortfall(roi)


def load() -> dict:
    if not PARAMS_PATH.exists():
        return {}
    return {k: v.get('params') for k, v in json.loads(PARAMS_PATH.read_text(encoding='utf-8')).items()
            if isinstance(v, dict)}


PARAMS = load()


def apply(roi: dict, rule: str | None = None) -> float | None:
    rule = rule or roi.get('roi_rule') or 'margins'
    return prob.apply(PARAMS.get(rule), score(roi, rule))


def main() -> int:
    import stats
    import trainset

    d = pd.read_csv(MARGINS, encoding='utf-8-sig')
    d = d[d.expert_roi.notna()].copy()
    d['fold'] = d.study.map(trainset.load_folds().fold)
    need = TOP_MM + D_TI_MM + BOTTOM_MM
    scores = {
        'scan_length': need - d.H.values.astype(float) * MM_PER_PX,
        'margins': np.array([shortfall(r) for r in d[['m_top_mm', 'm_bottom_mm', 'm_lat_mm']]
                             .to_dict('records')], float),
    }
    y = d.expert_roi.values.astype(float)
    out = {}
    for rule, s in scores.items():
        oof = np.full(len(d), np.nan)
        for k in sorted(d.fold.dropna().unique()):
            te = (d.fold == k).values
            pl = prob.fit(y[~te], s[~te])
            oof[te] = [prob.apply(pl, x) for x in s[te]]
        ok = np.isfinite(oof)
        pred = (s > 0).astype(int)
        c = stats.confusion(y.astype(int), pred)
        out[rule] = {
            'params': prob.fit(y, s),
            'score': ('129 - длина поля сканирования, мм' if rule == 'scan_length' else
                      'max(30 - m_top_mm, 30 - m_bottom_mm, 20 - m_lat_mm), мм'),
            'n': int(len(d)), 'positives': int(y.sum()),
            'tp': c['tp'], 'fp': c['fp'], 'fn': c['fn'], 'f1': round(c['f1'], 4),
            'roc_auc': round(float(stats.roc_auc(y.astype(int), s)), 4),
            'brier_oof': round(prob.brier(y[ok], oof[ok]), 4),
            'base_rate': round(float(y.mean()), 4),
        }
    PARAMS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
