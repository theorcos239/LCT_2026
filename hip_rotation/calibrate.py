# -*- coding: utf-8 -*-
"""Калибровка коридора нормы и честная оценка критерия ротации.

    python -m hip_rotation.calibrate

Коридор подбирается по перцентилям кадров, признанных экспертом корректными,
а не по максимуму F1: критерий двусторонний, и подгонка обеих границ по 36
положительным примерам — верный способ получить красивую цифру, которая не
повторится. Перцентиль — один параметр вместо двух.

Оценка out-of-fold: коридор считается по обучающим фолдам, применяется к
отложенному.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stats
import trainset
from hip_roi.kits import measure_roi_margins
from region_clf.features import featurize, flip_lr, preprocess
from spine_qc import probability as prob

from .detect import DEFAULTS, corridor_distance, lesser_trochanter_bulge

HERE = Path(__file__).resolve().parent
THRESHOLDS = HERE / 'thresholds.json'
METRICS = HERE / 'metrics.json'
IMG_SIZE = 128


def vector(img: np.ndarray, side: str) -> np.ndarray:
    """Вид кадра, приведённый к одной стороне — иначе lh и rh зеркальны."""
    return featurize(preprocess(flip_lr(img) if side == 'lh' else img, size=IMG_SIZE))


def measure(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows, vecs = [], []
    for r in df.itertuples():
        img = trainset.read(r.rel_path)
        T = measure_roi_margins(img, r.label, 'kit2').get('T_px')
        f = lesser_trochanter_bulge(img, r.label, int(T)) if T is not None else None
        rows.append({'study': r.study, 'rel_path': r.rel_path, 'side': r.label,
                     'fold': r.fold, 'y': r.y_rotation, 'T': T,
                     **{k: (f or {}).get(k) for k in
                        ('bulge_mm', 'area_mm2', 'span_mm', 'shaft_mm', 'relative')}})
        vecs.append(vector(img, r.label))
    return pd.DataFrame(rows), np.stack(vecs)


def corridor(area: np.ndarray, q: float) -> tuple[float, float]:
    lo, hi = np.percentile(area, [q, 100.0 - q])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description='Калибровка критерия ротации бедра')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    df = trainset.frames('hip')
    d, X = measure(df)
    ok = d.y.notna() & d.area_mm2.notna()
    print(f'{len(df)} кадров бедра, измерено {int(ok.sum())}, нарушений {int(d.y[ok].sum())}')

    y_all = d.y.values.astype(float)
    area = d.area_mm2.values.astype(float)
    folds = sorted(d.fold.unique())

    # выбор перцентиля по OOF
    best_q, best_f1 = 10.0, -1.0
    for q in (5.0, 10.0, 15.0, 20.0, 25.0):
        pred = np.zeros(len(d), int)
        for f in folds:
            te = (d.fold == f).values
            tr = ~te & ok.values & (y_all == 0)
            if tr.sum() < 10:
                continue
            lo, hi = corridor(area[tr], q)
            pred[te] = [(corridor_distance(a, {'area_lo': lo, 'area_hi': hi}) > 0) if np.isfinite(a) else 0
                        for a in area[te]]
        c = stats.confusion(y_all[ok].astype(int), pred[ok.values])
        print(f'  перцентиль {q:4.1f}: tp={c["tp"]:2d} fp={c["fp"]:2d} fn={c["fn"]:2d} '
              f'sens={c["sensitivity"]:.2f} spec={c["specificity"]:.2f} F1={c["f1"]:.2f}')
        if c['f1'] > best_f1:
            best_q, best_f1 = q, c['f1']

    # OOF-вердикт на выбранном перцентиле + модель вида кадра для сверки.
    # Вероятность (Платт по выходу за коридор) тоже учится только на обучающих
    # фолдах — иначе Brier на отложенном был бы оценкой на своих же данных.
    pred = np.zeros(len(d), int)
    oof_app = np.full(len(d), np.nan)
    oof_prob = np.full(len(d), np.nan)
    from spine_qc.appearance import make_model
    for f in folds:
        te = (d.fold == f).values
        tr_ok = ~te & ok.values
        lo, hi = corridor(area[tr_ok & (y_all == 0)], best_q)
        c = {'area_lo': lo, 'area_hi': hi}
        dist_f = np.array([corridor_distance(a, c) if np.isfinite(a) else np.nan for a in area])
        pred[te] = [(x > 0) if np.isfinite(x) else 0 for x in dist_f[te]]
        pl = prob.fit(y_all[tr_ok], dist_f[tr_ok])
        oof_prob[te] = [prob.apply(pl, x) if np.isfinite(x) else np.nan for x in dist_f[te]]
        m = make_model()
        m.fit(X[tr_ok], y_all[tr_ok].astype(int))
        oof_app[te] = m.predict_proba(X[te])[:, 1]

    yb = y_all[ok].astype(int)
    dist = np.array([corridor_distance(a, {'area_lo': corridor(area[ok & (d.y == 0)], best_q)[0],
                                           'area_hi': corridor(area[ok & (d.y == 0)], best_q)[1]})
                     for a in area[ok]])
    m = stats.evaluate(yb, pred[ok.values], dist, d.study.values[ok])
    m['appearance_auc'] = stats.roc_auc(yb, oof_app[ok.values])
    m['brier'] = round(prob.brier(y_all[ok.values], oof_prob[ok.values]), 4)
    m['prob_mean'] = round(float(np.nanmean(oof_prob[ok.values])), 4)
    m['base_rate'] = round(float(np.mean(yb)), 4)
    m['corridor_percentile'] = best_q
    m['positives'] = int(yb.sum())
    m['measured'] = int(ok.sum())
    print(f'\nOOF: {stats.fmt(m)}')
    print(f'  AUC выступа {stats.roc_auc(yb, dist):.3f}, вида кадра {m["appearance_auc"]:.3f}, '
          f'Brier {m["brier"]:.3f} (доля нарушений {m["base_rate"]:.2f})')

    lo, hi = corridor(area[ok.values & (y_all == 0)], best_q)
    thr = {**DEFAULTS, 'area_lo': round(lo, 1), 'area_hi': round(hi, 1),
           'corridor_percentile': best_q}
    # Вероятность нарушения для отчёта (колонка p_hip_rotation): Платт по
    # выходу за коридор поставляемых границ, на всей выборке — как у
    # критериев позвоночника (spine_qc/probability.py).
    full = np.array([corridor_distance(a, thr) if np.isfinite(a) else np.nan for a in area])
    pl = prob.fit(y_all, full)
    if pl is not None:
        thr['probability'] = pl
    # «кв. мм», а не «мм²»: «²» нет в cp1251, и в консоли Windows print падал
    # раньше, чем калибровка успевала сохранить пороги.
    print('коридор нормы для поставки: %.0f-%.0f кв. мм' % (lo, hi))

    if not args.no_save:
        THRESHOLDS.write_text(json.dumps(thr, ensure_ascii=False, indent=2), encoding='utf-8')
        METRICS.write_text(json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in m.items()},
                                      ensure_ascii=False, indent=2, default=float),
                           encoding='utf-8')
        d.to_csv(HERE / 'measurements.csv', index=False, encoding='utf-8')
        print(f'сохранено: {THRESHOLDS.name}, {METRICS.name}, measurements.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
