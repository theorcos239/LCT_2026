# -*- coding: utf-8 -*-
"""Калибровка коридора нормы и честная оценка критерия ротации.

    python -m hip_rotation.calibrate

Коридор подбирается по перцентилям кадров, признанных экспертом корректными.
Нижняя и верхняя границы — независимые перцентили (асимметрия пере- и
недоротации). Ненадёжные измерения (короткое окно / шейка не достигнута)
в вердикт не идут.

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

Q_LO = (5.0, 10.0, 15.0, 20.0)
Q_HI = (5.0, 10.0, 15.0, 20.0)


def vector(img: np.ndarray, side: str) -> np.ndarray:
    """Вид кадра, приведённый к одной стороне — иначе lh и rh зеркальны."""
    return featurize(preprocess(flip_lr(img) if side == 'lh' else img, size=IMG_SIZE))


def measure(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows, vecs = [], []
    for r in df.itertuples():
        img = trainset.read(r.rel_path)
        T = measure_roi_margins(img, r.label, 'kit2').get('T_px')
        f = lesser_trochanter_bulge(img, r.label, int(T)) if T is not None else None
        flags = list((f or {}).get('flags') or [])
        uncertain = (f is None
                     or 'rotation:short_window' in flags
                     or 'rotation:neck_not_reached' in flags)
        rows.append({'study': r.study, 'rel_path': r.rel_path, 'side': r.label,
                     'fold': r.fold, 'y': r.y_rotation, 'T': T,
                     'uncertain': uncertain,
                     **{k: (f or {}).get(k) for k in
                        ('bulge_mm', 'area_mm2', 'span_mm', 'shaft_mm',
                         'relative', 'area_over_shaft2')}})
        vecs.append(vector(img, r.label))
    return pd.DataFrame(rows), np.stack(vecs)


def corridor_asym(values: np.ndarray, q_lo: float, q_hi: float) -> tuple[float, float]:
    lo, hi = np.percentile(values, [q_lo, 100.0 - q_hi])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description='Калибровка критерия ротации бедра')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    df = trainset.frames('hip')
    d, X = measure(df)
    ok = d.y.notna() & d.area_mm2.notna()
    reliable = ok & ~d.uncertain.fillna(True)
    print(f'{len(df)} кадров бедра, измерено {int(ok.sum())}, '
          f'надёжных {int(reliable.sum())}, нарушений {int(d.y[ok].sum())}')

    y_all = d.y.values.astype(float)
    area = d.area_mm2.values.astype(float)
    folds = sorted(d.fold.unique())
    fold_v = d.fold.values

    best = {'f1': -1.0, 'auc': -1.0, 'q_lo': 15.0, 'q_hi': 15.0, 'score': 'area'}
    for score_kind, vals in (
        ('area', area),
        ('relative', d.relative.values.astype(float)),
        ('area_over_shaft2', d.area_over_shaft2.values.astype(float)),
    ):
        for q_lo in Q_LO:
            for q_hi in Q_HI:
                pred = np.zeros(len(d), int)
                dist_oof = np.full(len(d), np.nan)
                for f in folds:
                    te = fold_v == f
                    tr = (~te) & reliable.values & (y_all == 0) & np.isfinite(vals)
                    if tr.sum() < 10:
                        continue
                    lo, hi = corridor_asym(vals[tr], q_lo, q_hi)
                    c = {'area_lo': lo, 'area_hi': hi}
                    for i in np.flatnonzero(te):
                        if not reliable.values[i] or not np.isfinite(vals[i]):
                            pred[i] = 0
                            continue
                        dist_oof[i] = corridor_distance(float(vals[i]), c)
                        pred[i] = int(dist_oof[i] > 0)
                use = ok.values
                conf = stats.confusion(y_all[use].astype(int), pred[use])
                auc = stats.roc_auc(y_all[use].astype(int),
                                    np.nan_to_num(dist_oof[use], nan=0.0))
                print(f'  {score_kind:16} q_lo={q_lo:4.1f} q_hi={q_hi:4.1f}: '
                      f'tp={conf["tp"]:2d} fp={conf["fp"]:2d} fn={conf["fn"]:2d} '
                      f'F1={conf["f1"]:.2f} AUC={auc:.3f}')
                better = (conf['f1'] > best['f1'] + 1e-9
                          or (abs(conf['f1'] - best['f1']) <= 1e-9 and auc > best['auc']))
                if better:
                    best = {'f1': conf['f1'], 'auc': auc, 'q_lo': q_lo,
                            'q_hi': q_hi, 'score': score_kind}

    print(f"выбрано: score={best['score']} q_lo={best['q_lo']} q_hi={best['q_hi']} "
          f"OOF F1={best['f1']:.2f} AUC={best['auc']:.3f}")

    pred = np.zeros(len(d), int)
    oof_app = np.full(len(d), np.nan)
    oof_prob = np.full(len(d), np.nan)
    from spine_qc.appearance import make_model

    for f in folds:
        te = fold_v == f
        tr_ok = (~te) & reliable.values
        tr_neg = tr_ok & (y_all == 0) & np.isfinite(area)
        lo_a, hi_a = corridor_asym(area[tr_neg], best['q_lo'], best['q_hi'])
        c_area = {'area_lo': lo_a, 'area_hi': hi_a}
        if best['score'] == 'relative':
            vals = d.relative.values.astype(float)
            tr_v = tr_ok & (y_all == 0) & np.isfinite(vals)
            lo, hi = corridor_asym(vals[tr_v], best['q_lo'], best['q_hi'])
            c_v = {'area_lo': lo, 'area_hi': hi}
            dist_v = np.array([
                corridor_distance(float(vals[i]), c_v)
                if reliable.values[i] and np.isfinite(vals[i]) else np.nan
                for i in range(len(d))])
        elif best['score'] == 'area_over_shaft2':
            vals = d.area_over_shaft2.values.astype(float)
            tr_v = tr_ok & (y_all == 0) & np.isfinite(vals)
            lo, hi = corridor_asym(vals[tr_v], best['q_lo'], best['q_hi'])
            c_v = {'area_lo': lo, 'area_hi': hi}
            dist_v = np.array([
                corridor_distance(float(vals[i]), c_v)
                if reliable.values[i] and np.isfinite(vals[i]) else np.nan
                for i in range(len(d))])
        else:
            dist_v = np.array([
                corridor_distance(area[i], c_area)
                if reliable.values[i] and np.isfinite(area[i]) else np.nan
                for i in range(len(d))])
        pred[te] = [(x > 0) if np.isfinite(x) else 0 for x in dist_v[te]]
        area_dist = np.array([
            corridor_distance(area[i], c_area)
            if reliable.values[i] and np.isfinite(area[i]) else np.nan
            for i in range(len(d))])
        pl = prob.fit(y_all[tr_ok], area_dist[tr_ok])
        oof_prob[te] = [prob.apply(pl, x) if np.isfinite(x) else np.nan
                        for x in area_dist[te]]
        m = make_model()
        m.fit(X[tr_ok], y_all[tr_ok].astype(int))
        oof_app[te] = m.predict_proba(X[te])[:, 1]

    yb = y_all[ok].astype(int)
    lo_a, hi_a = corridor_asym(area[reliable.values & (y_all == 0)],
                               best['q_lo'], best['q_hi'])
    dist = np.array([
        corridor_distance(a, {'area_lo': lo_a, 'area_hi': hi_a}) if np.isfinite(a) else np.nan
        for a in area[ok]])
    m = stats.evaluate(yb, pred[ok.values], np.nan_to_num(dist, nan=0.0),
                       d.study.values[ok])
    m['appearance_auc'] = stats.roc_auc(yb, oof_app[ok.values])
    m['brier'] = round(prob.brier(y_all[ok.values], oof_prob[ok.values]), 4)
    m['prob_mean'] = round(float(np.nanmean(oof_prob[ok.values])), 4)
    m['base_rate'] = round(float(np.mean(yb)), 4)
    m['corridor_percentile_lo'] = best['q_lo']
    m['corridor_percentile_hi'] = best['q_hi']
    m['score'] = best['score']
    m['positives'] = int(yb.sum())
    m['measured'] = int(ok.sum())
    m['reliable'] = int(reliable.sum())
    print(f'\nOOF: {stats.fmt(m)}')
    print(f'  AUC выступа {stats.roc_auc(yb, np.nan_to_num(dist, nan=0.0)):.3f}, '
          f'вида кадра {m["appearance_auc"]:.3f}, '
          f'Brier {m["brier"]:.3f} (доля нарушений {m["base_rate"]:.2f})')

    thr = {**DEFAULTS, 'area_lo': round(lo_a, 1), 'area_hi': round(hi_a, 1),
           'corridor_percentile_lo': best['q_lo'],
           'corridor_percentile_hi': best['q_hi'],
           'score': best['score']}
    if best['score'] == 'relative':
        r = d.relative.values.astype(float)
        rlo, rhi = corridor_asym(
            r[reliable.values & (y_all == 0) & np.isfinite(r)],
            best['q_lo'], best['q_hi'])
        thr['relative_lo'] = round(rlo, 4)
        thr['relative_hi'] = round(rhi, 4)
    elif best['score'] == 'area_over_shaft2':
        v = d.area_over_shaft2.values.astype(float)
        vlo, vhi = corridor_asym(
            v[reliable.values & (y_all == 0) & np.isfinite(v)],
            best['q_lo'], best['q_hi'])
        thr['aoss_lo'] = round(vlo, 4)
        thr['aoss_hi'] = round(vhi, 4)

    full = np.array([
        corridor_distance(a, thr) if (np.isfinite(a) and not u) else np.nan
        for a, u in zip(area, d.uncertain.fillna(True).values)])
    pl = prob.fit(y_all, full)
    if pl is not None:
        thr['probability'] = pl
    print('коридор нормы для поставки: %.0f-%.0f кв. мм (q_lo=%.0f q_hi=%.0f)' % (
        lo_a, hi_a, best['q_lo'], best['q_hi']))

    if not args.no_save:
        THRESHOLDS.write_text(json.dumps(thr, ensure_ascii=False, indent=2),
                              encoding='utf-8')
        METRICS.write_text(json.dumps(
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()},
            ensure_ascii=False, indent=2, default=float), encoding='utf-8')
        d.to_csv(HERE / 'measurements.csv', index=False, encoding='utf-8')
        print(f'сохранено: {THRESHOLDS.name}, {METRICS.name}, measurements.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
