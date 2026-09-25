# -*- coding: utf-8 -*-
"""Калибровка порогов и честная оценка критериев позвоночника.

    python -m spine_qc.calibrate            # всё: model.joblib, thresholds.json, metrics.json
    python -m spine_qc.calibrate --no-save  # только напечатать метрики

Честность здесь — в том, что и порог, и модель вида кадра подбираются ТОЛЬКО
на обучающих фолдах, а метрика считается на отложенном. Порог, подобранный по
той же выборке, на которой потом меряют F1, завышает результат тем сильнее,
чем меньше положительных примеров — а их тут 6, 10 и 17.

В поставку идут модель и порог, обученные на всей выборке; в metrics.json —
оценка, полученная out-of-fold. Это разные числа, и подменять одно другим
нельзя.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stats
import trainset

from . import appearance as app
from .criteria import DEFAULTS, artifact_structures, axis, iliac_masses
from .geometry import spine_column

HERE = Path(__file__).resolve().parent
THRESHOLDS = HERE / 'thresholds.json'
METRICS = HERE / 'metrics.json'

# Как критерий превращается в вердикт.
#
# Вердикт всегда выносит геометрическое измерение, вторая оценка идёт в
# уверенность. Правило выбиралось не из удобства: сначала по метрикам,
# подобранным на всей выборке, лучшим казалось согласие обеих оценок для
# укладки (F1 0.73). Честная проверка out-of-fold это опровергла — согласие
# ловит 1 нарушение из 6 (F1 0.29), потому что порог, подобранный на фолде с
# одним-двумя положительными примерами, неустойчив. Таблица всех правил
# печатается по `--sweep` и лежит в metrics.json.
#
# Довод в пользу геометрии сверх метрики: её вердикт оператор проверяет
# глазами за секунду («гребни не визуализированы»), а вердикт модели вида
# кадра — нет.
RULES = {
    'axis':      {'geometry': 'angle_abs', 'rule': 'geometry', 'key': 'axis_deg'},
    'position':  {'geometry': 'iliac_neg', 'rule': 'geometry', 'key': 'iliac_area'},
    'artifacts': {'geometry': 'score', 'rule': 'geometry', 'key': 'artifact_score'},
}


def measure(df: pd.DataFrame) -> pd.DataFrame:
    """Геометрические измерения и векторы вида для всех кадров."""
    rows, vecs = [], []
    for r in df.itertuples():
        img = trainset.read(r.rel_path)
        col = spine_column(img)
        a = axis(col)
        il = iliac_masses(col)
        ar = artifact_structures(col)
        rows.append({
            'study': r.study, 'rel_path': r.rel_path, 'fold': r.fold,
            'angle_deg': a['angle_deg'], 'angle_edges_deg': a['angle_edges_deg'],
            'curvature_mm': a['curvature_mm'],
            'iliac_area': il['iliac_area'], 'artifact_score': ar['score'],
            'artifact_len_mm': ar['length_mm'],
            'y_position': r.y_position, 'y_axis': r.y_axis, 'y_artifacts': r.y_artifacts,
            'comment': r.comment,
        })
        vecs.append(app.vector(img))
    return pd.DataFrame(rows), np.stack(vecs)


def geometry_score(name: str, d: pd.DataFrame) -> np.ndarray:
    """Геометрическая оценка в виде «больше = вероятнее нарушение»."""
    how = RULES[name]['geometry']
    if how == 'angle_abs':
        return np.abs(np.nan_to_num(d.angle_deg.values.astype(float)))
    if how == 'iliac_neg':
        return -np.nan_to_num(d.iliac_area.values.astype(float))
    return np.nan_to_num(d.artifact_score.values.astype(float))


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Порог, максимизирующий F1. При ничьей берётся более строгий (меньше ложных)."""
    ok = np.isfinite(y)
    y, score = y[ok].astype(int), score[ok]
    if y.sum() == 0 or len(np.unique(score)) < 2:
        return float(np.median(score))
    grid = np.unique(np.percentile(score, np.arange(20.0, 99.6, 0.5)))
    best, thr = -1.0, float(grid[0])
    for t in grid:
        f1 = stats.confusion(y, (score > t).astype(int))['f1']
        if f1 > best + 1e-9:
            best, thr = f1, float(t)
    return thr


def combine(rule: str, geo_bad: np.ndarray, app_bad: np.ndarray | None) -> np.ndarray:
    if app_bad is None or rule == 'geometry':
        return geo_bad.astype(int)
    if rule == 'and':
        return (geo_bad & app_bad).astype(int)
    if rule == 'or':
        return (geo_bad | app_bad).astype(int)
    raise ValueError(rule)


def out_of_fold(d: pd.DataFrame, X: np.ndarray) -> tuple[dict, dict, dict]:
    """OOF-предсказания: порог и модель вида учатся только на обучающих фолдах.

    Возвращает вердикты по выбранному правилу, вероятности модели вида и
    вердикты по ВСЕМ правилам — последние нужны, чтобы выбор правила можно
    было перепроверить, а не принимать на веру.
    """
    folds = sorted(d.fold.unique())
    oof_pred = {k: np.zeros(len(d), int) for k in RULES}
    oof_app = {k: np.full(len(d), np.nan) for k in app.TARGETS}
    per_rule = {k: {} for k in RULES}

    for f in folds:
        te = (d.fold == f).values
        tr = ~te
        geo_bad, app_bad = {}, {}
        for name in app.TARGETS:
            y = d[f'y_{name}'].values.astype(float)
            ok = tr & np.isfinite(y)
            if ok.sum() < 10 or np.nansum(y[ok]) == 0:
                continue
            m = app.make_model()
            m.fit(X[ok], y[ok].astype(int))
            p_te = m.predict_proba(X[te])[:, 1]
            oof_app[name][te] = p_te
            app_bad[name] = p_te > best_threshold(y[tr], m.predict_proba(X[tr])[:, 1])

        for name, cfg in RULES.items():
            y = d[f'y_{name}'].values.astype(float)
            g = geometry_score(name, d)
            geo_bad[name] = g[te] > best_threshold(y[tr], g[tr])
            variants = {'geometry': geo_bad[name].astype(int)}
            if name in app_bad:
                variants['appearance'] = app_bad[name].astype(int)
                variants['and'] = (geo_bad[name] & app_bad[name]).astype(int)
                variants['or'] = (geo_bad[name] | app_bad[name]).astype(int)
            for rn, pred in variants.items():
                per_rule[name].setdefault(rn, np.zeros(len(d), int))[te] = pred
            oof_pred[name][te] = variants[cfg['rule']]
    return oof_pred, oof_app, per_rule


def report(d: pd.DataFrame, oof_pred: dict, oof_app: dict) -> dict:
    """Метрики с 95% ДИ, бутстрэп по исследованиям."""
    out = {}
    for name in RULES:
        y = d[f'y_{name}'].values.astype(float)
        ok = np.isfinite(y)
        g = geometry_score(name, d)
        m = stats.evaluate(y[ok].astype(int), oof_pred[name][ok], g[ok], d.study.values[ok])
        m['geometry_auc'] = stats.roc_auc(y[ok].astype(int), g[ok])
        if name in oof_app and np.isfinite(oof_app[name][ok]).all():
            m['appearance_auc'] = stats.roc_auc(y[ok].astype(int), oof_app[name][ok])
            m['rank_corr'] = float(pd.Series(g[ok]).corr(
                pd.Series(oof_app[name][ok]), method='spearman'))
        m['rule'] = RULES[name]['rule']
        m['positives'] = int(np.nansum(y))
        out[name] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Калибровка критериев позвоночника')
    ap.add_argument('--no-save', action='store_true', help='не писать модель и пороги')
    ap.add_argument('--sweep', action='store_true', help='таблица всех правил свёртки')
    args = ap.parse_args()

    df = trainset.frames('spine')
    print(f'{len(df)} кадров позвоночника, {df.study.nunique()} исследований')
    d, X = measure(df)

    oof_pred, oof_app, per_rule = out_of_fold(d, X)
    metrics = report(d, oof_pred, oof_app)

    print('\nOOF (порог и модель вида учились только на обучающих фолдах):')
    for name, m in metrics.items():
        print(f'  {name:10} {stats.fmt(m)}')
        extra = f"  AUC геометрии {m['geometry_auc']:.3f}"
        if 'appearance_auc' in m:
            extra += f", вида кадра {m['appearance_auc']:.3f}, корр. рангов {m['rank_corr']:.2f}"
        print(f"  {'':10} правило «{m['rule']}»,{extra}")

    sweep = {}
    for name in RULES:
        y = d[f'y_{name}'].values.astype(float)
        ok = np.isfinite(y)
        sweep[name] = {rn: stats.confusion(y[ok].astype(int), pred[ok])
                       for rn, pred in per_rule[name].items()}
    for name, m in metrics.items():
        m['rule_sweep'] = {rn: {k: c[k] for k in ('tp', 'fp', 'fn', 'sensitivity',
                                                  'specificity', 'f1')}
                           for rn, c in sweep[name].items()}
    if args.sweep:
        print('\nвсе правила свёртки (OOF) — на чём основан выбор:')
        for name, rules in sweep.items():
            print(f'  {name} ({metrics[name]["positives"]} положительных)')
            for rn, c in rules.items():
                mark = ' <- выбрано' if rn == RULES[name]['rule'] else ''
                print(f'      {rn:11} tp={c["tp"]:2d} fp={c["fp"]:2d} fn={c["fn"]:2d} '
                      f'sens={c["sensitivity"]:.2f} spec={c["specificity"]:.2f} '
                      f'F1={c["f1"]:.2f}{mark}')

    # пороги для поставки — на всей выборке
    thr = dict(DEFAULTS)
    for name, cfg in RULES.items():
        y = d[f'y_{name}'].values.astype(float)
        thr[cfg['key']] = round(float(best_threshold(y, geometry_score(name, d))), 4)
    thr['iliac_area'] = round(-thr['iliac_area'], 4)      # счёт был со знаком минус
    app_thr = {}
    for name in app.TARGETS:
        y = d[f'y_{name}'].values.astype(float)
        p = oof_app[name]
        if np.isfinite(p).all():
            app_thr[name] = round(float(best_threshold(y, p)), 4)
    thr['appearance_by_criterion'] = app_thr
    print('\nпороги для поставки:', json.dumps(thr, ensure_ascii=False))

    if not args.no_save:
        auc = {k: float(v.get('appearance_auc', float('nan'))) for k, v in metrics.items()}
        app.train(X, {k: d[f'y_{k}'].values.astype(float) for k in app.TARGETS}, auc)
        THRESHOLDS.write_text(json.dumps(thr, ensure_ascii=False, indent=2), encoding='utf-8')
        clean = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                     for kk, vv in v.items()} for k, v in metrics.items()}
        METRICS.write_text(json.dumps(clean, ensure_ascii=False, indent=2, default=float),
                           encoding='utf-8')
        d.to_csv(HERE / 'measurements.csv', index=False, encoding='utf-8')
        print(f'сохранено: {app.MODEL_PATH.name}, {THRESHOLDS.name}, {METRICS.name}, measurements.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
