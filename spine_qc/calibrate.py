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
from . import probability as prob
from .criteria import (DEFAULTS, artifact_structures, axis, axis_is_straight,
                       axis_decision, iliac_masses, top_margin_mm)
from .geometry import spine_column

HERE = Path(__file__).resolve().parent
THRESHOLDS = HERE / 'thresholds.json'
METRICS = HERE / 'metrics.json'


def _axis_fused(args) -> int:
    """Вложенный OOF-подбор дисперсий исправленного угла по модели точек."""
    from dxa_qc.spine_shape import (curve_descriptors, fused_tilt,
                                    source_variances, tilt_sources)
    z = np.load(args.axis_oof, allow_pickle=True)
    names = list(z['names'])
    thresholds = np.asarray(z['thresholds'])
    m = pd.read_csv(HERE / 'measurements.csv').set_index('study')
    rows = []
    for i, region in enumerate(z['regions']):
        study = str(z['studies'][i])
        if str(region) != 'spine' or study not in m.index:
            continue
        th = thresholds[i] if thresholds.ndim == 2 else thresholds
        visible = np.asarray(z['conf'][i]) >= th
        contour = m.loc[study]
        contour_dict = {'angle_deg': contour.angle_deg,
                        'curvature_mm': contour.curvature_mm}
        src = tilt_sources(z['pred'][i], visible, names, .6, contour_dict,
                           z['conf'][i])
        curve = curve_descriptors(z['pred'][i], visible, names, .6)
        if not curve.get('available'):
            continue
        rows.append({'study': study, 'fold': int(contour.fold),
                     'y': int(contour.y_axis),
                     'scoliosis': 'колиоз' in str(contour.comment).lower(),
                     'src': src, 'curve': curve,
                     'contour_curvature_mm': float(contour.curvature_mm)})
    if not rows:
        raise ValueError(f'{args.axis_oof}: нет сопоставимых spine-строк')

    grid = []
    for bc in (.5, 1., 2.):
        for be in (.5, 1., 2.):
            for bk in (.5, 1., 2.):
                for k in (0., .5, 1.):
                    grid.append({'b_c': bc, 'b_e': be, 'b_k': bk,
                                 'b_r': 1., 'tau': 1.5, 'uncertainty_k': k})

    def predict(indices, params):
        pred, state, angle = [], [], []
        for i in indices:
            row = rows[i]
            variance = source_variances(
                row['src'], row['curve'],
                {**params, 'contour_curvature_mm': row['contour_curvature_mm']})
            fused = fused_tilt(row['src'], variance)
            dec = axis_decision(fused['theta_fused_deg'], fused['sigma_deg'],
                                fused['Q'], fused['n_sources'],
                                {'uncertainty_k': params['uncertainty_k']})
            pred.append(dec['state'] == 'unsuitable')
            state.append(dec['state'])
            angle.append(fused['theta_fused_deg'])
        return np.asarray(pred, int), np.asarray(state), np.asarray(angle, float)

    folds = sorted({r['fold'] for r in rows})
    oof_pred = np.zeros(len(rows), int)
    oof_state = np.full(len(rows), '', object)
    chosen = []
    for fold in folds:
        train = np.array([i for i, r in enumerate(rows) if r['fold'] != fold])
        test = np.array([i for i, r in enumerate(rows) if r['fold'] == fold])
        best = None
        ytr = np.asarray([rows[i]['y'] for i in train])
        for params in grid:
            pred, state, _ = predict(train, params)
            c = stats.confusion(ytr, pred)
            und = float(np.mean(state == 'undetermined'))
            score = (c['f1'], -c['fn'], -und)
            if und <= .15 and (best is None or score > best[0]):
                best = (score, params)
        if best is None:
            best = ((-1, 0, 0), {'b_c': 1., 'b_e': 1., 'b_k': 1.,
                                  'b_r': 1., 'tau': 1.5, 'uncertainty_k': 0.})
        chosen.append(best[1])
        oof_pred[test], oof_state[test], _ = predict(test, best[1])

    y = np.asarray([r['y'] for r in rows])
    scoliosis = np.asarray([r['scoliosis'] for r in rows])
    c = stats.confusion(y, oof_pred)
    center_bad = np.asarray([
        abs(float(r['src']['theta_centers_deg'])) > 5
        if r['src']['theta_centers_deg'] is not None else False for r in rows
    ])
    full_idx = np.arange(len(rows))
    runtime_best = None
    for params in grid:
        pred, state, _ = predict(full_idx, params)
        cm = stats.confusion(y, pred)
        und = float(np.mean(state == 'undetermined'))
        score = (cm['f1'], -cm['fn'], -und)
        if und <= .15 and (runtime_best is None or score > runtime_best[0]):
            runtime_best = (score, params)
    runtime_params = (runtime_best[1] if runtime_best else
                      {'b_c': 1., 'b_e': 1., 'b_k': 1., 'b_r': 1.,
                       'tau': 1.5, 'uncertainty_k': 0.})
    result = {
        **{key: c[key] for key in ('tp', 'fp', 'fn', 'sensitivity', 'specificity', 'f1')},
        'n': len(rows), 'undetermined_fraction': float(np.mean(oof_state == 'undetermined')),
        'scoliosis_fp_centers': int(np.sum(center_bad & scoliosis & (y == 0))),
        'scoliosis_fp_fused': int(np.sum((oof_pred == 1) & scoliosis & (y == 0))),
        'fold_params': chosen,
        'runtime_params': runtime_params,
    }
    print('OOF исправленного угла:', json.dumps(result, ensure_ascii=False, indent=2))
    passed = (result['f1'] > .33 and result['undetermined_fraction'] <= .15
              and result['fn'] <= 1
              and result['scoliosis_fp_fused']
              <= .5 * max(result['scoliosis_fp_centers'], 1))
    result['gate_passed'] = bool(passed)
    out = Path(args.axis_out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'G2: {"ПРОЙДЕН" if passed else "НЕ ПРОЙДЕН"}; {out}')
    return 0

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
# Ось: geometry_and_straight — угол выше порога только на достаточно прямом
# треке (сколиоз эксперт нарушением оси не считает).
#
# Довод в пользу геометрии сверх метрики: её вердикт оператор проверяет
# глазами за секунду («гребни не визуализированы»), а вердикт модели вида
# кадра — нет.
RULES = {
    'axis':      {'geometry': 'angle_abs', 'rule': 'geometry_and_straight',
                  'key': 'axis_deg'},
    'position':  {'geometry': 'iliac_neg', 'rule': 'geometry', 'key': 'iliac_area'},
    'artifacts': {'geometry': 'score', 'rule': 'geometry', 'key': 'artifact_score'},
}

# Сетка OOF для порогов «прямой трек» (ось). Основной рычаг — согласие
# двух углов; изгиб/знаки — добор FP, грубая сетка.
AXIS_CURVED_GRID = (6.0, 8.0, 10.0, 12.0, 99.0)
AXIS_SIGN_GRID = (2, 3, 4, 99)
AXIS_HALF_GRID = (2.0, 4.0, 6.0, 99.0)


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
            'disagreement_deg': a.get('disagreement_deg'),
            'sign_changes': a.get('sign_changes'),
            'half_angle_diff_deg': a.get('half_angle_diff_deg'),
            'iliac_area': il['iliac_area'],
            'top_margin_mm': top_margin_mm(col),
            'artifact_score': ar['score'],
            'artifact_len_mm': ar['length_mm'],
            'y_position': r.y_position, 'y_axis': r.y_axis, 'y_artifacts': r.y_artifacts,
            'comment': r.comment,
        })
        vecs.append(app.vector(img))
    return pd.DataFrame(rows), np.stack(vecs)


def geometry_score(name: str, d: pd.DataFrame,
                   top_margin_min_mm: float = 0.0) -> np.ndarray:
    """Геометрическая оценка в виде «больше = вероятнее нарушение»."""
    how = RULES[name]['geometry']
    if how == 'angle_abs':
        return np.abs(np.nan_to_num(d.angle_deg.values.astype(float)))
    if how == 'iliac_neg':
        # Составной скор: гребни + (опционально) недостаток запаса сверху.
        iliac = -np.nan_to_num(d.iliac_area.values.astype(float))
        if top_margin_min_mm and top_margin_min_mm > 0 and 'top_margin_mm' in d.columns:
            top = d.top_margin_mm.values.astype(float)
            short = np.where(np.isfinite(top) & (top < top_margin_min_mm), 1.0, 0.0)
            # Добавка сдвигает скор вверх при коротком верхе, не ломая порядок
            # по iliac на кадрах с нормальным запасом.
            return iliac + short
        return iliac
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


def _straight_mask(d: pd.DataFrame, thr: dict) -> np.ndarray:
    """Кадры с согласованным и достаточно прямым треком."""
    out = np.zeros(len(d), bool)
    for i, r in enumerate(d.itertuples()):
        c = r.curvature_mm
        if c is None or (isinstance(c, float) and not np.isfinite(c)):
            out[i] = True
            continue
        disagree = getattr(r, 'disagreement_deg', None)
        if disagree is None or (isinstance(disagree, float) and not np.isfinite(disagree)):
            disagree = abs(float(r.angle_deg or 0) - float(r.angle_edges_deg or 0))
        out[i] = axis_is_straight(
            float(c), int(r.sign_changes or 0), float(r.half_angle_diff_deg or 0.0),
            thr, disagreement_deg=float(disagree))
    return out


def best_top_margin(d: pd.DataFrame) -> float:
    """OOF-подбор top_margin_min_mm для укладки; 0 = не использовать верх."""
    y = d.y_position.values.astype(float)
    iliac = d.iliac_area.values.astype(float)
    top = d.top_margin_mm.values.astype(float)
    folds = sorted(d.fold.unique())
    grid = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 20.0)
    best_t, best_f1, best_fp = 0.0, -1.0, 10**9
    for tmm in grid:
        pred = np.zeros(len(d), int)
        for f in folds:
            te = (d.fold == f).values
            tr = ~te & np.isfinite(y)
            thr_score = best_threshold(y[tr], -np.nan_to_num(iliac[tr]))
            iliac_bad = (-np.nan_to_num(iliac)) > thr_score
            top_bad = (tmm > 0) & np.isfinite(top) & (top < tmm)
            pred[te] = (iliac_bad[te] | top_bad[te]).astype(int)
        ok = np.isfinite(y)
        c = stats.confusion(y[ok].astype(int), pred[ok])
        better = (c['f1'] > best_f1 + 1e-9
                  or (abs(c['f1'] - best_f1) <= 1e-9 and c['fp'] < best_fp))
        if better:
            best_t, best_f1, best_fp = tmm, c['f1'], c['fp']
    print(f'  top_margin_min_mm OOF: {best_t} (F1={best_f1:.2f}, fp={best_fp})')
    return float(best_t)


def best_axis_joint(d: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> tuple[dict, np.ndarray]:
    """Совместный OOF-подбор порога угла и согласия двух конструкций оси.

    Возвращает пороги для поставки и бинарный вердикт на кадрах `te`.
    """
    y = d.y_axis.values.astype(float)
    ang = np.abs(np.nan_to_num(d.angle_deg.values.astype(float)))
    base = {
        'axis_deg': DEFAULTS['axis_deg'],
        'axis_disagree_deg': 2.0,
        'axis_curved_max_mm': 99.0,
        'axis_sign_max': 99,
        'axis_half_max_deg': 99.0,
    }
    best = {'f1': -1.0, 'fp': 10**9, 'thr': dict(base), 'ath': base['axis_deg']}
    ok = tr & np.isfinite(y)
    grid = np.unique(np.percentile(ang[ok], np.arange(40.0, 99.6, 1.0))) if ok.any() else [4.0]
    if ok.sum() >= 5 and np.nansum(y[ok]) > 0:
        for ath in grid:
            for ddeg in (1.0, 1.5, 2.0, 2.5, 3.0):
                t = {**base, 'axis_deg': float(ath), 'axis_disagree_deg': ddeg}
                straight = _straight_mask(d, t)
                pred = ((ang > ath) & straight).astype(int)
                c = stats.confusion(y[ok].astype(int), pred[ok])
                better = (c['f1'] > best['f1'] + 1e-9
                          or (abs(c['f1'] - best['f1']) <= 1e-9 and c['fp'] < best['fp']))
                if better:
                    best = {'f1': c['f1'], 'fp': c['fp'], 'thr': t, 'ath': float(ath)}
    t = best['thr']
    straight = _straight_mask(d, t)
    pred_te = ((ang[te] > t['axis_deg']) & straight[te]).astype(int) if te.any() else np.zeros(0, int)
    return t, pred_te


def out_of_fold(d: pd.DataFrame, X: np.ndarray) -> tuple[dict, dict, dict, dict, dict]:
    """OOF-предсказания: порог и модель вида учатся только на обучающих фолдах.

    Возвращает вердикты по выбранному правилу, вероятности модели вида,
    вердикты по ВСЕМ правилам и подобранные на каждом фолде пороги «прямого
    трека» для оси (для поставки усредняются / пересчитываются на всей выборке).
    """
    folds = sorted(d.fold.unique())
    oof_pred = {k: np.zeros(len(d), int) for k in RULES}
    oof_app = {k: np.full(len(d), np.nan) for k in app.TARGETS}
    oof_prob = {k: np.full(len(d), np.nan) for k in RULES}
    per_rule = {k: {} for k in RULES}
    oof_straight_thr = []

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
            ang_thr = best_threshold(y[tr], g[tr])
            geo_bad[name] = g[te] > ang_thr
            pl = prob.fit(y[tr], g[tr])                  # калибровка только по train
            if pl is not None:
                oof_prob[name][te] = [prob.apply(pl, v) for v in g[te]]
            variants = {'geometry': geo_bad[name].astype(int)}
            if name == 'axis':
                # Совместный подбор порога угла и согласия двух конструкций:
                # раздельный (сначала угол, потом straight) ронял OOF F1.
                st_thr, joint_pred_te = best_axis_joint(d, tr, te)
                oof_straight_thr.append(st_thr)
                variants['geometry_and_straight'] = joint_pred_te.astype(int)
            if name in app_bad:
                variants['appearance'] = app_bad[name].astype(int)
                variants['and'] = (geo_bad[name] & app_bad[name]).astype(int)
                variants['or'] = (geo_bad[name] | app_bad[name]).astype(int)
            for rn, pred in variants.items():
                per_rule[name].setdefault(rn, np.zeros(len(d), int))[te] = pred
            oof_pred[name][te] = variants[cfg['rule']]
    supply_straight, _ = best_axis_joint(d, np.ones(len(d), bool), np.zeros(len(d), bool))
    return oof_pred, oof_app, per_rule, oof_prob, supply_straight


def report(d: pd.DataFrame, oof_pred: dict, oof_app: dict,
           oof_prob: dict | None = None) -> dict:
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
        if oof_prob is not None and np.isfinite(oof_prob[name][ok]).any():
            pr = oof_prob[name][ok]
            m['brier'] = round(prob.brier(y[ok], pr), 4)
            m['prob_mean'] = round(float(np.nanmean(pr)), 4)
            m['base_rate'] = round(float(np.nanmean(y[ok])), 4)
        m['rule'] = RULES[name]['rule']
        m['positives'] = int(np.nansum(y))
        out[name] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Калибровка критериев позвоночника')
    ap.add_argument('--no-save', action='store_true', help='не писать модель и пороги')
    ap.add_argument('--sweep', action='store_true', help='таблица всех правил свёртки')
    ap.add_argument('--probability-only', action='store_true',
                    help='перекалибровать только вероятности; пороги и модель вида не трогать')
    ap.add_argument('--axis-fused', action='store_true',
                    help='оценить исправленный угол по OOF модели точек')
    ap.add_argument('--axis-oof', default='runs/keypoints_crops/oof.npz')
    ap.add_argument('--axis-out', default='runs/keypoints_crops/metrics_axis_fused.json')
    args = ap.parse_args()
    if args.axis_fused:
        return _axis_fused(args)

    df = trainset.frames('spine')
    print(f'{len(df)} кадров позвоночника, {df.study.nunique()} исследований')
    d, X = measure(df)

    oof_pred, oof_app, per_rule, oof_prob, supply_straight = out_of_fold(d, X)
    metrics = report(d, oof_pred, oof_app, oof_prob)
    if args.probability_only:
        return _save_probability(d, metrics, dry=args.no_save)

    print('\nOOF (порог и модель вида учились только на обучающих фолдах):')
    for name, m in metrics.items():
        print(f'  {name:10} {stats.fmt(m)}')
        extra = f"  AUC геометрии {m['geometry_auc']:.3f}"
        if 'brier' in m:
            extra += f", Brier {m['brier']:.3f} (доля нарушений {m['base_rate']:.2f})"
        if 'appearance_auc' in m:
            extra += f", вида кадра {m['appearance_auc']:.3f}, корр. рангов {m['rank_corr']:.2f}"
        print(f"  {'':10} правило «{m['rule']}»,{extra}")
    print('  пороги прямого трека (поставка):', supply_straight)

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

    # Порог запаса сверху для укладки: OOF-сетка, 0 = выключен (как раньше).
    top_thr = best_top_margin(d)
    # пороги для поставки — на всей выборке
    thr = dict(DEFAULTS)
    for name, cfg in RULES.items():
        y = d[f'y_{name}'].values.astype(float)
        g = geometry_score(name, d, top_margin_min_mm=top_thr if name == 'position' else 0.0)
        thr[cfg['key']] = round(float(best_threshold(y, g)), 4)
    thr['iliac_area'] = round(-thr['iliac_area'], 4)      # счёт был со знаком минус
    thr['top_margin_min_mm'] = float(top_thr)
    if 'axis_deg' in supply_straight:
        thr['axis_deg'] = round(float(supply_straight['axis_deg']), 4)
    thr['axis_disagree_deg'] = float(supply_straight['axis_disagree_deg'])
    thr['axis_curved_max_mm'] = float(supply_straight['axis_curved_max_mm'])
    thr['axis_sign_max'] = int(supply_straight['axis_sign_max'])
    thr['axis_half_max_deg'] = float(supply_straight['axis_half_max_deg'])
    app_thr = {}
    for name in app.TARGETS:
        y = d[f'y_{name}'].values.astype(float)
        p = oof_app[name]
        if np.isfinite(p).all():
            app_thr[name] = round(float(best_threshold(y, p)), 4)
    thr['appearance_by_criterion'] = app_thr
    thr['probability'] = {}
    for name in RULES:
        y = d[f'y_{name}'].values.astype(float)
        pl = prob.fit(y, geometry_score(name, d))
        if pl is not None:
            thr['probability'][name] = pl
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


def _save_probability(d: pd.DataFrame, metrics: dict, dry: bool = False) -> int:
    """Только калибровка Платта: вероятности в thresholds.json, Brier в metrics.json.

    Пороги вердикта и модель вида кадра остаются как были: их пересчёт в
    другом окружении (версия scikit-learn) дал бы другой model.joblib, а
    вероятность от них не зависит.
    """
    thr = json.loads(THRESHOLDS.read_text(encoding='utf-8'))
    saved = json.loads(METRICS.read_text(encoding='utf-8'))
    thr['probability'] = {}
    for name in RULES:
        y = d[f'y_{name}'].values.astype(float)
        pl = prob.fit(y, geometry_score(name, d))
        if pl is not None:
            thr['probability'][name] = pl
        for k in ('brier', 'prob_mean', 'base_rate'):
            if k in metrics[name]:
                saved[name][k] = metrics[name][k]
        print(f"  {name:10} Платт {pl}, OOF Brier {metrics[name].get('brier')} "
              f"(доля нарушений {metrics[name].get('base_rate')})")
    if not dry:
        THRESHOLDS.write_text(json.dumps(thr, ensure_ascii=False, indent=2), encoding='utf-8')
        METRICS.write_text(json.dumps(saved, ensure_ascii=False, indent=2, default=float),
                           encoding='utf-8')
        print(f'сохранено: {THRESHOLDS.name} (probability), {METRICS.name} (brier)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
