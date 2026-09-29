# -*- coding: utf-8 -*-
"""Сквозная оценка сервиса на обучающем наборе (ТЗ 8.4).

    python -m service.evaluate                  # метрики + service/metrics.json
    python -m service.evaluate --quick          # без бутстрэпа (быстро, без ДИ)

Считает то, что перечислено в ТЗ: чувствительность, специфичность,
сбалансированную точность, F1, ROC-AUC и PR-AUC — отдельно по каждой
анатомической области и каждому типу нарушения, плюс общий бинарный класс
«качественное / есть нарушение». Интервалы 95%: Уилсон для долей, бутстрэп по
ИССЛЕДОВАНИЯМ для F1 и AUC (см. stats.py).

**Про какие числа здесь речь.** Пороги критериев подобраны на этой же
выборке, поэтому метрики ниже — оценка сверху. Честные out-of-fold числа, где
порог и модель учились только на обучающих фолдах, лежат в
`spine_qc/metrics.json` и `hip_rotation/metrics.json` и печатаются здесь же
для сравнения. Разница между двумя таблицами — цена подгонки порога на выборке
с 6-36 положительными примерами, и её видно.

**Откуда берётся score для ROC-AUC и PR-AUC.** Для каждого из пяти критериев
это своя непрерывная величина до порога (угол оси в градусах, счёт top-hat,
отклонение от коридора нормы и т.п.), а не бинарное решение конвейера —
ранговая AUC на 0/1-скоре математически совпадает со сбалансированной
точностью и не проверяет ничего сверх неё. Для агрегатов «по области» и
«в целом» критерии в разных единицах сводятся через калиброванные
вероятности: у каждого критерия своя логистическая калибровка Платта, свод —
`quality_probability = 1 - П(1 - p)`, та же колонка, что в отчёте. Кадр без
вероятности (ни один критерий не измерился) откатывается на вердикт.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stats
import trainset

HERE = Path(__file__).resolve().parent
METRICS = HERE / 'metrics.json'

# Тип нарушения -> колонка эксперта для кадра этой области.
CRITERIA = {
    'spine_position':  ('spine', 'y_position'),
    'spine_axis':      ('spine', 'y_axis'),
    'spine_artifacts': ('spine', 'y_artifacts'),
    'hip_rotation':    ('hip', 'y_rotation'),
    'hip_roi':         ('hip', 'y_roi'),
}


def _risk_score(vtype: str, details: dict) -> float:
    """Непрерывная величина «насколько похоже на нарушение» до применения порога.

    Знак всегда один: больше -> ближе к нарушению. Нужна только для ROC-AUC и
    PR-AUC (см. docstring модуля) — сами вердикты этой функцией не считаются,
    они уже есть в `details` от конвейера.
    """
    try:
        if vtype == 'spine_axis':
            return abs(details['spine']['axis']['angle_deg'])
        if vtype == 'spine_position':
            return -details['spine']['position']['iliac_area']
        # Где вердикт выносит свод с сетью (cnn_qc), скор — вероятность свода:
        # иначе AUC мерила бы не то, по чему принято решение.
        if vtype == 'spine_artifacts':
            a = details['spine']['artifacts']
            return a['probability'] if a.get('p_cnn') is not None else a['score']
        if vtype == 'hip_rotation':
            r = details['hip_rotation']
            return r['probability'] if r.get('p_cnn') is not None else r['corridor_distance']
        if vtype == 'hip_roi':
            from hip_roi.probability import score
            v = score(details['hip_roi'])
            return float('nan') if v is None else v
    except (KeyError, TypeError):
        pass
    return float('nan')


def run(hip_method: str = 'kit2', keypoints: bool = False,
        keypoints_oof: bool = False, cnn: bool = True,
        roi_rule: str = 'scan_length') -> pd.DataFrame:
    """Прогон конвейера по всем уникальным кадрам обучающего набора.

    keypoints=True считает ось позвоночника и отступы ROI моделью ключевых
    точек. Сравнивать два прогона осмысленно: набор кадров и метки одни и те
    же, меняется только измеритель.

    keypoints_oof=True — то же, но честно: кадр считает модель только того
    фолда, который этот кадр в обучении не видел. Ансамбль всех фолдов на
    обучающем наборе — оценка на данных обучения: разметка точек есть у всех
    242 кадров pack, и каждый из них видели четыре модели из пяти.

    cnn=True — свод с нейросетью (cnn_qc) как в рабочем сервисе, но вероятность
    сети — out-of-fold (cnn_qc/<критерий>/oof.csv): поставляемая модель обучена
    на всех этих кадрах, и её оценка на них ничего бы не значила. cnn=False —
    одна геометрия, базовая линия до сети.
    """
    from .keypoints_backend import KeypointBackend
    from .pipeline import Analyzer

    df = trainset.frames()
    a = Analyzer.load(hip_method=hip_method, keypoints=keypoints or keypoints_oof,
                      cnn=False, cnn_oof=cnn, roi_rule=roi_rule)
    by_fold = {}
    if keypoints_oof:
        for k in sorted(df.fold.unique()):
            by_fold[int(k)] = KeypointBackend(folds=[int(k)])
        missing = [k for k, b in by_fold.items() if not b.available]
        if missing:
            raise SystemExit(f'для OOF нужны веса всех фолдов, нет: {missing}')
    rows = []
    for r in df.itertuples():
        px = trainset.read(r.rel_path)
        if by_fold:
            a.keypoints = by_fold[int(r.fold)]
        import time
        t0 = time.perf_counter()
        try:
            res = a.analyze_pixels(px)
            status, err = 'Success', ''
        except Exception as e:
            res = {'anatomical_region': 'unknown', 'violations': [], 'flags': [],
                   'quality_probability': None}
            status, err = 'Failure', f'{type(e).__name__}: {e}'
        rows.append({
            'study': r.study, 'rel_path': r.rel_path, 'expert_region': r.label,
            'region': res['anatomical_region'], 'violations': res['violations'],
            'flags': ';'.join(res['flags']), 'status': status, 'error': err,
            'seconds': time.perf_counter() - t0, 'details': res.get('details', {}),
            'quality_probability': res.get('quality_probability'),
            **{f'y_{c}': getattr(r, f'y_{c}') for c in trainset.CRITERIA},
        })
    return pd.DataFrame(rows)


def _binary_score(sub: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Скор бинарного класса: quality_probability, без неё — вердикт."""
    if 'quality_probability' not in sub:
        return pred.astype(float)
    q = pd.to_numeric(sub.quality_probability, errors='coerce').values.astype(float)
    return np.where(np.isfinite(q), q, pred.astype(float))


def evaluate(d: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Метрики по критериям, по областям и в целом."""
    out: dict = {'per_violation': {}, 'per_region': {}, 'overall': {}}

    for vtype, (region, col) in CRITERIA.items():
        sel = d.expert_region.isin(('lh', 'rh')) if region == 'hip' else (d.expert_region == region)
        sub = d[sel & d[col].notna()]
        if sub.empty:
            continue
        y = sub[col].values.astype(int)
        pred = sub.violations.apply(lambda v: int(vtype in v)).values
        # Скор для AUC — своя непрерывная величина критерия, не бинарный
        # вердикт (см. docstring модуля и _risk_score). Кадр без измерения
        # (сбой геометрии) откатывается на вердикт — он же в конфьюжн-матрице.
        raw = sub.details.apply(lambda dd: _risk_score(vtype, dd)).values.astype(float)
        score = np.where(np.isfinite(raw), raw, pred.astype(float))
        m = stats.evaluate(y, pred, score, sub.study.values, n_boot=n_boot)
        m['positives'] = int(y.sum())
        out['per_violation'][vtype] = m

    # бинарный класс по кадру: есть хоть одно нарушение
    for region, name in (('spine', 'spine'), ('hip', 'hip')):
        sel = d.expert_region.isin(('lh', 'rh')) if region == 'hip' else (d.expert_region == region)
        cols = [c for v, (r, c) in CRITERIA.items() if r == region]
        sub = d[sel]
        known = sub[cols].notna().all(axis=1)
        sub = sub[known]
        if sub.empty:
            continue
        y = (sub[cols].sum(axis=1) > 0).astype(int).values
        pred = sub.violations.apply(lambda v: int(bool(v))).values
        m = stats.evaluate(y, pred, _binary_score(sub, pred), sub.study.values, n_boot=n_boot)
        m['positives'] = int(y.sum())
        m['images'] = int(len(sub))
        out['per_region'][name] = m

    all_cols = [c for _, c in CRITERIA.values()]
    known = d[all_cols].notna().any(axis=1)
    sub = d[known]
    y = (sub[all_cols].fillna(0).sum(axis=1) > 0).astype(int).values
    pred = sub.violations.apply(lambda v: int(bool(v))).values
    m = stats.evaluate(y, pred, _binary_score(sub, pred), sub.study.values, n_boot=n_boot)
    m['positives'] = int(y.sum())
    m['images'] = int(len(sub))
    out['overall']['binary_quality_class'] = m

    f1s = [v['f1'] for v in out['per_violation'].values()]
    out['overall']['macro_f1'] = float(np.mean(f1s)) if f1s else float('nan')
    out['overall']['region_accuracy'] = float((d.region == d.expert_region).mean())
    out['overall']['processed'] = int((d.status == 'Success').sum())
    out['overall']['failed'] = int((d.status == 'Failure').sum())
    out['overall']['seconds_per_image_median'] = float(d.seconds.median())
    out['overall']['seconds_per_image_p95'] = float(d.seconds.quantile(0.95))
    return out


def honest(d: pd.DataFrame, n_boot: int = 2000, cnn: bool = True) -> dict:
    """Честная оценка всего конвейера: каждый вердикт — out-of-fold.

    evaluate() считает метрики с порогами, подобранными на этой же выборке, —
    это оценка сверху. Здесь каждый критерий взят в том виде, в каком его
    решение не видело отложенного фолда:

    * позвоночник (укладка, ось, артефакты без сети) — порог подобран на
      обучающих фолдах (spine_qc.calibrate.out_of_fold);
    * ротация и артефакты с сетью — вложенный свод cnn_qc: и сеть, и
      коэффициенты свода, и порог учились без отложенного фолда (oof.csv);
    * ротация без сети — коридор нормы по обучающим фолдам;
    * отступы ROI — пороги ТЗ (3 / 3 / 2 см), ничего не подбиралось.

    Бинарный класс кадра — «хоть одно нарушение» по этим OOF-вердиктам. Его
    ROC-AUC — по своду OOF-вероятностей критериев 1 - П(1 - p).
    """
    from hip_rotation import THRESHOLDS as ROT
    from hip_rotation.detect import corridor_distance
    from spine_qc import probability as prob
    from spine_qc.calibrate import geometry_score, measure, out_of_fold

    root = HERE.parent
    # pred — OOF-вердикт, score — непрерывная величина для AUC критерия,
    # probs — вероятность для свода по кадру
    pred, score, probs = {}, {}, {}

    ds, X = measure(trainset.frames('spine'))
    oof_pred, _, _, oof_prob = out_of_fold(ds, X)
    for name, key in (('position', 'spine_position'), ('axis', 'spine_axis'),
                      ('artifacts', 'spine_artifacts')):
        pred[key] = dict(zip(ds.rel_path, oof_pred[name]))
        score[key] = dict(zip(ds.rel_path, geometry_score(name, ds)))
        probs[key] = dict(zip(ds.rel_path, oof_prob[name]))

    m = pd.read_csv(root / 'hip_rotation' / 'measurements.csv')
    q = float(ROT.get('corridor_percentile', 15.0))
    pred['hip_rotation'], score['hip_rotation'], probs['hip_rotation'] = {}, {}, {}
    for k in sorted(m.fold.unique()):
        te = m.fold == k
        lo, hi = np.percentile(m.area_mm2[~te & (m.y == 0)].dropna(), [q, 100.0 - q])
        for r in m[te].itertuples():
            ok = np.isfinite(r.area_mm2)
            dist = corridor_distance(r.area_mm2, {'area_lo': lo, 'area_hi': hi}) if ok else np.nan
            pred['hip_rotation'][r.rel_path] = int(ok and dist > 0)
            score['hip_rotation'][r.rel_path] = dist
            probs['hip_rotation'][r.rel_path] = prob.apply(ROT.get('probability'), dist) if ok else None

    # С сетью: вердикт — вложенный свод (сеть, свод и порог без отложенного
    # фолда); скор — OOF-вероятность сети в своде с тремя коэффициентами по
    # всем OOF. Склеенные вероятности пяти вложенных сводов ранжируются хуже
    # любого из них, поэтому для AUC берётся p_stack (см. cnn_qc/train.py).
    for crit, key in (('rotation', 'hip_rotation'), ('artifacts', 'spine_artifacts')):
        f = root / 'cnn_qc' / crit / 'oof.csv'
        if cnn and f.exists():
            o = pd.read_csv(f)
            pred[key] = dict(zip(o.rel_path, o.pred_nested.astype(int)))
            col = 'p_verdict' if 'p_verdict' in o else 'p_stack'
            score[key] = dict(zip(o.rel_path, o[col]))
            probs[key] = dict(zip(o.rel_path, o[col]))

    from hip_roi import probability as roi_prob
    pred['hip_roi'] = {r.rel_path: int('hip_roi' in r.violations) for r in d.itertuples()}
    score['hip_roi'] = {r.rel_path: _risk_score('hip_roi', r.details) for r in d.itertuples()}
    probs['hip_roi'] = {r.rel_path: (r.details.get('hip_roi') or {}).get('probability')
                        for r in d.itertuples()}

    out: dict = {'per_violation': {}, 'per_region': {}, 'overall': {}}
    frame_pred = {r: 0 for r in d.rel_path}
    frame_prob: dict = {r: [] for r in d.rel_path}
    for vtype, (region, col) in CRITERIA.items():
        sel = d.expert_region.isin(('lh', 'rh')) if region == 'hip' else (d.expert_region == region)
        sub = d[sel & d[col].notna()]
        y = sub[col].values.astype(int)
        pv = np.array([pred[vtype].get(r, 0) for r in sub.rel_path], int)
        sc = np.array([score[vtype].get(r, np.nan) for r in sub.rel_path], float)
        sc = np.where(np.isfinite(sc), sc, pv.astype(float))
        mm = stats.evaluate(y, pv, sc, sub.study.values, n_boot=n_boot)
        mm['positives'] = int(y.sum())
        out['per_violation'][vtype] = mm
        for r, v in zip(sub.rel_path, pv):
            frame_pred[r] |= int(v)
        for r in d.rel_path[sel]:
            v = probs[vtype].get(r)
            if v is not None and np.isfinite(v):
                frame_prob[r].append(float(v))

    def binary(sub: pd.DataFrame, cols: list[str]) -> dict:
        y = (sub[cols].fillna(0).sum(axis=1) > 0).astype(int).values
        pv = np.array([frame_pred[r] for r in sub.rel_path], int)
        pq = np.array([prob.combine(frame_prob[r]) if frame_prob[r] else np.nan
                       for r in sub.rel_path], float)
        mm = stats.evaluate(y, pv, np.where(np.isfinite(pq), pq, pv), sub.study.values,
                            n_boot=n_boot)
        mm['positives'], mm['images'] = int(y.sum()), int(len(sub))
        return mm

    for region in ('spine', 'hip'):
        sel = d.expert_region.isin(('lh', 'rh')) if region == 'hip' else (d.expert_region == region)
        cols = [c for v, (r, c) in CRITERIA.items() if r == region]
        sub = d[sel]
        sub = sub[sub[cols].notna().all(axis=1)]
        out['per_region'][region] = binary(sub, cols)
    all_cols = [c for _, c in CRITERIA.values()]
    out['overall']['binary_quality_class'] = binary(d[d[all_cols].notna().any(axis=1)], all_cols)
    out['overall']['macro_f1'] = float(np.mean([v['f1'] for v in out['per_violation'].values()]))
    out['sources'] = {
        'spine_position': 'порог по обучающим фолдам',
        'spine_axis': 'порог по обучающим фолдам',
        'spine_artifacts': ('вложенный свод cnn_qc' if cnn and (root / 'cnn_qc' / 'artifacts' / 'oof.csv').exists()
                            else 'порог по обучающим фолдам'),
        'hip_rotation': ('вложенный свод cnn_qc' if cnn and (root / 'cnn_qc' / 'rotation' / 'oof.csv').exists()
                         else 'коридор по обучающим фолдам'),
        'hip_roi': 'длина поля по ТЗ и анатомии, без подбора',
    }
    return out


def _oof_reference() -> dict:
    """Честные OOF-метрики, сохранённые калибраторами критериев."""
    ref = {}
    p = Path(__file__).resolve().parent.parent
    sp = p / 'spine_qc' / 'metrics.json'
    if sp.exists():
        for k, v in json.loads(sp.read_text(encoding='utf-8')).items():
            ref[f'spine_{k}'] = v
    hr = p / 'hip_rotation' / 'metrics.json'
    if hr.exists():
        ref['hip_rotation'] = json.loads(hr.read_text(encoding='utf-8'))
    hroi = p / 'hip_roi' / 'eval' / 'metrics.json'
    if hroi.exists():
        m = json.loads(hroi.read_text(encoding='utf-8')).get('methods', {}).get('kit2')
        if m:
            ref['hip_roi'] = m
    return ref


def main() -> int:
    ap = argparse.ArgumentParser(description='Сквозная оценка сервиса на обучающем наборе')
    ap.add_argument('--quick', action='store_true', help='без бутстрэпа')
    ap.add_argument('--hip-method', default='kit2')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--keypoints', action='store_true',
                    help='ось и отступы ROI считать моделью ключевых точек (ансамбль)')
    ap.add_argument('--keypoints-oof', action='store_true',
                    help='то же, но каждый кадр считает модель фолда, не видевшая его')
    ap.add_argument('--no-cnn', action='store_true',
                    help='без нейросети cnn_qc: вердикт по одной геометрии (базовая линия)')
    ap.add_argument('--roi-rule', default='scan_length', choices=('scan_length', 'margins'))
    ap.add_argument('--out', type=Path, default=METRICS, help='куда положить метрики')
    args = ap.parse_args()

    d = run(args.hip_method, keypoints=args.keypoints, keypoints_oof=args.keypoints_oof,
            cnn=not args.no_cnn, roi_rule=args.roi_rule)
    cnn_flags = int(d['flags'].str.contains('cnn:').sum())
    if cnn_flags:
        print(f'у {cnn_flags} кадров нет вердикта сети (флаг cnn:*): в OOF-режиме это кадры '
              'без метки эксперта — их нет в обучающей выборке сети')
    print(f'{len(d)} кадров, {d.study.nunique()} исследований, '
          f'ошибок {(d.status == "Failure").sum()}')
    res = evaluate(d, n_boot=0 if args.quick else 2000)

    print('\nпо типам нарушений (пороги подобраны на этой же выборке):')
    for k, m in res['per_violation'].items():
        print(f'  {k:16} {stats.fmt(m)}')
    print('\nбинарный класс «есть нарушение»:')
    for k, m in res['per_region'].items():
        print(f'  {k:16} {stats.fmt(m)}')
    print(f"  {'всего':16} {stats.fmt(res['overall']['binary_quality_class'])}")
    o = res['overall']
    print(f"\nmacro-F1 по типам нарушений: {o['macro_f1']:.3f}")
    print(f"точность определения области: {o['region_accuracy']:.3f}")
    print(f"время на кадр: медиана {o['seconds_per_image_median']*1000:.0f} мс, "
          f"p95 {o['seconds_per_image_p95']*1000:.0f} мс")

    hon = honest(d, n_boot=0 if args.quick else 2000, cnn=not args.no_cnn)
    res['honest_oof'] = hon
    print('\nчестная оценка всего конвейера (каждый вердикт out-of-fold):')
    for k, m in hon['per_violation'].items():
        print(f'  {k:16} {stats.fmt(m)}')
    for k, m in hon['per_region'].items():
        print(f'  {k:16} {stats.fmt(m)}')
    print(f"  {'всего':16} {stats.fmt(hon['overall']['binary_quality_class'])}")
    print(f"  macro-F1 {hon['overall']['macro_f1']:.3f}")

    ref = _oof_reference()
    if ref:
        print('\nдля сравнения — честные OOF (порог и модель учились только на обучающих фолдах):')
        for k in ('spine_position', 'spine_axis', 'spine_artifacts', 'hip_rotation', 'hip_roi'):
            m = ref.get(k)
            if not m:
                continue
            se, sp = m.get('sensitivity'), m.get('specificity')
            f1 = m.get('f1')
            f1s = f'{f1:.2f}' if isinstance(f1, (int, float)) else '—'
            print(f'  {k:16} sens={se:.2f} spec={sp:.2f} F1={f1s}')

    if not args.no_save:
        res['oof_reference'] = ref
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                            encoding='utf-8')
        # Таблица кадров ложится рядом с метриками: прогон другим измерителем
        # не должен затирать зафиксированную базовую линию в service/.
        frames = args.out.parent / 'evaluation.csv'
        d.drop(columns=['violations', 'details']).assign(
            violations=d.violations.apply(';'.join)).to_csv(
            frames, index=False, encoding='utf-8')
        print(f'\nсохранено: {args.out}, {frames}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
