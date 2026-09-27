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
«в целом» такой общей шкалы нет (критерии в разных единицах), там AUC
действительно вырождается в сбалансированную точность — это не ошибка, а
следствие того, что бинарный вердикт «есть хоть одно нарушение» не сводится к
одному числу.
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
        if vtype == 'spine_artifacts':
            return details['spine']['artifacts']['score']
        if vtype == 'hip_rotation':
            return details['hip_rotation']['corridor_distance']
        if vtype == 'hip_roi':
            from hip_roi.geometry import BOTTOM_MM, LAT_MM, TOP_MM
            r = details['hip_roi']
            return max(TOP_MM - r['m_top_mm'], BOTTOM_MM - r['m_bottom_mm'],
                      LAT_MM - r['m_lat_mm'])
    except (KeyError, TypeError):
        pass
    return float('nan')


def run(hip_method: str = 'kit2', keypoints: bool = False) -> pd.DataFrame:
    """Прогон конвейера по всем уникальным кадрам обучающего набора.

    keypoints=True считает ось позвоночника и отступы ROI моделью ключевых
    точек. Сравнивать два прогона осмысленно: набор кадров и метки одни и те
    же, меняется только измеритель.
    """
    from .pipeline import Analyzer

    df = trainset.frames()
    a = Analyzer.load(hip_method=hip_method, keypoints=keypoints)
    rows = []
    for r in df.itertuples():
        px = trainset.read(r.rel_path)
        import time
        t0 = time.perf_counter()
        try:
            res = a.analyze_pixels(px)
            status, err = 'Success', ''
        except Exception as e:
            res = {'anatomical_region': 'unknown', 'violations': [], 'flags': []}
            status, err = 'Failure', f'{type(e).__name__}: {e}'
        rows.append({
            'study': r.study, 'rel_path': r.rel_path, 'expert_region': r.label,
            'region': res['anatomical_region'], 'violations': res['violations'],
            'flags': ';'.join(res['flags']), 'status': status, 'error': err,
            'seconds': time.perf_counter() - t0, 'details': res.get('details', {}),
            **{f'y_{c}': getattr(r, f'y_{c}') for c in trainset.CRITERIA},
        })
    return pd.DataFrame(rows)


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
        m = stats.evaluate(y, pred, pred.astype(float), sub.study.values, n_boot=n_boot)
        m['positives'] = int(y.sum())
        m['images'] = int(len(sub))
        out['per_region'][name] = m

    all_cols = [c for _, c in CRITERIA.values()]
    known = d[all_cols].notna().any(axis=1)
    sub = d[known]
    y = (sub[all_cols].fillna(0).sum(axis=1) > 0).astype(int).values
    pred = sub.violations.apply(lambda v: int(bool(v))).values
    m = stats.evaluate(y, pred, pred.astype(float), sub.study.values, n_boot=n_boot)
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
                    help='ось и отступы ROI считать моделью ключевых точек')
    ap.add_argument('--out', type=Path, default=METRICS, help='куда положить метрики')
    args = ap.parse_args()

    d = run(args.hip_method, keypoints=args.keypoints)
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
