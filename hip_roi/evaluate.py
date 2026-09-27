# -*- coding: utf-8 -*-
"""Оценка всех комплектов на 153 кадрах бедра против разметки эксперта.

    python -m hip_roi.evaluate                # всё: CSV, metrics.json, контактные листы
    python -m hip_roi.evaluate --methods kit2 # только один комплект

Что делает:
1. Для каждого кадра бедра из region_clf/labels.csv (эталон области) и каждого
   комплекта считает отступы; пишет hip_roi/eval/margins_<kit>.csv.
2. Сверяет вердикт с колонкой эксперта «корректности области интересов»
   (разметка.xlsx) — матрица ошибок, чувствительность/специфичность.
3. Сверяет найденную нижнюю точку седалищной кости (низ ROI по рисунку 6 ТЗ)
   с ручной разметкой точек (`contradictions/true_margins.csv`); пишет в
   metrics.json. Ручные точки — не эталон, эталон вердикта — xlsx.
4. Контактные листы: все расхождения с экспертом (disagree_<kit>.png), все
   7 помеченных экспертом кадров для всех комплектов (expert_positive.png).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from region_clf.features import read_image

from .geometry import D_TI_MM, MM_PER_PX, px2mm
from .kits import measure_roi_margins
from .overlay import contact_sheet, render_overlay

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'НД_для_обучения' / 'Исследования'
XLSX = ROOT / 'НД_для_обучения' / 'разметка.xlsx'
LABELS = ROOT / 'region_clf' / 'labels.csv'
OUT = Path(__file__).resolve().parent / 'eval'

EXPERT_COLS = ['n', 'study', 'sp_pos', 'sp_axis', 'sp_art', 'rh_rot', 'rh_roi', 'lh_rot', 'lh_roi',
               'tot_sp', 'tot_rh', 'tot_lh', 'comment']


def load_expert() -> pd.DataFrame:
    df = pd.read_excel(XLSX, header=None).iloc[2:, :13]
    df.columns = EXPERT_COLS
    return df


def hip_frames() -> pd.DataFrame:
    lab = pd.read_csv(LABELS)
    lab = lab[lab.label.isin(('lh', 'rh'))].reset_index(drop=True)
    ex = load_expert().set_index('study')
    flags = []
    for _, r in lab.iterrows():
        col = f'{r.label}_roi'
        v = ex[col].get(r.study, np.nan) if r.study in ex.index else np.nan
        flags.append(np.nan if pd.isna(v) else int(v))
    lab['expert_roi'] = flags
    return lab


def run_all(frames: pd.DataFrame, methods: list[str]):
    rows = {m: [] for m in methods}
    imgs = {}
    t0 = time.perf_counter()
    for i, r in frames.iterrows():
        img = read_image(DATA / r.rel_path)
        imgs[r.rel_path] = img
        for m in methods:
            t = time.perf_counter()
            res = measure_roi_margins(img, r.label, m)
            dt = time.perf_counter() - t
            row = {k: v for k, v in res.items() if k not in ('diag',)}
            row.update(study=r.study, rel_path=r.rel_path, rows=img.shape[0], expert_roi=r.expert_roi,
                       ms=round(dt * 1000, 1), flag_list=';'.join(res['flags']))
            row.pop('flags', None)
            rows[m].append(row)
    print(f'обработано {len(frames)} кадров x {len(methods)} комплектов за {time.perf_counter() - t0:.1f} с')
    return {m: pd.DataFrame(v) for m, v in rows.items()}, imgs


def confusion(df: pd.DataFrame) -> dict:
    d = df[df.expert_roi.isin([0, 1])].copy()
    pred = d.roi_ok.map({True: 0, False: 1})
    known = pred.notna()
    d, pred = d[known], pred[known].astype(int)
    y = d.expert_roi.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        'n': int(len(d)), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': round(tp / max(tp + fn, 1), 3),
        'specificity': round(tn / max(tn + fp, 1), 3),
        'undecided': int((~known).sum()),
        'violations_predicted': int((pred == 1).sum()),
        'violations_expert': int((y == 1).sum()),
    }


MANUAL = Path(__file__).resolve().parent / 'contradictions' / 'true_margins.csv'


def ischium_vs_manual(df2: pd.DataFrame) -> dict:
    """Низ ROI (нижняя точка седалищной кости) против ручной разметки точек.

    Проверяет сам измеритель, а не вердикт: вердикт сверяется с xlsx. Ручная
    точка `ischium_bottom` — разметка команды, в ней есть как минимум одна
    ошибка (точка на шейке бедра, …9759504, CR000000), она попадает в хвост.
    """
    if not MANUAL.exists():
        return {'note': f'нет {MANUAL.name}: python -m hip_roi.ground_truth'}
    t = pd.read_csv(MANUAL, encoding='utf-8-sig')[['rel_path', 'I_y', 'T_y']].dropna()
    d = df2.merge(t, on='rel_path').dropna(subset=['B_px'])
    err = (d.B_px - d.I_y).to_numpy() * MM_PER_PX
    dist = (d.I_y - d.T_y).to_numpy() * MM_PER_PX
    return {
        'n': int(len(d)),
        'error_median_mm': round(float(np.median(err)), 1),
        'abs_error_median_mm': round(float(np.median(np.abs(err))), 1),
        'abs_error_p90_mm': round(float(np.percentile(np.abs(err), 90)), 1),
        'over_5mm': int((np.abs(err) > 5).sum()),
        'over_10mm': [{'rel_path': r, 'error_mm': round(float(e), 1)}
                      for r, e in zip(d.rel_path, err) if abs(e) > 10],
        'manual_T_to_ischium_mm': {'median': round(float(np.median(dist)), 1),
                                   'p1': round(float(np.percentile(dist, 1)), 1),
                                   'p99': round(float(np.percentile(dist, 99)), 1),
                                   'prior_used_mm': D_TI_MM},
    }


def sheet_for(df: pd.DataFrame, imgs: dict, method: str, sel: pd.DataFrame, path: Path, cols=4):
    tiles = []
    for _, r in sel.iterrows():
        img = imgs[r.rel_path]
        res = measure_roi_margins(img, r.side, method)
        im = render_overlay(img, res, scale=2)
        from PIL import ImageDraw
        from .overlay import _font
        font, cyr = _font(12)
        ImageDraw.Draw(im).text((3, im.size[1] - 16),
                                f"{'эксперт' if cyr else 'expert'}={r.expert_roi} rows={r.rows} {r.study[-10:]}",
                                fill=(255, 255, 255), font=font)
        tiles.append(im)
    contact_sheet(tiles, cols=cols).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['kit0', 'kit1', 'kit2', 'kit3'])
    ap.add_argument('--no-sheets', action='store_true')
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    frames = hip_frames()
    print(f'кадров бедра: {len(frames)}, эксперт ROI=1: {int((frames.expert_roi == 1).sum())}, '
          f'ROI=0: {int((frames.expert_roi == 0).sum())}, NaN: {int(frames.expert_roi.isna().sum())}')
    results, imgs = run_all(frames, args.methods)

    metrics = {'n_frames': int(len(frames)), 'methods': {}}
    for m, df in results.items():
        df.to_csv(OUT / f'margins_{m}.csv', index=False, encoding='utf-8-sig')
        c = confusion(df)
        c['ms_per_frame_median'] = float(df.ms.median())
        c['flags_top'] = df['flag_list'][df['flag_list'] != ''].str.split(';').explode().value_counts().head(8).to_dict()
        metrics['methods'][m] = c
        print(f"{m}: n={c['n']} tp={c['tp']} fn={c['fn']} fp={c['fp']} tn={c['tn']} "
              f"sens={c['sensitivity']} spec={c['specificity']} undecided={c['undecided']} "
              f"{c['ms_per_frame_median']:.0f} мс/кадр")

    if 'kit2' in results:
        metrics['ischium_vs_manual'] = ischium_vs_manual(results['kit2'])
        print('седалищная кость против ручных точек (kit2):',
              {k: v for k, v in metrics['ischium_vs_manual'].items() if k != 'over_10mm'})
        d2 = results['kit2']
        buckets = pd.cut(d2.rows, [0, 210, 240, 270, 300, 330, 420])
        metrics['kit2_margins_by_rows'] = {
            str(k): {'n': int(len(g)), 'top_med': _med(g.m_top_mm), 'bottom_med': _med(g.m_bottom_mm),
                     'lat_med': _med(g.m_lat_mm), 'violations': int((g.roi_ok == False).sum())}
            for k, g in d2.groupby(buckets, observed=True)}
        # свип порогов: как меняется согласие с экспертом, если смягчить 3/3/2 см
        sweep = {}
        for bot in (30, 25, 20, 15):
            for lat in (20, 15):
                v = (d2.m_top_mm < 30) | (d2.m_bottom_mm < bot) | (d2.m_lat_mm < lat)
                k = d2.expert_roi.isin([0, 1]) & d2.m_bottom_mm.notna()
                y = d2.expert_roi[k] == 1
                sweep[f'bottom>={bot}&lat>={lat}'] = {
                    'tp': int((v[k] & y).sum()), 'fn': int((~v[k] & y).sum()),
                    'fp': int((v[k] & ~y).sum()), 'tn': int((~v[k] & ~y).sum())}
        metrics['kit2_threshold_sweep'] = sweep
        print('свип порогов kit2:', {k: f"tp{v['tp']}/fp{v['fp']}" for k, v in sweep.items()})
    # согласие комплектов между собой
    if len(results) > 1:
        agree = {}
        ms = list(results)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = results[ms[i]].roi_ok, results[ms[j]].roi_ok
                agree[f'{ms[i]}~{ms[j]}'] = round(float((a == b).mean()), 3)
        metrics['pairwise_verdict_agreement'] = agree

    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=str),
                                      encoding='utf-8')
    print('metrics ->', OUT / 'metrics.json')

    if args.no_sheets:
        return 0
    for m, df in results.items():
        if m == 'kit0':
            continue
        d = df[df.expert_roi.isin([0, 1])]
        pred = d.roi_ok.map({True: 0, False: 1})
        dis = d[(pred != d.expert_roi) | pred.isna()]
        if len(dis):
            sheet_for(df, imgs, m, dis, OUT / f'disagree_{m}.png')
            print(f'{m}: расхождений с экспертом {len(dis)} -> disagree_{m}.png')
    pos = results[args.methods[-1]][results[args.methods[-1]].expert_roi == 1]
    for m in args.methods:
        if m != 'kit0':
            sheet_for(results[m], imgs, m, pos, OUT / f'expert_positive_{m}.png', cols=4)
    return 0


def _med(s):
    s = s.dropna()
    return None if s.empty else round(float(s.median()), 1)


if __name__ == '__main__':
    sys.exit(main())
