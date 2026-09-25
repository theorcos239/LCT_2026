# -*- coding: utf-8 -*-
"""Истинные отступы ROI по настоящей ручной разметке ключевых точек эксперта.

    python -m hip_roi.ground_truth

Источник ground truth — `data/razmetka_aleksandra_*.json` (аннотатор —
отдельный человек, разметка сделана без доступа к вердиктам качества, см.
протокол `Processing/Протокол разметки ключевых точек - денситометрия.pdf`,
правило 1). Снимки в разметке идентифицированы непрозрачными id, поэтому
сначала их нужно сопоставить с файлами датасета — это делает `match_images()`
по пиксельной корреляции с кадрами из `region_clf/labels.csv`.

Критерий отсчёта полей ТЗ подтверждён протоколом (стр. 9, таблица):
    поле 3 см сверху -> trochanter_major (верхушка большого вертела)
    поле 3 см снизу  -> ischium_bottom  (нижняя точка седалищной кости!)
    поле 2 см сбоку  -> trochanter_lateral (наружная точка большого вертела)

Это НЕ то же самое, что использует остальной `hip_roi` (низ ROI там ищется
через окружение малого вертела) — расхождение и его следствия разобраны в
`contradictions/ПОЯСНЕНИЕ.txt`.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from region_clf.features import read_image

from .geometry import MM_PER_PX

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'НД_для_обучения' / 'Исследования'
LABELS = ROOT / 'region_clf' / 'labels.csv'
HTML_WITH_IMAGES = ROOT / 'Processing' / 'razmetka_all.html'
ANNOTATIONS_JSON = ROOT / 'data' / 'razmetka_aleksandra_2026-09-22_2241.json'
XLSX = ROOT / 'НД_для_обучения' / 'разметка.xlsx'
OUT = Path(__file__).resolve().parent / 'contradictions'

TOP_MM, BOTTOM_MM, LAT_MM = 30.0, 30.0, 20.0
EXPERT_COLS = ['n', 'study', 'sp_pos', 'sp_axis', 'sp_art', 'rh_rot', 'rh_roi', 'lh_rot', 'lh_roi',
               'tot_sp', 'tot_rh', 'tot_lh', 'comment']


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1
    return np.clip((x.astype(np.float64) - lo) / (hi - lo), 0, 1)


def load_html_images(path: Path = HTML_WITH_IMAGES) -> list[dict]:
    """Снимки, встроенные в веб-инструмент разметки как base64 PNG (id/w/h/region/src)."""
    s = open(path, encoding='utf-8').read()
    m = re.search(r'<script id="dxa-data" type="application/json">(.*?)</script>', s, re.S)
    return json.loads(m.group(1))['images']


def decode_png(im: dict) -> np.ndarray:
    b = base64.b64decode(im['src'].split(',', 1)[1])
    return np.array(Image.open(io.BytesIO(b)))


def match_images(html_images: list[dict] | None = None) -> pd.DataFrame:
    """Сопоставление id из разметки с файлами датасета по пиксельной корреляции.

    Инструмент разметки хранит снимки под непрозрачными id, не именами
    исследований — сторонней привязки id -> файл не существует. Размер кадра
    (w, h) не меняется относительно исходного DICOM (только контраст
    растянут для показа), поэтому корреляция нормализованных пикселей даёт
    однозначное совпадение: на всех 252 снимках corr = 1.000.
    """
    if html_images is None:
        html_images = load_html_images()
    lab = pd.read_csv(LABELS)

    cache: dict[tuple, list] = {}
    for _, r in lab.iterrows():
        img = read_image(DATA / r.rel_path)
        h, w = img.shape
        grp = 'spine' if r.label == 'spine' else 'hip'
        cache.setdefault((grp, w, h), []).append((r.rel_path, r.label, r.study, _norm(img)))

    rows = []
    for im in html_images:
        png = decode_png(im)
        pn = _norm(png)
        best = None
        for rel_path, label, study, na in cache.get((im['region'], im['w'], im['h']), []):
            corr = float(np.corrcoef(na.ravel(), pn.ravel())[0, 1])
            if best is None or corr > best[0]:
                best = (corr, rel_path, label, study)
        if best is None:
            rows.append(dict(id=im['id'], region=im['region'], w=im['w'], h=im['h'],
                             corr=None, rel_path=None, label=None, study=None))
        else:
            rows.append(dict(id=im['id'], region=im['region'], w=im['w'], h=im['h'],
                             corr=round(best[0], 5), rel_path=best[1], label=best[2], study=best[3]))
    return pd.DataFrame(rows)


def load_expert_xlsx() -> pd.DataFrame:
    df = pd.read_excel(XLSX, header=None).iloc[2:, :13]
    df.columns = EXPERT_COLS
    return df


def true_margins(match: pd.DataFrame | None = None) -> pd.DataFrame:
    """Отступы ROI по реальным точкам эксперта + сверка с разметка.xlsx."""
    d = json.load(open(ANNOTATIONS_JSON, encoding='utf-8'))
    if match is None:
        match = match_images()
    expert = load_expert_xlsx().set_index('study')

    rows = []
    for _, m in match[match.region == 'hip'].iterrows():
        v = d['images'][m.id]
        p = v.get('points', {})
        have_all = all(k in p for k in ('trochanter_major', 'trochanter_lateral', 'ischium_bottom'))
        row = dict(id=m.id, study=m.study, side=m.label, rel_path=m.rel_path, w=m.w, h=m.h,
                   ann_flags=';'.join(v.get('flags', [])), absent=';'.join(v.get('absent', [])),
                   have_all_margin_points=have_all)
        if have_all:
            T, L, I = p['trochanter_major']['y'], p['trochanter_lateral']['x'], p['ischium_bottom']['y']
            m_top = T * MM_PER_PX
            m_bottom = (m.h - I) * MM_PER_PX
            m_lat = (L if m.label == 'rh' else m.w - L) * MM_PER_PX
            row.update(T_y=round(T, 1), L_x=round(L, 1), I_y=round(I, 1),
                       m_top_true=round(m_top, 1), m_bottom_true=round(m_bottom, 1), m_lat_true=round(m_lat, 1),
                       top_ok=m_top >= TOP_MM, bottom_ok=m_bottom >= BOTTOM_MM, lat_ok=m_lat >= LAT_MM)
            row['roi_ok_true'] = row['top_ok'] and row['bottom_ok'] and row['lat_ok']
        else:
            row.update(m_top_true=None, m_bottom_true=None, m_lat_true=None,
                       top_ok=None, bottom_ok=None, lat_ok=None, roi_ok_true=None)
        col = f'{m.label}_roi'
        flag = np.nan
        if m.study in expert.index:
            v2 = expert.loc[m.study, col]
            flag = None if pd.isna(v2) else int(v2)
        row['expert_flag_xlsx'] = flag
        rows.append(row)
    return pd.DataFrame(rows)


def real_disagreements(margins: pd.DataFrame) -> pd.DataFrame:
    """Кадры, где булев вердикт разметка.xlsx противоречит измерению по точкам."""
    known = margins[margins.expert_flag_xlsx.notna() & margins.have_all_margin_points].copy()
    known['expert_flag_xlsx'] = known['expert_flag_xlsx'].astype(int)
    known['pred_violation'] = ~known.roi_ok_true.astype(bool)
    known['expert_violation'] = known.expert_flag_xlsx == 1
    return known[known.pred_violation != known.expert_violation]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    match = match_images()
    match.to_csv(OUT / 'image_match.csv', index=False, encoding='utf-8-sig')
    print(f'сопоставлено {len(match)} снимков, средняя corr = {match['corr'].mean():.4f}, '
          f'ниже 0.98: {(match['corr'] < 0.98).sum()}')

    margins = true_margins(match)
    margins.to_csv(OUT / 'true_margins.csv', index=False, encoding='utf-8-sig')
    have = margins[margins.have_all_margin_points]
    print(f'кадров бедра: {len(margins)}, с тремя опорными точками: {len(have)}')

    known = have[have.expert_flag_xlsx.notna()].copy()
    known['expert_flag_xlsx'] = known['expert_flag_xlsx'].astype(int)
    pred = ~known.roi_ok_true.astype(bool)
    y = known.expert_flag_xlsx == 1
    tp, fn = int((pred & y).sum()), int((~pred & y).sum())
    fp, tn = int((pred & ~y).sum()), int((~pred & ~y).sum())
    print(f'сверка с разметка.xlsx (n={len(known)}): tp={tp} fn={fn} fp={fp} tn={tn}  '
          f'sens={tp/max(tp+fn,1):.2f} spec={tn/max(tn+fp,1):.2f}')

    dis = real_disagreements(margins)
    dis.to_csv(OUT / 'all_real_disagreements.csv', index=False, encoding='utf-8-sig')
    print(f'настоящих расхождений: {len(dis)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
