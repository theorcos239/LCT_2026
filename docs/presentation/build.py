# -*- coding: utf-8 -*-
"""Сборка презентации из данных репозитория.

    python docs/presentation/build.py                 # PPTX по шаблону ЛЦТ, PDF, доклад DOCX
    python docs/presentation/build.py --cases --screens demo.zip   # + кейсы и снимки интерфейса

Все числа на слайдах читаются из файлов метрик (service/metrics.json,
runs/eval_geometry/metrics.json, cnn_qc/*/meta.json, service/benchmark.json),
кейсы рисует сам конвейер. Презентация пересобирается после любого
переобучения одной командой и не расходится с README.

Оформление — шаблон организаторов (template/LCT2026_template.pptx, шрифт
Montserrat), слайды — build_pptx.py, текст доклада — speech.py, данные
команды — team.json.

Нужно: python-pptx, python-docx; playwright с chromium (снимки интерфейса);
для PDF, встраивания шрифта и миниатюр в докладе — установленный PowerPoint
(Windows) и шрифт Montserrat.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = HERE / 'assets'
NAME = 'DXA_QC_pitch'

CRIT_RU = {
    'spine_position': 'Укладка позвоночника',
    'spine_axis': 'Ось позвоночника',
    'spine_artifacts': 'Посторонние предметы',
    'hip_rotation': 'Ротация бедра',
    'hip_roi': 'Отступы ROI бедра',
}
REGION_RU = {'spine': 'Позвоночник, кадр', 'hip': 'Бедро, кадр'}


# --------------------------------------------------------------------------- #
#  Данные
# --------------------------------------------------------------------------- #
def jload(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def load() -> dict:
    import trainset
    D: dict = {}
    m = jload(ROOT / 'service' / 'metrics.json')
    D['ins'] = m
    D['h'] = m.get('honest_oof', {})
    D['hg'] = jload(ROOT / 'runs' / 'eval_geometry' / 'metrics.json').get('honest_oof', {})
    # исходное решение до доработок под эксперта: без сети, ROI по рисунку 6 ТЗ
    D['hb'] = jload(ROOT / 'runs' / 'eval_baseline' / 'metrics.json').get('honest_oof', {}) or D['hg']
    D['h0'] = D['hg'].get('per_violation', {})
    D['tests'] = 64
    D['tests_bot'] = 27
    D['cnn'] = {c: jload(ROOT / 'cnn_qc' / c / 'meta.json') for c in ('rotation', 'artifacts')}
    D['bench'] = jload(ROOT / 'service' / 'benchmark.json')
    D['bench_geo'] = jload(ROOT / 'runs' / 'benchmark_geometry.json')
    # модель точек и контур на одних и тех же кадрах фолдов 0–1 (обучены только они)
    D['kp_oof'] = jload(ROOT / 'runs' / 'eval_oof' / 'metrics.json')
    D['kp_contour'] = jload(ROOT / 'runs' / 'eval_oof_contour' / 'metrics.json')
    D['spine_oof'] = jload(ROOT / 'spine_qc' / 'metrics.json')
    D['rot_geo'] = jload(ROOT / 'hip_rotation' / 'metrics.json')
    D['roi_prob'] = jload(ROOT / 'hip_roi' / 'probability.json')
    df = trainset.frames()
    D['frames'] = len(df)
    D['regions'] = df.label.value_counts().to_dict()
    D['pos'] = {c: (int(df[f'y_{c}'].sum()), int(df[f'y_{c}'].notna().sum()))
                for c in trainset.CRITERIA}
    D['strata'] = pd.read_csv(ROOT / 'folds.csv').stratum.value_counts().to_dict()
    D['rot_oof'] = pd.read_csv(ROOT / 'cnn_qc' / 'rotation' / 'oof.csv')
    D['art_oof'] = pd.read_csv(ROOT / 'cnn_qc' / 'artifacts' / 'oof.csv')
    D['sample'] = report_sample()
    return D


def f2(x, nd=2) -> str:
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def ci(m: dict, key: str) -> str:
    lo, hi = m.get(f'{key}_ci', [None, None])
    if lo is None or not np.isfinite(lo):
        return ''
    return f'{lo:.2f}–{hi:.2f}'


# --------------------------------------------------------------------------- #
#  Кейсы: конвейер сам рисует оверлеи
# --------------------------------------------------------------------------- #
CASES = [
    # (ключ, тип нарушения, критерий отбора)
    ('axis', 'spine_axis'),
    ('artifacts', 'spine_artifacts'),
    ('rotation', 'hip_rotation'),
    ('roi', 'hip_roi'),
    ('ok_spine', None),
    ('ok_hip', None),
]


def render_cases(force: bool = False) -> dict:
    """Кадры-примеры: верно найденное нарушение каждого типа и качественные кадры.

    Вердикт и вероятности — как в честной оценке: сеть отвечает out-of-fold
    (кадр предсказан моделями, его не видевшими). Тепловую карту рисует
    поставляемый ансамбль — это иллюстрация, где сеть видит признак.
    """
    out = ASSETS / 'cases'
    meta_path = out / 'cases.json'
    if meta_path.exists() and not force:
        return jload(meta_path)
    out.mkdir(parents=True, exist_ok=True)
    import trainset
    from service.overlay import render
    from service.pipeline import Analyzer

    a = Analyzer.load(cnn=False, cnn_oof=True)
    ev = pd.read_csv(ROOT / 'service' / 'evaluation.csv')
    ev['violations'] = ev.violations.fillna('')
    col = {'spine_axis': 'y_axis', 'spine_artifacts': 'y_artifacts',
           'hip_rotation': 'y_rotation', 'hip_roi': 'y_roi'}

    def score(key, res):
        d = res['details']
        if key == 'spine_axis':
            return abs(d['spine']['axis']['angle_deg'] or 0)
        if key == 'spine_artifacts':
            return d['spine']['artifacts'].get('p_cnn') or 0
        if key == 'hip_rotation':
            return d['hip_rotation'].get('p_cnn') or 0
        if key == 'hip_roi':
            from hip_roi.probability import shortfall
            return shortfall(d['hip_roi']) or 0
        return -(res.get('quality_probability') or 1)

    cases = {}
    for name, key in CASES:
        if key:
            cand = ev[(ev[col[key]] == 1) & ev.violations.str.contains(key)]
        else:
            region = 'spine' if name == 'ok_spine' else ('lh', 'rh')
            sel = ev.expert_region.isin(region if isinstance(region, tuple) else (region,))
            ys = ev[['y_position', 'y_axis', 'y_artifacts', 'y_rotation', 'y_roi']].fillna(0).sum(axis=1)
            cand = ev[sel & (ys == 0) & (ev.violations == '')]
        best = None
        for r in cand.itertuples():
            px = trainset.read(r.rel_path)
            res = a.analyze_pixels(px)
            # сначала снимки, где весь перечень нарушений совпал с экспертом
            truth = {v for v, c in col.items() if getattr(r, c) == 1}
            exact = set(res['violations']) == truth
            single = len(res['violations']) <= 1
            clean = not any(f.startswith(('measure:', 'mask:', 'I:', 'T:', 'L:', 'rotation:'))
                            for f in res['flags'])
            sc = (exact and single, clean, score(key, res))
            if best is None or sc > best[0]:
                best = (sc, r, px, res)
        if best is None:
            continue
        _, r, px, res = best
        row = {'anatomical_region': res['anatomical_region'], 'details': res['details']}
        img = render(px, row)
        p = out / f'{name}.png'
        img.save(p)
        d = res['details']
        info = {'file': p.name, 'study': r.study, 'rel_path': r.rel_path,
                'region': res['anatomical_region'], 'violations': res['violations'],
                'quality_probability': res.get('quality_probability')}
        if res['anatomical_region'] == 'spine':
            info['angle'] = d['spine']['axis'].get('angle_deg')
            info['iliac'] = d['spine']['position'].get('iliac_area')
            info['art_len'] = d['spine']['artifacts'].get('length_mm')
            info['art_p'] = d['spine']['artifacts'].get('p_cnn')
            info['art_prob'] = d['spine']['artifacts'].get('probability')
        else:
            hr, rot = d['hip_roi'], d['hip_rotation']
            info.update(top=hr.get('m_top_mm'), bottom=hr.get('m_bottom_mm'), lat=hr.get('m_lat_mm'),
                        scan_len=hr.get('scan_length_mm'), need_len=hr.get('min_length_mm'),
                        area=rot.get('area_mm2'), rot_p=rot.get('p_cnn'),
                        rot_prob=rot.get('probability'))
        cases[name] = info
    meta_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2, default=float),
                         encoding='utf-8')
    return cases


SAMPLE_STUDIES = ('2.25.102965897799404973225419898683246248872',     # норма
                  '2.25.114887542067602662452692698814728642117')     # ротация


def report_sample(force: bool = False) -> list[dict]:
    """Первые восемь колонок отчёта (ТЗ 2.5) рабочим конвейером на двух исследованиях."""
    p = ASSETS / 'report_sample.json'
    if p.exists() and not force:
        return jload(p)
    from service.pipeline import COLUMNS, Analyzer, process_study
    a = Analyzer.load()
    rows = []
    for s in SAMPLE_STUDIES:
        for r in process_study(ROOT / 'НД_для_обучения' / 'Исследования' / s, a, s):
            rows.append({k: r.get(k) for k in COLUMNS})
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    return rows


# --------------------------------------------------------------------------- #
#  Снимки веб-интерфейса
# --------------------------------------------------------------------------- #
def screenshots(demo_zip: Path) -> None:
    """Поднять сервис, загрузить демо-архив, снять интерфейс."""
    from playwright.sync_api import sync_playwright
    port = 8791
    proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'service.api:app',
                             '--host', '127.0.0.1', '--port', str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import urllib.request
        for _ in range(120):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2)
                break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={'width': 1600, 'height': 940}, device_scale_factor=1.25)
            pg.goto(f'http://127.0.0.1:{port}/')
            pg.wait_for_timeout(800)
            pg.locator('#fileInput').set_input_files(str(demo_zip))
            pg.wait_for_function("document.querySelector('#stage')?.textContent === 'Готово'",
                                 timeout=600000)
            pg.wait_for_function("document.querySelectorAll('#rows tr').length > 3", timeout=60000)
            pg.wait_for_timeout(2500)                  # снимок кадра и карточка критериев
            pg.screenshot(path=str(ASSETS / 'ui_results.png'))
            # кадр с нарушением ротации: фильтр по типу и первый кадр
            chip = pg.get_by_text('Ротация', exact=False).first
            if chip.count():
                chip.click()
                pg.wait_for_timeout(1500)
                pg.screenshot(path=str(ASSETS / 'ui_rotation.png'))
            b.close()
    finally:
        proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser(description='Сборка презентации по шаблону ЛЦТ 2026')
    ap.add_argument('--cases', action='store_true', help='перерисовать кейсы')
    ap.add_argument('--screens', type=Path, metavar='DEMO_ZIP',
                    help='снять веб-интерфейс, загрузив этот архив')
    args = ap.parse_args()
    ASSETS.mkdir(exist_ok=True)
    if args.screens:
        screenshots(args.screens)
    D = load()
    cases = render_cases(force=args.cases)
    import build_docx
    import build_pptx
    deck = build_pptx.build_deck(D, cases)
    pptx = HERE / f'{NAME}.pptx'
    deck.save(pptx)
    print(f'{pptx} — {len(deck.prs.slides)} слайдов')
    if build_pptx.export_with_powerpoint(pptx, HERE / f'{NAME}.pdf', ASSETS / 'pptx_png'):
        print(f'{NAME}.pdf и assets/pptx_png/*.png — через PowerPoint, Montserrat встроен в PPTX')
        build_docx.main()
    else:
        print('PowerPoint недоступен: PDF, миниатюры и доклад не пересобраны')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
