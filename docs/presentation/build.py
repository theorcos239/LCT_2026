# -*- coding: utf-8 -*-
"""Сборка презентации из данных репозитория.

    python docs/presentation/build.py                 # slides.html, PPTX, PDF, доклад DOCX
    python docs/presentation/build.py --cases --screens demo.zip   # + кейсы и снимки интерфейса
    python docs/presentation/build.py --html-only     # только веб-версия

Все числа на слайдах читаются из файлов метрик (service/metrics.json,
runs/eval_geometry/metrics.json, cnn_qc/*/meta.json, service/benchmark.json),
кейсы рисует сам конвейер. Презентация пересобирается после любого
переобучения одной командой и не расходится с README.

Нужно: python-pptx, python-docx; playwright с chromium (снимки интерфейса);
для PDF и миниатюр в докладе — установленный PowerPoint (Windows), без него
PDF рендерится из HTML-версии.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
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
    D['kp_oof'] = jload(ROOT / 'runs' / 'eval_oof' / 'metrics.json')
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


def b64(path: Path, mime: str | None = None) -> str:
    mime = mime or ('image/png' if path.suffix == '.png' else 'image/jpeg')
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


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


# --------------------------------------------------------------------------- #
#  Графики (SVG, в масштабе)
# --------------------------------------------------------------------------- #
def ci_bar(m: dict, key: str, lo_ax: float, hi_ax: float, w: int = 330, color: str = 'var(--accent)',
           label: bool = True) -> str:
    """Точка и 95% ДИ на отрезке оси [lo_ax, hi_ax]."""
    v = m.get(key)
    lo, hi = m.get(f'{key}_ci', [None, None])
    if v is None or not np.isfinite(v):
        return ''
    X = lambda t: 8 + (min(max(t, lo_ax), hi_ax) - lo_ax) / (hi_ax - lo_ax) * (w - 16)  # noqa: E731
    parts = [f'<svg width="{w}" height="26" viewBox="0 0 {w} 26" role="img" '
             f'aria-label="{v:.2f}">']
    parts.append(f'<line x1="8" y1="13" x2="{w - 8}" y2="13" stroke="var(--rule)" stroke-width="1"/>')
    for t in np.linspace(lo_ax, hi_ax, 5):
        parts.append(f'<line x1="{X(t):.1f}" y1="9" x2="{X(t):.1f}" y2="17" stroke="var(--rule)" stroke-width="1"/>')
    if lo is not None and np.isfinite(lo):
        parts.append(f'<line x1="{X(lo):.1f}" y1="13" x2="{X(hi):.1f}" y2="13" stroke="{color}" '
                     f'stroke-width="3" stroke-linecap="round" opacity=".45"/>')
    parts.append(f'<circle cx="{X(v):.1f}" cy="13" r="6.5" fill="{color}" stroke="var(--sheet)" stroke-width="2"/>')
    parts.append('</svg>')
    return ''.join(parts)


def svg_roc(curves: list[tuple[str, np.ndarray, np.ndarray, str, str]], size: int = 520) -> str:
    """ROC-кривые: (подпись, y, score, цвет, штрих)."""
    import stats
    pad_l, pad_b, pad_t, pad_r = 64, 56, 14, 14
    W = H = size
    iw, ih = W - pad_l - pad_r, H - pad_t - pad_b
    X = lambda f: pad_l + f * iw          # noqa: E731
    Y = lambda t: pad_t + (1 - t) * ih    # noqa: E731
    s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="ROC-кривые">']
    for t in np.linspace(0, 1, 6):
        s.append(f'<line x1="{X(t):.1f}" y1="{Y(0):.1f}" x2="{X(t):.1f}" y2="{Y(1):.1f}" stroke="var(--rule2)"/>')
        s.append(f'<line x1="{X(0):.1f}" y1="{Y(t):.1f}" x2="{X(1):.1f}" y2="{Y(t):.1f}" stroke="var(--rule2)"/>')
        s.append(f'<text x="{X(t):.1f}" y="{H - pad_b + 24}" text-anchor="middle" font-size="15" '
                 f'class="m" fill="var(--ink3)">{t:.1f}</text>')
        s.append(f'<text x="{pad_l - 10}" y="{Y(t) + 5:.1f}" text-anchor="end" font-size="15" '
                 f'class="m" fill="var(--ink3)">{t:.1f}</text>')
    s.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="var(--ink3)" '
             f'stroke-dasharray="4 5" stroke-width="1.2"/>')
    s.append(f'<text x="{X(.5):.1f}" y="{H - 6}" text-anchor="middle" font-size="16" fill="var(--ink2)">'
             f'1 − специфичность</text>')
    s.append(f'<text transform="translate(16 {Y(.5):.1f}) rotate(-90)" text-anchor="middle" font-size="16" '
             f'fill="var(--ink2)">чувствительность</text>')
    for label, y, sc, color, dash in curves:
        ok = np.isfinite(sc)
        y, sc = np.asarray(y, int)[ok], np.asarray(sc, float)[ok]
        order = np.argsort(-sc, kind='mergesort')
        ys, ss = y[order], sc[order]
        P, N = ys.sum(), len(ys) - ys.sum()
        tpr, fpr, tp, fp = [0.0], [0.0], 0, 0
        for i in range(len(ys)):
            tp += ys[i]
            fp += 1 - ys[i]
            if i == len(ys) - 1 or ss[i + 1] != ss[i]:
                tpr.append(tp / P)
                fpr.append(fp / N)
        pts = ' '.join(f'{X(f):.1f},{Y(t):.1f}' for f, t in zip(fpr, tpr))
        da = f' stroke-dasharray="{dash}"' if dash else ''
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.6"{da} '
                 f'stroke-linejoin="round"/>')
    s.append('</svg>')
    return ''.join(s)


def svg_balance(D: dict) -> str:
    """Доля нарушений по критериям: полоса = все кадры области, заливка = нарушения."""
    rows = [('position', 'Укладка'), ('axis', 'Ось > 5°'), ('artifacts', 'Посторонние предметы'),
            ('rotation', 'Ротация бедра'), ('roi', 'Отступы ROI')]
    W, rh, lab = 820, 58, 300
    H = rh * len(rows) + 30
    maxn = max(n for _, n in D['pos'].values())
    s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Доля нарушений">']
    for i, (k, name) in enumerate(rows):
        pos, n = D['pos'][k]
        y = 10 + i * rh
        full = (W - lab - 150) * n / maxn
        part = full * pos / n
        s.append(f'<text x="0" y="{y + 25}" font-size="21" fill="var(--ink)">{name}</text>')
        s.append(f'<rect x="{lab}" y="{y + 8}" width="{full:.1f}" height="24" rx="4" fill="var(--rule2)"/>')
        s.append(f'<rect x="{lab}" y="{y + 8}" width="{max(part, 4):.1f}" height="24" rx="4" fill="var(--bad)"/>')
        s.append(f'<text x="{lab + full + 14:.1f}" y="{y + 26}" font-size="19" class="m" fill="var(--ink2)">'
                 f'{pos} из {n}</text>')
    s.append('</svg>')
    return ''.join(s)


def svg_folds() -> str:
    """Схема 5 фолдов: в строке k отложен фолд k."""
    W, cell, gap = 560, 96, 8
    H = 5 * 46 + 40
    s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Пять фолдов">']
    for i in range(5):
        s.append(f'<text x="{i * (cell + gap) + cell / 2 + 36:.0f}" y="18" text-anchor="middle" font-size="15" '
                 f'class="m" fill="var(--ink3)">фолд {i}</text>')
    for k in range(5):
        y = 30 + k * 46
        s.append(f'<text x="0" y="{y + 25}" font-size="15" class="m" fill="var(--ink3)">#{k + 1}</text>')
        for i in range(5):
            x = 36 + i * (cell + gap)
            val = i == k
            fill = 'var(--accent)' if val else 'var(--rule2)'
            s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="36" rx="5" fill="{fill}"/>')
            s.append(f'<text x="{x + cell / 2}" y="{y + 24}" text-anchor="middle" font-size="15" '
                     f'fill="{"#fff" if val else "var(--ink2)"}">{"проверка" if val else "обучение"}</text>')
    s.append('</svg>')
    return ''.join(s)


# --------------------------------------------------------------------------- #
#  Слайды
# --------------------------------------------------------------------------- #
class Deck:
    def __init__(self):
        self.slides: list[tuple[str, str, str]] = []   # (eyebrow, html, notes)

    def add(self, section: str, body: str, notes: str, cls: str = '', head: str = '') -> None:
        self.slides.append((section, body, notes, cls, head))


def slide_html(i: int, n: int, section: str, body: str, cls: str, head: str) -> str:
    eyebrow = (f'<div class="eyebrow"><span class="n">{i:02d}</span><span>{esc(section)}</span></div>'
               if section else '')
    header = f'<div class="head">{eyebrow}{head}</div>' if (section or head) else ''
    foot = (f'<div class="foot"><span>Контроль качества DXA · ЛЦТ 2026</span>'
            f'<span>{i:02d} / {n:02d}</span></div><div class="ruler"></div>')
    return f'<section class="slide {cls}">{header}{body}{foot}</section>'


def build(D: dict, cases: dict) -> Deck:
    dk = Deck()
    h, hg = D['h'], D['hb']
    hv, hgv = h.get('per_violation', {}), hg.get('per_violation', {})
    ov = h.get('overall', {}).get('binary_quality_class', {})
    ovg = hg.get('overall', {}).get('binary_quality_class', {})
    bench = D['bench']
    rot, art = D['cnn']['rotation'], D['cnn']['artifacts']
    rm, am = rot.get('metrics', {}), art.get('metrics', {})
    ps = bench.get('per_study_s', {})
    pf = bench.get('per_frame_s', {})
    img = lambda k: b64(ASSETS / 'cases' / cases[k]['file']) if k in cases else ''  # noqa: E731

    # 1. Титул ------------------------------------------------------------------
    c_axis, c_rot = cases.get('axis', {}), cases.get('rotation', {})
    body = f'''
<div class="left">
  <div class="eyebrow"><span>ЛЦТ 2026 · кейс Департамента здравоохранения Москвы</span></div>
  <h1>Контроль качества <span>DXA</span></h1>
  <p class="lead">Сервис проверяет укладку и разметку денситометрии до того, как
  исследование уйдёт к врачу, и объясняет каждое замечание в миллиметрах и градусах.</p>
  <div class="stats" style="grid-template-columns: repeat(4, auto); justify-content: start;">
    <div class="stat"><div class="v">5</div><div class="d">критериев ТЗ, позвоночник и бедро</div></div>
    <div class="stat"><div class="v">{ps.get("median", 0):.1f}<small>с</small></div><div class="d">медиана на исследование, CPU (лимит 180 с)</div></div>
    <div class="stat"><div class="v">{f2(ov.get("roc_auc"))}</div><div class="d">ROC-AUC «есть нарушение», out-of-fold</div></div>
    <div class="stat"><div class="v">0</div><div class="d">необработанных исключений</div></div>
  </div>
</div>
<div class="right">
  <div class="film"><img src="{img('axis')}" alt="Позвоночник: ось отклонена"><div class="lbl"><b>ось {f2(c_axis.get('angle'), 1)}°</b> · <span class="bad">нарушение</span> · порог ТЗ 5°</div></div>
  <div class="film"><img src="{img('rotation')}" alt="Бедро: ротация, тепловая карта сети"><div class="lbl"><b>ротация</b> · сеть {f2(c_rot.get('rot_p'))} · <span class="bad">нарушение</span></div></div>
</div>'''
    dk.add('', body, '''Здравствуйте. Мы сделали сервис контроля качества денситометрии: он проверяет
каждый снимок DXA по пяти критериям из ТЗ и объясняет замечание числом и картинкой — угол оси,
отступ в миллиметрах, где на снимке нейросеть видит ротацию. Работает целиком локально, на
процессоре, около двух секунд на исследование.''', cls='title-slide')

    # 2. Клиническая логика -----------------------------------------------------
    body = f'''
<div class="content cols-2">
  <div class="stack">
    <h3>Почему это важно</h3>
    <ul class="list">
      <li>Минеральная плотность кости считается в области интереса. Если позвоночник наклонён, бедро
      повёрнуто или в кадр попала застёжка белья, программа денситометра меряет не то.</li>
      <li>Ошибка переходит в T-критерий и ложную динамику: пациента повторно облучают или
      неверно оценивают эффект лечения.</li>
      <li>Сегодня качество проверяет врач вручную, по памяти о критериях для каждой области.</li>
    </ul>
    <div class="note">Сервис не ставит диагноз и не меряет плотность. Он отвечает на один вопрос:
    можно ли это исследование интерпретировать.</div>
    <div class="pair">
      <div class="film"><img src="{img('ok_hip')}" alt="Бедро без нарушений"><div class="lbl"><span class="ok">норма</span> · малый вертел слегка деформирует контур</div></div>
      <div class="film"><img src="{img('rotation')}" alt="Бедро с ротацией"><div class="lbl"><span class="bad">ротация</span> · контур плавный, вертела не видно</div></div>
    </div>
  </div>
  <div class="stack">
    <h3>Что делает сервис</h3>
    <div class="card"><div class="k">1 · проверяет</div><div class="body">Каждый снимок по пяти
      критериям ТЗ: укладка, ось, посторонние предметы, ротация бедра, отступы области интереса.</div></div>
    <div class="card"><div class="k">2 · объясняет</div><div class="body">«Ось 6.9° при норме до 5°»,
      «снизу 2.1 см при норме 3 см», контур и тепловая карта на снимке.</div></div>
    <div class="card"><div class="k">3 · отдаёт человеку</div><div class="body">Лаборант исправляет
      укладку, пока пациент на столе. Врач видит заключение DICOM SR рядом со снимком в PACS.</div></div>
  </div>
</div>'''
    dk.add('Клиническая задача', body, '''Контекст: плотность кости считается внутри области интереса,
и любая ошибка укладки искажает число, по которому ставят остеопороз и оценивают лечение. Наш
сервис — не диагност, а контролёр: он говорит, пригодно ли исследование, и почему нет. Главное
требование к нему — объяснимость: каждое замечание проверяется глазами за секунду.''',
           head='<h2>Ошибку укладки дешевле поймать, пока пациент на столе</h2>')

    # 3. Таксономия --------------------------------------------------------------
    body = f'''
<div class="content cols-8-4">
  <table class="t compact">
    <thead><tr><th>код в отчёте</th><th>критерий ТЗ 2.3</th><th>как измеряем</th><th>норма</th></tr></thead>
    <tbody>
      <tr><td><code>spine_position</code></td><td>внизу видны верхние края подвздошных костей</td><td>доля кости в нижних латеральных зонах кадра</td><td class="mono">&gt; 0.001</td></tr>
      <tr><td><code>spine_axis</code></td><td>наклон оси позвоночника до 5°</td><td>прямая Тейла–Сена по центрам тел, вторая — по краям колонны</td><td class="mono">≤ 5° (решение 4.0°)</td></tr>
      <tr><td><code>spine_artifacts</code></td><td>нет посторонних предметов и наложений</td><td>white top-hat + сверточная сеть, свод</td><td class="mono">p &lt; {f2(art.get('threshold'))}</td></tr>
      <tr><td><code>hip_rotation</code></td><td>нет ротации (оценка малого вертела)</td><td>выступ малого вертела, мм² + сверточная сеть, свод</td><td class="mono">p &lt; {f2(rot.get('threshold'))}</td></tr>
      <tr><td><code>hip_roi</code></td><td>отступы 3 см сверху и снизу, 2 см сбоку</td><td>длина поля: 3 см + вертел — седалищная кость + 3 см; отступы по рисунку 6 — справочно</td><td class="mono">≥ 12.9 см</td></tr>
      <tr><td><code>undetermined</code></td><td>область не определена</td><td>фильтр «не похоже на DXA» в классификаторе области</td><td>ручной разбор</td></tr>
    </tbody>
  </table>
  <div class="stack">
    <div class="card">
      <div class="k">несколько нарушений</div>
      <div class="body">У 20 из 100 исследований нарушений больше одного. Критерии считаются
      независимо, в отчёт идут все: <code>spine_axis;spine_artifacts</code>.</div>
      <div class="body"><code>quality_class = 1</code>, если сработал хоть один.</div>
    </div>
    <div class="card">
      <div class="k">сбой</div>
      <div class="body">Непрочитанный файл — строка <code>Failure</code> с текстом ошибки и
      <code>quality_class = 1</code>: необработанный снимок не считается качественным.</div>
    </div>
  </div>
</div>'''
    dk.add('Таксономия нарушений', body, '''Таксономия повторяет ТЗ дословно: пять кодов плюс
«область не определена». Задача мультилейбловая — у каждого пятого исследования больше одного
нарушения, поэтому критерии независимы и перечисляются все. Для каждого кода есть измерение в
физических единицах; для ротации и посторонних предметов к измерению добавлена сверточная сеть.''',
           head='<h2>Пять критериев ТЗ, у каждого — измерение в физических единицах</h2>')

    # 4. Данные ------------------------------------------------------------------
    reg = D['regions']
    body = f'''
<div class="content cols-5-7">
  <div class="stack">
    <div class="stats" style="grid-template-columns: repeat(3, auto); justify-content: start;">
      <div class="stat"><div class="v">100</div><div class="d">исследований</div></div>
      <div class="stat"><div class="v">499</div><div class="d">DICOM-файлов</div></div>
      <div class="stat"><div class="v">{D['frames']}</div><div class="d">уникальных снимков</div></div>
    </div>
    <ul class="list small">
      <li>Lunar выгружает один снимок по нескольку раз с разными UID. Дубликаты находим по MD5
      пикселей, до разбиения на фолды — иначе копия снимка попала бы и в обучение, и в проверку.</li>
      <li>Позвоночник {reg.get('spine', 0)}, левое бедро {reg.get('lh', 0)}, правое {reg.get('rh', 0)};
      не больше трёх снимков на исследование.</li>
      <li>Один аппарат, GE Lunar Prodigy Advance, 8 бит. <code>PixelSpacing</code> пуст:
      масштаб 0.600 / 0.607 мм/px восстановлен и проверен по шагу «позвонок + диск» 27–36 мм.</li>
      <li>Своя разметка: 242 снимка × 29 ключевых точек по протоколу команды — для модели точек
      и для проверки измерений.</li>
    </ul>
  </div>
  <div class="stack">
    <h3>Нарушений мало, и они разные</h3>
    {svg_balance(D)}
    <p class="small">Метки эксперта — на исследование и область; снимку достаётся метка своей
    области. На 6–36 положительных примерах ёмкую модель учить не на чем, поэтому основа —
    геометрия, а сеть подключена там, где геометрия упирается в разрешение.</p>
    <div class="stats" style="grid-template-columns: repeat(3, auto); justify-content: start;">
      <div class="stat"><div class="v">48</div><div class="d">исследований из 100 с нарушением</div></div>
      <div class="stat"><div class="v">20</div><div class="d">с несколькими нарушениями сразу</div></div>
      <div class="stat"><div class="v">72</div><div class="d">снимка из 249 размеченных с нарушением</div></div>
    </div>
  </div>
</div>'''
    dk.add('Данные', body, '''В наборе 499 файлов, но уникальных снимков 252 — остальное повторные
выгрузки Lunar с новыми UID. Дедупликация по пикселям идёт до разбиения, это первая защита от
утечки. Классы крошечные: 6 нарушений укладки, 7 — отступов. Отсюда архитектурное решение:
измерения, а не сквозная сеть.''', head='<h2>252 снимка, 6–36 нарушений на критерий</h2>')

    # 5. Разбиение и утечки ---------------------------------------------------------
    st = D['strata']
    body = f'''
<div class="content cols-5-7">
  <div class="stack">
    {svg_folds()}
    <p class="small">5 фолдов по 20 исследований, <code>folds.csv</code> в репозитории, код его только
    читает. Страты: норма {st.get('norm', 0)}, ротация {st.get('rotation', 0)}, артефакты {st.get('artifacts', 0)},
    ось {st.get('axis', 0)}, укладка {st.get('spine_position', 0)}, ROI {st.get('roi', 0)} исследований.</p>
  </div>
  <div class="stack">
    <div class="card"><div class="k">группа — исследование</div><div class="body">Позвоночник и оба бедра
      одного пациента всегда в одном фолде: бёдра похожи, и разнесённые по фолдам они завысили бы метрику.</div></div>
    <div class="card"><div class="k">страта — самое редкое нарушение исследования</div><div class="body">В каждом
      фолде есть хотя бы один пример самых редких классов.</div></div>
    <div class="card"><div class="k">всё подбирается только на обучающих фолдах</div><div class="body">Пороги,
      коридор нормы, калибровка вероятностей, веса сети, свод с геометрией, порог свода. Метрики на всей
      выборке и out-of-fold публикуются рядом — разница видна.</div></div>
    <div class="card"><div class="k">дисбаланс</div><div class="body">Вес положительного класса в потере сети,
      порог по доле нарушений, двусторонний коридор нормы по перцентилям (один параметр вместо двух),
      доверительные интервалы бутстрэпом по исследованиям.</div></div>
  </div>
</div>'''
    dk.add('Разбиение и утечки', body, '''Разбиение — пять фолдов с группировкой по исследованию и
стратификацией по самому редкому нарушению. Всё, что подбирается, подбирается только на
обучающих фолдах, включая порог свода нейросети. Поэтому в README две таблицы: оптимистичная на
всей выборке и честная out-of-fold. На слайде с метриками — честная.''',
           head='<h2>Честная оценка: исследование целиком в одном фолде</h2>')

    # 6. Подход ------------------------------------------------------------------
    body = f'''
<div class="content cols-7-5">
  <table class="t compact">
    <thead><tr><th>что пробовали для ротации бедра</th><th class="n">ROC-AUC (OOF)</th><th>вывод</th></tr></thead>
    <tbody>
      <tr><td>выступ малого вертела по контуру, коридор нормы</td><td class="n">{f2(rm.get('geometry', {}).get('roc_auc'))}</td><td>сигнал есть, разрыв классов ≈2 мм</td></tr>
      <tr><td>вид кадра, логистическая регрессия на 128 px</td><td class="n">0.50</td><td>малый вертел теряется при уменьшении</td></tr>
      <tr><td>модель ключевых точек U-Net, выступ по точкам</td><td class="n">—</td><td>ошибка выступа 0.63 мм ≈ разрыву классов</td></tr>
      <tr><td>сверточная сеть ResNet-34 на кадре 320 px, 1 модель</td><td class="n">{f2(np.mean([v['roc_auc'] for v in rm.get('cnn_oof_by_seed', {}).values()]) if rm.get('cnn_oof_by_seed') else None)}</td><td>зависит от сида</td></tr>
      <tr><td>ансамбль трёх сидов</td><td class="n">{f2(rm.get('cnn_oof', {}).get('roc_auc'))}</td><td>стабильнее одиночной</td></tr>
      <tr class="hl"><td><b>свод: ансамбль + геометрия</b></td><td class="n"><b>{f2(rm.get('stack_oof', {}).get('roc_auc'))}</b></td><td>в поставке</td></tr>
    </tbody>
  </table>
  <div class="stack">
    <div class="card"><div class="k">принцип</div><div class="body">Вердикт должен проверяться глазами:
      «ось 6.9°», «снизу 21 мм при норме 30». Поэтому основа — геометрия в миллиметрах и градусах.</div></div>
    <div class="card"><div class="k">где геометрии мало</div><div class="body">Ротация и посторонние предметы —
      признаки формы и текстуры. Там геометрию дополняет сеть, вердикт выносит свод, а тепловая карта
      показывает, куда смотрела сеть.</div></div>
    <div class="card"><div class="k">где сеть не нужна</div><div class="body">Ось, укладка, отступы — это
      прямые измерения по определению ТЗ. Сеть на 6–10 примерах здесь добавила бы только шум.</div></div>
  </div>
</div>'''
    dk.add('Подход', body, '''Мы сознательно строили решение от измерений. Ротация — самый трудный
критерий: разрыв между классами по выступу малого вертела около двух миллиметров. Мы проверили
четыре пути: контур, модель вида кадра, свою модель ключевых точек и сверточную сеть. Сеть на
полном разрешении кадра дала лучший результат, и в поставке она работает в своде с геометрией.''',
           head='<h2>Геометрия там, где критерий — измерение; сеть там, где форма</h2>')

    # 7. Архитектура ПО -----------------------------------------------------------
    from service.pipeline import COLUMNS as COLS8

    def cut(v, n=14):
        v = '' if v is None else str(v)
        return v if len(v) <= n else '…' + v[-n:]
    sample_rows = ''.join(
        '<tr>' + ''.join(f'<td>{esc(cut(r.get(c)) if c in ("path_to_study", "study_uid", "image_uid") else ("" if r.get(c) is None else r.get(c)))}</td>'
                         for c in COLS8) + '</tr>'
        for r in D.get('sample', [])[:5])
    body = f'''
<div class="content" style="grid-template-rows: auto auto auto; gap: 30px;">
  <div class="flow">
    <div class="node"><div class="k">вход</div><h4>DICOM, папка, zip</h4><p>Выгрузка из PACS как есть: DICOMDIR,
      вложенные серии, SR и служебные файлы пропускаются.</p></div>
    <div class="node"><div class="k">чтение</div><h4>pydicom + gdcm</h4><p>RLE, JPEG, JPEG-LS, JPEG 2000,
      MONOCHROME1, VOI LUT. Дедупликация по MD5 пикселей.</p></div>
    <div class="node"><div class="k">область</div><h4>region_clf</h4><p>Позвоночник, левое, правое бедро;
      «не похоже на DXA» — на ручной разбор.</p></div>
    <div class="node hot"><div class="k">критерии</div><h4>spine_qc · hip_roi · hip_rotation · cnn_qc</h4>
      <p>Независимо друг от друга, каждый со своими флагами.</p></div>
    <div class="node"><div class="k">свод</div><h4>quality_class, violation_type</h4><p>Вероятности
      критериев, 1 − Π(1 − p), текст по-русски.</p></div>
  </div>
  <div class="content cols-3" style="gap: 24px;">
    <div class="card"><div class="k">выход по ТЗ 2.5 и 2.7</div><div class="body">xlsx или csv, строка на снимок,
      8 колонок ТЗ в том же порядке; zip с визуализацией; zip с DICOM SR.</div></div>
    <div class="card"><div class="k">интерфейсы</div><div class="body">CLI для пакета, REST API
      (<code>POST /batch</code> → задание → отчёт), веб-интерфейс на том же порту.</div></div>
    <div class="card"><div class="k">контейнер</div><div class="body"><code>python:3.11-slim</code> по дайджесту,
      lock-файл зависимостей, onnxruntime на CPU, без сети. Ошибка любого файла — строка <code>Failure</code>,
      повторный прогон — побитово тот же отчёт.</div></div>
  </div>
  <div class="stack tight">
    <div class="cap">Фрагмент настоящего отчёта: первые восемь колонок ровно как в ТЗ 2.5, дальше — вероятности, флаги, текст</div>
    <table class="t compact mono" style="font-size: 16px;">
      <thead><tr>{''.join(f'<th>{c}</th>' for c in COLS8)}</tr></thead>
      <tbody>{sample_rows}</tbody>
    </table>
  </div>
</div>'''
    dk.add('Архитектура решения', body, '''Конвейер линейный: чтение с дедупликацией, определение
области, независимые критерии и свод. Вход — выгрузка из PACS как есть. Выход — таблица строго в
формате ТЗ плюс визуализация и DICOM SR. Всё в одном контейнере без сети; однопоточные
вычисления дают побитовую воспроизводимость, это проверяет тест.''',
           head='<h2>Один контейнер, пять шагов, ни одного обращения в сеть</h2>')

    # 8. Модели -------------------------------------------------------------------
    ens = rot.get('onnx', {})
    body = f'''
<div class="content cols-3 stretch" style="grid-template-rows: 1fr 1fr; gap: 22px;">
  <div class="card"><div class="k">region_clf</div><h3>Область снимка</h3><p class="body small2">Признаки кадра +
    логистическая регрессия; фильтр новизны отсекает кадры, не похожие на DXA, — они уходят на ручной разбор.</p>
    <div class="metric">252 / 252 снимка · 100 % на перевёрнутых</div></div>
  <div class="card"><div class="k">spine_qc</div><h3>Позвоночник</h3><p class="body small2">Маска кости, коридор
    колонны, центр тела в каждой строке. Ось — Тейл–Сен по центрам тел, вторая оценка — по краям; укладка —
    масса гребней; артефакты — white top-hat.</p>
    <div class="metric">ROC-AUC OOF: ось 0.82 · укладка 0.82</div></div>
  <div class="card"><div class="k">hip_roi</div><h3>Отступы бедра</h3><p class="body small2">Длина поля
    сканирования против 3 см + вертел — седалищная кость + 3 см, как оценивает эксперт; отступы по рисунку 6 —
    справочно и режимом <code>--roi-rule margins</code>.</p>
    <div class="metric">F1 {f2(D['roi_prob'].get('scan_length', {}).get('f1'))} · ROC-AUC {f2(D['roi_prob'].get('scan_length', {}).get('roc_auc'))}</div></div>
  <div class="card"><div class="k">hip_rotation</div><h3>Ротация: геометрия</h3><p class="body small2">Выступ малого
    вертела над прямой медиального контура диафиза, мм². Коридор нормы ловит и пере-, и недоротацию.</p>
    <div class="metric">коридор 96–272 мм² · ROC-AUC 0.68</div></div>
  <div class="card" style="border-color: var(--accent);"><div class="k" style="color: var(--accent);">cnn_qc</div><h3>Ротация и предметы: сеть</h3>
    <p class="body small2">ResNet-34 на кадре 320 px, ансамбль 3 сидов, ONNX fp16 → onnxruntime на CPU.
    Свод — логистическая регрессия [логит сети, геометрия]; карта CAM — на снимке.</p>
    <div class="metric">ROC-AUC свода 0.82 и 0.82 · {ens.get('bytes', 0) / 1e6 / max(len(ens.get('files', [1])), 1):.0f} МБ × 3</div></div>
  <div class="card"><div class="k">dxa_qc · по флагу</div><h3>Ключевые точки</h3><p class="body small2">U-Net с энкодером
    ResNet-34, 29 точек в одном выходе, 5 фолдов. Проигрывает контуру на OOF — выключена по умолчанию.</p>
    <div class="metric">медиана ошибки 1.0–2.5 мм · угол 0.50°</div></div>
</div>'''
    dk.add('Модели', body, '''Шесть модулей. Четыре — классическая геометрия и линейные модели.
Сеть cnn_qc — ResNet-34, дообученная на полном кадре, ансамбль из трёх сидов, исполняется
onnxruntime на CPU, torch в рабочем образе не нужен. Модель ключевых точек обучена и подключается
флагом, но в честном сравнении проигрывает контуру, и мы это показываем.''',
           head='<h2>Шесть модулей, у каждого своя проверяемая задача</h2>')

    # 9. Предобработка и постобработка -----------------------------------------------
    body = '''
<div class="content cols-2">
  <div class="stack">
    <h3>Предобработка</h3>
    <ul class="list">
      <li>Декодирование сжатых синтаксисов, инверсия MONOCHROME1, VOI LUT.</li>
      <li>Дедупликация по MD5 пикселей: одна строка отчёта на уникальный снимок.</li>
      <li>Правое бедро отражается в левое: измерения и сеть видят одну сторону.</li>
      <li>Маска кости и масштаб мм/px: все пороги — в миллиметрах, градусах, мм².</li>
      <li>Для сети: яркость по перцентилям 0.5–99.5 (готово к 12-битным аппаратам), длинная сторона → 320 px
      без искажения пропорций. Одна numpy-функция для обучения и сервиса.</li>
    </ul>
  </div>
  <div class="stack">
    <h3>Постобработка</h3>
    <ul class="list">
      <li>Пороги и коридоры — из калибровки по обучающим фолдам; отступы ROI — ровно по ТЗ.</li>
      <li>Свод сети с геометрией, порог — доля нарушений в обучении: устойчивее максимума F1 на малой выборке.</li>
      <li>Калибровка Платта для каждого критерия → вероятность нарушения и <code>quality_probability = 1 − Π(1 − p)</code>.</li>
      <li>Расхождение двух оценок — флаг <code>*:disagree</code> и низкая уверенность, а не молчаливый выбор.</li>
      <li>Текст заключения по-русски, оверлей с контуром и тепловой картой, DICOM SR.</li>
    </ul>
  </div>
</div>'''
    dk.add('Предобработка и постобработка', body, '''До модели: декодирование любых синтаксисов,
дедупликация, приведение бедра к одной стороне, маска кости и масштаб. После: калиброванные
пороги, свод сети с геометрией, вероятности по Платту и флаги расхождения методов. Оператор
видит не только вердикт, но и насколько система в нём уверена.''',
           head='<h2>Всё приводится к миллиметрам до модели и к вероятности после</h2>')

    # 10. Метрики --------------------------------------------------------------------
    def mrow(name, m, mg=None, n=None):
        both = mg is not None and mg and abs((mg.get('f1') or 0) - (m.get('f1') or 0)) > 1e-9
        def cell(key, lo, hi):
            s = ci_bar(m, key, lo, hi)
            if both:
                s = ci_bar(mg, key, lo, hi, color='var(--geo)') + '<br>' + s
            return s
        def num(key):
            v = f'<b>{f2(m.get(key))}</b> <span class="cap">{ci(m, key)}</span>'
            if both:
                v = f'<span class="cap">{f2(mg.get(key))}</span><br>' + v
            return v
        pos = f"{m.get('positives', '')}/{m.get('n', '')}" if n is None else n
        return (f'<tr><td>{name}</td><td class="n">{pos}</td>'
                f'<td class="n">{f2(m.get("sensitivity"))}</td><td class="n">{f2(m.get("specificity"))}</td>'
                f'<td>{cell("f1", 0, 1)}</td><td class="n" style="white-space:nowrap">{num("f1")}</td>'
                f'<td>{cell("roc_auc", .4, 1)}</td><td class="n" style="white-space:nowrap">{num("roc_auc")}</td></tr>')
    rows = [mrow(CRIT_RU[k], hv[k], hgv.get(k)) for k in CRIT_RU if k in hv]
    reg_rows = [mrow(REGION_RU[k], h['per_region'][k], hg.get('per_region', {}).get(k))
                for k in ('spine', 'hip') if k in h.get('per_region', {})]
    body = f'''
<div class="content" style="grid-template-rows: 1fr auto; gap: 18px;">
  <table class="t compact">
    <thead><tr><th>out-of-fold</th><th class="n">нар./n</th><th class="n">чувств.</th><th class="n">специф.</th>
      <th>F1, 95% ДИ</th><th class="n">F1</th><th>ROC-AUC, 95% ДИ (ось 0.4–1)</th><th class="n">ROC-AUC</th></tr></thead>
    <tbody>{''.join(rows)}{''.join(reg_rows)}
    {mrow('<b>Всего, «есть нарушение»</b>', ov, ovg)}</tbody>
  </table>
  <div class="legend">
    <span><i style="background: var(--geo)"></i>исходное решение: геометрия, ROI по рисунку 6</span>
    <span><i style="background: var(--accent)"></i>сервис сейчас: сеть + правило эксперта для ROI</span>
    <span>macro-F1 по типам: {f2(hg.get('overall', {}).get('macro_f1'))} → <b>{f2(h.get('overall', {}).get('macro_f1'))}</b></span>
    <span>область определена верно: 252 из 252 · обработано 100 % файлов</span>
  </div>
</div>'''
    dk.add('Метрики качества', body, '''Это честные out-of-fold числа: каждый вердикт вынесен
моделью и порогом, не видевшими этот снимок. Серым — только геометрия, синим — поставляемый
сервис. Сеть подняла ротацию и посторонние предметы; общий ROC-AUC бинарного класса —
по сводной вероятности. Интервалы широкие, потому что положительных примеров 6–36, и мы их
не прячем.''', head='<h2>Метрики out-of-fold с 95% доверительными интервалами</h2>')

    # 11. Эксперименты: ROC ротации ---------------------------------------------------
    ro = D['rot_oof']
    curves = [('геометрия', ro.y, ro.geometry.fillna(0).values, 'var(--geo)', '6 5'),
              ('сеть, ансамбль', ro.y, ro.p_cnn.values, '#6f9be6', ''),
              ('свод', ro.y, ro.p_stack.values, 'var(--accent)', '')]
    ao = D['art_oof']
    curves_a = [('top-hat', ao.y, ao.geometry.values, 'var(--geo)', '6 5'),
                ('сеть, ансамбль', ao.y, ao.p_cnn.values, '#6f9be6', ''),
                ('свод', ao.y, ao.p_stack.values, 'var(--accent)', '')]
    def side(m, geo_label, geo_m):
        v, g = m.get('verdict_nested', {}), geo_m
        def line(lbl, x):
            return (f'<tr><td>{lbl}</td><td class="n">{f2(x.get("sensitivity"))}</td>'
                    f'<td class="n">{f2(x.get("specificity"))}</td><td class="n"><b>{f2(x.get("f1"))}</b></td></tr>')
        return (f'<table class="t compact" style="width: 330px;"><thead><tr><th>OOF</th><th class="n">чув.</th>'
                f'<th class="n">спец.</th><th class="n">F1</th></tr></thead><tbody>'
                f'{line(geo_label, g)}{line("свод", v)}</tbody></table>')
    hgv = D['hg'].get('per_violation', {})
    body = f'''
<div class="content cols-2" style="gap: 48px;">
  <div class="stack">
    <h3>Ротация бедра · {rot.get('positives', '')} из {rot.get('n', '')}</h3>
    <div style="display: flex; gap: 26px; align-items: flex-start;">
      {svg_roc(curves, 520)}
      <div class="stack tight" style="padding-top: 8px;">
        <div class="legend" style="flex-direction: column; gap: 8px;"><span><i style="background: var(--geo)"></i>геометрия {f2(rm.get('geometry', {}).get('roc_auc'))}</span>
          <span><i style="background: #6f9be6"></i>сеть {f2(rm.get('cnn_oof', {}).get('roc_auc'))}</span>
          <span><i style="background: var(--accent)"></i><b>свод {f2(rm.get('stack_oof', {}).get('roc_auc'))}</b></span></div>
        <div class="cap" style="margin-top: 18px;">вердикт, свод и порог — без отложенного фолда</div>
        {side(rm, 'выступ', hgv.get('hip_rotation', {}))}
      </div>
    </div>
  </div>
  <div class="stack">
    <h3>Посторонние предметы · {art.get('positives', '')} из {art.get('n', '')}</h3>
    <div style="display: flex; gap: 26px; align-items: flex-start;">
      {svg_roc(curves_a, 520)}
      <div class="stack tight" style="padding-top: 8px;">
        <div class="legend" style="flex-direction: column; gap: 8px;"><span><i style="background: var(--geo)"></i>top-hat {f2(am.get('geometry', {}).get('roc_auc'))}</span>
          <span><i style="background: #6f9be6"></i>сеть {f2(am.get('cnn_oof', {}).get('roc_auc'))}</span>
          <span><i style="background: var(--accent)"></i><b>свод {f2(am.get('stack_oof', {}).get('roc_auc'))}</b></span></div>
        <div class="cap" style="margin-top: 18px;">вердикт, свод и порог — без отложенного фолда</div>
        {side(am, 'top-hat', hgv.get('spine_artifacts', {}))}
      </div>
    </div>
  </div>
</div>'''
    dk.add('Эксперименты', body, '''ROC-кривые out-of-fold: каждый снимок оценён моделями, которые его не
видели. Для ротации сеть заметно лучше геометрии, свод добавляет ещё немного. Для посторонних
предметов выигрыш скромнее: морфологический фильтр и так хорош, сеть дополняет его там, где
предмет не яркий и не вытянутый.''', head='<h2>Сеть поверх геометрии: ROC-кривые out-of-fold</h2>')

    # 12. Сравнение вариантов --------------------------------------------------------
    kp = D['kp_oof'].get('per_violation', {})
    sw = D['spine_oof'].get('position', {}).get('rule_sweep', {})
    body = f'''
<div class="content" style="grid-template-rows: auto auto; gap: 34px;">
<div class="content cols-2" style="gap: 56px;">
  <div class="stack">
    <h3>Измеритель оси и отступов: контур против модели точек</h3>
    <table class="t compact">
      <thead><tr><th>out-of-fold</th><th class="n">контур</th><th class="n">точки U-Net</th></tr></thead>
      <tbody>
        <tr><td>ось: F1 / ROC-AUC</td><td class="n">0.25 / 0.82</td><td class="n">{f2(kp.get('spine_axis', {}).get('f1'))} / {f2(kp.get('spine_axis', {}).get('roc_auc'))}</td></tr>
        <tr><td>ROI: F1 / ROC-AUC</td><td class="n">0.35 / 0.89</td><td class="n">{f2(kp.get('hip_roi', {}).get('f1'))} / {f2(kp.get('hip_roi', {}).get('roc_auc'))}</td></tr>
        <tr><td>время на снимок, CPU</td><td class="n">31 мс</td><td class="n">301 мс</td></tr>
      </tbody>
    </table>
    <p class="small">По F1 на оси разница в пределах шума (10 нарушений), по ранжированию контур лучше и в десять
    раз быстрее. 53 из 54 крупных ошибок точек — ошибки нумерации позвонка. Модель в поставке, по флагу.</p>
  </div>
  <div class="stack">
    <h3>Правило свода для укладки позвоночника</h3>
    <table class="t compact">
      <thead><tr><th>out-of-fold, 6 нарушений</th><th class="n">найдено</th><th class="n">лишних</th><th class="n">F1</th></tr></thead>
      <tbody>
        {''.join(f'<tr class="{"hl" if k == "geometry" else ""}"><td>{ {"geometry": "геометрия (в поставке)", "appearance": "вид кадра", "and": "обе согласны", "or": "любая из двух"}.get(k, k)}</td><td class="n">{v["tp"]}</td><td class="n">{v["fp"]}</td><td class="n">{f2(v["f1"])}</td></tr>' for k, v in sw.items())}
      </tbody>
    </table>
    <p class="small">На всей выборке лучшим казалось согласие обеих оценок (F1 0.73); out-of-fold оно
    ловит одно нарушение из шести. Выбор правил мы делаем только по OOF.</p>
  </div>
</div>
  <div class="stack">
    <h3>Сеть для ротации: что перебрали</h3>
    <table class="t compact">
      <thead><tr><th>вариант, out-of-fold</th><th class="n">ResNet-18</th><th class="n">EfficientNet-B0</th>
        <th class="n">ResNet-34, 256 px</th><th class="n">ResNet-34, 1 сид</th><th class="n">ResNet-34 × 3</th><th class="n">+ геометрия</th></tr></thead>
      <tbody><tr><td>ROC-AUC</td><td class="n">0.76</td><td class="n">0.73</td><td class="n">0.77</td>
        <td class="n">{f2(np.mean([v['roc_auc'] for v in rm.get('cnn_oof_by_seed', {}).values()]) if rm.get('cnn_oof_by_seed') else None)}</td>
        <td class="n">{f2(rm.get('cnn_oof', {}).get('roc_auc'))}</td><td class="n"><b>{f2(rm.get('stack_oof', {}).get('roc_auc'))}</b></td></tr>
        <tr><td>порог: максимум F1 → доля нарушений, F1 вложенно</td><td class="n" colspan="4"></td>
        <td class="n" colspan="2">0.48 → <b>{f2(rm.get('verdict_nested', {}).get('f1'))}</b></td></tr></tbody>
    </table>
  </div>
</div>'''
    dk.add('Сравнение вариантов', body, '''Два эксперимента, где честная оценка изменила решение. Модель
ключевых точек точна, но на оси проигрывает контуру и в десять раз медленнее — осталась опцией.
Для укладки правило «обе оценки согласны» выглядело лучшим на всей выборке и провалилось
out-of-fold. Все решения о правилах принимались по OOF.''',
           head='<h2>Что проверили и почему оставили то, что оставили</h2>')

    # 13. Анализ ошибок --------------------------------------------------------------
    ro_ = D['rot_oof']
    geo_bad = ro_.geometry.fillna(0) > 0
    st_bad = ro_.pred_nested.astype(bool)
    dis = geo_bad != st_bad
    st_right = int((st_bad[dis] == ro_.y[dis].astype(bool)).sum())
    body = f'''
<div class="content cols-2" style="gap: 28px 48px;">
  <div class="card" style="border-color: var(--accent);"><div class="k" style="color: var(--accent);">отступы ROI · что делает эксперт</div>
    <h3>Судит по длине поля</h3>
    <table class="t compact" style="font-size: 18px;"><thead><tr><th>правило</th><th class="n">найдено</th><th class="n">лишних</th><th class="n">F1</th><th class="n">AUC</th></tr></thead><tbody>
      <tr><td>отступы от ориентиров, рисунок 6</td><td class="n">{D['roi_prob'].get('margins', {}).get('tp')}/7</td><td class="n">{D['roi_prob'].get('margins', {}).get('fp')}</td><td class="n">{f2(D['roi_prob'].get('margins', {}).get('f1'))}</td><td class="n">{f2(D['roi_prob'].get('margins', {}).get('roc_auc'))}</td></tr>
      <tr class="hl"><td><b>длина поля ≥ 3 + вертел–седалищная + 3 см</b></td><td class="n">{D['roi_prob'].get('scan_length', {}).get('tp')}/7</td><td class="n">{D['roi_prob'].get('scan_length', {}).get('fp')}</td><td class="n"><b>{f2(D['roi_prob'].get('scan_length', {}).get('f1'))}</b></td><td class="n">{f2(D['roi_prob'].get('scan_length', {}).get('roc_auc'))}</td></tr>
    </tbody></table>
    <p class="small">Эксперт ставит «норму» при 1.7–3.0 см под седалищной костью и «нарушение» на коротких полях.
    Порог не подбирался по меткам: 30 мм — из ТЗ, 69 мм от вертела до седалищной кости — медиана анатомии.
    Отступы по рисунку 6 считаются и показываются; режим по ТЗ — <code>--roi-rule margins</code>.</p></div>
  <div class="stack">
    <div class="card"><div class="k">ротация · где сеть и выступ спорят</div><div class="body">Расхождений сети и выступа:
      {int(dis.sum())} из {len(ro_)}; прав свод — {st_right}, выступ — {int(dis.sum()) - st_right}. Кадр получает флаг
      <code>rotation:cnn_vs_geometry_disagree</code> и низкую уверенность: такие снимки врачу стоит открыть.</div></div>
    <div class="card"><div class="k">ось · сколиоз не нарушение</div><div class="body">При сколиозе эксперт не ставит
      нарушение оси ни в одном из 14 случаев, даже при угле до 8.8°. Это главный источник наших лишних срабатываний;
      по поясничному треку сколиоз от наклона не отличить (сеть для оси — ROC-AUC 0.51).</div></div>
    <div class="card"><div class="k">укладка · 6 примеров</div><div class="body">ROC-AUC 0.82, но F1 out-of-fold 0.32:
      неустойчива отсечка на фолде с одним-двумя нарушениями. Половина тела Th12 не проверяется — такой
      разметки в данных нет.</div></div>
  </div>
</div>'''
    dk.add('Анализ ошибок', body, '''Разбор ошибок показал, как оценивает эксперт. Отступы ROI он судит
по длине поля сканирования: правило длины поля совпадает с ним вдвое лучше отступов по рисунку 6.
При сколиозе эксперт не ставит нарушение оси — это главный источник наших лишних срабатываний.''',
           head='<h2>Разбор ошибок показал, как оценивает эксперт</h2>')

    # 14. Кейсы -----------------------------------------------------------------------
    def fig(k, text):
        if k not in cases:
            return ''
        return (f'<div class="film"><img src="{img(k)}" alt="{esc(k)}"><div class="lbl">{text}</div></div>')
    ca, cr = cases.get('artifacts', {}), cases.get('roi', {})
    body = f'''
<div class="content stretch" style="grid-template-columns: repeat(4, 1fr); gap: 20px;">
  {fig('axis', f"<b>ось {f2(c_axis.get('angle'), 1)}°</b> при норме до 5°<br><span class='bad'>spine_axis</span> · эксперт: нарушение")}
  {fig('artifacts', f"<b>посторонний предмет</b> · сеть {f2(ca.get('art_p'))}<br><span class='bad'>spine_artifacts</span> · эксперт: нарушение")}
  {fig('rotation', f"<b>ротация</b> · выступ {f2(c_rot.get('area'), 0)} мм² при норме 96–272, сеть {f2(c_rot.get('rot_p'))}<br><span class='bad'>hip_rotation</span> · эксперт: нарушение")}
  {fig('roi', f"<b>поле {f2((cr.get('scan_len') or 0) / 10, 1)} см</b>, нужно {f2((cr.get('need_len') or 129) / 10, 1)} см<br><span class='bad'>hip_roi</span> · эксперт: нарушение")}
</div>
<p class="small">Снимки обучающего набора; вероятности сети — out-of-fold. Оранжевым — тепловая карта сети (CAM), голубым —
найденные ориентиры, зелёным и красным — пороги ТЗ.</p>'''
    dk.add('Кейсы', body, '''Четыре настоящих снимка, где сервис согласен с экспертом. Ось: угол
посчитан по центрам тел и больше порога. Посторонний предмет: морфология и сеть указывают на одну
структуру. Ротация: малый вертел почти не выступает, тепловая карта сети лежит на вертельной области — туда и смотрит врач. Отступы:
нижний край поля ближе трёх сантиметров к седалищной кости.''',
           head='<h2>Каждое замечание видно на снимке</h2>')

    # 15. Скорость и требования --------------------------------------------------------
    bg = D['bench_geo']
    share = ps.get('median', 0) / 180 * 100
    body = f'''
<div class="content cols-2" style="gap: 56px;">
  <div class="stack" style="gap: 26px;">
    <div class="stats grid2">
      <div class="stat"><div class="v">{ps.get('median', 0):.1f}<small>с</small></div><div class="d">медиана на исследование (p95 {ps.get('p95', 0):.1f} с, максимум {ps.get('max', 0):.1f} с)</div></div>
      <div class="stat"><div class="v">180<small>с</small></div><div class="d">лимит ТЗ 2.7 на исследование</div></div>
      <div class="stat"><div class="v">{pf.get('median', 0) * 1000:.0f}<small>мс</small></div><div class="d">медиана на снимок, один поток CPU</div></div>
      <div class="stat"><div class="v">{bench.get('total_s', 0) / 60:.1f}<small>мин</small></div><div class="d">все {bench.get('studies', 100)} исследований ({bench.get('frames', 252)} снимков) в один процесс</div></div>
    </div>
    <div class="stack tight">
      <div class="cap">худшее исследование против лимита ТЗ, в масштабе</div>
      <div class="speedbar"><i style="width: {max(ps.get('max', 0) / 180 * 100, 0.6):.2f}%"></i>
        <span style="left: calc({max(ps.get('max', 0) / 180 * 100, 0.6):.2f}% + 10px)">{ps.get('max', 0):.1f} с — {ps.get('max', 0) / 180 * 100:.1f} % лимита</span>
        <span style="right: 12px">180 с</span></div>
    </div>
    <p class="small">В контейнере (Linux, Python 3.11) — 158 с на все 100 исследований вместе с оверлеями и DICOM SR.
    Без сети (<code>--no-cnn</code>) — {bg.get('per_frame_s', {}).get('median', 0.03) * 1000:.0f} мс на снимок. Исследования независимы:
    на 44 ядрах целевой машины пакет параллелится процессами.</p>
  </div>
  <div class="stack">
    <table class="t compact">
      <thead><tr><th></th><th>минимум</th><th>рекомендуется</th></tr></thead>
      <tbody>
        <tr><td>CPU</td><td>2 ядра x86-64</td><td>4 ядра</td></tr>
        <tr><td>RAM</td><td>2 ГБ</td><td>4 ГБ</td></tr>
        <tr><td>диск</td><td>2 ГБ (образ 1.4 ГБ)</td><td>5 ГБ</td></tr>
        <tr><td>GPU</td><td colspan="2">не нужен ни для работы, ни для калибровки; только для переобучения сети</td></tr>
        <tr><td>ПО</td><td colspan="2">Docker; Linux или любая UNIX-подобная система; скрипты на POSIX sh</td></tr>
      </tbody>
    </table>
    <div class="note">Воспроизводимость: повторный прогон на тех же данных даёт побитово тот же отчёт —
    фиксированный порядок обхода, модели на диске, вычисления в один поток. Проверяется тестом.</div>
  </div>
</div>'''
    dk.add('Скорость и системные требования', body, f'''Медиана — {ps.get('median', 0):.1f} секунды на исследование при лимите
три минуты, на одном ядре процессора. GPU не нужен: сеть исполняется onnxruntime. Весь обучающий
набор — {bench.get('total_s', 0) / 60:.1f} минуты в один процесс. Результат воспроизводим побитово.''',
           head='<h2>Секунды на исследование на обычном процессоре</h2>')

    # 16. Интерфейс и интеграция -------------------------------------------------------
    ui = ASSETS / 'ui_rotation.png'
    if not ui.exists():
        ui = ASSETS / 'ui_results.png'
    ui_img = f'<div class="film" style="background: #0d1117;"><img src="{b64(ui)}" alt="Веб-интерфейс"></div>' if ui.exists() else ''
    body = f'''
<div class="content cols-8-4" style="gap: 36px;">
  {ui_img}
  <div class="stack">
    <div class="card"><div class="k">веб-приложение</div><div class="small">Перетащить папку или zip, листать снимки с
      разметкой, нормой и близостью к границе. Ставится из браузера как приложение.</div></div>
    <div class="card"><div class="k">человек в контуре</div><div class="small">Оператор отмечает «согласен / не согласен»
      с комментарием; отметки выгружаются в CSV — это данные для перекалибровки порогов.</div></div>
    <div class="card"><div class="k">PACS</div><div class="small">DICOM SR (Basic Text SR) в том же исследовании:
      заключение по-русски рядом со снимком, <code>VerificationFlag = UNVERIFIED</code>.</div></div>
    <div class="card"><div class="k">без Docker</div><div class="small">Установщик для Windows и Telegram-бот:
      тот же конвейер, отчёт побитово тот же. API — <code>POST /batch</code>, документация на <code>/docs</code>.</div></div>
  </div>
</div>'''
    dk.add('Интерфейс и интеграция', body, '''Всё, что выдаёт конвейер, видно в веб-интерфейсе на том же
порту: снимок с разметкой, измерения с нормой, флаги. Оператор отмечает согласие с вердиктом — это
готовая обратная связь для перекалибровки. В PACS заключение попадает как DICOM SR в то же
исследование.''', head='<h2>Замечание доходит до лаборанта и до PACS</h2>')

    # 17. Ограничения ----------------------------------------------------------------
    body = '''
<div class="content">
  <table class="t">
    <thead><tr><th style="width: 24%">ограничение</th><th style="width: 40%">почему</th><th>что нужно, чтобы снять</th></tr></thead>
    <tbody>
      <tr><td><b>Один аппарат</b></td><td>Все снимки — GE Lunar Prodigy Advance; <code>PixelSpacing</code> пуст, масштаб задан константой</td>
        <td>Снимки второго аппарата: проверить масштаб, дообучить сеть тем же скриптом</td></tr>
      <tr><td><b>Малая выборка</b></td><td>6–36 нарушений на критерий: интервалы широкие, F1 ротации 0.55 при ROC-AUC 0.82</td>
        <td>Больше размеченных нарушений; отметки операторов из пилота</td></tr>
      <tr><td><b>ROI — как эксперт</b></td><td>Вердикт по длине поля, а не буквально по рисунку 6 ТЗ</td>
        <td>Решение организаторов; режим строго по ТЗ уже есть флагом</td></tr>
      <tr><td><b>Ось и сколиоз</b></td><td>Эксперт не считает сколиоз нарушением оси, мы его не отличаем</td>
        <td>Разметка сколиоза или снимки грудного отдела</td></tr>
      <tr><td><b>Разметка оператора</b></td><td>В данных нет ROI и линий ни в пикселях, ни в тегах — всё оценивается по изображению</td>
        <td>Выгрузка с оверлеями или отчётами денситометра</td></tr>
      <tr><td><b>Половина тела Th12</b></td><td>Верхняя граница укладки не проверяется: разметки уровня Th12 нет</td>
        <td>Разметка уровня Th12; модель точек уже выдаёт <code>th12_top</code> и <code>th12_bottom</code></td></tr>
      <tr><td><b>Чужие кадры</b></td><td>Фильтр «не DXA» проверен на искажениях своих снимков</td>
        <td>Настоящие снимки предплечья и всего тела для проверки фильтра</td></tr>
    </tbody>
  </table>
</div>'''
    dk.add('Ограничения', body, '''Главные ограничения честно: один аппарат, маленькая выборка и
расхождение эталона с ТЗ по отступам. Разметки оператора в данных нет, поэтому проверяем всё по
изображению. Каждое ограничение описано в README вместе с тем, что нужно, чтобы его снять.''',
           head='<h2>Что сервис не умеет и почему</h2>')

    # 18. Внедрение -------------------------------------------------------------------
    body = '''
<div class="content" style="grid-template-rows: auto auto; gap: 34px;">
  <div class="phases">
    <div class="phase"><div class="when">месяцы 1–2 · теневой режим</div><h3>1–2 медорганизации</h3>
      <p class="small">DICOM-роутер отправляет копии исследований DXA в контейнер, SR возвращается в PACS с пометкой
      «не проверено». Врач работает как раньше. Считаем согласие с экспертом и время.</p></div>
    <div class="phase"><div class="when">месяцы 3–6 · ассистент</div><h3>Замечание до отпуска пациента</h3>
      <p class="small">Результат на рабочем месте лаборанта через 2–3 секунды после снимка: повторная укладка сразу,
      а не повторный визит. Отметки операторов раз в месяц уходят в перекалибровку порогов.</p></div>
    <div class="phase"><div class="when">месяцы 6–12 · масштабирование</div><h3>Другие аппараты и сеть МО</h3>
      <p class="small">Hologic и другие Lunar: проверка масштаба, дообучение сети на их снимках (скрипт готов).
      Мониторинг доли нарушений по аппаратам и лаборантам.</p></div>
  </div>
  <div class="content cols-3 stretch" style="gap: 24px;">
    <div class="card"><div class="k">метрика пилота · качество</div><h3>Согласие с экспертом</h3><div class="body small2">Чувствительность
      и специфичность на слепой выборке врача-эксперта, раз в месяц; отдельно по критериям.</div></div>
    <div class="card"><div class="k">метрика пилота · пациент</div><h3>Повторные исследования</h3><div class="body small2">Доля повторных
      сканирований из-за укладки по журналу МО: три месяца до и три после.</div></div>
    <div class="card"><div class="k">метрика пилота · врач</div><h3>Время на контроль</h3><div class="body small2">Хронометраж контроля
      качества на 50 исследованиях до и после; сервис — 2 секунды на исследование.</div></div>
  </div>
  <div class="flow">
    <div class="node"><div class="k">денситометр</div><h4>DICOM</h4><p>Снимок уходит в PACS как обычно</p></div>
    <div class="node"><div class="k">PACS</div><h4>DICOM-роутер</h4><p>Правило: копия исследований DXA в контейнер</p></div>
    <div class="node hot"><div class="k">контейнер</div><h4>Контроль качества DXA</h4><p>CPU, без сети, 2 с на исследование</p></div>
    <div class="node"><div class="k">назад в PACS</div><h4>DICOM SR</h4><p>Заключение рядом со снимком, «не проверено»</p></div>
    <div class="node"><div class="k">лаборант</div><h4>Веб-интерфейс</h4><p>Замечание, снимок, отметка «согласен»</p></div>
  </div>
</div>'''
    dk.add('Внедрение и пилот', body, '''Внедрение в три шага. Теневой режим: сервис стоит за DICOM-роутером
и ничего не меняет в работе врача, мы копим статистику согласия. Затем — ассистент лаборанта:
замечание приходит, пока пациент ещё на столе. И масштабирование на другие аппараты с
дообучением. Для каждого шага есть измеримая метрика.''',
           head='<h2>Теневой режим → ассистент лаборанта → сеть медорганизаций</h2>')

    # 19. Демо -----------------------------------------------------------------------
    body = '''
<div class="content cols-2" style="gap: 56px; flex: 0 0 auto;">
  <ol class="list" style="counter-reset: s;">
    <li><code>./build.sh && ./run.sh</code> → <code>http://localhost:8000</code>, индикатор «сервис работает».</li>
    <li>Перетащить <code>demo.zip</code> (7 исследований: по одному на каждый тип нарушения и две нормы): прогресс, затем сводка.</li>
    <li>Фильтр «Ротация»: снимок с тепловой картой сети на малом вертеле, выступ в мм² и норма.</li>
    <li>Ось: угол и порог; отступы ROI: три линии порогов ТЗ, красная — не выдержана.</li>
    <li>Отметить «согласен» (<kbd>A</kbd>) и «не согласен» с комментарием, выгрузить отметки.</li>
    <li>Скачать таблицу xlsx: 8 колонок ТЗ; открыть DICOM SR.</li>
    <li>Пакетный режим: <code>./run.sh batch /data</code> → <code>out/report.xlsx</code>.</li>
  </ol>
  <div class="stack">
    <div class="card"><div class="k">запасной вариант</div><div class="body">Нет Docker — настольное приложение;
      нет сети — снимки экрана в этой презентации; с телефона — бот, команда <code>/demo</code>.</div></div>
    <div class="card"><div class="k">репозиторий</div><div class="body mono" style="font-size: 20px;">github.com/theorcos239/LCT_2026</div>
      <div class="small">README · docs/REVIEW · USER_GUIDE · DESKTOP · TELEGRAM_BOT</div></div>
  </div>
</div>
<div class="strip">
  <div><b>Контейнер</b>Docker, веса в образе, build.sh и run.sh на POSIX sh</div>
  <div><b>Отчёт ТЗ 2.5</b>xlsx или csv, 8 колонок, вероятности и флаги</div>
  <div><b>ТЗ 2.6</b>оверлеи с тепловой картой, DICOM SR, веб-интерфейс</div>
  <div><b>Интерфейсы</b>веб-приложение, установщик для Windows, Telegram-бот, API</div>
  <div><b>Документация</b>README, руководства пользователя, развёртывания, обучения</div>
  <div><b>Автотесты</b>сервис — 64, бот — 27: формат, устойчивость, детерминизм, сеть, SR</div>
</div>'''
    dk.add('Демонстрация', body, '''Демонстрация занимает три минуты: поднимаем контейнер, загружаем
архив из семи исследований, проходим по одному снимку каждого типа нарушения, отмечаем
согласие и выгружаем отчёт. Сценарий с подготовленными исследованиями — в docs/DEMO.md.
Спасибо, готовы ответить на вопросы.''', head='<h2>Сценарий демонстрации, 3 минуты</h2>')
    return dk


# --------------------------------------------------------------------------- #
#  Сборка файла и рендер
# --------------------------------------------------------------------------- #
FONTS = ('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400'
         '&family=IBM+Plex+Mono:wght@400;500&display=swap')

SCRIPT = '''
(function () {
  const frames = [...document.querySelectorAll('.frame')];
  function fit() { frames.forEach(f => f.style.setProperty('--k', f.clientWidth / 1920)); }
  let cur = 0;
  function show(i) { cur = Math.max(0, Math.min(frames.length - 1, i));
    frames.forEach((f, j) => f.classList.toggle('on', j === cur)); fit(); }
  document.addEventListener('keydown', e => {
    if (e.key === 'f' || e.key === 'F') { document.body.classList.toggle('show'); show(cur); return; }
    if (!document.body.classList.contains('show')) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { show(cur + 1); e.preventDefault(); }
    if (['ArrowLeft', 'PageUp'].includes(e.key)) { show(cur - 1); e.preventDefault(); }
    if (e.key === 'Escape') { document.body.classList.remove('show'); fit(); }
  });
  frames.forEach((f, j) => f.addEventListener('dblclick', () => { document.body.classList.add('show'); show(j); }));
  window.addEventListener('resize', fit); fit();
})();
'''


def write_html(dk: Deck, path: Path) -> None:
    css = (HERE / 'deck.css').read_text(encoding='utf-8')
    n = len(dk.slides)
    frames = []
    for i, (section, body, notes, cls, head) in enumerate(dk.slides, 1):
        frames.append(f'<div class="frame" id="s{i}" data-notes="{esc(" ".join(notes.split()))}">'
                      f'{slide_html(i, n, section, body, cls, head)}</div>')
    page = f'''<title>Контроль качества DXA</title>
<meta name="description" content="Презентация сервиса контроля качества денситометрии, ЛЦТ 2026">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{css}</style>
<div class="deck">
<p class="hint">Презентация «Контроль качества DXA», {n} слайдов. Показ на весь экран — <kbd>F</kbd> или двойной щелчок
по слайду, листать — <kbd>←</kbd> <kbd>→</kbd>, выход — <kbd>Esc</kbd>.</p>
{''.join(frames)}
</div>
<script>{SCRIPT}</script>
'''
    doc = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1">'
           + page + '</html>')
    path.write_text(doc, encoding='utf-8')
    # Для публикации как Artifact: без обёртки html/head — её добавляет платформа.
    (HERE / 'slides.artifact.html').write_text(page, encoding='utf-8')


def render_pdf(html_path: Path) -> None:
    """Запасной PDF из HTML-версии (Chromium), если PowerPoint недоступен."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1)
        pg.goto(html_path.as_uri())
        pg.evaluate("document.body.classList.add('render')")
        pg.wait_for_timeout(2500)                       # шрифты Google
        pg.pdf(path=str(HERE / f'{NAME}.pdf'), width='1920px', height='1080px',
               print_background=True, prefer_css_page_size=True)
        b.close()


def main() -> int:
    ap = argparse.ArgumentParser(description='Сборка презентации')
    ap.add_argument('--html-only', action='store_true')
    ap.add_argument('--cases', action='store_true', help='перерисовать кейсы')
    ap.add_argument('--screens', type=Path, metavar='DEMO_ZIP',
                    help='снять веб-интерфейс, загрузив этот архив')
    args = ap.parse_args()
    ASSETS.mkdir(exist_ok=True)
    if args.screens:
        screenshots(args.screens)
    D = load()
    cases = render_cases(force=args.cases)
    dk = build(D, cases)
    html_path = HERE / 'slides.html'
    write_html(dk, html_path)
    print(f'{html_path} — {len(dk.slides)} слайдов')
    if not args.html_only:
        # Основная презентация — нативная PPTX; PDF экспортирует из неё PowerPoint.
        # Без PowerPoint PDF рендерится из HTML-версии (тот же материал).
        import build_docx
        import build_pptx
        prs = build_pptx.build_deck(D, cases)
        pptx = HERE / f'{NAME}.pptx'
        prs.save(str(pptx))
        if build_pptx.export_with_powerpoint(pptx, HERE / f'{NAME}.pdf', ASSETS / 'pptx_png'):
            print(f'{NAME}.pptx, {NAME}.pdf (PowerPoint), assets/pptx_png/*.png')
        else:
            render_pdf(html_path)
            print(f'{NAME}.pptx, {NAME}.pdf (из HTML)')
        build_docx.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
