# -*- coding: utf-8 -*-
"""Нативная презентация PPTX: редактируемые тексты, таблицы и графики.

    python docs/presentation/build_pptx.py          # DXA_QC_pitch.pptx (+ PDF и PNG через PowerPoint)

Данные — те же, что у HTML-версии (build.load): метрики, кейсы, скриншоты.
Заметки докладчика — speech.py. Если на машине есть PowerPoint, презентация
им же экспортируется в PDF и в PNG по слайдам (PNG нужны доклад-DOCX и
проверке вёрстки); без PowerPoint остаётся только PPTX.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import build  # noqa: E402
import speech  # noqa: E402
from pptx_kit import (CW, MONO, MX, SANS, SANS_B, C, H, W, bullets, card, frame, line,  # noqa: E402
                      notes, oval, picture_fit, rect, stat, table, text)

NAME = 'DXA_QC_pitch'
ASSETS = HERE / 'assets'
CY = Inches(1.95)                    # верх контента под двухстрочным заголовком
f2 = build.f2
I = Inches


def ci(m, key):
    lo, hi = m.get(f'{key}_ci', [None, None])
    return '' if lo is None or not np.isfinite(lo) else f'{lo:.2f}–{hi:.2f}'


# --------------------------------------------------------------------------- #
#  Графики
# --------------------------------------------------------------------------- #
def roc_points(y, s):
    ok = np.isfinite(s)
    y, s = np.asarray(y, int)[ok], np.asarray(s, float)[ok]
    o = np.argsort(-s, kind='mergesort')
    ys, ss = y[o], s[o]
    P, N = ys.sum(), len(ys) - ys.sum()
    fpr, tpr, tp, fp = [0.0], [0.0], 0, 0
    for i in range(len(ys)):
        tp += ys[i]
        fp += 1 - ys[i]
        if i == len(ys) - 1 or ss[i + 1] != ss[i]:
            # группа одинаковых значений скора — один отрезок (диагональ при
            # ничьих), как в стандартной ROC
            fpr.append(fp / N)
            tpr.append(tp / P)
    return fpr, tpr


def roc_chart(slide, x, y, w, h, curves):
    cd = XyChartData()
    for label, yy, ss, _, _ in curves:
        s = cd.add_series(label)
        for a, b in zip(*roc_points(yy, ss)):
            s.add_data_point(a, b)
    diag = cd.add_series('случайно')
    diag.add_data_point(0, 0)
    diag.add_data_point(1, 1)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, x, y, w, h, cd)
    ch = gf.chart
    ch.has_legend = False
    ch.font.size = Pt(10)
    ch.font.name = MONO
    ch.font.color.rgb = C['ink3']
    for ax in (ch.category_axis, ch.value_axis):
        ax.minimum_scale, ax.maximum_scale, ax.major_unit = 0, 1, 0.2
        ax.has_major_gridlines = True
        ax.major_gridlines.format.line.color.rgb = C['rule2']
        ax.format.line.color.rgb = C['rule']
        ax.tick_labels.number_format = '0.0'
        ax.tick_labels.number_format_is_linked = False
    styles = [(c[3], c[4]) for c in curves] + [(C['ink3'], 'dash')]
    for series, (color, dash) in zip(ch.series, styles):
        ln = series.format.line
        ln.color.rgb = color
        ln.width = Pt(2.25 if dash != 'dash' else 1)
        if dash:
            from pptx.enum.dml import MSO_LINE_DASH_STYLE
            ln.dash_style = MSO_LINE_DASH_STYLE.DASH
        series.smooth = False
    return gf


def ci_track(slide, x, y, w, m, key, lo_ax, hi_ax, color, d=I(0.13)):
    v = m.get(key)
    lo, hi = m.get(f'{key}_ci', [None, None])
    if v is None or not np.isfinite(v):
        return
    X = lambda t: x + int((min(max(t, lo_ax), hi_ax) - lo_ax) / (hi_ax - lo_ax) * w)  # noqa: E731
    line(slide, x, y, x + w, y, C['rule'], 0.75)
    for t in np.linspace(lo_ax, hi_ax, 5):
        line(slide, X(t), y - I(0.04), X(t), y + I(0.04), C['rule'], 0.75)
    if lo is not None and np.isfinite(lo):
        line(slide, X(lo), y, X(hi), y, color, 3)
    oval(slide, X(v), y, d, color, line=C['paper'], line_w=1.5)


# --------------------------------------------------------------------------- #
def build_deck(D: dict, cases: dict) -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    SP = speech.slides(D, cases)
    total = len(SP)
    h, hb0 = D['h'], D['hb']
    hv, h0v = h['per_violation'], hb0['per_violation']
    ov, ov0 = h['overall']['binary_quality_class'], hb0['overall']['binary_quality_class']
    rot, art = D['cnn']['rotation'], D['cnn']['artifacts']
    rm, am = rot['metrics'], art['metrics']
    ps, pf = D['bench']['per_study_s'], D['bench']['per_frame_s']
    roi = D['roi_prob']
    rs, rmg = roi.get('scan_length', {}), roi.get('margins', {})
    case = lambda k: ASSETS / 'cases' / cases[k]['file']  # noqa: E731
    ca, cr, cart, croi = (cases.get(k, {}) for k in ('axis', 'rotation', 'artifacts', 'roi'))
    n = 0

    def new(section, title):
        nonlocal n
        n += 1
        s = frame(prs, n, total, section, title)
        notes(s, SP[n - 1]['text'] + ['Если спросят: ' + ' '.join(SP[n - 1]['ask'])])
        return s

    # 1. Титул ---------------------------------------------------------------
    n += 1
    s = frame(prs, n, total)
    notes(s, SP[0]['text'])
    rect(s, I(7.05), 0, W - I(7.05), H - I(0.12), fill=C['film'])
    half = (W - I(7.05)) // 2
    picture_fit(s, case('axis'), I(7.05), I(0.25), half - I(0.02), H - I(0.5), film=False,
                caption=f'ось {f2(ca.get("angle"), 1)}° · нарушение · порог ТЗ 5°', cap_h=I(0.42))
    picture_fit(s, case('rotation'), I(7.05) + half, I(0.25), half, H - I(0.5), film=False,
                caption=f'ротация · сеть {f2(cr.get("rot_p"))} · нарушение', cap_h=I(0.42))
    text(s, MX, I(1.25), I(6.2), I(0.3), 'ЛЦТ 2026 · кейс Департамента здравоохранения Москвы',
         size=11, font=MONO, color=C['accent'], caps=True, tracking=1.0)
    text(s, MX, I(1.7), I(6.3), I(1.9), [[('Контроль', {})], [('качества ', {}), ('DXA', {'color': C['accent']})]],
         size=54, font=SANS_B, spacing=0.95)
    text(s, MX, I(3.55), I(6.0), I(1.2), 'Проверяет укладку и разметку денситометрии до того, как '
         'исследование уйдёт к врачу, и объясняет каждое замечание в миллиметрах и градусах.',
         size=15.5, color=C['ink2'], spacing=1.25)
    xs = [MX, MX + I(1.2), MX + I(2.75), MX + I(4.3)]
    stat(s, xs[0], I(5.0), I(1.0), '5', '', 'критериев\nТЗ', 30, divider=False)
    stat(s, xs[1], I(5.0), I(1.3), f'{ps["median"]:.1f}', ' с', 'на исследование,\nлимит 180 с', 30)
    stat(s, xs[2], I(5.0), I(1.3), f2(ov['roc_auc']), '', 'ROC-AUC против\nэксперта, OOF', 30)
    stat(s, xs[3], I(5.0), I(1.3), f2(ov['f1']), '', 'F1 против\nэксперта, OOF', 30)

    # 2. Клиническая задача ---------------------------------------------------
    s = new('Клиническая задача', 'Ошибку укладки дешевле поймать,\nпока пациент на столе')
    text(s, MX, CY, I(5.9), I(0.35), 'Почему это важно', size=17, font=SANS_B)
    bullets(s, MX, CY + I(0.45), I(5.9), I(2.3), [
        'Плотность кости считается в области интереса. Позвоночник наклонён, бедро повёрнуто, в кадре '
        'застёжка белья — программа денситометра меряет не то.',
        'Ошибка уходит в T-критерий и ложную динамику: повторное облучение или неверная оценка лечения.',
        'Сегодня качество проверяет врач вручную, по памяти о критериях для каждой области.'],
        size=12.5)
    picture_fit(s, case('ok_hip'), MX, I(4.6), I(2.9), I(2.3),
                caption='норма: вертел виден', cap_h=I(0.38), cap_size=10)
    picture_fit(s, case('rotation'), MX + I(3.0), I(4.6), I(2.9), I(2.3),
                caption='ротация: контур плавный', cap_h=I(0.38), cap_size=10)
    text(s, I(6.95), CY, I(5.7), I(0.35), 'Что делает сервис', size=17, font=SANS_B)
    for i, (k, b) in enumerate([
            ('1 · проверяет', 'Каждый снимок по пяти критериям ТЗ: укладка, ось, посторонние предметы, '
                              'ротация бедра, отступы области интереса.'),
            ('2 · объясняет', '«Ось 6.9° при норме до 5°», «поле 12.4 см, нужно 12.9 см», контур и '
                              'тепловая карта на снимке.'),
            ('3 · отдаёт человеку', 'Лаборант исправляет укладку, пока пациент на столе; врач видит '
                                    'заключение DICOM SR рядом со снимком в PACS.')]):
        card(s, I(6.95), CY + I(0.5) + i * I(1.52), I(5.75), I(1.4), k, '', b, body_size=13)

    # 3. Таксономия ------------------------------------------------------------
    s = new('Таксономия нарушений', 'Пять критериев ТЗ, у каждого —\nизмерение в физических единицах')
    rows = [['код', 'критерий ТЗ 2.3', 'как измеряем', 'норма'],
            [[('spine_position', {'font': MONO})], 'видны верхние края подвздошных костей',
             'доля кости в нижних латеральных зонах', '> 0.001'],
            [[('spine_axis', {'font': MONO})], 'наклон оси до 5°', 'Тейл–Сен по центрам тел; вторая оценка — по краям',
             '≤ 5° (порог 4°)'],
            [[('spine_artifacts', {'font': MONO})], 'нет посторонних предметов', 'white top-hat + сеть, свод',
             f'p < {f2(art["threshold"])}'],
            [[('hip_rotation', {'font': MONO})], 'нет ротации (малый вертел)', 'выступ малого вертела, мм² + сеть, свод',
             f'p < {f2(rot["threshold"])}'],
            [[('hip_roi', {'font': MONO})], 'отступы 3 / 3 / 2 см', 'длина поля: 3 см + вертел — седалищная + 3 см',
             '≥ 12.9 см'],
            [[('undetermined', {'font': MONO})], 'область не определена', 'фильтр «не похоже на DXA»', 'ручной разбор']]
    table(s, MX, CY, I(8.25), rows, [2.0, 2.9, 3.6, 1.45], size=12, row_h=I(0.52),
          align=['l', 'l', 'l', 'l'], mono_cols=(3,))
    card(s, I(9.15), CY, I(3.55), I(2.25), 'несколько нарушений', '',
         'У 20 из 100 исследований нарушений больше одного. Критерии независимы, в отчёт идут все: '
         'spine_axis;spine_artifacts. quality_class = 1, если сработал хоть один.', body_size=13)
    card(s, I(9.15), CY + I(2.4), I(3.55), I(1.75), 'сбой', '',
         'Непрочитанный файл — строка Failure с текстом ошибки и quality_class = 1.', body_size=13)

    # 4. Данные ------------------------------------------------------------------
    s = new('Данные', f'{D["frames"]} снимков, 6–36 нарушений\nна критерий')
    stat(s, MX, CY, I(1.4), '100', '', 'исследований', 38, divider=False)
    stat(s, MX + I(1.65), CY, I(1.4), '499', '', 'DICOM-файлов', 38)
    stat(s, MX + I(3.3), CY, I(1.8), str(D['frames']), '', 'уникальных снимков', 38)
    reg = D['regions']
    bullets(s, MX, CY + I(1.5), I(5.6), I(3.2), [
        'Дубликаты (Lunar выгружает снимок по нескольку раз с разными UID) находим по MD5 пикселей — '
        'до разбиения на фолды.',
        f'Позвоночник {reg.get("spine", 0)}, левое бедро {reg.get("lh", 0)}, правое {reg.get("rh", 0)}; '
        'не больше трёх снимков на исследование.',
        'Один аппарат, GE Lunar Prodigy Advance. PixelSpacing пуст: масштаб 0.600 / 0.607 мм/px '
        'восстановлен по шагу «позвонок + диск».',
        'Своя разметка: 242 снимка × 29 ключевых точек по протоколу команды.'], size=13)
    text(s, I(6.7), CY, I(6.0), I(0.35), 'Нарушений мало, и они разные', size=17, font=SANS_B)
    items = [('position', 'Укладка'), ('axis', 'Ось > 5°'), ('artifacts', 'Посторонние предметы'),
             ('rotation', 'Ротация бедра'), ('roi', 'Отступы ROI')]
    maxn = max(nn for _, nn in D['pos'].values())
    for i, (k, lab) in enumerate(items):
        pos, nn = D['pos'][k]
        yy = CY + I(0.55) + i * I(0.48)
        text(s, I(6.7), yy, I(2.1), I(0.3), lab, size=13)
        full = int(I(2.85) * nn / maxn)
        rect(s, I(8.85), yy + I(0.04), full, I(0.22), fill=C['rule2'], radius=0.3)
        rect(s, I(8.85), yy + I(0.04), max(int(full * pos / nn), I(0.06)), I(0.22), fill=C['bad'], radius=0.3)
        text(s, I(8.85) + full + I(0.12), yy, I(1.1), I(0.3), f'{pos} из {nn}', size=12, font=MONO,
             color=C['ink2'])
    stat(s, I(6.7), I(5.15), I(1.75), '48', '', 'исследований из 100\nс нарушением', 30, divider=False)
    stat(s, I(8.6), I(5.15), I(1.8), '20', '', 'с несколькими\nнарушениями', 30)
    stat(s, I(10.6), I(5.15), I(1.9), '72', '', 'снимка из 249\nс нарушением', 30)

    # 5. Разбиение ---------------------------------------------------------------
    s = new('Разбиение и утечки', 'Честная оценка: исследование\nцеликом в одном фолде')
    cw, gap = I(0.95), I(0.07)
    for i in range(5):
        text(s, MX + I(0.35) + i * (cw + gap), CY, cw, I(0.25), f'фолд {i}', size=10, font=MONO,
             color=C['ink3'], align=PP_ALIGN.CENTER)
    for k in range(5):
        yy = CY + I(0.32) + k * I(0.45)
        text(s, MX, yy + I(0.07), I(0.3), I(0.25), f'#{k + 1}', size=10, font=MONO, color=C['ink3'])
        for i in range(5):
            val = i == k
            b = rect(s, MX + I(0.35) + i * (cw + gap), yy, cw, I(0.36), fill=C['accent'] if val else C['rule2'],
                     radius=0.15)
            tf = b.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = 'проверка' if val else 'обучение'
            r.font.size, r.font.name = Pt(10.5), SANS
            r.font.color.rgb = C['sheet'] if val else C['ink2']
    st = D['strata']
    text(s, MX, CY + I(2.75), I(5.6), I(1.3),
         f'5 фолдов по 20 исследований, folds.csv в репозитории. Страты: норма {st.get("norm", 0)}, '
         f'ротация {st.get("rotation", 0)}, артефакты {st.get("artifacts", 0)}, ось {st.get("axis", 0)}, '
         f'укладка {st.get("spine_position", 0)}, ROI {st.get("roi", 0)} исследований.',
         size=13, color=C['ink2'], spacing=1.25)
    for i, (k, b) in enumerate([
            ('группа — исследование', 'Позвоночник и оба бедра пациента всегда в одном фолде: '
                                      'разнесённые бёдра завысили бы метрику.'),
            ('страта — редчайшее нарушение', 'В каждом фолде есть примеры самых редких классов.'),
            ('подбор только на обучающих фолдах', 'Пороги, коридор нормы, веса сети, свод и его порог. '
                                                  'Метрики на всей выборке и out-of-fold — рядом.'),
            ('дисбаланс', 'Вес класса в потере сети, порог по доле нарушений, ДИ бутстрэпом по '
                          'исследованиям.')]):
        card(s, I(6.7), CY + i * I(1.18), I(6.0), I(1.07), k, '', b, body_size=13)

    # 6. Подход -------------------------------------------------------------------
    s = new('Подход', 'Геометрия там, где критерий — измерение;\nсеть там, где форма')
    seeds = rm.get('cnn_oof_by_seed', {})
    one = np.mean([v['roc_auc'] for v in seeds.values()]) if seeds else None
    rows = [['что пробовали для ротации', 'ROC-AUC', 'вывод'],
            ['выступ малого вертела по контуру', f2(rm['geometry']['roc_auc']), 'разрыв классов ≈ 2 мм'],
            ['вид кадра, логистическая регрессия', '0.50', 'вертел теряется при уменьшении'],
            ['модель ключевых точек, выступ по точкам', '—', 'ошибка 0.63 мм ≈ разрыву классов'],
            ['ResNet-34 на кадре 320 px, одна модель', f2(one), 'зависит от сида'],
            ['ансамбль трёх сидов', f2(rm['cnn_oof']['roc_auc']), 'стабильнее'],
            [[('свод: ансамбль + геометрия', {'bold': True})], [(f2(rm['stack_oof']['roc_auc']), {'bold': True})],
             'в поставке']]
    table(s, MX, CY, I(7.4), rows, [3.6, 1.1, 2.7], size=12.5, row_h=I(0.55), align=['l', 'r', 'l'],
          highlight=(6,), mono_cols=(1,))
    for i, (k, b) in enumerate([
            ('принцип', 'Вердикт проверяется глазами: «ось 6.9°», «поле 12.4 см». Основа — геометрия '
                        'в миллиметрах и градусах.'),
            ('где геометрии мало', 'Ротация и посторонние предметы — признаки формы. Там геометрию '
                                   'дополняет сеть, вердикт выносит свод, тепловая карта показывает, куда '
                                   'смотрела сеть.'),
            ('где сеть не нужна', 'Ось, укладка, отступы — прямые измерения. Сеть для оси проверили: '
                                  'ROC-AUC 0.51, сигнала нет.')]):
        card(s, I(8.35), CY + i * I(1.62), I(4.35), I(1.52), k, '', b, body_size=12)

    # 7. Архитектура ------------------------------------------------------------------
    s = new('Архитектура решения', 'Один контейнер, пять шагов,\nни одного обращения в сеть')
    nodes = [('вход', 'DICOM, zip', 'выгрузка из PACS как есть'),
             ('чтение', 'pydicom + gdcm', 'сжатие, ориентация, дубликаты'),
             ('область', 'region_clf', 'позвоночник, бёдра, «не DXA»'),
             ('критерии', 'геометрия + CNN', 'независимо, со своими флагами'),
             ('свод', 'quality_class', 'вероятности, текст, SR')]
    nw, ng = I(2.2), I(0.27)
    for i, (k, t, b) in enumerate(nodes):
        xx = MX + i * (nw + ng)
        card(s, xx, CY, nw, I(1.4), k, t, b, accent=(i == 3), body_size=11, title_size=14)
        if i:
            line(s, xx - ng + I(0.03), CY + I(0.7), xx - I(0.03), CY + I(0.7), C['ink3'], 1.25, arrow=True)
    for i, (k, b) in enumerate([
            ('выход по ТЗ 2.5 и 2.7', 'xlsx или csv, строка на снимок, 8 колонок ТЗ; zip с визуализацией; '
                                      'zip с DICOM SR.'),
            ('интерфейсы', 'CLI для пакета, REST API (POST /batch → задание → отчёт), веб-интерфейс.'),
            ('контейнер', 'python:3.11-slim по дайджесту, lock-файл, onnxruntime на CPU, без сети, '
                          'один поток — побитово тот же отчёт.')]):
        card(s, MX + i * I(4.1), CY + I(1.55), I(3.9), I(1.3), k, '', b, body_size=12)
    from service.pipeline import COLUMNS as COLS8

    def cut(v, k=13):
        v = '' if v is None else str(v)
        return v if len(v) <= k else '…' + v[-k:]
    sample = [COLS8] + [[cut(r.get(c)) if c in ('path_to_study', 'study_uid', 'image_uid')
                         else ('' if r.get(c) is None else str(r.get(c))) for c in COLS8]
                        for r in D['sample'][:4]]
    text(s, MX, CY + I(2.95), CW, I(0.25), 'фрагмент настоящего отчёта: восемь колонок ТЗ 2.5',
         size=10, font=MONO, color=C['ink3'], caps=True, tracking=0.6)
    table(s, MX, CY + I(3.25), CW, sample, [1.6, 1.6, 1.6, 1.6, 1.25, 1.35, 1.6, 1.6], size=10.5,
          header_size=8.5, row_h=I(0.3), align=['l'] * 8, mono_cols=tuple(range(8)))

    # 8. Модели --------------------------------------------------------------------------
    s = new('Модели', 'Шесть модулей, у каждого\nсвоя проверяемая задача')
    ens = rot.get('onnx', {})
    mb = ens.get('bytes', 0) / 1e6 / max(len(ens.get('files', [1])), 1)
    cards = [
        ('region_clf', 'Область снимка', 'Признаки кадра + логистическая регрессия; фильтр новизны '
         'отсекает не-DXA на ручной разбор.', '252 / 252 снимка', False),
        ('spine_qc', 'Позвоночник', 'Маска кости, коридор колонны, центры тел. Ось — Тейл–Сен, укладка — '
         'масса гребней, артефакты — white top-hat.', 'ROC-AUC OOF: ось 0.82 · укладка 0.82', False),
        ('hip_roi', 'Отступы бедра', 'Длина поля сканирования против 3 см + вертел — седалищная + 3 см; '
         'отступы по рисунку 6 — справочно.', f'F1 {f2(rs.get("f1"))} · ROC-AUC {f2(rs.get("roc_auc"))}', False),
        ('hip_rotation', 'Ротация: геометрия', 'Выступ малого вертела над прямой медиального контура '
         'диафиза, мм². Коридор ловит пере- и недоротацию.', 'коридор 96–272 мм²', False),
        ('cnn_qc', 'Ротация и предметы: сеть', 'ResNet-34, 320 px, ансамбль 3 сидов, ONNX → onnxruntime. '
         'Свод с геометрией, тепловая карта CAM.', f'ROC-AUC свода 0.82 · {mb:.0f} МБ × 3', True),
        ('dxa_qc · по флагу', 'Ключевые точки', 'U-Net + ResNet-34, 29 точек, 5 фолдов. Проигрывает '
         'контуру на OOF — выключена по умолчанию.', 'ошибка 1.0–2.5 мм · угол 0.50°', False)]
    cw3, ch2 = I(3.95), I(2.3)
    for i, (k, t, b, m, acc) in enumerate(cards):
        card(s, MX + (i % 3) * (cw3 + I(0.12)), CY + (i // 3) * (ch2 + I(0.15)), cw3, ch2, k, t, b,
             accent=acc, body_size=13, metric=m)

    # 9. Предобработка -----------------------------------------------------------------------
    s = new('Предобработка и постобработка', 'Всё приводится к миллиметрам до модели\nи к вероятности после')
    text(s, MX, CY, I(5.8), I(0.35), 'Предобработка', size=17, font=SANS_B)
    bullets(s, MX, CY + I(0.5), I(5.8), I(4.3), [
        'Сжатые синтаксисы, инверсия MONOCHROME1, VOI LUT.',
        'Ориентация по PatientOrientation: зеркальная выгрузка приводится к стандартному AP-виду.',
        'Дедупликация по MD5 пикселей: одна строка отчёта на снимок.',
        'Правое бедро отражается в левое: измерения и сеть видят одну сторону.',
        'Маска кости и масштаб мм/px: пороги в миллиметрах, градусах, мм².',
        'Для сети: яркость по перцентилям 0.5–99.5, длинная сторона → 320 px без искажения.'], size=14)
    text(s, I(6.9), CY, I(5.8), I(0.35), 'Постобработка', size=17, font=SANS_B)
    bullets(s, I(6.9), CY + I(0.5), I(5.8), I(4.3), [
        'Пороги и коридоры — из калибровки по обучающим фолдам.',
        'Отступы ROI — по длине поля, как оценивает эксперт; порог из ТЗ и анатомии.',
        'Свод сети с геометрией; порог — доля нарушений в обучении.',
        'Платт для каждого критерия → quality_probability = 1 − Π(1 − p).',
        'Расхождение методов — флаг и низкая уверенность, а не молчаливый выбор.',
        'Текст по-русски, оверлей с тепловой картой, DICOM SR.'], size=14)

    # 10. Метрики ------------------------------------------------------------------------------
    s = new('Метрики качества', 'Метрики out-of-fold с 95%\nдоверительными интервалами')
    names = {'spine_position': 'Укладка', 'spine_axis': 'Ось', 'spine_artifacts': 'Посторонние предметы',
             'hip_rotation': 'Ротация бедра', 'hip_roi': 'Отступы ROI'}
    rowsm = [(names[k], hv[k], h0v.get(k)) for k in names] + \
        [('Позвоночник, кадр', h['per_region']['spine'], hb0['per_region']['spine']),
         ('Бедро, кадр', h['per_region']['hip'], hb0['per_region']['hip']),
         ('Всего, «есть нарушение»', ov, ov0)]
    cols_x = dict(name=MX, pos=I(3.05), sens=I(3.9), spec=I(4.65), f1bar=I(5.45), f1=I(8.25),
                  aucbar=I(9.3), auc=I(12.1))
    yy = CY
    for key, lab in (('name', 'out-of-fold'), ('pos', 'нар./n'), ('sens', 'чувств.'), ('spec', 'специф.'),
                     ('f1bar', 'F1, 95% ДИ'), ('f1', 'F1'), ('aucbar', 'ROC-AUC, 95% ДИ (0.4–1)'),
                     ('auc', 'AUC')):
        text(s, cols_x[key], yy, I(2.6), I(0.25), lab, size=9, font=MONO, color=C['ink3'], caps=True)
    line(s, MX, yy + I(0.3), W - MX, yy + I(0.3), C['ink'], 1.25)
    rh = I(0.5)
    for i, (lab, m, m0) in enumerate(rowsm):
        y0 = yy + I(0.38) + i * rh
        both = m0 is not None and abs((m0.get('f1') or 0) - (m.get('f1') or 0)) > 1e-9
        last = i == len(rowsm) - 1
        text(s, cols_x['name'], y0 + I(0.1), I(2.4), I(0.3), lab, size=12.5, font=SANS_B if last else SANS)
        text(s, cols_x['pos'], y0 + I(0.1), I(0.8), I(0.3), f"{m['positives']}/{m['n']}", size=11.5, font=MONO)
        text(s, cols_x['sens'], y0 + I(0.1), I(0.7), I(0.3), f2(m['sensitivity']), size=11.5, font=MONO)
        text(s, cols_x['spec'], y0 + I(0.1), I(0.7), I(0.3), f2(m['specificity']), size=11.5, font=MONO)
        for key, lo_ax, xb, xn in (('f1', 0, cols_x['f1bar'], cols_x['f1']),
                                   ('roc_auc', 0.4, cols_x['aucbar'], cols_x['auc'])):
            if both:
                ci_track(s, xb, y0 + I(0.12), I(2.55), m0, key, lo_ax, 1, C['geo'], d=I(0.11))
                ci_track(s, xb, y0 + I(0.33), I(2.55), m, key, lo_ax, 1, C['accent'])
                text(s, xn, y0 + I(0.02), I(0.9), I(0.22), f2(m0.get(key)), size=10, font=MONO, color=C['ink3'])
                text(s, xn, y0 + I(0.22), I(0.9), I(0.25), f2(m.get(key)), size=12, font=MONO, bold=True)
            else:
                ci_track(s, xb, y0 + I(0.22), I(2.55), m, key, lo_ax, 1, C['accent'])
                text(s, xn, y0 + I(0.1), I(0.9), I(0.25), f2(m.get(key)), size=12, font=MONO, bold=True)
        line(s, MX, y0 + rh - I(0.02), W - MX, y0 + rh - I(0.02), C['rule'], 0.5)
    ly = yy + I(0.45) + len(rowsm) * rh
    rect(s, MX, ly + I(0.07), I(0.3), I(0.1), fill=C['geo'])
    text(s, MX + I(0.38), ly, I(3.9), I(0.25), 'исходное решение: геометрия, ROI по рисунку 6', size=10.5,
         color=C['ink2'])
    rect(s, I(4.85), ly + I(0.07), I(0.3), I(0.1), fill=C['accent'])
    text(s, I(5.23), ly, I(3.4), I(0.25), 'сейчас: сеть + правило эксперта для ROI', size=10.5,
         color=C['ink2'])
    text(s, I(8.6), ly, I(4.2), I(0.25),
         f"macro-F1 {f2(hb0['overall']['macro_f1'])} → {f2(h['overall']['macro_f1'])} · область 252/252",
         size=11, color=C['ink2'], font=MONO)

    # 11. ROC ----------------------------------------------------------------------------------------
    s = new('Эксперименты', 'Сеть поверх геометрии:\nROC-кривые out-of-fold')
    ro, ao = D['rot_oof'], D['art_oof']
    for j, (title, dd, m, geo_lab) in enumerate((
            (f'Ротация бедра · {rot["positives"]} из {rot["n"]}', ro, rm, 'выступ'),
            (f'Посторонние предметы · {art["positives"]} из {art["n"]}', ao, am, 'top-hat'))):
        x0 = MX + j * I(6.15)
        text(s, x0, CY, I(5.9), I(0.35), title, size=16, font=SANS_B)
        curves = [(geo_lab, dd.y, dd.geometry.fillna(0).values, C['geo'], 'dash'),
                  ('сеть', dd.y, dd.p_cnn.values, C['cnn'], None),
                  ('свод', dd.y, dd.p_stack.values, C['accent'], None)]
        roc_chart(s, x0 - I(0.1), CY + I(0.4), I(3.6), I(3.6), curves)
        lx = x0 + I(3.65)
        for k, (lab, col, key) in enumerate(((geo_lab, C['geo'], 'geometry'), ('сеть', C['cnn'], 'cnn_oof'),
                                              ('свод', C['accent'], 'stack_oof'))):
            rect(s, lx, CY + I(0.62) + k * I(0.34), I(0.28), I(0.1), fill=col)
            text(s, lx + I(0.38), CY + I(0.53) + k * I(0.34), I(1.9), I(0.28),
                 f'{lab} {f2(m[key]["roc_auc"])}', size=12, bold=(key == 'stack_oof'))
        text(s, lx, CY + I(1.72), I(2.3), I(0.5), 'вердикт вложенно: свод и порог без отложенного фолда',
             size=10, color=C['ink3'], spacing=1.1)
        g, v = D['h0'][('hip_rotation' if j == 0 else 'spine_artifacts')], m['verdict_nested']
        table(s, lx, CY + I(2.3), I(2.35), [['', 'чув.', 'спец.', 'F1'],
                                           [geo_lab, f2(g['sensitivity']), f2(g['specificity']), f2(g['f1'])],
                                           ['свод', f2(v['sensitivity']), f2(v['specificity']),
                                            [(f2(v['f1']), {'bold': True})]]],
              [1.1, 0.75, 0.75, 0.7], size=11.5, header_size=8.5, row_h=I(0.34), mono_cols=(1, 2, 3))
        text(s, x0, CY + I(4.1), I(3.5), I(0.25), '1 − специфичность →   ↑ чувствительность', size=9.5,
             font=MONO, color=C['ink3'])

    # 12. Сравнение вариантов -----------------------------------------------------------------
    s = new('Сравнение вариантов', 'Что проверили и почему\nоставили то, что оставили')
    kp = D['kp_oof'].get('per_violation', {})
    text(s, MX, CY, I(5.8), I(0.35), 'Ось и отступы: контур против модели точек', size=15, font=SANS_B)
    table(s, MX, CY + I(0.45), I(5.8), [
        ['out-of-fold', 'контур', 'точки U-Net'],
        ['ось: F1 / ROC-AUC', '0.25 / 0.82', f"{f2(kp.get('spine_axis', {}).get('f1'))} / "
                                            f"{f2(kp.get('spine_axis', {}).get('roc_auc'))}"],
        ['время на снимок, CPU', '31 мс', '301 мс'],
        ['ошибка точек, медиана', '—', '1.0–2.5 мм']], [2.6, 1.5, 1.7], size=12.5, row_h=I(0.42),
        mono_cols=(1, 2))
    text(s, MX, CY + I(2.3), I(5.8), I(0.9), 'По F1 разница в пределах шума, по ранжированию контур '
         'лучше и в десять раз быстрее; 53 из 54 крупных ошибок точек — нумерация позвонка.', size=12,
         color=C['ink2'], spacing=1.2)
    sw = D['spine_oof'].get('position', {}).get('rule_sweep', {})
    ru = {'geometry': 'геометрия (в поставке)', 'appearance': 'вид кадра', 'and': 'обе согласны',
          'or': 'любая из двух'}
    text(s, I(6.9), CY, I(5.8), I(0.35), 'Правило свода для укладки', size=15, font=SANS_B)
    table(s, I(6.9), CY + I(0.45), I(5.8), [['out-of-fold, 6 нарушений', 'найдено', 'лишних', 'F1']] +
          [[ru.get(k, k), str(v['tp']), str(v['fp']), f2(v['f1'])] for k, v in sw.items()],
          [2.8, 1.0, 1.0, 0.9], size=12.5, row_h=I(0.36), highlight=(1,), mono_cols=(1, 2, 3))
    text(s, I(6.9), CY + I(2.3), I(5.8), I(0.9), 'На всей выборке лучшим казалось согласие обеих '
         'оценок (F1 0.73); out-of-fold оно ловит одно нарушение из шести.', size=12, color=C['ink2'],
         spacing=1.2)
    text(s, MX, CY + I(3.2), CW, I(0.35), 'Сеть для ротации: что перебрали (ROC-AUC out-of-fold)', size=15,
         font=SANS_B)
    table(s, MX, CY + I(3.65), CW, [
        ['ResNet-18', 'EfficientNet-B0', 'ResNet-34, 256 px', 'ResNet-34, 1 сид', 'энкодер модели точек',
         'ResNet-34 × 3', '+ геометрия'],
        ['0.76', '0.73', '0.77', f2(one), '0.79', f2(rm['cnn_oof']['roc_auc']),
         [(f2(rm['stack_oof']['roc_auc']), {'bold': True})]]],
        [1] * 7, size=13, row_h=I(0.4), align=['r'] * 7, mono_cols=tuple(range(7)))

    # 13. Эксперт ---------------------------------------------------------------------------------
    s = new('Анализ ошибок и эксперт', 'Разбор ошибок показал,\nкак оценивает эксперт')
    card(s, MX, CY, I(6.0), I(4.6), 'отступы roi · что делает эксперт', 'Судит по длине поля', '', accent=True)
    table(s, MX + I(0.22), CY + I(1.0), I(5.55), [
        ['правило', 'найдено', 'лишних', 'F1', 'AUC'],
        ['отступы от ориентиров, рисунок 6', f"{rmg.get('tp')}/7", str(rmg.get('fp')), f2(rmg.get('f1')),
         f2(rmg.get('roc_auc'))],
        [[('длина поля ≥ 3 + вертел–седалищная + 3 см', {'bold': True})], f"{rs.get('tp')}/7",
         str(rs.get('fp')), [(f2(rs.get('f1')), {'bold': True})], f2(rs.get('roc_auc'))]],
        [3.2, 0.9, 0.8, 0.6, 0.6], size=11.5, row_h=I(0.46), highlight=(2,), mono_cols=(1, 2, 3, 4))
    text(s, MX + I(0.22), CY + I(2.55), I(5.55), I(1.9),
         'Эксперт ставит «норму» при 1.7–3.0 см под седалищной костью и «нарушение» на коротких полях. '
         'Правило длины поля не подбиралось по меткам: 30 мм сверху и снизу — из ТЗ, 69 мм между вертелом '
         'и седалищной костью — медиана анатомии. Отступы по рисунку 6 считаются и показываются; режим '
         'строго по ТЗ — флаг --roi-rule margins.', size=12.5, color=C['ink2'], spacing=1.22)
    ro_ = D['rot_oof']
    geo_bad = ro_.geometry.fillna(0) > 0
    st_bad = ro_.pred_nested.astype(bool)
    dis = geo_bad != st_bad
    right = int((st_bad[dis] == ro_.y[dis].astype(bool)).sum())
    for i, (k, b) in enumerate([
            ('ось · сколиоз не нарушение', 'При сколиозе эксперт не ставит нарушение оси ни в одном из '
                                           '14 случаев, даже при угле до 8.8°. Это главный источник наших лишних '
                                           'срабатываний по оси; отличить сколиоз от наклона по поясничному '
                                           'треку не удалось (сеть — ROC-AUC 0.51).'),
            ('ротация · сеть против выступа', f'Расхождений сети и выступа: {int(dis.sum())} из {len(ro_)}; '
                                              f'прав свод — {right}, выступ — {int(dis.sum()) - right}. '
                                              'Такие снимки помечены флагом.'),
            ('укладка · 6 примеров', 'ROC-AUC 0.82, но отсечка на фолде с одним-двумя нарушениями '
                                     'неустойчива: F1 out-of-fold 0.32.')]):
        card(s, I(6.85), CY + [0, I(1.75), I(3.15)][i], I(5.85), [I(1.6), I(1.25), I(1.45)][i], k, '', b,
             body_size=12.5)

    # 14. Кейсы ------------------------------------------------------------------------------------
    s = new('Кейсы', 'Каждое замечание видно на снимке')
    caps = [('axis', f'ось {f2(ca.get("angle"), 1)}°, норма до 5°\nspine_axis ✓ эксперт'),
            ('artifacts', f'предмет · сеть {f2(cart.get("art_p"))}\nspine_artifacts ✓ эксперт'),
            ('rotation', f'выступ {f2(cr.get("area"), 0)} мм² (96–272)\nhip_rotation ✓ эксперт'),
            ('roi', f'поле {f2((croi.get("scan_len") or 0) / 10, 1)} см < 12.9\nhip_roi ✓ эксперт')]
    fw = (CW - 3 * I(0.15)) // 4
    for i, (k, cap) in enumerate(caps):
        if k in cases:
            picture_fit(s, case(k), MX + i * (fw + I(0.15)), I(1.45), fw, I(5.0), caption=cap, cap_h=I(0.62),
                        cap_size=10.5)
    text(s, MX, I(6.6), CW, I(0.3), 'Снимки обучающего набора, вероятности сети — out-of-fold. Оранжевым — '
         'тепловая карта сети, голубым — ориентиры, зелёным и красным — пороги.', size=10.5,
         color=C['ink2'])

    # 15. Скорость ---------------------------------------------------------------------------------
    s = new('Скорость и системные требования', 'Секунды на исследование\nна обычном процессоре')
    stat(s, MX, CY, I(2.4), f'{ps["median"]:.1f}', ' с', f'медиана на исследование\n(максимум {ps["max"]:.1f} с)',
         40, divider=False)
    stat(s, MX + I(2.85), CY, I(2.2), '180', ' с', 'лимит ТЗ 2.7\nна исследование', 40)
    stat(s, MX, CY + I(1.55), I(2.4), f'{pf["median"] * 1000:.0f}', ' мс', 'медиана на снимок,\nодин поток CPU',
         40, divider=False)
    stat(s, MX + I(2.85), CY + I(1.55), I(2.5), f'{D["bench"]["total_s"] / 60:.1f}', ' мин',
         'все 100 исследований\nв один процесс', 40)
    text(s, MX, CY + I(3.15), I(5.5), I(0.25), 'худшее исследование против лимита, в масштабе', size=9.5,
         font=MONO, color=C['ink3'], caps=True)
    rect(s, MX, CY + I(3.45), I(5.4), I(0.3), fill=C['rule2'], radius=0.2)
    rect(s, MX, CY + I(3.45), max(int(I(5.4) * ps['max'] / 180), I(0.05)), I(0.3), fill=C['accent'])
    text(s, MX + I(0.2), CY + I(3.49), I(4), I(0.25), f'{ps["max"]:.1f} с — {ps["max"] / 180 * 100:.1f} % лимита',
         size=10.5, font=MONO, color=C['ink2'])
    text(s, MX, CY + I(3.95), I(5.6), I(0.8), 'В контейнере (Linux, Python 3.11) — 158 с на 100 исследований '
         'с оверлеями и SR. Без сети — 27 мс на снимок.', size=12, color=C['ink2'], spacing=1.2)
    table(s, I(6.7), CY, I(6.0), [['', 'минимум', 'рекомендуется'],
                                  ['CPU', '2 ядра x86-64', '4 ядра'], ['RAM', '2 ГБ', '4 ГБ'],
                                  ['диск', '2 ГБ (образ 1.4 ГБ)', '5 ГБ'], ['GPU', 'не нужен', 'не нужен'],
                                  ['ПО', 'Docker, POSIX sh', 'Linux']],
          [1.0, 2.5, 2.5], size=13, row_h=I(0.45), align=['l', 'l', 'l'])
    card(s, I(6.7), CY + I(3.0), I(6.0), I(1.5), 'воспроизводимость', '',
         'Повторный прогон на тех же данных даёт побитово тот же отчёт: фиксированный порядок обхода, '
         'модели на диске, вычисления в один поток. Проверяется тестом.', body_size=13)

    # 16. Интерфейс ------------------------------------------------------------------------------------
    s = new('Интерфейс и интеграция', 'Замечание доходит до лаборанта\nи до PACS')
    ui = ASSETS / 'ui_rotation.png'
    if ui.exists():
        picture_fit(s, ui, MX, CY, I(8.0), I(4.8))
    for i, (k, b) in enumerate([
            ('веб-приложение', 'Папка или zip; снимок с разметкой и нормой; ставится из браузера '
                               'как приложение.'),
            ('человек в контуре', 'Оператор отмечает «согласен / не согласен»; отметки — CSV для '
                                  'перекалибровки.'),
            ('PACS и API', 'DICOM SR в том же исследовании; POST /batch → отчёт, оверлеи, SR.'),
            ('без Docker', 'Установщик для Windows и Telegram-бот: тот же конвейер, отчёт побитово '
                           'тот же.')]):
        card(s, I(8.85), CY + i * I(1.22), I(3.85), I(1.14), k, '', b, body_size=11)

    # 17. Ограничения ------------------------------------------------------------------------------------
    s = new('Ограничения', 'Что сервис не умеет и почему')
    table(s, MX, I(1.55), CW, [
        ['ограничение', 'почему', 'что нужно, чтобы снять'],
        [[('Один аппарат', {'bold': True})], 'Все снимки — GE Lunar Prodigy Advance; PixelSpacing пуст',
         'Снимки второго аппарата: проверить масштаб, дообучить сеть'],
        [[('Малая выборка', {'bold': True})], '6–36 нарушений на критерий, интервалы широкие',
         'Больше разметки; отметки операторов из пилота'],
        [[('ROI — как эксперт', {'bold': True})], 'Вердикт по длине поля, а не буквально по рисунку 6',
         'Решение организаторов; режим по ТЗ уже есть флагом'],
        [[('Ось и сколиоз', {'bold': True})], 'Эксперт не считает сколиоз нарушением оси, мы его не отличаем',
         'Разметка сколиоза или грудного отдела'],
        [[('Разметка оператора', {'bold': True})], 'В данных нет ROI и линий — всё по изображению',
         'Выгрузка с оверлеями денситометра'],
        [[('Половина тела Th12', {'bold': True})], 'Разметки уровня Th12 нет',
         'Разметка Th12; модель точек уже выдаёт th12_top/bottom']],
        [2.2, 4.4, 4.4], size=13, row_h=I(0.66), align=['l', 'l', 'l'])

    # 18. Пилот ---------------------------------------------------------------------------------------
    s = new('Внедрение и пилот', 'Теневой режим → ассистент лаборанта\n→ сеть медорганизаций')
    phases = [('месяцы 1–2 · теневой режим', '1–2 медорганизации',
               'DICOM-роутер отправляет копии в контейнер, SR возвращается в PACS «не проверено». '
               'Считаем согласие с экспертом.'),
              ('месяцы 3–6 · ассистент', 'Замечание до отпуска пациента',
               'Результат у лаборанта через 2–3 с после снимка. Отметки операторов — в перекалибровку.'),
              ('месяцы 6–12 · масштабирование', 'Другие аппараты и сеть МО',
               'Проверка масштаба, дообучение сети на снимках аппарата, мониторинг доли нарушений.')]
    pw = CW // 3
    line(s, MX, CY + I(0.05), W - MX, CY + I(0.05), C['ink'], 1.5)
    for i, (w_, t, b) in enumerate(phases):
        x0 = MX + i * pw
        oval(s, x0 + I(0.08), CY + I(0.05), I(0.16), C['paper'], line=C['ink'], line_w=1.5)
        text(s, x0, CY + I(0.25), pw - I(0.2), I(0.25), w_, size=10, font=MONO, color=C['accent'], caps=True)
        text(s, x0, CY + I(0.55), pw - I(0.2), I(0.35), t, size=15, font=SANS_B)
        text(s, x0, CY + I(0.95), pw - I(0.3), I(1.1), b, size=12, color=C['ink2'], spacing=1.2)
    for i, (k, t, b) in enumerate([('качество', 'Согласие с экспертом', 'Чувствительность и специфичность '
                                    'на слепой выборке врача, раз в месяц.'),
                                   ('пациент', 'Повторные исследования', 'Доля повторов из-за укладки '
                                    'по журналу МО: до и после.'),
                                   ('врач', 'Время на контроль', 'Хронометраж на 50 исследованиях; '
                                    'сервис — 2 с на исследование.')]):
        card(s, MX + i * (pw + I(0.02)), CY + I(2.05), pw - I(0.12), I(1.55), 'метрика пилота · ' + k, t, b,
             body_size=11.5, title_size=15)
    fl = [('денситометр', 'DICOM'), ('PACS', 'DICOM-роутер'), ('контейнер', 'сервис QC'),
          ('назад в PACS', 'DICOM SR'), ('лаборант', 'веб-интерфейс')]
    fw5 = (CW - 4 * I(0.25)) // 5
    for i, (k, t) in enumerate(fl):
        x0 = MX + i * (fw5 + I(0.25))
        card(s, x0, CY + I(3.8), fw5, I(0.92), k, t, '', accent=(i == 2), title_size=13.5)
        if i:
            line(s, x0 - I(0.22), CY + I(4.26), x0 - I(0.03), CY + I(4.26), C['ink3'], 1.25, arrow=True)

    # 19. Демо -------------------------------------------------------------------------------------------
    s = new('Демонстрация', 'Сценарий демонстрации, 3 минуты')
    steps = ['./build.sh && ./run.sh → http://localhost:8000',
             'Перетащить demo.zip: 7 исследований, все типы нарушений',
             'Фильтр «Ротация»: тепловая карта, свод, выступ в мм²',
             'Ось с порогом; отступы: длина поля и линии рисунка 6',
             'Отметить «согласен» (A) и «не согласен», выгрузить отметки',
             'Скачать xlsx (8 колонок ТЗ) и DICOM SR',
             'Пакетно: ./run.sh batch /data → out/report.xlsx']
    for i, st_ in enumerate(steps):
        yy = I(1.55) + i * I(0.52)
        oval(s, MX + I(0.17), yy + I(0.17), I(0.34), C['accent'])
        text(s, MX, yy + I(0.04), I(0.34), I(0.3), str(i + 1), size=12, font=MONO, color=C['sheet'],
             align=PP_ALIGN.CENTER)
        text(s, MX + I(0.5), yy + I(0.02), I(6.2), I(0.4), st_, size=14)
    card(s, I(7.6), I(1.55), I(5.1), I(1.3), 'запасной вариант', '',
         'Нет Docker — настольное приложение; нет сети — слайды «Кейсы» и «Интерфейс»; '
         'с телефона — бот, команда /demo.', body_size=12)
    card(s, I(7.6), I(3.0), I(5.1), I(1.6), 'репозиторий', 'github.com/theorcos239/LCT_2026',
         'README · REVIEW · USER_GUIDE · DESKTOP · TELEGRAM_BOT', body_size=12, title_size=14)
    strip = [('Контейнер', 'веса в образе, build.sh / run.sh'), ('Отчёт ТЗ 2.5', 'xlsx/csv, 8 колонок'),
             ('ТЗ 2.6', 'оверлеи, SR, интерфейс'), ('Интерфейсы', 'веб, Windows, бот, API'),
             ('Документация', 'README и руководства'),
             ('Автотесты', f'сервис — {D.get("tests", 64)}, бот — {D.get("tests_bot", 27)}')]
    sw_ = CW // 6
    line(s, MX, I(5.4), W - MX, I(5.4), C['ink'], 1.5)
    for i, (t, b) in enumerate(strip):
        text(s, MX + i * sw_, I(5.55), sw_ - I(0.15), I(0.3), t, size=13.5, font=SANS_B)
        text(s, MX + i * sw_, I(5.9), sw_ - I(0.15), I(0.6), b, size=11.5, color=C['ink2'], spacing=1.15)
    return prs


# --------------------------------------------------------------------------- #
def export_with_powerpoint(pptx: Path, pdf: Path, png_dir: Path) -> bool:
    """PDF и PNG слайдов через установленный PowerPoint. Без него — False."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return False
    pythoncom.CoInitialize()
    app = None
    try:
        app = win32com.client.DispatchEx('PowerPoint.Application')
        pres = app.Presentations.Open(str(pptx), ReadOnly=True, Untitled=False, WithWindow=False)
        pres.SaveAs(str(pdf), 32)                            # ppSaveAsPDF
        png_dir.mkdir(parents=True, exist_ok=True)
        for old in png_dir.glob('*.png'):
            old.unlink()
        for i in range(1, pres.Slides.Count + 1):
            pres.Slides(i).Export(str(png_dir / f'{i:02d}.png'), 'PNG', 1920, 1080)
        pres.Close()
        return True
    except Exception as e:                                   # noqa: BLE001
        print('PowerPoint недоступен:', e)
        return False
    finally:
        if app is not None:
            app.Quit()


def main() -> int:
    D = build.load()
    cases = build.render_cases()
    prs = build_deck(D, cases)
    out = HERE / f'{NAME}.pptx'
    prs.save(str(out))
    print(out, len(prs.slides), 'слайдов')
    if export_with_powerpoint(out, HERE / f'{NAME}.pdf', ASSETS / 'pptx_png'):
        print('PDF и PNG — через PowerPoint')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
