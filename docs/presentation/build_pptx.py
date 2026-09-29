# -*- coding: utf-8 -*-
"""Презентация по шаблону ЛЦТ 2026: редактируемые тексты, таблицы и графики.

    python docs/presentation/build_pptx.py      # DXA_QC_pitch.pptx (+ PDF и PNG через PowerPoint)

Основа — шаблон организаторов (template/LCT2026_template.pptx): его фоны с
логотипами, Montserrat и палитра темы. Первые четыре слайда — обязательные
слайды шаблона (титул, о команде, команда, история), они заполняются как есть;
данные команды — в team.json. Дальше — презентация решения в стиле шаблона.
Числа — из файлов метрик (build.load), текст доклада и заметки — speech.py.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pptx.chart.data import XyChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import build  # noqa: E402
import speech  # noqa: E402
from lct_kit import (CW, FONT, XR, C, H, I, TemplateDeck, W, X0, _run, bullets, card, label, line,  # noqa: E402
                     number_badge, oval, picture, picture_fit, rect, stat, table, text)

NAME = 'DXA_QC_pitch'
ASSETS = HERE / 'assets'
TEMPLATE = HERE / 'template' / 'LCT2026_template.pptx'
MANUAL_ASSETS = HERE.parents[1] / 'docs' / 'manuals' / 'assets'
REPO = 'github.com/theorcos239/LCT_2026'
BOT = '@LCT_bone_density_research_bot'
Y0 = I(2.0)                      # верх контента под заголовком-утверждением
f2 = build.f2


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
            fpr.append(fp / N)                  # группа одинаковых значений — один отрезок
            tpr.append(tp / P)
    return fpr, tpr


def roc_chart(slide, x, y, w, h, curves):
    cd = XyChartData()
    for lab, yy, ss, _, _ in curves:
        s = cd.add_series(lab)
        for a, b in zip(*roc_points(yy, ss)):
            s.add_data_point(a, b)
    diag = cd.add_series('случайно')
    diag.add_data_point(0, 0)
    diag.add_data_point(1, 1)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, x, y, w, h, cd)
    ch = gf.chart
    ch.has_legend = False
    ch.font.size = Pt(10)
    ch.font.name = FONT
    ch.font.color.rgb = C['ink3']
    for ax in (ch.category_axis, ch.value_axis):
        ax.minimum_scale, ax.maximum_scale, ax.major_unit = 0, 1, 0.2
        ax.has_major_gridlines = True
        ax.major_gridlines.format.line.color.rgb = C['lilac']
        ax.format.line.color.rgb = C['line']
        ax.tick_labels.number_format = '0.0'
        ax.tick_labels.number_format_is_linked = False
    styles = [(c[3], c[4]) for c in curves] + [(C['ink3'], 'dash')]
    for series, (color, dash) in zip(ch.series, styles):
        ln = series.format.line
        ln.color.rgb = color
        ln.width = Pt(2.5 if dash != 'dash' else 1)
        if dash:
            ln.dash_style = MSO_LINE_DASH_STYLE.DASH
        series.smooth = False
    return gf


def ci_track(slide, x, y, w, m, key, lo_ax, hi_ax, color, d=I(0.13)):
    v = m.get(key)
    lo, hi = m.get(f'{key}_ci', [None, None])
    if v is None or not np.isfinite(v):
        return
    X = lambda t: x + int((min(max(t, lo_ax), hi_ax) - lo_ax) / (hi_ax - lo_ax) * w)  # noqa: E731
    line(slide, x, y, x + w, y, C['line'], 0.75)
    for t in np.linspace(lo_ax, hi_ax, 5):
        line(slide, X(t), y - I(0.04), X(t), y + I(0.04), C['line'], 0.75)
    if lo is not None and np.isfinite(lo):
        line(slide, X(lo), y, X(hi), y, color, 3)
    oval(slide, X(v), y, d, color, line=C['white'], line_w=1.5)


# --------------------------------------------------------------------------- #
#  Заполнение обязательных слайдов шаблона
# --------------------------------------------------------------------------- #
def shape_by_id(slide, sid):
    return next(s for s in slide.shapes if s.shape_id == sid)


def set_paras(shape, paras, size=None):
    """Заменить текст, сохранив оформление абзацев и фрагментов шаблона.
    paras: [[текст фрагмента, ...], ...] или [(текст, {'bold':..}), ...]."""
    tf = shape.text_frame
    ps = list(tf.paragraphs)
    for k, runs in enumerate(paras):
        if k >= len(ps):
            new = copy.deepcopy(ps[-1]._p)
            ps[-1]._p.addnext(new)
            ps = list(tf.paragraphs)
        p = ps[k]
        rs = list(p.runs)
        for j, item in enumerate(runs):
            t, o = (item, {}) if isinstance(item, str) else item
            if j < len(rs):
                r = rs[j]
            else:
                r = p.add_run() if not rs else None
                if r is None:
                    new_r = copy.deepcopy(rs[-1]._r)
                    rs[-1]._r.addnext(new_r)
                    r = list(p.runs)[j]
            r.text = t
            if 'bold' in o:
                r.font.bold = o['bold']
            if 'italic' in o:
                r.font.italic = o['italic']
            if size or 'size' in o:
                r.font.size = Pt(o.get('size', size))
        for r in list(p.runs)[len(runs):]:
            r._r.getparent().remove(r._r)
    for p in list(tf.paragraphs)[len(paras):]:
        p._p.getparent().remove(p._p)


def remove(shape):
    shape._element.getparent().remove(shape._element)


def fill_mandatory(deck: TemplateDeck, T: dict, D: dict, notes: dict):
    s_title, s_about, s_team, s_story = deck.src

    # 1. Титул ---------------------------------------------------------------
    s = s_title
    for sh in list(s.shapes):
        if sh.is_placeholder and sh.placeholder_format.idx == 11:         # логотип — своей картинкой
            remove(sh)
    picture(s, ASSETS / 'brand' / 'moscow_medicine_white.png', I(0.55), I(0.62), h=I(0.9))
    text(s, I(0.55), I(1.62), I(5.4), I(0.3), 'постановщик задачи — Департамент здравоохранения Москвы',
         size=11, color=C['white'])
    for sh in list(s.shapes):
        if sh.is_placeholder and sh.placeholder_format.idx in (0, 12):
            remove(sh)
    text(s, I(0.55), I(4.15), I(8.0), I(0.8), 'Контроль качества DXA', size=42, bold=True, color=C['white'])
    text(s, I(0.55), I(4.95), I(7.0), I(0.8), 'Сервис ИИ для оценки качества исследований плотности костей',
         size=18, color=C['white'], spacing=1.1)
    rect(s, I(0.55), I(6.05), I(0.6), Emu(38100), fill=C['pink'])
    text(s, I(0.55), I(6.2), I(8.0), I(0.4), f'Команда {T["team"]} · ЛЦТ 2026', size=16, bold=True,
         color=C['white'])
    deck.notes(s, notes['title'])

    # 2. О команде и решении -------------------------------------------------------
    s = s_about
    for sh in list(s.shapes):
        if sh.is_placeholder and sh.placeholder_format.idx == 10:         # место под фото — снимок решения
            x, y, w, h = sh.left, sh.top, sh.width, sh.height
            remove(sh)
            ui = ASSETS / 'ui_results.png'
            if ui.exists():
                pic = s.shapes.add_picture(str(ui), x, y, w, h)
                with Image.open(ui) as im:
                    iw, ih = im.size
                k = (w / h) / (iw / ih)                                 # обрезка до пропорций места
                if k > 1:
                    pic.crop_bottom = 1 - 1 / k
                else:
                    pic.crop_right = 1 - k
                s.shapes._spTree.remove(pic._element)
                s.shapes._spTree.insert(2, pic._element)               # под панелью «О команде»
    ttl = next(sh for sh in s.shapes if sh.is_placeholder and sh.placeholder_format.idx == 0)
    ttl.text_frame.text = ''
    _run(ttl.text_frame.paragraphs[0], f'Команда {T["team"]}', 22, C['purple'], bold=True)
    set_paras(shape_by_id(s, 14), [
        [('Капитан: ', {'bold': True}), (T['captain'], {'bold': False})],
        [('Кол-во участников: ', {'bold': True}), (f'{len(T["members"])}', {'bold': False})],
        [('Краткое описание: ', {'bold': True})],
        [(T['about'][0], {'italic': True})],
        [(T['about'][1] if len(T['about']) > 1 else '', {'italic': True})],
        [('Город и регион: ', {'bold': True}), (T['city'], {'bold': False})]])
    body1, body2 = shape_by_id(s, 5), shape_by_id(s, 8)
    body1.height, body2.height = I(1.85), I(1.5)
    set_paras(body1, [[
        f'Сервис проверяет DXA-снимок позвоночника и бедра по пяти критериям ТЗ до того, как исследование '
        f'уйдёт к врачу: определяет область, находит нарушения укладки и отступов и объясняет каждое в '
        f'миллиметрах и градусах. Локально на CPU, {D["bench"]["per_study_s"]["median"]:.1f} с на исследование.']],
        size=13)
    set_paras(body2, [[
        'Вердикт проверяется глазами: измерение с нормой из ТЗ, нейросеть — там, где признак — форма, с '
        'тепловой картой. Отступы ROI оцениваются так, как их оценивает эксперт.']], size=13)
    deck.notes(s, notes['team_about'])

    # 3. Команда -------------------------------------------------------------------
    s = s_team
    ttl = next(sh for sh in s.shapes if sh.is_placeholder and sh.placeholder_format.idx == 0)
    ttl.text_frame.text = 'КОМАНДА'
    for sh in list(s.shapes):
        if sh.is_placeholder and sh.placeholder_format.idx not in (0, 4, 27):
            remove(sh)                                                  # места под фото
    cards = [(17, 15, 9), (56, 58, 57), (59, 61, 60), (62, 64, 63), (65, 67, 66)]
    members = T['members'][:5]
    n = len(members)
    cw, gap = I(2.4), I(0.12)
    x_start = int((W - (n * cw + (n - 1) * gap)) / 2)
    for i, (cid, nid, did) in enumerate(cards):
        c_, nm, dt = shape_by_id(s, cid), shape_by_id(s, nid), shape_by_id(s, did)
        if i >= n:
            for sh in (c_, nm, dt):
                remove(sh)
            continue
        dx = x_start + i * (cw + gap) - c_.left
        for sh in (c_, nm, dt):
            sh.left = sh.left + dx
        m = members[i]
        set_paras(nm, [[m['name']]])
        set_paras(dt, [[m['role']], [m['nick']], [m['phone']], [m['org']]])
        initials = ''.join(w[0] for w in m['name'].split()[:2]) if not m['name'].startswith('[') else '?'
        b = oval(s, c_.left + cw // 2, I(2.75), I(1.35), C['pink_soft'])
        tf = b.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _run(tf.paragraphs[0], initials.upper(), 30, C['purple'], bold=True)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    deck.notes(s, notes['team'])

    # 4. История и вызовы -----------------------------------------------------------
    s = s_story
    ttl = next(sh for sh in s.shapes if sh.is_placeholder and sh.placeholder_format.idx == 0)
    ttl.text_frame.text = 'ИСТОРИЯ И ВЫЗОВЫ'
    set_paras(shape_by_id(s, 37), [[T['history']]])
    b43 = shape_by_id(s, 43)
    b43.width, b43.height = I(10.9), I(1.05)
    set_paras(b43, [['Ошибка укладки незаметно портит измерение плотности кости и оценку лечения, а проверяет '
                     'её сегодня только врач вручную. И её можно решить прозрачно: каждое замечание — число '
                     'с нормой из ТЗ, которое врач проверит глазами.']])
    b40 = shape_by_id(s, 40)
    b40.width, b40.height = I(11.2), I(1.25)
    set_paras(b40, [
        ['Мало нарушений (6–36 на критерий) — честная out-of-fold оценка по исследованиям вместо подбора '
         'на всей выборке; сеть только там, где геометрии мало.'],
        ['Эталон расходился с буквой ТЗ по отступам ROI — разобрали все расхождения и нашли, как оценивает '
         'эксперт: по длине поля. Пустой PixelSpacing — масштаб восстановили по анатомии.']])
    deck.notes(s, notes['team_story'])


# --------------------------------------------------------------------------- #
#  Слайды решения
# --------------------------------------------------------------------------- #
def build_deck(D: dict, cases: dict) -> TemplateDeck:
    T = speech.team()
    SP = {s['key']: s for s in speech.slides(D, cases)}
    notes = {k: v['text'] + (['Если спросят: ' + ' '.join(v['ask'])] if v['ask'] else [])
             for k, v in SP.items()}
    deck = TemplateDeck(TEMPLATE)
    fill_mandatory(deck, T, D, notes)

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

    def new(key, style, section, headline=''):
        s = deck.new(style, section, headline)
        deck.notes(s, notes[key])
        return s

    # 5. Коротко о решении ------------------------------------------------------------
    s = new('brief', 'dark', 'Коротко о решении')
    cy, chh, cw2 = I(1.32), I(5.46), I(5.95)
    for j, (ttl_, items) in enumerate((
            ('Техническая суть решения', [
                'Вход — DICOM или zip как из PACS: дубликаты схлопываются по пикселям, ориентация '
                'приводится к стандартному виду.',
                'Область снимка — позвоночник, левое и правое бедро или «не DXA»: верно на 252 из 252.',
                'Пять критериев ТЗ — измерения в мм и градусах; для ротации и посторонних предметов — '
                'ансамбль ResNet-34 в своде с геометрией.',
                'Выход — таблица ТЗ 2.5, снимки с разметкой и тепловой картой, DICOM SR.',
                f'Out-of-fold: F1 {f2(ov["f1"])}, ROC-AUC {f2(ov["roc_auc"])}; {ps["median"]:.1f} с на '
                'исследование на CPU, без сети.']),
            ('Маркетинговая суть решения', [
                'Для кого: лаборанты и врачи отделений денситометрии, служба контроля качества лучевой '
                'диагностики.',
                'Ценность: замечание приходит, пока пациент на столе, — меньше повторных исследований и '
                'ошибок в плотности кости.',
                'Внедрение: DICOM-роутер → сервис → заключение DICOM SR в PACS; теневой режим → ассистент '
                '→ сеть медорганизаций.',
                'Поставка: контейнер для сервера, установщик для Windows, бот для показа; GPU не нужен.']))):
        x = X0 + j * (cw2 + I(0.48))
        rect(s, x, cy, cw2, chh, fill=C['white'], radius=0.035)
        text(s, x + I(0.3), cy + I(0.3), cw2 - I(0.6), I(0.5), ttl_, size=20, bold=True, color=C['purple'])
        bullets(s, x + I(0.3), cy + I(0.95), cw2 - I(0.6), chh - I(1.1), items, size=13, gap=8)

    # 6. Клиническая задача ---------------------------------------------------------------
    s = new('task', 'city', 'Клиническая задача', 'Ошибку укладки дешевле поймать, пока пациент на столе')
    label(s, X0, Y0, I(5.8), 'почему это важно')
    bullets(s, X0, Y0 + I(0.35), I(6.1), I(2.0), [
        'Плотность кости считается в области интереса. Позвоночник наклонён, бедро повёрнуто, в кадре '
        'застёжка белья — денситометр меряет не то.',
        'Ошибка уходит в T-критерий и ложную динамику: повторное облучение или неверная оценка лечения.',
        'Сегодня качество проверяет врач вручную, по памяти о критериях для каждой области.'], size=12.5)
    picture_fit(s, case('ok_hip'), X0, I(4.3), I(3.0), I(2.5), caption='норма: вертел виден',
                cap_h=I(0.36), cap_size=10.5)
    picture_fit(s, case('rotation'), X0 + I(3.12), I(4.3), I(3.0), I(2.5), caption='ротация: контур плавный',
                cap_h=I(0.36), cap_size=10.5)
    label(s, I(6.95), Y0, I(5.5), 'что делает сервис')
    for i, (t_, b_) in enumerate([
            ('Проверяет', 'Каждый снимок по пяти критериям ТЗ: укладка, ось, посторонние предметы, ротация '
                          'бедра, отступы области интереса.'),
            ('Объясняет', '«Ось 6.9° при норме до 5°», «поле 12.4 см, нужно 12.9 см», ориентиры и тепловая '
                          'карта на снимке.'),
            ('Отдаёт человеку', 'Лаборант исправляет укладку, пока пациент на столе; врач видит заключение '
                                'DICOM SR рядом со снимком в PACS.')]):
        y = Y0 + I(0.4) + i * I(1.5)
        number_badge(s, I(6.95), y, i + 1)
        text(s, I(7.6), y - I(0.02), I(5.3), I(0.35), t_, size=16, bold=True, color=C['purple'])
        text(s, I(7.6), y + I(0.38), I(5.3), I(0.95), b_, size=12.5, color=C['ink2'], spacing=1.12)

    # 7. Таксономия --------------------------------------------------------------------------
    s = new('taxonomy', 'light', 'Таксономия нарушений', 'Пять критериев ТЗ, у каждого — измерение в физических единицах')
    rows = [['код', 'критерий ТЗ 2.3', 'как измеряем', 'норма'],
            [[('spine_position', {'bold': True})], 'видны верхние края подвздошных костей',
             'доля кости в нижних латеральных зонах', '> 0.001'],
            [[('spine_axis', {'bold': True})], 'наклон оси до 5°', 'прямая по центрам тел; вторая оценка — по краям',
             '≤ 5° (порог 4°)'],
            [[('spine_artifacts', {'bold': True})], 'нет посторонних предметов', 'white top-hat + сеть, свод',
             f'p < {f2(art["threshold"])}'],
            [[('hip_rotation', {'bold': True})], 'нет ротации (малый вертел)', 'выступ малого вертела + сеть, свод',
             f'p < {f2(rot["threshold"])}'],
            [[('hip_roi', {'bold': True})], 'отступы 3 / 3 / 2 см', 'длина поля: 3 см + вертел — седалищная + 3 см',
             '≥ 12.9 см'],
            [[('undetermined', {'bold': True})], 'область не определена', 'фильтр «не похоже на DXA»', 'ручной разбор']]
    table(s, X0, Y0, I(8.55), rows, [1.95, 2.75, 3.4, 1.45], size=11.5, row_h=I(0.6),
          align=['l', 'l', 'l', 'l'])
    card(s, I(9.2), Y0, I(3.73), I(2.35), 'Несколько нарушений',
         'У 20 из 100 исследований нарушений больше одного. Критерии независимы, в отчёт идут все: '
         'spine_axis;spine_artifacts. quality_class = 1, если сработал хоть один.', dark=False, body_size=12)
    card(s, I(9.2), Y0 + I(2.5), I(3.73), I(1.7), 'Сбой чтения',
         'Непрочитанный файл — строка Failure с текстом ошибки и quality_class = 1.', dark=False, body_size=12)

    # 8. Данные -------------------------------------------------------------------------------
    s = new('data', 'dark', 'Данные', f'{D["frames"]} снимков, 6–36 нарушений на критерий')
    rect(s, X0, Y0, I(6.1), I(4.8), fill=C['white'], radius=0.04)
    for i, (v, cap) in enumerate((('100', 'исследований'), ('499', 'DICOM-файлов'),
                                  (str(D['frames']), 'уникальных снимков'))):
        stat(s, X0 + I(0.3) + i * I(1.9), Y0 + I(0.25), I(1.8), v, cap, size=32)
    reg = D['regions']
    bullets(s, X0 + I(0.3), Y0 + I(1.45), I(5.5), I(3.3), [
        'Дубликаты (Lunar выгружает снимок по нескольку раз с разными UID) находим по MD5 пикселей — до '
        'разбиения на фолды.',
        f'Позвоночник {reg.get("spine", 0)}, левое бедро {reg.get("lh", 0)}, правое {reg.get("rh", 0)}; '
        'не больше трёх снимков на исследование.',
        'Один аппарат — GE Lunar Prodigy Advance. PixelSpacing пуст: масштаб 0.600 / 0.607 мм/px '
        'восстановлен по шагу «позвонок + диск».',
        'Своя разметка: 242 снимка × 29 ключевых точек.'], size=12)
    rect(s, I(6.7), Y0, I(6.23), I(4.8), fill=C['white'], radius=0.04)
    text(s, I(7.0), Y0 + I(0.25), I(5.7), I(0.4), 'Нарушений мало, и они разные', size=16, bold=True,
         color=C['purple'])
    items = [('position', 'Укладка'), ('axis', 'Ось > 5°'), ('artifacts', 'Посторонние предметы'),
             ('rotation', 'Ротация бедра'), ('roi', 'Отступы ROI')]
    maxn = max(nn for _, nn in D['pos'].values())
    for i, (k, lab) in enumerate(items):
        pos, nn = D['pos'][k]
        yy = Y0 + I(0.8) + i * I(0.46)
        text(s, I(7.0), yy, I(2.3), I(0.3), lab, size=12)
        full = int(I(2.3) * nn / maxn)
        rect(s, I(9.35), yy + I(0.04), full, I(0.22), fill=C['pink_soft'], radius=0.4)
        rect(s, I(9.35), yy + I(0.04), max(int(full * pos / nn), I(0.08)), I(0.22), fill=C['pink'], radius=0.4)
        text(s, I(9.35) + full + I(0.1), yy, I(1.2), I(0.3), f'{pos} из {nn}', size=11.5, bold=True,
             color=C['ink2'])
    for i, (v, cap) in enumerate((('48', 'исследований из 100\nс нарушением'), ('20', 'с несколькими\nнарушениями'),
                                  ('72', 'снимка из 249\nс нарушением'))):
        stat(s, I(7.0) + i * I(1.95), Y0 + I(3.3), I(1.9), v, cap, size=28, color=C['purple'])

    # 9. Разбиение ------------------------------------------------------------------------------
    s = new('folds', 'light', 'Разбиение и утечки', 'Честная оценка: исследование целиком в одном фолде')
    cw_, gap = I(0.98), I(0.08)
    for i in range(5):
        text(s, X0 + I(0.4) + i * (cw_ + gap), Y0, cw_, I(0.25), f'фолд {i}', size=10, bold=True,
             color=C['ink3'], align=PP_ALIGN.CENTER)
    for k in range(5):
        yy = Y0 + I(0.32) + k * I(0.46)
        text(s, X0, yy + I(0.08), I(0.35), I(0.25), f'#{k + 1}', size=10, bold=True, color=C['ink3'])
        for i in range(5):
            val = i == k
            b = rect(s, X0 + I(0.4) + i * (cw_ + gap), yy, cw_, I(0.37),
                     fill=C['pink'] if val else C['lilac'], radius=0.2)
            tf = b.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = 0
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
            _run(tf.paragraphs[0], 'проверка' if val else 'обучение', 10, C['white'] if val else C['purple'],
                 bold=val)
    st = D['strata']
    text(s, X0, Y0 + I(2.8), I(5.8), I(1.4),
         f'5 фолдов по 20 исследований, folds.csv в репозитории. Страты: норма {st.get("norm", 0)}, ротация '
         f'{st.get("rotation", 0)}, артефакты {st.get("artifacts", 0)}, ось {st.get("axis", 0)}, укладка '
         f'{st.get("spine_position", 0)}, ROI {st.get("roi", 0)} исследований.', size=12.5, color=C['ink2'],
         spacing=1.15)
    for i, (t_, b_) in enumerate([
            ('Группа — исследование', 'Позвоночник и оба бедра пациента всегда в одном фолде: разнесённые '
                                      'бёдра завысили бы метрику.'),
            ('Страта — редчайшее нарушение', 'В каждом фолде есть примеры самых редких классов.'),
            ('Подбор только на обучающих фолдах', 'Пороги, коридор нормы, веса сети, свод и его порог.'),
            ('Дисбаланс', 'Вес класса в потере сети, порог по доле нарушений, ДИ бутстрэпом по исследованиям.')]):
        card(s, I(6.7), Y0 + i * I(1.2), I(6.23), I(1.08), t_, b_, dark=False, title_size=14, body_size=11.5)

    # 10. Подход ----------------------------------------------------------------------------------
    s = new('approach', 'dark', 'Подход', 'Геометрия там, где критерий — измерение; сеть — где форма')
    seeds = rm.get('cnn_oof_by_seed', {})
    one = np.mean([v['roc_auc'] for v in seeds.values()]) if seeds else None
    rect(s, X0, Y0, I(7.7), I(4.55), fill=C['white'], radius=0.035)
    text(s, X0 + I(0.3), Y0 + I(0.22), I(7.1), I(0.35), 'Что пробовали для ротации бедра', size=15, bold=True,
         color=C['purple'])
    table(s, X0 + I(0.3), Y0 + I(0.7), I(7.1), [
        ['вариант', 'ROC-AUC', 'вывод'],
        ['выступ малого вертела по контуру', f2(rm['geometry']['roc_auc']), 'разрыв классов ≈ 2 мм'],
        ['вид кадра, логистическая регрессия', '0.50', 'вертел теряется'],
        ['модель ключевых точек', '—', 'ошибка выступа 0.75 мм'],
        ['ResNet-34, одна модель', f2(one), 'зависит от сида'],
        ['ансамбль трёх сидов', f2(rm['cnn_oof']['roc_auc']), 'стабильнее'],
        [[('свод: ансамбль + геометрия', {'bold': True})], [(f2(rm['stack_oof']['roc_auc']), {'bold': True})],
         [('в поставке', {'bold': True})]]],
        [3.4, 1.0, 2.7], size=12, row_h=I(0.5), align=['l', 'r', 'l'], highlight=(6,))
    for i, (t_, b_) in enumerate([
            ('Принцип', 'Вердикт проверяется глазами: «ось 6.9°», «поле 12.4 см». Основа — геометрия в мм '
                        'и градусах.'),
            ('Где геометрии мало', 'Ротация и посторонние предметы — признаки формы: там геометрию дополняет '
                                   'сеть, вердикт — свод, тепловая карта показывает, куда смотрела сеть.'),
            ('Где сеть не нужна', 'Ось, укладка, отступы — прямые измерения. Сеть для оси: ROC-AUC 0.51.')]):
        card(s, I(8.3), Y0 + i * I(1.55), I(4.63), I(1.43), t_, b_, title_size=14, body_size=11.5)

    # 11. Архитектура -------------------------------------------------------------------------------
    s = new('architecture', 'light', 'Архитектура', 'Один контейнер, пять шагов, ни одного обращения в сеть')
    nodes = [('DICOM, zip', 'выгрузка из PACS как есть'), ('pydicom + gdcm', 'сжатие, ориентация, дубликаты'),
             ('region_clf', 'позвоночник, бёдра, «не DXA»'), ('геометрия + CNN', 'пять критериев независимо'),
             ('свод', 'класс, вероятности, текст, SR')]
    step = CW // 5
    line(s, X0 + I(0.23), Y0 + I(0.23), X0 + 4 * step + I(0.23), Y0 + I(0.23), C['pink'], 1.25)
    for i, (t_, b_) in enumerate(nodes):
        x = X0 + i * step
        number_badge(s, x, Y0, i + 1)
        text(s, x, Y0 + I(0.6), step - I(0.2), I(0.3), t_, size=14, bold=True, color=C['purple'])
        text(s, x, Y0 + I(0.95), step - I(0.25), I(0.5), b_, size=11.5, color=C['ink2'])
    for i, (t_, b_) in enumerate([
            ('Выход по ТЗ 2.5 и 2.7', 'xlsx или csv, строка на снимок, 8 колонок ТЗ; zip с визуализацией; '
                                      'zip с DICOM SR.'),
            ('Интерфейсы', 'Веб-приложение, приложение для Windows, Telegram-бот, CLI и REST API.'),
            ('Контейнер', 'python:3.11-slim по дайджесту, lock-файл, onnxruntime на CPU, один поток — '
                          'побитово тот же отчёт.')]):
        card(s, X0 + i * I(4.22), Y0 + I(1.6), I(4.1), I(1.3), t_, b_, dark=False, title_size=14, body_size=11.5)
    from service.pipeline import COLUMNS as COLS8

    def cut(v, k=13):
        v = '' if v is None else str(v)
        return v if len(v) <= k else '…' + v[-k:]
    sample = [COLS8] + [[cut(r.get(c)) if c in ('path_to_study', 'study_uid', 'image_uid')
                         else ('' if r.get(c) is None else str(r.get(c))) for c in COLS8]
                        for r in D['sample'][:4]]
    label(s, X0, Y0 + I(3.05), CW, 'фрагмент настоящего отчёта: восемь колонок ТЗ 2.5', color=C['purple'])
    table(s, X0, Y0 + I(3.35), CW, sample, [1.6, 1.6, 1.6, 1.55, 1.2, 1.35, 1.55, 1.55], size=10,
          header_size=8.5, row_h=I(0.3), align=['l'] * 8)

    # 12. Модели ----------------------------------------------------------------------------------------
    s = new('models', 'dark', 'Модели', 'Шесть модулей, у каждого своя проверяемая задача')
    ens = rot.get('onnx', {})
    mb = ens.get('bytes', 0) / 1e6 / max(len(ens.get('files', [1])), 1)
    cards = [
        ('region_clf', 'Область снимка', 'Признаки кадра + логистическая регрессия; фильтр новизны отсекает '
         'не-DXA на ручной разбор.', '252 / 252 снимка', False),
        ('spine_qc', 'Позвоночник', 'Маска кости, коридор колонны, центры тел. Ось — Тейл–Сен, укладка — масса '
         'гребней, артефакты — white top-hat.', 'ROC-AUC OOF: ось 0.82 · укладка 0.82', False),
        ('hip_roi', 'Отступы бедра', 'Длина поля сканирования против 3 см + вертел — седалищная + 3 см; отступы '
         'по рисунку 6 — справочно.', f'F1 {f2(rs.get("f1"))} · ROC-AUC {f2(rs.get("roc_auc"))}', False),
        ('hip_rotation', 'Ротация: геометрия', 'Выступ малого вертела над медиальным контуром диафиза, мм²; '
         'коридор ловит пере- и недоротацию.', 'коридор 96–272 мм²', False),
        ('cnn_qc', 'Ротация и предметы: сеть', 'ResNet-34, 320 px, ансамбль 3 сидов, ONNX на CPU; свод с '
         'геометрией, тепловая карта.', f'ROC-AUC свода 0.82 и 0.85 · {mb:.0f} МБ × 3', True),
        ('dxa_qc · по флагу', 'Ключевые точки', 'U-Net + ResNet-34, 29 точек; обучены фолды 0–1 из 5. По оси не '
         'хуже контура, но в 12 раз медленнее — по флагу.', 'ошибка 0.9–2.4 мм · угол 0.50°', False)]
    cw3, ch2 = I(4.08), I(2.3)
    for i, (k, t_, b_, m_, acc) in enumerate(cards):
        card(s, X0 + (i % 3) * (cw3 + I(0.155)), Y0 + (i // 3) * (ch2 + I(0.17)), cw3, ch2, t_, b_, kicker=k,
             metric=m_, accent=acc, title_size=15, body_size=11.5)

    # 13. Предобработка -----------------------------------------------------------------------------------
    s = new('prepost', 'city', 'Предобработка и постобработка',
            'Всё приводится к миллиметрам до модели и к вероятности после')
    for j, (lab, items) in enumerate((
            ('предобработка', ['Сжатые синтаксисы, инверсия MONOCHROME1, VOI LUT.',
                               'Ориентация по PatientOrientation: зеркальная выгрузка → стандартный AP-вид.',
                               'Дедупликация по MD5 пикселей: одна строка отчёта на снимок.',
                               'Правое бедро отражается в левое: измерения и сеть видят одну сторону.',
                               'Маска кости и масштаб мм/px: пороги в мм, градусах, мм².',
                               'Для сети: яркость по перцентилям 0.5–99.5, длинная сторона → 320 px.']),
            ('постобработка', ['Пороги и коридоры — из калибровки на обучающих фолдах.',
                               'Отступы ROI — по длине поля, как оценивает эксперт.',
                               'Свод сети с геометрией; порог — доля нарушений в обучении.',
                               'Платт для каждого критерия → quality_probability = 1 − Π(1 − p).',
                               'Расхождение методов — флаг и низкая уверенность.',
                               'Текст по-русски, снимок с тепловой картой, DICOM SR.']))):
        x = X0 + j * I(6.3)
        label(s, x, Y0, I(5.9), lab)
        bullets(s, x, Y0 + I(0.4), I(5.9), I(4.4), items, size=13.5, gap=9)

    # 14. Метрики ---------------------------------------------------------------------------------------------
    s = new('metrics', 'light', 'Метрики качества', 'Out-of-fold с 95% доверительными интервалами')
    names = {'spine_position': 'Укладка', 'spine_axis': 'Ось', 'spine_artifacts': 'Посторонние предметы',
             'hip_rotation': 'Ротация бедра', 'hip_roi': 'Отступы ROI'}
    rowsm = [(names[k], hv[k], h0v.get(k)) for k in names] + \
        [('Позвоночник, кадр', h['per_region']['spine'], hb0['per_region']['spine']),
         ('Бедро, кадр', h['per_region']['hip'], hb0['per_region']['hip']),
         ('Всего, «есть нарушение»', ov, ov0)]
    cx = dict(name=X0, pos=I(3.1), sens=I(3.95), spec=I(4.75), f1bar=I(5.6), f1=I(8.3), aucbar=I(9.25),
              auc=I(12.05))
    yy = Y0 - I(0.1)
    for key, lab in (('name', 'out-of-fold'), ('pos', 'нар./n'), ('sens', 'чувств.'), ('spec', 'специф.'),
                     ('f1bar', 'F1, 95% ДИ'), ('f1', 'F1'), ('aucbar', 'ROC-AUC, 95% ДИ (0.4–1)'), ('auc', 'AUC')):
        text(s, cx[key], yy, I(2.6), I(0.25), lab, size=9, bold=True, color=C['purple'], caps=True)
    line(s, X0, yy + I(0.3), XR, yy + I(0.3), C['purple'], 1.5)
    rh = I(0.5)
    for i, (lab, m, m0) in enumerate(rowsm):
        y0 = yy + I(0.38) + i * rh
        both = m0 is not None and abs((m0.get('f1') or 0) - (m.get('f1') or 0)) > 1e-9
        last = i == len(rowsm) - 1
        text(s, cx['name'], y0 + I(0.11), I(2.6), I(0.3), lab, size=12, bold=last)
        for key, val in (('pos', f"{m['positives']}/{m['n']}"), ('sens', f2(m['sensitivity'])),
                         ('spec', f2(m['specificity']))):
            text(s, cx[key], y0 + I(0.11), I(0.8), I(0.3), val, size=11.5)
        for key, lo_ax, xb, xn in (('f1', 0, cx['f1bar'], cx['f1']), ('roc_auc', 0.4, cx['aucbar'], cx['auc'])):
            if both:
                ci_track(s, xb, y0 + I(0.13), I(2.5), m0, key, lo_ax, 1, C['lav'], d=I(0.11))
                ci_track(s, xb, y0 + I(0.34), I(2.5), m, key, lo_ax, 1, C['pink'])
                text(s, xn, y0 + I(0.02), I(0.9), I(0.22), f2(m0.get(key)), size=10, color=C['ink3'])
                text(s, xn, y0 + I(0.22), I(0.9), I(0.25), f2(m.get(key)), size=12, bold=True)
            else:
                ci_track(s, xb, y0 + I(0.23), I(2.5), m, key, lo_ax, 1, C['pink'])
                text(s, xn, y0 + I(0.11), I(0.9), I(0.25), f2(m.get(key)), size=12, bold=True)
        line(s, X0, y0 + rh - I(0.02), XR, y0 + rh - I(0.02), C['line'], 0.5)
    ly = yy + I(0.45) + len(rowsm) * rh
    rect(s, X0, ly + I(0.07), I(0.3), I(0.1), fill=C['lav'])
    text(s, X0 + I(0.38), ly, I(4.3), I(0.25), 'исходное решение: геометрия, ROI по рисунку 6', size=10.5,
         color=C['ink2'])
    rect(s, I(5.0), ly + I(0.07), I(0.3), I(0.1), fill=C['pink'])
    text(s, I(5.38), ly, I(3.6), I(0.25), 'сейчас: сеть + правило эксперта для ROI', size=10.5, color=C['ink2'])
    text(s, I(9.0), ly, I(3.93), I(0.25), f"macro-F1 {f2(hb0['overall']['macro_f1'])} → "
         f"{f2(h['overall']['macro_f1'])} · область 252/252", size=10.5, bold=True, color=C['purple'],
         align=PP_ALIGN.RIGHT)

    # 15. Эксперименты --------------------------------------------------------------------------------------
    s = new('experiments', 'light', 'Эксперименты', 'Сеть поверх геометрии: ROC-кривые out-of-fold')
    ro, ao = D['rot_oof'], D['art_oof']
    for j, (ttl_, dd, m, geo_lab) in enumerate((
            (f'Ротация бедра · {rot["positives"]} из {rot["n"]}', ro, rm, 'выступ'),
            (f'Посторонние предметы · {art["positives"]} из {art["n"]}', ao, am, 'top-hat'))):
        x0 = X0 + j * I(6.35)
        text(s, x0, Y0, I(6.0), I(0.35), ttl_, size=15, bold=True, color=C['purple'])
        curves = [(geo_lab, dd.y, dd.geometry.fillna(0).values, C['lav'], 'dash'),
                  ('сеть', dd.y, dd.p_cnn.values, C['pink2'], None),
                  ('свод', dd.y, dd.p_stack.values, C['purple'], None)]
        roc_chart(s, x0 - I(0.1), Y0 + I(0.4), I(3.55), I(3.55), curves)
        lx = x0 + I(3.6)
        for k, (lab, col, key) in enumerate(((geo_lab, C['lav'], 'geometry'), ('сеть', C['pink2'], 'cnn_oof'),
                                              ('свод', C['purple'], 'stack_oof'))):
            rect(s, lx, Y0 + I(0.6) + k * I(0.34), I(0.28), I(0.1), fill=col)
            text(s, lx + I(0.38), Y0 + I(0.51) + k * I(0.34), I(2.2), I(0.28), f'{lab} {f2(m[key]["roc_auc"])}',
                 size=12, bold=(key == 'stack_oof'))
        text(s, lx, Y0 + I(1.7), I(2.45), I(0.5), 'вердикт вложенно: свод и порог без отложенного фолда',
             size=10, color=C['ink3'], spacing=1.05)
        g, v = D['h0'][('hip_rotation' if j == 0 else 'spine_artifacts')], m['verdict_nested']
        table(s, lx, Y0 + I(2.3), I(2.5), [['', 'чув.', 'спец.', 'F1'],
                                          [geo_lab, f2(g['sensitivity']), f2(g['specificity']), f2(g['f1'])],
                                          ['свод', f2(v['sensitivity']), f2(v['specificity']),
                                           [(f2(v['f1']), {'bold': True})]]],
              [1.1, 0.75, 0.75, 0.7], size=11, header_size=8.5, row_h=I(0.34))
        text(s, x0, Y0 + I(4.05), I(3.5), I(0.25), '1 − специфичность →   ↑ чувствительность', size=9.5,
             color=C['ink3'])

    # 16. Сравнение вариантов ----------------------------------------------------------------------------------
    s = new('comparison', 'city', 'Сравнение вариантов', 'Что проверили и почему оставили то, что оставили')
    # Модель точек обучена на фолдах 0–1 из 5; контур и точки — на одних и тех же кадрах этих фолдов
    kp, kc = D.get('kp_oof', {}), D.get('kp_contour', {})

    def axis(m):
        a = m.get('per_violation', {}).get('spine_axis', {})
        return f"{f2(a.get('f1'))} / {f2(a.get('roc_auc'))}"

    def ms(m):
        v = m.get('overall', {}).get('seconds_per_image_median')
        return '—' if v is None else f'{v * 1000:.0f} мс'
    n_ax = kc.get('per_violation', {}).get('spine_axis', {}).get('positives', 0)
    text(s, X0, Y0, I(6.0), I(0.35), 'Контур против модели точек', size=14, bold=True, color=C['purple'])
    table(s, X0, Y0 + I(0.42), I(6.0), [
        ['фолды 0–1, out-of-fold', 'контур', 'точки U-Net'],
        ['ось: F1 / ROC-AUC', axis(kc), axis(kp)],
        ['macro-F1 по типам', f2(kc.get('overall', {}).get('macro_f1'), 3), f2(kp.get('overall', {}).get('macro_f1'), 3)],
        ['время на снимок, CPU', ms(kc), ms(kp)]], [2.6, 1.5, 1.7], size=12, row_h=I(0.42))
    text(s, X0, Y0 + I(2.2), I(6.0), I(0.9), f'Нарушений оси на этих фолдах всего {n_ax}. Модель обучена на '
         'двух фолдах из пяти и на порядок медленнее — по умолчанию выключена.', size=11.5, color=C['ink2'],
         spacing=1.12)
    sw = D['spine_oof'].get('position', {}).get('rule_sweep', {})
    ru = {'geometry': 'геометрия (в поставке)', 'appearance': 'вид кадра', 'and': 'обе согласны',
          'or': 'любая из двух'}
    text(s, I(6.75), Y0, I(6.18), I(0.35), 'Правило свода для укладки', size=14, bold=True, color=C['purple'])
    table(s, I(6.75), Y0 + I(0.42), I(6.18), [['out-of-fold, 6 нарушений', 'найдено', 'лишних', 'F1']] +
          [[ru.get(k, k), str(v['tp']), str(v['fp']), f2(v['f1'])] for k, v in sw.items()],
          [2.8, 1.0, 1.0, 0.9], size=12, row_h=I(0.36), highlight=(1,))
    text(s, I(6.75), Y0 + I(2.2), I(6.18), I(0.9), 'На всей выборке лучшим казалось согласие обеих оценок '
         '(F1 0.73); out-of-fold оно ловит одно нарушение из шести.', size=11.5, color=C['ink2'], spacing=1.12)
    text(s, X0, Y0 + I(3.15), CW, I(0.35), 'Сеть для ротации: что перебрали (ROC-AUC out-of-fold)', size=14,
         bold=True, color=C['purple'])
    table(s, X0, Y0 + I(3.57), CW, [
        ['ResNet-18', 'EfficientNet-B0', 'ResNet-34, 256 px', 'ResNet-34, 1 сид', 'энкодер точек', 'ResNet-34 × 3',
         '+ геометрия'],
        ['0.76', '0.73', '0.77', f2(one), '0.79', f2(rm['cnn_oof']['roc_auc']),
         [(f2(rm['stack_oof']['roc_auc']), {'bold': True, 'color': C['pink']})]]],
        [1] * 7, size=13, header_size=9, row_h=I(0.42), align=['c'] * 7)

    # 17. Эксперт -----------------------------------------------------------------------------------------------
    s = new('expert', 'dark', 'Анализ ошибок', 'Разбор ошибок показал, как оценивает эксперт')
    rect(s, X0, Y0, I(6.25), I(4.75), fill=C['white'], line=C['pink'], line_w=2, radius=0.035)
    label(s, X0 + I(0.3), Y0 + I(0.22), I(5.7), 'отступы ROI · что делает эксперт')
    text(s, X0 + I(0.3), Y0 + I(0.5), I(5.7), I(0.4), 'Судит по длине поля', size=18, bold=True, color=C['purple'])
    table(s, X0 + I(0.3), Y0 + I(1.05), I(5.65), [
        ['правило', 'найдено', 'лишних', 'F1', 'AUC'],
        ['отступы по рисунку 6', f"{rmg.get('tp')}/7", str(rmg.get('fp')), f2(rmg.get('f1')), f2(rmg.get('roc_auc'))],
        [[('длина поля ≥ 12.9 см', {'bold': True})], f"{rs.get('tp')}/7", str(rs.get('fp')),
         [(f2(rs.get('f1')), {'bold': True})], f2(rs.get('roc_auc'))]],
        [2.3, 0.95, 0.95, 0.65, 0.65], size=11.5, row_h=I(0.44), highlight=(2,))
    text(s, X0 + I(0.3), Y0 + I(2.5), I(5.65), I(2.1),
         'Эксперт ставит «норму» при 1.7–3.0 см под седалищной костью и «нарушение» на коротких полях. Правило '
         'длины поля не подбиралось по меткам: 30 мм — из ТЗ, 69 мм от вертела до седалищной кости — медиана '
         'анатомии. Отступы по рисунку 6 считаются и показываются; режим строго по ТЗ — флаг --roi-rule margins.',
         size=12, color=C['ink2'], spacing=1.14)
    ro_ = D['rot_oof']
    geo_bad = ro_.geometry.fillna(0) > 0
    st_bad = ro_.pred_nested.astype(bool)
    dis = geo_bad != st_bad
    right = int((st_bad[dis] == ro_.y[dis].astype(bool)).sum())
    for i, (t_, b_, hh) in enumerate([
            ('Ось: сколиоз — не нарушение', 'При сколиозе эксперт не ставит нарушение оси ни в одном из 14 '
             'случаев, даже при угле до 8.8°. Это главный источник наших лишних срабатываний; по поясничному '
             'треку сколиоз от наклона не отличить (сеть — ROC-AUC 0.51).', I(1.85)),
            ('Ротация: сеть против выступа', f'Расхождений сети и выступа: {int(dis.sum())} из {len(ro_)}; прав '
             f'свод — {right}, выступ — {int(dis.sum()) - right}. Такие снимки помечены флагом.', I(1.3)),
            ('Укладка: 6 примеров', 'ROC-AUC 0.82, но отсечка на фолде с одним-двумя нарушениями неустойчива: '
             'F1 out-of-fold 0.32.', I(1.3))]):
        y = Y0 + [0, I(1.98), I(3.43)][i]
        card(s, I(6.85), y, I(6.08), hh, t_, b_, title_size=14, body_size=11.5)

    # 18. Кейсы ------------------------------------------------------------------------------------------------
    s = new('cases', 'dark', 'Кейсы', 'Каждое замечание видно на снимке')
    caps = [('axis', f'ось {f2(ca.get("angle"), 1)}°, норма до 5°\nspine_axis · эксперт согласен'),
            ('artifacts', f'посторонний предмет · сеть {f2(cart.get("art_p"))}\nspine_artifacts · эксперт согласен'),
            ('rotation', f'выступ {f2(cr.get("area"), 0)} мм² (норма 96–272)\nhip_rotation · эксперт согласен'),
            ('roi', f'поле {f2((croi.get("scan_len") or 0) / 10, 1)} см < 12.9 см\nhip_roi · эксперт согласен')]
    fw = (CW - 3 * I(0.17)) // 4
    for i, (k, cap) in enumerate(caps):
        if k in cases:
            picture_fit(s, case(k), X0 + i * (fw + I(0.17)), Y0, fw, I(4.55), caption=cap, cap_h=I(0.62),
                        cap_size=10.5)
    text(s, X0, Y0 + I(4.65), CW, I(0.3), 'Снимки обучающего набора, вероятности сети — out-of-fold. Оранжевым — '
         'тепловая карта сети, голубым — ориентиры, зелёным и красным — пороги.', size=10.5, color=C['white'])

    # 19. Скорость ---------------------------------------------------------------------------------------------
    s = new('speed', 'light', 'Скорость и требования', 'Секунды на исследование на обычном процессоре')
    for i, (v, u, cap) in enumerate(((f'{ps["median"]:.1f}', ' с', f'медиана на исследование\n(максимум {ps["max"]:.1f} с)'),
                                     ('180', ' с', 'лимит ТЗ 2.7\nна исследование'),
                                     (f'{pf["median"] * 1000:.0f}', ' мс', 'медиана на снимок,\nодин поток CPU'),
                                     (f'{D["bench"]["total_s"] / 60:.1f}', ' мин', 'все 100 исследований\nв один процесс'))):
        stat(s, X0 + (i % 2) * I(2.95), Y0 + (i // 2) * I(1.45), I(2.8), v, cap, unit=u, size=36,
             color=C['pink'] if i != 1 else C['purple'])
    label(s, X0, Y0 + I(3.05), I(5.8), 'худшее исследование против лимита, в масштабе', color=C['purple'])
    rect(s, X0, Y0 + I(3.38), I(5.6), I(0.32), fill=C['lilac'], radius=0.3)
    rect(s, X0, Y0 + I(3.38), max(int(I(5.6) * ps['max'] / 180), I(0.08)), I(0.32), fill=C['pink'], radius=0.3)
    text(s, X0 + I(0.25), Y0 + I(3.43), I(4.5), I(0.25), f'{ps["max"]:.1f} с — {ps["max"] / 180 * 100:.1f} % лимита',
         size=10.5, bold=True, color=C['purple'])
    text(s, X0, Y0 + I(3.9), I(5.8), I(0.8), 'В контейнере (Linux, Python 3.11) — 158 с на 100 исследований с '
         'оверлеями и SR. Без сети — 27 мс на снимок.', size=11.5, color=C['ink2'], spacing=1.12)
    table(s, I(6.8), Y0, I(6.13), [['', 'минимум', 'рекомендуется'],
                                   ['CPU', '2 ядра x86-64', '4 ядра'], ['RAM', '2 ГБ', '4 ГБ'],
                                   ['диск', '2 ГБ (образ 1.4 ГБ)', '5 ГБ'], ['GPU', 'не нужен', 'не нужен'],
                                   ['ПО', 'Docker или Windows 10/11', 'Linux-сервер']],
          [1.0, 2.6, 2.4], size=12.5, row_h=I(0.45), align=['l', 'l', 'l'])
    card(s, I(6.8), Y0 + I(2.95), I(6.13), I(1.7), 'Воспроизводимость',
         'Повторный прогон на тех же данных даёт побитово тот же отчёт: фиксированный порядок обхода, модели на '
         'диске, вычисления в один поток. Контейнер, приложение и Python сверены на 252 снимках.',
         dark=False, title_size=14, body_size=11.5)

    # 20. Интерфейсы -------------------------------------------------------------------------------------------
    s = new('interfaces', 'city', 'Интерфейсы', 'Четыре входа в один конвейер')
    bx, by, bw, bh = X0, Y0, I(7.05), I(4.65)
    rect(s, bx, by, bw, bh, fill=C['white'], line=C['purple'], line_w=1.25, radius=0.03)
    for k, col in enumerate((C['pink'], C['lav'], C['lilac'])):
        oval(s, bx + I(0.25) + k * I(0.2), by + I(0.2), I(0.12), col)
    rect(s, bx + I(1.0), by + I(0.1), bw - I(1.2), I(0.22), fill=C['lilac'], radius=0.5)
    text(s, bx + I(1.15), by + I(0.11), I(4), I(0.2), 'localhost:8000 — веб-приложение и окно для Windows', size=9,
         color=C['purple'])
    ui = ASSETS / 'ui_rotation.png'
    if ui.exists():
        picture_fit(s, ui, bx + I(0.08), by + I(0.42), bw - I(0.16), bh - I(0.5), frame=False)
    phone = phone_image()
    px, pw_, ph_ = I(7.65), I(2.35), I(4.65)
    rect(s, px, Y0, pw_, ph_, fill=C['deep'], radius=0.12)
    if phone:
        picture_fit(s, phone, px + I(0.1), Y0 + I(0.3), pw_ - I(0.2), ph_ - I(0.45), frame=False)
    rect(s, px + pw_ // 2 - I(0.35), Y0 + I(0.1), I(0.7), I(0.1), fill=C['ink'], radius=0.5)
    for i, (t_, b_) in enumerate([
            ('Веб-приложение', 'Папка или zip, снимки с разметкой; ставится из браузера.'),
            ('Windows', 'Установщик без прав администратора, консоль для пакетов.'),
            ('Telegram-бот', 'Вердикт, снимки и отчёт в чате; обезличенные данные.'),
            ('API и PACS', 'POST /batch → отчёт; DICOM SR в то же исследование.')]):
        card(s, I(10.2), Y0 + i * I(1.18), I(2.73), I(1.08), t_, b_, dark=False, title_size=12, body_size=9.5)

    # 21. Ценность ----------------------------------------------------------------------------------------------
    s = new('value', 'dark', 'Ценность для здравоохранения', 'Качество исследования — в момент съёмки, а не на повторе')
    for i, (k, t_, b_) in enumerate([
            ('лаборант', 'Подсказка у аппарата', 'Замечание с числом и нормой через 2 секунды: укладку '
             'исправляют, пока пациент на столе.'),
            ('врач', 'Исследование, которое можно читать', 'Заключение о качестве DICOM SR рядом со снимком; '
             'сомнительные снимки помечены флагами.'),
            ('медорганизация', 'Меньше повторов, единый стандарт', 'Одни и те же критерии ТЗ для всех '
             'аппаратов и смен; доля нарушений — в мониторинге.')]):
        card(s, X0 + i * I(4.23), Y0, I(4.1), I(2.35), t_, b_, kicker=k, title_size=15, body_size=12)
    for i, (t_, b_) in enumerate([
            ('Как поставляется', ['Контейнер для сервера МО или облака Департамента.',
                                  'Установщик для Windows — ноутбук в кабинете без Docker.',
                                  'GPU и интернет не нужны, снимки не уходят наружу.']),
            ('Как измеряем эффект', ['Доля повторных исследований из-за укладки: до и после.',
                                     'Согласие с экспертом на слепой выборке раз в месяц.',
                                     'Время врача на контроль качества.'])]):
        card(s, X0 + i * I(6.35), Y0 + I(2.55), I(6.2), I(2.2), t_, b_, title_size=15, body_size=12)

    # 22. Ограничения -------------------------------------------------------------------------------------------
    s = new('limits', 'city', 'Ограничения', 'Что сервис не умеет и почему')
    table(s, X0, Y0, CW, [
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
         'Разметка Th12; модель точек уже выдаёт th12_top/bottom'],
        [[('Бот и Telegram', {'bold': True})], 'Файлы идут через серверы Telegram (ТЗ 3.2)',
         'Только обезличенные данные; основной контур — без сети']],
        [2.3, 4.4, 4.4], size=12, row_h=I(0.58), align=['l', 'l', 'l'])

    # 23. Пилот -------------------------------------------------------------------------------------------------
    s = new('pilot', 'light', 'Внедрение и пилот', 'Теневой режим → ассистент лаборанта → сеть медорганизаций')
    phases = [('месяцы 1–2 · теневой режим', '1–2 медорганизации', 'DICOM-роутер отправляет копии в сервис, SR '
               'возвращается в PACS «не проверено». Считаем согласие с экспертом.'),
              ('месяцы 3–6 · ассистент', 'Замечание до отпуска пациента', 'Результат у лаборанта через 2–3 с после '
               'снимка. Отметки операторов — в перекалибровку.'),
              ('месяцы 6–12 · масштабирование', 'Другие аппараты и сеть МО', 'Проверка масштаба, дообучение сети '
               'на снимках аппарата, мониторинг доли нарушений.')]
    pw = CW // 3
    line(s, X0 + I(0.23), Y0 + I(0.23), X0 + 2 * pw + I(0.23), Y0 + I(0.23), C['pink'], 1.25)
    for i, (w_, t_, b_) in enumerate(phases):
        x0 = X0 + i * pw
        number_badge(s, x0, Y0, i + 1)
        label(s, x0, Y0 + I(0.6), pw - I(0.3), w_)
        text(s, x0, Y0 + I(0.88), pw - I(0.25), I(0.35), t_, size=15, bold=True, color=C['purple'])
        text(s, x0, Y0 + I(1.25), pw - I(0.3), I(0.75), b_, size=11, color=C['ink2'], spacing=1.1)
    for i, (k, t_, b_) in enumerate([('качество', 'Согласие с экспертом', 'Чувствительность и специфичность на '
                                      'слепой выборке врача, раз в месяц.'),
                                     ('пациент', 'Повторные исследования', 'Доля повторов из-за укладки по '
                                      'журналу МО: до и после.'),
                                     ('врач', 'Время на контроль', 'Хронометраж на 50 исследованиях; сервис — '
                                      '2 с на исследование.')]):
        card(s, X0 + i * (pw + I(0.02)), Y0 + I(2.1), pw - I(0.12), I(1.5), t_, b_, kicker='метрика пилота · ' + k,
             dark=False, title_size=14, body_size=11.5)
    fl = [('денситометр', 'DICOM'), ('PACS', 'DICOM-роутер'), ('сервис', 'контроль качества'),
          ('назад в PACS', 'DICOM SR'), ('лаборант', 'веб-интерфейс')]
    fw5 = (CW - 4 * I(0.28)) // 5
    for i, (k, t_) in enumerate(fl):
        x0 = X0 + i * (fw5 + I(0.28))
        acc = i == 2
        rect(s, x0, Y0 + I(3.8), fw5, I(0.9), fill=C['pink'] if acc else C['white'],
             line=None if acc else C['line'], radius=0.12)
        label(s, x0 + I(0.18), Y0 + I(3.92), fw5 - I(0.3), k, color=C['white'] if acc else C['pink'], size=9)
        text(s, x0 + I(0.18), Y0 + I(4.2), fw5 - I(0.3), I(0.35), t_, size=13, bold=True,
             color=C['white'] if acc else C['purple'])
        if i:
            line(s, x0 - I(0.25), Y0 + I(4.25), x0 - I(0.03), Y0 + I(4.25), C['purple'], 1.25, arrow=True)

    # 24. Планы -------------------------------------------------------------------------------------------------
    s = new('roadmap', 'dark', 'Планы развития', 'Что сделаем дальше')
    plans = [('Пилот', 'Теневой режим в 1–2 медорганизациях; отметки операторов — в перекалибровку раз в месяц.'),
             ('Больше нарушений', 'Разметка ротации и оси: вложенный F1 ротации 0.55 при ROC-AUC 0.82 — отсечке '
              'не хватает примеров.'),
             ('Модель точек', 'Дообучить фолды 2–4; согласованная расшифровка позвоночника уже снизила ошибки '
              'уровня. Дальше — Th12 для верхней границы укладки.'),
             ('Второй аппарат', 'Hologic и другие Lunar: проверка масштаба, дообучение сети тем же скриптом.'),
             ('Интеграция', 'DICOM-роутер и SR в PACS, мониторинг доли нарушений по аппаратам и сменам.')]
    for i, (t_, b_) in enumerate(plans):
        x = X0 + (i % 3) * I(4.23)
        y = Y0 + (i // 3) * I(2.45)
        card(s, x, y, I(4.1), I(2.3), t_, b_, kicker=f'шаг {i + 1}', title_size=15, body_size=12)
    card(s, X0 + 2 * I(4.23), Y0 + I(2.45), I(4.1), I(2.3), 'Уже готово', [
        'Контейнер, API, веб-приложение', 'Установщик для Windows', 'Telegram-бот, документация'],
        kicker='сейчас', title_size=15, body_size=12, accent=True)

    # 25. Демонстрация ------------------------------------------------------------------------------------------
    s = new('demo', 'light', 'Демонстрация', 'Сценарий демонстрации, 3 минуты')
    steps = ['./build.sh && ./run.sh → http://localhost:8000 (или приложение)',
             'Перетащить demo.zip: 7 исследований, все типы нарушений',
             'Фильтр «Ротация»: тепловая карта, свод, выступ в мм²',
             'Ось с порогом; отступы: длина поля и линии рисунка 6',
             'Отметить «согласен» (A) и «не согласен», выгрузить отметки',
             'Скачать xlsx (8 колонок ТЗ) и DICOM SR',
             'С телефона: Telegram-бот, команда /demo']
    for i, st_ in enumerate(steps):
        yy = Y0 - I(0.05) + i * I(0.56)
        number_badge(s, X0, yy, i + 1, size=I(0.42))
        text(s, X0 + I(0.6), yy + I(0.07), I(6.3), I(0.4), st_, size=13.5)
    card(s, I(7.6), Y0, I(5.33), I(1.5), 'Запасной вариант',
         'Нет Docker — приложение для Windows; нет сети — слайды «Кейсы» и «Интерфейсы»; сценарий — docs/DEMO.md.',
         dark=False, title_size=14, body_size=12)
    card(s, I(7.6), Y0 + I(1.65), I(5.33), I(1.6), 'Репозиторий', [REPO, 'README · REVIEW · инструкции · DEMO'],
         dark=False, title_size=14, body_size=12)
    strip = [('Контейнер', 'build.sh / run.sh'), ('Отчёт ТЗ 2.5', 'xlsx/csv, 8 колонок'),
             ('ТЗ 2.6', 'оверлеи, SR, интерфейс'), ('Интерфейсы', 'веб, Windows, бот, API'),
             ('Автотесты', f'сервис — {D.get("tests", 64)}, бот — {D.get("tests_bot", 27)}')]
    sw_ = CW // 5
    line(s, X0, I(6.05), XR, I(6.05), C['purple'], 1.5)
    for i, (t_, b_) in enumerate(strip):
        text(s, X0 + i * sw_, I(6.15), sw_ - I(0.15), I(0.3), t_, size=12.5, bold=True, color=C['purple'])
        text(s, X0 + i * sw_, I(6.45), sw_ - I(0.15), I(0.3), b_, size=11, color=C['ink2'])

    # 26. Спасибо -------------------------------------------------------------------------------------------------
    s = deck.closing()
    deck.notes(s, notes['thanks'])
    text(s, I(0.6), I(1.3), I(6.5), I(1.2), 'Спасибо!', size=54, bold=True, color=C['white'])
    text(s, I(0.6), I(2.5), I(6.3), I(0.5), 'Готовы ответить на вопросы', size=20, color=C['white'])
    rows = [('Репозиторий', REPO), ('Установщик для Windows', 'release/DXA-QC-Setup-1.0.0.exe в репозитории'),
            ('Telegram-бот', BOT), ('Для проверяющих', 'docs/REVIEW.md — запуск за 5 минут')]
    for i, (k, v) in enumerate(rows):
        y = I(3.45) + i * I(0.72)
        label(s, I(0.6), y, I(5), k, color=C['pink_soft'], size=10)
        text(s, I(0.6), y + I(0.26), I(6.8), I(0.35), v, size=15, bold=True, color=C['white'])
    qr = qr_png()
    if qr:
        rect(s, I(0.6) + I(6.55), I(3.45), I(1.75), I(1.75), fill=C['white'], radius=0.08)
        picture(s, qr, I(0.6) + I(6.65), I(3.55), w=I(1.55))
        text(s, I(7.15), I(5.3), I(1.75), I(0.3), 'репозиторий', size=10, color=C['white'], align=PP_ALIGN.CENTER)
    return deck


def phone_image() -> Path | None:
    """Верх отрисовки ответа бота — в пропорциях телефона."""
    src = MANUAL_ASSETS / 'bot_result.png'
    if not src.exists():
        return None
    out = ASSETS / 'bot_phone.png'
    with Image.open(src) as im:
        w, h = im.size
        im.crop((0, 0, w, min(h, int(w * 2.0)))).save(out)
    return out


def qr_png() -> Path | None:
    out = ASSETS / 'qr_repo.png'
    if out.exists():
        return out
    try:
        import segno
    except ImportError:
        return None
    segno.make(f'https://{REPO}', error='m').save(str(out), scale=12, border=1, dark='#2D1451')
    return out


# --------------------------------------------------------------------------- #
def export_with_powerpoint(pptx: Path, pdf: Path, png_dir: Path) -> bool:
    """Через PowerPoint: пересохранить PPTX со встроенным Montserrat, затем PDF и
    PNG слайдов. Без PowerPoint — False (остаётся PPTX от python-pptx)."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return False
    pythoncom.CoInitialize()
    app = None
    try:
        app = win32com.client.DispatchEx('PowerPoint.Application')
        tmp = pptx.with_name(pptx.stem + '.embed.pptx')
        pres = app.Presentations.Open(str(pptx), ReadOnly=False, Untitled=False, WithWindow=False)
        # 24 = ppSaveAsOpenXMLPresentation; третий аргумент — встроить шрифты (Montserrat, OFL)
        pres.SaveAs(str(tmp), 24, True)
        pres.SaveAs(str(pdf), 32)                            # ppSaveAsPDF
        png_dir.mkdir(parents=True, exist_ok=True)
        for old in png_dir.glob('*.png'):
            old.unlink()
        for i in range(1, pres.Slides.Count + 1):
            pres.Slides(i).Export(str(png_dir / f'{i:02d}.png'), 'PNG', 1920, 1080)
        pres.Close()
        tmp.replace(pptx)
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
    deck = build_deck(D, cases)
    out = HERE / f'{NAME}.pptx'
    deck.save(out)
    print(out, len(deck.prs.slides), 'слайдов')
    if export_with_powerpoint(out, HERE / f'{NAME}.pdf', ASSETS / 'pptx_png'):
        print('PDF и PNG — через PowerPoint, Montserrat встроен в PPTX')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
