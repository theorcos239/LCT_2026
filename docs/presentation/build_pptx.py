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
        f'миллиметрах и градусах. Работает локально, без видеокарты: '
        f'{D["bench"]["per_study_s"]["median"]:.1f} с на исследование.']],
        size=13)
    set_paras(body2, [[
        'Вердикт проверяется глазами: измерение на снимке сравнивается с нормой из ТЗ. Нейросеть '
        'используется только там, где признак нельзя измерить линейкой, — это форма, а не размер. Отступы '
        'области измерения (ROI) оцениваются так, как их оценивает эксперт.']], size=13)
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
    set_paras(b43, [['Ошибка укладки (положения пациента при съёмке) незаметно портит результаты измерения '
                     'плотности кости и оценку необходимости лечения, а замечает её сегодня только врач, '
                     'вручную. Задачу можно решить прозрачно: каждое замечание — это численный показатель, '
                     'который сравнивается с нормой, и врач проверяет его глазами.']])
    b40 = shape_by_id(s, 40)
    b40.width, b40.height = I(11.2), I(1.25)
    set_paras(b40, [
        ['Примеров нарушений в данных мало (от 6 до 36 на каждый тип). Поэтому качество считаем честно: '
         'каждый снимок проверяет модель, которая его не видела, а нейросеть используется только там, где '
         'измерения недостаточно.'],
        ['Разметка эксперта расходилась с ТЗ по отступам области измерения (ROI) — разобрали все '
         'расхождения и поняли, по какому правилу судит эксперт: по длине поля сканирования. Размер пикселя '
         'в файлах не указан — масштаб восстановили по анатомии.']])
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
                'Вход — снимки в медицинском формате DICOM (папка или zip), как их выгружает архив клиники '
                'PACS; повторные снимки отсеиваются по совпадению изображения.',
                'Область снимка — позвоночник, левое или правое бедро, «не денситометрия» — определяется '
                'верно на 252 снимках из 252.',
                'Три из пяти критериев проверяются прямым измерением в мм и градусах. Для ротации бедра и '
                'посторонних предметов к измерению добавлена нейросеть; вердикт объединяет обе оценки.',
                'Выход — таблица в формате ТЗ (п. 2.5), снимки с разметкой и подсветкой зон, куда смотрела '
                'нейросеть, заключение DICOM SR для PACS.',
                f'Оценка на снимках, которых модель не видела: F1 {f2(ov["f1"])} (баланс найденных '
                f'нарушений и ложных тревог), ROC-AUC {f2(ov["roc_auc"])} (1.0 — идеал). '
                f'{ps["median"]:.1f} с на исследование, без интернета.']),
            ('Маркетинговая суть решения', [
                'Для кого: лаборанты и врачи отделений денситометрии, служба контроля качества лучевой '
                'диагностики.',
                'Ценность: предупреждение приходит, пока пациент на аппарате, — меньше повторных '
                'исследований и ошибок в измерении плотности кости.',
                'Внедрение: снимок с аппарата автоматически уходит в сервис через DICOM-роутер, '
                'заключение возвращается в архив клиники. Сначала параллельно с врачом, затем как помощник '
                'лаборанта, затем — в других медорганизациях.',
                'Поставка: контейнер для сервера, полностью локальное приложение для Windows, программный '
                'интерфейс API. Telegram-бот — демонстрация альтернативного интерфейса. Видеокарта не нужна.']))):
        x = X0 + j * (cw2 + I(0.48))
        rect(s, x, cy, cw2, chh, fill=C['white'], radius=0.035)
        text(s, x + I(0.3), cy + I(0.3), cw2 - I(0.6), I(0.5), ttl_, size=20, bold=True, color=C['purple'])
        bullets(s, x + I(0.3), cy + I(0.9), cw2 - I(0.6), chh - I(1.05), items, size=12, gap=6)

    # 6. Клиническая задача ---------------------------------------------------------------
    s = new('task', 'city', 'Клиническая задача', 'Ошибку укладки дешевле поймать, пока пациент на аппарате')
    label(s, X0, Y0, I(5.8), 'почему это важно')
    bullets(s, X0, Y0 + I(0.35), I(6.1), I(2.1), [
        'Плотность кости считается внутри области измерения. Если позвоночник наклонён, бедро повёрнуто '
        'или в кадре посторонний предмет, аппарат измеряет не тот участок кости.',
        'Ошибка уходит в T-критерий (число, по которому ставят остеопороз) и в ложное изменение по '
        'сравнению с прошлым снимком: лишнее облучение или неверная оценка лечения.',
        'Сегодня врач проверяет качество на глаз, помня критерии для каждой области.'], size=12)
    picture_fit(s, case('ok_hip'), X0, I(4.5), I(3.0), I(2.35), caption='норма: малый вертел виден',
                cap_h=I(0.36), cap_size=10.5)
    picture_fit(s, case('rotation'), X0 + I(3.12), I(4.5), I(3.0), I(2.35),
                caption='бедро повёрнуто: вертел не виден', cap_h=I(0.36), cap_size=10.5)
    label(s, I(6.95), Y0, I(5.5), 'что делает сервис')
    for i, (t_, b_) in enumerate([
            ('Проверяет', 'Каждый снимок по пяти критериям ТЗ: укладка, наклон оси, посторонние предметы, '
                          'поворот (ротация) бедра, отступы области измерения.'),
            ('Объясняет', 'Пишет причину в цифрах — «наклон 8.7° при норме до 5°», «поле 12.5 см, нужно '
                          '12.9 см» — и подсвечивает место на снимке.'),
            ('Передаёт человеку', 'Лаборант исправляет укладку, пока пациент на аппарате; врач видит '
                                  'заключение DICOM SR рядом со снимком в архиве (PACS).')]):
        y = Y0 + I(0.4) + i * I(1.5)
        number_badge(s, I(6.95), y, i + 1)
        text(s, I(7.6), y - I(0.02), I(5.3), I(0.35), t_, size=16, bold=True, color=C['purple'])
        text(s, I(7.6), y + I(0.38), I(5.3), I(0.95), b_, size=12.5, color=C['ink2'], spacing=1.12)

    # 7. Таксономия --------------------------------------------------------------------------
    s = new('taxonomy', 'light', 'Таксономия нарушений', 'Пять критериев ТЗ, у каждого — измерение в физических единицах')
    rows = [['код', 'критерий ТЗ 2.3', 'как измеряем', 'норма'],
            [[('spine_position', {'bold': True})], 'видны верхние края подвздошных костей (таза)',
             'кость видна в нижних углах снимка — гребни подвздошных костей', 'есть с обеих сторон'],
            [[('spine_axis', {'bold': True})], 'наклон оси до 5°',
             'линия через центры позвонков; контроль — по краям позвоночника', '≤ 5° (порог 4°)'],
            [[('spine_artifacts', {'bold': True})], 'нет посторонних предметов',
             'яркие тонкие структуры вне кости (фильтр top-hat) + нейросеть', f'уверенность < {f2(art["threshold"])}'],
            [[('hip_rotation', {'bold': True})], 'нет ротации (по малому вертелу)',
             'выступ малого вертела — костного бугорка — над контуром кости, мм² + нейросеть',
             f'уверенность < {f2(rot["threshold"])}'],
            [[('hip_roi', {'bold': True})], 'отступы 3 / 3 / 2 см',
             'длина поля: 3 см + от вертела до седалищной кости + 3 см', '≥ 12.9 см'],
            [[('undetermined', {'bold': True})], 'область не определена', 'снимок не похож на денситометрию',
             'ручной разбор']]
    table(s, X0, Y0, I(8.55), rows, [1.85, 2.6, 3.4, 1.7], size=11, row_h=I(0.62),
          align=['l', 'l', 'l', 'l'])
    card(s, I(9.2), Y0, I(3.73), I(2.35), 'Несколько нарушений',
         'У каждого пятого исследования нарушений несколько. Критерии проверяются независимо, в отчёт '
         'попадают все, например «spine_axis; spine_artifacts». Класс качества = 1 («есть нарушение»), '
         'если сработал хоть один.', dark=False, body_size=12)
    card(s, I(9.2), Y0 + I(2.5), I(3.73), I(1.7), 'Сбой чтения',
         'Файл, который не удалось прочитать, попадает в отчёт строкой Failure с текстом ошибки; '
         'обработка не прерывается.', dark=False, body_size=12)

    # 8. Данные -------------------------------------------------------------------------------
    s = new('data', 'dark', 'Данные', f'{D["frames"]} снимка, по каждому критерию всего 6–36 нарушений')
    rect(s, X0, Y0, I(6.1), I(4.8), fill=C['white'], radius=0.04)
    for i, (v, cap) in enumerate((('100', 'исследований'), ('499', 'DICOM-файлов'),
                                  (str(D['frames']), 'уникальных снимков'))):
        stat(s, X0 + I(0.3) + i * I(1.9), Y0 + I(0.25), I(1.8), v, cap, size=32)
    reg = D['regions']
    bullets(s, X0 + I(0.3), Y0 + I(1.45), I(5.5), I(3.3), [
        'Один снимок в выгрузке встречается по нескольку раз под разными идентификаторами. Повторы '
        'находим по совпадению изображения — ещё до деления данных на обучение и проверку.',
        f'Из них {reg.get("spine", 0)} снимков позвоночника, {reg.get("lh", 0)} левого и {reg.get("rh", 0)} '
        'правого бедра — до трёх снимков на исследование. У трёх снимков бедра нет полной оценки эксперта, '
        'поэтому метрики считаются на 249.',
        'Все снимки с одного аппарата (GE Lunar Prodigy Advance). Размер пикселя в файлах не указан: масштаб '
        '0.6 мм на пиксель восстановили по известной высоте «позвонок + диск».',
        'Сделали собственную разметку: 242 снимка, по 29 опорных точек на каждом.'], size=11.5)
    rect(s, I(6.7), Y0, I(6.23), I(4.8), fill=C['white'], radius=0.04)
    text(s, I(7.0), Y0 + I(0.25), I(5.7), I(0.4), 'Нарушений мало, и они разные', size=16, bold=True,
         color=C['purple'])
    items = [('position', 'Укладка'), ('axis', 'Наклон оси > 5°'), ('artifacts', 'Посторонние предметы'),
             ('rotation', 'Ротация бедра'), ('roi', 'Отступы поля')]
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
    for i, (v, cap) in enumerate((('48', 'исследований из 100\nс нарушением'), ('20', 'с двумя и более\nнарушениями'),
                                  ('72', 'снимка из 249 оценённых\nс нарушением'))):
        stat(s, I(7.0) + i * I(1.95), Y0 + I(3.3), I(1.9), v, cap, size=28, color=C['purple'])

    # 9. Разбиение ------------------------------------------------------------------------------
    s = new('folds', 'light', 'Как мы проверяем себя',
            'Каждый снимок оценивает модель, которая этот снимок не видела')
    cw_, gap = I(0.98), I(0.08)
    for i in range(5):
        text(s, X0 + I(0.4) + i * (cw_ + gap), Y0, cw_, I(0.25), f'часть {i + 1}', size=10, bold=True,
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
    text(s, X0, Y0 + I(2.8), I(5.8), I(1.7),
         'Модель здесь — это пороги нормы для измерений и нейросеть (о них на следующем слайде). Сто '
         'исследований делим на пять частей по двадцать. Пять раз по очереди четыре части идут на обучение, '
         'пятая — на проверку: в итоге каждый снимок оценён моделью, которая его не видела. При делении '
         'каждое исследование отнесено к самому редкому из своих нарушений, и эти группы разложены по частям '
         f'поровну: без нарушений {st.get("norm", 0)}, ротация {st.get("rotation", 0)}, посторонние предметы '
         f'{st.get("artifacts", 0)}, ось {st.get("axis", 0)}, укладка {st.get("spine_position", 0)}, отступы '
         f'{st.get("roi", 0)}.', size=11, color=C['ink2'], spacing=1.12)
    for i, (t_, b_) in enumerate([
            ('Пациент целиком в одной части', 'Позвоночник и оба бедра одного человека всегда попадают в '
             'одну часть — иначе модель «узнавала» бы пациента, и оценка была бы завышена.'),
            ('Редкие нарушения — в каждой части', 'Деление сделано так, чтобы в каждой из пяти частей были '
             'примеры даже самых редких нарушений.'),
            ('Настройка — только на обучающих частях', 'Пороги, границы нормы, параметры нейросети и '
             'правило объединения нейросети с измерением подбираются на четырёх частях, проверяются на пятой.'),
            ('Мало нарушений — широкие интервалы', 'На каждый тип нарушения всего 6–36 примеров, поэтому к '
             'каждому показателю приводим 95% доверительный интервал — диапазон, где он лежит при такой выборке.')]):
        card(s, I(6.7), Y0 + i * I(1.2), I(6.23), I(1.1), t_, b_, dark=False, title_size=13.5, body_size=10.5)

    # 10. Подход ----------------------------------------------------------------------------------
    s = new('approach', 'dark', 'Подход', 'Измерения там, где критерий — размер; нейросеть — там, где форма')
    seeds = rm.get('cnn_oof_by_seed', {})
    one = np.mean([v['roc_auc'] for v in seeds.values()]) if seeds else None
    rect(s, X0, Y0, I(7.7), I(4.55), fill=C['white'], radius=0.035)
    text(s, X0 + I(0.3), Y0 + I(0.22), I(7.1), I(0.35), 'Что пробовали для ротации бедра', size=15, bold=True,
         color=C['purple'])
    table(s, X0 + I(0.3), Y0 + I(0.65), I(7.1), [
        ['что пробовали', 'ROC-AUC', 'вывод'],
        ['измерение выступа малого вертела по контуру кости', f2(rm['geometry']['roc_auc']),
         'норма и нарушение отличаются на ≈ 2 мм'],
        ['простая модель по всему снимку целиком', '0.50', 'не лучше монетки'],
        ['опорные точки, найденные нейросетью', '—', 'ошибка до 2.2 мм при разнице классов 2.1 мм'],
        ['одна нейросеть ResNet-34', f2(one), 'результат плавает от запуска к запуску'],
        ['три копии той же нейросети, ответы усреднены', f2(rm['cnn_oof']['roc_auc']), 'стабильнее'],
        [[('нейросеть + измерение выступа → один вердикт', {'bold': True})],
         [(f2(rm['stack_oof']['roc_auc']), {'bold': True})], [('выбрано', {'bold': True})]]],
        [3.5, 0.9, 2.7], size=11.5, row_h=I(0.47), align=['l', 'r', 'l'], highlight=(6,))
    text(s, X0 + I(0.3), Y0 + I(4.08), I(7.1), I(0.4),
         'ROC-AUC — насколько хорошо оценка отделяет нарушения от нормы: 0.5 — наугад, 1.0 — идеально.',
         size=10.5, color=C['ink3'], spacing=1.05)
    for i, (t_, b_) in enumerate([
            ('Принцип', 'Вердикт можно проверить глазами: «наклон 8.7°», «поле 12.5 см». Основа — '
                        'измерения в миллиметрах и градусах.'),
            ('Где измерения недостаточно', 'Ротация и посторонние предметы — это форма, а не размер. Здесь '
             'к измерению добавляем нейросеть и объединяем обе оценки; подсветка показывает, куда она '
             'смотрела.'),
            ('Где нейросеть не нужна', 'Наклон оси, укладка, отступы — прямые измерения. Нейросеть для '
                                       'оси проверили: ROC-AUC 0.51 — не лучше монетки.')]):
        card(s, I(8.3), Y0 + i * I(1.55), I(4.63), I(1.43), t_, b_, title_size=13.5, body_size=10.5)

    # 11. Архитектура -------------------------------------------------------------------------------
    s = new('architecture', 'light', 'Архитектура', 'Один контейнер, пять шагов, ни одного обращения в интернет')
    nodes = [('DICOM, zip', 'снимки из архива клиники как есть'), ('чтение', 'распаковка, разворот, дубликаты'),
             ('область', 'позвоночник, бёдра, «не DXA»'), ('критерии', 'измерения + нейросеть, каждый отдельно'),
             ('вердикт и отчёт', 'класс, вероятности, заключение')]
    step = CW // 5
    line(s, X0 + I(0.23), Y0 + I(0.23), X0 + 4 * step + I(0.23), Y0 + I(0.23), C['pink'], 1.25)
    for i, (t_, b_) in enumerate(nodes):
        x = X0 + i * step
        number_badge(s, x, Y0, i + 1)
        text(s, x, Y0 + I(0.6), step - I(0.2), I(0.3), t_, size=14, bold=True, color=C['purple'])
        text(s, x, Y0 + I(0.95), step - I(0.25), I(0.5), b_, size=11.5, color=C['ink2'])
    for i, (t_, b_) in enumerate([
            ('Результат — ровно по ТЗ (п. 2.5, 2.7)', 'Таблица xlsx или csv (строка на снимок, восемь '
                                                       'колонок ТЗ), zip со снимками с разметкой, zip с '
                                                       'заключениями DICOM SR.'),
            ('Интерфейсы', 'Веб-приложение, приложение для Windows, программный интерфейс API и консоль; '
                           'Telegram-бот — демонстрация.'),
            ('Контейнер', 'Python 3.11, версии пакетов зафиксированы, нейросеть на процессоре, один поток: '
                          'повторный запуск даёт в точности тот же отчёт.')]):
        card(s, X0 + i * I(4.22), Y0 + I(1.6), I(4.1), I(1.42), t_, b_, dark=False, title_size=13.5, body_size=11)
    from service.pipeline import COLUMNS as COLS8

    def cut(v, k=13):
        v = '' if v is None else str(v)
        return v if len(v) <= k else '…' + v[-k:]
    sample = [COLS8] + [[cut(r.get(c)) if c in ('path_to_study', 'study_uid', 'image_uid')
                         else ('' if r.get(c) is None else str(r.get(c))) for c in COLS8]
                        for r in D['sample'][:4]]
    label(s, X0, Y0 + I(3.17), CW, 'фрагмент настоящего отчёта: восемь колонок ТЗ 2.5', color=C['purple'])
    table(s, X0, Y0 + I(3.44), CW, sample, [1.6, 1.6, 1.6, 1.55, 1.2, 1.35, 1.55, 1.55], size=10,
          header_size=8.5, row_h=I(0.28), align=['l'] * 8)
    text(s, X0, Y0 + I(4.9), CW, I(0.22), 'lh / rh — левое и правое бедро, spine — позвоночник; класс 0 — '
         'качественный, 1 — есть нарушение', size=9, color=C['ink3'])

    # 12. Модели ----------------------------------------------------------------------------------------
    s = new('models', 'dark', 'Модели', 'Внутри шагов 3–4 конвейера — шесть модулей, у каждого своя задача')
    cards = [
        ('region_clf', 'Область снимка', 'Простая модель по яркости и форме снимка; снимок, не похожий на '
         'денситометрию, уходит на ручной разбор.', '252 / 252 снимка', False),
        ('spine_qc', 'Позвоночник', 'Выделяем кость и центры позвонков. Наклон оси — прямая через центры; '
         'укладка — кость в зонах гребней таза; предметы — фильтр ярких тонких структур (top-hat).',
         'ROC-AUC: ось 0.82 · укладка 0.82', False),
        ('hip_roi', 'Отступы бедра', 'Проверяем, что поле сканирования не короче нормы: 3 см + расстояние '
         'от вертела до седалищной кости + 3 см; отступы из ТЗ (рис. 6) — для сведения.',
         f'F1 {f2(rs.get("f1"))} · ROC-AUC {f2(rs.get("roc_auc"))}', False),
        ('hip_rotation', 'Ротация: измерение', 'Площадь выступа малого вертела над контуром кости, мм²; '
         'выход за коридор нормы означает избыточный или недостаточный поворот.', 'норма 96–272 мм²', False),
        ('cnn_qc', 'Ротация и предметы: нейросеть', 'Готовая нейросеть ResNet-34, дообученная на наших '
         'снимках; три копии, ответ усредняется; без видеокарты; подсвечивает, куда смотрела.',
         'ROC-AUC с измерением 0.82 · 0.85', True),
        ('dxa_qc · выключен по умолчанию', 'Опорные точки', 'Нейросеть находит 29 опорных точек; обучена '
         'на 2 частях данных из 5. Ось — не хуже контурного метода, но в 12 раз медленнее; включается '
         'настройкой.', 'точки: ошибка 0.9–2.4 мм · ось: 0.5°', False)]
    cw3, ch2 = I(4.08), I(2.3)
    for i, (k, t_, b_, m_, acc) in enumerate(cards):
        card(s, X0 + (i % 3) * (cw3 + I(0.155)), Y0 + (i // 3) * (ch2 + I(0.17)), cw3, ch2, t_, b_, kicker=k,
             metric=m_, accent=acc, title_size=14.5, body_size=10.5)

    # 13. Предобработка -----------------------------------------------------------------------------------
    s = new('prepost', 'city', 'Предобработка и постобработка',
            'До модели всё приводится к миллиметрам, после — к вероятности')
    for j, (lab, items) in enumerate((
            ('до проверки критериев', [
                'Читаем сжатые файлы DICOM, инвертированные снимки и таблицы яркости.',
                'Зеркально выгруженный снимок разворачиваем к стандартному виду по служебной метке в файле.',
                'Одинаковые снимки под разными именами схлопываем в один — по совпадению изображения.',
                'Правое бедро отражаем в левое: все измерения и нейросеть работают с одной стороной.',
                'Выделяем кость и переводим пиксели в миллиметры: пороги заданы в мм, градусах и мм².',
                'Для нейросети выравниваем яркость и уменьшаем снимок до 320 точек по длинной стороне.']),
            ('после проверки', [
                'Пороги и границы нормы подобраны на обучающих частях данных, а не на проверяемых снимках.',
                'Отступы области измерения оцениваем по длине поля сканирования — так же, как врач-эксперт.',
                'По ротации бедра и посторонним предметам ответ дают и измерение, и нейросеть — их оценки '
                'объединяются в один вердикт.',
                'Каждое измерение переводится в вероятность нарушения; общая вероятность — что нарушен '
                'хотя бы один критерий.',
                'Если измерение и нейросеть разошлись, вероятность остаётся около порога, а снимок получает '
                'пометку «проверьте глазами».',
                'На выходе — заключение по-русски, снимок с разметкой и подсветкой, DICOM SR для архива.']))):
        x = X0 + j * I(6.3)
        label(s, x, Y0, I(5.9), lab)
        bullets(s, x, Y0 + I(0.4), I(5.9), I(4.5), items, size=12.5, gap=8)

    # 14. Метрики ---------------------------------------------------------------------------------------------
    s = new('metrics', 'light', 'Метрики качества', 'Каждый снимок проверен моделью, которая его не видела')
    names = {'spine_position': 'Укладка', 'spine_axis': 'Наклон оси', 'spine_artifacts': 'Посторонние предметы',
             'hip_rotation': 'Ротация бедра', 'hip_roi': 'Отступы поля'}
    rowsm = [(names[k], hv[k], h0v.get(k)) for k in names] + \
        [('Позвоночник (любое)', h['per_region']['spine'], hb0['per_region']['spine']),
         ('Бедро (любое)', h['per_region']['hip'], hb0['per_region']['hip']),
         ('Все снимки (любое)', ov, ov0)]
    cx = dict(name=X0, pos=I(2.95), sens=I(3.95), spec=I(4.75), f1bar=I(5.85), f1=I(8.4), aucbar=I(9.35),
              auc=I(12.05))
    yy = Y0 - I(0.1)
    for key, lab in (('name', 'критерий'), ('pos', 'нар./всего'), ('sens', 'найдено'), ('spec', 'без ложных'),
                     ('f1bar', 'F1, 95% интервал'), ('f1', 'F1'), ('aucbar', 'ROC-AUC, 95% интервал (0.4–1)'),
                     ('auc', 'AUC')):
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
    text(s, X0 + I(0.38), ly, I(4.3), I(0.25), 'было: только измерения, отступы по рисунку 6',
         size=10.5, color=C['ink2'])
    rect(s, I(5.0), ly + I(0.07), I(0.3), I(0.1), fill=C['pink'])
    text(s, I(5.38), ly, I(3.7), I(0.25), 'стало: + нейросеть, отступы как у эксперта', size=10.5,
         color=C['ink2'])
    text(s, I(9.1), ly, I(3.83), I(0.25), f"macro-F1 {f2(hb0['overall']['macro_f1'])} → "
         f"{f2(h['overall']['macro_f1'])} · область 252/252", size=10.5, bold=True, color=C['purple'],
         align=PP_ALIGN.RIGHT)
    text(s, X0, ly + I(0.34), CW, I(0.25), '«найдено» — доля нарушений, которые сервис нашёл (чувствительность); '
         '«без ложных» — доля нормальных снимков, которые он не тревожил зря (специфичность); F1 — их баланс.',
         size=9.5, color=C['ink3'])

    # 15. Эксперименты --------------------------------------------------------------------------------------
    s = new('experiments', 'light', 'Эксперименты',
            'Нейросеть плюс измерение точнее каждого по отдельности: чем выше и левее кривая, тем лучше')
    ro, ao = D['rot_oof'], D['art_oof']
    for j, (ttl_, dd, m, geo_lab) in enumerate((
            (f'Ротация бедра · {rot["positives"]} из {rot["n"]}', ro, rm, 'выступ'),
            (f'Посторонние предметы · {art["positives"]} из {art["n"]}', ao, am, 'top-hat'))):
        x0 = X0 + j * I(6.35)
        text(s, x0, Y0, I(6.0), I(0.35), ttl_, size=15, bold=True, color=C['purple'])
        curves = [(geo_lab, dd.y, dd.geometry.fillna(0).values, C['lav'], 'dash'),
                  ('нейросеть', dd.y, dd.p_cnn.values, C['pink2'], None),
                  ('нейросеть + измерение', dd.y, dd.p_stack.values, C['purple'], None)]
        roc_chart(s, x0 - I(0.1), Y0 + I(0.4), I(3.55), I(3.55), curves)
        lx = x0 + I(3.45)
        text(s, lx, Y0 + I(0.28), I(2.85), I(0.25), 'ROC-AUC (1.0 — идеал):', size=9.5, color=C['ink3'])
        for k, (lab, col, key) in enumerate(((geo_lab, C['lav'], 'geometry'), ('нейросеть', C['pink2'], 'cnn_oof'),
                                              ('нейросеть + измерение', C['purple'], 'stack_oof'))):
            rect(s, lx, Y0 + I(0.65) + k * I(0.34), I(0.28), I(0.1), fill=col)
            text(s, lx + I(0.38), Y0 + I(0.56) + k * I(0.34), I(2.5), I(0.28), f'{lab} {f2(m[key]["roc_auc"])}',
                 size=11, bold=(key == 'stack_oof'))
        text(s, lx, Y0 + I(1.75), I(2.85), I(0.55), 'что реально ставит сервис: правило объединения и порог '
             'настроены на других снимках, не на проверочных', size=9.5, color=C['ink3'], spacing=1.05)
        g, v = D['h0'][('hip_rotation' if j == 0 else 'spine_artifacts')], m['verdict_nested']
        table(s, lx, Y0 + I(2.4), I(2.85), [['', 'нашли', 'без ложных', 'F1'],
                                           [geo_lab, f2(g['sensitivity']), f2(g['specificity']), f2(g['f1'])],
                                           ['нейросеть + измерение', f2(v['sensitivity']), f2(v['specificity']),
                                            [(f2(v['f1']), {'bold': True})]]],
              [1.15, 0.55, 0.65, 0.5], size=9.5, header_size=6.5, row_h=I(0.3))
        text(s, x0, Y0 + I(4.05), I(3.5), I(0.25), 'по горизонтали — доля ложных тревог, по вертикали — доля '
             'найденных нарушений', size=9.5, color=C['ink3'])

    # 16. Сравнение вариантов ----------------------------------------------------------------------------------
    s = new('comparison', 'city', 'Сравнение вариантов', 'Что проверили и что в итоге оставили')
    # Модель точек обучена на фолдах 0–1 из 5; контур и точки — на одних и тех же кадрах этих фолдов
    kp, kc = D.get('kp_oof', {}), D.get('kp_contour', {})

    def axis(m):
        a = m.get('per_violation', {}).get('spine_axis', {})
        return f"{f2(a.get('f1'))} / {f2(a.get('roc_auc'))}"

    def ms(m):
        v = m.get('overall', {}).get('seconds_per_image_median')
        return '—' if v is None else f'{v * 1000:.0f} мс'
    n_ax = kc.get('per_violation', {}).get('spine_axis', {}).get('positives', 0)
    text(s, X0, Y0, I(6.0), I(0.35), 'Два способа найти ось: контур и опорные точки', size=14, bold=True,
         color=C['purple'])
    table(s, X0, Y0 + I(0.42), I(6.0), [
        ['на снимках, которых модель не видела (части 1–2)', 'контур', 'опорные точки'],
        ['ось: F1 / ROC-AUC', axis(kc), axis(kp)],
        ['средний F1 по типам нарушений', f2(kc.get('overall', {}).get('macro_f1'), 3), f2(kp.get('overall', {}).get('macro_f1'), 3)],
        ['время на снимок (без видеокарты)', ms(kc), ms(kp)]], [2.9, 1.4, 1.7], size=11.5, row_h=I(0.42))
    text(s, X0, Y0 + I(2.2), I(6.0), I(0.9), f'На этих снимках всего {n_ax} нарушения оси — сравнение неточное. '
         'Модель опорных точек обучена пока на двух частях из пяти и в 12 раз медленнее, поэтому по умолчанию '
         'выключена.', size=11.5, color=C['ink2'], spacing=1.12)
    sw = D['spine_oof'].get('position', {}).get('rule_sweep', {})
    ru = {'geometry': 'измерение (выбрано)', 'appearance': 'оценка по виду снимка', 'and': 'обе оценки согласны',
          'or': 'любая из двух'}
    text(s, I(6.75), Y0, I(6.18), I(0.35), 'Как объединять две оценки укладки', size=14, bold=True,
         color=C['purple'])
    table(s, I(6.75), Y0 + I(0.42), I(6.18), [['на снимках, которых модель не видела; 6 нарушений', 'найдено',
                                              'лишних', 'F1']] +
          [[ru.get(k, k), str(v['tp']), str(v['fp']), f2(v['f1'])] for k, v in sw.items()],
          [3.0, 1.0, 1.0, 0.9], size=11.5, row_h=I(0.36), highlight=(1,))
    text(s, I(6.75), Y0 + I(2.35), I(6.18), I(0.8), 'На обучающих снимках лучшим выглядело правило «обе оценки '
         'согласны» (F1 0.73), но на новых снимках оно находит лишь одно нарушение из шести — поэтому оставили '
         'измерение.', size=11, color=C['ink2'], spacing=1.1)
    text(s, X0, Y0 + I(3.15), CW, I(0.35), 'Нейросеть для ротации бедра: какие варианты перебрали (ROC-AUC '
         'на снимках, которых модель не видела)', size=14, bold=True, color=C['purple'])
    table(s, X0, Y0 + I(3.57), CW, [
        ['ResNet-18', 'EfficientNet-B0', 'ResNet-34, снимок 256 px', 'ResNet-34, одна нейросеть',
         'переучена с модели точек', 'ResNet-34, три нейросети', '+ измерение выступа'],
        ['0.76', '0.73', '0.77', f2(one), '0.79', f2(rm['cnn_oof']['roc_auc']),
         [(f2(rm['stack_oof']['roc_auc']), {'bold': True, 'color': C['pink']})]]],
        [1] * 7, size=13, header_size=9, row_h=I(0.42), align=['c'] * 7)

    # 17. Эксперт -----------------------------------------------------------------------------------------------
    s = new('expert', 'dark', 'Анализ ошибок', 'Разбор ошибок показал, как оценивает эксперт')
    rect(s, X0, Y0, I(6.25), I(4.75), fill=C['white'], line=C['pink'], line_w=2, radius=0.035)
    label(s, X0 + I(0.3), Y0 + I(0.22), I(5.7), 'отступы области измерения · что делает эксперт')
    text(s, X0 + I(0.3), Y0 + I(0.5), I(5.7), I(0.4), 'Эксперт смотрит на длину поля', size=17,
         bold=True, color=C['purple'])
    table(s, X0 + I(0.3), Y0 + I(1.05), I(5.65), [
        ['правило', 'нашли', 'лишних тревог', 'F1', 'AUC'],
        ['отступы по схеме из ТЗ', f"{rmg.get('tp')}/7", str(rmg.get('fp')), f2(rmg.get('f1')),
         f2(rmg.get('roc_auc'))],
        [[('длина поля ≥ 12.9 см', {'bold': True})], f"{rs.get('tp')}/7", str(rs.get('fp')),
         [(f2(rs.get('f1')), {'bold': True})], f2(rs.get('roc_auc'))]],
        [2.55, 0.85, 1.05, 0.55, 0.65], size=11.5, row_h=I(0.44), highlight=(2,))
    text(s, X0 + I(0.3), Y0 + I(2.5), I(5.65), I(2.1),
         'Эксперт ставит «норму», даже когда под седалищной костью остаётся 1.7–3.0 см вместо трёх, и '
         '«нарушение» на коротких полях. Порог 12.9 см мы не подгоняли под его ответы: 3 см взяты из ТЗ, '
         '6.9 см — типичное расстояние от вертела до седалищной кости. Отступы по схеме из ТЗ сервис тоже '
         'считает и показывает; если эталонная разметка сделана строго по схеме, режим меняется одной '
         'настройкой. F1 и AUC — доля верных ответов, чем ближе к 1, тем лучше.',
         size=11, color=C['ink2'], spacing=1.14)
    ro_ = D['rot_oof']
    geo_bad = ro_.geometry.fillna(0) > 0
    st_bad = ro_.pred_nested.astype(bool)
    dis = geo_bad != st_bad
    right = int((st_bad[dis] == ro_.y[dis].astype(bool)).sum())
    for i, (t_, b_, hh) in enumerate([
            ('Ось: сколиоз — не нарушение', 'При сколиозе эксперт не ставит нарушение оси ни в одном из 14 '
             'случаев, даже при угле до 8.8°. Это главный источник наших лишних срабатываний: по '
             'поясничному отделу сколиоз не отличить от наклона пациента (нейросеть здесь угадывает как '
             'монетка — ROC-AUC 0.51).', I(1.85)),
            ('Ротация: нейросеть и измерение расходятся', f'Разошлись на {int(dis.sum())} снимке из {len(ro_)}, и '
             f'в каждом спорном случае прав ровно один: объединённый вердикт — в {right}, одно измерение — в '
             f'{int(dis.sum()) - right}. Такие снимки помечены «проверьте глазами».', I(1.3)),
            ('Укладка: в данных лишь 6 нарушений', 'ROC-AUC 0.82, но граница «норма / нарушение», подобранная '
             'по одному-двум примерам, неустойчива: F1 на новых снимках 0.32.', I(1.3))]):
        y = Y0 + [0, I(1.98), I(3.43)][i]
        card(s, I(6.85), y, I(6.08), hh, t_, b_, title_size=14, body_size=11.5)

    # 18. Кейсы ------------------------------------------------------------------------------------------------
    s = new('cases', 'dark', 'Кейсы', 'Четыре снимка, где сервис и эксперт сошлись: каждое замечание видно на снимке')
    caps = [('axis', f'ось позвоночника отклонена на {f2(ca.get("angle"), 1)}° при норме до 5°\nпроверка spine_axis'),
            ('artifacts', f'посторонний предмет; нейросеть уверена на {(cart.get("art_p") or 0) * 100:.0f} %\n'
                          'проверка spine_artifacts'),
            ('rotation', f'малый вертел {f2(cr.get("area"), 0)} мм² при норме 96–272 → бедро повёрнуто\n'
                         'проверка hip_rotation'),
            ('roi', f'поле сканирования {f2((croi.get("scan_len") or 0) / 10, 1)} см, нужно не меньше 12.9\n'
                    'проверка hip_roi')]
    fw = (CW - 3 * I(0.17)) // 4
    for i, (k, cap) in enumerate(caps):
        if k in cases:
            picture_fit(s, case(k), X0 + i * (fw + I(0.17)), Y0, fw, I(4.55), caption=cap, cap_h=I(0.7),
                        cap_size=10)
    text(s, X0, Y0 + I(4.65), CW, I(0.3), 'Снимки из обучающего набора, но оценку давали модели, которые именно '
         'этот снимок не видели. Оранжевым — куда смотрела нейросеть, голубым — найденные ориентиры, зелёным и '
         'красным — допустимые границы.', size=10.5, color=C['white'])

    # 19. Скорость ---------------------------------------------------------------------------------------------
    s = new('speed', 'light', 'Скорость и требования', 'Секунды на исследование на обычном процессоре')
    for i, (v, u, cap) in enumerate(((f'{ps["median"]:.1f}', ' с', f'типичное время на исследование\n(максимум {ps["max"]:.1f} с)'),
                                     ('180', ' с', 'лимит по п. 2.7 ТЗ\nна исследование'),
                                     (f'{pf["median"] * 1000:.0f}', ' мс', 'типичное время на снимок,\nодно ядро процессора'),
                                     (f'{D["bench"]["total_s"] / 60:.1f}', ' мин', 'все 100 исследований\nподряд, без параллелизма'))):
        stat(s, X0 + (i % 2) * I(2.95), Y0 + (i // 2) * I(1.45), I(2.8), v, cap, unit=u, size=36,
             color=C['pink'] if i != 1 else C['purple'])
    label(s, X0, Y0 + I(3.05), I(5.8), 'самое долгое исследование и лимит ТЗ', color=C['purple'])
    rect(s, X0, Y0 + I(3.38), I(5.6), I(0.32), fill=C['lilac'], radius=0.3)
    rect(s, X0, Y0 + I(3.38), max(int(I(5.6) * ps['max'] / 180), I(0.12)), I(0.32), fill=C['pink'], radius=0.3)
    text(s, X0 + I(0.25), Y0 + I(3.43), I(5.2), I(0.25), f'{ps["max"]:.1f} с — это {ps["max"] / 180 * 100:.1f} % '
         'от лимита в 180 с (розовая полоска слева)', size=10.5, bold=True, color=C['purple'])
    text(s, X0, Y0 + I(3.9), I(5.8), I(0.8), 'В контейнере (Linux) 100 исследований вместе со снимками с '
         'разметкой и заключениями SR — 158 с. Одни измерения без нейросети — около 27 мс на снимок.',
         size=11.5, color=C['ink2'], spacing=1.12)
    table(s, I(6.8), Y0, I(6.13), [['', 'минимум', 'рекомендуется'],
                                   ['процессор', '2 ядра обычного ПК', '4 ядра'], ['память', '2 ГБ', '4 ГБ'],
                                   ['диск', '2 ГБ (образ 1.4 ГБ)', '5 ГБ'], ['видеокарта', 'не нужна', 'не нужна'],
                                   ['ПО', 'Docker или Windows 10/11', 'Linux-сервер']],
          [1.2, 2.5, 2.4], size=12.5, row_h=I(0.45), align=['l', 'l', 'l'])
    card(s, I(6.8), Y0 + I(2.95), I(6.13), I(1.7), 'Воспроизводимость',
         'Повторный прогон на тех же данных даёт в точности тот же отчёт: фиксированный порядок обработки, '
         'модели на диске, вычисления без параллелизма. Контейнер, приложение для Windows и запуск из кода '
         'дают одинаковый результат на всех 252 снимках.', dark=False, title_size=14, body_size=11.5)

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
            ('Веб-приложение', 'Загружаете папку или zip — получаете снимки с разметкой. Можно установить '
                               'из браузера как приложение.'),
            ('Для Windows', 'Работает полностью локально: снимки не покидают компьютер. Обновляется '
                            'установкой новой версии.'),
            ('Telegram-бот', 'Тот же анализ прямо в чате: показывает, что сервис легко встроить в другие '
                             'системы. Только обезличенные снимки.'),
            ('Интеграция: API и PACS', 'Внешняя система присылает снимки — получает отчёт; заключение '
                                       'записывается в то же исследование в архиве.')]):
        card(s, I(10.2), Y0 + i * I(1.18), I(2.73), I(1.12), t_, b_, dark=False, title_size=11.5, body_size=8.5)

    # 21. Ценность ----------------------------------------------------------------------------------------------
    s = new('value', 'dark', 'Ценность для здравоохранения', 'Качество исследования — в момент съёмки, а не на повторе')
    for i, (k, t_, b_) in enumerate([
            ('лаборант', 'Подсказка у аппарата', f'Через {ps["median"]:.1f} с после съёмки — замечание с '
             'измеренным значением и нормой: укладку исправляют, пока пациент на аппарате.'),
            ('врач', 'Снимок, пригодный для заключения', 'Заключение о качестве сохраняется в архиве рядом со '
             'снимком (DICOM SR); сомнительные снимки помечены «проверьте глазами».'),
            ('медорганизация', 'Меньше повторов, единый стандарт', 'Единые критерии ТЗ для всех аппаратов и '
             'смен; доля нарушений видна в мониторинге.')]):
        card(s, X0 + i * I(4.23), Y0, I(4.1), I(2.35), t_, b_, kicker=k, title_size=15, body_size=12)
    for i, (t_, b_) in enumerate([
            ('Как поставляется', ['Готовый серверный пакет (контейнер) для медорганизации или облака '
                                  'Департамента.',
                                  'Приложение для Windows: работает на обычном ноутбуке в кабинете, '
                                  'полностью локально.',
                                  'Видеокарта и интернет не нужны, снимки не уходят наружу.']),
            ('Как измеряем эффект', ['Доля повторных исследований из-за неверной укладки — до и после '
                                     'внедрения.',
                                     'Раз в месяц сверяем вердикты сервиса с экспертом на случайных снимках.',
                                     'Время, которое врач тратит на контроль качества.'])]):
        card(s, X0 + i * I(6.35), Y0 + I(2.55), I(6.2), I(2.2), t_, b_, title_size=15, body_size=12)

    # 22. Ограничения -------------------------------------------------------------------------------------------
    s = new('limits', 'city', 'Ограничения', 'Что сервис не умеет и почему')
    table(s, X0, Y0, CW, [
        ['ограничение', 'почему', 'что нужно, чтобы снять'],
        [[('Один аппарат', {'bold': True})], 'Все снимки — с одного денситометра GE Lunar Prodigy Advance; '
         'размер пикселя в файлах не указан', 'Снимки второго аппарата: проверить масштаб, дообучить нейросеть'],
        [[('Малая выборка', {'bold': True})], 'От 6 до 36 нарушений на критерий, поэтому доверительные '
         'интервалы широкие', 'Больше размеченных снимков; отметки операторов из пилота'],
        [[('Отступы — как у эксперта', {'bold': True})], 'Отступы проверяем по длине поля сканирования, как '
         'делает эксперт, а не строго по схеме из ТЗ (рис. 6)', 'Если эталонная разметка сделана строго по схеме '
         'из ТЗ, режим меняется одной настройкой'],
        [[('Ось и сколиоз', {'bold': True})], 'Для эксперта сколиоз — не ошибка укладки, а сервис по одному '
         'углу не отличает сколиоз от наклона пациента', 'Размеченные случаи сколиоза или снимки грудного отдела'],
        [[('Разметка оператора', {'bold': True})], 'Линий и областей, которые оператор ставит на аппарате, в '
         'файлах нет — всё оценивается по изображению', 'Выгрузка снимков вместе с разметкой оператора'],
        [[('Верхняя граница укладки', {'bold': True})], 'Позвонок Th12 (12-й грудной) в данных не размечен',
         'Разметить Th12; модель опорных точек уже находит этот уровень'],
        [[('Telegram-бот', {'bold': True})], 'Файлы проходят через серверы Telegram, а ТЗ (п. 3.2) запрещает '
         'передавать снимки наружу', 'Бот — демонстрация на обезличенных данных; сам сервис в медорганизации '
         'работает без интернета']],
        [2.3, 4.4, 4.4], size=11, row_h=I(0.6), align=['l', 'l', 'l'])

    # 23. Пилот -------------------------------------------------------------------------------------------------
    s = new('pilot', 'light', 'Внедрение и пилот',
            'Сначала параллельно с врачом, затем помощник лаборанта, затем другие медорганизации')
    phases = [('месяцы 1–2 · параллельно с врачом', '1–2 медорганизации', 'Копии снимков идут в сервис через '
               'DICOM-роутер, заключение возвращается в архив с пометкой «не проверено». Считаем согласие с '
               'экспертом.'),
              ('месяцы 3–6 · помощник лаборанта', 'Подсказка у аппарата', f'Результат у лаборанта через '
               f'{ps["median"]:.1f} с (максимум {ps["max"]:.1f} с) после снимка. По отметкам лаборантов уточняем '
               'границы «норма / нарушение».'),
              ('месяцы 6–12 · масштабирование', 'Другие аппараты и клиники', 'Подключаем новые аппараты: '
               'проверяем размер пикселя, дообучаем нейросеть на их снимках, следим за долей нарушений.')]
    pw = CW // 3
    line(s, X0 + I(0.23), Y0 + I(0.23), X0 + 2 * pw + I(0.23), Y0 + I(0.23), C['pink'], 1.25)
    for i, (w_, t_, b_) in enumerate(phases):
        x0 = X0 + i * pw
        number_badge(s, x0, Y0, i + 1)
        label(s, x0, Y0 + I(0.6), pw - I(0.3), w_)
        text(s, x0, Y0 + I(0.88), pw - I(0.25), I(0.35), t_, size=15, bold=True, color=C['purple'])
        text(s, x0, Y0 + I(1.25), pw - I(0.3), I(0.8), b_, size=10.5, color=C['ink2'], spacing=1.1)
    for i, (k, t_, b_) in enumerate([('качество', 'Согласие с экспертом', 'Раз в месяц: доля найденных нарушений '
                                      'и доля ложных тревог на снимках, которые врач проверяет вслепую.'),
                                     ('пациент', 'Повторные исследования', 'Доля повторных снимков из-за '
                                      'положения пациента по журналу медорганизации — до и после.'),
                                     ('врач', 'Время на контроль', 'Замер времени на 50 исследованиях; сервису '
                                      f'нужно {ps["median"]:.1f} с на исследование.')]):
        card(s, X0 + i * (pw + I(0.02)), Y0 + I(2.1), pw - I(0.12), I(1.5), t_, b_, kicker='метрика пилота · ' + k,
             dark=False, title_size=14, body_size=11.5)
    fl = [('денситометр', 'снимок DICOM'), ('архив PACS', 'DICOM-роутер'), ('сервис', 'контроль качества'),
          ('назад в архив', 'заключение SR'), ('лаборант', 'веб-интерфейс')]
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
    plans = [('Пилот', 'Работа параллельно с врачом в 1–2 медорганизациях; по отметкам операторов раз в '
              'месяц уточняем границы «норма / нарушение».'),
             ('Больше примеров нарушений', 'Разметка ротации и оси: нейросеть уже отличает ротацию (ROC-AUC '
              '0.82), но порог «да / нет» на 36 примерах пока нестабилен (F1 0.55).'),
             ('Модель опорных точек', 'Дообучить на всех данных (сейчас — 2 части из 5). Дальше — 12-й грудной '
              'позвонок (Th12) для верхней границы области сканирования.'),
             ('Второй аппарат', 'Аппараты Hologic и другие модели Lunar: проверить размер пикселя, дообучить '
              'нейросеть тем же скриптом.'),
             ('Интеграция и обновления', 'Автоприём снимков с аппарата и запись заключения в архив (PACS), '
              'мониторинг доли нарушений; автообновление приложения (сейчас — новым установщиком), '
              'встраивание в другие сервисы клиники.')]
    for i, (t_, b_) in enumerate(plans):
        x = X0 + (i % 3) * I(4.23)
        y = Y0 + (i // 3) * I(2.45)
        card(s, x, y, I(4.1), I(2.3), t_, b_, kicker=f'шаг {i + 1}', title_size=15, body_size=12)
    card(s, X0 + 2 * I(4.23), Y0 + I(2.45), I(4.1), I(2.3), 'Уже готово', [
        'Сервер в Docker, программный интерфейс API, веб-приложение',
        'Установщик для Windows, работает без интернета', 'Telegram-бот для демонстрации; документация'],
        kicker='сейчас', title_size=15, body_size=12, accent=True)

    # 25. Демонстрация ------------------------------------------------------------------------------------------
    s = new('demo', 'light', 'Демонстрация', 'Сценарий демонстрации, 3 минуты')
    steps = ['Запуск одной командой (Docker) → сервис в браузере, или приложение для Windows',
             'Загрузить архив demo.zip: 7 исследований, все типы нарушений',
             'Фильтр «Ротация»: подсветка нейросети, вердикт, выступ вертела в мм²',
             'Наклон оси с допуском; отступы: длина поля и линии схемы из ТЗ',
             'Отметить «согласен» / «не согласен» одной клавишей, выгрузить отметки',
             'Скачать таблицу xlsx (8 колонок ТЗ) и заключения DICOM SR',
             'С телефона: Telegram-бот, команда /demo — тот же сценарий в чате']
    for i, st_ in enumerate(steps):
        yy = Y0 - I(0.05) + i * I(0.56)
        number_badge(s, X0, yy, i + 1, size=I(0.42))
        text(s, X0 + I(0.6), yy + I(0.07), I(6.5), I(0.4), st_, size=13)
    card(s, I(7.6), Y0, I(5.33), I(1.5), 'Запасной вариант',
         'Нет Docker — приложение для Windows; нет интернета — слайды «Кейсы» и «Интерфейсы»; сценарий — docs/DEMO.md.',
         dark=False, title_size=14, body_size=12)
    card(s, I(7.6), Y0 + I(1.65), I(5.33), I(1.6), 'Репозиторий',
         [REPO, 'описание (README), обзор для проверяющих (REVIEW), инструкции, пояснительная записка'],
         dark=False, title_size=14, body_size=12)
    strip = [('Запуск', 'одна команда (Docker)'), ('Отчёт (п. 2.5 ТЗ)', 'xlsx/csv, 8 колонок'),
             ('Визуализация (п. 2.6)', 'снимки с разметкой, SR, интерфейс'), ('Интерфейсы', 'веб, Windows, API, бот'),
             ('Автотесты', f'{D.get("tests", 65) + D.get("tests_bot", 27)}: сервис {D.get("tests", 65)}, '
                           f'бот {D.get("tests_bot", 27)}')]
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
    rows = [('Репозиторий', REPO), ('Установщик для Windows', 'файл DXA-QC-Setup-1.0.0.exe в папке release'),
            ('Telegram-бот (демонстрация)', BOT), ('Для проверяющих', 'docs/REVIEW.md — запуск за 5 минут')]
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
