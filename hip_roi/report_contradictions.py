# -*- coding: utf-8 -*-
"""Архив «экспертная оценка против критерия ТЗ» — по настоящим точкам эксперта.

    python -m hip_roi.report_contradictions

Собирает hip_roi/contradictions/expert_vs_tz.zip:
  - 9 показательных случаев: подписанный снимок (реальные точки эксперта,
    пороговые линии ТЗ, размерные стрелки, вердикт) + рядом ОРИГИНАЛ без
    изменений (тот же кадр, без разметки; имя файла несёт тот же
    идентификатор исследования, что и подписанная версия);
  - cases.csv — сводка по этим 9 случаям;
  - all_real_disagreements.csv — все настоящие расхождения (не только показанные);
  - ПОЯСНЕНИЕ.md — текст рядом с архивом.

В отличие от первой версии этого отчёта (см. git-историю), здесь используются
НЕ оценки алгоритма `hip_roi.kits`, а настоящая ручная разметка ключевых точек
(`hip_roi.ground_truth`, источник — `data/razmetka_aleksandra_*.json`) и
официальный протокол разметки, который называет `ischium_bottom` (а не малый
вертел) точкой отсчёта нижнего поля ТЗ. Подробности — в ПОЯСНЕНИЕ.md.
"""
from __future__ import annotations

import json
import shutil
import sys
import textwrap
import zipfile

import pandas as pd
from PIL import Image, ImageDraw

from .geometry import BOTTOM_MM, LAT_MM, MM_PER_PX, TOP_MM
from .ground_truth import (ANNOTATIONS_JSON, OUT, decode_png, load_html_images,
                           match_images, real_disagreements, true_margins)
from .overlay import _font

ZIP_NAME = 'expert_vs_tz.zip'
NOTE_NAME = 'ПОЯСНЕНИЕ.md'
S = 3

RED = (255, 70, 70)
GREEN = (90, 230, 90)
CYAN = (0, 220, 255)
YELLOW = (255, 230, 0)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)

# (имя, суффикс study, сторона, группа, обоснование)
CASES = [
    ('A1_top_violation', '116822802042655971810859435524105064465', 'lh',
     'A. Эксперт: норма. По точкам: нарушение сверху',
     'Верхушка большого вертела (trochanter_major) в 2.6 см от верхнего края, '
     'требуется 3 см. Остальные поля с запасом.'),
    ('A2_bottom_violation', '173753313169335820519632380691065649420', 'lh',
     'A. Эксперт: норма. По точкам: нарушение снизу',
     'Нижняя точка седалищной кости (ischium_bottom) в 2.2 см от нижнего края, '
     'требуется 3 см. Верх и латераль в норме.'),
    ('A3_bottom_violation', '234651502352368356456298825489565282446', 'lh',
     'A. Эксперт: норма. По точкам: нарушение снизу',
     'Седалищная кость обрезана до 1.9 см от нижнего края при требовании 3 см.'),
    ('A4_bottom_violation', '234651502352368356456298825489565282446', 'rh',
     'A. Эксперт: норма. По точкам: нарушение снизу',
     'То же исследование, правое бедро: 2.4 см снизу — тоже меньше 3 см. '
     'На обоих бёдрах одного пациента нижний край обрезан одинаково.'),
    ('A5_lateral_violation', '17706555930735581384953296958427669848', 'rh',
     'A. Эксперт: норма. По точкам: нарушение латерально',
     'Наружный край большого вертела (trochanter_lateral) в 1.7 см от бокового '
     'края, требуется 2 см. Верх и низ с большим запасом.'),
    ('A6_lateral_violation', '175903805423871105435148927871625232853', 'rh',
     'A. Эксперт: норма. По точкам: нарушение латерально',
     'Наружный край вертела в 1.8 см от бокового края при требовании 2 см.'),
    ('A7_lateral_violation', '181196659018348153828200695057897449628', 'lh',
     'A. Эксперт: норма. По точкам: нарушение латерально',
     'Наружный край вертела в 1.6 см от бокового края — самый тесный случай '
     'в выборке. Требуется 2 см.'),
    ('B1_all_margins_ok', '277752758923974029595977676354128739670', 'lh',
     'B. Эксперт: некорректная область интересов. По точкам: все поля в норме',
     'Верхушка вертела, седалищная кость и наружный край — все в 3.7-5.0 см от '
     'краёв, с большим запасом. Пометка флагом minor_flat относится к малому '
     'вертелу (ротация), не к полям области интереса.'),
    ('B2_all_margins_ok', '13325817932854000792938662637654885530', 'lh',
     'B. Эксперт: некорректная область интересов. По точкам: все поля в норме',
     'Верх 3.5 см, низ 3.3 см, латераль 2.09 см — на грани, но не менее 2 см. '
     'Самый пограничный из всех «B»-случаев: 0.9 мм запаса.'),
]


def render_annotated(img_u8, side, w, h, points, verdict, name, group, why):
    T = points['trochanter_major']
    L = points['trochanter_lateral']
    I = points['ischium_bottom']
    lat_left = side == 'rh'

    PAD, CAP = 24, 190
    base = Image.fromarray(img_u8).convert('RGB').resize((w * S, h * S), Image.LANCZOS)
    canvas = Image.new('RGB', (w * S + 2 * PAD, h * S + 2 * PAD + CAP), (18, 18, 18))
    canvas.paste(base, (PAD, PAD))
    d = ImageDraw.Draw(canvas)
    font, _ = _font(15)
    font_b, _ = _font(17)

    def P(x, y):
        return PAD + x * S, PAD + y * S

    d.rectangle([PAD - 1, PAD - 1, PAD + w * S, PAD + h * S], outline=GRAY, width=1)

    top_px = TOP_MM / MM_PER_PX
    bot_px = h - BOTTOM_MM / MM_PER_PX
    d.line([P(0, top_px), P(w, top_px)], fill=GRAY, width=1)
    d.line([P(0, bot_px), P(w, bot_px)], fill=GRAY, width=1)
    xt = (LAT_MM / MM_PER_PX) if lat_left else (w - LAT_MM / MM_PER_PX)
    d.line([P(xt, 0), P(xt, h)], fill=GRAY, width=1)

    for pt in (T, L, I):
        x, y = P(pt['x'], pt['y'])
        d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=YELLOW, width=2)
    d.line([P(0, T['y']), P(w, T['y'])], fill=CYAN, width=2)
    d.line([P(0, I['y']), P(w, I['y'])], fill=CYAN, width=2)
    d.line([P(L['x'], 0), P(L['x'], h)], fill=CYAN, width=2)

    m_top = T['y'] * MM_PER_PX
    m_bottom = (h - I['y']) * MM_PER_PX
    m_lat = (L['x'] if lat_left else w - L['x']) * MM_PER_PX
    top_ok, bottom_ok, lat_ok = m_top >= TOP_MM, m_bottom >= BOTTOM_MM, m_lat >= LAT_MM

    def arrow_v(x, y0, y1, color):
        d.line([(x, y0), (x, y1)], fill=color, width=3)
        for yy, s in ((y0, 1), (y1, -1)):
            d.polygon([(x, yy), (x - 5, yy + s * 9), (x + 5, yy + s * 9)], fill=color)

    xa = P(T['x'] + (12 if lat_left else -12), 0)[0]
    arrow_v(xa, P(0, 0)[1] + 2, P(0, T['y'])[1], GREEN if top_ok else RED)
    d.text((xa + (8 if lat_left else -8), P(0, T['y'] / 2)[1]),
           'сверху %.1f см %s 3 см' % (m_top / 10, '>=' if top_ok else '<'),
           fill=GREEN if top_ok else RED, font=font_b, anchor='lm' if lat_left else 'rm')

    xb = P(I['x'] + (12 if lat_left else -12), 0)[0]
    arrow_v(xb, P(0, I['y'])[1], P(0, h)[1] - 2, GREEN if bottom_ok else RED)
    d.text((xb + (8 if lat_left else -8), P(0, (I['y'] + h) / 2)[1]),
           'снизу %.1f см %s 3 см' % (m_bottom / 10, '>=' if bottom_ok else '<'),
           fill=GREEN if bottom_ok else RED, font=font_b, anchor='lm' if lat_left else 'rm')

    x_edge = P(0, 0)[0] + 2 if lat_left else P(w, 0)[0] - 2
    yl = P(0, min(T['y'] + 12 / MM_PER_PX, h - 4))[1]
    d.line([(x_edge, yl), (P(L['x'], 0)[0], yl)], fill=GREEN if lat_ok else RED, width=3)
    d.text((P(L['x'], 0)[0] + (6 if lat_left else -6), yl - 12),
           'латерально %.1f см %s 2 см' % (m_lat / 10, '>=' if lat_ok else '<'),
           fill=GREEN if lat_ok else RED, font=font_b, anchor='la' if lat_left else 'ra')

    y0 = PAD + h * S + 12
    side_ru = 'правое бедро' if side == 'rh' else 'левое бедро'
    d.text((PAD, y0),
           '%s  |  %s, кадр %d x %d px, точки - реальная разметка эксперта' % (name, side_ru, w, h),
           fill=WHITE, font=font)
    d.text((PAD, y0 + 24), group, fill=WHITE, font=font_b)
    d.text((PAD, y0 + 48), 'Вердикт по точкам: %s' % ('норма' if verdict else 'нарушение'),
           fill=GREEN if verdict else RED, font=font_b)
    for i, ln in enumerate(textwrap.wrap(why, width=max(60, (w * S) // 8))[:4]):
        d.text((PAD, y0 + 78 + 21 * i), ln, fill=WHITE, font=font)
    return canvas


def main(argv=None) -> int:
    OUT.mkdir(exist_ok=True)
    html_images = load_html_images()
    d = json.load(open(ANNOTATIONS_JSON, encoding='utf-8'))
    match = match_images(html_images)
    html_by_id = {im['id']: im for im in html_images}

    tmp = OUT / '_build'
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    rows = []
    for name, study_suffix, side, group, why in CASES:
        cand = match[match.study.str.endswith(study_suffix) & (match.label == side)]
        if len(cand) != 1:
            raise SystemExit(f'не найден однозначный id для {name}: {len(cand)} кандидатов')
        jid = cand.iloc[0]['id']
        im = html_by_id[jid]
        img = decode_png(im)
        pts = d['images'][jid]['points']
        m_top = pts['trochanter_major']['y'] * MM_PER_PX
        m_bottom = (im['h'] - pts['ischium_bottom']['y']) * MM_PER_PX
        lat_x = pts['trochanter_lateral']['x']
        m_lat = (lat_x if side == 'rh' else im['w'] - lat_x) * MM_PER_PX
        verdict = m_top >= TOP_MM and m_bottom >= BOTTOM_MM and m_lat >= LAT_MM

        canvas = render_annotated(img, side, im['w'], im['h'], pts, verdict, name, group, why)
        canvas.save(tmp / f'{name}.png')

        study = cand.iloc[0]['study']
        suffix10 = study.split('.')[-1][-10:]
        orig_name = f'original_{side}_{suffix10}.png'
        Image.fromarray(img).save(tmp / orig_name)

        rows.append(dict(case=name, id=jid, study=study, side=side,
                         m_top_mm=round(m_top, 1), m_bottom_mm=round(m_bottom, 1), m_lat_mm=round(m_lat, 1),
                         verdict_true=('норма' if verdict else 'нарушение'), original_file=orig_name,
                         rel_path=cand.iloc[0]['rel_path']))
        print(f'{name}: верх={m_top:.1f} низ={m_bottom:.1f} лат={m_lat:.1f} -> '
              f'{"норма" if verdict else "НАРУШЕНИЕ"}  оригинал -> {orig_name}')

    pd.DataFrame(rows).to_csv(tmp / 'cases.csv', index=False, encoding='utf-8-sig')

    margins = true_margins(match)
    dis = real_disagreements(margins)
    dis.to_csv(tmp / 'all_real_disagreements.csv', index=False, encoding='utf-8-sig')
    print(f'\nвсего настоящих расхождений (по точкам эксперта): {len(dis)}')

    note = OUT / NOTE_NAME
    zpath = OUT / ZIP_NAME
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.iterdir()):
            z.write(p, p.name)
        if note.exists():
            z.write(note, NOTE_NAME)
    shutil.rmtree(tmp)
    print(f'архив: {zpath} ({zpath.stat().st_size / 1e6:.2f} МБ)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
