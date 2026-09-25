# -*- coding: utf-8 -*-
"""Архив «экспертная оценка против критерия ТЗ» по отступам ROI.

    python -m hip_roi.report_contradictions

Собирает hip_roi/contradictions/expert_vs_tz.zip:
  - 8 кадров с нанесёнными границами ROI, размерными стрелками
    (красная — отступ меньше порога ТЗ, зелёная — выдержан) и подписью
    «эксперт / ТЗ / обоснование»;
  - contradictions.csv — все кадры, где вердикт kit2 по порогам ТЗ не
    совпал с оценкой эксперта, с отступами в мм;
  - ПОЯСНЕНИЕ.md — тот же текст, что лежит рядом с архивом.

Кадры — все расхождения основного комплекта (kit2) с экспертом на 153 кадрах
бедра: 4 случая «эксперт норма / ТЗ нарушение» и 4 обратных. Каждый просмотрен
глазами: верхушка вертела и латеральный край на них найдены верно.
"""
from __future__ import annotations

import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from region_clf.features import read_image

from .evaluate import DATA, hip_frames
from .geometry import BOTTOM_MM, LAT_MM, MM_PER_PX, TOP_MM, mm2px, to_uint8
from .kits import measure_roi_margins
from .overlay import _font

OUT = Path(__file__).resolve().parent / 'contradictions'
ZIP_NAME = 'expert_vs_tz.zip'
NOTE_NAME = 'ПОЯСНЕНИЕ.md'

# (суффикс study, сторона, имя файла, группа, обоснование)
CASES = [
    ('4105064465', 'lh', 'A1_top_2.7cm_lh_235',
     'A. Эксперт: норма. ТЗ: нарушение сверху',
     'Над верхушкой большого вертела 2.7 см при требовании 3 см. Верхушка '
     'найдена по контуру и подтверждена вторым способом (расхождение 0 мм).'),
    ('7897449628', 'lh', 'A2_lateral_1.6cm_lh_291',
     'A. Эксперт: норма. ТЗ: нарушение латерально',
     'Латеральный край большого вертела в 1.6 см от края кадра при требовании '
     '2 см — самый тесный латеральный отступ в выборке. Верх и низ в норме.'),
    ('8427669848', 'rh', 'A3_lateral_1.8cm_rh_261',
     'A. Эксперт: норма. ТЗ: нарушение латерально',
     'Латеральный край вертела в 1.8 см от края кадра, требуется 2 см. '
     'Верх и низ в норме (3.9 / 7.0 см).'),
    ('1625232853', 'rh', 'A4_lateral_1.8cm_rh_289',
     'A. Эксперт: норма. ТЗ: нарушение латерально',
     'Латеральный край вертела в 1.8 см от края кадра, требуется 2 см. '
     'Верх и низ в норме (3.6 / 8.9 см).'),
    ('7654885530', 'lh', 'B1_expert_flag_lat_2.1cm_lh_235',
     'B. Эксперт: некорректная область интересов. ТЗ: все отступы в норме',
     'Единственный тесный отступ — латеральный, 2.1 см, порог ТЗ 2 см выдержан. '
     'Вместе с A2-A4 (1.6-1.8 см, экспертом НЕ помечены) это прямое '
     'противоречие внутри разметки: 1.6 см — норма, 2.1 см — нарушение.'),
    ('0069487890', 'rh', 'B2_expert_flag_bottom_3.0cm_rh_206',
     'B. Эксперт: некорректная область интересов. ТЗ: все отступы в норме',
     'Ниже области интереса 3.0 см — ровно на пороге. Низ области интереса '
     'здесь поставлен по анатомическому якорю (измерить по кадру не удалось), '
     'его точность ±1.5 см, поэтому случай пограничный в обе стороны.'),
    ('1065649420', 'rh', 'B3_expert_flag_all_ok_rh_207',
     'B. Эксперт: некорректная область интересов. ТЗ: все отступы в норме',
     'Верх 3.8 см, низ 4.2 см, латераль 3.6 см — все три отступа выдержаны. '
     'Кадр короткий (12.6 см), но вертелы в него попали целиком.'),
    ('4128739670', 'lh', 'B4_expert_flag_all_ok_lh_261',
     'B. Эксперт: некорректная область интересов. ТЗ: все отступы в норме',
     'Верх 3.8 см, низ 7.5 см, латераль 5.0 см — с запасом по всем трём. '
     'Возможно, эксперт оценивал другое: головка бедра и вертлужная впадина '
     'здесь упираются в верхний край кадра.'),
]

RED = (255, 70, 70)
GREEN = (90, 230, 90)
CYAN = (0, 220, 255)
GRAY = (150, 150, 150)
WHITE = (240, 240, 240)
S = 3            # масштаб
PAD = 24
CAPTION_H = 190


def _arrow(d, p0, p1, color, width=3, head=9):
    """Двусторонняя стрелка между точками p0 и p1 (горизонтальная или вертикальная)."""
    d.line([p0, p1], fill=color, width=width)
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1:                                   # вертикальная
        s = 1 if y1 > y0 else -1
        d.polygon([(x0, y0), (x0 - head // 2, y0 + s * head), (x0 + head // 2, y0 + s * head)], fill=color)
        d.polygon([(x1, y1), (x1 - head // 2, y1 - s * head), (x1 + head // 2, y1 - s * head)], fill=color)
    else:                                          # горизонтальная
        s = 1 if x1 > x0 else -1
        d.polygon([(x0, y0), (x0 + s * head, y0 - head // 2), (x0 + s * head, y0 + head // 2)], fill=color)
        d.polygon([(x1, y1), (x1 - s * head, y1 - head // 2), (x1 - s * head, y1 + head // 2)], fill=color)


def _label(d, xy, text, color, font, anchor='la'):
    """Текст с тёмной подложкой."""
    box = d.textbbox(xy, text, font=font, anchor=anchor)
    d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0))
    d.text(xy, text, fill=color, font=font, anchor=anchor)


def _label_beside(d, x, y, text, color, font, right_first: bool, canvas_w: int, gap: int = 8):
    """Подпись рядом с вертикальной стрелкой в точке x: с предпочтительной
    стороны, а если текст вылезает за холст — с другой."""
    w = d.textlength(text, font=font)
    if right_first:
        if x + gap + w + 6 <= canvas_w:
            _label(d, (x + gap, y), text, color, font, 'lm')
        else:
            _label(d, (x - gap, y), text, color, font, 'rm')
    else:
        if x - gap - w - 6 >= 0:
            _label(d, (x - gap, y), text, color, font, 'rm')
        else:
            _label(d, (x + gap, y), text, color, font, 'lm')


def render_case(img, res, case, expert_flag):
    suf, side, name, group, why = case
    u8 = to_uint8(img)
    H, W = u8.shape
    font, cyr = _font(15)
    font_b, _ = _font(17)
    base = Image.fromarray(u8).convert('RGB').resize((W * S, H * S), Image.LANCZOS)
    canvas = Image.new('RGB', (W * S + 2 * PAD, H * S + 2 * PAD + CAPTION_H), (18, 18, 18))
    canvas.paste(base, (PAD, PAD))
    d = ImageDraw.Draw(canvas)

    def P(x, y):
        return PAD + x * S, PAD + y * S

    T, B, L = res['T_px'], res['B_px'], res['L_px']
    ax, ay = res['apex_xy']
    lat_left = side == 'rh'
    edge_x = 0 if lat_left else W - 1
    cm = lambda mm: f'{mm / 10:.1f} см'

    # рамка кадра — край поля сканирования
    d.rectangle([PAD - 1, PAD - 1, PAD + W * S, PAD + H * S], outline=GRAY, width=1)

    # границы ROI (циан)
    d.line([P(0, T), P(W, T)], fill=CYAN, width=2)
    d.line([P(0, B), P(W, B)], fill=CYAN, width=2)
    d.line([P(L, 0), P(L, H)], fill=CYAN, width=2)
    d.ellipse([P(ax, ay)[0] - 6, P(ax, ay)[1] - 6, P(ax, ay)[0] + 6, P(ax, ay)[1] + 6], outline=(255, 230, 0), width=2)
    med_anchor = 'ra' if lat_left else 'la'
    med_x = PAD + W * S - 6 if lat_left else PAD + 6
    _label(d, (med_x, P(0, T)[1] + 4), 'верх ROI — верхушка большого вертела', CYAN, font, med_anchor)
    _label(d, (med_x, P(0, B)[1] + 4), 'низ ROI — конец вертельной массы (якорь, ±1.5 см)', CYAN, font, med_anchor)
    _label(d, (P(L, 0)[0] + (6 if lat_left else -6), PAD + H * S - 44), 'латеральный край ROI', CYAN, font,
           'la' if lat_left else 'ra')

    # пороги ТЗ (тонкие серые)
    for y, txt in ((mm2px(TOP_MM), f'{TOP_MM / 10:.0f} см от верхнего края'),
                   (H - mm2px(BOTTOM_MM), f'{BOTTOM_MM / 10:.0f} см от нижнего края')):
        d.line([P(0, y), P(W, y)], fill=GRAY, width=1)
        _label(d, (PAD + W * S // 2, P(0, y)[1] - 2), txt, GRAY, font, 'mb')
    xt = mm2px(LAT_MM) if lat_left else W - 1 - mm2px(LAT_MM)
    d.line([P(xt, 0), P(xt, H)], fill=GRAY, width=1)
    _label(d, (P(xt, 0)[0] + (4 if lat_left else -4), PAD + 6), f'{LAT_MM / 10:.0f} см от края', GRAY, font,
           'la' if lat_left else 'ra')

    # размерные стрелки
    m_top, m_bot, m_lat = res['m_top_mm'], res['m_bottom_mm'], res['m_lat_mm']
    off = -12 if lat_left else 12
    # низ: от B до нижнего края, латеральнее диафиза
    xb = min(max(L + off, 4), W - 5)
    col = GREEN if res['bottom_ok'] else RED
    _arrow(d, P(xb, B), (P(xb, H)[0], P(xb, H)[1] - 2), col)
    sign = '≥' if res['bottom_ok'] else '<'
    _label_beside(d, P(xb, 0)[0], P(0, (B + H) // 2)[1], f'снизу {cm(m_bot)} {sign} {BOTTOM_MM / 10:.0f} см',
                  col, font_b, right_first=not lat_left, canvas_w=canvas.size[0])
    # верх: от верхнего края до T, латеральнее верхушки
    xa = min(max(ax + off, 4), W - 5)
    col = GREEN if res['top_ok'] else RED
    _arrow(d, (P(xa, 0)[0], P(xa, 0)[1] + 2), P(xa, T), col)
    sign = '≥' if res['top_ok'] else '<'
    _label_beside(d, P(xa, 0)[0], P(0, T // 2)[1], f'сверху {cm(m_top)} {sign} {TOP_MM / 10:.0f} см',
                  col, font_b, right_first=not lat_left, canvas_w=canvas.size[0])
    # латераль: от края до L на уровне чуть ниже верхушки
    yl = min(T + mm2px(12), H - 4)
    col = GREEN if res['lat_ok'] else RED
    x_edge = P(0, 0)[0] + 2 if lat_left else P(W, 0)[0] - 2
    _arrow(d, (x_edge, P(0, yl)[1]), (P(L, yl)[0], P(0, yl)[1]), col)
    sign = '≥' if res['lat_ok'] else '<'
    _label(d, (P(L, 0)[0] + (6 if lat_left else -6), P(0, yl)[1] - 12),
           f'латерально {cm(m_lat)} {sign} {LAT_MM / 10:.0f} см', col, font_b, 'lb' if lat_left else 'rb')

    # подпись
    y0 = PAD + H * S + 12
    x0 = PAD
    side_ru = 'правое бедро' if side == 'rh' else 'левое бедро'
    expert_txt = 'некорректная область интересов' if expert_flag == 1 else 'норма'
    tz_txt = res['violation_text'] or 'все отступы выдержаны'
    d.text((x0, y0), f'{name}   |   исследование …{suf}, {side_ru}, кадр {W}×{H} px '
                     f'({W * MM_PER_PX / 10:.1f} × {H * MM_PER_PX / 10:.1f} см)', fill=WHITE, font=font)
    d.text((x0, y0 + 24), f'Оценка эксперта: {expert_txt}', fill=WHITE, font=font_b)
    d.text((x0, y0 + 48), f'По критерию ТЗ (3 / 3 / 2 см): {tz_txt}',
           fill=GREEN if res['roi_ok'] else RED, font=font_b)
    lines = textwrap.wrap(why, width=max(60, (W * S) // 8))
    for i, ln in enumerate(lines[:4]):
        d.text((x0, y0 + 78 + 21 * i), ln, fill=WHITE, font=font)
    return canvas


def build_contradictions_csv(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in frames.iterrows():
        if r.expert_roi not in (0, 1):
            continue
        img = read_image(DATA / r.rel_path)
        res = measure_roi_margins(img, r.label, 'kit2')
        pred = None if res['roi_ok'] is None else int(not res['roi_ok'])
        if pred is None or pred != int(r.expert_roi):
            rows.append(dict(study=r.study, side=r.label, file=r.rel_path, rows=img.shape[0],
                             m_top_mm=res['m_top_mm'], m_bottom_mm=res['m_bottom_mm'], m_lat_mm=res['m_lat_mm'],
                             expert_roi=int(r.expert_roi), tz_violation=pred,
                             violation_text=res['violation_text'], flags=';'.join(res['flags'])))
    return pd.DataFrame(rows).sort_values(['expert_roi', 'm_bottom_mm'])


def main(argv=None):
    OUT.mkdir(exist_ok=True)
    tmp = OUT / '_build'
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    frames = hip_frames()
    for case in CASES:
        suf, side, name = case[:3]
        r = frames[frames.study.str.endswith(suf) & (frames.label == side)].iloc[0]
        img = read_image(DATA / r.rel_path)
        res = measure_roi_margins(img, side, 'kit2')
        render_case(img, res, case, r.expert_roi).save(tmp / f'{name}.png')
        print(f'{name}: верх {res["m_top_mm"]} низ {res["m_bottom_mm"]} лат {res["m_lat_mm"]} мм, '
              f'эксперт={r.expert_roi}, ТЗ={"норма" if res["roi_ok"] else res["violation_text"]}')
    csv = build_contradictions_csv(frames)
    csv.to_csv(tmp / 'contradictions.csv', index=False, encoding='utf-8-sig')
    print(f'всего расхождений kit2 с экспертом: {len(csv)} -> contradictions.csv')
    note = OUT / NOTE_NAME
    zpath = OUT / ZIP_NAME
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.iterdir()):
            z.write(p, p.name)
        if note.exists():
            z.write(note, NOTE_NAME)
    shutil.rmtree(tmp)
    print(f'архив: {zpath} ({zpath.stat().st_size / 1e6:.1f} МБ)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
