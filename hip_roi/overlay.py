# -*- coding: utf-8 -*-
"""Визуализация результата: маска кости, границы ROI, пороговые линии.

Это «дополнительная серия с визуализацией нарушения» из ТЗ (п. 2.6):
на исходном кадре рисуются найденные границы ROI (T, B, L) и линии порогов
3/3/2 см от краёв. Линия порога зелёная, если отступ выдержан, красная — нет.
T и L измерены по кадру, B — анатомический якорь от T с поправкой (±1.5 см).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import BOTTOM_MM, LAT_MM, TOP_MM, bone_mask, mm2px, to_uint8

_FONT_CANDIDATES = (
    'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/tahoma.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf',
)


def _font(size: int = 12):
    """TrueType-шрифт с кириллицей; штатный растровый шрифт PIL её не умеет."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size), True
        except OSError:
            continue
    return ImageFont.load_default(), False

CYAN = (0, 220, 255)
GREEN = (60, 220, 60)
RED = (255, 60, 60)
YELLOW = (255, 230, 0)
BLUE = (70, 90, 255)


def render_overlay(img: np.ndarray, res: dict, scale: int = 3, show_mask: bool = True) -> Image.Image:
    img8 = to_uint8(img)
    h, w = img8.shape
    rgb = np.stack([img8] * 3, axis=-1).astype(np.int16)
    if show_mask:
        m = bone_mask(img8)
        edge = m & ~np.pad(m, 1)[1:-1, 1:-1]
        rgb[m] = (rgb[m] * 0.75 + np.array(BLUE) * 0.25)
        rgb[edge] = BLUE
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((w * scale, h * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    S = scale
    side = res['side']
    lat_x = 0 if side == 'rh' else w - 1

    def hline(y, color, width=1):
        d.line([(0, y * S), (w * S, y * S)], fill=color, width=width)

    def vline(x, color, width=1):
        d.line([(x * S, 0), (x * S, h * S)], fill=color, width=width)

    # пороговые линии
    col = lambda ok: GREEN if ok else (RED if ok is False else YELLOW)
    hline(mm2px(TOP_MM), col(res['top_ok']))
    hline(h - mm2px(BOTTOM_MM), col(res['bottom_ok']))
    vline(mm2px(LAT_MM) if side == 'rh' else w - 1 - mm2px(LAT_MM), col(res['lat_ok']))

    # границы ROI
    if res['T_px'] is not None:
        hline(res['T_px'], CYAN, 2)
    if res['B_px'] is not None and res['B_px'] < h:
        hline(res['B_px'], CYAN, 2)
    if res['L_px'] is not None:
        vline(res['L_px'], CYAN, 2)
    if res['apex_xy'] is not None:
        x, y = res['apex_xy']
        d.ellipse([(x * S - 4, y * S - 4), (x * S + 4, y * S + 4)], outline=YELLOW, width=2)

    font, cyr = _font(12)
    lab = ('верх', 'низ', 'лат') if cyr else ('top', 'bot', 'lat')
    verdict = _verdict(res['roi_ok'], cyr)
    txt = (f"{res['method']} {side}  {lab[0]} {_cm(res['m_top_mm'])}  {lab[1]} {_cm(res['m_bottom_mm'])}  "
           f"{lab[2]} {_cm(res['m_lat_mm'])}  -> {verdict}")
    d.rectangle([(0, 0), (min(w * S, 7 * len(txt) + 6), 16)], fill=(0, 0, 0))
    d.text((3, 2), txt, fill=YELLOW, font=font)
    if res['flags']:
        d.text((3, 18), ', '.join(res['flags'])[:120], fill=(255, 160, 0), font=font)
    return im


def _cm(mm):
    return '—' if mm is None else f'{mm / 10:.1f}'


def _verdict(ok, cyr=True):
    return {True: 'ok', False: 'НАРУШЕНИЕ' if cyr else 'VIOLATION', None: '?'}[ok]


def contact_sheet(tiles: list[Image.Image], cols: int = 4, bg=(30, 30, 30)) -> Image.Image:
    if not tiles:
        return Image.new('RGB', (10, 10), bg)
    tw = max(t.size[0] for t in tiles)
    th = max(t.size[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, rows * th), bg)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    return sheet
