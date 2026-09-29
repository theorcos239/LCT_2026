# -*- coding: utf-8 -*-
"""Иконка приложения: четыре позвонка L1–L4 на синем квадрате.

    python desktop/make_icons.py

Пишет desktop/assets/icon.ico (16–256 px) и PNG для веб-приложения
(service/static/icons/): 192 и 512 px, вариант maskable с полями под обрезку.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOP, BOTTOM = (0x2A, 0x70, 0xE2), (0x14, 0x4C, 0xB4)     # акцент презентации #1B5FD0


def draw(size: int = 1024, pad: float = 0.0, radius: float = 0.22) -> Image.Image:
    s = size
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    grad = Image.new('RGBA', (s, s))
    gd = ImageDraw.Draw(grad)
    for y in range(s):
        t = y / (s - 1)
        gd.line([(0, y), (s, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(TOP, BOTTOM)) + (255,))
    mask = Image.new('L', (s, s), 0)
    inset = int(s * pad)
    ImageDraw.Draw(mask).rounded_rectangle([inset, inset, s - inset - 1, s - inset - 1],
                                           radius=int(s * radius) if not pad else 0, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # позвонки: сверху вниз шире и выше, как L1 -> L4
    area = s * (1 - 2 * pad) if pad else s
    x0 = s / 2
    widths = [0.40, 0.44, 0.48, 0.52]
    heights = [0.130, 0.140, 0.150, 0.160]
    gap = 0.045
    total = sum(heights) + gap * 3
    y = s / 2 - total * area / 2
    for w, h in zip(widths, heights):
        ww, hh = w * area, h * area
        d.rounded_rectangle([x0 - ww / 2, y, x0 + ww / 2, y + hh], radius=hh * 0.32,
                            fill=(255, 255, 255, 255))
        y += hh + gap * area
    return img


def main() -> int:
    assets = HERE / 'assets'
    icons = ROOT / 'service' / 'static' / 'icons'
    assets.mkdir(exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)
    big = draw()
    big.resize((256, 256), Image.LANCZOS).save(assets / 'icon.ico',
                                               sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                                      (64, 64), (128, 128), (256, 256)])
    big.resize((256, 256), Image.LANCZOS).save(assets / 'icon.png')
    for n in (192, 512):
        big.resize((n, n), Image.LANCZOS).save(icons / f'icon-{n}.png', optimize=True)
    # maskable: фон до краёв, рисунок в безопасной зоне (80 % по центру)
    draw(pad=0.1).resize((512, 512), Image.LANCZOS).save(icons / 'maskable-512.png', optimize=True)
    big.resize((180, 180), Image.LANCZOS).save(icons / 'apple-touch-icon.png', optimize=True)
    big.resize((32, 32), Image.LANCZOS).save(icons / 'favicon-32.png', optimize=True)
    print(assets / 'icon.ico', icons)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
