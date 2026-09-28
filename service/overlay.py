# -*- coding: utf-8 -*-
"""Единая визуализация вердикта: позвоночник или бедро — по области кадра.

Картинка обязана совпадать с отчётом. Поэтому критерии позвоночника
пересчитываются с калиброванными порогами (spine_qc/thresholds.json), а не с
порогами по умолчанию, и если в строке отчёта есть детали — вердикты берутся
оттуда: в них уже учтены модель точек и свод с нейросетью (cnn_qc).

Если нарушение подтвердила нейросеть, поверх кадра ложится её тепловая карта
(CAM) — где на снимке сеть видит признаки нарушения (ТЗ 2.6: «маска, контур,
ключевые точки или тепловая карта»).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HEAT = np.array([255, 96, 0], dtype=np.float32)     # оранжевый: не спутать с красным «нарушение»


@lru_cache(maxsize=1)
def _spine_thresholds() -> dict:
    from spine_qc.criteria import DEFAULTS
    p = Path(__file__).resolve().parent.parent / 'spine_qc' / 'thresholds.json'
    thr = dict(DEFAULTS)
    if p.exists():
        thr.update(json.loads(p.read_text(encoding='utf-8')))
    return thr


@lru_cache(maxsize=1)
def _cnn():
    try:
        from cnn_qc import CnnQC
        return CnnQC()
    except Exception:                       # noqa: BLE001 — без сети просто нет карты
        return None


def _heat(img: Image.Image, px: np.ndarray, crit: str, region: str, label: str) -> Image.Image:
    """Тепловая карта сети поверх готовой картинки."""
    net = _cnn()
    if net is None or not net.available(crit):
        return img
    try:
        from cnn_qc.predict import heatmap
        _, cam = net.p_cnn(crit, px, region)
        hm = heatmap(cam, px.shape, region)
    except Exception:                       # noqa: BLE001 — визуализация не критична
        return img
    hm = np.asarray(Image.fromarray(hm, mode='F').resize(img.size, Image.BILINEAR))
    # Показываем только верхнюю половину карты и с нарастанием: CAM 10x10,
    # растянутая на кадр, иначе красит полкадра ровным слоем.
    a = (np.clip((hm - 0.45) / 0.55, 0.0, 1.0) ** 1.5 * 0.6)[..., None]
    rgb = np.asarray(img.convert('RGB'), dtype=np.float32)
    out = Image.fromarray(np.clip(rgb * (1 - a) + HEAT * a, 0, 255).astype(np.uint8))
    from hip_roi.overlay import _font
    font, cyr = _font(12)
    text = label if cyr else 'CNN heatmap'
    d = ImageDraw.Draw(out)
    tw = int(d.textlength(text, font=font)) + 10
    d.rectangle([(out.width - tw, 0), (out.width, 18)], fill=(0, 0, 0))    # правый верх: низ занят вердиктом
    d.text((out.width - tw + 5, 2), text, fill=tuple(int(v) for v in HEAT), font=font)
    return out


def render(px: np.ndarray, row: dict, scale: int = 2) -> Image.Image:
    """Кадр + разметка того, на чём построен вердикт."""
    region = row.get('anatomical_region', 'unknown')
    details = row.get('details') or {}
    if region == 'spine':
        from spine_qc.criteria import analyze
        from spine_qc.overlay import render_overlay
        res = analyze(px, None, _spine_thresholds())
        rep = details.get('spine') or {}
        for k, c in res['criteria'].items():
            if k in rep and 'violated' in rep[k]:
                c['violated'] = bool(rep[k]['violated'])
                c['text'] = rep[k].get('text', c.get('text', ''))
        res['violations'] = [c['text'] for c in res['criteria'].values()
                             if c['violated'] and c.get('text')]
        img = render_overlay(px, res, scale=scale)
        art = rep.get('artifacts') or {}
        if art.get('violated') and art.get('p_cnn') is not None:
            img = _heat(img, px, 'artifacts', region,
                        f"нейросеть: артефакт {art['p_cnn']:.2f}")
        return img
    if region in ('lh', 'rh'):
        from hip_roi.kits import measure_roi_margins
        from hip_roi.overlay import render_overlay as hip_overlay
        img = hip_overlay(px, measure_roi_margins(px, region, 'kit2'))
        rot = details.get('hip_rotation') or {}
        if rot.get('violated') and rot.get('p_cnn') is not None:
            img = _heat(img, px, 'rotation', region,
                        f"нейросеть: ротация {rot['p_cnn']:.2f}")
        return img
    from hip_roi.geometry import to_uint8
    img = Image.fromarray(to_uint8(px)).convert('RGB')
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)
