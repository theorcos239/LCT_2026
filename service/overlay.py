# -*- coding: utf-8 -*-
"""Единая визуализация вердикта: позвоночник или бедро — по области кадра."""
from __future__ import annotations

import numpy as np
from PIL import Image


def render(px: np.ndarray, row: dict, scale: int = 2) -> Image.Image:
    """Кадр + разметка того, на чём построен вердикт."""
    region = row.get('anatomical_region', 'unknown')
    if region == 'spine':
        from spine_qc.overlay import render_overlay
        from spine_qc.criteria import analyze
        return render_overlay(px, analyze(px), scale=scale)
    if region in ('lh', 'rh'):
        from hip_roi.overlay import render_overlay as hip_overlay
        from hip_roi.kits import measure_roi_margins
        return hip_overlay(px, measure_roi_margins(px, region, 'kit2'))
    from hip_roi.geometry import to_uint8
    img = Image.fromarray(to_uint8(px)).convert('RGB')
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)
