# -*- coding: utf-8 -*-
"""Детерминированная проверка отступов ROI на снимке проксимального отдела бедра.

    from hip_roi import measure_roi_margins
    res = measure_roi_margins(pixel_array, side='lh', method='kit2')
"""
from .kits import METHODS, measure_roi_margins  # noqa: F401
