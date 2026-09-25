# -*- coding: utf-8 -*-
"""Ротация проксимального отдела бедра по выступу малого вертела."""
import json
from pathlib import Path

from .detect import DEFAULTS, detect, lesser_trochanter_bulge

_THR = Path(__file__).resolve().parent / 'thresholds.json'
THRESHOLDS = {**DEFAULTS, **(json.loads(_THR.read_text(encoding='utf-8')) if _THR.exists() else {})}

__all__ = ['detect', 'lesser_trochanter_bulge', 'THRESHOLDS', 'DEFAULTS']
