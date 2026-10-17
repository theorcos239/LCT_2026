# -*- coding: utf-8 -*-
"""Ротация проксимального отдела бедра по малому вертелу (ТЗ, п. 2.3, бедро).

Критерий ТЗ двусторонний, и это определяет всю конструкцию:

    корректно     — контур слегка деформирован малым вертелом
    переротация   — контур плавный, малого вертела не видно
    недоротация   — малый вертел слишком большой

То есть нарушением считается и слишком маленький выступ, и слишком большой, а
норма — середина. Поэтому вердикт выносится не порогом «больше/меньше», а
выходом измерения за коридор нормы, калиброванный на обучающей выборке.

Границы коридора подбираются независимо (разные перцентили снизу и сверху):
переротация и недоротация — разные механизмы, симметричный коридор их смешивает.

**Это самый слабый критерий в решении, и это измеренный факт, а не оговорка.**
OOF AUC около 0.68 против 0.79-0.82 у критериев позвоночника. Причина известна
заранее: разрыв между классами по выступу малого вертела
составляет около 2 мм при ширине диафиза 35 мм, а выступ — это разность двух
координат, то есть ошибки складываются. Чтобы уверенно различать классы, обе
точки нужно находить точнее пикселя — это задача модели ключевых точек
(`src/dxa_qc`), а не контурной геометрии. Здесь сделано лучшее, что даёт
силуэт, и помечено флагом низкой уверенности.
"""
from __future__ import annotations

import numpy as np

from hip_roi.geometry import (bone_mask, lateral_profile, medial_profile, mm2px,
                              moving_average, px2mm, to_lateral_left, to_uint8)
from hip_roi.kits import measure_roi_margins
from hip_roi.landmarks import _theil_sen as theil_sen

# Коридор нормы по площади выступа, мм². Границы — перцентили кадров,
# признанных экспертом корректными (calibrate.py); lo и hi — разные.
DEFAULTS = {
    'area_lo': 70.0,
    'area_hi': 302.0,
    'appearance': 0.5,
    'neck_mm': 14.0,        # отклонение контура, после которого это уже шейка
    'fit_mm': 55.0,         # длина участка диафиза для опорной прямой
    'score': 'area',        # 'area' | 'relative' | 'area_over_shaft2'
}


def lesser_trochanter_bulge(img: np.ndarray, side: str, T: int,
                            neck_mm: float = 14.0, fit_mm: float = 55.0) -> dict | None:
    """Выступ малого вертела над прямой медиального контура диафиза.

    Опорная прямая строится по медиальному контуру диафиза (Тейл-Сен, нижние
    `fit_mm` строк с костью, но не выше T + 55 мм — выше начинается вертельная
    зона). Дальше идём вверх от диафиза и **останавливаемся на входе в шейку**:
    отклонение контура больше `neck_mm` — это уже не вертел.

    Остановка обязательна. Без неё максимум отклонения приходится на
    медиальный край шейки, измерение даёт 60-70 мм вместо 3-8 мм и не имеет
    отношения к малому вертелу (проверено: AUC падает до 0.63 при бессмысленных
    абсолютных значениях).
    """
    m = to_lateral_left(bone_mask(to_uint8(img)), side)
    h, w = m.shape
    xl = lateral_profile(m)
    xm = medial_profile(m, xl)
    valid = (xl < w) & (xm < w) & ((xm - xl) > mm2px(12.0))
    ys = np.arange(h)
    av = ys[valid]
    if av.size < mm2px(50.0):
        return None

    bottom = int(av[-1])
    fit = av[(av >= max(bottom - mm2px(fit_mm), T + mm2px(55.0))) & (av <= bottom)]
    if fit.size < 8:
        fit = av[av >= bottom - mm2px(fit_mm)]
    if fit.size < 8:
        return None

    smooth = moving_average(xm.astype(float), 5)
    slope, intercept = theil_sen(fit.astype(float), smooth[fit])
    dev = smooth - (slope * ys + intercept)

    top_fit = int(fit[0])
    limit = mm2px(neck_mm)
    stop, y = top_fit, top_fit
    while y > max(T, 0) and valid[y] and dev[y] < limit:
        stop = y
        y -= 1
    win = ys[(ys >= stop) & (ys <= top_fit) & valid]
    if win.size < 4:
        return None

    dv = dev[win]
    peak = float(np.max(dv))
    shaft = px2mm(float(np.median((xm - xl)[fit])))
    flags = []
    if dev[max(y, 0)] < limit:
        flags.append('rotation:neck_not_reached')
    if win.size < mm2px(12.0):
        flags.append('rotation:short_window')
    area = float(np.sum(np.clip(dv, 0.0, None)) * px2mm(1.0) * px2mm(1.0))
    bulge = px2mm(peak)
    return {
        'bulge_mm': round(bulge, 2),
        'area_mm2': round(area, 1),
        'span_mm': round(px2mm(float(win.size)), 1),
        'shaft_mm': round(shaft, 1),
        'relative': round(bulge / max(shaft, 1e-6), 3),
        'area_over_shaft2': round(area / max(shaft * shaft, 1e-6), 4),
        'peak_offset_mm': round(px2mm(float(win[int(np.argmax(dv))] - T)), 1),
        'flags': flags,
    }


def corridor_distance(area_mm2: float, thr: dict) -> float:
    """Насколько измерение вышло за коридор нормы, мм². Внутри коридора — 0."""
    return float(max(thr['area_lo'] - area_mm2, area_mm2 - thr['area_hi'], 0.0))


def geometry_score(area_mm2: float, thr: dict,
                   relative: float | None = None,
                   area_over_shaft2: float | None = None) -> float:
    """Нормированный выход за коридор: больше = вероятнее нарушение.

    Пере- и недоротация нормируются на свою границу, чтобы асимметричный
    коридор не перекашивал скор в одну сторону.
    """
    lo, hi = float(thr['area_lo']), float(thr['area_hi'])
    kind = thr.get('score', 'area')
    if kind == 'relative' and relative is not None:
        # относительный выступ: те же границы, пересчитанные как доля
        # (calibrate кладёт relative_lo/hi при выборе score=relative)
        lo = float(thr.get('relative_lo', lo))
        hi = float(thr.get('relative_hi', hi))
        val = float(relative)
    elif kind == 'area_over_shaft2' and area_over_shaft2 is not None:
        lo = float(thr.get('aoss_lo', lo))
        hi = float(thr.get('aoss_hi', hi))
        val = float(area_over_shaft2)
    else:
        val = float(area_mm2)
    under = max(lo - val, 0.0) / max(lo, 1e-6)
    over = max(val - hi, 0.0) / max(hi, 1e-6)
    return float(max(under, over))


def detect(img: np.ndarray, side: str, appearance_p: float | None = None,
           thr: dict | None = None, T: int | None = None) -> dict:
    """Ротация одного кадра бедра.

    T (верхушка большого вертела) берётся из hip_roi/kit2, если не передан:
    тот же ориентир, от которого модуль ROI считает верхний отступ, так что
    два критерия бедра опираются на одну и ту же анатомию, а не на две разные.
    """
    t = {**DEFAULTS, **(thr or {})}
    if T is None:
        T = measure_roi_margins(img, side, 'kit2').get('T_px')
    if T is None:
        return {'violated': None, 'text': 'ротация: верхушка большого вертела не найдена',
                'flags': ['rotation:no_anchor'], 'confidence': 'low'}

    f = lesser_trochanter_bulge(img, side, int(T), t['neck_mm'], t['fit_mm'])
    if f is None:
        return {'violated': None, 'text': 'ротация: медиальный контур не прослеживается',
                'flags': ['rotation:no_contour'], 'confidence': 'low'}

    flags = list(f.pop('flags'))
    # Неуверенное измерение: не объявляем переротацию «по отсутствию вертела».
    if 'rotation:short_window' in flags or 'rotation:neck_not_reached' in flags:
        dist = corridor_distance(f['area_mm2'], t)
        return {**f, 'corridor_distance': round(dist, 1),
                'geometry_score': round(geometry_score(
                    f['area_mm2'], t, f.get('relative'), f.get('area_over_shaft2')), 4),
                'appearance_p': None if appearance_p is None else round(float(appearance_p), 3),
                'violated': None,
                'text': 'ротация: измерение ненадёжно (короткое окно или шейка не достигнута)',
                'confidence': 'low', 'flags': flags, 'T_px': int(T)}

    dist = corridor_distance(f['area_mm2'], t)   # мм² — для cnn_qc и probability
    score = geometry_score(f['area_mm2'], t, f.get('relative'), f.get('area_over_shaft2'))
    kind = t.get('score', 'area')
    if kind == 'relative' and 'relative_lo' in t:
        geo_bad = (f['relative'] < t['relative_lo'] or f['relative'] > t['relative_hi'])
        low = f['relative'] < t['relative_lo']
    elif kind == 'area_over_shaft2' and 'aoss_lo' in t:
        geo_bad = (f['area_over_shaft2'] < t['aoss_lo']
                   or f['area_over_shaft2'] > t['aoss_hi'])
        low = f['area_over_shaft2'] < t['aoss_lo']
    else:
        geo_bad = dist > 0.0
        low = f['area_mm2'] < t['area_lo']
    if appearance_p is not None and (appearance_p > t['appearance']) != geo_bad:
        flags.append('rotation:estimates_disagree')

    if geo_bad:
        label = ('переротация: малый вертел почти не выступает'
                 if low else
                 'недоротация: малый вертел выступает слишком сильно')
        text = (f'{label} (выступ {f["area_mm2"]:.0f} мм², '
                f'норма {t["area_lo"]:.0f}-{t["area_hi"]:.0f})')
    else:
        text = ''
    return {**f, 'corridor_distance': round(dist, 1),
            'geometry_score': round(score, 4),
            'appearance_p': None if appearance_p is None else round(float(appearance_p), 3),
            'violated': bool(geo_bad), 'text': text,
            # уверенность низкая всегда: разделяющая способность критерия
            # измерена и невысока, обещать большего нельзя
            'confidence': 'low', 'flags': flags, 'T_px': int(T)}
