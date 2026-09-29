# -*- coding: utf-8 -*-
"""Сборка вариантов алгоритма («комплекты») и единая точка входа.

    kit0 — без ориентиров: только длина сканирования H·s (контрольная точка)
    kit1 — «Профили»:  L2 + T1 + низ по седалищной кости (B1 — в диагностике)
    kit2 — «Контур»:   L4 -> T3 -> низ по седалищной кости I1 -> L3  (основной)
    kit3 — «Бедро отдельно от таза»: водораздел -> ось диафиза -> T3/T5 -> низ по I1
    custom — любая комбинация L*/T*/B* из landmarks.py

Отступы — как на рисунке 6 ТЗ, от области интереса до границы поля
сканирования (ответ организаторов, 27.09.2026): сверху — от верхушки большого
вертела, сбоку — от наружного контура бедренной кости, снизу — от нижней точки
седалищной кости. До этого ответа низ ROI ставился на уровне малого вертела
(якорь T + 50 мм), в среднем на 20 мм выше седалищной кости.

Все комплекты возвращают словарь одной формы (см. _result). Координаты в
результате — в ИСХОДНОЙ ориентации кадра.
"""
from __future__ import annotations

import numpy as np

from . import landmarks as lm
from .geometry import (BOTTOM_MM, D_TB_MM, D_TI_MM, D_TI_RANGE_MM, LAT_MM,
                       MM_PER_PX, TOP_MM, bone_mask, detect_dual_femur, mm2px,
                       px2mm, to_lateral_left, to_uint8, x_to_original)
from .separate import Rotation, separate_femur, shaft_axis

METHODS = ('kit0', 'kit1', 'kit2', 'kit3', 'custom')


# --------------------------------------------------------------------------- #
#  Общий результат
# --------------------------------------------------------------------------- #
def _fmt_cm(mm):
    return '—' if mm is None else f'{mm / 10:.1f} см'


def _result(method, side, shape, T, B, L, apex, flags, diag):
    h, w = shape
    m_top = None if T is None else px2mm(T)
    m_bottom = None if B is None else px2mm(h - B)
    m_lat = None if L is None else px2mm(L)
    top_ok = None if m_top is None else bool(m_top >= TOP_MM)
    bottom_ok = None if m_bottom is None else bool(m_bottom >= BOTTOM_MM)
    lat_ok = None if m_lat is None else bool(m_lat >= LAT_MM)
    oks = [o for o in (top_ok, bottom_ok, lat_ok) if o is not None]
    if any(o is False for o in oks):
        roi_ok = False
    elif len(oks) == 3:
        roi_ok = True
    else:
        roi_ok = None
    if roi_ok is True and 'measure:implausible_shaft' in flags:
        roi_ok = None                 # «норму» по недостоверной маске не выдаём

    parts = []
    if top_ok is False:
        parts.append(f'сверху {_fmt_cm(m_top)} < {TOP_MM / 10:.0f} см')
    if bottom_ok is False:
        parts.append(f'снизу {_fmt_cm(m_bottom)} < {BOTTOM_MM / 10:.0f} см')
    if lat_ok is False:
        parts.append(f'латерально {_fmt_cm(m_lat)} < {LAT_MM / 10:.0f} см')
    if roi_ok is False:
        text = 'ROI: ' + '; '.join(parts)
    elif roi_ok is None:
        text = 'ROI: не определено (' + ', '.join(flags) + ')'
    else:
        text = ''

    return {
        'method': method, 'side': side, 'H': int(h), 'W': int(w), 'scale_mm_per_px': MM_PER_PX,
        'T_px': None if T is None else int(T),
        'B_px': None if B is None else int(B),
        'L_px': None if L is None else int(x_to_original(L, w, side)),
        'apex_xy': None if apex is None else (int(x_to_original(apex[1], w, side)), int(apex[0])),
        'm_top_mm': None if m_top is None else round(m_top, 1),
        'm_bottom_mm': None if m_bottom is None else round(m_bottom, 1),
        'm_lat_mm': None if m_lat is None else round(m_lat, 1),
        'top_ok': top_ok, 'bottom_ok': bottom_ok, 'lat_ok': lat_ok, 'roi_ok': roi_ok,
        'flags': list(dict.fromkeys(flags)),
        'violation_text': text,
        'diag': diag,
    }


def _prepare(img, side):
    """Кадр и маска кости в нормализованной ориентации + проверка «две ноги»."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError('ожидается двумерный массив пикселей')
    norm = to_lateral_left(to_uint8(img), side)
    mask = to_lateral_left(bone_mask(img), side)
    flags = []
    if not mask.any():
        flags.append('mask:empty')
    return norm, mask, flags


def _ischium_B(img, side, norm, mask, T, h, flags, diag):
    """Низ ROI по рисунку 6 ТЗ: нижняя точка седалищной кости (landmarks.I1).

    Кандидат принимается, только если лежит на анатомически возможном
    расстоянии от верхушки вертела (D_TI_RANGE_MM; по ручной разметке 52–83 мм).
    Иначе — повтор по строгой маске: на полутоновых кадрах слабый порог
    сливает кость с мягкими тканями, и «кончик» уходит по их границе вниз. Если
    не помог и он — анатомический запас T + D_TI_MM с флагом: вердикт по низу
    на таком кадре читать как «не уверены».

    Отступ меряется до нижнего края кадра, а не до незасканированного выреза в
    углу поля: на рисунке 6 нижняя стрелка идёт мимо выреза до края поля.
    """
    lo, hi = D_TI_RANGE_MM
    tried = []
    for strict in (False, True):
        m = to_lateral_left(bone_mask(img, weak_frac=1.0), side) if strict else mask
        y, info = lm.I1_ischium_bottom(norm, m, T)
        tried.append({'strict': strict, 'y': y, **info})
        if y is not None and lo <= px2mm(y - T) <= hi:
            if strict:
                flags.append('I:strict_mask')
            diag['I'] = tried
            return min(int(y), h)
    flags.append('I:anatomical_prior')
    diag['I'] = tried
    return min(int(T) + mm2px(D_TI_MM), h)


SHAFT_MM_RANGE = (20.0, 48.0)      # физически возможная ширина диафиза бедра


def _check_shaft_plausible(norm, mask, flags, diag):
    """Внутренняя линейка: ширина диафиза вне физических пределов — маска или
    масштаб не в порядке, «норму» такому кадру выдавать нельзя.

    Вердикт при этом не переворачивается: найденное нарушение остаётся
    нарушением (обрезанный кадр как раз и даёт широкий «диафиз»), а вот
    «всё в порядке» превращается в «не определено» (см. _result).
    """
    sw = lm.shaft_width_mm(norm, mask)
    diag['shaft_width_mm'] = sw
    lo, hi = SHAFT_MM_RANGE
    if sw is None or not (lo <= sw <= hi):
        flags.append('measure:implausible_shaft')
    return sw


def _check_shaft_visible(T, B, h, flags, slack_mm: float = 40.0):
    """Кадр длиннее T + d_TB + slack, а диафиз «не найден» — значит, маска
    подозрительна (мягкие ткани слились с костью), а не кадр короткий.
    Вердикт по низу — «не определено», а не ложное нарушение."""
    if B is not None and B >= h and any(f in flags for f in ('B:diaphysis_not_reached', 'B:not_stabilised')):
        if h - T >= mm2px(D_TB_MM + slack_mm):
            flags.append('mask:suspect_wide_shaft')
            return None
    return B


def _finish_B(T, B1, B2, h, flags, diag, agree_mm=10.0):
    """Консенсус B1/B2 -> B4 запас -> нижняя граница по анатомическому прибору."""
    if B1 >= h and (B2 is None or B2 >= h):
        return h
    if B1 >= h and B2 is not None:
        flags.append('B:single_source_B2')
        B = B2
    elif B2 is None:
        flags.append('B:single_source_B1')
        B = B1
    elif B2 >= h:
        flags.append('B:single_source_B1')
        B = B1
    elif abs(B1 - B2) <= mm2px(agree_mm):
        B = int(round((B1 + B2) / 2))
    else:
        B, info4 = lm.B4_anatomical_offset(T)
        flags += info4['flags'] + ['B:B1_B2_disagree']
    b_min = T + mm2px(0.8 * D_TB_MM)
    if B < b_min:
        flags.append('B:clamped_to_prior')
        B = b_min
    return min(B, h)


# --------------------------------------------------------------------------- #
#  Длина поля сканирования — критерий так, как его ставит эксперт
# --------------------------------------------------------------------------- #
def scan_field(img) -> dict:
    """Хватает ли длины поля, чтобы отступы 3 см сверху и снизу вообще поместились.

    Нужная длина — 3 см + расстояние «верхушка большого вертела — нижняя
    точка седалищной кости» (медиана по обучающему набору, 69 мм) + 3 см =
    129 мм. Порог не подбирался по меткам эксперта: он собран из ТЗ и
    анатомии. С разметкой эксперта это правило согласуется намного лучше, чем
    отступы от найденных ориентиров по рисунку 6 (F1 0.71 против 0.35 на
    150 кадрах): эксперт, судя по всему, судит о поле целиком, а «норму»
    ставит и при 1.7-3.0 см под седалищной костью (DATASET.md).
    """
    h = int(np.asarray(img).shape[0])
    length = px2mm(h)
    need = TOP_MM + D_TI_MM + BOTTOM_MM
    return {'scan_length_mm': round(float(length), 1), 'min_length_mm': round(float(need), 1),
            'field_ok': bool(length >= need)}


# --------------------------------------------------------------------------- #
#  kit0 — только длина сканирования
# --------------------------------------------------------------------------- #
def kit0(img, side):
    """Минимальная длина кадра, при которой критерий вообще выполним:
    TOP + d_TI + BOTTOM (d_TI — медиана расстояния «верхушка вертела ->
    седалищная кость»). Ни верх, ни латераль не проверяет."""
    h, w = np.asarray(img).shape
    length = px2mm(h)
    min_len = TOP_MM + D_TI_MM + BOTTOM_MM
    ok = bool(length >= min_len)
    return {
        'method': 'kit0', 'side': side, 'H': int(h), 'W': int(w), 'scale_mm_per_px': MM_PER_PX,
        'T_px': None, 'B_px': None, 'L_px': None, 'apex_xy': None,
        'm_top_mm': None, 'm_bottom_mm': round(length - min_len + BOTTOM_MM, 1), 'm_lat_mm': None,
        'top_ok': None, 'bottom_ok': ok, 'lat_ok': None, 'roi_ok': ok,
        'flags': ['proxy:scan_length_only'],
        'violation_text': '' if ok else f'ROI: длина сканирования {length / 10:.1f} см < {min_len / 10:.0f} см',
        'diag': {'scan_length_mm': round(length, 1), 'min_length_mm': min_len},
    }


# --------------------------------------------------------------------------- #
#  kit1 — «Профили»
# --------------------------------------------------------------------------- #
def kit1(img, side):
    norm, mask, flags = _prepare(img, side)
    h, w = mask.shape
    diag = {}
    L, iL = lm.L2_jump_guard(mask)
    flags += iL['flags']
    if L is None:
        return _result('kit1', side, mask.shape, None, None, None, None, flags, diag)
    T, iT = lm.T1_wide_part(mask, L, iL['yL'])
    flags += iT['flags']
    B1, iB1 = lm.B1_width_stabilisation(mask, T)     # оставлен для сравнения
    B = _ischium_B(img, side, norm, mask, T, h, flags, diag)
    _check_shaft_plausible(norm, mask, flags, diag)
    diag.update(L=iL, T=iT, B1=iB1)
    return _result('kit1', side, mask.shape, T, B, L, iT['apex'], flags, diag)


# --------------------------------------------------------------------------- #
#  kit2 — «Контур» (основной)
# --------------------------------------------------------------------------- #
def kit2(img, side):
    norm, mask, flags = _prepare(img, side)
    h, w = mask.shape
    diag = {}
    L, iL, T, iT = _find_LT(mask, flags)
    if L is None:
        return _result('kit2', side, mask.shape, None, None, None, None, flags, diag)
    if T == 0:
        # контур ушёл по тазу до верхнего края: на полутоновых кадрах слабый
        # порог сливает кость с мягкими тканями. Повторяем по строгой маске.
        strict = to_lateral_left(bone_mask(img, weak_frac=1.0), side)
        f2 = []
        L2, iL2, T2v, iT2 = _find_LT(strict, f2)
        if T2v is not None and T2v > 0:
            flags.append('mask:strict_fallback')
            mask, L, iL, T, iT = strict, L2, iL2, T2v, iT2
            flags += f2
            norm = norm
    if T is None:
        return _result('kit2', side, mask.shape, None, None, L, None, flags, diag)
    B = _ischium_B(img, side, norm, mask, T, h, flags, diag)
    L3, iL3 = lm.L3_in_range(mask, T, B)
    if L3 is not None:
        L = L3
    _check_shaft_plausible(norm, mask, flags, diag)
    diag.update(L=iL, T=iT, L3=iL3)
    return _result('kit2', side, mask.shape, T, B, L, iT['apex'], flags, diag)


def _find_LT(mask, flags):
    """Латеральный край и верхушка вертела на данной маске."""
    L, iL = lm.L4_bulge_vs_shaft_line(mask)
    flags += iL['flags']
    if L is None:
        return None, iL, None, {}
    T, iT = lm.T3_contour_first_min(mask, L, iL['yL'])
    if T is None or T == 0:
        # контур без седловины (None) или ушёл по тазу до верхнего края (0):
        # запас T2 — верх компоненты в полосе 25 мм от латерального края.
        # Если и он даёт 0 — вертел действительно срезан краем кадра.
        T2, iT2 = lm.T2_band_component_top(mask, L, iL['yL'])
        if T2 is not None and (T is None or T2 > 0):
            flags.append('T:fallback_T2')
            T, iT = T2, {**iT, **iT2}
    flags += iT['flags']
    return L, iL, T, iT


# --------------------------------------------------------------------------- #
#  kit3 — «Бедро отдельно от таза»
# --------------------------------------------------------------------------- #
def kit3(img, side):
    norm, mask, flags = _prepare(img, side)
    h, w = mask.shape
    diag = {}
    if not mask.any():
        return _result('kit3', side, mask.shape, None, None, None, None, flags, diag)
    femur, iS = separate_femur(mask)
    flags += iS['flags']
    phi, iA = shaft_axis(femur)
    flags += iA['flags']
    rot = Rotation(mask.shape, phi)
    Fr = rot.apply(femur)
    diag.update(sep=iS, axis={'phi_deg': round(phi, 2), **iA})

    Lr, iLr = lm.L2_jump_guard(Fr)
    if Lr is None:
        return _result('kit3', side, mask.shape, None, None, None, None, flags + iLr['flags'], diag)
    Tr, iTr = lm.T3_contour_first_min(Fr, Lr, iLr['yL'])
    T5r, iT5 = lm.T5_component_top(Fr, Lr)
    if Tr is None or (Tr == 0 and T5r is not None and T5r > 0):
        Tr, iTr = T5r, {**iTr, **iT5}
        flags.append('T:fallback_T5')
    if Tr is None:
        return _result('kit3', side, mask.shape, None, None, None, None, flags + iTr['flags'], diag)
    if T5r is not None and abs(T5r - Tr) > mm2px(5):
        flags.append('T:T3_T5_disagree')
    flags += iTr['flags']
    Br, iBr = lm.B3_distance_ridge(Fr, Tr)           # оставлен для сравнения

    # обратно в исходную систему координат
    apex = rot.point_to_orig(*iTr['apex'])
    T = min(max(apex[0], 0), h - 1)
    B = _ischium_B(img, side, norm, mask, T, h, flags, diag)
    ys, xs = np.nonzero(femur)
    sel = (ys >= T) & (ys <= min(B, h - 1))
    L = int(xs[sel].min()) if sel.any() else int(xs.min())
    _check_shaft_plausible(norm, mask, flags, diag)
    diag.update(L=iLr, T=iTr, T5=iT5, B3=iBr, B3_rotated=Br, rot_shape=Fr.shape)
    return _result('kit3', side, mask.shape, T, B, L, (T, apex[1]), flags, diag)


# --------------------------------------------------------------------------- #
#  custom — произвольная комбинация
# --------------------------------------------------------------------------- #
def kit_custom(img, side, L='L2', T='T3', B='B1'):
    _, mask, flags = _prepare(img, side)
    h, w = mask.shape
    diag = {'combo': (L, T, B)}
    Lv, iL = lm.L_VARIANTS[L](mask)
    flags += iL['flags']
    if Lv is None:
        return _result('custom', side, mask.shape, None, None, None, None, flags, diag)
    Tv, iT = lm.T_VARIANTS[T](mask, Lv, iL['yL'])
    flags += iT['flags']
    if Tv is None:
        return _result('custom', side, mask.shape, None, None, Lv, None, flags, diag)
    if B == 'B4':
        Bv, iB = lm.B4_anatomical_offset(Tv)
    else:
        Bv, iB = lm.B_VARIANTS[B](mask, Tv)
    flags += iB['flags']
    if Bv is None:
        Bv, iB4 = lm.B4_anatomical_offset(Tv)
        flags += iB4['flags']
    diag.update(L=iL, T=iT, B=iB)
    return _result('custom', side, mask.shape, Tv, min(Bv, h), Lv, iT['apex'], flags, diag)


# --------------------------------------------------------------------------- #
#  Точка входа
# --------------------------------------------------------------------------- #
def measure_roi_margins(img, side: str, method: str = 'kit2', **combo):
    """Отступы ROI для кадра бедра.

    img    — двумерный массив пикселей (uint8 или любой числовой), кость светлая
    side   — 'lh' | 'rh' (от region_clf)
    method — 'kit0' | 'kit1' | 'kit2' | 'kit3' | 'custom' (+ L=, T=, B=)
    """
    if method == 'kit0':
        return kit0(img, side)
    if method == 'kit1':
        return kit1(img, side)
    if method == 'kit2':
        return kit2(img, side)
    if method == 'kit3':
        return kit3(img, side)
    if method == 'custom':
        return kit_custom(img, side, **combo)
    raise ValueError(f'неизвестный метод {method!r}; доступны {METHODS}')
