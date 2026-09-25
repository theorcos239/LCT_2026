# -*- coding: utf-8 -*-
"""Подпрограммы поиска ориентиров ROI. По одной функции на каждый вариант.

Все функции работают в НОРМАЛИЗОВАННОЙ ориентации: латеральный край бедра
слева (x = 0), верх кадра — y = 0. Возвращают (значение, info), где info —
словарь с флагами и диагностикой. Значение None означает «не найдено».

Ориентиры:
    L — столбец самой латеральной точки бедра          (L1, L2, L3)
    T — строка верхушки большого вертела               (T1, T2, T3, T5)
    B — низ ROI, конец вертельной массы                (B1, B2, B3, B4, B5)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .geometry import (D_TB_MM, EIGHT, component_containing, lateral_profile,
                       medial_profile, mm2px, moving_average, px2mm,
                       top_profile, trace_contour, unscanned_region,
                       width_profile)


def _profiles(mask: np.ndarray):
    xl = lateral_profile(mask)
    xm = medial_profile(mask, xl)
    w = width_profile(xl, xm, mask.shape[1])
    return xl, xm, w


# =========================================================================== #
#  L — латеральный край
# =========================================================================== #
def L1_lower_fraction(mask: np.ndarray, frac: float = 0.30):
    """L1. Минимум латерального профиля по строкам ниже frac·H.

    Верхние 30 % кадра отсекаются, чтобы не зацепить крыло подвздошной кости.
    Слабость: доля кадра, а не анатомия — на длинных кадрах вертел может
    оказаться выше отсечки.
    """
    h, w = mask.shape
    xl = lateral_profile(mask)
    y0 = int(h * frac)
    rows = np.flatnonzero(xl[y0:] < w) + y0
    if rows.size == 0:
        return None, {'flags': ['L:no_bone'], 'yL': None}
    L = int(xl[rows].min())
    yL = int(rows[np.flatnonzero(xl[rows] == L)[0]])     # верхняя строка минимума
    return L, {'flags': [], 'yL': yL}


def L2_jump_guard(mask: np.ndarray, jump_mm: float = 8.0):
    """L2. Идём снизу вверх по латеральному профилю, пока он непрерывен.

    Останавливаемся на пустой строке или на скачке профиля латерально больше
    jump_mm между соседними строками (подвздошная кость через мягкотканный
    промежуток). L — минимум на пройденном участке.
    """
    h, w = mask.shape
    xl = lateral_profile(mask)
    rows = np.flatnonzero(xl < w)
    if rows.size == 0:
        return None, {'flags': ['L:no_bone'], 'yL': None}
    jump = mm2px(jump_mm)
    y = int(rows.max())
    L, yL, prev = int(xl[y]), y, int(xl[y])
    stop_reason = 'top'
    while y > 0:
        nxt = int(xl[y - 1])
        if nxt >= w:
            stop_reason = 'empty_row'
            break
        if nxt < prev - jump:
            stop_reason = 'lateral_jump'
            break
        y -= 1
        prev = nxt
        if nxt <= L:                         # <= : берём самую верхнюю строку минимума
            L, yL = nxt, y
    return L, {'flags': [], 'yL': yL, 'stop_row': y, 'stop_reason': stop_reason}


def L3_in_range(mask: np.ndarray, T: int, B: int):
    """L3. Минимум латерального профиля строго внутри диапазона ROI [T, B]."""
    h, w = mask.shape
    xl = lateral_profile(mask)
    lo, hi = max(0, int(T)), min(h - 1, int(B))
    if hi < lo:
        return None, {'flags': ['L:empty_range'], 'yL': None}
    seg = xl[lo:hi + 1]
    rows = np.flatnonzero(seg < w)
    if rows.size == 0:
        return None, {'flags': ['L:no_bone_in_range'], 'yL': None}
    L = int(seg[rows].min())
    yL = lo + int(rows[np.flatnonzero(seg[rows] == L)[0]])
    return L, {'flags': [], 'yL': yL}


def L4_bulge_vs_shaft_line(mask: np.ndarray, fit_frac: float = 0.30, min_fit_mm: float = 15.0,
                           bulge_mm: float = 3.0, jump_mm: float = 8.0):
    """L4. Большой вертел как латеральная выпуклость относительно прямой диафиза.

    Прямая (Тейл—Сен) по латеральному профилю нижних fit_frac строк с костью;
    выше неё ищем строки, где профиль латеральнее прямой больше чем на
    bulge_mm — это вертел. L — самая латеральная точка выпуклости, yL — её
    (верхняя) строка. Нужен потому, что у отведённого бедра самая латеральная
    точка всего бедра — низ диафиза, а не вертел (L2 там стартует контур с
    нижнего края кадра). Без выпуклости — запас L2.
    """
    h, w = mask.shape
    xl = lateral_profile(mask)
    rows = np.flatnonzero(xl < w)
    if rows.size == 0:
        return None, {'flags': ['L:no_bone'], 'yL': None}
    y_last, y_first = int(rows.max()), int(rows.min())
    n = y_last - y_first + 1
    k = max(mm2px(min_fit_mm), int(round(fit_frac * n)))
    fit_rows = np.arange(max(y_first, y_last - k + 1), y_last + 1)
    fit_rows = fit_rows[xl[fit_rows] < w]
    if fit_rows.size < 3:
        L, info = L2_jump_guard(mask, jump_mm)
        info['flags'] = info['flags'] + ['L:L4_fallback_L2']
        return L, info
    a, b = _theil_sen(fit_rows.astype(float), xl[fit_rows].astype(float))
    jump, bulge = mm2px(jump_mm), bulge_mm / px2mm(1)
    best_dev, L, yL, prev = 0.0, None, None, int(xl[fit_rows.min()])
    y = int(fit_rows.min())
    while y > 0:                                  # вверх от зоны подгонки, как в L2
        nxt = int(xl[y - 1])
        if nxt >= w or nxt < prev - jump:
            break
        y -= 1
        prev = nxt
        dev = nxt - (a * y + b)                   # < 0: латеральнее прямой
        if dev <= -bulge and (L is None or nxt <= L):
            L, yL = nxt, y
        best_dev = min(best_dev, dev)
    info = {'flags': [], 'line': (a, b), 'max_bulge_mm': round(px2mm(-best_dev), 1), 'stop_row': y}
    if L is None:
        L, info2 = L2_jump_guard(mask, jump_mm)
        info2['flags'] = info2['flags'] + ['L:no_bulge_fallback_L2']
        info2['max_bulge_mm'] = info['max_bulge_mm']
        return L, info2
    info['yL'] = yL
    return L, info


# =========================================================================== #
#  T — верхушка большого вертела
# =========================================================================== #
def T1_wide_part(mask: np.ndarray, L: int, yL: int, delta_mm: float = 5.0):
    """T1. Верх «широкой части»: идём вверх от yL, пока x_lat(y) <= L + delta.

    Даёт не верхушку, а верх широкой части вертела — систематически ниже
    истинной верхушки, т.е. ЗАВЫШАЕТ верхний отступ. Нижняя оценка.
    """
    xl = lateral_profile(mask)
    d = mm2px(delta_mm)
    y = int(yL)
    while y > 0 and xl[y - 1] <= L + d:
        y -= 1
    flags = ['T:at_top_edge'] if y == 0 else []
    return y, {'flags': flags, 'apex': (y, int(xl[y]) if xl[y] < mask.shape[1] else L)}


def T2_band_component_top(mask: np.ndarray, L: int, yL: int, band_mm: float = 25.0):
    """T2. Верхняя точка кости в полосе столбцов [L, L + band_mm].

    Полоса вырезается из маски, берётся её 8-связная компонента, содержащая
    точку (yL, L): подвздошная кость, отделённая промежутком, в неё не попадает,
    а связь через головку/вертлужную впадину разрывается краем полосы.
    """
    h, w = mask.shape
    x1 = min(w, L + mm2px(band_mm))
    sub = mask[:, L:x1]
    comp = component_containing(sub, (yL, 0))
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return None, {'flags': ['T:band_empty'], 'apex': None}
    T = int(ys.min())
    x_apex = L + int(xs[ys == T].min())
    flags = ['T:at_top_edge'] if T == 0 else []
    return T, {'flags': flags, 'apex': (T, x_apex)}


def T3_contour_first_min(mask: np.ndarray, L: int, yL: int, smooth_mm: float = 4.0,
                         drop_mm: float = 3.0, medial_shift_mm: float = 2.0,
                         plateau_mm: float = 12.0, flat_tol_mm: float = 3.0):
    """T3. Трассировка внешнего контура от точки L вверх; верхушка = первый
    локальный минимум y с выраженностью >= drop_mm.

    Контур поднимается по латеральному краю вертела, достигает верхушки и
    опускается в седловину шейки. Держим бегущий минимум y (сглаженного окном
    smooth_mm вдоль дуги). Два правила остановки, срабатывает первое:
      (a) контур поднялся над минимумом на drop_mm И сместился медиально на
          medial_shift_mm — обычная седловина шейки;
      (b) контур ушёл медиально от точки минимума на plateau_mm, оставаясь в
          пределах flat_tol_mm по высоте — плоский верх вертела, который в
          проекции переходит прямо в таз (приведённое бедро, седловины нет).
    В обоих случаях верхушка — бегущий минимум. Если контур упёрся в y = 0
    раньше — вертел срезан краем: T = 0 и флаг.
    """
    h, w = mask.shape
    comp = component_containing(mask, (yL, L))
    C = trace_contour(comp, (yL, L))
    n = len(C)
    if n < 8:
        return None, {'flags': ['T:contour_too_short'], 'apex': None}

    # направление обхода, в котором y сначала убывает (вверх)
    k = min(10, n - 1)
    fwd = C[1:k + 1, 0].mean()
    bwd = C[-k:, 0].mean()
    if bwd < fwd:
        C = np.concatenate((C[:1], C[1:][::-1]))
    ys = moving_average(C[:, 0], 2 * mm2px(smooth_mm) + 1)
    xs = C[:, 1].astype(float)

    drop = drop_mm / px2mm(1)
    shift = medial_shift_mm / px2mm(1)
    plateau = plateau_mm / px2mm(1)
    flat = flat_tol_mm / px2mm(1)
    run_min, run_idx = ys[0], 0
    for i in range(1, n):
        if C[i, 0] == 0:
            return 0, {'flags': ['T:at_top_edge'], 'apex': (0, int(C[i, 1])), 'contour_len': n}
        if ys[i] < run_min:
            run_min, run_idx = ys[i], i
            continue
        rise, travel = ys[i] - run_min, xs[i] - xs[run_idx]
        rule = 'saddle' if (rise >= drop and travel >= shift) else                'plateau' if (travel >= plateau and rise <= flat) else None
        if rule:
            # уточняем по несглаженному контуру в окрестности бегущего минимума
            half = mm2px(smooth_mm)
            lo, hi = max(0, run_idx - half), min(n, run_idx + half + 1)
            j = lo + int(np.argmin(C[lo:hi, 0]))
            return int(C[j, 0]), {'flags': [], 'apex': (int(C[j, 0]), int(C[j, 1])),
                                  'contour_len': n, 'arc_idx': j, 'rule': rule}
    return None, {'flags': ['T:no_local_min'], 'apex': None, 'contour_len': n}


def T5_component_top(femur: np.ndarray, L: int):
    """T5. Верхняя точка латеральной половины компоненты бедра.

    Только для отделённого от таза бедра (комплект 3): после разделения головка
    отходит к тазу, и самая верхняя точка латеральной половины — верхушка
    большого вертела. Если разделение прошло по головке, а не по шейке —
    даст верх головки (завышает нарушение), поэтому в комплекте 3 это
    перекрёстная проверка к T3, а не основной ответ.
    """
    ys, xs = np.nonzero(femur)
    if ys.size == 0:
        return None, {'flags': ['T:no_femur'], 'apex': None}
    x_mid = L + (int(xs.max()) - L) // 2
    sel = (xs >= L) & (xs <= x_mid)
    if not sel.any():
        return None, {'flags': ['T:empty_half'], 'apex': None}
    T = int(ys[sel].min())
    x_apex = int(xs[sel][ys[sel] == T].min())
    flags = ['T:at_top_edge'] if T == 0 else []
    return T, {'flags': flags, 'apex': (T, x_apex)}


# =========================================================================== #
#  B — низ ROI
# =========================================================================== #
def _shaft_reference(w: np.ndarray, T: int, ref_mm: float, max_shaft_mm: float):
    """Опорная ширина диафиза по нижним ref_mm строкам с костью."""
    rows = np.flatnonzero(w > 0)
    rows = rows[rows >= T]
    if rows.size == 0:
        return None, None, ['B:no_bone_below_T']
    y_last = int(rows.max())
    seg = w[max(T, y_last - mm2px(ref_mm) + 1):y_last + 1]
    seg = seg[seg > 0]
    w_ref = float(np.median(seg))
    flags = []
    if px2mm(w_ref) > max_shaft_mm:
        flags.append('B:diaphysis_not_reached')
    return w_ref, y_last, flags


def B1_width_stabilisation(mask: np.ndarray, T: int, ref_mm: float = 15.0,
                           slope_mm_per_cm: float = 2.0, run_mm: float = 20.0,
                           level_tol: float = 1.35, max_shaft_mm: float = 40.0,
                           min_tb_mm: float = 45.0):
    """B1. Стабилизация ширины бедра: первая строка ниже T + min_tb, с которой
    ширина на отрезке run_mm сужается медленнее slope_mm_per_cm И не превышает
    level_tol·w_ref (w_ref — медиана ширины по нижним ref_mm кадра).

    Критерий по наклону, а не по уровню: диафиз сужается непрерывно
    (~0.9 мм/см на выборке), поэтому «ширина <= 1.1·опорной» ставило B на
    2 см ниже истинного низа малого вертела. Межвертельная зона сужается
    5-9 мм/см, малый вертел — 4-5 мм/см, диафиз — < 1.5 мм/см.
    Окно run_mm = 20 мм: на плато самого малого вертела (~15 мм) ширина
    тоже локально постоянна, и окно в 10 мм ловило его вместо диафиза.
    Окно короче порога 30 мм, поэтому ложного нарушения дать не может.

    Защита от коротких кадров: если w_ref > max_shaft_mm, нижний край кадра
    ещё не диафиз (межвертельная зона 40-60 мм; сам диафиз в выборке
    26-36 мм) -> B = H, m_bottom = 0.
    """
    h, w_ = mask.shape
    xl, xm, w = _profiles(mask)
    w_ref, y_last, flags = _shaft_reference(w, T, ref_mm, max_shaft_mm)
    info = {'flags': flags, 'w_ref_mm': None if w_ref is None else round(px2mm(w_ref), 1)}
    if w_ref is None or 'B:diaphysis_not_reached' in flags:
        return h, info
    run = mm2px(run_mm)
    ws = moving_average(w.astype(float), mm2px(3.0) * 2 + 1)
    max_drop = slope_mm_per_cm * run_mm / 10.0 / px2mm(1)       # px за run_mm
    for y in range(int(T) + mm2px(min_tb_mm), y_last - run + 2):
        seg = w[y:y + run]
        if not (seg > 0).all():
            continue
        if (ws[y] - ws[y + run - 1]) <= max_drop and ws[y] <= level_tol * w_ref:
            return int(y), info
    info['flags'] = flags + ['B:not_stabilised']
    return h, info


def _theil_sen(y: np.ndarray, x: np.ndarray):
    """Детерминированная робастная прямая x = a·y + b (медиана попарных наклонов)."""
    n = len(y)
    if n < 2:
        return 0.0, float(x[0]) if n else 0.0
    ii, jj = np.triu_indices(n, k=1)
    dy = y[jj] - y[ii]
    ok = dy != 0
    slopes = (x[jj][ok] - x[ii][ok]) / dy[ok]
    a = float(np.median(slopes)) if slopes.size else 0.0
    b = float(np.median(x - a * y))
    return a, b


def B2_medial_line_deviation(mask: np.ndarray, T: int, fit_frac: float = 0.25, dev_mm: float = 2.0,
                             max_shaft_mm: float = 40.0, min_tb_mm: float = 45.0):
    """B2. Прямая по медиальному контуру диафиза (нижние fit_frac строк, Тейл—Сен);
    идём снизу вверх, B — первая строка, где контур ушёл медиальнее прямой
    больше чем на dev_mm (начался бугор малого вертела).

    Если отклонения нет вплоть до T + min_tb — малый вертел не виден
    (переротация): возвращает None, комплект берёт запасное правило B4.
    Известная слабость: без малого вертела первое отклонение — начало
    дуги Адамса/шейки, оно ВЫШЕ истинного низа ROI.
    """
    h, w_ = mask.shape
    xl, xm, w = _profiles(mask)
    w_ref, y_last, flags = _shaft_reference(w, T, 15.0, max_shaft_mm)
    info = {'flags': flags, 'w_ref_mm': None if w_ref is None else round(px2mm(w_ref), 1)}
    if w_ref is None or 'B:diaphysis_not_reached' in flags:
        return h, info
    y_min = int(T) + mm2px(min_tb_mm)
    n_rows = y_last - y_min + 1
    if n_rows < mm2px(10):
        info['flags'] = flags + ['B:too_short_for_fit']
        return h, info
    k = max(mm2px(10), int(round(fit_frac * n_rows)))
    fit_rows = np.arange(y_last - k + 1, y_last + 1)
    fit_rows = fit_rows[w[fit_rows] > 0]
    a, b = _theil_sen(fit_rows.astype(float), xm[fit_rows].astype(float))
    info.update(line=(a, b), fit_rows=(int(fit_rows.min()), int(fit_rows.max())))
    dev = mm2px(dev_mm)
    for y in range(y_last - k, y_min - 1, -1):
        if w[y] == 0:
            continue
        if xm[y] - (a * y + b) > dev:
            return int(y), info
    info['flags'] = flags + ['B:lesser_troch_not_visible']
    return None, info


def B3_distance_ridge(mask: np.ndarray, T: int, win_mm: float = 15.0, tol_mm: float = 1.5,
                      max_half_mm: float = 16.0, min_tb_mm: float = 45.0, edge_mm: float = 8.0):
    """B3. Гребень дистанционного преобразования: r(y) = max D по отрезку бедра
    в строке. На диафизе r(y) почти постоянна (естественное сужение ~0.45 мм/см
    по полуширине). B — первая строка ниже T + min_tb, где на окне win_mm
    разброс r меньше tol_mm и r <= max_half_mm.

    Не нуждается в явном медиальном контуре; отрезок бедра в строке берётся
    между x_lat и первым разрывом, поэтому седалищная кость не мешает.
    Последние edge_mm строк с костью не используются: если низ кости — срез
    (косой край повёрнутого кадра), D там занижена краем, а не анатомией.
    """
    h, w_ = mask.shape
    xl, xm, w = _profiles(mask)
    D = ndi.distance_transform_edt(mask)
    r = np.zeros(h)
    for y in range(h):
        if w[y] > 0:
            r[y] = D[y, xl[y]:xm[y]].max()
    rows = np.flatnonzero(w > 0)
    rows = rows[rows >= T]
    if rows.size == 0:
        return h, {'flags': ['B:no_bone_below_T']}
    y_last = int(rows.max()) - mm2px(edge_mm)
    win, tol, max_half = mm2px(win_mm), tol_mm / px2mm(1), max_half_mm / px2mm(1)
    for y in range(int(T) + mm2px(min_tb_mm), y_last - win + 2):
        seg = r[y:y + win]
        if (seg > 0).all() and (seg.max() - seg.min()) < tol and seg[0] <= max_half:
            x_ridge = xl[y] + int(np.argmax(D[y, xl[y]:xm[y]]))
            return int(y), {'flags': [], 'ridge_x': x_ridge, 'r_mm': round(px2mm(seg[0]), 1)}
    return h, {'flags': ['B:ridge_not_flat']}


def row_mass(img: np.ndarray, mask: np.ndarray, max_width_mm: float = 55.0):
    """«Масса» кости в строке: сумма яркости по сегменту бедра / 255.

    Это тон, а не силуэт: в bone map яркость пропорциональна поверхностной
    плотности, поэтому сумма по строке — эффективная толщина кости в этой
    строке. Строки, где сегмент шире max_width_mm, помечаются недостоверными:
    такой ширины у бедра ниже вертелов не бывает, значит медиальный край
    «ушёл» на таз через зону наложения.

    Строки, задетые незасканированным вырезом поля, тоже недостоверны:
    там сегмент обрезан границей поля, а не костью.

    Возвращает (mass, valid).
    """
    arr = np.asarray(img, dtype=np.float64)
    xl, xm, w = _profiles(mask)
    h, wd = mask.shape
    notch = unscanned_region(img)
    mass = np.zeros(h)
    cut = np.zeros(h, bool)
    for y in range(h):
        if w[y] > 0:
            a, b = xl[y], min(xm[y], wd)
            mass[y] = arr[y, a:b].sum() / 255.0
            lo, hi = max(0, a - 2), min(wd, b + 2)
            cut[y] = bool(notch[y, lo:hi].any())     # строку задел вырез поля
    valid = (w > 0) & (w <= mm2px(max_width_mm)) & ~cut
    return mass, valid


def B5_mass_minimum(img: np.ndarray, mask: np.ndarray, T: int, min_tb_mm: float = 35.0,
                    smooth_mm: float = 6.0, tail_mm: float = 10.0, max_width_mm: float = 55.0):
    """B5. Субтрохантерный уровень: минимум массы строки ниже вертелов.

    Вертельная масса добавляет кости, дальше книзу идёт сужение до самого
    узкого места бедра — субтрохантерного, ниже которого кортикал утолщается
    и масса снова растёт. Минимум массы и есть конец вертельной массы, то есть
    низ области интереса.

    Почему не «стабилизация ширины» (B1) и не «отклонение медиального контура»
    (B2): проксимальный диафиз расширяется кверху плавно, без излома, поэтому
    оба правила срабатывают на 2-4 см ниже конца вертельной массы (проверено
    на кадрах, где эксперт не видит нарушения, а ТЗ по старому B давало
    «снизу < 3 см»). Масса же имеет настоящий экстремум.

    Если минимум пришёлся на последние tail_mm достоверных строк, значит
    сужение ещё не пройдено — кадр обрезан выше субтрохантерного уровня:
    возвращается H (отступ снизу 0).
    """
    h = mask.shape[0]
    mass, valid = row_mass(img, mask, max_width_mm)
    y0 = max(int(T) + mm2px(min_tb_mm), 0)
    idx = np.flatnonzero(valid)
    idx = idx[idx >= y0]
    info = {'flags': [], 'n_valid': int(idx.size)}
    if idx.size < mm2px(tail_mm):
        info['flags'].append('B:no_valid_rows')
        return h, info
    ms = moving_average(mass, 2 * mm2px(smooth_mm) + 1)
    j = int(idx[np.argmin(ms[idx])])
    info.update(mass_min=round(float(ms[j]), 1), last_valid=int(idx.max()),
                mass_at_last=round(float(ms[idx.max()]), 1))
    if idx.max() - j < mm2px(tail_mm):
        info['flags'].append('B:narrowing_not_passed')
        return h, info
    return j, info


def B6_trochanteric_mass_end(img: np.ndarray, mask: np.ndarray, T: int, min_tb_mm: float = 25.0,
                             delta_frac: float = 0.05, run_mm: float = 6.0, smooth_mm: float = 6.0,
                             tail_mm: float = 10.0, max_width_mm: float = 55.0):
    """B6. Низ ROI = нижняя граница добавочной массы вертелов.

    Работает по тону, а не по силуэту. Масса строки (сумма яркости по сегменту
    бедра) вдоль бедра ведёт себя так: вертелы дают избыток кости, ниже них
    масса падает до субтрохантерного минимума, а дальше книзу медленно растёт
    вместе с толщиной кортикала. Значит:

      1. субтрохантерный минимум массы — опорный уровень «чистого» бедра;
      2. низ ROI — первая строка ВЫШЕ него, где масса устойчиво (run_mm)
         превысила этот уровень на delta_frac: там начинается вертельная масса.

    Почему не «стабилизация ширины» (B1) и не «минимум массы» (B5): силуэт
    проксимального диафиза расширяется кверху плавно, без излома, поэтому B1
    ставит низ ROI на 2-4 см ниже вертелов; сам минимум массы лежит ниже
    вертелов примерно на столько же. Проверено на кадрах, где эксперт не видит
    нарушения, а прежнее правило давало «снизу < 3 см».

    Если ниже минимума нет хотя бы tail_mm достоверных строк, сужение не
    пройдено — кадр обрезан выше субтрохантерного уровня: H (отступ снизу 0).
    """
    h = mask.shape[0]
    mass, valid = row_mass(img, mask, max_width_mm)
    ms = moving_average(mass, 2 * mm2px(smooth_mm) + 1)
    idx = np.flatnonzero(valid)
    idx = idx[idx >= int(T) + mm2px(min_tb_mm)]
    info = {'flags': [], 'n_valid': int(idx.size)}
    if idx.size < mm2px(tail_mm) * 2:
        info['flags'].append('B:no_valid_rows')
        return h, info
    y_min = int(idx[np.argmin(ms[idx])])
    info.update(y_min=y_min, mass_min=round(float(ms[y_min]), 1))
    if idx.max() - y_min < mm2px(tail_mm):
        info['flags'].append('B:narrowing_not_passed')       # кадр обрезан выше сужения
        return h, info
    limit = float(ms[y_min]) * (1.0 + delta_frac)
    info['limit'] = round(limit, 1)
    run = mm2px(run_mm)
    above = idx[idx < y_min]
    for y in above[::-1]:                                     # вверх от минимума
        seg = ms[max(0, y - run + 1):y + 1]
        if (seg > limit).all():
            info['mass_at_B'] = round(float(ms[y]), 1)
            return int(y), info
    info['flags'].append('B:no_trochanteric_mass')
    return h, info


def B7_medial_local_deviation(img: np.ndarray, mask: np.ndarray, T: int, dev_mm: float = 2.0,
                              fit_lo_mm: float = 8.0, fit_hi_mm: float = 45.0,
                              smooth_mm: float = 2.0, min_tb_mm: float = 25.0,
                              max_width_mm: float = 55.0):
    """B7. Низ ROI по медиальному контуру с ЛОКАЛЬНОЙ опорной прямой.

    Для каждой строки прямая (Тейл—Сен) строится по контуру на 8-45 мм ниже
    неё и сравнивается с самим контуром: выше низа вертелов контур отходит
    медиальнее прямой. B — верхняя строка непрерывного участка, где отклонение
    держится ниже dev_mm.

    Отличие от B2: там прямая одна на весь кадр и берётся по дистальному
    диафизу, а диафиз изогнут, поэтому экстраполяция вверх уходит латеральнее
    истинного контура и отклонение «обнаруживается» на 2-4 см ниже вертелов.
    """
    h = mask.shape[0]
    xl, xm, w = _profiles(mask)
    _, valid = row_mass(img, mask, max_width_mm)
    xs = moving_average(xm.astype(float), 2 * mm2px(smooth_mm) + 1)
    dev = dev_mm / px2mm(1)
    lo, hi, need = mm2px(fit_lo_mm), mm2px(fit_hi_mm), mm2px(15.0)
    info = {'flags': []}
    y0 = int(T) + mm2px(min_tb_mm)
    hits = []
    for y in range(y0, h):
        if not valid[y]:
            continue
        rows = np.flatnonzero(valid[y + lo:min(h, y + hi)]) + y + lo
        if rows.size < need:
            continue
        a, b = _theil_sen(rows.astype(float), xs[rows])
        hits.append((y, xs[y] - (a * y + b)))
    if not hits:
        info['flags'].append('B:no_fit_rows')
        return h, info
    inside = [y for y, r in hits if r <= dev]
    if not inside:
        info['flags'].append('B:contour_never_straight')
        return h, info
    info['dev_top_mm'] = round(px2mm(hits[0][1]), 1)
    return int(min(inside)), info


def B4_anatomical_offset(T: int, d_tb_mm: float = D_TB_MM):
    """B4. Запасное правило: B = T + d_TB (калиброванная анатомическая константа)."""
    return int(T) + mm2px(d_tb_mm), {'flags': ['B:fallback_anatomical']}


# =========================================================================== #
#  Реестры вариантов (для сборки произвольных комбинаций)
# =========================================================================== #
L_VARIANTS = {'L1': L1_lower_fraction, 'L2': L2_jump_guard, 'L4': L4_bulge_vs_shaft_line}
T_VARIANTS = {'T1': T1_wide_part, 'T2': T2_band_component_top, 'T3': T3_contour_first_min}
B_VARIANTS = {'B1': B1_width_stabilisation, 'B2': B2_medial_line_deviation,
              'B3': B3_distance_ridge}
# B5 принимает ещё и полутоновый кадр, поэтому живёт отдельно от реестра.
