# -*- coding: utf-8 -*-
"""Три критерия качества кадра поясничного отдела (ТЗ, п. 2.3, пункт 1).

    укладка    — видны верхние края подвздошных костей и половина тела Th12
    ось        — наклон оси позвоночника не больше 5°
    артефакты  — нет посторонних предметов и наложений

Каждый критерий измеряется ДВУМЯ независимыми способами и они сверяются.
Так устроен весь проект (ср. `B:estimates_disagree` в hip_roi): на выборке с
6-17 положительными примерами одиночное число нечем проверить, а расхождение
двух оценок — честный признак «здесь измерение ненадёжно».

Важно, что вторые оценки именно независимы. Первая попытка сверять морфологию
top-hat с логистической регрессией по тем же четырём признакам ничего не дала:
оценки расходились на 1 кадре из 99, потому что смотрели на одно и то же.
Работающие пары опираются на разные свидетельства:

| критерий  | оценка A (измерение)             | оценка B (независимая)        | корр. |
|-----------|----------------------------------|-------------------------------|-------|
| ось       | угол по центрам тел позвонков    | угол по боковым краям колонны | 0.59  |
| укладка   | костная масса гребней внизу      | вид кадра целиком (LR)        | 0.66  |
| артефакты | top-hat: тонкие яркие структуры  | вид кадра целиком (LR)        | 0.18  |

Правило свёртки выбрано по OOF-F1 отдельно для каждого критерия и в каждом
случае разное — см. calibrate.py и README. Расхождение оценок всегда выносится
во флаг, даже когда вердикт принимает одна из них.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import probability
from .geometry import (AXIS_LIMIT_DEG, SpineColumn, mm2px, px2mm, spine_column,
                       theil_sen, white_tophat)

# --------------------------------------------------------------------------- #
#  Пороги. Значения по умолчанию — из calibrate.py (OOF по 5 фолдам).
# --------------------------------------------------------------------------- #
DEFAULTS = {
    # Наш оценщик читает наклон примерно на градус ниже конструкции эксперта
    # (медиана на нарушениях 4.0° против порога ТЗ 5°), поэтому порог решения
    # калибруется, а не берётся из ТЗ буквально. Измеренный угол в отчёт идёт
    # как есть — вердикт объясним числом.
    'axis_deg': 4.0,
    'axis_disagree_deg': 2.0,       # расхождение двух конструкций -> не уверены
    # Подавление «сколиоз ≠ наклон»: нарушение оси только на достаточно прямом
    # треке. Пороги подбираются OOF в calibrate.py (geometry_and_straight).
    'axis_curved_max_mm': 99.0,     # изгиб в вердикте отключён; см. calibrate
    'axis_sign_max': 99,            # признаки пишутся в details
    'axis_half_max_deg': 99.0,      # основной фильтр — согласие двух углов
    'iliac_area': 0.010,            # доля костных пикселей в нижних латеральных зонах
    'top_margin_min_mm': 0.0,       # 0 = верх кадра не участвует в вердикте
    'artifact_score': 24.0,         # длина x вытянутость x контраст / 100
    'artifact_min_elongation': 2.0, # короткие пятна отбрасываем как ложный кортикал
    'artifact_max_vert_deg': 25.0,  # почти вертикальные короткие компоненты — шум
    'appearance': 0.5,              # порог вероятности у модели вида кадра
}


def _verdict(violated: bool | None, text: str) -> dict:
    return {'violated': violated, 'text': text}


# --------------------------------------------------------------------------- #
#  Ось позвоночника
# --------------------------------------------------------------------------- #
def _axis_track_features(y: np.ndarray, center: np.ndarray, valid: np.ndarray,
                         slope: float, intercept: float) -> tuple[float, int, float]:
    """Изгиб, число перемен знака сглаженного остатка и |угол верха − угол низа|.

    Остаток сглаживается окном ~12 мм: без сглаживания пиксельный шум даёт
    десятки перемен знака на любом кадре, и признак не отличает сколиоз.
    """
    from .geometry import moving_average
    b = valid
    resid = center[b] - (slope * y[b] + intercept)
    curvature = px2mm(float(np.abs(resid).max()))
    win = max(5, mm2px(12.0) | 1)          # нечётное окно
    smooth = moving_average(resid.astype(float), win)
    # Порог «ноль»: мелкие колебания после сглаживания не считаем.
    dead = max(mm2px(0.8), 1.0)
    sgn = np.sign(np.where(np.abs(smooth) < dead, 0.0, smooth))
    sgn = sgn[sgn != 0]
    sign_changes = int(np.sum(sgn[1:] * sgn[:-1] < 0)) if sgn.size >= 2 else 0
    idx = np.flatnonzero(b)
    mid = idx[len(idx) // 2]
    top, bot = b.copy(), b.copy()
    top[mid:] = False
    bot[:mid] = False
    if top.sum() >= 8 and bot.sum() >= 8:
        sl_t, _ = theil_sen(y[top], center[top])
        sl_b, _ = theil_sen(y[bot], center[bot])
        half_diff = abs(float(np.degrees(np.arctan(sl_t)) - np.degrees(np.arctan(sl_b))))
    else:
        half_diff = 0.0
    return curvature, sign_changes, half_diff


def axis_is_straight(curvature_mm: float, sign_changes: int, half_diff_deg: float,
                     thr: dict, disagreement_deg: float | None = None) -> bool:
    """Трек достаточно согласован и прямой, чтобы угол считать наклоном.

    Две конструкции угла должны сходиться (иначе измерение ненадёжно — на
    таких кадрах OOF F1 оси выше, если вердикт не ставить). Дополнительно
    изгиб/перемены знака/расхождение половин подавляют сколиоз-подобные треки.
    """
    if disagreement_deg is not None and disagreement_deg > thr['axis_disagree_deg']:
        return False
    if curvature_mm > thr['axis_curved_max_mm']:
        return False
    # Подавляем только когда И много перемен знака, И половины расходятся —
    # иначе один признак на малой выборке слишком агрессивен.
    if (sign_changes > thr['axis_sign_max']
            and half_diff_deg > thr['axis_half_max_deg']):
        return False
    return True


def axis(col: SpineColumn, thr: dict | None = None) -> dict:
    """Наклон оси позвоночника, °. Положительный угол — низ уходит вправо.

    Оценка A — прямая Тейла-Сена по центрам тел позвонков. Оценка B — среднее
    прямых по левому и правому краям колонны: та же ось, но по силуэту, а не
    по плотности. Вердикт принимает A (AUC 0.83 против 0.80 у B), B проверяет:
    на кадрах, где они сходятся в пределах 2°, F1 критерия 0.50 против 0.41 на
    всех кадрах, то есть расхождение действительно метит ненадёжные измерения.

    Угол выше порога на изогнутом треке (сколиоз) нарушением не считаем:
    эксперт на таких кадрах `spine_axis` не ставит. Признаки изгиба и пороги —
    OOF в calibrate.py.
    """
    t = {**DEFAULTS, **(thr or {})}
    y = np.arange(col.shape[0], dtype=float)
    b = col.valid
    if b.sum() < 20:
        return {'angle_deg': None, 'angle_edges_deg': None, 'curvature_mm': None,
                'sign_changes': None, 'half_angle_diff_deg': None,
                'flags': ['axis:column_not_found'], **_verdict(None, 'ось: колонна не найдена')}

    slope_c, intercept = theil_sen(y[b], col.center[b])
    ang_c = float(np.degrees(np.arctan(slope_c)))
    sl_l, _ = theil_sen(y[b], col.left[b])
    sl_r, _ = theil_sen(y[b], col.right[b])
    ang_edges = float(np.degrees(np.arctan((sl_l + sl_r) / 2.0)))

    curvature, sign_changes, half_diff = _axis_track_features(
        y, col.center, b, slope_c, intercept)

    flags = []
    disagree = abs(ang_c - ang_edges)
    if disagree > t['axis_disagree_deg']:
        flags.append('axis:estimates_disagree')
    if curvature > t['axis_curved_max_mm']:
        flags.append('axis:curved')

    angle_bad = abs(ang_c) > t['axis_deg']
    straight = axis_is_straight(curvature, sign_changes, half_diff, t,
                                disagreement_deg=disagree)
    if angle_bad and not straight:
        flags.append('axis:curved_suppressed')
        violated = False
    else:
        violated = bool(angle_bad)
    text = (f'ось позвоночника отклонена на {abs(ang_c):.1f}° '
            f'(допустимо {AXIS_LIMIT_DEG:.0f}°)') if violated else ''
    return {'angle_deg': round(ang_c, 2), 'angle_edges_deg': round(ang_edges, 2),
            'curvature_mm': round(curvature, 1),
            'sign_changes': int(sign_changes),
            'half_angle_diff_deg': round(half_diff, 2),
            'disagreement_deg': round(disagree, 2),
            'flags': flags, **_verdict(bool(violated), text)}


# --------------------------------------------------------------------------- #
#  Укладка: гребни подвздошных костей
# --------------------------------------------------------------------------- #
def iliac_masses(col: SpineColumn) -> dict:
    """Костная масса в нижних латеральных зонах кадра — прокси гребней.

    Критерий ТЗ: «на нижнем уровне сканирования визуализированы верхние края
    подвздошных костей». На кадре это две крупные яркие массы по сторонам от
    позвоночника в нижней трети. Если поле сканирования поднято, их нет вовсе:
    у всех 6 помеченных экспертом кадров доля костных пикселей в этих зонах
    равна нулю против медианы 0.086 у нормы.
    """
    h, w = col.shape
    lat = col.center_x
    gap = mm2px(35.0)                       # 3.5 см от центра колонны — вне тел и отростков
    xs = np.arange(w)
    half = mm2px(8.0)
    out = {}
    for name, sel in (('left', xs < lat - gap), ('right', xs > lat + gap)):
        sub = col.mask[:, sel]
        if sub.size == 0 or not sub.any():
            out[f'iliac_{name}_area'] = 0.0
            out[f'iliac_{name}_top'] = None
            continue
        bottom = sub[int(h * 0.5):]
        out[f'iliac_{name}_area'] = float(bottom.sum() / max(bottom.size, 1))
        rowsum = sub.sum(axis=1)
        big = np.flatnonzero(rowsum > half)
        big = big[big >= int(h * 0.35)]
        out[f'iliac_{name}_top'] = int(big[0]) if big.size else None
    out['iliac_area'] = min(out['iliac_left_area'], out['iliac_right_area'])
    return out


def top_margin_mm(col: SpineColumn) -> float | None:
    """Сколько миллиметров колонны видно от верха кадра до первого valid-ряда.

    Прокси верхней границы укладки ТЗ («половина тела Th12»). Порог
    `top_margin_min_mm` (если > 0) включает это число в геометрический вердикт.
    """
    b = np.flatnonzero(col.valid)
    return px2mm(float(b[0])) if b.size else None


def position(col: SpineColumn, appearance_p: float | None = None, thr: dict | None = None) -> dict:
    """Укладка поясничного отдела.

    Геометрия: нет гребней снизу ИЛИ слишком малый запас колонны сверху.
    С appearance (если есть) — AND, как раньше: на OOF согласие поднимает
    специфичность. Порог верха калибруется в calibrate.py; 0 = выключен.
    """
    t = {**DEFAULTS, **(thr or {})}
    f = iliac_masses(col)
    top = top_margin_mm(col)
    iliac_bad = f['iliac_area'] < t['iliac_area']
    top_thr = float(t.get('top_margin_min_mm') or 0.0)
    top_bad = bool(top_thr > 0 and top is not None and top < top_thr)
    geo_bad = bool(iliac_bad or top_bad)
    flags = []
    if f['iliac_left_area'] < t['iliac_area'] or f['iliac_right_area'] < t['iliac_area']:
        flags.append('position:iliac_missing_one_side')
    if top_bad:
        flags.append('position:top_margin_short')

    if appearance_p is None:
        violated = bool(geo_bad)
        flags.append('position:no_appearance_model')
    else:
        app_bad = appearance_p > t['appearance']
        violated = bool(geo_bad and app_bad)
        if geo_bad != app_bad:
            flags.append('position:estimates_disagree')

    parts = []
    if violated and iliac_bad:
        parts.append('не визуализированы верхние края подвздошных костей')
    if violated and top_bad:
        parts.append(f'малый запас сверху ({top:.1f} мм)')
    text = ('укладка: ' + '; '.join(parts)) if parts else ''
    return {'iliac_area': round(f['iliac_area'], 4),
            'iliac_left_area': round(f['iliac_left_area'], 4),
            'iliac_right_area': round(f['iliac_right_area'], 4),
            'top_margin_mm': None if top is None else round(top, 1),
            'appearance_p': None if appearance_p is None else round(float(appearance_p), 3),
            'flags': flags, **_verdict(bool(violated), text)}


# --------------------------------------------------------------------------- #
#  Посторонние предметы и артефакты
# --------------------------------------------------------------------------- #
def artifact_structures(col: SpineColumn, radius_mm: float = 3.0, k: float = 2.5,
                        thr: dict | None = None) -> dict:
    """Тонкие яркие структуры вне колонны позвоночника.

    Косточка бюстгальтера — длинная дуга шириной 2-4 px, ярче окружающих
    мягких тканей. Абсолютная яркость её не выдаёт: экспорт Lunar уже насыщен,
    max = 255 на всех 99 кадрах, и кортикальная кость такая же белая. Зато
    она выдаётся формой — white top-hat со структурным элементом шире косточки
    и уже тела позвонка гасит и кость, и мягкие ткани, а металл оставляет.

    Колонна позвоночника из зоны поиска исключается: её собственная текстура
    (отростки, замыкательные пластинки) даёт такой же отклик.
    """
    a = col.img.astype(float)
    h, w = col.shape
    th = white_tophat(a, mm2px(radius_mm))

    xs = np.arange(w)[None, :]
    outside = np.abs(xs - col.center_x) > col.half_width + mm2px(6.0)
    lit = a > np.percentile(a[a > 0], 20) if (a > 0).any() else np.zeros_like(a, bool)
    zone = np.broadcast_to(outside, (h, w)) & lit

    empty = {'n_structures': 0, 'length_mm': 0.0, 'elongation': 0.0,
             'contrast': 0.0, 'area_frac': 0.0, 'score': 0.0}
    v = th[zone]
    if v.size < 100:
        return empty
    # порог по разбросу самого отклика: median + k*(p84 - median) — устойчивая
    # замена «среднее + k*сигма», не сбиваемая самими артефактами
    level = np.median(v) + k * (np.percentile(v, 84) - np.median(v))
    cand = ndi.binary_opening((th > max(level, 8.0)) & zone, np.ones((2, 2), bool))
    lbl, n = ndi.label(cand, np.ones((3, 3), bool))
    if n == 0:
        return empty

    t = {**DEFAULTS, **(thr or {})}
    min_el = float(t.get('artifact_min_elongation', 2.0))
    max_vert = float(t.get('artifact_max_vert_deg', 25.0))
    found = []
    for key in range(1, n + 1):
        ys, xs_ = np.nonzero(lbl == key)
        if len(ys) < 15:
            continue
        p = np.stack([ys.astype(float), xs_.astype(float)])
        p -= p.mean(axis=1, keepdims=True)
        cov = np.cov(p)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if ev[1] <= 1e-9:
            continue
        length, width = 4 * np.sqrt(ev[0]), 4 * np.sqrt(ev[1])
        elong = float(length / max(width, 1e-6))
        # Угол главной оси к вертикали кадра (0° = вертикаль).
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        # axis[0] — dy, axis[1] — dx; угол к вертикали.
        vert_deg = float(np.degrees(np.arctan2(abs(axis[1]), abs(axis[0]) + 1e-9)))
        # Короткие почти вертикальные пятна у края кости — ложный кортикал.
        if elong < min_el and vert_deg < max_vert:
            continue
        if elong < min_el:
            continue
        found.append({'area': len(ys), 'length_mm': px2mm(length),
                      'elongation': elong, 'vert_deg': vert_deg,
                      'contrast': float(th[lbl == key].mean())})
    if not found:
        return empty

    # ведущая структура — самая длинная и вытянутая: косточка белья длиннее и
    # ровнее любого случайного пятна мягких тканей
    top = max(found, key=lambda f: f['length_mm'] * f['elongation'])
    # Штраф за компактность: log1p(elong) и так слабее на круглых пятнах.
    score = top['length_mm'] * np.log1p(top['elongation']) * top['contrast'] / 100.0
    return {'n_structures': len(found),
            'length_mm': round(top['length_mm'], 1),
            'elongation': round(top['elongation'], 2),
            'contrast': round(top['contrast'], 1),
            'area_frac': round(sum(f['area'] for f in found) / (h * w), 5),
            'score': round(float(score), 2)}


def artifacts(col: SpineColumn, appearance_p: float | None = None, thr: dict | None = None) -> dict:
    """Посторонние предметы и наложения.

    Вердикт принимает морфология (OOF F1 0.65 против 0.42 у вида кадра), вид
    кадра сверяет. Согласие обеих оценок поднимает специфичность до 0.99, но
    роняет чувствительность до 0.47, поэтому в вердикт оно не идёт — только в
    уверенность: при расхождении кадр помечается как требующий внимания.
    """
    t = {**DEFAULTS, **(thr or {})}
    f = artifact_structures(col, thr=t)
    geo_bad = f['score'] > t['artifact_score']

    flags = []
    if appearance_p is None:
        flags.append('artifacts:no_appearance_model')
        confidence = 'low'
    else:
        app_bad = appearance_p > t['appearance']
        if geo_bad != app_bad:
            flags.append('artifacts:estimates_disagree')
            confidence = 'low'
        else:
            confidence = 'high'

    text = (f'посторонний предмет: яркая структура длиной {f["length_mm"]:.0f} мм '
            f'вне позвоночника') if geo_bad else ''
    return {**f, 'appearance_p': None if appearance_p is None else round(float(appearance_p), 3),
            'confidence': confidence, 'flags': flags, **_verdict(bool(geo_bad), text)}


# --------------------------------------------------------------------------- #
#  Все критерии сразу
# --------------------------------------------------------------------------- #
def analyze(img: np.ndarray, appearance: dict | None = None, thr: dict | None = None) -> dict:
    """Кадр позвоночника -> три критерия, измерения, флаги.

    appearance — вероятности от модели вида кадра: {'position': p, 'artifacts': p}.
    Без неё критерии считаются по одной геометрии (флаг *:no_appearance_model).
    """
    app = appearance or {}
    col = spine_column(img)
    res = {
        'axis': axis(col, thr),
        'position': position(col, app.get('position'), thr),
        'artifacts': artifacts(col, app.get('artifacts'), thr),
    }
    # Калиброванная вероятность рядом с вердиктом: вердикт нужен оператору,
    # вероятность — отчёту, по ней организатор считает ROC-AUC.
    params = (thr or {}).get('probability', {})
    for name, crit in res.items():
        crit['probability'] = probability.apply(params.get(name),
                                                probability.score_of(name, crit))

    violations = [v['text'] for v in res.values() if v['violated'] and v['text']]
    flags = list(col.flags) + [f for v in res.values() for f in v['flags']]
    return {'criteria': res, 'violations': violations, 'flags': flags,
            'quality_class': int(any(v['violated'] for v in res.values())),
            'quality_probability': probability.combine(
                v['probability'] for v in res.values()),
            'column': col}
