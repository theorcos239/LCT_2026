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
    'iliac_area': 0.010,            # доля костных пикселей в нижних латеральных зонах
    'artifact_score': 24.0,         # длина x вытянутость x контраст / 100
    'appearance': 0.5,              # порог вероятности у модели вида кадра
}


def _verdict(violated: bool | None, text: str) -> dict:
    return {'violated': violated, 'text': text}


# --------------------------------------------------------------------------- #
#  Ось позвоночника
# --------------------------------------------------------------------------- #
def axis(col: SpineColumn, thr: dict | None = None) -> dict:
    """Наклон оси позвоночника, °. Положительный угол — низ уходит вправо.

    Оценка A — прямая Тейла-Сена по центрам тел позвонков. Оценка B — среднее
    прямых по левому и правому краям колонны: та же ось, но по силуэту, а не
    по плотности. Вердикт принимает A (AUC 0.83 против 0.80 у B), B проверяет:
    на кадрах, где они сходятся в пределах 2°, F1 критерия 0.50 против 0.41 на
    всех кадрах, то есть расхождение действительно метит ненадёжные измерения.
    """
    t = {**DEFAULTS, **(thr or {})}
    y = np.arange(col.shape[0], dtype=float)
    b = col.valid
    if b.sum() < 20:
        return {'angle_deg': None, 'angle_edges_deg': None, 'curvature_mm': None,
                'flags': ['axis:column_not_found'], **_verdict(None, 'ось: колонна не найдена')}

    slope_c, intercept = theil_sen(y[b], col.center[b])
    ang_c = float(np.degrees(np.arctan(slope_c)))
    sl_l, _ = theil_sen(y[b], col.left[b])
    sl_r, _ = theil_sen(y[b], col.right[b])
    ang_edges = float(np.degrees(np.arctan((sl_l + sl_r) / 2.0)))

    resid = col.center[b] - (slope_c * y[b] + intercept)
    curvature = px2mm(float(np.abs(resid).max()))

    flags = []
    disagree = abs(ang_c - ang_edges)
    if disagree > t['axis_disagree_deg']:
        flags.append('axis:estimates_disagree')
    # Сколиоз даёт малый наклон при большом изгибе: ось формально вертикальна,
    # но позвоночник изогнут. В выборке эксперт пометил такие исследования
    # «сколиоз» в комментарии; критерий ТЗ их не описывает, поэтому это флаг,
    # а не нарушение.
    if curvature > 12.0:
        flags.append('axis:curved')

    violated = abs(ang_c) > t['axis_deg']
    text = (f'ось позвоночника отклонена на {abs(ang_c):.1f}° '
            f'(допустимо {AXIS_LIMIT_DEG:.0f}°)') if violated else ''
    return {'angle_deg': round(ang_c, 2), 'angle_edges_deg': round(ang_edges, 2),
            'curvature_mm': round(curvature, 1), 'disagreement_deg': round(disagree, 2),
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
    """Сколько миллиметров колонны видно выше самого верхнего целого позвонка.

    Вторая половина критерия укладки — «верхний уровень: половина тела Th12».
    Отдельной разметки уровня Th12 в данных нет, проверить такое правило не на
    чем, поэтому число считается и кладётся в отчёт как диагностика, но вердикт
    на него не опирается. Это ограничение, а не недоделка: см. README.
    """
    b = np.flatnonzero(col.valid)
    return px2mm(float(b[0])) if b.size else None


def position(col: SpineColumn, appearance_p: float | None = None, thr: dict | None = None) -> dict:
    """Укладка поясничного отдела.

    Вердикт — согласие двух оценок (геометрия И вид кадра). Именно «И», а не
    «или» и не одна из них: на OOF геометрия даёт F1 0.43, вид кадра 0.67, их
    согласие 0.73 при специфичности 0.99 (1 ложное срабатывание на 93 нормы).
    Пропускать нарушение укладки неприятно, но 12 ложных тревог на 93 кадра
    (столько даёт одна геометрия) делают проверку бесполезной для оператора.
    """
    t = {**DEFAULTS, **(thr or {})}
    f = iliac_masses(col)
    geo_bad = f['iliac_area'] < t['iliac_area']
    flags = []
    if f['iliac_left_area'] < t['iliac_area'] or f['iliac_right_area'] < t['iliac_area']:
        flags.append('position:iliac_missing_one_side')

    if appearance_p is None:
        violated = bool(geo_bad)
        flags.append('position:no_appearance_model')
    else:
        app_bad = appearance_p > t['appearance']
        violated = bool(geo_bad and app_bad)
        if geo_bad != app_bad:
            flags.append('position:estimates_disagree')

    text = ('укладка: не визуализированы верхние края подвздошных костей'
            if violated else '')
    return {'iliac_area': round(f['iliac_area'], 4),
            'iliac_left_area': round(f['iliac_left_area'], 4),
            'iliac_right_area': round(f['iliac_right_area'], 4),
            'top_margin_mm': top_margin_mm(col),
            'appearance_p': None if appearance_p is None else round(float(appearance_p), 3),
            'flags': flags, **_verdict(bool(violated), text)}


# --------------------------------------------------------------------------- #
#  Посторонние предметы и артефакты
# --------------------------------------------------------------------------- #
def artifact_structures(col: SpineColumn, radius_mm: float = 3.0, k: float = 2.5) -> dict:
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

    found = []
    for key in range(1, n + 1):
        ys, xs_ = np.nonzero(lbl == key)
        if len(ys) < 15:
            continue
        p = np.stack([ys.astype(float), xs_.astype(float)])
        p -= p.mean(axis=1, keepdims=True)
        ev = np.sort(np.linalg.eigvalsh(np.cov(p)))[::-1]
        if ev[1] <= 1e-9:
            continue
        length, width = 4 * np.sqrt(ev[0]), 4 * np.sqrt(ev[1])
        found.append({'area': len(ys), 'length_mm': px2mm(length),
                      'elongation': float(length / max(width, 1e-6)),
                      'contrast': float(th[lbl == key].mean())})
    if not found:
        return empty

    # ведущая структура — самая длинная и вытянутая: косточка белья длиннее и
    # ровнее любого случайного пятна мягких тканей
    top = max(found, key=lambda f: f['length_mm'] * f['elongation'])
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
    f = artifact_structures(col)
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
    violations = [v['text'] for v in res.values() if v['violated'] and v['text']]
    flags = list(col.flags) + [f for v in res.values() for f in v['flags']]
    return {'criteria': res, 'violations': violations, 'flags': flags,
            'quality_class': int(any(v['violated'] for v in res.values())),
            'column': col}
