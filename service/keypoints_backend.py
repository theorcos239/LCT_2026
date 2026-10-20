# -*- coding: utf-8 -*-
"""Критерии по ключевым точкам: мост между `src/dxa_qc` и конвейером сервиса.

Модель точек в сервисе **необязательна**. Без torch или без весов бэкенд
просто недоступен (`available == False`), и конвейер работает на контурной
геометрии, как работал. Поэтому torch остаётся в requirements-train.txt, а
базовый образ — 450 МБ.

## Что бэкенд забирает себе, а что нет

Забирает ровно то, для чего есть сквозная метрика «геометрия по предсказанным
точкам против геометрии по разметке эксперта» (`runs/keypoints/metrics_end_to_end.csv`,
рабочий инференс: TTA и согласованная расшифровка позвоночника):

| величина        | медиана | p90    | порог ТЗ | вердикт |
|-----------------|---------|--------|----------|---------|
| угол оси        | 0.50°   | 1.41°  | 5°       | да      |
| отступ сверху   | 0.52 мм | 1.50 мм| 30 мм    | да      |
| отступ снизу    | 0.50 мм | 1.12 мм| 30 мм    | да      |
| отступ латерально| 0.51 мм| 1.52 мм| 20 мм    | да      |

(out-of-fold на кадрах фолдов 0–1: обучены два фолда из пяти.)

**Отступы ROI — по рисунку 6 ТЗ у обоих методов.** Низ — нижняя точка
седалищной кости: у модели это `ischium_bottom`, у контурного `hip_roi` —
ориентир `I1`. Отступы двух методов сходятся с медианой разницы около 1 мм,
поэтому для ROI модель точек ничего не добавляет к контуру; при включении
контурный замер остаётся в деталях, а расхождение больше 5 мм идёт во флаг.

Не забирает:

- **Ротация бедра.** Выступ малого вертела по точкам считается и уходит в
  диагностику, но вердикт остаётся за `hip_rotation`: ошибка выступа на OOF
  0.75 мм (p90 2.18) при разрыве между классами около 2.1 мм и требовании
  ТЗ ≈0.5 мм — это ровно та точность, которой не хватает контурной геометрии.
  Подменять один неуверенный измеритель другим неуверенным незачем.
- **Гребни подвздошных (часть укладки).** Каналы `iliac_right/left` остаются
  диагностикой: нарушений укладки в эталоне шесть, на фолдах 0–1 — два,
  переводить вердикт гребней на них нельзя. Вердикт гребней — за `spine_qc`.
- **Половина тела Th12 (вторая часть укладки ТЗ).** Контурный
  `top_margin_mm` это не измерял (порог OOF = 0). По точкам
  `th12_top` / `th12_bottom` и шагу L1–L2 считается доля видимого тела;
  при `frac < 0.5` укладка нарушена. На OOF `keypoints_crops` согласие с
  геометрией разметки точек F1 ≈ 0.93. Гребни и Th12 в вердикте — через OR.
- **Артефакты.** Канала под посторонние предметы в модели нет и не
  планировалось: это отдельная задача и отдельная модель.

## Ориентация

Два канонических кадра в проекте противоположны, и путать их нельзя:

    hip_roi.to_lateral_left   латеральный край СЛЕВА  ('rh' как есть, 'lh' зеркалим)
    dxa_qc.points             латеральный край СПРАВА ('lhip' как есть, 'rhip' зеркалим)

Проверено по разметке: в кадре модели `trochanter_lateral` правее
`femoral_head_center` на всех 149 бёдрах, `shaft_lateral` правее `shaft_medial`
на всех 149. Анатомия у модулей одна, канонический бок разный.
"""
from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path

import numpy as np

# Исследовательская часть живёт в src/ и пакетом не ставится: сервис импортирует
# region_clf/spine_qc/hip_roi как пакеты верхнего уровня, а dxa_qc — отсюда.
_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dxa_qc.metrics import axis_angle, axis_curvature   # noqa: E402 — после sys.path
from dxa_qc.spine_shape import (apply_scoliosis_to_axis, assess_scoliosis,
                                curve_descriptors, fused_tilt, source_variances,
                                tilt_sources)
from hip_roi.geometry import BOTTOM_MM, LAT_MM, MM_PER_PX, TOP_MM, px2mm
from spine_qc.criteria import axis_decision

# dxa_qc.metrics — чистая геометрия на numpy/pandas, torch она не тянет: бэкенд
# импортируется и там, где torch не установлен, и сообщает об этом флагом.

# Названия областей: у region_clf 'lh'/'rh'/'spine', у модели точек 'lhip'/'rhip'/'spine'.
SIDE = {'lh': 'lhip', 'rh': 'rhip', 'spine': 'spine'}

SPINE_AXIS_POINTS = [f'l{i}_center' for i in range(1, 5)]
SPINE_EDGE_PAIRS = [(f'l{i}_edge_left', f'l{i}_edge_right') for i in range(1, 5)]
ROI_POINTS = {'top': 'trochanter_major', 'bottom': 'ischium_bottom',
              'lateral': 'trochanter_lateral'}

# Расхождение двух конструкций оси, после которого измерение считается
# ненадёжным. Тот же порог, что у spine_qc: конструкции разные, смысл один.
AXIS_DISAGREE_DEG = 2.0


# --------------------------------------------------------------------------- #
#  Подготовка кадра
# --------------------------------------------------------------------------- #
def to_model_uint8(px: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Кадр в том виде, в каком модель его видела на обучении.

    Обучение шло на сыром `ds.pixel_array`: в этих данных он 8-битный
    MONOCHROME2 без VOI LUT, то есть ровно 0..255. Сервис читает кадр через
    `apply_voi_lut` и отдаёт float32 с теми же значениями, поэтому округления
    достаточно. Кадр, вышедший за 0..255 (другая разрядность, сработавший LUT),
    растягиваем по min-max и помечаем флагом: это уже не то распределение
    яркости, на котором модель училась.
    """
    a = np.asarray(px)
    if a.dtype == np.uint8:
        return a, []
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if 0.0 <= lo and hi <= 255.0:
        return np.clip(np.round(a), 0, 255).astype(np.uint8), []
    if hi <= lo:
        return np.zeros(a.shape, np.uint8), ['keypoints:blank_frame']
    scaled = np.round((a - lo) / (hi - lo) * 255).astype(np.uint8)
    return scaled, ['keypoints:rescaled_intensity']


def _flip_x(x: float, width: int) -> float:
    return width - 1 - x


# --------------------------------------------------------------------------- #
#  Геометрия поверх точек
# --------------------------------------------------------------------------- #
def axis_from_points(get, params: dict | None = None) -> dict | None:
    """Ось позвоночника по центрам тел и, независимо, по боковым краям.

    Две конструкции, как в `spine_qc`: центры тел (вердикт) и середины боковых
    краёв (проверка). Разница в том, что здесь обе берутся у одной модели, а
    значит коррелируют сильнее, чем плотность и силуэт, — поэтому расхождение
    идёт во флаг, а не в вердикт.

    Порог берётся из ТЗ как есть (5°), без поправки 4°, которой пользуется
    `spine_qc`: поправка компенсирует систематический недобор контурного
    измерителя, а модель воспроизводит ту самую конструкцию эксперта, по
    которой поправка и меряна (сквозная ошибка угла: медиана 0.50°, p90 1.41°).
    """
    centers = [get(n) for n in SPINE_AXIS_POINTS]
    if any(c is None for c in centers):
        return None
    centers = np.array(centers, dtype=float)
    ang = axis_angle(centers)

    mids = []
    for left, right in SPINE_EDGE_PAIRS:
        a, b = get(left), get(right)
        if a is not None and b is not None:
            mids.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    # Меньше трёх уровней — прямая по краям не конструкция, а шум.
    ang_edges = axis_angle(np.array(mids, dtype=float)) if len(mids) >= 3 else None

    flags, curvature = [], axis_curvature(centers)
    disagree = None if ang_edges is None else abs(ang - ang_edges)
    if disagree is None:
        flags.append('axis:edges_not_found')
    elif disagree > AXIS_DISAGREE_DEG:
        flags.append('axis:estimates_disagree')
    if curvature > 12.0:
        flags.append('axis:curved')

    # Исправленный угол строится ниже по тем же точкам: centers + пластинки.
    shape_names = SPINE_AXIS_POINTS + [n for pair in SPINE_EDGE_PAIRS for n in pair]
    shape_points = np.zeros((len(shape_names), 2), float)
    shape_visible = np.zeros(len(shape_names), bool)
    for i, name in enumerate(shape_names):
        point = get(name)
        if point is not None:
            shape_points[i], shape_visible[i] = point, True
    sources = tilt_sources(shape_points, shape_visible, shape_names, MM_PER_PX)
    curve = curve_descriptors(shape_points, shape_visible, shape_names, MM_PER_PX)
    params = params or {}
    sigmas = source_variances(sources, curve, params,
                              point_sigma_px=params.get('point_sigma_px'))
    fused = fused_tilt(sources, sigmas)
    decision = axis_decision(fused['theta_fused_deg'], fused['sigma_deg'],
                             fused['Q'], fused['n_sources'],
                             {'uncertainty_k': params.get('uncertainty_k', 1.0)})
    # Сколиотическая дуга даёт большой угол хорды без дефекта укладки.
    # Контур гасит такие кадры флагом curved_suppressed; здесь тот же смысл
    # по кривизне центров (порог 12 мм — флаг axis:curved) + явный признак
    # по segment_range L1–L4 (assess_scoliosis / apply_scoliosis_to_axis).
    if decision['state'] == 'unsuitable' and 'axis:curved' in flags:
        decision = {**decision, 'state': 'undetermined', 'violated': None,
                    'text': '', 'reasons': [*decision['reasons'],
                                           'scoliosis_like_curve']}
    if decision['state'] == 'undetermined':
        flags.append('axis:needs_review')
    correction = (None if fused['theta_fused_deg'] is None
                  else float(ang - fused['theta_fused_deg']))
    result = {'angle_deg': (None if fused['theta_fused_deg'] is None
                            else round(float(fused['theta_fused_deg']), 2)),
              'angle_centers_deg': round(ang, 2),
              'angle_edges_deg': None if ang_edges is None else round(ang_edges, 2),
              'curvature_mm': round(curvature, 1),
              'disagreement_deg': None if disagree is None else round(disagree, 2),
              'sigma_deg': fused['sigma_deg'], 'Q': fused['Q'],
              'weights': fused['weights'], 'sources': sources, 'curve': curve,
              'variance_params': params,
              'correction_deg': correction, 'state': decision['state'],
              'reasons': decision['reasons'],
              'source': 'keypoints_fused', 'flags': flags,
              'violated': decision['violated'], 'text': decision['text'],
              'scoliosis': assess_scoliosis(curve)}
    return apply_scoliosis_to_axis(result, result['scoliosis'])


def fuse_axis_with_contour(point_axis: dict, contour_axis: dict) -> dict:
    """Добавить независимый контурный угол и повторить свёртку."""
    if not point_axis:
        return point_axis
    sources = dict(point_axis.get('sources') or {})
    sources['theta_contour_deg'] = contour_axis.get('angle_deg')
    curve = dict(point_axis.get('curve') or {})
    params = {**(point_axis.get('variance_params') or {}),
              'contour_curvature_mm': contour_axis.get('curvature_mm') or 0.0}
    sigmas = source_variances(sources, curve, params,
                              point_sigma_px=params.get('point_sigma_px'))
    fused = fused_tilt(sources, sigmas)
    decision = axis_decision(fused['theta_fused_deg'], fused['sigma_deg'],
                             fused['Q'], fused['n_sources'],
                             {'uncertainty_k': params.get('uncertainty_k', 1.0)})
    # Контурная конструкция пометила большой угол как изгиб. При конфликте
    # не превращаем его в автоматический брак: у модели точек и контура разные
    # ошибки. Явный сколиоз по segment_range дожимается ниже.
    if (decision['state'] == 'unsuitable'
            and 'axis:curved_suppressed' in contour_axis.get('flags', [])):
        decision = {**decision, 'state': 'undetermined', 'violated': None,
                    'text': '', 'reasons': [*decision['reasons'],
                                           'scoliosis_like_contour']}
    out = dict(point_axis)
    scoliosis = (point_axis.get('scoliosis')
                 or assess_scoliosis(point_axis.get('curve')))
    out.update({
        'angle_deg': (None if fused['theta_fused_deg'] is None
                      else round(float(fused['theta_fused_deg']), 2)),
        'sigma_deg': fused['sigma_deg'], 'Q': fused['Q'],
        'weights': fused['weights'], 'sources': sources,
        'correction_deg': (None if fused['theta_fused_deg'] is None
                           else float(point_axis['angle_centers_deg']
                                      - fused['theta_fused_deg'])),
        'state': decision['state'], 'reasons': decision['reasons'],
        'violated': decision['violated'], 'text': decision['text'],
        'scoliosis': scoliosis,
    })
    flags = [f for f in out.get('flags', []) if f != 'axis:needs_review']
    if decision['state'] == 'undetermined':
        flags.append('axis:needs_review')
    out['flags'] = flags
    return apply_scoliosis_to_axis(out, scoliosis)


def axis_from_contour_only(contour_axis: dict, params: dict | None = None,
                           *, points_required: bool = False) -> dict:
    """Та же трёхсостоянийная политика при недоступных точках.

    Применяется только после G2; до этого сервис использует исходный
    контурный критерий с его собственным калиброванным порогом.
    """
    params = params or {}
    if points_required:
        decision = axis_decision(None, None, None, 0, params)
        return {**contour_axis, **decision, 'source': 'contour_fused_policy',
                'flags': [*contour_axis.get('flags', []), 'axis:points_missing']}
    angle = contour_axis.get('angle_deg')
    if angle is None:
        decision = axis_decision(None, None, None, 0, params)
        return {**contour_axis, **decision, 'source': 'contour_fused_policy'}
    src = {'theta_contour_deg': float(angle)}
    sig = source_variances(src, {'step_mm': 30.0},
                           {**params, 'contour_curvature_mm':
                            contour_axis.get('curvature_mm') or 0.0})
    fused = fused_tilt(src, sig)
    decision = axis_decision(
        fused['theta_fused_deg'], fused['sigma_deg'], fused['Q'], 1,
        {**params, 'min_sources': 1})
    return {**contour_axis, **decision, 'source': 'contour_fused_policy',
            'angle_deg': fused['theta_fused_deg'], 'sigma_deg': fused['sigma_deg'],
            'Q': fused['Q'], 'weights': fused['weights']}


def roi_from_points(get, shape: tuple[int, int], side: str) -> dict | None:
    """Отступы ROI бедра по трём точкам: верхушка вертела, седалищная кость, латеральный край.

    Возвращает тот же набор ключей, что `hip_roi.kits._result`, — отчёт,
    оверлеи и сквозная оценка разбирают оба источника одинаково.
    """
    h, w = shape
    p = {k: get(n) for k, n in ROI_POINTS.items()}
    if any(v is None for v in p.values()):
        return None

    T = float(p['top'][1])
    B = float(p['bottom'][1])
    # Латеральный край: в кадре модели он справа, в исходном кадре — со стороны
    # тела. Отступ считается до ближайшего к нему края снимка.
    lat_x = float(p['lateral'][0])
    m_top, m_bottom, m_lat = px2mm(T), px2mm(h - B), px2mm(w - 1 - lat_x)

    top_ok = bool(m_top >= TOP_MM)
    bottom_ok = bool(m_bottom >= BOTTOM_MM)
    lat_ok = bool(m_lat >= LAT_MM)
    roi_ok = bool(top_ok and bottom_ok and lat_ok)

    def cm(mm):
        return f'{mm / 10:.1f} см'

    parts = []
    if not top_ok:
        parts.append(f'сверху {cm(m_top)} < {TOP_MM / 10:.0f} см')
    if not bottom_ok:
        parts.append(f'снизу {cm(m_bottom)} < {BOTTOM_MM / 10:.0f} см')
    if not lat_ok:
        parts.append(f'латерально {cm(m_lat)} < {LAT_MM / 10:.0f} см')

    x_orig = lat_x if side == 'lh' else _flip_x(lat_x, w)
    apex = p['top']
    apex_orig = apex[0] if side == 'lh' else _flip_x(apex[0], w)
    return {
        'method': 'keypoints', 'side': side, 'H': int(h), 'W': int(w),
        'scale_mm_per_px': MM_PER_PX,
        'T_px': int(round(T)), 'B_px': int(round(B)), 'L_px': int(round(x_orig)),
        'apex_xy': (int(round(apex_orig)), int(round(apex[1]))),
        'm_top_mm': round(m_top, 1), 'm_bottom_mm': round(m_bottom, 1),
        'm_lat_mm': round(m_lat, 1),
        'top_ok': top_ok, 'bottom_ok': bottom_ok, 'lat_ok': lat_ok, 'roi_ok': roi_ok,
        'flags': [], 'violation_text': ('ROI: ' + '; '.join(parts)) if parts else '',
    }


def bulge_from_points(get) -> dict | None:
    """Выступ малого вертела над медиальным контуром диафиза, мм. Только диагностика.

    В кадре модели медиальная сторона слева, поэтому выступ — это
    `shaft_medial.x - trochanter_minor.x`. По разметке величина лежит в
    2.5–10.6 мм (5–95 %, медиана 6.0), по предсказаниям ошибка 0.75 мм при p90 2.18 —
    в полтора-четыре раза больше требуемых ТЗ ≈0.5 мм. Вердикт остаётся за `hip_rotation`.
    """
    minor, medial = get('trochanter_minor'), get('shaft_medial')
    if minor is None or medial is None:
        return None
    return {'bulge_mm': round(px2mm(float(medial[0]) - float(minor[0])), 2),
            'source': 'keypoints', 'verdict': None}


# Высота тела Th12 / шаг L1–L2 по 29 кадрам с обоими краями в pack (медиана 0.95).
# В structure.py для полосы декодера заложено ~0.8 — там допуск, здесь калибр покрытия.
TH12_BODY_TO_STEP = 0.95
TH12_HALF_FRAC = 0.5          # ТЗ: видна не меньше половины тела Th12
# Hard-reject только при видимом th12_bottom и очень малом frac.
# Сетка 0.05–0.50 на 92 live-кадрах (Processing/eval_th12_full/hard_frac_sweep_live.csv):
# плато expert F1=0.80 / GT precision=1.0 при hard∈[0.05, 0.13]. Правый край
# (0.13) на полной НД дал 2 FP / 0 TP vs эксперту (frac 0.066 и 0.091) —
# берём левый край плато, mid-band остаётся флагом без undetermined.
TH12_HARD_FRAC = 0.05
TH12_BORDER_FRAC = 0.05
# Порог ТЗ для оси (°) — сверка контура с диагностическими точками при
# закрытом axis-gate (см. reconcile_contour_axis_with_points).
AXIS_TZ_LIMIT_DEG = 5.0
# Контурный брак гасим, только если точки явно «норма» (далеко от порога).
# Без требования corroboration ≥4°: оно роняло recall оси на НД.
AXIS_POINT_CLEAR_DEG = 3.0
# Узкий override: контур «норма», но края расходятся и точки ≥ ТЗ → брак.
# На НД ловит FN (контур 2.7° / точки 5.7° / disagree 3.1°) без blanket-OR.
AXIS_POINT_OVERRIDE_DISAGREE_DEG = 2.0
# Выступ малого вертела по точкам: на pack 5–95% ≈ 2.5–10.6 мм, медиана 6.0.
# Если свод (геометрия/CNN) ставит брак, а точки в «ядре нормы» — гасим FP.
# Сетка на НД: [3, 8] мм → F1 0.60→0.67, P 0.51→0.70.
ROTATION_KP_NORMAL_LO_MM = 3.0
ROTATION_KP_NORMAL_HI_MM = 8.0
# Медиана |th12_bottom.y − th12_top.y| на pack, px — если L1/L2 не видимы.
TH12_BODY_PRIOR_PX = 38.0


def th12_coverage_from_points(
    get,
    image_height: int,
    body_to_step: float = TH12_BODY_TO_STEP,
    half_frac: float = TH12_HALF_FRAC,
    hard_frac: float = TH12_HARD_FRAC,
    border: float = TH12_BORDER_FRAC,
    body_prior_px: float = TH12_BODY_PRIOR_PX,
) -> dict:
    """Доля тела Th12 в кадре по ключевым точкам (верхняя граница укладки ТЗ).

    Двухпороговое правило (ТЗ закрыт, F1 по эксперту не роняем):
      * suitable — оба края или frac ≥ half_frac (0.5);
      * unsuitable (hard → spine_position) — виден th12_bottom и frac < hard_frac;
      * undetermined (review) — bottom_missing, mid-band [hard, half), prior, граница.
    Hard только когда нижняя замыкательная в кадре: иначе модель часто
    «не видит» точку на нормальных кадрах и даёт ложные срабатывания.
    """
    del image_height  # API стабилен; y=0 — верх того же массива, что у get().
    top, bot = get('th12_top'), get('th12_bottom')
    l1, l2 = get('l1_center'), get('l2_center')
    disc = get('disc_th12_l1')

    body_meas = (abs(float(bot[1]) - float(top[1]))
                 if top is not None and bot is not None else None)
    step = (float(np.linalg.norm(np.asarray(l2, float) - np.asarray(l1, float)))
            if l1 is not None and l2 is not None else None)
    body_source = 'measured'
    if body_meas is not None:
        body_est = body_meas
    elif step is not None and step > 1:
        body_est = body_to_step * step
        body_source = 'l1_l2_step'
    elif body_prior_px and body_prior_px > 1:
        body_est = float(body_prior_px)
        body_source = 'prior'
    else:
        body_est = None
        body_source = 'none'

    flags: list[str] = []
    if top is not None and bot is None:
        return {
            'state': 'undetermined', 'violated': None, 'frac': None,
            'reason': 'top_without_bottom', 'needs_review': True,
            'flags': ['position:th12_undetermined'],
            'text': '', 'source': 'keypoints', 'body_source': body_source,
            'body_est_px': None if body_est is None else round(float(body_est), 1),
            'visible_px': None, 'half_frac': half_frac, 'hard_frac': hard_frac,
            'body_to_step': body_to_step,
        }
    if body_est is None or body_est <= 1:
        return {
            'state': 'undetermined', 'violated': None, 'frac': None,
            'reason': 'body_height_unknown', 'needs_review': True,
            'flags': ['position:th12_undetermined'],
            'text': '', 'source': 'keypoints', 'body_source': body_source,
            'body_est_px': None, 'visible_px': None,
            'half_frac': half_frac, 'hard_frac': hard_frac,
            'body_to_step': body_to_step,
        }

    if top is not None and bot is not None:
        visible_px = float(body_meas)
        frac = 1.0
        reason = 'both_endplates'
    elif bot is not None:
        visible_px = max(0.0, float(bot[1]))
        frac = visible_px / float(body_est)
        reason = 'bottom_only'
    else:
        visible_px = 0.0
        frac = 0.0
        reason = 'bottom_missing'

    if disc is not None and bot is not None and float(disc[1]) + 1.0 < float(bot[1]):
        flags.append('position:th12_disc_inconsistent')

    needs_review = False
    if reason == 'both_endplates' or frac >= half_frac:
        state, violated = 'suitable', False
    elif reason == 'bottom_missing':
        # Нет нижней замыкательной: ТЗ не подтверждён, но hard-reject по
        # «невидимости» даёт десятки FP vs эксперту (ошибка видимости модели).
        state, violated = 'undetermined', None
        needs_review = True
        flags.append('position:th12_bottom_missing')
    elif body_source == 'prior':
        state, violated = 'undetermined', None
        needs_review = True
        flags.append('position:th12_prior_uncertain')
    elif frac < hard_frac:
        state, violated = 'unsuitable', True
        flags.append('position:th12_short')
    else:
        # hard_frac ≤ frac < half_frac: зона ТЗ, без автоматического брака.
        state, violated = 'undetermined', None
        needs_review = True
        flags.append('position:th12_review')
        if abs(frac - half_frac) <= border:
            flags.append('position:th12_borderline')

    text = ''
    if violated:
        text = (f'укладка: видна малая доля тела Th12 '
                f'({frac:.0%} при hard-пороге {hard_frac:.0%}, норма ТЗ ≥{half_frac:.0%})')
    return {
        'state': state, 'violated': violated, 'frac': round(float(frac), 3),
        'reason': reason, 'needs_review': needs_review,
        'flags': flags, 'text': text, 'source': 'keypoints',
        'body_source': body_source,
        'body_est_px': round(float(body_est), 1),
        'visible_px': round(float(visible_px), 1),
        'half_frac': half_frac, 'hard_frac': hard_frac,
        'body_to_step': body_to_step,
    }


def apply_th12_to_position(position: dict, th12: dict | None) -> dict:
    """Вшить Th12 в укладку: hard → OR в spine_position; иначе review-флаги."""
    if not th12:
        return position
    out = {**position, 'th12_coverage': th12}
    if th12.get('frac') is not None:
        out['th12_frac'] = th12['frac']
    flags = list(out.get('flags') or [])
    for flag in th12.get('flags') or []:
        if flag not in flags:
            flags.append(flag)

    if th12.get('violated') is True:
        out['violated'] = True
        out['th12_violated'] = True
        out['th12_needs_review'] = False
        old = (out.get('text') or '').removeprefix('укладка:').strip()
        th12_msg = (th12.get('text') or '').removeprefix('укладка:').strip()
        parts = [p for p in (old, th12_msg) if p]
        uniq: list[str] = []
        for part in parts:
            if part not in uniq:
                uniq.append(part)
        out['text'] = ('укладка: ' + '; '.join(uniq)) if uniq else ''
    elif th12.get('violated') is False:
        out['th12_violated'] = False
        out['th12_needs_review'] = False
    else:
        out['th12_violated'] = None
        out['th12_needs_review'] = bool(th12.get('needs_review'))
        if 'position:th12_undetermined' not in flags and out['th12_needs_review']:
            flags.append('position:th12_undetermined')
    out['flags'] = flags
    return out


def record_th12_review(out: dict, position: dict) -> None:
    """Mid-band / bottom_missing Th12 → флаги review, без undetermined.

    Раньше сюда писался ``undetermined`` в violations: на НД это давало
    ~16 ложных «есть нарушение» на чистых кадрах и роняло binary F1.
    Hard-Th12 по-прежнему OR'ится в spine_position через apply_th12.
    """
    if not position.get('th12_needs_review'):
        return
    if position.get('violated'):
        return  # уже брак по гребням или hard-Th12
    if 'position:th12_needs_review' not in out['flags']:
        out['flags'].append('position:th12_needs_review')


def reconcile_contour_axis_with_points(
    contour_axis: dict | None,
    point_axis: dict | None,
    *,
    tz_limit_deg: float = AXIS_TZ_LIMIT_DEG,
    clear_deg: float = AXIS_POINT_CLEAR_DEG,
    override_disagree_deg: float = AXIS_POINT_OVERRIDE_DISAGREE_DEG,
) -> dict | None:
    """Точки — судья для контурной оси при закрытом gate.

    Полный fused-verdict закрыт OOF-гейтом. Правила (НД, spine):
      * контур «брак», |θ_точек| < clear_deg → гасим;
      * контур «норма», |θ_точек| ≥ tz_limit и расхождение краёв
        > override_disagree_deg → брак по точкам (``axis:point_overrides_contour``);
      * контур «норма», |θ_точек| ≥ tz_limit без расхождения → только флаг
        ``axis:point_tilt_over_tz`` (без автобрака).
    Сильный сколиоз гасится позже ``apply_scoliosis_to_axis``.
    """
    if not contour_axis:
        return contour_axis
    if not point_axis:
        return contour_axis
    point_ang = point_axis.get('angle_deg')
    if point_ang is None:
        point_ang = point_axis.get('angle_centers_deg')
    if point_ang is None:
        return contour_axis
    point_abs = abs(float(point_ang))
    out = dict(contour_axis)
    flags = list(out.get('flags') or [])
    if 'axis:points_missing' in flags:
        return contour_axis

    out['point_angle_deg'] = round(float(point_ang), 2)
    disagree = out.get('disagreement_deg')
    if disagree is None and 'axis:estimates_disagree' in flags:
        disagree = override_disagree_deg + 0.1

    if out.get('violated') is True and point_abs < clear_deg:
        out['violated'] = False
        out['text'] = ''
        if 'state' in out:
            out['state'] = 'suitable'
        flags = [f for f in flags if f != 'axis:needs_review']
        if 'axis:point_clears_contour' not in flags:
            flags.append('axis:point_clears_contour')
        out['flags'] = flags
        return out

    if (out.get('violated') is not True and point_abs >= tz_limit_deg
            and disagree is not None
            and float(disagree) > override_disagree_deg):
        out['violated'] = True
        out['text'] = (f'ось позвоночника отклонена на {point_abs:.1f}° по точкам '
                       f'(контур ненадёжен: расхождение краёв {float(disagree):.1f}°; '
                       f'допустимо {tz_limit_deg:.0f}°)')
        if 'state' in out:
            out['state'] = 'unsuitable'
        if 'axis:point_overrides_contour' not in flags:
            flags.append('axis:point_overrides_contour')
        # убрать «мягкий» флаг tilt, если был
        flags = [f for f in flags if f != 'axis:point_tilt_over_tz']
        out['flags'] = flags
        return out

    if out.get('violated') is not True and point_abs >= tz_limit_deg:
        if 'axis:point_tilt_over_tz' not in flags:
            flags.append('axis:point_tilt_over_tz')
        out['flags'] = flags
        return out

    return out


def judge_rotation_with_points(
    rotation: dict | None,
    kp_bulge_mm: float | None,
    *,
    normal_lo: float = ROTATION_KP_NORMAL_LO_MM,
    normal_hi: float = ROTATION_KP_NORMAL_HI_MM,
) -> dict | None:
    """Точки — судья в споре по ротации: нормальный выступ снимает брак.

    Никогда не создаёт нового нарушения (только гасит FP). Если точек нет —
    вердикт свода геометрия+CNN не трогаем.
    """
    if not rotation or rotation.get('violated') is not True:
        return rotation
    if kp_bulge_mm is None or not np.isfinite(float(kp_bulge_mm)):
        return rotation
    bulge = float(kp_bulge_mm)
    out = dict(rotation)
    out['keypoints_bulge_mm'] = round(bulge, 2)
    flags = list(out.get('flags') or [])
    if normal_lo <= bulge <= normal_hi:
        out['violated'] = False
        out['text'] = ''
        out['geometry_violated'] = out.get('geometry_violated')
        if 'rotation:points_normal_bulge' not in flags:
            flags.append('rotation:points_normal_bulge')
        out['flags'] = flags
        out['confidence'] = 'low'
        out['method'] = (str(out.get('method') or 'ротация')
                         + '; точки: выступ в норме → брак снят')
        return out
    if 'rotation:points_abnormal_bulge' not in flags:
        flags.append('rotation:points_abnormal_bulge')
    out['flags'] = flags
    return out


# --------------------------------------------------------------------------- #
#  Бэкенд
# --------------------------------------------------------------------------- #
class KeypointBackend:
    """Модель ключевых точек как источник измерений для конвейера.

    Создаётся один раз на процесс: веса двух фолдов — 187 МБ, чтение с диска
    небыстрое. Ни одна ошибка загрузки наружу не выходит — недоступный бэкенд
    это не отказ сервиса, а возврат к контурной геометрии.
    """

    def __init__(self, run_dir: str | Path | None = None, device: str = 'cpu',
                 quiet: bool = True, folds: list[int] | None = None,
                 multicrop: bool = False, use_for_verdict: bool = True):
        self.predictor, self.error = None, None
        self.multicrop_requested = bool(multicrop)
        self.multicrop = False
        self.multicrop_variant = 'auto'
        self.multicrop_gate_passed = False
        self.use_for_verdict = bool(use_for_verdict)
        self.axis_gate_passed = False
        self.axis_params = {}
        try:
            from dxa_qc.infer import RUN_DIR, fold_checkpoints, load_predictor
            self.predictor = load_predictor(run_dir, device, folds)
            run_path = Path(run_dir or RUN_DIR)
            def same_weights(expected):
                if not isinstance(expected, dict) or not expected:
                    return False
                for checkpoint in fold_checkpoints(run_path, folds):
                    fold = checkpoint.stem.removeprefix('fold')
                    digest = hashlib.sha256()
                    with checkpoint.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                    if expected.get(fold) != digest.hexdigest():
                        return False
                return set(expected) == {p.stem.removeprefix('fold')
                                         for p in fold_checkpoints(run_path, folds)}
            gate_path = Path(run_dir or RUN_DIR) / 'metrics_axis_fused.json'
            if gate_path.exists():
                gate = json.loads(gate_path.read_text(encoding='utf-8'))
                self.axis_gate_passed = (bool(gate.get('gate_passed'))
                                         and same_weights(gate.get('checkpoint_sha256')))
                self.axis_params = dict(gate.get('runtime_params') or {})
            semantic_gate_path = Path(run_dir or RUN_DIR) / 'metrics_semantic_gates.json'
            consensus_gate_path = Path(run_dir or RUN_DIR) / 'metrics_crop_consensus.json'
            if consensus_gate_path.exists():
                consensus = json.loads(consensus_gate_path.read_text(encoding='utf-8'))
                from dxa_qc.multicrop import DEFAULT_SEMANTIC_CONFIG
                self.multicrop_gate_passed = (bool(consensus.get('gate_passed'))
                    and bool((consensus.get('checks') or {}).get('five_folds'))
                    and same_weights(consensus.get('checkpoint_sha256'))
                    and hashlib.sha256(DEFAULT_SEMANTIC_CONFIG.read_bytes()).hexdigest()
                    == consensus.get('semantic_config_sha256'))
                if self.multicrop_gate_passed:
                    self.multicrop_variant = 'consensus'
            if semantic_gate_path.exists():
                semantic = json.loads(semantic_gate_path.read_text(encoding='utf-8'))
                if isinstance(semantic, list):
                    auto = next((r for r in semantic if r.get('variant') == 'semantic_auto'), {})
                    auto_gate_passed = bool(auto.get('gate_passed')) and bool(
                        auto.get('provenance_verified', False)) and bool(
                            (auto.get('checks') or {}).get('five_fold_oof')) and same_weights(
                            auto.get('checkpoint_sha256'))
                    anchor_path = run_path / 'thresholds_anchor.csv'
                    auto_gate_passed = (auto_gate_passed and anchor_path.exists()
                                        and hashlib.sha256(anchor_path.read_bytes()).hexdigest()
                                        == auto.get('anchor_thresholds_sha256'))
                    if auto_gate_passed and not self.multicrop_gate_passed:
                        self.multicrop_gate_passed = True
                        self.multicrop_variant = 'auto'
            self.multicrop = self.multicrop_requested and self.multicrop_gate_passed
        except Exception as e:                       # noqa: BLE001 — причина уходит во флаг
            self.error = f'{type(e).__name__}: {e}'
            if not quiet:
                print(f'модель точек недоступна: {self.error}')

    @property
    def available(self) -> bool:
        return self.predictor is not None

    @property
    def names(self) -> list[str]:
        return list(self.predictor.names) if self.available else []

    # ------------------------------------------------------------------ #
    def points(self, px: np.ndarray, region: str) -> dict:
        """Кадр и область -> точки в координатах ИСХОДНОГО кадра.

        Правое бедро приводится к левому перед прогоном и возвращается обратно:
        в наборе точек бедра стороны нет, обучение шло только на левых.
        """
        img, flags = to_model_uint8(px)
        if self.multicrop_requested and not self.multicrop_gate_passed:
            flags.append('keypoints:semantic_gate_failed')
        side = SIDE.get(region, region)
        w = img.shape[1]
        if side == 'rhip':
            img = img[:, ::-1].copy()

        if side == 'spine' and self.multicrop:
            out = (self.predictor.predict_consensus(img)
                   if self.multicrop_variant == 'consensus'
                   else self.predictor.predict_semantic(img))
        else:
            out = self.predictor.predict(img)
        coords = np.asarray(out['coords'], dtype=float).copy()
        if side == 'rhip':
            coords[:, 0] = _flip_x(coords[:, 0], w)

        multicrop_raw = out.get('multicrop')
        multicrop = ({k: v for k, v in multicrop_raw.items() if k != 'sources'}
                     if multicrop_raw else None)
        if multicrop and multicrop.get('abstain'):
            flags += ['keypoints:crop_disagree', 'keypoints:numbering_doubt']
        return {
            'names': list(out['names']),
            'coords': coords,                       # (C, 2) в исходном кадре
            'coords_model': np.asarray(out['coords'], dtype=float),
            'confidence': np.asarray(out['confidence'], dtype=float),
            'visible': np.asarray(out['visible'], dtype=bool),
            'region': out['region'],                # 'spine' | 'hip' | 'out_of_scope'
            'flags': flags,
            'multicrop': multicrop,
        }

    # ------------------------------------------------------------------ #
    def analyze(self, px: np.ndarray, region: str) -> dict:
        """Кадр -> измерения по точкам. Вердикт выносится только там, где он проверен.

        `axis` и `roi` не None — конвейер берёт их вместо контурных.
        `th12_coverage` — верхняя граница укладки ТЗ; конвейер OR'ит её в
        `spine_position` (гребни по-прежнему за контуром).
        `rotation` и `iliac` — всегда только диагностика.
        """
        kp = self.points(px, region)
        idx = {n: i for i, n in enumerate(kp['names'])}
        flags = list(kp['flags'])

        # Точка идёт в геометрию только если модель считает её видимой:
        # координата невидимого канала — это позиция максимума шума.
        def get(name):
            i = idx.get(name)
            if i is None or not kp['visible'][i]:
                return None
            # Модель работает в своём кадре: бедро приведено к левому, и вся
            # геометрия ниже написана для него.
            return tuple(kp['coords_model'][i])

        expected = 'spine' if region == 'spine' else 'hip'
        if kp['region'] != expected:
            flags.append(f"keypoints:region_disagree({kp['region']})")

        res = {
            'region_model': kp['region'],
            'axis': None, 'roi': None, 'rotation': None, 'iliac': None,
            'th12_coverage': None,
            'axis_diagnostic': None, 'roi_diagnostic': None,
            'multicrop': kp.get('multicrop'),
            'points': {n: {'x': round(float(kp['coords'][i][0]), 1),
                           'y': round(float(kp['coords'][i][1]), 1),
                           'confidence': round(float(kp['confidence'][i]), 3),
                           'visible': bool(kp['visible'][i])}
                       for n, i in idx.items()},
            'flags': flags,
        }

        if region == 'spine':
            axis = axis_from_points(get, self.axis_params)
            res['axis_diagnostic'] = axis
            abstain = bool((kp.get('multicrop') or {}).get('abstain'))
            res['axis'] = (axis if self.use_for_verdict and self.axis_gate_passed
                           and not abstain else None)
            if axis is None:
                flags.append('keypoints:axis_points_missing')
            else:
                flags += axis['flags']
            if abstain:
                flags.append('keypoints:axis_abstained')
            if self.use_for_verdict and not self.axis_gate_passed:
                flags.append('keypoints:axis_gate_failed')
            res['iliac'] = {side: bool(kp['visible'][idx[f'iliac_{side}']])
                            for side in ('right', 'left') if f'iliac_{side}' in idx}
            # Покрытие Th12 не зависит от axis-gate: контур эту половину ТЗ
            # не меряет, а геометрическое правило по точкам проверено на OOF.
            th12 = th12_coverage_from_points(get, int(px.shape[0]))
            if abstain and th12.get('violated') is not None:
                th12 = {
                    **th12,
                    'state': 'undetermined',
                    'violated': None,
                    'flags': list(th12.get('flags') or []) + [
                        'position:th12_numbering_doubt'],
                    'text': '',
                }
            res['th12_coverage'] = th12
            flags += [f for f in th12.get('flags') or [] if f not in flags]
        else:
            roi = roi_from_points(get, px.shape, region)
            res['roi_diagnostic'] = roi
            res['roi'] = roi if self.use_for_verdict else None
            if roi is None:
                flags.append('keypoints:roi_points_missing')
            res['rotation'] = bulge_from_points(get)
        return res
