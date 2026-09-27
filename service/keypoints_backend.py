# -*- coding: utf-8 -*-
"""Критерии по ключевым точкам: мост между `src/dxa_qc` и конвейером сервиса.

Модель точек в сервисе **необязательна**. Без torch или без весов бэкенд
просто недоступен (`available == False`), и конвейер работает на контурной
геометрии, как работал. Поэтому torch остаётся в requirements-train.txt, а
базовый образ — 450 МБ.

## Что бэкенд забирает себе, а что нет

Забирает ровно то, для чего есть сквозная метрика «геометрия по предсказанным
точкам против геометрии по разметке эксперта» (`runs/keypoints/metrics_end_to_end.csv`):

| величина        | медиана | p90    | порог ТЗ | вердикт |
|-----------------|---------|--------|----------|---------|
| угол оси        | 0.34°   | 1.17°  | 5°       | да      |
| отступ сверху   | 0.44 мм | 1.17 мм| 30 мм    | да      |
| отступ снизу    | 0.37 мм | 0.92 мм| 30 мм    | да      |
| отступ латерально| 0.76 мм| 1.47 мм| 20 мм    | да      |

**Низ ROI — особый случай.** Точность у модели есть (0.37 мм), но меряет она
другой уровень: `ischium_bottom` лежит примерно на 20 мм ниже, чем якорь
`hip_roi` (верхушка большого вертела + 50 мм, уровень малого вертела). Верх и
латераль у двух методов сходятся в пределах 2 мм, низ расходится на 20 — это
спор об определении «3 см снизу от области интереса», а не о точности, и
решать его измерением нельзя. Поэтому по умолчанию бэкенд выключен, а при
включении контурный замер остаётся в деталях и расхождение идёт во флаг.

Не забирает:

- **Ротация бедра.** Выступ малого вертела по точкам считается и уходит в
  диагностику, но вердикт остаётся за `hip_rotation`: ошибка выступа на OOF
  0.82 мм (p90 1.76) при разрыве между классами около 2.1 мм и требовании
  ТЗ ≈0.5 мм — это ровно та точность, которой не хватает контурной геометрии.
  Подменять один неуверенный измеритель другим неуверенным незачем.
- **Укладка позвоночника.** Формально критерий — «видны верхние края
  подвздошных костей», и у модели на это есть каналы `iliac_right/left` с
  калиброванным порогом видимости. Но на 18 OOF-кадрах позвоночника гребни
  были видны **всегда**: голова видимости для них не проверена ни одним
  отрицательным примером. Видимость уходит в диагностику, вердикт — за
  `spine_qc`.
- **Артефакты.** Канала под посторонние предметы в модели нет и не
  планировалось (TODO.md: это отдельная задача, отдельная модель).

## Ориентация

Два канонических кадра в проекте противоположны, и путать их нельзя:

    hip_roi.to_lateral_left   латеральный край СЛЕВА  ('rh' как есть, 'lh' зеркалим)
    dxa_qc.points             латеральный край СПРАВА ('lhip' как есть, 'rhip' зеркалим)

Проверено по разметке: в кадре модели `trochanter_lateral` правее
`femoral_head_center` на всех 31 бедре, `shaft_lateral` правее `shaft_medial`
на всех 31. Анатомия у модулей одна, канонический бок разный.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Исследовательская часть живёт в src/ и пакетом не ставится: сервис импортирует
# region_clf/spine_qc/hip_roi как пакеты верхнего уровня, а dxa_qc — отсюда.
_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dxa_qc.metrics import axis_angle, axis_curvature   # noqa: E402 — после sys.path
from hip_roi.geometry import BOTTOM_MM, LAT_MM, MM_PER_PX, TOP_MM, px2mm
from spine_qc.geometry import AXIS_LIMIT_DEG

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
def axis_from_points(get) -> dict | None:
    """Ось позвоночника по центрам тел и, независимо, по боковым краям.

    Две конструкции, как в `spine_qc`: центры тел (вердикт) и середины боковых
    краёв (проверка). Разница в том, что здесь обе берутся у одной модели, а
    значит коррелируют сильнее, чем плотность и силуэт, — поэтому расхождение
    идёт во флаг, а не в вердикт.

    Порог берётся из ТЗ как есть (5°), без поправки 4°, которой пользуется
    `spine_qc`: поправка компенсирует систематический недобор контурного
    измерителя, а модель воспроизводит ту самую конструкцию эксперта, по
    которой поправка и меряна (сквозная ошибка угла: медиана 0.34°, p90 1.17°).
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

    violated = abs(ang) > AXIS_LIMIT_DEG
    text = (f'ось позвоночника отклонена на {abs(ang):.1f}° '
            f'(допустимо {AXIS_LIMIT_DEG:.0f}°)') if violated else ''
    return {'angle_deg': round(ang, 2),
            'angle_edges_deg': None if ang_edges is None else round(ang_edges, 2),
            'curvature_mm': round(curvature, 1),
            'disagreement_deg': None if disagree is None else round(disagree, 2),
            'source': 'keypoints', 'flags': flags, 'violated': bool(violated), 'text': text}


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
    2.6–9.0 мм (медиана 5.6), по предсказаниям ошибка 0.82 мм при p90 1.76 —
    вдвое-втрое больше требуемых ТЗ ≈0.5 мм. Вердикт остаётся за `hip_rotation`.
    """
    minor, medial = get('trochanter_minor'), get('shaft_medial')
    if minor is None or medial is None:
        return None
    return {'bulge_mm': round(px2mm(float(medial[0]) - float(minor[0])), 2),
            'source': 'keypoints', 'verdict': None}


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
                 quiet: bool = True):
        self.predictor, self.error = None, None
        try:
            from dxa_qc.infer import load_predictor
            self.predictor = load_predictor(run_dir, device)
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
        side = SIDE.get(region, region)
        w = img.shape[1]
        if side == 'rhip':
            img = img[:, ::-1].copy()

        out = self.predictor.predict(img)
        coords = np.asarray(out['coords'], dtype=float).copy()
        if side == 'rhip':
            coords[:, 0] = _flip_x(coords[:, 0], w)

        return {
            'names': list(out['names']),
            'coords': coords,                       # (C, 2) в исходном кадре
            'coords_model': np.asarray(out['coords'], dtype=float),
            'confidence': np.asarray(out['confidence'], dtype=float),
            'visible': np.asarray(out['visible'], dtype=bool),
            'region': out['region'],                # 'spine' | 'hip' | 'out_of_scope'
            'flags': flags,
        }

    # ------------------------------------------------------------------ #
    def analyze(self, px: np.ndarray, region: str) -> dict:
        """Кадр -> измерения по точкам. Вердикт выносится только там, где он проверен.

        `axis` и `roi` не None — конвейер берёт их вместо контурных.
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
            'points': {n: {'x': round(float(kp['coords'][i][0]), 1),
                           'y': round(float(kp['coords'][i][1]), 1),
                           'confidence': round(float(kp['confidence'][i]), 3),
                           'visible': bool(kp['visible'][i])}
                       for n, i in idx.items()},
            'flags': flags,
        }

        if region == 'spine':
            res['axis'] = axis_from_points(get)
            if res['axis'] is None:
                flags.append('keypoints:axis_points_missing')
            else:
                flags += res['axis']['flags']
            res['iliac'] = {side: bool(kp['visible'][idx[f'iliac_{side}']])
                            for side in ('right', 'left') if f'iliac_{side}' in idx}
        else:
            res['roi'] = roi_from_points(get, px.shape, region)
            if res['roi'] is None:
                flags.append('keypoints:roi_points_missing')
            res['rotation'] = bulge_from_points(get)
        return res
