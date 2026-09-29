# -*- coding: utf-8 -*-
"""Конвейер контроля качества: DICOM-исследование -> строки отчёта.

Порядок работы над кадром и почему он такой:

    1. область (region_clf)   кадр приходит без BodyPartExamined и Laterality,
                              и пока область неизвестна, непонятно, какие
                              критерии применять. Здесь же отбраковывается
                              то, что вообще не DXA.
    2. критерии области       позвоночник: укладка, ось, артефакты
                              бедро:       ротация, отступы ROI
    3. свод                   quality_class = 1, если сработал хоть один
                              критерий; типы нарушений перечисляются все

Мультилейбловость здесь не надстройка, а исходное свойство задачи: в обучающем
наборе у 20 исследований из 100 больше одного нарушения. Поэтому критерии
считаются независимо и ни один не «выигрывает» у остальных.

Ни одно исключение наружу не выходит: любая ошибка становится строкой отчёта
со статусом Failure и текстом ошибки (ТЗ 2.7).
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spine_qc import probability

from .dicom_io import Frame, unique_frames

# --------------------------------------------------------------------------- #
#  Таксономия нарушений (ТЗ 2.3)
# --------------------------------------------------------------------------- #
VIOLATIONS = {
    'spine_position':  'поясничный отдел: некорректная укладка',
    'spine_axis':      'поясничный отдел: ось отклонена более чем на 5°',
    'spine_artifacts': 'поясничный отдел: посторонние предметы или артефакты',
    'hip_rotation':    'проксимальный отдел бедра: некорректная ротация',
    'hip_roi':         'проксимальный отдел бедра: недостаточные отступы области интереса',
    'undetermined':    'анатомическая область не определена',
}

REGION_RU = {'spine': 'поясничный отдел позвоночника',
             'lh': 'проксимальный отдел левого бедра',
             'rh': 'проксимальный отдел правого бедра',
             'unknown': 'не определена'}

COLUMNS = ['path_to_study', 'study_uid', 'image_uid', 'anatomical_region',
           'quality_class', 'violation_type', 'processing_status', 'time_of_processing']

# Расхождение контурного измерителя и модели точек, после которого кадр
# помечается как ненадёжный. Пороги те же, что внутри критериев: 2° — это
# `axis_disagree_deg` у spine_qc, 5 мм — шестая часть запаса ROI в 30 мм.
AXIS_METHODS_DISAGREE_DEG = 2.0
ROI_METHODS_DISAGREE_MM = 5.0


@dataclass
class _Unavailable:
    """Заглушка бэкенда, который не удалось даже импортировать (нет torch, нет src/).

    Нужна, чтобы причина дошла до отчёта: `None` вместо бэкенда означал бы
    «точки не запрашивали», а это другое.
    """
    error: str
    available: bool = False


@dataclass
class Analyzer:
    """Загруженные модели. Создаётся один раз на процесс — веса читаются с диска."""
    region: object
    spine: object
    hip_method: str = 'kit2'
    strict_region: bool = True      # непринятый кадр не отправляем на критерии
    keypoints: object = None        # модель ключевых точек; None — контурная геометрия
    cnn: object = None              # cnn_qc: второе мнение по ротации и артефактам
    # Вердикт по отступам ROI: 'scan_length' — хватает ли длины поля (так
    # отступы оценивает эксперт, F1 0.71 против его разметки), 'margins' —
    # отступы от найденных ориентиров буквально по рисунку 6 ТЗ (F1 0.35).
    # Отступы по рисунку 6 считаются и показываются в обоих режимах.
    roi_rule: str = 'scan_length'

    @classmethod
    def load(cls, hip_method: str = 'kit2', strict_region: bool = True,
             keypoints: bool = False, keypoints_dir=None, cnn: bool = True,
             cnn_oof: bool = False, roi_rule: str = 'scan_length') -> 'Analyzer':
        """keypoints=True подключает модель ключевых точек (`runs/keypoints`).

        Она забирает себе два измерения — угол оси позвоночника и отступы ROI
        бедра, — потому что сквозная ошибка по ним измерена и мала (0.50° и
        0.4-0.5 мм out-of-fold против порогов 5° и 30/20 мм). Остальные критерии остаются
        за контурной геометрией: см. `keypoints_backend`. Недоступная модель
        (нет torch, нет весов) не роняет сервис: кадры считаются контурной
        геометрией, а причина уходит во флаг каждой строки отчёта — молчаливой
        подмены метода быть не должно.
        """
        from region_clf import RegionClassifier
        from spine_qc import SpineQC
        net = None
        if cnn or cnn_oof:
            try:
                from cnn_qc import CnnQC, OofCnnQC
                net = OofCnnQC() if cnn_oof else CnnQC()
            except Exception as e:      # noqa: BLE001 — нет onnxruntime или каталога
                net = _Unavailable(f'{type(e).__name__}: {e}')
        kp = None
        if keypoints:
            try:
                from .keypoints_backend import KeypointBackend
                kp = KeypointBackend(keypoints_dir)
            except Exception as e:      # noqa: BLE001 — нет torch, нет src/dxa_qc
                kp = _Unavailable(f'{type(e).__name__}: {e}')
        return cls(region=RegionClassifier(), spine=SpineQC(),
                   hip_method=hip_method, strict_region=strict_region, keypoints=kp,
                   cnn=net, roi_rule=roi_rule)

    # ------------------------------------------------------------------ #
    def _keypoints(self, px: np.ndarray, region: str, out: dict) -> dict | None:
        """Измерения по ключевым точкам или None, если модель их не дала.

        Сбой модели не должен ронять кадр: критерии посчитаются контурной
        геометрией, а причина уйдёт во флаг. Молча подменять метод нельзя —
        иначе по отчёту не понять, чем именно померено.
        """
        if self.keypoints is None:
            return None
        if not self.keypoints.available:
            out['flags'].append(f'keypoints:unavailable({self.keypoints.error})')
            return None
        try:
            kp = self.keypoints.analyze(px, region)
        except Exception as e:                       # noqa: BLE001 — причина во флаг
            out['flags'].append(f'keypoints:failed({type(e).__name__}: {e})')
            return None
        out['flags'] += kp['flags']
        out['details']['keypoints'] = {'region_model': kp['region_model'],
                                       'points': kp['points'],
                                       'iliac_visible': kp['iliac']}
        return kp

    def _cnn(self, crit: str, px: np.ndarray, region: str, geometry, out: dict) -> dict | None:
        """Свод сети с геометрией по критерию или None, если сеть недоступна.

        Недоступная сеть не роняет кадр и не подменяет метод молча: вердикт
        остаётся за геометрией, причина уходит во флаг.
        """
        if self.cnn is None:
            return None
        check = getattr(self.cnn, 'available', False)
        if not (check(crit) if callable(check) else check):
            err = getattr(self.cnn, 'errors', {}).get(crit) or getattr(self.cnn, 'error', '')
            out['flags'].append(f'cnn:unavailable({crit}: {err})')
            return None
        try:
            return self.cnn.assess(crit, px, region, geometry)
        except Exception as e:                       # noqa: BLE001 — причина во флаг
            out['flags'].append(f'cnn:failed({crit}: {type(e).__name__}: {e})')
            return None

    @staticmethod
    def _merge_cnn(c: dict, a: dict, name: str, out: dict) -> None:
        """Вердикт критерия -> свод сети и геометрии; геометрия остаётся в деталях."""
        geo = c.get('violated')
        c['geometry_violated'] = geo
        c['p_cnn'] = a['p_cnn']
        c['probability'] = a['probability']
        c['probability_threshold'] = a['threshold']
        c['violated'] = a['violated']
        c['method'] = 'свод: геометрия + нейросеть (cnn_qc)'
        if geo is not None and bool(geo) != a['violated']:
            out['flags'].append(f'{name}:cnn_vs_geometry_disagree')
            c['confidence'] = 'low'
        elif geo is not None:
            c['confidence'] = 'high'
        out.setdefault('heatmaps', {})[name] = a['heatmap']

    def _spine_probabilities(self, criteria: dict, out: dict) -> None:
        """Калиброванная вероятность нарушения по каждому критерию позвоночника.

        Считается после возможной подмены оси моделью точек: вероятность
        должна идти от того же измерения, что и вердикт.
        """
        params = self.spine.thresholds.get('probability', {})
        for crit, key in (('position', 'spine_position'), ('axis', 'spine_axis'),
                          ('artifacts', 'spine_artifacts')):
            c = criteria[crit]
            if c.get('probability') is None:
                c['probability'] = probability.apply(
                    params.get(crit), probability.score_of(crit, c))
            out['probabilities'][key] = c['probability']

    # ------------------------------------------------------------------ #
    def analyze_pixels(self, px: np.ndarray) -> dict:
        """Пиксели -> область, критерии, нарушения. Без чтения файлов."""
        r = self.region.classify(px)
        region, accepted = r['label'], r['accepted']
        out = {
            'anatomical_region': region if accepted or not self.strict_region else 'unknown',
            'region_confidence': round(float(r['confidence']), 4),
            'region_accepted': bool(accepted),
            'region_novelty': round(float(r['novelty']), 4),
            'violations': [], 'details': {}, 'flags': [],
            'probabilities': {},        # калиброванные вероятности по критериям
        }
        if not accepted and self.strict_region:
            # Кадр не похож на то, на чём учились модели. Пропустить его как
            # качественный нельзя — это молчаливый пропуск брака, поэтому он
            # помечается нарушением «область не определена» и уходит человеку.
            out['violations'].append('undetermined')
            out['flags'].append(f"region:{r['reason']}")
            return out

        kp = self._keypoints(px, region, out)

        if region == 'spine':
            res = self.spine.analyze(px)
            out['details']['spine'] = res['criteria']
            out['flags'] += res['flags']
            # Ось — единственный критерий позвоночника, который модель точек
            # забирает: она меряет ту же конструкцию, что чертит эксперт, и
            # ошибка угла на OOF 0.50° при пороге 5°.
            if kp and kp['axis'] is not None:
                contour_axis = res['criteria']['axis']
                out['details']['spine']['axis_contour'] = contour_axis
                # Две конструкции внутри модели точек коррелируют (общий
                # энкодер) и расходятся редко — настоящая независимая проверка
                # здесь именно контурный угол.
                a, b = kp['axis']['angle_deg'], contour_axis.get('angle_deg')
                if b is not None and abs(a - b) > AXIS_METHODS_DISAGREE_DEG:
                    out['flags'].append('axis:keypoints_vs_contour_disagree')
                out['details']['spine']['axis'] = kp['axis']
                res['criteria']['axis'] = kp['axis']
            art = res['criteria']['artifacts']
            a = self._cnn('artifacts', px, 'spine', art.get('score'), out)
            if a is not None:
                self._merge_cnn(art, a, 'artifacts', out)
                if art['violated'] and not art['geometry_violated']:
                    art['text'] = (f"посторонний предмет или наложение: по виду кадра "
                                   f"(нейросеть {a['p_cnn']:.2f}, вероятность нарушения "
                                   f"{a['probability']:.2f})")
                elif not art['violated']:
                    art['text'] = ''
            self._spine_probabilities(res['criteria'], out)
            for crit, key in (('position', 'spine_position'), ('axis', 'spine_axis'),
                              ('artifacts', 'spine_artifacts')):
                if res['criteria'][crit]['violated']:
                    out['violations'].append(key)
        else:
            from hip_roi import probability as roi_probability
            from hip_roi.kits import measure_roi_margins, scan_field
            from hip_rotation import THRESHOLDS as ROT_THR
            from hip_rotation import detect as detect_rotation

            contour = measure_roi_margins(px, region, self.hip_method)
            # Отступы по точкам вытесняют контурные: ошибка 0.4-0.8 мм против
            # порогов 30 и 20 мм. Контурный замер остаётся в диагностике —
            # расхождение двух методов видно в деталях и никуда не прячется.
            roi = contour
            if kp and kp['roi'] is not None:
                out['details']['hip_roi_contour'] = {k: v for k, v in contour.items()
                                                     if k != 'diag'}
                worst = max((abs(kp['roi'][k] - contour[k])
                             for k in ('m_top_mm', 'm_bottom_mm', 'm_lat_mm')
                             if contour.get(k) is not None), default=0.0)
                if worst > ROI_METHODS_DISAGREE_MM:
                    out['flags'].append('roi:keypoints_vs_contour_disagree')
                roi = kp['roi']
            out['details']['hip_roi'] = {k: v for k, v in roi.items() if k != 'diag'}
            out['flags'] += list(roi.get('flags', []))
            hr = out['details']['hip_roi']
            hr['margins_ok'] = roi.get('roi_ok')          # отступы по рисунку 6 ТЗ
            hr.update(scan_field(px))
            hr['roi_rule'] = self.roi_rule
            if self.roi_rule == 'scan_length':
                hr['roi_ok'] = hr['field_ok']
                hr['violation_text'] = '' if hr['field_ok'] else (
                    f"поле сканирования {hr['scan_length_mm'] / 10:.1f} см короче "
                    f"{hr['min_length_mm'] / 10:.1f} см: отступы 3 см сверху и снизу "
                    f"от области интереса не помещаются")
            if hr.get('roi_ok') is False:
                out['violations'].append('hip_roi')
            # Вероятность — от того же измерения, что и вердикт.
            p_roi = roi_probability.apply(hr, self.roi_rule)
            out['details']['hip_roi']['probability'] = p_roi
            out['probabilities']['hip_roi'] = p_roi

            # Ротация берёт ту же верхушку большого вертела, что и отступы ROI:
            # один ориентир на оба критерия бедра, а не два независимых.
            #
            # T сюда идёт КОНТУРНЫЙ, даже когда точки доступны: коридор нормы
            # по площади выступа (95.7-271.8 мм²) откалиброван именно на нём.
            # Подставить более точную верхушку, не пересчитав коридор, значит
            # сдвинуть измерение относительно порога.
            rot = detect_rotation(px, region, thr=ROT_THR, T=contour.get('T_px'))
            if kp and kp['rotation'] is not None:
                rot = {**rot, 'keypoints_bulge_mm': kp['rotation']['bulge_mm']}
            rot['probability'] = probability.apply(ROT_THR.get('probability'),
                                                   rot.get('corridor_distance'))
            a = self._cnn('rotation', px, region, rot.get('corridor_distance'), out)
            if a is not None:
                self._merge_cnn(rot, a, 'rotation', out)
                if rot['violated'] and not rot['geometry_violated']:
                    area = rot.get('area_mm2')
                    rot['text'] = ('ротация: вид малого вертела нетипичен для корректной '
                                   f"укладки (нейросеть {a['p_cnn']:.2f}"
                                   + (f', выступ {area:.0f} мм² в пределах коридора'
                                      if area is not None else '') + ')')
                elif not rot['violated']:
                    rot['text'] = ''
            out['probabilities']['hip_rotation'] = rot['probability']
            out['details']['hip_rotation'] = rot
            out['flags'] += list(rot.get('flags', []))
            if rot.get('violated'):
                out['violations'].append('hip_rotation')
        # Свод по области: 1 - П(1 - p) по критериям, у которых есть измерение.
        # Кадр, где ни один критерий не измерился, остаётся с None.
        out['quality_probability'] = probability.combine(out['probabilities'].values())
        return out

    # ------------------------------------------------------------------ #
    def analyze_frame(self, frame: Frame, path_to_study: str | None = None) -> dict:
        """Кадр -> строка отчёта. Исключения превращаются в статус Failure."""
        t0 = time.perf_counter()
        row = {
            'path_to_study': path_to_study or str(frame.path.parent),
            'study_uid': frame.study_uid,
            'image_uid': frame.image_uid,
            'anatomical_region': 'unknown',
            'quality_class': 1,
            'violation_type': '',
            'processing_status': 'Success',
            'time_of_processing': 0.0,
        }
        try:
            res = self.analyze_pixels(frame.pixels)
            row['anatomical_region'] = res['anatomical_region']
            # DXA позвоночника и бедра по протоколу — передне-задняя проекция;
            # кадр приведён к ней по PatientOrientation при чтении (dicom_io).
            row['projection'] = 'AP' if res['anatomical_region'] in ('spine', 'lh', 'rh') else 'unknown'
            orient = (frame.meta or {}).get('orientation', '')
            if orient in ('mirrored_lr', 'mirrored_hf', 'rotated_180'):
                res['flags'].append(f'orientation:{orient}')
            elif orient and orient != 'L\\F':
                res['flags'].append(f'orientation:nonstandard({orient})')
            row['quality_class'] = int(bool(res['violations']))
            row['quality_probability'] = res.get('quality_probability')
            for key, pv in res.get('probabilities', {}).items():
                row[f'p_{key}'] = pv
            row['violation_type'] = ';'.join(res['violations'])
            row['violation_description'] = '; '.join(VIOLATIONS[v] for v in res['violations'])
            row['region_confidence'] = res['region_confidence']
            row['region_accepted'] = res['region_accepted']
            row['flags'] = ';'.join(res['flags'])
            row['details'] = res['details']
            row['error'] = ''
        except Exception as e:
            row['processing_status'] = 'Failure'
            row['quality_class'] = 1          # необработанный кадр не считается качественным
            row['violation_type'] = ''
            row['violation_description'] = ''
            row['error'] = f'{type(e).__name__}: {e}'
            row['traceback'] = traceback.format_exc(limit=4)
            row['details'] = {}
            row['flags'] = ''
        row['time_of_processing'] = round(time.perf_counter() - t0, 4)
        row['file_name'] = frame.path.name
        row['duplicates'] = len(frame.duplicates)
        return row


# --------------------------------------------------------------------------- #
#  Обработка исследования и пачки
# --------------------------------------------------------------------------- #
def process_study(study: str | Path, analyzer: Analyzer,
                  path_to_study: str | None = None) -> list[dict]:
    """Одно исследование -> строки отчёта (по одной на уникальный кадр)."""
    study = Path(study)
    shown = path_to_study or str(study)
    frames, failed = unique_frames(study)

    rows = [analyzer.analyze_frame(f, shown) for f in frames]
    for path, err in failed:
        rows.append({
            'path_to_study': shown, 'study_uid': '', 'image_uid': '',
            'anatomical_region': 'unknown', 'quality_class': 1, 'violation_type': '',
            'violation_description': '', 'processing_status': 'Failure',
            'time_of_processing': 0.0, 'error': f'{type(err).__name__}: {err}',
            'file_name': path.name, 'flags': '', 'details': {}, 'duplicates': 0,
        })
    if not rows:
        rows.append({
            'path_to_study': shown, 'study_uid': '', 'image_uid': '',
            'anatomical_region': 'unknown', 'quality_class': 1, 'violation_type': '',
            'violation_description': '', 'processing_status': 'Failure',
            'time_of_processing': 0.0, 'error': 'в исследовании не найдено DICOM-снимков',
            'file_name': '', 'flags': '', 'details': {}, 'duplicates': 0,
        })
    return rows


def process_batch(root: str | Path, analyzer: Analyzer | None = None,
                  progress=None) -> list[dict]:
    """Архив/папка с исследованиями -> строки отчёта по всем кадрам."""
    from .dicom_io import study_dirs
    analyzer = analyzer or Analyzer.load()
    root = Path(root)
    studies = study_dirs(root)
    rows: list[dict] = []
    for i, s in enumerate(studies, 1):
        rel = s.relative_to(root) if s != root and root in s.parents else s.name
        rows += process_study(s, analyzer, str(rel))
        if progress:
            progress(i, len(studies), s)
    return rows
