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

    @classmethod
    def load(cls, hip_method: str = 'kit2', strict_region: bool = True,
             keypoints: bool = False, keypoints_dir=None) -> 'Analyzer':
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
        kp = None
        if keypoints:
            try:
                from .keypoints_backend import KeypointBackend
                kp = KeypointBackend(keypoints_dir)
            except Exception as e:      # noqa: BLE001 — нет torch, нет src/dxa_qc
                kp = _Unavailable(f'{type(e).__name__}: {e}')
        return cls(region=RegionClassifier(), spine=SpineQC(),
                   hip_method=hip_method, strict_region=strict_region, keypoints=kp)

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
            for crit, key in (('position', 'spine_position'), ('axis', 'spine_axis'),
                              ('artifacts', 'spine_artifacts')):
                if res['criteria'][crit]['violated']:
                    out['violations'].append(key)
        else:
            from hip_roi.kits import measure_roi_margins
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
            if roi.get('roi_ok') is False:
                out['violations'].append('hip_roi')

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
            out['details']['hip_rotation'] = rot
            out['flags'] += list(rot.get('flags', []))
            if rot.get('violated'):
                out['violations'].append('hip_rotation')
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
            row['quality_class'] = int(bool(res['violations']))
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
