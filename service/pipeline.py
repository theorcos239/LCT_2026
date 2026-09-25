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


@dataclass
class Analyzer:
    """Загруженные модели. Создаётся один раз на процесс — веса читаются с диска."""
    region: object
    spine: object
    hip_method: str = 'kit2'
    strict_region: bool = True      # непринятый кадр не отправляем на критерии

    @classmethod
    def load(cls, hip_method: str = 'kit2', strict_region: bool = True) -> 'Analyzer':
        from region_clf import RegionClassifier
        from spine_qc import SpineQC
        return cls(region=RegionClassifier(), spine=SpineQC(),
                   hip_method=hip_method, strict_region=strict_region)

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

        if region == 'spine':
            res = self.spine.analyze(px)
            out['details']['spine'] = res['criteria']
            out['flags'] += res['flags']
            for crit, key in (('position', 'spine_position'), ('axis', 'spine_axis'),
                              ('artifacts', 'spine_artifacts')):
                if res['criteria'][crit]['violated']:
                    out['violations'].append(key)
        else:
            from hip_roi.kits import measure_roi_margins
            from hip_rotation import THRESHOLDS as ROT_THR
            from hip_rotation import detect as detect_rotation

            roi = measure_roi_margins(px, region, self.hip_method)
            out['details']['hip_roi'] = {k: v for k, v in roi.items() if k != 'diag'}
            out['flags'] += list(roi.get('flags', []))
            if roi.get('roi_ok') is False:
                out['violations'].append('hip_roi')

            # Ротация берёт ту же верхушку большого вертела, что и отступы ROI:
            # один ориентир на оба критерия бедра, а не два независимых.
            rot = detect_rotation(px, region, thr=ROT_THR, T=roi.get('T_px'))
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
            'time_of_processing': 0.0, 'error': 'в исследовании не найдено DICOM-файлов',
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
