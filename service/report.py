# -*- coding: utf-8 -*-
"""Итоговая таблица и дополнительные серии с визуализацией.

Формат колонок задан ТЗ 2.5 и соблюдается буквально: первые восемь колонок
идут в указанном порядке и с указанными именами. Всё остальное, что считает
конвейер (уверенность классификатора области, флаги, измерения в миллиметрах),
добавляется правее — это не мешает проверяющему скрипту читать обязательную
часть, но позволяет разобрать любое решение вручную.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from .pipeline import COLUMNS

EXTRA = ['violation_description', 'quality_probability', 'p_spine_position',
         'p_spine_axis', 'p_spine_artifacts', 'region_confidence', 'region_accepted',
         'file_name', 'duplicates', 'flags', 'error']


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Строки конвейера -> DataFrame с обязательными колонками впереди."""
    df = pd.DataFrame(rows)
    for c in COLUMNS + EXTRA:
        if c not in df.columns:
            df[c] = ''
    df['quality_class'] = df['quality_class'].fillna(1).astype(int)
    df['time_of_processing'] = df['time_of_processing'].astype(float).round(4)
    ordered = COLUMNS + [c for c in EXTRA if c in df.columns]
    rest = [c for c in df.columns if c not in ordered and c not in ('details', 'traceback')]
    return df[ordered + rest]


def write_table(rows: list[dict], path: str | Path) -> Path:
    """CSV или XLSX — по расширению.

    CSV пишется в чистом utf-8, БЕЗ BOM. utf-8-sig удобнее для Excel, но
    стандартный `csv.reader`, открывший файл как utf-8, получает первую
    колонку с именем `﻿path_to_study` — а проверяющий скрипт ищет
    `path_to_study`. pandas BOM переваривает, `csv` нет, и рисковать разбором
    отчёта ради удобства двойного клика не стоит: кому нужен Excel, тот берёт
    .xlsx (формат по умолчанию), где проблемы кодировки нет вовсе.
    """
    p = Path(path)
    df = to_frame(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in ('.xlsx', '.xlsm'):
        df.to_excel(p, index=False, sheet_name='quality')
    else:
        df.to_csv(p, index=False, encoding='utf-8')
    return p


def write_details(rows: list[dict], path: str | Path) -> Path:
    """Все измерения в JSON: то, чего не видно в таблице (мм, градусы, флаги)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [{'path_to_study': r.get('path_to_study'), 'image_uid': r.get('image_uid'),
                'file_name': r.get('file_name'),
                'anatomical_region': r.get('anatomical_region'),
                'quality_class': r.get('quality_class'),
                'quality_probability': r.get('quality_probability'),
                'violation_type': r.get('violation_type'),
                'violation_description': r.get('violation_description'),
                'processing_status': r.get('processing_status'),
                'time_of_processing': r.get('time_of_processing'),
                'flags': r.get('flags'),
                'details': r.get('details', {})} for r in rows]
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return p


def summary(rows: list[dict]) -> dict:
    """Сводка для лога и ответа API."""
    df = to_frame(rows)
    ok = df.processing_status == 'Success'
    types: dict[str, int] = {}
    for v in df.violation_type:
        for t in str(v).split(';'):
            if t:
                types[t] = types.get(t, 0) + 1
    return {
        'images': int(len(df)),
        'studies': int(df.path_to_study.nunique()),
        'success': int(ok.sum()),
        'failure': int((~ok).sum()),
        'with_violation': int(df.loc[ok, 'quality_class'].sum()),
        'by_region': df.loc[ok, 'anatomical_region'].value_counts().to_dict(),
        'by_violation': dict(sorted(types.items())),
        'seconds_total': round(float(df.time_of_processing.sum()), 2),
        'seconds_per_image_median': round(float(df.time_of_processing.median()), 4),
    }


def write_overlays(rows: list[dict], frames_by_uid: dict, path: str | Path,
                   only_violations: bool = True) -> Path:
    """Zip с визуализацией нарушений (ТЗ 2.6 и 2.7).

    По умолчанию кладём только кадры с нарушением: архив со всеми кадрами
    подряд оператор не откроет, а разбирать он будет именно эти. Веб-интерфейс
    просит все (`only_violations=False`), чтобы показывать и качественные кадры
    с разметкой — «вот почему снимок принят».
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import io

    from .overlay import render
    with zipfile.ZipFile(p, 'w', zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            if r.get('processing_status') != 'Success':
                continue
            if only_violations and not r.get('quality_class'):
                continue
            frame = frames_by_uid.get(r.get('image_uid'))
            if frame is None:
                continue
            try:
                img = render(frame.pixels, r)
            except Exception:
                continue                     # визуализация не критична для отчёта
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            # Имя записи содержит уникальный хвост image_uid: внутри исследования файлы
            # называются одинаково (CR000000.dcm), и без него записи перетирались бы.
            uid = str(r.get('image_uid') or '')
            stem = Path(str(r.get('file_name') or uid or 'frame')).stem
            name = f"{r.get('path_to_study', 'study')}/{stem}__{uid[-12:]}.png"
            z.writestr(name.replace('\\', '/'), buf.getvalue())
    return p


def write_sr(rows: list[dict], frames_by_uid: dict, path: str | Path) -> Path:
    """Zip с DICOM SR на каждый обработанный кадр (ТЗ 2.6).

    В отличие от оверлеев, SR пишется и для качественных кадров: вывод
    «нарушений не обнаружено» — тоже заключение, и PACS должен его видеть
    рядом со снимком. Заголовок исходного DICOM перечитывается без пикселей —
    теги пациента и исследования в Frame не хранятся.
    """
    import io

    import pydicom

    from .dicom_sr import build_sr
    from .pipeline import REGION_RU, VIOLATIONS

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, r in enumerate(rows, 1):
            if r.get('processing_status') != 'Success':
                continue
            frame = frames_by_uid.get(r.get('image_uid'))
            if frame is None:
                continue
            try:
                src = pydicom.dcmread(str(frame.path), stop_before_pixels=True)
                sr = build_sr(src, r, VIOLATIONS, REGION_RU, instance_number=i)
                buf = io.BytesIO()
                sr.save_as(buf, enforce_file_format=True)
            except Exception:
                continue                     # SR — дополнительная серия, отчёт важнее
            name = f"{r.get('path_to_study', 'study')}/{r.get('file_name', r.get('image_uid'))}.sr.dcm"
            z.writestr(name.replace('\\', '/'), buf.getvalue())
    return p
