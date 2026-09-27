# -*- coding: utf-8 -*-
"""HTTP API сервиса контроля качества DXA (ТЗ 3.2).

    uvicorn service.api:app --host 0.0.0.0 --port 8000

Эндпоинты:

    GET  /health                 жив ли сервис и загружены ли модели
    GET  /version                версии, модели, пороги
    GET  /taxonomy               перечень типов нарушений
    POST /analyze                один DICOM-файл -> вердикт в JSON
    POST /analyze/overlay        один DICOM-файл -> PNG с визуализацией
    POST /analyze/sr             один DICOM-файл -> DICOM SR с текстовым заключением
    POST /batch                  zip с исследованиями -> задание на обработку
    GET  /jobs                   список заданий
    GET  /jobs/{id}              статус и сводка
    GET  /jobs/{id}/report       итоговая таблица (.xlsx или .csv)
    GET  /jobs/{id}/details      все измерения в JSON
    GET  /jobs/{id}/overlays     zip с визуализацией нарушений
    GET  /jobs/{id}/sr           zip с DICOM SR на каждый кадр

Пакетная обработка идёт заданием в фоне, а не в запросе: закрытый тестовый
набор может оказаться большим, а держать HTTP-соединение открытым полчаса —
надёжный способ получить таймаут на прокси. Для маленьких пачек есть
`?wait=true`, тогда ответ приходит сразу готовым.

Всё работает локально: ни один эндпоинт никуда не ходит по сети, изображения
за пределы контейнера не уходят (ТЗ 3.2).
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
except ImportError as e:                    # noqa: F841
    raise SystemExit('нужен fastapi: pip install fastapi uvicorn python-multipart') from e

from .dicom_io import extract_archive, read_frame
from .pipeline import REGION_RU, VIOLATIONS, Analyzer, process_batch
from .report import summary, write_details, write_overlays, write_sr, write_table

VERSION = '1.0.0'

app = FastAPI(title='DXA Quality Control', version=VERSION,
              description='Сервис оценки качества денситометрических исследований')

_analyzer: Analyzer | None = None
_lock = threading.Lock()
JOBS_DIR = Path(tempfile.gettempdir()) / 'dxa_qc_jobs'


def analyzer() -> Analyzer:
    """Модели грузятся один раз на процесс: чтение весов дороже самого анализа.

    DXA_QC_KEYPOINTS=1 подключает модель ключевых точек (нужен torch и веса в
    runs/keypoints). Переменной окружения, а не параметра запроса: веса читаются
    при старте процесса, переключать измеритель между запросами нечестно —
    отчёты станут несравнимы между собой.
    """
    global _analyzer
    with _lock:
        if _analyzer is None:
            _analyzer = Analyzer.load(
                keypoints=os.environ.get('DXA_QC_KEYPOINTS', '') not in ('', '0', 'false'))
    return _analyzer


# --------------------------------------------------------------------------- #
#  Задания
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    state: str = 'queued'               # queued | running | done | failed
    created: float = field(default_factory=time.time)
    finished: float | None = None
    total: int = 0
    done: int = 0
    summary: dict = field(default_factory=dict)
    error: str = ''
    workdir: Path | None = None

    def public(self) -> dict:
        return {'id': self.id, 'state': self.state, 'progress': f'{self.done}/{self.total}',
                'created': self.created, 'finished': self.finished,
                'summary': self.summary, 'error': self.error,
                'report': f'/jobs/{self.id}/report' if self.state == 'done' else None}


JOBS: dict[str, Job] = {}


def _run_job(job: Job, data_dir: Path, fmt: str, overlays: bool, sr: bool = False) -> None:
    try:
        job.state = 'running'

        def progress(i, n, _study):
            job.done, job.total = i, n

        rows = process_batch(data_dir, analyzer(), progress=progress)
        out = job.workdir / f'report.{fmt}'
        write_table(rows, out)
        write_details(rows, job.workdir / 'details.json')
        if overlays or sr:
            from .dicom_io import study_dirs, unique_frames
            frames = {}
            for s in study_dirs(data_dir):
                for fr in unique_frames(s)[0]:
                    frames[fr.image_uid] = fr
            if overlays:
                write_overlays(rows, frames, job.workdir / 'overlays.zip')
            if sr:
                write_sr(rows, frames, job.workdir / 'sr.zip')
        job.summary = summary(rows)
        job.state = 'done'
    except Exception as e:
        job.state, job.error = 'failed', f'{type(e).__name__}: {e}'
    finally:
        job.finished = time.time()
        shutil.rmtree(data_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
#  Служебные
# --------------------------------------------------------------------------- #
@app.get('/health')
def health() -> dict:
    try:
        a = analyzer()
        return {'status': 'ok', 'models': {'region_clf': True,
                                           'spine_appearance': a.spine.appearance.available}}
    except Exception as e:
        raise HTTPException(503, f'модели не загружены: {e}')


@app.get('/version')
def version() -> dict:
    a = analyzer()
    from hip_rotation import THRESHOLDS as ROT
    return {'version': VERSION, 'hip_method': a.hip_method,
            'spine_thresholds': {k: v for k, v in a.spine.thresholds.items()
                                 if not isinstance(v, dict)},
            'rotation_corridor_mm2': [ROT['area_lo'], ROT['area_hi']],
            'region_classes': list(REGION_RU)}


@app.get('/taxonomy')
def taxonomy() -> dict:
    return {'violations': VIOLATIONS, 'regions': REGION_RU,
            'quality_class': {'0': 'качественное', '1': 'есть нарушение'}}


# --------------------------------------------------------------------------- #
#  Один кадр
# --------------------------------------------------------------------------- #
def _read_upload(data: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as t:
        t.write(data)
        tmp = Path(t.name)
    try:
        return read_frame(tmp).pixels
    finally:
        tmp.unlink(missing_ok=True)


@app.post('/analyze')
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    """Один DICOM-файл -> область, класс качества, перечень нарушений."""
    t0 = time.perf_counter()
    try:
        px = _read_upload(await file.read())
    except Exception as e:
        raise HTTPException(400, f'не удалось прочитать DICOM: {e}')
    try:
        res = analyzer().analyze_pixels(px)
    except Exception as e:
        return JSONResponse({'processing_status': 'Failure', 'error': f'{type(e).__name__}: {e}',
                             'time_of_processing': round(time.perf_counter() - t0, 4)},
                            status_code=200)
    return JSONResponse({
        'file_name': file.filename,
        'anatomical_region': res['anatomical_region'],
        'anatomical_region_ru': REGION_RU.get(res['anatomical_region'], ''),
        'quality_class': int(bool(res['violations'])),
        'violation_type': ';'.join(res['violations']),
        'violation_description': [VIOLATIONS[v] for v in res['violations']],
        'flags': res['flags'],
        'details': res['details'],
        'processing_status': 'Success',
        'time_of_processing': round(time.perf_counter() - t0, 4),
    })


@app.post('/analyze/overlay')
async def analyze_overlay(file: UploadFile = File(...)) -> StreamingResponse:
    """Тот же кадр, но PNG с разметкой вердикта (ТЗ 2.6)."""
    from .overlay import render
    try:
        px = _read_upload(await file.read())
        res = analyzer().analyze_pixels(px)
        img = render(px, {'anatomical_region': res['anatomical_region']})
    except Exception as e:
        raise HTTPException(400, f'не удалось построить визуализацию: {e}')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')


@app.post('/analyze/sr')
async def analyze_sr(file: UploadFile = File(...)) -> StreamingResponse:
    """Тот же кадр, но DICOM SR с текстовым заключением (ТЗ 2.6).

    SR ссылается на исходный снимок и наследует теги пациента и исследования,
    поэтому PACS покажет его в том же исследовании.
    """
    import pydicom

    from .dicom_sr import build_sr
    data = await file.read()
    try:
        src = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
        px = _read_upload(data)
    except Exception as e:
        raise HTTPException(400, f'не удалось прочитать DICOM: {e}')
    try:
        res = analyzer().analyze_pixels(px)
        sr = build_sr(src, res, VIOLATIONS, REGION_RU)
        buf = io.BytesIO()
        sr.save_as(buf, enforce_file_format=True)
    except Exception as e:
        raise HTTPException(500, f'не удалось построить SR: {type(e).__name__}: {e}')
    buf.seek(0)
    name = Path(file.filename or 'image').stem
    return StreamingResponse(buf, media_type='application/dicom',
                             headers={'Content-Disposition': f'attachment; filename="{name}.sr.dcm"'})


# --------------------------------------------------------------------------- #
#  Пачка
# --------------------------------------------------------------------------- #
@app.post('/batch')
async def batch(background: BackgroundTasks,
                file: UploadFile = File(..., description='zip с исследованиями'),
                fmt: str = Query('xlsx', pattern='^(xlsx|csv)$'),
                overlays: bool = Query(False, description='приложить zip с визуализацией'),
                sr: bool = Query(False, description='приложить zip с DICOM SR'),
                wait: bool = Query(False, description='дождаться результата в этом же запросе')):
    """Zip с исследованиями -> задание на пакетную обработку."""
    job = Job(id=uuid.uuid4().hex[:12])
    job.workdir = JOBS_DIR / job.id
    job.workdir.mkdir(parents=True, exist_ok=True)
    data = job.workdir / 'data'
    archive = job.workdir / 'upload.zip'
    archive.write_bytes(await file.read())
    try:
        extract_archive(archive, data)
    except Exception as e:
        shutil.rmtree(job.workdir, ignore_errors=True)
        raise HTTPException(400, f'не удалось распаковать архив: {e}')
    archive.unlink(missing_ok=True)
    JOBS[job.id] = job

    if wait:
        _run_job(job, data, fmt, overlays, sr)
        return JSONResponse(job.public())
    background.add_task(_run_job, job, data, fmt, overlays, sr)
    return JSONResponse(job.public(), status_code=202)


@app.get('/jobs')
def jobs() -> dict:
    return {'jobs': [j.public() for j in sorted(JOBS.values(), key=lambda j: -j.created)]}


def _job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, 'задание не найдено')
    return job


@app.get('/jobs/{job_id}')
def job_status(job_id: str) -> dict:
    return _job(job_id).public()


def _artifact(job_id: str, patterns: list[str], media: str, name: str) -> FileResponse:
    job = _job(job_id)
    if job.state != 'done':
        raise HTTPException(409, f'задание в состоянии {job.state}')
    for p in patterns:
        f = job.workdir / p
        if f.exists():
            return FileResponse(f, media_type=media, filename=f'{name}{f.suffix}')
    raise HTTPException(404, 'файл не найден для этого задания')


@app.get('/jobs/{job_id}/report')
def job_report(job_id: str) -> FileResponse:
    return _artifact(job_id, ['report.xlsx', 'report.csv'],
                     'application/octet-stream', f'report_{job_id}')


@app.get('/jobs/{job_id}/details')
def job_details(job_id: str) -> FileResponse:
    return _artifact(job_id, ['details.json'], 'application/json', f'details_{job_id}')


@app.get('/jobs/{job_id}/overlays')
def job_overlays(job_id: str) -> FileResponse:
    return _artifact(job_id, ['overlays.zip'], 'application/zip', f'overlays_{job_id}')


@app.get('/jobs/{job_id}/sr')
def job_sr(job_id: str) -> FileResponse:
    return _artifact(job_id, ['sr.zip'], 'application/zip', f'sr_{job_id}')
