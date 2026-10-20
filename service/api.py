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
    GET  /jobs/{id}/overlays/{name}  один кадр из этого zip (PNG) — для интерфейса
    GET  /                       веб-интерфейс (ТЗ 2.6)
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
    import zipfile

    from fastapi import Header, BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as e:                    # noqa: F841
    raise SystemExit('нужен fastapi: pip install fastapi uvicorn python-multipart') from e

from .dicom_io import extract_archive, read_frame
from .pipeline import REGION_RU, VIOLATIONS, Analyzer, process_batch
from .report import summary, write_details, write_overlays, write_sr, write_table

VERSION = '1.0.0'

# Встроенные страницы /docs и /redoc FastAPI подгружают Swagger UI и ReDoc с
# CDN — единственное место, откуда интерфейс мог бы пойти в интернет. Поэтому
# они выключены, а /docs собирается ниже из копии Swagger UI в static/swagger.
app = FastAPI(title='DXA Quality Control', version=VERSION,
              description='Сервис оценки качества денситометрических исследований',
              docs_url=None, redoc_url=None)

_analyzer: Analyzer | None = None
_lock = threading.Lock()
JOBS_DIR = Path(tempfile.gettempdir()) / 'dxa_qc_jobs'


def analyzer() -> Analyzer:
    """Модели грузятся один раз на процесс: чтение весов дороже самого анализа.

    DXA_QC_KEYPOINTS=1 подключает диагностические точки. Отдельные переменные
    DXA_QC_KEYPOINTS_MULTICROP=1 и DXA_QC_KEYPOINTS_VERDICT=1 включают
    экспериментальную нарезку и разрешают подмену контура. DXA_QC_CNN=0 выключает
    (cnn_qc) — вердикт тогда выносит одна геометрия. Переменной окружения, а
    не параметра запроса: веса читаются при старте процесса, переключать
    измеритель между запросами нечестно — отчёты станут несравнимы между собой.
    """
    global _analyzer
    with _lock:
        if _analyzer is None:
            kp_enabled = any(
                os.environ.get(name, '') not in ('', '0', 'false')
                for name in ('DXA_QC_KEYPOINTS', 'DXA_QC_KEYPOINTS_MULTICROP',
                             'DXA_QC_KEYPOINTS_VERDICT')
            )
            _analyzer = Analyzer.load(
                keypoints=kp_enabled,
                keypoints_dir=os.environ.get('DXA_QC_KEYPOINTS_DIR') or None,
                keypoints_multicrop=os.environ.get(
                    'DXA_QC_KEYPOINTS_MULTICROP', '') not in ('', '0', 'false'),
                keypoints_verdict=os.environ.get(
                    'DXA_QC_KEYPOINTS_VERDICT', '') not in ('', '0', 'false'),
                cnn=os.environ.get('DXA_QC_CNN', '1') not in ('0', 'false'),
                roi_rule=os.environ.get('DXA_QC_ROI_RULE', 'margins'))
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
    owner: str = ''                     # идентификатор браузера, загрузившего архив

    def public(self) -> dict:
        return {'id': self.id, 'state': self.state, 'progress': f'{self.done}/{self.total}',
                'created': self.created, 'finished': self.finished,
                'summary': self.summary, 'error': self.error,
                'report': f'/jobs/{self.id}/report' if self.state == 'done' else None}


JOBS: dict[str, Job] = {}

# Сколько хранить результаты завершённого задания. Без срока каталог заданий
# растёт без предела, и долго работающий сервис однажды заполнит диск.
JOB_TTL_HOURS = float(os.environ.get('DXA_QC_JOB_TTL_HOURS', '24'))


def _cleanup_jobs(now: float | None = None) -> None:
    """Удаляет завершённые задания старше JOB_TTL_HOURS вместе с их файлами.

    Каталоги, которых нет в JOBS, остались от прошлого запуска процесса: открыть
    их уже нельзя, поэтому они удаляются по тому же сроку.
    """
    now = time.time() if now is None else now
    for job_id, job in list(JOBS.items()):
        if job.finished is not None and now - job.finished > JOB_TTL_HOURS * 3600:
            _drop_job(job_id)
    if JOBS_DIR.exists():
        for d in JOBS_DIR.iterdir():
            try:
                if d.name not in JOBS and now - d.stat().st_mtime > JOB_TTL_HOURS * 3600:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass


def _drop_job(job_id: str) -> None:
    job = JOBS.pop(job_id, None)
    if job is not None and job.workdir is not None:
        shutil.rmtree(job.workdir, ignore_errors=True)


def _client_id(value: str | None) -> str:
    """Заголовок X-DXA-Client: случайный идентификатор браузера, не секрет пользователя."""
    v = (value or '').strip()
    return v if 8 <= len(v) <= 64 and all(c.isalnum() or c == '-' for c in v) else ''


def _run_job(job: Job, data_dir: Path, fmt: str, overlays: bool, sr: bool = False,
             all_frames: bool = False) -> None:
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
                write_overlays(rows, frames, job.workdir / 'overlays.zip',
                               only_violations=not all_frames)
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
        cnn = a.cnn
        nets = {f'cnn_{c}': bool(callable(getattr(cnn, 'available', None)) and cnn.available(c))
                for c in ('rotation', 'artifacts')}
        kp = a.keypoints
        return {'status': 'ok', 'models': {'region_clf': True,
                                           'spine_appearance': a.spine.appearance.available,
                                           'keypoints': bool(kp and getattr(kp, 'available', False)),
                                           **nets}}
    except Exception as e:
        raise HTTPException(503, f'модели не загружены: {e}')


@app.get('/version')
def version() -> dict:
    a = analyzer()
    from hip_rotation import THRESHOLDS as ROT
    from hip_roi.geometry import BOTTOM_MM, LAT_MM, TOP_MM
    from spine_qc.geometry import AXIS_LIMIT_DEG
    kp = a.keypoints
    return {'version': VERSION, 'hip_method': a.hip_method, 'roi_rule': a.roi_rule,
            'keypoints_requested': kp is not None,
            'keypoints_available': bool(kp and getattr(kp, 'available', False)),
            'keypoints_error': getattr(kp, 'error', None) if kp is not None else None,
            'keypoints_multicrop_active': bool(kp and getattr(kp, 'multicrop', False)),
            'keypoints_axis_verdict_active': bool(
                kp and getattr(kp, 'use_for_verdict', False)
                and getattr(kp, 'axis_gate_passed', False)),
            'spine_thresholds': {k: v for k, v in a.spine.thresholds.items()
                                 if not isinstance(v, dict)},
            'rotation_corridor_mm2': [ROT['area_lo'], ROT['area_hi']],
            # пороги, по которым выносится вердикт, — чтобы интерфейс показывал
            # настоящие нормы, а не зашитые в страницу числа
            'roi_margins_mm': {'top': TOP_MM, 'bottom': BOTTOM_MM, 'lat': LAT_MM},
            'axis_limit_deg_tz': AXIS_LIMIT_DEG,
            'keypoints': bool(kp and getattr(kp, 'available', False)),
            'cnn': {c: {'threshold': m.get('threshold'), 'stack': m.get('stack')}
                    for c, m in getattr(a.cnn, 'meta', {}).items()},
            'region_classes': list(REGION_RU),
            'job_ttl_hours': JOB_TTL_HOURS}


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
        'projection': 'AP' if res['anatomical_region'] in ('spine', 'lh', 'rh') else 'unknown',
        'quality_class': int(bool(res['violations'])),
        'violation_type': ';'.join(res['violations']),
        'violation_description': [VIOLATIONS[v] for v in res['violations']],
        'scoliosis': res.get('scoliosis', ''),
        'region_confidence': res.get('region_confidence'),
        'quality_probability': res.get('quality_probability'),
        'probabilities': res.get('probabilities', {}),
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
        img = render(px, {'anatomical_region': res['anatomical_region'],
                          'details': res.get('details', {})})
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
                all_frames: bool = Query(False, description='оверлеи и для качественных кадров'),
                wait: bool = Query(False, description='дождаться результата в этом же запросе'),
    x_dxa_client: str | None = Header(None, description='идентификатор клиента: '
                                      'GET /jobs покажет задание только ему')):
    """Zip с исследованиями -> задание на пакетную обработку."""
    _cleanup_jobs()
    # Полный uuid: на открытом стенде номер задания — единственный ключ к
    # чужим результатам, подобрать его перебором нельзя.
    job = Job(id=uuid.uuid4().hex, owner=_client_id(x_dxa_client))
    job.workdir = JOBS_DIR / job.id
    job.workdir.mkdir(parents=True, exist_ok=True)
    data = job.workdir / 'data'
    archive = job.workdir / 'upload.zip'
    # Потоком на диск, а не file.read(): архив закрытого набора может весить
    # гигабайты, держать его целиком в памяти процесса незачем.
    with archive.open('wb') as fh:
        shutil.copyfileobj(file.file, fh, length=1 << 20)
    try:
        extract_archive(archive, data)
    except Exception as e:
        shutil.rmtree(job.workdir, ignore_errors=True)
        raise HTTPException(400, f'не удалось распаковать архив: {e}')
    archive.unlink(missing_ok=True)
    JOBS[job.id] = job

    if wait:
        _run_job(job, data, fmt, overlays, sr, all_frames)
        return JSONResponse(job.public())
    background.add_task(_run_job, job, data, fmt, overlays, sr, all_frames)
    return JSONResponse(job.public(), status_code=202)


@app.get('/jobs')
def jobs(x_dxa_client: str | None = Header(None)) -> dict:
    """Задания этого клиента (заголовок X-DXA-Client). Чужие задания не
    показываются: сервис может быть общим для нескольких пользователей."""
    _cleanup_jobs()
    owner = _client_id(x_dxa_client)
    mine = [j for j in JOBS.values() if owner and j.owner == owner]
    return {'jobs': [j.public() for j in sorted(mine, key=lambda j: -j.created)]}


@app.delete('/jobs/{job_id}')
def job_delete(job_id: str) -> dict:
    """Удалить задание и все его файлы с сервера (таблица, оверлеи, SR)."""
    job = _job(job_id)
    if job.state in ('queued', 'running'):
        raise HTTPException(409, 'задание ещё выполняется')
    _drop_job(job_id)
    return {'deleted': job_id}


def _job(job_id: str) -> Job:
    _cleanup_jobs()
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


@app.get('/jobs/{job_id}/overlays/{name}')
def job_overlay_one(job_id: str, name: str) -> StreamingResponse:
    """Один кадр из архива оверлеев.

    Пиксели задания удаляются сразу после обработки, перерисовать кадр нельзя —
    поэтому интерфейс берёт готовую картинку из уже собранного zip.
    """
    job = _job(job_id)
    archive = (job.workdir or Path('.')) / 'overlays.zip'
    if job.state != 'done' or not archive.exists():
        raise HTTPException(404, 'оверлеи для этого задания не собраны')
    # Интерфейс просит кадр по image_uid, оператор может знать имя записи —
    # принимаем оба варианта.
    safe = Path(name).name
    key = safe[:-4] if safe.lower().endswith('.png') else safe
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        match = next((n for n in names if Path(n).name == safe), None)
        if match is None and key:
            match = next((n for n in names if n.endswith(f'__{key[-12:]}.png')), None)
        if match is None:
            raise HTTPException(404, 'кадр не найден в архиве')
        data = z.read(match)
    return StreamingResponse(io.BytesIO(data), media_type='image/png')


@app.get('/jobs/{job_id}/sr')
def job_sr(job_id: str) -> FileResponse:
    return _artifact(job_id, ['sr.zip'], 'application/zip', f'sr_{job_id}')


# --------------------------------------------------------------------------- #
#  Веб-интерфейс (ТЗ 2.6)
# --------------------------------------------------------------------------- #
# Раздаём страницу из самого сервиса, а не отдельным origin: тогда не нужен
# CORS, контейнер остаётся один, и утверждение «изображения за пределы
# контейнера не уходят» остаётся верным.
STATIC = Path(__file__).resolve().parent / 'static'
if STATIC.exists():
    app.mount('/static', StaticFiles(directory=STATIC), name='static')

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / 'index.html').read_text(encoding='utf-8'))

    # Веб-приложение (PWA): манифест и сервис-воркер отдаются с корня — только
    # так область действия воркера покрывает весь интерфейс.
    @app.get('/manifest.webmanifest', include_in_schema=False)
    def manifest() -> FileResponse:
        return FileResponse(STATIC / 'manifest.webmanifest', media_type='application/manifest+json')

    @app.get('/sw.js', include_in_schema=False)
    def service_worker() -> FileResponse:
        return FileResponse(STATIC / 'sw.js', media_type='text/javascript',
                            headers={'Cache-Control': 'no-cache'})

    @app.get('/favicon.ico', include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(STATIC / 'icons' / 'favicon-32.png', media_type='image/png')

    # Описание API (Swagger UI) — из локальной копии static/swagger, без CDN:
    # приложение и контейнер не обращаются в интернет ни одной страницей.
    if (STATIC / 'swagger' / 'swagger-ui-bundle.js').exists():
        from fastapi.openapi.docs import get_swagger_ui_html

        @app.get('/docs', include_in_schema=False)
        def api_docs() -> HTMLResponse:
            return get_swagger_ui_html(openapi_url=app.openapi_url or '/openapi.json',
                                       title=f'{app.title} — API',
                                       swagger_js_url='/static/swagger/swagger-ui-bundle.js',
                                       swagger_css_url='/static/swagger/swagger-ui.css',
                                       swagger_favicon_url='/static/icons/favicon-32.png')
