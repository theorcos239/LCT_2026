# -*- coding: utf-8 -*-
"""Тесты сервиса: формат отчёта, дедупликация, устойчивость, воспроизводимость.

    python -m service.test_service              # всё
    python -m service.test_service --fast       # без прогона по всей выборке

Проверяется то, на чём решение может незаметно сломаться при сдаче: формат
колонок (его читает скрипт организатора), дедупликация (без неё строк втрое
больше), обработка мусора на входе (ТЗ 2.7: необработанных исключений быть
не должно) и воспроизводимость (ТЗ 2.7: повторный запуск даёт то же самое).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'НД_для_обучения' / 'Исследования'

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = '') -> None:
    (PASS if ok else FAIL).append(name)
    print(f'  [{"ok" if ok else "ПРОВАЛ"}] {name}{" — " + detail if detail else ""}')


# --------------------------------------------------------------------------- #
def test_report_format() -> None:
    """Первые восемь колонок — ровно те, что требует ТЗ 2.5, и в том же порядке."""
    from .pipeline import COLUMNS
    from .report import to_frame
    expected = ['path_to_study', 'study_uid', 'image_uid', 'anatomical_region',
                'quality_class', 'violation_type', 'processing_status', 'time_of_processing']
    check('COLUMNS совпадает с ТЗ 2.5', COLUMNS == expected)

    df = to_frame([{'path_to_study': 'x', 'quality_class': 1, 'time_of_processing': 0.1}])
    check('таблица строится из неполной строки', list(df.columns[:8]) == expected)
    check('quality_class целочисленный', df.quality_class.dtype.kind == 'i')
    check('time_of_processing вещественный', df.time_of_processing.dtype.kind == 'f')

    # BOM в CSV ломает stdlib csv.reader: колонка приезжает как '﻿path_to_study'
    import csv as _csv
    from .report import write_table
    tmp = Path(tempfile.mkdtemp(prefix='dxa_csv_'))
    try:
        f = write_table([{'path_to_study': 'x', 'quality_class': 0,
                          'time_of_processing': 0.1}], tmp / 'r.csv')
        with open(f, encoding='utf-8', newline='') as fh:
            first = next(_csv.reader(fh))[0]
        check('CSV читается стандартным csv.reader без BOM', first == 'path_to_study',
              repr(first))
        xl = write_table([{'path_to_study': 'x', 'quality_class': 0,
                           'time_of_processing': 0.1}], tmp / 'r.xlsx')
        import pandas as pd
        check('XLSX читается и колонки на месте',
              list(pd.read_excel(xl).columns[:8]) == expected)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_taxonomy() -> None:
    """Каждый тип нарушения, который умеет ставить конвейер, описан в таксономии."""
    from .pipeline import VIOLATIONS
    produced = {'spine_position', 'spine_axis', 'spine_artifacts',
                'hip_rotation', 'hip_roi', 'undetermined'}
    check('таксономия покрывает все типы', produced <= set(VIOLATIONS),
          f'нет описания: {produced - set(VIOLATIONS)}' if produced - set(VIOLATIONS) else '')


def test_bad_input() -> None:
    """Мусор на входе даёт строку Failure, а не исключение наружу."""
    from .pipeline import Analyzer, process_study
    tmp = Path(tempfile.mkdtemp(prefix='dxa_bad_'))
    try:
        (tmp / 'broken.dcm').write_bytes(b'not a dicom at all')
        (tmp / 'empty.dcm').write_bytes(b'')
        rows = process_study(tmp, Analyzer.load(), 'broken_study')
        check('битые файлы не роняют обработку', len(rows) >= 1)
        check('статус Failure проставлен', all(r['processing_status'] == 'Failure' for r in rows))
        check('ошибка записана в отчёт', all(r.get('error') for r in rows))
        check('непрочитанный кадр не считается качественным',
              all(r['quality_class'] == 1 for r in rows))

        empty = Path(tempfile.mkdtemp(prefix='dxa_empty_'))
        rows = process_study(empty, Analyzer.load(), 'empty_study')
        check('пустая папка даёт строку, а не падение', len(rows) == 1
              and rows[0]['processing_status'] == 'Failure')
        shutil.rmtree(empty, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pixel_robustness() -> None:
    """Нетипичные кадры проходят конвейер без исключений."""
    from .pipeline import Analyzer
    a = Analyzer.load()
    cases = {
        'нули': np.zeros((300, 300), np.float32),
        'константа': np.full((260, 280), 120.0, np.float32),
        'шум': np.random.default_rng(0).normal(120, 40, (300, 300)).astype(np.float32),
        'узкий': np.zeros((400, 40), np.float32),
        'крошечный': np.zeros((12, 12), np.float32),
        'огромные значения': np.full((300, 300), 1e6, np.float32),
        'отрицательные': np.full((300, 300), -500.0, np.float32),
    }
    ok = True
    for name, px in cases.items():
        try:
            r = a.analyze_pixels(px)
            assert isinstance(r['violations'], list)
        except Exception as e:
            ok = False
            print(f'      {name}: {type(e).__name__}: {e}')
    check('нетипичные кадры не роняют конвейер', ok)


def test_archive_traversal() -> None:
    """Архив не может записать файл за пределы каталога назначения."""
    import zipfile
    from .dicom_io import extract_archive
    tmp = Path(tempfile.mkdtemp(prefix='dxa_zip_'))
    try:
        arc = tmp / 'evil.zip'
        with zipfile.ZipFile(arc, 'w') as z:
            z.writestr('../../escaped.txt', 'x')
        try:
            extract_archive(arc, tmp / 'dest')
            check('path traversal в архиве отклонён', False, 'архив распаковался')
        except ValueError:
            check('path traversal в архиве отклонён', True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dedup_and_regions() -> None:
    """Дедупликация и определение области на всём обучающем наборе."""
    import trainset
    from .dicom_io import dicom_files, study_dirs, unique_frames
    from .pipeline import Analyzer

    files = sum(len(dicom_files(s)) for s in study_dirs(DATA))
    frames = []
    for s in study_dirs(DATA):
        frames += unique_frames(s)[0]
    check('дедупликация: 499 файлов -> 252 кадра', files == 499 and len(frames) == 252,
          f'{files} -> {len(frames)}')
    check('в исследовании не больше 3 кадров',
          max(sum(1 for f in frames if f.study_key == s.name) for s in study_dirs(DATA)) <= 3)

    expected = {r.rel_path.split('/')[0] + '|' + r.px_hash: r.label
                for r in trainset.frames().itertuples()}
    a = Analyzer.load()
    hits = total = 0
    for f in frames:
        key = f.study_key + '|' + f.px_hash
        if key not in expected:
            continue
        total += 1
        hits += int(a.analyze_pixels(f.pixels)['anatomical_region'] == expected[key])
    check('область определена верно на всех кадрах', hits == total, f'{hits}/{total}')


def test_determinism() -> None:
    """Два прогона одного исследования дают побитово одинаковый результат."""
    from .pipeline import Analyzer, process_study
    study = sorted(DATA.iterdir())[0]
    a = Analyzer.load()
    keys = ('anatomical_region', 'quality_class', 'violation_type', 'image_uid')
    r1 = [{k: r[k] for k in keys} for r in process_study(study, a, 's')]
    r2 = [{k: r[k] for k in keys} for r in process_study(study, a, 's')]
    check('повторный прогон даёт то же самое', r1 == r2)

    a2 = Analyzer.load()                      # модели перезагружены с нуля
    r3 = [{k: r[k] for k in keys} for r in process_study(study, a2, 's')]
    check('перезагрузка моделей ничего не меняет', r1 == r3)


def test_timing() -> None:
    """Время обработки исследования против лимита ТЗ 2.7 — 3 минуты."""
    from .pipeline import Analyzer, process_study
    a = Analyzer.load()
    times = []
    for s in sorted(DATA.iterdir())[:10]:
        t = time.perf_counter()
        process_study(s, a, s.name)
        times.append(time.perf_counter() - t)
    worst = max(times)
    check('исследование обрабатывается быстрее 180 с', worst < 180.0,
          f'худшее {worst:.2f} с, медиана {np.median(times):.2f} с')


def test_api() -> None:
    """Эндпоинты отвечают и отдают обязательные поля."""
    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError):
        # RuntimeError — так starlette сообщает об отсутствии httpx: в рабочем
        # образе тестовых зависимостей нет, и пропуск лучше падения всего набора.
        check('API (пропущено: нет fastapi или httpx)', True)
        return
    from .api import app
    c = TestClient(app)
    check('GET /health', c.get('/health').status_code == 200)
    check('GET /taxonomy', 'violations' in c.get('/taxonomy').json())
    dcm = next(sorted(DATA.iterdir())[0].rglob('*.dcm'))
    r = c.post('/analyze', files={'file': (dcm.name, dcm.read_bytes(), 'application/dicom')})
    j = r.json()
    need = {'anatomical_region', 'quality_class', 'violation_type',
            'processing_status', 'time_of_processing'}
    check('POST /analyze отдаёт поля ТЗ', r.status_code == 200 and need <= set(j))
    r = c.post('/analyze', files={'file': ('x.dcm', b'garbage', 'application/dicom')})
    check('POST /analyze отвергает мусор', r.status_code == 400)
    page = c.get('/')
    check('GET / отдаёт веб-интерфейс', page.status_code == 200 and 'id="drop"' in page.text
          and 'id="rows"' in page.text)
    v = c.get('/version').json()
    check('GET /version отдаёт пороги для интерфейса',
          {'roi_margins_mm', 'axis_limit_deg_tz', 'rotation_corridor_mm2', 'spine_thresholds'} <= set(v))


def test_keypoints() -> None:
    """Модель ключевых точек: выключена по умолчанию, включённая не меняет формат.

    Главное, что проверяется, — отсутствие модели не ломает сервис: без torch
    или без весов кадр обязан посчитаться контурной геометрией, а причина —
    попасть во флаг, а не потеряться.
    """
    from .pipeline import Analyzer, process_study
    study = sorted(DATA.iterdir())[0]

    check('по умолчанию модель точек выключена', Analyzer.load().keypoints is None)

    missing = Analyzer.load(keypoints=True, keypoints_dir=Path(tempfile.gettempdir()) / 'нет_весов')
    rows = process_study(study, missing, 's')
    check('без весов кадры считаются и помечаются флагом',
          all(r['processing_status'] == 'Success' for r in rows)
          and any('keypoints:unavailable' in r['flags'] for r in rows))

    a = Analyzer.load(keypoints=True)
    if a.keypoints is None or not a.keypoints.available:
        check('модель точек (пропущено: веса или torch недоступны)', True)
        return

    base = {r['image_uid']: r for r in process_study(study, Analyzer.load(), 's')}
    rows = process_study(study, a, 's')
    check('с моделью точек формат строки не меняется',
          all(set(r) >= set(base[r['image_uid']]) for r in rows))
    check('с моделью точек нет необработанных исключений',
          all(r['processing_status'] == 'Success' for r in rows))

    kp_rows = [r for r in rows if 'keypoints' in r['details']]
    check('измерения по точкам попали в детали', bool(kp_rows))
    spine = [r for r in rows if r['anatomical_region'] == 'spine']
    hip = [r for r in rows if r['anatomical_region'] in ('lh', 'rh')]
    check('у позвоночника ось посчитана точками',
          all(r['details']['spine']['axis'].get('source') == 'keypoints' for r in spine),
          f'кадров {len(spine)}')
    check('у бедра отступы посчитаны точками и контур сохранён',
          all(r['details']['hip_roi']['method'] == 'keypoints'
              and 'hip_roi_contour' in r['details'] for r in hip),
          f'кадров {len(hip)}')

    r2 = process_study(study, a, 's')
    keys = ('anatomical_region', 'quality_class', 'violation_type')
    check('повторный прогон с точками даёт то же самое',
          [{k: r[k] for k in keys} for r in rows] == [{k: r[k] for k in keys} for r in r2])


def test_cnn() -> None:
    """Нейросетевое второе мнение (cnn_qc): подключено по умолчанию и не роняет сервис.

    Как и с моделью точек, главное — поведение без неё: нет onnxruntime или
    весов — кадр считается геометрией, причина уходит во флаг.
    """
    from cnn_qc import CnnQC

    from .dicom_io import unique_frames
    from .overlay import render
    from .pipeline import Analyzer, process_study
    study = sorted(DATA.iterdir())[0]

    a = Analyzer.load()
    ok = callable(getattr(a.cnn, 'available', None)) and all(
        a.cnn.available(c) for c in ('rotation', 'artifacts'))
    check('сеть cnn_qc подключена по умолчанию', ok, str(getattr(a.cnn, 'errors', '')))

    missing = Analyzer.load()
    missing.cnn = CnnQC(root=Path(tempfile.gettempdir()) / 'нет_сети')
    rows = process_study(study, missing, 's')
    check('без весов сети кадры считаются геометрией и помечаются флагом',
          all(r['processing_status'] == 'Success' for r in rows)
          and any('cnn:unavailable' in r['flags'] for r in rows))
    if not ok:
        return

    rows = process_study(study, a, 's')
    spine = [r for r in rows if r['anatomical_region'] == 'spine']
    hip = [r for r in rows if r['anatomical_region'] in ('lh', 'rh')]
    check('артефакты позвоночника — свод геометрии и сети',
          all(r['details']['spine']['artifacts'].get('p_cnn') is not None for r in spine),
          f'кадров {len(spine)}')
    check('ротация бедра — свод геометрии и сети',
          all(r['details']['hip_rotation'].get('p_cnn') is not None for r in hip),
          f'кадров {len(hip)}')
    check('вероятность нарушения есть у каждого кадра',
          all(r.get('quality_probability') is not None for r in rows))
    r2 = process_study(study, a, 's')
    keys = ('quality_class', 'violation_type', 'quality_probability')
    check('повторный прогон с сетью даёт то же самое',
          [{k: r[k] for k in keys} for r in rows] == [{k: r[k] for k in keys} for r in r2])

    frames, _ = unique_frames(study)
    by_uid = {f.image_uid: f for f in frames}
    imgs = [render(by_uid[r['image_uid']].pixels, r) for r in rows if r['image_uid'] in by_uid]
    check('визуализация строится по деталям вердикта', len(imgs) == len(rows)
          and all(i.size[0] > 0 for i in imgs))


def test_roi_rule() -> None:
    """Отступы ROI: вердикт по длине поля (как эксперт), отступы по рисунку 6 — в деталях."""
    from .dicom_io import unique_frames
    from .pipeline import Analyzer
    study = sorted(DATA.iterdir())[0]
    frames, _ = unique_frames(study)
    a, m = Analyzer.load(cnn=False), Analyzer.load(cnn=False, roi_rule='margins')
    hips = [(a.analyze_pixels(f.pixels), m.analyze_pixels(f.pixels)) for f in frames]
    hips = [(x, y) for x, y in hips if x['anatomical_region'] in ('lh', 'rh')]
    ok = bool(hips) and all(
        x['details']['hip_roi']['roi_rule'] == 'scan_length'
        and x['details']['hip_roi']['roi_ok'] == x['details']['hip_roi']['field_ok']
        and x['details']['hip_roi']['margins_ok'] == y['details']['hip_roi']['roi_ok']
        and x['details']['hip_roi']['m_top_mm'] == y['details']['hip_roi']['m_top_mm']
        for x, y in hips)
    check('ROI: вердикт по длине поля, отступы по рисунку 6 сохранены', ok, f'кадров {len(hips)}')


def test_orientation() -> None:
    """Зеркальная выгрузка (PatientOrientation = R, F) даёт тот же вердикт и ту же сторону."""
    import pydicom

    from .dicom_io import read_frame
    from .pipeline import Analyzer
    a = Analyzer.load(cnn=False)
    hip = next(p for p in sorted(DATA.rglob('*.dcm'))
               if a.analyze_pixels(read_frame(p).pixels)['anatomical_region'] == 'lh')
    base = a.analyze_frame(read_frame(hip))
    ds = pydicom.dcmread(str(hip))
    ds.PixelData = np.ascontiguousarray(ds.pixel_array[:, ::-1]).tobytes()
    ds.PatientOrientation = ['R', 'F']
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'mirrored.dcm'
        ds.save_as(str(p))
        fr = read_frame(p)
        mir = a.analyze_frame(fr)
    check('зеркальный кадр (R, F) приводится к стандартному виду',
          fr.meta.get('orientation') == 'mirrored_lr' and np.array_equal(fr.pixels, read_frame(hip).pixels))
    check('сторона бедра и вердикт не зависят от зеркальной выгрузки',
          (mir['anatomical_region'], mir['violation_type']) == (base['anatomical_region'], base['violation_type'])
          and 'orientation:mirrored_lr' in mir['flags'],
          f"{base['anatomical_region']} -> {mir['anatomical_region']}")
    check('проекция в отчёте', base.get('projection') == 'AP')


def test_dicom_sr() -> None:
    """DICOM SR (ТЗ 2.6): валидный Basic Text SR, связан со снимком, читается обратно.

    Главное — связь с исходным снимком: SR с чужим StudyInstanceUID или без
    ссылки на изображение PACS покажет отдельным исследованием, и заключение
    потеряется.
    """
    import io
    import zipfile

    import pydicom

    from .dicom_io import unique_frames
    from .dicom_sr import SR_SOP_CLASS, build_sr
    from .pipeline import REGION_RU, VIOLATIONS, Analyzer, process_study
    from .report import write_sr

    study = sorted(DATA.iterdir())[0]
    frames, _ = unique_frames(study)
    fr = frames[0]
    src = pydicom.dcmread(str(fr.path), stop_before_pixels=True)
    res = Analyzer.load().analyze_pixels(fr.pixels)
    res['violations'] = ['spine_axis']              # заключение с нарушением
    buf = io.BytesIO()
    build_sr(src, res, VIOLATIONS, REGION_RU).save_as(buf, enforce_file_format=True)
    back = pydicom.dcmread(io.BytesIO(buf.getvalue()))

    check('SR: класс Basic Text SR и модальность SR',
          back.SOPClassUID == SR_SOP_CLASS and back.Modality == 'SR')
    check('SR: то же исследование, новая серия',
          back.StudyInstanceUID == src.StudyInstanceUID
          and back.SeriesInstanceUID != src.get('SeriesInstanceUID'))
    ref = back.ContentSequence[1].ReferencedSOPSequence[0]
    check('SR: ссылается на исходный снимок',
          ref.ReferencedSOPInstanceUID == src.SOPInstanceUID
          and ref.ReferencedSOPClassUID == src.SOPClassUID)
    text = back.ContentSequence[0].TextValue
    check('SR: заключение по-русски с текстом нарушения',
          VIOLATIONS['spine_axis'] in text and REGION_RU[res['anatomical_region']] in text,
          text.splitlines()[0])

    rows = process_study(study, Analyzer.load(), 's')
    tmp = Path(tempfile.mkdtemp(prefix='dxa_sr_'))
    try:
        z = zipfile.ZipFile(write_sr(rows, {f.image_uid: f for f in frames}, tmp / 'sr.zip'))
        ok = [r for r in rows if r['processing_status'] == 'Success']
        check('SR: в архиве по файлу на каждый обработанный кадр',
              len(z.namelist()) == len(ok), f'{len(z.namelist())} из {len(ok)}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError):
        check('POST /analyze/sr (пропущено: нет fastapi или httpx)', True)
        return
    from .api import app
    r = TestClient(app).post('/analyze/sr', files={
        'file': (fr.path.name, fr.path.read_bytes(), 'application/dicom')})
    ok = r.status_code == 200
    if ok:
        ok = pydicom.dcmread(io.BytesIO(r.content)).SOPClassUID == SR_SOP_CLASS
    check('POST /analyze/sr отдаёт DICOM SR', ok, f'status {r.status_code}')


def test_compressed_dicom() -> None:
    """Сжатый DICOM читается без потерь (ТЗ 8.2: качество предобработки DICOM).

    Обучающий набор несжатый, но из PACS снимки приходят в RLE, JPEG Lossless,
    JPEG-LS. Без декодера такой кадр ушёл бы в Failure — проверяем, что
    пиксели совпадают с исходными побайтно.
    """
    import pydicom
    from pydicom.uid import RLELossless

    from .dicom_io import dicom_files, read_frame

    src = dicom_files(sorted(DATA.iterdir())[0])[0]
    ref = read_frame(src).pixels
    tmp = Path(tempfile.mkdtemp(prefix='dxa_ts_'))
    try:
        ds = pydicom.dcmread(str(src))
        ds.compress(RLELossless)
        ds.save_as(str(tmp / 'rle.dcm'), enforce_file_format=True)
        variants = {'RLE': tmp / 'rle.dcm'}
        try:
            import gdcm
        except ImportError:
            gdcm = None
        if gdcm is not None:
            for name, ts in (('JPEG Lossless', gdcm.TransferSyntax.JPEGLosslessProcess14_1),
                             ('JPEG-LS', gdcm.TransferSyntax.JPEGLSLossless)):
                r = gdcm.ImageReader()
                r.SetFileName(str(src))
                r.Read()
                ch = gdcm.ImageChangeTransferSyntax()
                ch.SetTransferSyntax(gdcm.TransferSyntax(ts))
                ch.SetInput(r.GetImage())
                ch.Change()
                w = gdcm.ImageWriter()
                w.SetFile(r.GetFile())
                w.SetImage(ch.GetOutput())
                out = tmp / f'{name}.dcm'
                w.SetFileName(str(out))
                w.Write()
                variants[name] = out
        else:
            check('JPEG Lossless / JPEG-LS (пропущено: нет python-gdcm)', True)
        for name, f in variants.items():
            try:
                same = np.array_equal(read_frame(f).pixels, ref)
                check(f'сжатый DICOM {name} читается без потерь', same)
            except Exception as e:
                check(f'сжатый DICOM {name} читается без потерь', False, f'{type(e).__name__}: {e}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pacs_export_layout() -> None:
    """Выгрузка из PACS: DICOMDIR в корне, служебные файлы, SR рядом со снимками.

    Раньше DICOMDIR в корне превращал всю выгрузку в одно исследование, а
    каждый служебный файл давал строку Failure. Не снимок — не строка отчёта;
    битый .dcm — по-прежнему Failure.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from .dicom_io import study_dirs, unique_frames
    from .dicom_sr import build_sr
    from .pipeline import REGION_RU, VIOLATIONS, Analyzer, process_batch

    tmp = Path(tempfile.mkdtemp(prefix='dxa_pacs_'))
    try:
        studies = sorted(DATA.iterdir())[:2]
        for st in studies:
            shutil.copytree(st, tmp / st.name)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.1.3.10'      # Media Storage Directory
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ddir = FileDataset(None, {}, file_meta=meta, preamble=bytes(128))
        ddir.SOPClassUID = meta.MediaStorageSOPClassUID
        ddir.save_as(str(tmp / 'DICOMDIR'), enforce_file_format=True)

        first = tmp / studies[0].name
        mac = tmp / '__MACOSX' / first.name
        mac.mkdir(parents=True)
        (mac / '._CR000000.dcm').write_bytes(bytes([0, 5, 22, 7]) + bytes(60))
        (first / '.DS_Store').write_bytes(bytes([0, 0, 0, 1]) + b'Bud1' + bytes(40))
        fr = unique_frames(first)[0][0]
        src = pydicom.dcmread(str(fr.path), stop_before_pixels=True)
        sr = build_sr(src, Analyzer.load().analyze_pixels(fr.pixels), VIOLATIONS, REGION_RU)
        sr.save_as(str(first / 'report.sr.dcm'), enforce_file_format=True)
        (first / 'broken.dcm').write_bytes(b'not a dicom at all')

        check('DICOMDIR в корне не склеивает выгрузку в одно исследование',
              len(study_dirs(tmp)) == len(studies), f'{len(study_dirs(tmp))} из {len(studies)}')
        rows = process_batch(tmp, Analyzer.load())
        fails = [r for r in rows if r['processing_status'] == 'Failure']
        expected = sum(len(unique_frames(st)[0]) for st in studies)
        check('служебные файлы и SR не дают строк, битый .dcm — даёт Failure',
              len(rows) == expected + 1 and [r['file_name'] for r in fails] == ['broken.dcm'],
              f'строк {len(rows)}, ожидалось {expected + 1}; '
              f'Failure: {[r["file_name"] for r in fails]}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_batch_api() -> None:
    """Пакетный путь API целиком: zip -> задание -> отчёт, SR -> очистка по сроку."""
    import io
    import zipfile

    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError):
        check('POST /batch (пропущено: нет fastapi или httpx)', True)
        return
    from . import api

    study = sorted(DATA.iterdir())[0]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for f in study.rglob('*'):
            if f.is_file():
                z.write(f, f'{study.name}/{f.relative_to(study).as_posix()}')
    c = TestClient(api.app)
    r = c.post('/batch?fmt=csv&wait=true&sr=true&overlays=true&all_frames=true',
               files={'file': ('studies.zip', buf.getvalue(), 'application/zip')})
    job = r.json()
    check('POST /batch?wait=true завершает задание',
          r.status_code == 200 and job.get('state') == 'done', str(job.get('error') or job.get('state')))
    if job.get('state') != 'done':
        return
    rep = c.get(f"/jobs/{job['id']}/report")
    check('GET /jobs/{id}/report отдаёт таблицу ТЗ',
          rep.status_code == 200 and rep.content.decode('utf-8').startswith('path_to_study'))
    det = c.get(f"/jobs/{job['id']}/details").json()
    check('details.json: поля для интерфейса у каждого кадра',
          all({'error', 'study_uid', 'processing_status', 'details'} <= set(d) for d in det))
    uid = next((d['image_uid'] for d in det if d['image_uid']), '')
    one = c.get(f"/jobs/{job['id']}/overlays/{uid}.png")
    check('оверлей кадра по image_uid', one.status_code == 200 and one.content[:4] == b'\x89PNG')
    sr = c.get(f"/jobs/{job['id']}/sr")
    n_sr = len(zipfile.ZipFile(io.BytesIO(sr.content)).namelist()) if sr.status_code == 200 else 0
    check('GET /jobs/{id}/sr отдаёт SR на каждый кадр',
          n_sr == job['summary']['success'], f"{n_sr} из {job['summary']['success']}")

    workdir = api.JOBS[job['id']].workdir
    api._cleanup_jobs(now=time.time() + api.JOB_TTL_HOURS * 3600 + 1)
    check('задание старше срока удаляется вместе с файлами',
          job['id'] not in api.JOBS and not workdir.exists())

def test_wrapper_folder() -> None:
    """Архив, сжатый из папки: папка-обёртка не склеивает исследования.

    Раньше единственная верхняя папка считалась одним исследованием на всех
    пациентов. При этом исследование из двух серий-подпапок, завёрнутое в
    папки, должно остаться одним исследованием.
    """
    from .dicom_io import study_dirs

    studies = [d for d in sorted(DATA.iterdir()) if d.is_dir()]
    two_series = next((d for d in studies if len([s for s in d.iterdir() if s.is_dir()]) > 1), None)
    tmp = Path(tempfile.mkdtemp(prefix='dxa_wrap_'))
    try:
        for st in studies[:3]:
            shutil.copytree(st, tmp / 'wrap' / 'Исследования' / st.name)
        found = study_dirs(tmp / 'wrap')
        check('папка-обёртка: исследования не склеиваются', len(found) == 3, f'{len(found)} из 3')
        if two_series is not None:
            shutil.copytree(two_series, tmp / 'one' / 'deeper' / two_series.name)
            found = study_dirs(tmp / 'one')
            check('исследование из двух серий в обёртках остаётся одним', len(found) == 1,
                  f'{len(found)}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description='Тесты сервиса контроля качества DXA')
    ap.add_argument('--fast', action='store_true', help='без прогона по всей выборке')
    args = ap.parse_args()

    print('формат отчёта и таксономия')
    test_report_format()
    test_taxonomy()
    print('устойчивость')
    test_bad_input()
    test_pixel_robustness()
    test_archive_traversal()
    print('API')
    test_api()

    if not args.fast:
        if not DATA.exists():
            print(f'  (обучающий набор не найден: {DATA} — тесты по данным пропущены)')
        else:
            print('данные')
            test_dedup_and_regions()
            test_compressed_dicom()
            test_pacs_export_layout()
            test_wrapper_folder()
            test_batch_api()
            test_determinism()
            test_timing()
            print('модель ключевых точек')
            test_keypoints()
            print('нейросетевое второе мнение')
            test_cnn()
            print('ориентация и проекция')
            test_orientation()
            test_roi_rule()
            print('DICOM SR')
            test_dicom_sr()

    print(f'\nпройдено {len(PASS)}, провалено {len(FAIL)}')
    for f in FAIL:
        print(f'  ПРОВАЛ: {f}')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
