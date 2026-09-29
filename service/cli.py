# -*- coding: utf-8 -*-
"""Пакетная обработка исследований из командной строки.

    python -m service.cli НД_для_обучения/Исследования --out report.xlsx
    python -m service.cli исследования.zip --out report.csv --overlays overlays.zip
    python -m service.cli одно_исследование/ --out report.csv --details details.json
    python -m service.cli исследования/ --out report.xlsx --sr sr.zip
    python -m service.cli снимок.dcm --json

На вход принимается папка с исследованиями, одно исследование, отдельный файл
или zip-архив. На выходе — таблица в формате ТЗ 2.5 и, по запросу, архив с
визуализацией нарушений, JSON со всеми измерениями и zip с DICOM SR
(текстовое заключение по каждому кадру, ТЗ 2.6).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog='service.cli',
        description='Контроль качества денситометрических исследований (DXA)')
    ap.add_argument('target', help='папка с исследованиями, исследование, файл .dcm или .zip')
    ap.add_argument('--out', type=Path, default=Path('report.xlsx'),
                    help='итоговая таблица .xlsx или .csv (по умолчанию report.xlsx)')
    ap.add_argument('--overlays', type=Path, help='zip с визуализацией нарушений')
    ap.add_argument('--details', type=Path, help='JSON со всеми измерениями')
    ap.add_argument('--sr', type=Path, help='zip с DICOM SR (Basic Text SR) на каждый кадр')
    ap.add_argument('--hip-method', default='kit2', choices=('kit0', 'kit1', 'kit2', 'kit3'),
                    help='комплект hip_roi для отступов бедра')
    ap.add_argument('--lenient-region', action='store_true',
                    help='оценивать кадры, не принятые фильтром области')
    ap.add_argument('--keypoints', action='store_true',
                    help='ось позвоночника и отступы ROI считать моделью ключевых '
                         'точек (нужен torch и веса в runs/keypoints)')
    ap.add_argument('--keypoints-dir', type=Path,
                    help='каталог с весами модели точек (по умолчанию runs/keypoints)')
    ap.add_argument('--roi-rule', default='scan_length', choices=('scan_length', 'margins'),
                    help='отступы ROI: длина поля, как оценивает эксперт (по умолчанию), '
                         'или отступы от ориентиров буквально по рисунку 6 ТЗ')
    ap.add_argument('--no-cnn', action='store_true',
                    help='без нейросетевого второго мнения (cnn_qc): вердикт по одной геометрии')
    ap.add_argument('--json', action='store_true', help='сводку вывести как JSON')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    from .dicom_io import extract_archive
    from .pipeline import Analyzer, process_batch
    from .report import summary, write_details, write_overlays, write_sr, write_table

    target = Path(args.target)
    if not target.exists():
        print(f'не найдено: {target}', file=sys.stderr)
        return 1

    tmp = None
    if target.suffix.lower() == '.zip':
        tmp = tempfile.TemporaryDirectory(prefix='dxa_qc_')
        target = extract_archive(target, tmp.name)

    t0 = time.perf_counter()
    analyzer = Analyzer.load(hip_method=args.hip_method,
                             strict_region=not args.lenient_region,
                             keypoints=args.keypoints,
                             keypoints_dir=args.keypoints_dir,
                             cnn=not args.no_cnn, roi_rule=args.roi_rule)
    if args.keypoints and analyzer.keypoints is not None and not analyzer.keypoints.available:
        print(f'модель точек не загружена: {analyzer.keypoints.error}\n'
              'кадры будут посчитаны контурной геометрией', file=sys.stderr)

    def progress(i, n, study):
        if not args.quiet:
            print(f'\r[{i}/{n}] {study.name[:60]:60}', end='', file=sys.stderr, flush=True)

    rows = process_batch(target, analyzer, progress=progress if not args.quiet else None)
    if not args.quiet:
        print('\r' + ' ' * 72 + '\r', end='', file=sys.stderr)

    write_table(rows, args.out)
    if args.details:
        write_details(rows, args.details)
    if args.overlays or args.sr:
        # кадры перечитываются: держать 252 массива в памяти ради архива не нужно
        from .dicom_io import study_dirs, unique_frames
        frames = {}
        for s in study_dirs(target):
            for fr in unique_frames(s)[0]:
                frames[fr.image_uid] = fr
        if args.overlays:
            write_overlays(rows, frames, args.overlays)
        if args.sr:
            write_sr(rows, frames, args.sr)

    s = summary(rows)
    s['wall_seconds'] = round(time.perf_counter() - t0, 2)
    s['report'] = str(args.out)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    elif not args.quiet:
        print(f"исследований {s['studies']}, кадров {s['images']}: "
              f"успешно {s['success']}, ошибок {s['failure']}")
        print(f"с нарушением {s['with_violation']}, качественных "
              f"{s['success'] - s['with_violation']}")
        print('по областям: ' + ', '.join(f'{k}={v}' for k, v in s['by_region'].items()))
        if s['by_violation']:
            print('по нарушениям: ' + ', '.join(f'{k}={v}' for k, v in s['by_violation'].items()))
        print(f"время: {s['wall_seconds']} с всего, "
              f"{s['seconds_per_image_median']} с на кадр (медиана)")
        print(f'таблица: {args.out}')
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
