# -*- coding: utf-8 -*-
"""Замер скорости рабочего конвейера (ТЗ 2.7: не более 3 минут на исследование).

    python -m service.benchmark                     # все исследования обучающего набора
    python -m service.benchmark --limit 20 --no-cnn

Считает то, что увидит проверяющий: чтение DICOM, дедупликацию, все критерии и
нейросеть — по исследованию целиком, в один процесс, как `service.cli`.
Результат — service/benchmark.json (медиана, p95, максимум на кадр и на
исследование, общее время, конфигурация машины).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'НД_для_обучения' / 'Исследования'


def main() -> int:
    ap = argparse.ArgumentParser(description='Скорость конвейера на исследованиях')
    ap.add_argument('root', nargs='?', type=Path, default=DATA)
    ap.add_argument('--limit', type=int, default=0, help='только первые N исследований')
    ap.add_argument('--no-cnn', action='store_true')
    ap.add_argument('--out', type=Path, default=HERE / 'benchmark.json')
    args = ap.parse_args()

    from .dicom_io import study_dirs
    from .pipeline import Analyzer, process_study

    t_load = time.perf_counter()
    a = Analyzer.load(cnn=not args.no_cnn)
    t_load = time.perf_counter() - t_load
    studies = study_dirs(args.root)
    if args.limit:
        studies = studies[:args.limit]

    per_study, per_frame, frames, failures = [], [], 0, 0
    t_all = time.perf_counter()
    for s in studies:
        t = time.perf_counter()
        rows = process_study(s, a, s.name)
        per_study.append(time.perf_counter() - t)
        per_frame += [r['time_of_processing'] for r in rows]
        frames += len(rows)
        failures += sum(r['processing_status'] != 'Success' for r in rows)
    total = time.perf_counter() - t_all

    q = lambda x, p: round(float(np.percentile(x, p)), 3)          # noqa: E731
    res = {
        'studies': len(studies), 'frames': frames, 'failures': failures,
        'cnn': not args.no_cnn, 'model_load_s': round(t_load, 2), 'total_s': round(total, 1),
        'per_study_s': {'median': q(per_study, 50), 'p95': q(per_study, 95),
                        'max': round(max(per_study), 3)},
        'per_frame_s': {'median': q(per_frame, 50), 'p95': q(per_frame, 95),
                        'max': round(max(per_frame), 3)},
        'threads': {k: os.environ.get(k, '') for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'machine': {'cpu': platform.processor() or platform.machine(),
                    'cores': os.cpu_count(), 'python': platform.python_version(),
                    'system': platform.system()},
    }
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
