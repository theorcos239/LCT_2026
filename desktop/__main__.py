# -*- coding: utf-8 -*-
"""python -m desktop            окно приложения
python -m desktop <команда>  batch | serve | bot | gui | version (см. desktop/cli.py)"""
import os
import sys

# Один поток BLAS и onnxruntime — условие побитовой воспроизводимости (как в Dockerfile)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

if len(sys.argv) > 1:
    from desktop.cli import main
else:
    from desktop.app import main

raise SystemExit(main())
