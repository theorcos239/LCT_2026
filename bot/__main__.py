# -*- coding: utf-8 -*-
"""python -m bot — запуск Telegram-бота (см. bot/app.py)."""
import os

# Один поток BLAS и onnxruntime — те же условия, что у сервиса в контейнере:
# вердикты бота совпадают с CLI и API побитово.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from bot.app import main  # noqa: E402

raise SystemExit(main())
