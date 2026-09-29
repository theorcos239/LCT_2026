# -*- coding: utf-8 -*-
"""Точка входа PyInstaller для обоих exe установленного приложения.

DXA-QC.exe (оконный) открывает приложение; dxa-qc-cli.exe (консольный)
разбирает команду: batch, serve, bot, gui, version.
"""
import multiprocessing
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    exe = os.path.splitext(os.path.basename(sys.executable))[0].lower()
    if exe.endswith('-cli'):
        from desktop.cli import main
    else:
        from desktop.app import main
    raise SystemExit(main())
