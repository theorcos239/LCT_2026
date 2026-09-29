# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: настольное приложение «Контроль качества DXA».

    pyinstaller desktop/dxa_qc.spec --noconfirm --distpath dist --workpath build/pyinstaller

Собирается из корня репозитория в окружении Python 3.11 с requirements.lock
(desktop/build_windows.ps1 делает всё сам). Результат — dist/DXA-QC/:

    DXA-QC.exe        окно приложения (WebView2)
    dxa-qc-cli.exe    консоль: batch | serve | bot | gui | version
    _internal/        Python, пакеты, веса моделей — общие для обоих exe
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent            # noqa: F821  (SPECPATH задаёт PyInstaller)

# Файлы, которые конвейер читает во время работы (пути — относительно модулей)
datas = [
    (str(ROOT / 'service' / 'static'), 'service/static'),
    (str(ROOT / 'service' / 'metrics.json'), 'service'),
    (str(ROOT / 'region_clf' / 'model.joblib'), 'region_clf'),
    (str(ROOT / 'spine_qc' / 'model.joblib'), 'spine_qc'),
    (str(ROOT / 'spine_qc' / 'thresholds.json'), 'spine_qc'),
    (str(ROOT / 'hip_rotation' / 'thresholds.json'), 'hip_rotation'),
    (str(ROOT / 'hip_roi' / 'probability.json'), 'hip_roi'),
    (str(ROOT / 'desktop' / 'assets' / 'icon.ico'), 'desktop/assets'),
]
# версия сборки: коммит и дата (пишет desktop/build_windows.ps1)
if (ROOT / 'build' / 'build_info.json').exists():
    datas.append((str(ROOT / 'build' / 'build_info.json'), 'desktop'))
for crit in ('rotation', 'artifacts'):
    for f in sorted((ROOT / 'cnn_qc' / crit).glob('model_*.onnx')) + [ROOT / 'cnn_qc' / crit / 'meta.json']:
        datas.append((str(f), f'cnn_qc/{crit}'))

binaries, hiddenimports = [], []
for pkg in ('gdcm', 'onnxruntime', 'webview', 'pythonnet', 'clr_loader'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('pydicom.pixels')
hiddenimports += [
    # классы в моделях joblib: PyInstaller их в коде не видит
    'sklearn.decomposition._pca', 'sklearn.linear_model._logistic', 'sklearn.pipeline',
    'sklearn.preprocessing._data',
    'python_multipart', 'multipart', 'openpyxl', 'clr',
    # пакеты проекта: часть модулей импортируется внутри функций
    'service.api', 'service.cli', 'service.overlay', 'service.dicom_sr', 'service.report',
    'spine_qc.criteria', 'spine_qc.overlay', 'bot.app', 'desktop.app', 'desktop.cli',
]

excludes = ['torch', 'torchvision', 'matplotlib', 'IPython', 'jupyter', 'notebook', 'pytest',
            'tensorboard', 'cv2', 'src', 'dxa_qc', 'service.keypoints_backend']

a = Analysis(                                                            # noqa: F821
    [str(ROOT / 'desktop' / 'entry.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)                                                        # noqa: F821

icon = str(ROOT / 'desktop' / 'assets' / 'icon.ico')
common = dict(exclude_binaries=True, debug=False, bootloader_ignore_signals=False,
              strip=False, upx=False, icon=icon, version=None)

# UTF-8 mode: вывод в канал и файлы — в UTF-8, а не в cp1251 консоли Windows
utf8 = [('X utf8', None, 'OPTION')]
gui = EXE(pyz, a.scripts, utf8, name='DXA-QC', console=False, **common)   # noqa: F821
cli = EXE(pyz, a.scripts, utf8, name='dxa-qc-cli', console=True, **common)  # noqa: F821

coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False,    # noqa: F821
               name='DXA-QC')
