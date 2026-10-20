# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: настольное приложение «Контроль качества DXA» для macOS.

    pyinstaller desktop/dxa_qc_macos.spec --noconfirm --distpath dist --workpath build/pyinstaller-macos

Собирается из корня репозитория в окружении Python 3.11 с requirements.lock
(desktop/build_macos.sh делает всё сам). Результат — dist/DXA-QC.app:

    Contents/MacOS/DXA-QC       окно приложения (WebKit через pywebview)
    Contents/MacOS/dxa-qc-cli   консоль: batch | serve | gui | version
    Contents/Resources/         Python, пакеты, веса моделей — общие для обоих
"""
import json
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
    (str(ROOT / 'desktop' / 'assets' / 'icon.png'), 'desktop/assets'),
]
build_info = ROOT / 'build' / 'build_info.json'
if build_info.exists():
    datas.append((str(build_info), 'desktop'))
for crit in ('rotation', 'artifacts'):
    for f in sorted((ROOT / 'cnn_qc' / crit).glob('model_*.onnx')) + [ROOT / 'cnn_qc' / crit / 'meta.json']:
        datas.append((str(f), f'cnn_qc/{crit}'))

binaries, hiddenimports = [], []
for pkg in ('gdcm', 'onnxruntime', 'webview'):
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
    'python_multipart', 'multipart', 'openpyxl',
    'objc', 'Foundation', 'AppKit', 'WebKit', 'Quartz',
    # пакеты проекта: часть модулей импортируется внутри функций
    'service.api', 'service.cli', 'service.overlay', 'service.dicom_sr', 'service.report',
    'spine_qc.criteria', 'spine_qc.overlay', 'desktop.app', 'desktop.cli',
]

excludes = ['torch', 'torchvision', 'matplotlib', 'IPython', 'jupyter', 'notebook', 'pytest',
            'tensorboard', 'cv2', 'src', 'dxa_qc', 'service.keypoints_backend',
            'clr', 'pythonnet', 'clr_loader']

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

common = dict(exclude_binaries=True, debug=False, bootloader_ignore_signals=False,
              strip=False, upx=False, target_arch=None,
              codesign_identity=None, entitlements_file=None)
utf8 = [('X utf8', None, 'OPTION')]
gui = EXE(pyz, a.scripts, utf8, name='DXA-QC', console=False, argv_emulation=False,  # noqa: F821
          **common)
cli = EXE(pyz, a.scripts, utf8, name='dxa-qc-cli', console=True, **common)             # noqa: F821

coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False,    # noqa: F821
               name='DXA-QC')

version = (json.loads(build_info.read_text(encoding='utf-8')).get('version', '1.0.0')
           if build_info.exists() else '1.0.0')
app = BUNDLE(                                                            # noqa: F821
    coll,
    name='DXA-QC.app',
    icon=str(ROOT / 'build' / 'icon.icns'),
    bundle_identifier='ru.lct2026.dxa-qc',
    version=version,
    info_plist={
        'CFBundleName': 'DXA-QC',
        'CFBundleDisplayName': 'Контроль качества DXA',
        'CFBundleShortVersionString': version,
        'LSMinimumSystemVersion': '12.0',
        'NSHighResolutionCapable': True,
        'LSApplicationCategoryType': 'public.app-category.medical',
    },
)
