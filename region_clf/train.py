# -*- coding: utf-8 -*-
"""Обучение классификатора области DXA-снимка: lh / rh / spine.

Оценка — StratifiedGroupKFold(5) с группировкой по исследованию: кадры одного
пациента не попадают одновременно в обучение и в контроль.

Запуск:
    python -m region_clf.train                 # CV + обучение на всех данных
    python -m region_clf.train --stress        # ещё и проверка на искажениях
    python -m region_clf.train --jitter 0      # без геометрической аугментации
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (CLASSES, FLIP_LABEL, IMG_SIZE, contrast, featurize,
                       flip_lr, flip_ud, preprocess, read_dicom)

CROP = False        # обрезка фона выключена: в стресс-тесте не дала выигрыша (см. README)
PCA_K = 20          # компонент в детекторе «чужого кадра»
REJECT_Q = 0.99     # доля своих кадров, которую детектор обязан пропускать

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDIES = ROOT / 'НД_для_обучения' / 'Исследования'
MODEL_PATH = HERE / 'model.joblib'


def make_model() -> object:
    """Линейная модель на стандартизованных признаках.

    Выбрана по стресс-тесту: на чистых данных логрег, линейный SVM и бустинг
    неразличимы (0-1 ошибка из 252), но на искажённых кадрах бустинг падает до
    ~0.72, а логрег держит ~0.97.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000))


# --------------------------------------------------------------------------- #
#  Данные и аугментация
# --------------------------------------------------------------------------- #
def load_dataset(labels_csv: Path = HERE / 'labels.csv') -> tuple[list, np.ndarray, np.ndarray]:
    lab = pd.read_csv(labels_csv)
    raw = [read_dicom(STUDIES / p) for p in lab.rel_path]
    return raw, lab.label.to_numpy(), lab.study.to_numpy()


def _jitter(px: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Поворот / масштаб / сдвиг / гамма — разброс укладки и экспозиции."""
    from PIL import Image
    scale = max(float(px.max()), 1e-6)
    im = Image.fromarray(np.clip(px / scale * 255, 0, 255).astype(np.uint8))
    w, h = im.size
    im = im.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR, fillcolor=0)
    z = rng.uniform(0.9, 1.12)
    im = im.resize((max(16, int(w * z)), max(16, int(h * z))), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    dx = int(rng.uniform(-.07, .07) * a.shape[1])
    dy = int(rng.uniform(-.07, .07) * a.shape[0])
    a = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    return (a ** rng.uniform(0.8, 1.25)).astype(np.float32)


def build_views(raw: list, y: np.ndarray, n_jitter: int, seed: int = 7):
    """Матрицы признаков: исходная + зеркальная + джиттер-копии.

    Зеркалирование — не просто «больше данных»: оно задаёт модели симметрию
    задачи (отражённое левое бедро обязано быть правым) и вдвое увеличивает
    выборку для самого трудного различения lh/rh.
    """
    rng = np.random.default_rng(seed)
    y_flip = np.array([FLIP_LABEL[v] for v in y])

    base = [preprocess(p, crop=CROP) for p in raw]
    X = np.stack([featurize(i) for i in base])
    views = [(X, y), (np.stack([featurize(flip_lr(i)) for i in base]), y_flip)]
    for _ in range(n_jitter):
        jit = [preprocess(_jitter(p, rng), crop=CROP) for p in raw]
        views.append((np.stack([featurize(i) for i in jit]), y))
        views.append((np.stack([featurize(flip_lr(i)) for i in jit]), y_flip))
    return X, views


def _fit_on(views, idx):
    Xtr = np.vstack([V[idx] for V, _ in views])
    ytr = np.concatenate([t[idx] for _, t in views])
    return make_model().fit(Xtr, ytr)


# --------------------------------------------------------------------------- #
#  Оценка
# --------------------------------------------------------------------------- #
def cross_validate(X, views, y, groups, n_splits=5, seed=0, extra_X=None):
    """Возвращает out-of-fold предсказания (и их же на искажённых копиях)."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    oof_extra = [np.empty(len(y), dtype=object) for _ in (extra_X or [])]
    for tr, te in skf.split(X, y, groups=groups):
        m = _fit_on(views, tr)
        oof[te] = m.predict(X[te])
        for k, Xe in enumerate(extra_X or []):
            oof_extra[k][te] = m.predict(Xe[te])
    return oof, oof_extra


# --------------------------------------------------------------------------- #
#  Детектор «чужого кадра»
# --------------------------------------------------------------------------- #
def make_detector():
    """PCA-реконструкция признаков: насколько кадр похож на то, что видели.

    Нужен потому, что классификатор стоит ПЕРВЫМ в конвейере: до него снимок
    никто не фильтрует, а сам он обязан выдать одну из трёх меток. Уверенность
    для этого не годится — на чистом шуме модель даёт spine с p=1.000.
    """
    return make_pipeline(StandardScaler(), PCA(n_components=PCA_K, random_state=0))


def novelty(detector, A: np.ndarray) -> np.ndarray:
    """Средний квадрат ошибки восстановления признаков (в стандартизованном виде)."""
    sc, pca = detector.named_steps['standardscaler'], detector.named_steps['pca']
    Z = sc.transform(A)
    return np.mean((Z - pca.inverse_transform(pca.transform(Z))) ** 2, axis=1)


def fit_detector(X, Xflip, y, groups, n_splits=5, seed=0):
    """Возвращает детектор, порог и долю своих кадров, которую он отсекает.

    Порог калибруется out-of-fold: PCA обучается на train-фолде и оценивает
    отложенные кадры, иначе своя же выборка выглядит подозрительно похожей.
    """
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y))
    for tr, te in skf.split(X, y, groups=groups):
        det = make_detector().fit(np.vstack([X[tr], Xflip[tr]]))
        oof[te] = novelty(det, X[te])
    thr = float(np.quantile(oof, REJECT_Q))
    det = make_detector().fit(np.vstack([X, Xflip]))
    return det, thr, float(np.mean(oof > thr)), oof


def fit_orientation(X, Xflip, Xud, Xud_flip, y, groups, n_splits=5, seed=0):
    """Проверка «кадр не перевёрнут»: обучаемая, а не эвристическая.

    Детектор новизны на перевёрнутом кадре бесполезен (ловит 17%): это всё ещё
    похожая на DXA картинка. При этом модель выдаёт на нём уверенную и НЕВЕРНУЮ
    сторону, потому что переворот меняет яркостную асимметрию таза. Отличать
    норму от переворота умеет отдельный линейный классификатор: размеченные
    примеры для него получаются из обучающей выборки бесплатно.
    """
    Xpos, Xneg = np.vstack([X, Xflip]), np.vstack([Xud, Xud_flip])
    Xo = np.vstack([Xpos, Xneg])
    yo = np.r_[np.zeros(len(Xpos)), np.ones(len(Xneg))]     # 1 = перевёрнут
    g2 = np.tile(groups, 4)
    y2 = np.tile(y, 4)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(yo))
    for tr, te in skf.split(Xo, y2, groups=g2):
        oof[te] = make_model().fit(Xo[tr], yo[tr]).predict(Xo[te])
    acc = float(accuracy_score(yo, oof))
    return make_model().fit(Xo, yo), acc


def stress_views(raw, seed=0, n=3):
    """Сильнее, чем обучающая аугментация: поворот ±12°, масштаб, сдвиг, шум."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        imgs = [_jitter(p, rng) for p in raw]
        noisy = [np.clip(i + rng.normal(0, 0.03, i.shape), 0, 1).astype(np.float32) for i in imgs]
        out.append(np.stack([featurize(preprocess(i, crop=CROP)) for i in noisy]))
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description='Обучение классификатора lh/rh/spine')
    ap.add_argument('--jitter', type=int, default=3, help='геометрических копий на кадр')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--stress', action='store_true', help='проверить на искажённых кадрах')
    ap.add_argument('--out', type=Path, default=MODEL_PATH)
    args = ap.parse_args()

    t0 = time.time()
    raw, y, groups = load_dataset()
    print(f'{len(raw)} кадров, {len(set(groups))} исследований, '
          f'классы: {dict(pd.Series(y).value_counts())}')

    X, views = build_views(raw, y, args.jitter)
    print(f'признаков: {X.shape[1]}, обучающих представлений на кадр: {len(views)}')

    extra = stress_views(raw) if args.stress else None
    oof, oof_stress = cross_validate(X, views, y, groups, args.folds, extra_X=extra)

    acc = accuracy_score(y, oof)
    print(f'\n=== Кросс-валидация по исследованиям ({args.folds} фолдов) ===')
    print(f'accuracy = {acc:.4f}  ({int(round(acc * len(y)))}/{len(y)})')
    print(classification_report(y, oof, labels=list(CLASSES), digits=3, zero_division=0))
    print('матрица ошибок (строки — факт, столбцы — предсказание):')
    print(pd.DataFrame(confusion_matrix(y, oof, labels=list(CLASSES)),
                       index=list(CLASSES), columns=list(CLASSES)).to_string())
    for i in np.flatnonzero(oof != y):
        print(f'  ошибка: {y[i]} -> {oof[i]}  {pd.read_csv(HERE / "labels.csv").rel_path[i]}')

    stress_acc = None
    if extra:
        stress_acc = float(np.mean([accuracy_score(y, p) for p in oof_stress]))
        print(f'\nустойчивость (поворот ±8°, масштаб ±10%, сдвиг ±7%, гамма, шум): '
              f'accuracy = {stress_acc:.4f}')

    # детектор чужого кадра + нижняя граница контраста (пустой кадр)
    Xflip = views[1][0]
    det, thr, rejected, _ = fit_detector(X, Xflip, y, groups, args.folds)
    min_contrast = float(min(contrast(preprocess(p, crop=CROP)) for p in raw))
    print(f'\nдетектор чужого кадра: порог новизны {thr:.3f} '
          f'(отсекает {rejected:.1%} своих кадров), '
          f'минимальный контраст своих {min_contrast:.4f}')

    # проверка ориентации (перевёрнутый кадр детектор новизны не ловит)
    base = [preprocess(p, crop=CROP) for p in raw]
    Xud = np.stack([featurize(flip_ud(i)) for i in base])
    Xud_flip = np.stack([featurize(flip_lr(flip_ud(i))) for i in base])
    orient, orient_acc = fit_orientation(X, Xflip, Xud, Xud_flip, y, groups, args.folds)
    print(f'проверка ориентации: accuracy {orient_acc:.4f} (кросс-валидация по исследованиям)')

    model = _fit_on(views, np.arange(len(y)))
    import joblib
    joblib.dump({'model': model, 'classes': list(CLASSES), 'img_size': IMG_SIZE,
                 'crop': CROP,          # инференс обязан повторять препроцессинг обучения
                 'detector': det, 'novelty_threshold': thr,
                 'orientation': orient,
                 'min_contrast': min_contrast * 0.5,   # запас к самому тусклому своему кадру
                 'trained_on': len(y), 'cv_accuracy': float(acc)}, args.out)
    meta = {'cv_accuracy': float(acc), 'stress_accuracy': stress_acc,
            'novelty_threshold': thr, 'novelty_rejects_own': rejected,
            'orientation_accuracy': orient_acc,
            'n_images': int(len(y)), 'n_studies': int(len(set(groups))),
            'folds': args.folds, 'jitter': args.jitter,
            'seconds': round(time.time() - t0, 1)}
    (HERE / 'metrics.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    print(f'\nмодель сохранена: {args.out}  ({time.time() - t0:.1f} c)')


if __name__ == '__main__':
    main()
