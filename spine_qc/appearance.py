# -*- coding: utf-8 -*-
"""Вторая оценка критериев позвоночника — по виду кадра целиком.

Логистическая регрессия на тех же признаках, что использует region_clf:
уменьшенное изображение плюс гистограммы ориентаций градиента. Она ничего не
знает про гребни подвздошных костей и косточки белья и смотрит на кадр как на
картинку — именно поэтому годится в сверку к геометрическому измерению.

Независимость проверена, а не предположена: корреляция рангов с геометрией
0.66 по укладке и 0.18 по артефактам. Для сравнения, регрессия, обученная на
тех же четырёх морфологических признаках, что складывает геометрический счёт,
расходилась с ним на 1 кадре из 99 — такая сверка не проверяет ничего.

Ось этой моделью не сверяется: OOF AUC 0.57, то есть сигнала нет. Ось сверяется
второй геометрической конструкцией угла (см. criteria.axis).

    python -m spine_qc.appearance        # обучить и сохранить model.joblib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from region_clf.features import featurize, preprocess

MODEL_PATH = Path(__file__).resolve().parent / 'model.joblib'
TARGETS = ('position', 'artifacts')
IMG_SIZE = 96          # крупнее 64 у region_clf: косточка белья тонкая


def vector(img: np.ndarray) -> np.ndarray:
    return featurize(preprocess(img, size=IMG_SIZE))


def make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # C=0.05: признаков 2688 на 99 кадров, без сильной регуляризации регрессия
    # запоминает выборку. class_weight выравнивает 6 и 17 положительных.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, class_weight='balanced', max_iter=4000))


class AppearanceModel:
    """Обёртка над сохранёнными регрессиями (по одной на критерий)."""

    def __init__(self, path: str | Path = MODEL_PATH):
        import joblib
        p = Path(path)
        self.models: dict = {}
        self.oof_auc: dict = {}
        if p.exists():
            blob = joblib.load(p)
            self.models = blob.get('models', {})
            self.oof_auc = blob.get('oof_auc', {})

    @property
    def available(self) -> bool:
        return bool(self.models)

    def probabilities(self, img: np.ndarray) -> dict[str, float]:
        """Кадр -> {'position': p, 'artifacts': p}. Пусто, если модели нет."""
        if not self.models:
            return {}
        x = vector(img)[None]
        return {k: float(m.predict_proba(x)[0, 1]) for k, m in self.models.items()}

    def probabilities_from_vectors(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {k: m.predict_proba(X)[:, 1] for k, m in self.models.items()}


def train(vectors: np.ndarray, targets: dict[str, np.ndarray], oof_auc: dict | None = None,
          path: str | Path = MODEL_PATH) -> None:
    """Обучение на всей выборке и сохранение. Честная оценка — в calibrate.py."""
    import joblib
    models = {}
    for name, y in targets.items():
        ok = np.isfinite(y)
        m = make_model()
        m.fit(vectors[ok], y[ok].astype(int))
        models[name] = m
    joblib.dump({'models': models, 'oof_auc': oof_auc or {}, 'img_size': IMG_SIZE}, path)


if __name__ == '__main__':
    from .calibrate import main
    raise SystemExit(main())
