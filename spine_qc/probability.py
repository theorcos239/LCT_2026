# -*- coding: utf-8 -*-
"""Калиброванная вероятность нарушения вместо голого «да/нет».

Зачем, если ROC-AUC от этого не меняется (она инвариантна к монотонному
преобразованию счёта): вероятность нужна отчёту. По ТЗ метрики считает
организатор, а посчитать AUC по бинарному вердикту нельзя — нужен непрерывный
скор. Заодно вероятности правильно собираются в вердикт по области:
1 - Π(1-p) вместо OR жёстких флагов.

Калибровка — логистическая по Платту: p = sigmoid(w * score + b), коэффициенты
подбираются на обучающих фолдах. Обратная пропорциональность углу (1/angle)
дала бы ту же AUC, но числа нельзя было бы читать как вероятность.
"""
from __future__ import annotations

import numpy as np

# Счёт критерия в виде «больше = вероятнее нарушение». Те же соглашения,
# что в spine_qc.calibrate.geometry_score — при изменении править оба места.
SCORE = {
    'axis': lambda c: None if c.get('angle_deg') is None else abs(float(c['angle_deg'])),
    'position': lambda c: None if c.get('iliac_area') is None else -float(c['iliac_area']),
    'artifacts': lambda c: None if c.get('score') is None else float(c['score']),
}


def score_of(name: str, criterion: dict) -> float | None:
    fn = SCORE.get(name)
    return fn(criterion) if fn else None


def apply(params: dict | None, score: float | None) -> float | None:
    """p = sigmoid(w * score + b); None, если нет калибровки или измерения."""
    if params is None or score is None or not np.isfinite(score):
        return None
    z = float(params['w']) * float(score) + float(params['b'])
    return float(round(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), 4))


def fit(y: np.ndarray, score: np.ndarray) -> dict | None:
    """Коэффициенты Платта по обучающей выборке. Требует обоих классов.

    Скор перед подгонкой стандартизуется, а коэффициенты пересчитываются
    обратно в его единицы. Без этого L2-регуляризация логистической регрессии
    зависит от масштаба: у укладки скор — доля пикселей (~0.1), и штраф
    прижимал вес к нулю — вероятность выходила одинаковой (0.06) у всех
    кадров, а свод по позвоночнику ранжировал хуже голого вердикта.
    """
    from sklearn.linear_model import LogisticRegression

    ok = np.isfinite(y) & np.isfinite(score)
    y, score = y[ok].astype(int), score[ok].astype(float)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return None
    mu, sd = float(score.mean()), float(score.std())
    sd = sd if sd > 1e-12 else 1.0
    m = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    m.fit(((score - mu) / sd).reshape(-1, 1), y)
    w, b = float(m.coef_[0][0]) / sd, float(m.intercept_[0]) - float(m.coef_[0][0]) * mu / sd
    return {'w': round(w, 6), 'b': round(b, 6)}


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Насколько вероятности откалиброваны: 0 — идеально, 0.25 — постоянные 0.5."""
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean((p[ok] - y[ok].astype(float)) ** 2))


def combine(probs) -> float | None:
    """Вероятность «хоть одно нарушение» при независимости критериев."""
    ps = [p for p in probs if p is not None]
    return None if not ps else float(round(1.0 - np.prod([1.0 - p for p in ps]), 4))
