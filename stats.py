# -*- coding: utf-8 -*-
"""Метрики бинарной классификации с 95% доверительными интервалами.

ТЗ (п. 8.4) требует чувствительность, специфичность, сбалансированную точность,
F1 и ROC-AUC «желательно с 95% доверительными интервалами». Выборка маленькая
(3-36 положительных примеров на критерий), поэтому интервал здесь — не
украшение, а единственный честный способ показать, что разница между двумя
вариантами алгоритма может быть шумом.

Два вида интервалов, каждый на своём месте:

* **Уилсон** — для доли (чувствительность, специфичность). Точен при малом
  числе наблюдений и не выходит за [0, 1], в отличие от нормального
  приближения, которое на 1 из 7 даёт отрицательную границу.
* **Бутстрэп по исследованиям** — для F1, сбалансированной точности и AUC.
  Ресэмплим ИССЛЕДОВАНИЯ, а не кадры: два бедра одного пациента похожи, и
  интервал по кадрам окажется уже настоящего.
"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Интервал Уилсона для доли k/n."""
    if n == 0:
        return (float('nan'), float('nan'))
    p, z2 = k / n, z * z
    d = 1.0 + z2 / n
    c = (p + z2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return (float(max(0.0, c - half)), float(min(1.0, c + half)))


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    """tp/fp/tn/fn и производные. y и pred — 0/1, без пропусков."""
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    prec = tp / (tp + fp) if tp + fp else float('nan')
    f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0
    return {'n': len(y), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': sens, 'specificity': spec, 'precision': prec,
            'f1': f1, 'balanced_accuracy': (sens + spec) / 2,
            'accuracy': (tp + tn) / len(y) if len(y) else float('nan')}


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC через ранги (эквивалент U-статистики Манна-Уитни); ничьи усредняются."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), float)
    s = score[order]
    i = 0
    while i < len(s):                       # средний ранг на группе одинаковых значений
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC (средняя точность). На редких классах информативнее ROC-AUC."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    if y.sum() == 0:
        return float('nan')
    order = np.argsort(-score, kind='mergesort')
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def _bootstrap(y, pred, score, groups, fn, n_boot, seed):
    """Перевыборка групп с возвращением; NaN-повторы (нет одного из классов) отбрасываются."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in take])
        v = fn(y[idx], pred[idx], score[idx] if score is not None else None)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(50, n_boot // 20):
        return (float('nan'), float('nan'))
    lo, hi = np.percentile(vals, (2.5, 97.5))
    return (float(lo), float(hi))


def evaluate(y, pred, score=None, groups=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Полный набор метрик с 95% ДИ.

    y       — эталон 0/1
    pred    — предсказанный класс 0/1
    score   — непрерывная оценка (больше = вероятнее нарушение); нужна для AUC
    groups  — ключ исследования для бутстрэпа; без него ДИ считаются по кадрам
    """
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    score = None if score is None else np.asarray(score, float)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)

    m = confusion(y, pred)
    m['sensitivity_ci'] = wilson(m['tp'], m['tp'] + m['fn'])
    m['specificity_ci'] = wilson(m['tn'], m['tn'] + m['fp'])
    m['f1_ci'] = _bootstrap(y, pred, score, groups,
                            lambda a, b, _s: confusion(a, b)['f1'], n_boot, seed)
    m['balanced_accuracy_ci'] = _bootstrap(
        y, pred, score, groups,
        lambda a, b, _s: confusion(a, b)['balanced_accuracy'], n_boot, seed)
    if score is not None:
        m['roc_auc'] = roc_auc(y, score)
        m['pr_auc'] = average_precision(y, score)
        m['roc_auc_ci'] = _bootstrap(y, pred, score, groups,
                                     lambda a, _b, s: roc_auc(a, s), n_boot, seed)
    return m


def fmt(m: dict) -> str:
    """Одна строка метрик для лога и README."""
    def ci(key):
        lo, hi = m.get(key + '_ci', (float('nan'),) * 2)
        return f'[{lo:.2f}-{hi:.2f}]' if np.isfinite(lo) else '[—]'
    parts = [f"n={m['n']}", f"tp={m['tp']} fp={m['fp']} fn={m['fn']}",
             f"sens={m['sensitivity']:.2f}{ci('sensitivity')}",
             f"spec={m['specificity']:.2f}{ci('specificity')}",
             f"F1={m['f1']:.2f}{ci('f1')}"]
    if 'roc_auc' in m:
        parts.append(f"AUC={m['roc_auc']:.2f}{ci('roc_auc')}")
    return '  '.join(parts)
