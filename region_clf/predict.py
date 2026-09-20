# -*- coding: utf-8 -*-
"""Определение области DXA-снимка: lh (левое бедро) / rh (правое) / spine.

CLI:
    python -m region_clf.predict снимок.dcm
    python -m region_clf.predict папка_исследования            # рекурсивно
    python -m region_clf.predict папка --csv out.csv --json

API:
    from region_clf.predict import RegionClassifier
    clf = RegionClassifier()
    clf.predict('снимок.dcm')        -> 'lh'
    clf.predict_proba('снимок.dcm')  -> {'lh': 0.99, 'rh': 0.004, 'spine': 0.006}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .features import (CLASS_RU, CLASSES, contrast, featurize, preprocess,
                       read_image)

MODEL_PATH = Path(__file__).resolve().parent / 'model.joblib'
SUFFIXES = {'.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.tif', '.tiff', ''}


class RegionClassifier:
    """Обёртка над сохранённой моделью."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        import joblib
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f'Модель не найдена: {p}. Сначала обучите: python -m region_clf.train')
        blob = joblib.load(p)
        self.model = blob['model']
        self.classes = list(blob.get('classes', CLASSES))
        self.crop = bool(blob.get('crop', False))    # ровно тот препроцессинг, что при обучении
        self.cv_accuracy = blob.get('cv_accuracy')
        self.detector = blob.get('detector')         # см. train.make_detector
        self.novelty_threshold = blob.get('novelty_threshold')
        self.orientation = blob.get('orientation')   # 1 = кадр перевёрнут
        self.min_contrast = blob.get('min_contrast', 0.0)

    # ---------------------------------------------------------------- #
    def prepare(self, src) -> tuple[np.ndarray, float]:
        """Путь / массив пикселей -> вектор признаков и контраст кадра."""
        px = src if isinstance(src, np.ndarray) else read_image(src)
        img = preprocess(px, crop=self.crop)
        return featurize(img), contrast(img)

    def vector(self, src) -> np.ndarray:
        return self.prepare(src)[0]

    def novelty(self, X: np.ndarray) -> np.ndarray:
        """Насколько кадры непохожи на обучающие (ошибка PCA-реконструкции)."""
        if self.detector is None:
            return np.zeros(len(X))
        sc, pca = self.detector.named_steps['standardscaler'], self.detector.named_steps['pca']
        Z = sc.transform(X)
        return np.mean((Z - pca.inverse_transform(pca.transform(Z))) ** 2, axis=1)

    def predict(self, src) -> str:
        return str(self.model.predict(self.vector(src)[None])[0])

    def predict_proba(self, src) -> dict[str, float]:
        p = self.model.predict_proba(self.vector(src)[None])[0]
        return {str(c): float(v) for c, v in zip(self.model.classes_, p)}

    def rows_from_vectors(self, sources: list, X: np.ndarray,
                          contrasts: list[float] | None = None) -> list[dict]:
        """Готовые признаки -> строки результата (один вызов модели на пачку).

        `accepted` = кадр похож на то, на чём модель училась. Классификатор
        стоит первым в конвейере, поэтому решение «это вообще не наш снимок»
        принимать больше некому: непринятый кадр нельзя отправлять на разметку.
        """
        proba = self.model.predict_proba(X)
        labels = self.model.classes_
        nov = self.novelty(X)
        thr = self.novelty_threshold
        upside = (self.orientation.predict(X).astype(bool) if self.orientation is not None
                  else np.zeros(len(X), bool))
        cs = contrasts if contrasts is not None else [None] * len(sources)
        out = []
        for src, row, nv, ct, up in zip(sources, proba, nov, cs, upside):
            k = int(np.argmax(row))
            blank = ct is not None and ct < self.min_contrast
            accepted = bool((thr is None or nv <= thr) and not blank and not up)
            out.append({
                'path': str(src),
                'label': str(labels[k]),
                'label_ru': CLASS_RU.get(str(labels[k]), ''),
                'confidence': float(row[k]),
                'accepted': accepted,
                'reason': ('' if accepted else
                           'низкий контраст' if blank else
                           'кадр перевёрнут' if up else 'не похож на DXA'),
                'novelty': float(nv),
                **{f'p_{c}': float(v) for c, v in zip(labels, row)},
            })
        return out

    def predict_batch(self, sources: list) -> list[dict]:
        if not sources:
            return []
        prepared = [self.prepare(s) for s in sources]
        return self.rows_from_vectors(sources, np.stack([v for v, _ in prepared]),
                                      [c for _, c in prepared])

    def classify(self, src) -> dict:
        """Один снимок -> полный результат, включая отказ от классификации."""
        return self.predict_batch([src])[0]


# -------------------------------------------------------------------- #
def collect(targets: list[str]) -> list[Path]:
    """Разворачивает файлы/папки/маски в список снимков."""
    files: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            files += [f for f in sorted(p.rglob('*')) if f.is_file()
                      and f.suffix.lower() in SUFFIXES]
        elif p.exists():
            files.append(p)
        else:
            files += [f for f in sorted(Path().glob(t)) if f.is_file()]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Классификация области DXA-снимка: lh / rh / spine')
    ap.add_argument('targets', nargs='+', help='файл .dcm, папка (рекурсивно) или маска')
    ap.add_argument('--model', type=Path, default=MODEL_PATH)
    ap.add_argument('--csv', type=Path, help='сохранить результат в CSV')
    ap.add_argument('--json', action='store_true', help='вывести JSON вместо таблицы')
    ap.add_argument('--quiet', action='store_true', help='только метки, по одной на строку')
    ap.add_argument('--strict', action='store_true',
                    help='непринятым кадрам ставить метку unknown вместо лучшей догадки')
    args = ap.parse_args()

    files = collect(args.targets)
    if not files:
        print('Не найдено ни одного снимка', file=sys.stderr)
        return 1

    clf = RegionClassifier(args.model)
    ok, vecs, conts, failed = [], [], [], []
    for f in files:
        try:
            v, c = clf.prepare(f)
            vecs.append(v); conts.append(c); ok.append(f)
        except Exception as e:                      # битый файл не должен ронять пачку
            failed.append((f, e))
    rows = clf.rows_from_vectors(ok, np.stack(vecs), conts) if vecs else []
    if args.strict:
        for r in rows:
            if not r['accepted']:
                r['label'], r['label_ru'] = 'unknown', 'не определено'
    if not rows:
        for f, e in failed:
            print(f'не прочитан {f}: {e}', file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.quiet:
        for r in rows:
            print(r['label'])
    else:
        w = min(70, max((len(Path(r['path']).name) for r in rows), default=10))
        print(f'{"файл":{w}} {"область":8} {"увер.":>6} {"новизна":>8}  статус')
        for r in rows:
            print(f'{Path(r["path"]).name:{w}} {r["label"]:8} {r["confidence"]:>6.3f} '
                  f'{r["novelty"]:>8.3f}  {r["reason"] or "ok"}')
        counts = {c: sum(r['label'] == c for r in rows) for c in CLASSES}
        n_rej = sum(not r['accepted'] for r in rows)
        print(f'\nвсего {len(rows)}: ' + ', '.join(f'{c}={n}' for c, n in counts.items())
              + (f'; не принято: {n_rej}' if n_rej else ''))

    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False, encoding='utf-8')
        print(f'CSV: {args.csv}')
    for f, e in failed:
        print(f'не прочитан {f}: {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
