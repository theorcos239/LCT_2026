# -*- coding: utf-8 -*-
"""Прогон модели по всем файлам обучающего набора.

Проверяет не только точность, но и то, что инференс повторяет препроцессинг
обучения: дубликаты кадров внутри исследования обязаны получать одну метку, а
зеркальный кадр — противоположную сторону.

Запуск:  python -m region_clf.test_region_clf
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FLIP_LABEL, flip_lr, preprocess, read_dicom
from .predict import RegionClassifier

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent / 'НД_для_обучения' / 'Исследования'


def main() -> int:
    if not STUDIES.exists():
        # набор организаторов в репозиторий не входит: в нём снимки
        print(f'обучающий набор не найден: {STUDIES} — проверка пропущена')
        return 0
    lab = pd.read_csv(HERE / 'labels.csv')
    truth = dict(zip(lab.px_hash, lab.label))
    clf = RegionClassifier()
    files = sorted(STUDIES.rglob('*.dcm'))

    px = {f: read_dicom(f) for f in files}
    hashes = {f: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
              for f, a in px.items()}
    rows = clf.rows_from_vectors(files, np.stack([clf.vector(px[f]) for f in files]))

    errors = [(f, truth[hashes[f]], r['label'])
              for f, r in zip(files, rows) if truth[hashes[f]] != r['label']]
    conf = np.array([r['confidence'] for r in rows])
    print(f'всех файлов: {len(files)}, ошибок: {len(errors)}, '
          f'минимальная уверенность: {conf.min():.3f}')
    for f, t, p in errors[:20]:
        print(f'  {t} -> {p}: {f.relative_to(STUDIES)}')

    # зеркальный кадр обязан менять сторону (и не менять позвоночник)
    sample = lab.sample(40, random_state=0)
    flip_bad = []
    for r in sample.itertuples():
        img = preprocess(read_dicom(STUDIES / r.rel_path), crop=clf.crop)
        got = clf.predict(flip_lr(img))
        if got != FLIP_LABEL[r.label]:
            flip_bad.append((r.rel_path, r.label, got))
    print(f'проверка симметрии на {len(sample)} кадрах: несоответствий {len(flip_bad)}')
    for p, t, g in flip_bad:
        print(f'  {t} зеркально -> ожидалось {FLIP_LABEL[t]}, получено {g}: {p}')

    # все свои кадры обязаны проходить фильтр «чужого кадра»
    own_rejected = [r for r in rows if not r['accepted']]
    print(f'фильтр чужих кадров: своих отклонено {len(own_rejected)}/{len(rows)}')

    # а заведомо чужие — не проходить
    rng = np.random.default_rng(0)
    spine = px[files[0]]
    for f in files:
        a = px[f]
        if a.shape[1] < 300:
            hip = a
            break
    ood = {
        'чистый шум': rng.random((280, 280)).astype(np.float32),
        'пустой кадр': np.zeros((280, 280), np.float32),
        'поворот 90°': np.rot90(hip).copy(),
        'вверх ногами': np.flipud(hip).copy(),
        'фрагмент': hip[:60, :60].copy(),
        'инверсия': spine.max() - spine,
        'полосы': np.tile(np.r_[np.ones(10), np.zeros(10)].astype(np.float32), (280, 14)),
    }
    passed = [n for n, a in ood.items() if clf.classify(a)['accepted']]
    print(f'фильтр чужих кадров: чужих пропущено {len(passed)}/{len(ood)}'
          + (f' ({", ".join(passed)})' if passed else ''))

    ok = (not errors and not flip_bad and conf.min() > 0.5
          and not own_rejected and not passed)
    print('ИТОГ:', 'OK' if ok else 'ЕСТЬ ПРОБЛЕМЫ')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
