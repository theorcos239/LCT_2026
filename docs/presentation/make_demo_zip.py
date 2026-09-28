# -*- coding: utf-8 -*-
"""Демо-архив для показа: 7–8 исследований обучающего набора, все типы нарушений и норма.

    python docs/presentation/make_demo_zip.py            # -> demo.zip в корне репозитория
    python docs/presentation/make_demo_zip.py --out /tmp/demo.zip

Исследования выбираются по `service/evaluation.csv` (сквозная оценка): для
каждого типа нарушения — исследование, где сервис и эксперт согласны, плюс
два полностью нормальных и одно с несколькими нарушениями. Выбор
детерминирован (первое по имени), список печатается и совпадает с docs/DEMO.md.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'НД_для_обучения' / 'Исследования'
COL = {'spine_position': 'y_position', 'spine_axis': 'y_axis', 'spine_artifacts': 'y_artifacts',
       'hip_rotation': 'y_rotation', 'hip_roi': 'y_roi'}


def choose() -> list[tuple[str, str]]:
    ev = pd.read_csv(ROOT / 'service' / 'evaluation.csv')
    ev['violations'] = ev.violations.fillna('')
    ys = ev[list(COL.values())].fillna(0)
    picked: list[tuple[str, str]] = []
    used: set[str] = set()

    def take(study: str, why: str) -> None:
        if study not in used:
            used.add(study)
            picked.append((study, why))

    for v, c in COL.items():
        hit = ev[(ev[c] == 1) & ev.violations.str.contains(v)].sort_values('study')
        hit = hit[~hit.study.isin(used)]
        if len(hit):
            take(hit.study.iloc[0], f'{v}: найдено, эксперт согласен')
    # исследование, где у одного снимка больше одного верно найденного нарушения
    multi = ev[(ys.sum(axis=1) > 1) & (ev.violations.str.count(';') >= 1)].sort_values('study')
    multi = multi[~multi.study.isin(used)]
    if len(multi):
        take(multi.study.iloc[0], 'несколько нарушений на одном снимке')
    # полностью нормальные исследования: ни эксперт, ни сервис не нашли нарушений
    by_study = ev.assign(any_y=ys.sum(axis=1) > 0, any_p=ev.violations != '').groupby('study')
    clean = [s for s, g in by_study if not g.any_y.any() and not g.any_p.any()]
    for s in sorted(clean)[:2]:
        take(s, 'норма: нарушений нет ни у эксперта, ни у сервиса')
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'demo.zip')
    args = ap.parse_args()
    picked = choose()
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for study, _ in picked:
            for f in sorted((DATA / study).rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(DATA).as_posix())
    for study, why in picked:
        print(f'{study}  —  {why}')
    print(f'\n{args.out}: {len(picked)} исследований, {args.out.stat().st_size / 1e6:.1f} МБ')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
