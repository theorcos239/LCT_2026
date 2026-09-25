# -*- coding: utf-8 -*-
"""Единая точка входа для критериев позвоночника.

    python -m spine_qc.predict снимок.dcm
    python -m spine_qc.predict папка --csv out.csv
    python -m spine_qc.predict снимок.dcm --json --overlay out.png

    from spine_qc import SpineQC
    qc = SpineQC()
    res = qc.analyze('снимок.dcm')
    res['quality_class']          # 0 / 1
    res['violations']             # ['ось позвоночника отклонена на 6.9° (допустимо 5°)']
    res['criteria']['axis']       # измерения и флаги
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from region_clf.features import read_image

from . import criteria
from .appearance import AppearanceModel
from .calibrate import THRESHOLDS

CRITERIA_RU = {'position': 'укладка', 'axis': 'ось', 'artifacts': 'артефакты'}


class SpineQC:
    """Три критерия позвоночника по одному кадру.

    Модель вида кадра и калиброванные пороги подхватываются автоматически,
    если лежат рядом (`model.joblib`, `thresholds.json`). Без них модуль
    работает на одной геометрии и порогах по умолчанию, помечая это флагом:
    сервис должен оставаться работоспособным, даже если веса не подложены.
    """

    def __init__(self, model_path: str | Path | None = None,
                 thresholds_path: str | Path | None = None):
        self.appearance = AppearanceModel(model_path) if model_path else AppearanceModel()
        p = Path(thresholds_path) if thresholds_path else THRESHOLDS
        self.thresholds = dict(criteria.DEFAULTS)
        if p.exists():
            self.thresholds.update(json.loads(p.read_text(encoding='utf-8')))
        self._by_criterion = self.thresholds.get('appearance_by_criterion', {})

    def analyze(self, src) -> dict:
        """Путь или массив пикселей -> вердикт по трём критериям."""
        img = src if isinstance(src, np.ndarray) else read_image(src)
        probs = self.appearance.probabilities(img)
        thr = dict(self.thresholds)
        res = criteria.analyze(img, probs or None, thr)
        # у каждого критерия свой порог вероятности: положительных 6 и 17,
        # одна общая отсечка сместила бы более редкий критерий
        for name, p in probs.items():
            if name in self._by_criterion and name in res['criteria']:
                res['criteria'][name]['appearance_threshold'] = self._by_criterion[name]
        res.pop('column', None)
        return res

    def analyze_batch(self, sources: list) -> list[dict]:
        return [self.analyze(s) for s in sources]


def _row(path, res) -> dict:
    c = res['criteria']
    return {
        'path': str(path),
        'quality_class': res['quality_class'],
        'violations': '; '.join(res['violations']),
        'angle_deg': c['axis']['angle_deg'],
        'curvature_mm': c['axis']['curvature_mm'],
        'iliac_area': c['position']['iliac_area'],
        'artifact_len_mm': c['artifacts']['length_mm'],
        'artifact_score': c['artifacts']['score'],
        'flags': ';'.join(res['flags']),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Критерии качества кадра позвоночника')
    ap.add_argument('targets', nargs='+', help='файл .dcm, папка (рекурсивно) или маска')
    ap.add_argument('--csv', type=Path, help='сохранить таблицу')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--overlay', type=Path, help='визуализация (только для одного кадра)')
    args = ap.parse_args()

    from region_clf.predict import collect
    files = collect(args.targets)
    if not files:
        print('Не найдено ни одного снимка', file=sys.stderr)
        return 1

    qc = SpineQC()
    if not qc.appearance.available:
        print('модель вида кадра не найдена, работаю на одной геометрии', file=sys.stderr)

    rows, failed = [], []
    for f in files:
        try:
            rows.append(_row(f, qc.analyze(f)))
        except Exception as e:
            failed.append((f, e))

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=float))
    else:
        w = min(48, max((len(Path(r['path']).name) for r in rows), default=10))
        print(f'{"файл":{w}} {"класс":>5} {"угол":>7} {"гребни":>7} {"артефакт":>9}  нарушения')
        for r in rows:
            print(f'{Path(r["path"]).name:{w}} {r["quality_class"]:>5} '
                  f'{(r["angle_deg"] if r["angle_deg"] is not None else float("nan")):>6.1f}° '
                  f'{r["iliac_area"]:>7.3f} {r["artifact_len_mm"]:>8.0f}мм  {r["violations"]}')
        bad = sum(r['quality_class'] for r in rows)
        print(f'\nвсего {len(rows)}: с нарушением {bad}, качественных {len(rows) - bad}')

    if args.overlay and len(files) == 1:
        from .overlay import render_overlay
        render_overlay(read_image(files[0]), qc.analyze(files[0])).save(args.overlay)
        print(f'оверлей: {args.overlay}')
    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False, encoding='utf-8')
        print(f'CSV: {args.csv}')
    for f, e in failed:
        print(f'не обработан {f}: {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
