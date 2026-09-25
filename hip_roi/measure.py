# -*- coding: utf-8 -*-
"""CLI: отступы ROI для одного кадра или папки.

    python -m hip_roi.measure снимок.dcm --side lh
    python -m hip_roi.measure снимок.dcm --side auto --method kit3 --overlay out.png
    python -m hip_roi.measure папка --side auto --csv out.csv
    python -m hip_roi.measure снимок.dcm --side rh --method custom --L L1 --T T2 --B B3

--side auto берёт метку из region_clf (кадры не-бедра пропускаются).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from region_clf.features import read_image

from .kits import METHODS, measure_roi_margins
from .overlay import render_overlay

SUFFIXES = {'.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.tif', '.tiff', ''}


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            yield p


def resolve_side(path: Path, side: str, clf):
    if side != 'auto':
        return side, None
    r = clf.classify(str(path))
    if r['label'] not in ('lh', 'rh') or not r['accepted']:
        return None, r
    return r['label'], r


def main(argv=None):
    ap = argparse.ArgumentParser(description='Отступы ROI на снимке проксимального отдела бедра')
    ap.add_argument('path')
    ap.add_argument('--side', default='auto', choices=['lh', 'rh', 'auto'])
    ap.add_argument('--method', default='kit2', choices=METHODS)
    ap.add_argument('--L', default='L2')
    ap.add_argument('--T', default='T3')
    ap.add_argument('--B', default='B1')
    ap.add_argument('--overlay', help='PNG с визуализацией (только для одного файла)')
    ap.add_argument('--csv', help='выгрузка результатов в CSV')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)

    clf = None
    if args.side == 'auto':
        from region_clf import RegionClassifier
        clf = RegionClassifier()

    rows = []
    for p in iter_files(Path(args.path)):
        side, info = resolve_side(p, args.side, clf)
        if side is None:
            rows.append({'file': str(p), 'skipped': info['label'] if info else '?'})
            continue
        img = read_image(p)
        combo = {'L': args.L, 'T': args.T, 'B': args.B} if args.method == 'custom' else {}
        res = measure_roi_margins(img, side, args.method, **combo)
        res['file'] = str(p)
        rows.append(res)
        if args.overlay:
            render_overlay(img, res).save(args.overlay)

    if args.json:
        out = [{k: v for k, v in r.items() if k != 'diag'} for r in rows]
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    else:
        for r in rows:
            if 'skipped' in r:
                print(f"{r['file']}: пропущен ({r['skipped']})")
                continue
            print(f"{r['file']}: {r['side']} верх={r['m_top_mm']} низ={r['m_bottom_mm']} "
                  f"лат={r['m_lat_mm']} мм -> {'ok' if r['roi_ok'] else r['violation_text']}"
                  f"{'  [' + ', '.join(r['flags']) + ']' if r['flags'] else ''}")
    if args.csv:
        import pandas as pd
        pd.DataFrame([{k: v for k, v in r.items() if k != 'diag'} for r in rows]).to_csv(
            args.csv, index=False, encoding='utf-8-sig')
    return 0


if __name__ == '__main__':
    sys.exit(main())
