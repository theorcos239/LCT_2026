"""Сохранить настоящие OOF heatmap рабочего Predictor для проверки траекторий.

Нужны data/keypoints.npz и fold0.pt/fold1.pt. Каждый снимок обрабатывается
только моделью фолда, в обучении которого этот снимок не участвовал.

PYTHONPATH=src python tools/export_oof_heatmaps.py \\
  --run runs/keypoints --pack data/keypoints.npz \\
  --out Processing/oof_heatmaps.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dxa_qc import folds as foldsmod
from dxa_qc import pack as packmod
from dxa_qc.infer import Predictor, fold_checkpoints


def export(run: Path, pack: Path, out: Path, device: str) -> int:
    if not pack.exists():
        raise FileNotFoundError(f"Нет исходного набора снимков: {pack}")
    checkpoints = fold_checkpoints(run)
    if not checkpoints:
        raise FileNotFoundError(f"В {run} нет fold*.pt; по одним OOF-координатам карты не восстановить")
    data = packmod.load(pack)
    studies = np.asarray(data["studies"], dtype=str)
    regions = np.asarray(data["regions"], dtype=str)
    rows = []
    names = None
    for checkpoint in checkpoints:
        predictor = Predictor([checkpoint], device)
        if names is None:
            names = list(predictor.names)
        elif names != list(predictor.names):
            raise ValueError("У фолдов различаются каналы модели")
        fold = int(checkpoint.stem.removeprefix("fold"))
        split = foldsmod.load(predictor.cfg["data"]["folds"], set(studies))
        fold_of = dict(zip(split.study, split.fold))
        for j in np.flatnonzero((regions == "spine") &
                                np.array([fold_of[str(s)] == fold for s in studies])):
            image = np.asarray(data["images"][j], dtype=np.uint8)
            h, w = image.shape
            heatmap = predictor.heatmaps(image).cpu().numpy()[:, :h, :w]
            rows.append((int(j), h, w, heatmap))
    if not rows:
        raise ValueError("Не нашлись отложенные снимки позвоночника для доступных фолдов")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Один снимок попал в несколько отложенных фолдов")
    height = max(r[1] for r in rows)
    width = max(r[2] for r in rows)
    heatmaps = np.zeros((len(rows), len(names), height, width), dtype=np.float16)
    for n, (_, h, w, value) in enumerate(rows):
        heatmaps[n, :, :h, :w] = value.astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, heatmaps=heatmaps,
                        index=np.array([r[0] for r in rows], dtype=np.int32),
                        image_hw=np.array([[r[1], r[2]] for r in rows], dtype=np.int16),
                        names=np.array(names))
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("runs/keypoints"))
    p.add_argument("--pack", type=Path, default=Path("data/keypoints.npz"))
    p.add_argument("--out", type=Path, default=Path("Processing/oof_heatmaps.npz"))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    n = export(a.run, a.pack, a.out, a.device)
    print(f"Сохранено {n} OOF-карт позвоночника: {a.out}")


if __name__ == "__main__":
    main()
