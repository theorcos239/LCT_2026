"""Кросс-фолдовая калибровка видимости на врачебных spine-кропах."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dxa_qc import crops, folds, metrics  # noqa: E402
from dxa_qc.infer import Predictor, fold_checkpoints  # noqa: E402
from dxa_qc.pack import ABSENT, LABELED, load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/keypoints_crops")
    parser.add_argument("--pack", default="data/keypoints.npz")
    parser.add_argument("--crops", default="Processing/dxa-crops.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    source = load(args.pack)
    crop_data = crops.load(args.crops, source)
    names = list(source["names"])
    split = folds.load("folds.csv", set(map(str, source["studies"])))
    fold_of = dict(zip(split.study.astype(str), split.fold.astype(int)))
    parts = {}
    for checkpoint in fold_checkpoints(args.run):
        fold = int(checkpoint.stem.removeprefix("fold"))
        pred = Predictor([checkpoint], args.device)
        rows = np.flatnonzero([fold_of.get(str(s), -1) == fold
                               for s in crop_data["studies"]])
        conf = []
        for j in rows:
            result = pred.predict(np.asarray(crop_data["images"][j], np.uint8), chain=False)
            conf.append(result["confidence"])
        state = crop_data["state"][rows]
        parts[fold] = {
            "conf": np.asarray(conf), "visible": state == LABELED,
            "labeled": (state == LABELED) | (state == ABSENT),
        }

    output = []
    for target in sorted(parts):
        train = [part for fold, part in parts.items() if fold != target]
        if not train:
            train = [parts[target]]
        threshold = metrics.calibrate_thresholds(
            names,
            np.concatenate([p["conf"] for p in train]),
            np.concatenate([p["visible"] for p in train]),
            np.concatenate([p["labeled"] for p in train]),
            objective="balanced",
        )
        output.extend({"fold": target, "name": name, "threshold": float(value)}
                      for name, value in threshold.items())
    out = Path(args.out or Path(args.run) / "thresholds_crop.csv")
    pd.DataFrame(output).to_csv(out, index=False)
    print(f"{out}: {len(parts)} фолдов, {len(output)} порогов")


if __name__ == "__main__":
    main()
