"""Подбор порога разрыва гипотез на остальных OOF-фолдах.

Порог для фолда k выбирается только по кадрам других фолдов. Максимальная
потеря правильных якорей на обучающей части ограничена 5 процентами.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def choose(train: pd.DataFrame, max_false_reject: float = .05) -> float:
    valid = train.dropna(subset=["score_gap_top2"])
    good = valid.loc[~valid.level_error, "score_gap_top2"].to_numpy(float)
    bad = valid.loc[valid.level_error, "score_gap_top2"].to_numpy(float)
    if not len(good) or not len(bad):
        return .15
    candidates = np.unique(np.r_[.15, valid.score_gap_top2.to_numpy(float) + 1e-6])
    feasible = [float(value) for value in candidates
                if float((good < value).mean()) <= max_false_reject]
    return max(feasible, key=lambda value: ((bad < value).mean(), -value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="runs/keypoints_crops/metrics_anchor_frames.csv")
    parser.add_argument("--out", default="runs/keypoints_crops/thresholds_anchor.csv")
    args = parser.parse_args()
    data = pd.read_csv(args.frames)
    rows = []
    for fold in sorted(data.fold.unique()):
        train = data[data.fold != fold]
        test = data[data.fold == fold]
        threshold = choose(train)
        rows.append({
            "fold": int(fold), "gap_min": threshold,
            "train_correct_rejected": int(((~train.level_error)
                                           & (train.score_gap_top2 < threshold)).sum()),
            "test_correct_rejected": int(((~test.level_error)
                                          & (test.score_gap_top2 < threshold)).sum()),
            "test_level_errors_rejected": int(((test.level_error)
                                               & (test.score_gap_top2 < threshold)).sum()),
        })
    rows.append({"fold": -1, "gap_min": choose(data),
                 "train_correct_rejected": 0, "test_correct_rejected": 0,
                 "test_level_errors_rejected": 0})
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
