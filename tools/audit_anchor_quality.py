"""OOF-аудит отказов якорной цепочки на текущих весах без изменения модели."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dxa_qc.folds import load as load_folds  # noqa: E402
from dxa_qc.infer import Predictor, fold_checkpoints  # noqa: E402
from dxa_qc.multicrop import anchor_quality, load_semantic_config  # noqa: E402
from dxa_qc.pack import LABELED, load as load_pack  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/keypoints_crops")
    parser.add_argument("--pack", default="data/keypoints.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="runs/keypoints_crops/metrics_anchor_quality.csv")
    args = parser.parse_args()
    pack = load_pack(args.pack)
    names = list(pack["names"])
    idx = {name: i for i, name in enumerate(names)}
    cols = [idx[f"l{i}_center"] for i in range(1, 5)]
    split = load_folds("folds.csv", set(map(str, pack["studies"])))
    fold_of = dict(zip(split.study.astype(str), split.fold.astype(int)))
    cfg = load_semantic_config()
    rows = []
    for checkpoint in fold_checkpoints(args.run):
        fold = int(checkpoint.stem.removeprefix("fold"))
        model = Predictor([checkpoint], args.device)
        for i, (study, region) in enumerate(zip(pack["studies"], pack["regions"])):
            if str(region) != "spine" or fold_of[str(study)] != fold:
                continue
            if not np.all(pack["state"][i, cols] == LABELED):
                continue
            pred = model.predict(np.asarray(pack["images"][i], np.uint8))
            quality = anchor_quality(pred, cfg)
            gt = np.asarray(pack["coords"][i, cols], float)
            th12 = gt[0] - (gt[1] - gt[0])
            if np.all(pack["state"][i, [idx["th12_top"], idx["th12_bottom"]]] == LABELED):
                th12 = pack["coords"][i, [idx["th12_top"], idx["th12_bottom"]]].mean(axis=0)
            chain = np.vstack([th12, gt, gt[-1] + gt[-1] - gt[-2]])
            wrong = any(np.argmin(np.linalg.norm(chain - pred["coords"][col], axis=1)) != k + 1
                        for k, col in enumerate(cols))
            rows.append({
                "study": str(study), "image_id": str(pack["ids"][i]), "fold": fold,
                "level_error": bool(wrong), "anchor_reliable": bool(quality["reliable"]),
                "score_gap_top2": (pred.get("chain_hypotheses") or {}).get("score_gap_top2"),
                "reasons": ";".join(quality["reasons"]),
            })
    detail = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out.with_name("metrics_anchor_frames.csv"), index=False)
    summary = []
    for fold, group in [("all", detail)] + list(detail.groupby("fold")):
        summary.append({
            "fold": fold, "frames": len(group),
            "level_errors": int(group.level_error.sum()),
            "reliable_frames": int(group.anchor_reliable.sum()),
            "level_errors_accepted": int(group.loc[group.anchor_reliable, "level_error"].sum()),
            "level_errors_rejected": int(group.loc[~group.anchor_reliable, "level_error"].sum()),
            "coverage": float(group.anchor_reliable.mean()),
            "reason_counts": dict(Counter(r for item in group.reasons
                                          for r in item.split(";") if r)),
        })
    pd.DataFrame(summary).to_csv(out, index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
