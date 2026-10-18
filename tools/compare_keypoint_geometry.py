"""Paired service comparison on the keypoint pack, without local DICOM files.

The pack contains the same 242 annotated frames as region_clf/labels.csv.
Expert quality labels come from the tracked service/evaluation.csv. No image
pixels or per-patient predictions are written to the output CSV.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import trainset  # noqa: E402
from dxa_qc import infer  # noqa: E402
from service import evaluate as service_eval  # noqa: E402


def inputs(pack_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Restore service-orientation pixels and attach the existing quality labels."""
    with np.load(pack_path, allow_pickle=True) as pack:
        images: dict[str, np.ndarray] = {}
        scoliosis: dict[str, bool] = {}
        for image, meta in zip(pack["images"], pack["meta"]):
            px = np.asarray(image)
            if meta[3] == "rhip":
                px = px[:, ::-1]
            px = np.ascontiguousarray(px.astype(np.float32))
            digest = hashlib.md5(px.tobytes()).hexdigest()
            images[digest] = px
            scoliosis[digest] = "scoliosis" in str(meta[4]).split(",")
        count = len(pack["images"])

    labels = pd.read_csv(ROOT / "region_clf/labels.csv")
    folds = pd.read_csv(ROOT / "folds.csv", dtype={"study": str})[["study", "fold"]]
    saved = pd.read_csv(ROOT / "service/evaluation.csv")
    truth = ["rel_path", *(f"y_{criterion}" for criterion in trainset.CRITERIA)]
    frame = (labels.merge(folds, on="study").merge(saved[truth], on="rel_path")
             .loc[lambda d: d.px_hash.isin(images)].sort_values("rel_path")
             .reset_index(drop=True))
    if len(frame) != count or frame.px_hash.duplicated().any():
        raise ValueError("Pack, region labels and quality labels do not align one-to-one")
    frame["scoliosis"] = frame.px_hash.map(scoliosis).astype(bool)
    by_path = {r.rel_path: images[r.px_hash] for r in frame.itertuples()}
    return frame, by_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT / "data/keypoints.npz")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/keypoints_crops")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--out", type=Path, default=ROOT / "runs/keypoints_crops/metrics_service_geometry_comparison.csv")
    args = parser.parse_args()

    frame, by_path = inputs(args.pack)

    def frames(region: str | None = None) -> pd.DataFrame:
        if region == "hip":
            return frame[frame.label.isin(("lh", "rh"))].reset_index(drop=True)
        if region is not None:
            return frame[frame.label == region].reset_index(drop=True)
        return frame.copy()

    trainset.frames = frames
    trainset.read = by_path.__getitem__
    infer.RUN_DIR = args.run.resolve()

    rows = []
    scoliosis_by_path = frame.set_index("rel_path").scoliosis
    for method, use_points in (("geometry", False), ("keypoints_hypotheses", True)):
        result = service_eval.run(cnn=False, keypoints_oof=use_points,
                                  roi_rule="scan_length", folds=args.folds)
        if len(result) != int(frame.fold.isin(args.folds).sum()):
            raise ValueError(f"{method}: unexpected number of evaluated frames")
        result = result.assign(scoliosis=result.rel_path.map(scoliosis_by_path))
        for fold_name, part in [("all", result),
                                *((str(k), result[result.study.isin(
                                    frame.loc[frame.fold == k, "study"])])
                                  for k in args.folds)]:
            for subset, selected in (("all", part),
                                     ("without_scoliosis", part[~part.scoliosis])):
                metrics = service_eval.evaluate(selected, n_boot=0)
                quality = metrics["overall"]["binary_quality_class"]
                rows.append({"method": method, "fold": fold_name,
                             "subset": subset, "frames": len(selected),
                             "failures": int((selected.status == "Failure").sum()),
                             "positives": quality["positives"],
                             **{key: quality[key] for key in
                                ("tp", "fp", "tn", "fn", "f1", "sensitivity", "specificity")},
                             "macro_f1": metrics["overall"]["macro_f1"]})
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
