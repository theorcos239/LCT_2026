"""Собрать единый G1-отчёт full-frame/oracle/auto из результатов evaluate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dxa_qc.metrics import endplate_angle  # noqa: E402


def _endplate_p90(path: Path) -> float:
    z = np.load(path)
    names = list(z["names"])
    idx = {n: i for i, n in enumerate(names)}
    pairs = [(idx[f"l{i}_edge_left"], idx[f"l{i}_edge_right"]) for i in range(1, 5)]
    errors = []
    for row, region in enumerate(z["regions"]):
        if str(region) != "spine":
            continue
        active = [pair for pair in pairs if z["visible"][row, list(pair)].all()]
        if len(active) < 3:
            continue
        ep = np.asarray(active)
        errors.append(abs(endplate_angle(z["pred"][row, ep])
                          - endplate_angle(z["true"][row, ep])))
    return float(np.percentile(errors, 90)) if errors else float("nan")


def _row(name: str, directory: Path) -> dict:
    summary = pd.read_csv(directory / "metrics_eval_summary.csv").iloc[0]
    accepted = int(summary.get("spine_frames_after_abstain", summary.get("spine_frames", 0)))
    total = accepted + int(summary.get("multicrop_abstain_frames", 0))
    raw_levels = int(summary["level_err_frames_chain"])
    effective_levels = int(summary.get("level_err_frames_after_abstain", raw_levels))
    return {
        "variant": name,
        "level_error_frames_raw": raw_levels,
        "level_error_frames": effective_levels,
        "spine_p90_mm": float(summary["spine_p90_mm_chain"]),
        "axis_median_deg": float(summary.get("e2e_axis_deg_median", np.nan)),
        "axis_p90_deg": float(summary.get("e2e_axis_deg_p90", np.nan)),
        "endplate_p90_deg": _endplate_p90(directory / "oof_evaluated.npz"),
        "abstain_frames": int(summary.get("multicrop_abstain_frames", 0)),
        "accepted_spine_frames": accepted,
        "coverage": accepted / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="Processing/validation_crops4_baseline")
    parser.add_argument("--oracle", default="Processing/validation_semantic_oracle")
    parser.add_argument("--auto", default="Processing/validation_semantic_auto")
    parser.add_argument("--out", default="runs/keypoints_crops/metrics_semantic_gates.csv")
    args = parser.parse_args()
    rows = [_row("full_frame", Path(args.baseline)),
            _row("semantic_oracle", Path(args.oracle)),
            _row("semantic_auto", Path(args.auto))]
    base = rows[0]
    for row in rows[1:]:
        checks = {
            "levels": row["level_error_frames"] <= 8,
            "localization": row["spine_p90_mm"] <= 10.14,
            "axis": row["axis_p90_deg"] <= 1.25,
            "endplates": row["endplate_p90_deg"] <= base["endplate_p90_deg"],
            "coverage": row["coverage"] >= .80,
        }
        row["checks"] = json.dumps(checks, ensure_ascii=False)
        row["gate_passed"] = all(checks.values())
    base["checks"], base["gate_passed"] = "", True
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    (out.with_suffix(".json")).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
