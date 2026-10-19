"""Парный OOF-отчёт по алгоритму фиксированных кропов и консенсуса.

Запуск: python tools/report_crop_consensus.py --baseline ... --consensus ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.metrics import MM_PER_PX  # noqa: E402
from tools.report_semantic_gates import metrics, pair_oof, read_oof  # noqa: E402


def compare(baseline_dir: Path, consensus_dir: Path) -> dict:
    baseline, bm = read_oof(baseline_dir)
    consensus, cm = read_oof(consensus_dir)
    for key in ("pack_sha256", "checkpoint_sha256", "folds",
                "threshold_protocol", "crops_per_image"):
        if bm.get(key) != cm.get(key):
            raise ValueError(f"различается {key}")
    if cm.get("multicrop") not in ("consensus", "consensus_adaptive") or not cm.get("semantic_config_sha256"):
        raise ValueError("ожидался OOF consensus с зафиксированной конфигурацией")
    consensus = pair_oof(baseline, consensus)
    summary, details = {}, {}
    for name, data in (("full_frame", baseline), ("consensus", consensus)):
        result, detail = metrics(data)
        idx = {n: i for i, n in enumerate(data["names"])}
        mask = np.asarray(data["regions"]) == "spine"
        centers = [idx[f"l{i}_center"] for i in range(1, 5)]
        gt_visible = data["visible"][mask][:, centers]
        error = np.linalg.norm(data["pred"][mask][:, centers]
                               - data["true"][mask][:, centers], axis=2) * MM_PER_PX
        values = error[gt_visible]
        result["centers_p90_mm"] = float(np.percentile(values, 90))
        result["center_errors_over_15mm"] = int(np.sum(values > 15))
        result["level_error_image_ids"] = detail.loc[detail.level_error == True,
                                                       "image_id"].tolist()
        summary[name] = result
        details[name] = detail.set_index(["study", "image_id"])
    # Решение о включении не принимается по «красивой» метрике на отобранных
    # кадрах: сырые ошибки и покрытие тоже обязательны.
    b, c = summary["full_frame"], summary["consensus"]
    paired = details["full_frame"].join(
        details["consensus"][["accepted", "level_error"]],
        lsuffix="_baseline", rsuffix="_consensus")
    accepted = paired[paired.accepted_consensus & paired.level_evaluable]
    base_bad = accepted.level_error_baseline.astype(bool)
    new_bad = accepted.level_error_consensus.astype(bool)
    c["baseline_level_errors_on_accepted"] = int(base_bad.sum())
    c["corrected_levels_on_accepted"] = int(
        (base_bad & ~new_bad).sum())
    c["introduced_levels_on_accepted"] = int(
        (~base_bad & new_bad).sum())
    checks = {
        "five_folds": bm["folds"] == [0, 1, 2, 3, 4],
        "levels_raw_not_worse": c["level_error_frames_raw"] <= b["level_error_frames_raw"],
        "spine_p90_not_worse": c["spine_p90_mm"] <= b["spine_p90_mm"],
        "axis_p90_not_worse": c["axis_p90_deg"] <= b["axis_p90_deg"],
        "coverage_at_least_80pct": c["coverage"] >= .8,
    }
    return {"baseline": summary["full_frame"], "consensus": summary["consensus"],
            "checks": checks, "gate_passed": all(checks.values()),
            "folds": bm["folds"], "mode": cm["multicrop"],
            "checkpoint_sha256": bm["checkpoint_sha256"],
            "semantic_config_sha256": cm["semantic_config_sha256"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--consensus", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    result = compare(args.baseline, args.consensus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks": result["checks"],
                      "baseline": {k: result["baseline"][k] for k in
                                   ("level_error_frames_raw", "spine_p90_mm", "axis_p90_deg")},
                      "consensus": {k: result["consensus"][k] for k in
                                    ("level_error_frames_raw", "spine_p90_mm", "axis_p90_deg")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
