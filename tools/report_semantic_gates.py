"""Парное сравнение G1 на OOF одного набора кадров и чекпоинтов."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.metrics import MM_PER_PX, axis_angle, endplate_angle  # noqa: E402


def read_oof(directory: Path) -> tuple[dict, dict]:
    """Старые отчёты без manifest намеренно не принимаются."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with np.load(directory / "oof_evaluated.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    required = {"names", "ids", "studies", "regions", "pred", "true", "visible",
                "conf", "thresholds", "abstain", "folds"}
    if missing := required - data.keys():
        raise ValueError(f"{directory}: неполный OOF: {sorted(missing)}")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{directory}: неизвестный manifest.json")
    return data, manifest


def pair_oof(baseline: dict, other: dict) -> dict:
    """Выравнивать по исследованию/кадру, запрещая подмену состава OOF."""
    def keys(z):
        return list(zip(map(str, z["studies"]), map(str, z["ids"])))

    reference, incoming = keys(baseline), keys(other)
    if len(set(reference)) != len(reference) or len(set(incoming)) != len(incoming):
        raise ValueError("повторяющиеся study/image ID")
    if set(reference) != set(incoming):
        raise ValueError("наборы отложенных кадров различаются")
    positions = {key: i for i, key in enumerate(incoming)}
    order = [positions[key] for key in reference]
    aligned = {
        key: value[order] if key != "names" and value.ndim and len(value) == len(order)
        else value for key, value in other.items()
    }
    for key in ("names", "regions", "visible", "true", "folds"):
        if not np.array_equal(baseline[key], aligned[key]):
            raise ValueError(f"OOF имеет разные {key}")
    if not np.allclose(baseline["thresholds"], aligned["thresholds"]):
        raise ValueError("пороги видимости различаются")
    return aligned


def metrics(z: dict) -> tuple[dict, pd.DataFrame]:
    names = list(map(str, z["names"]))
    idx = {name: i for i, name in enumerate(names)}
    centers = [idx[f"l{i}_center"] for i in range(1, 5)]
    edges = [(idx[f"l{i}_edge_left"], idx[f"l{i}_edge_right"])
             for i in range(1, 5)]
    rows, point_errors, accepted_errors, axis_errors, endplate_errors = [], [], [], [], []
    accepted_axis_errors, accepted_endplate_errors = [], []
    for row, region in enumerate(z["regions"]):
        if str(region) != "spine":
            continue
        gt_visible = z["visible"][row]
        found = z["conf"][row] >= z["thresholds"][row]
        accepted = not bool(z["abstain"][row])
        true, pred = z["true"][row], z["pred"][row]
        errors = np.linalg.norm(pred - true, axis=1) * MM_PER_PX
        point_errors.extend(errors[gt_visible].tolist())
        if accepted:
            accepted_errors.extend(errors[gt_visible].tolist())
        has_centers = bool(gt_visible[centers].all())
        shifted = None
        if has_centers:
            gt = true[centers]
            th12 = gt[0] - (gt[1] - gt[0])
            if gt_visible[idx["th12_top"]] and gt_visible[idx["th12_bottom"]]:
                th12 = true[[idx["th12_top"], idx["th12_bottom"]]].mean(axis=0)
            chain = np.vstack([th12, gt, gt[-1] + gt[-1] - gt[-2]])
            shifted = any(
                np.argmin(np.linalg.norm(chain - pred[col], axis=1)) != k + 1
                for k, col in enumerate(centers)
            )
            axis_error = abs(axis_angle(pred[centers]) - axis_angle(gt))
            axis_errors.append(axis_error)
            if accepted:
                accepted_axis_errors.append(axis_error)
        gt_pairs = [pair for pair in edges if gt_visible[list(pair)].all()]
        available_pairs = sum(bool(found[list(pair)].all()) for pair in gt_pairs)
        eligible_edges = len(gt_pairs) >= 3
        if eligible_edges:
            pairs = np.asarray(gt_pairs)
            edge_error = abs(endplate_angle(pred[pairs]) - endplate_angle(true[pairs]))
            endplate_errors.append(edge_error)
            if accepted:
                accepted_endplate_errors.append(edge_error)
        rows.append({
            "study": str(z["studies"][row]), "image_id": str(z["ids"][row]),
            "fold": int(z["folds"][row]), "accepted": accepted,
            "level_evaluable": has_centers, "level_error": shifted,
            "edge_eligible": eligible_edges,
            "three_edge_pairs_found": available_pairs >= 3,
        })
    detail = pd.DataFrame(rows)
    level_rows = detail[detail.level_evaluable]
    edge_rows = detail[detail.edge_eligible]
    def p90(values):
        return float(np.percentile(values, 90)) if values else None
    result = {
        "spine_frames": len(detail), "level_evaluable_frames": len(level_rows),
        "level_error_frames_raw": int(level_rows.level_error.sum()),
        "level_error_frames_accepted": int(level_rows.loc[level_rows.accepted, "level_error"].sum()),
        "accepted_spine_frames": int(detail.accepted.sum()),
        "abstain_frames": int((~detail.accepted).sum()),
        "coverage": float(detail.accepted.mean()),
        "spine_p90_mm": p90(point_errors),
        "accepted_spine_p90_mm": p90(accepted_errors),
        "axis_p90_deg": p90(axis_errors),
        "accepted_axis_p90_deg": p90(accepted_axis_errors),
        "endplate_p90_deg": p90(endplate_errors),
        "accepted_endplate_p90_deg": p90(accepted_endplate_errors),
        "edge_eligible_frames": len(edge_rows),
        "three_edge_pairs_rate": (float(edge_rows.three_edge_pairs_found.mean())
                                  if len(edge_rows) else 0.0),
    }
    return result, detail


def compare(sources: dict[str, tuple[dict, dict]]) -> tuple[list[dict], pd.DataFrame]:
    base_data, base_manifest = sources["full_frame"]
    if base_manifest["multicrop"] != "off":
        raise ValueError("baseline должен быть full-frame")
    reports, details = {}, {}
    for name, (data, manifest) in sources.items():
        if name != "full_frame":
            for key in ("pack_sha256", "checkpoint_sha256", "folds",
                        "threshold_protocol", "crops_per_image"):
                if manifest.get(key) != base_manifest.get(key):
                    raise ValueError(f"{name}: отличается {key}")
            if name == "semantic_auto" and not manifest.get("anchor_thresholds_sha256"):
                raise ValueError("semantic_auto: нет подтверждения crossfit-порога якоря")
            data = pair_oof(base_data, data)
        reports[name], details[name] = metrics(data)
        reports[name]["variant"] = name
        reports[name]["provenance_verified"] = True
        if name == "semantic_auto":
            reports[name]["anchor_thresholds_sha256"] = manifest["anchor_thresholds_sha256"]
    baseline = reports["full_frame"]
    if any(baseline[key] is None for key in
           ("spine_p90_mm", "axis_p90_deg", "endplate_p90_deg")):
        raise ValueError("baseline: недостаточно размеченных точек для G1")
    for row in reports.values():
        row["checkpoint_sha256"] = base_manifest["checkpoint_sha256"]
        row["pack_sha256"] = base_manifest["pack_sha256"]
    baseline_detail = details["full_frame"].set_index(["study", "image_id"])
    for name in ("semantic_oracle", "semantic_auto"):
        row = reports[name]
        detail = details[name].set_index(["study", "image_id"])
        selected = baseline_detail.loc[detail.index[detail.accepted]]
        row["baseline_level_errors_on_accepted"] = int(
            selected.loc[selected.level_evaluable, "level_error"].sum()
        )
        subset_abstain = np.zeros(len(base_data["ids"]), bool)
        subset_abstain[np.asarray(base_data["regions"]) == "spine"] = ~detail.accepted.to_numpy(bool)
        baseline_subset = {**base_data, "abstain": subset_abstain}
        subset_metrics, _ = metrics(baseline_subset)
        row["baseline_spine_p90_on_accepted"] = subset_metrics["accepted_spine_p90_mm"]
        row["baseline_axis_p90_on_accepted"] = subset_metrics["accepted_axis_p90_deg"]
        row["baseline_endplate_p90_on_accepted"] = subset_metrics["accepted_endplate_p90_deg"]
        checks = {
            "five_fold_oof": base_manifest["folds"] == [0, 1, 2, 3, 4],
            "levels_raw": row["level_error_frames_raw"] <= 8,
            "levels_accepted": row["level_error_frames_accepted"] <= 8,
            "localization": (row["spine_p90_mm"] is not None and
                             row["spine_p90_mm"] <= min(10.14, baseline["spine_p90_mm"])),
            "axis": (row["axis_p90_deg"] is not None and
                     row["axis_p90_deg"] <= min(1.25, baseline["axis_p90_deg"])),
            "endplates": (row["endplate_p90_deg"] is not None and
                          row["endplate_p90_deg"] <= baseline["endplate_p90_deg"]),
            "edge_pairs": row["three_edge_pairs_rate"] >= .80,
            "coverage": row["coverage"] >= .80,
        }
        row["checks"], row["gate_passed"] = checks, all(checks.values())
    baseline["checks"], baseline["gate_passed"] = {}, None
    paired = details["full_frame"].rename(columns=lambda col: f"baseline_{col}"
                                            if col not in {"study", "image_id", "fold"} else col)
    for name in ("semantic_oracle", "semantic_auto"):
        paired = paired.merge(details[name].drop(columns="fold").rename(
            columns=lambda col: f"{name}_{col}" if col not in {"study", "image_id"} else col),
            on=["study", "image_id"], validate="one_to_one")
    return list(reports.values()), paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="Processing/validation_crops4_baseline")
    parser.add_argument("--oracle", default="Processing/validation_semantic_oracle")
    parser.add_argument("--auto", default="Processing/validation_semantic_auto")
    parser.add_argument("--out", default="runs/keypoints_crops/metrics_semantic_gates.csv")
    args = parser.parse_args()
    sources = {name: read_oof(Path(path)) for name, path in
               (("full_frame", args.baseline), ("semantic_oracle", args.oracle),
                ("semantic_auto", args.auto))}
    report, paired = compare(sources)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(report)
    table.to_csv(out, index=False)
    out.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    paired.to_csv(out.with_name("metrics_semantic_paired.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
