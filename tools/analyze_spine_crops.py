"""Получить семантические спецификации кропов позвоночника из dxa-crops-v1.

Runtime читает сгенерированный YAML и не зависит от врачебного JSON.
Частоты считаются только по подтверждённым spine-кропам, присутствующим в pack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dxa_qc.pack import LABELED, load as load_pack  # noqa: E402
from dxa_qc.points import SPINE_CORE, SPINE_EDGES  # noqa: E402

SPEC_NAMES = ("t12_l1", "l1_l2", "l2_l3", "l3_l4")
SPINE_NAMES = SPINE_CORE + SPINE_EDGES
# Фиксированные доли высоты исходного кадра. Границы включают все 92
# размеченных позвоночника с запасом >=10 px у целевых центров.
FIXED_BANDS = ((0.0, .44), (0.0, .59), (.19, .79), (.37, 1.0))
EXPECTED_CENTERS = (("l1_center",), ("l1_center", "l2_center"),
                    ("l2_center", "l3_center"), ("l3_center", "l4_center"))


def _projection_t(point: np.ndarray, l1: np.ndarray, step: np.ndarray) -> float:
    den = float(step @ step)
    if den < 1e-6:
        raise ValueError("нулевой шаг L1-L4")
    return float((point - l1) @ step / den)


def derive(export: dict, pack: dict) -> dict:
    if export.get("format") != "dxa-crops-v1" or export.get("schema_version") != 1:
        raise ValueError("ожидается dxa-crops-v1 schema_version=1")
    names = list(pack["names"])
    index = {name: i for i, name in enumerate(names)}
    by_id = {str(uid): i for i, uid in enumerate(pack["ids"])}
    by_image: dict[str, list[dict]] = defaultdict(list)
    for item in export.get("crops", []):
        uid = str(item.get("image_id", ""))
        if (item.get("status") == "confirmed" and item.get("region") == "spine"
                and uid in by_id):
            by_image[uid].append(item)

    grouped: dict[int, list[tuple[dict, int]]] = defaultdict(list)
    for uid, items in by_image.items():
        ordered = sorted(items, key=lambda x: (x["box"]["y0"], x["box"]["y1"]))
        h = int(ordered[0]["image_size"]["h"])
        for ordinal, item in enumerate(ordered):
            if len(ordered) == 4:
                rank = ordinal
            else:
                cy = (item["box"]["y0"] + item["box"]["y1"]) / (2 * h)
                rank = int(np.argmin(np.abs(np.asarray((.19, .315, .48, .725)) - cy)))
            grouped[rank].append((item, by_id[uid]))

    specs = {}
    all_output: set[str] = set()
    spine_indices = [i for i, region in enumerate(pack["regions"]) if region == "spine"]
    for rank, spec_name in enumerate(SPEC_NAMES):
        rows = grouped[rank]
        if not rows:
            raise ValueError(f"нет кропов ранга {rank}")
        counts = {name: 0 for name in SPINE_NAMES if name in index}
        point_presence = {name: {"selected": 0, "inside_unselected": 0,
                                 "outside": 0, "unknown": 0}
                          for name in counts}
        window_t: list[tuple[float, float]] = []
        for item, parent in rows:
            selected = set(item["included_points"])
            for name in counts:
                counts[name] += int(name in selected)
            state, coords = pack["state"][parent], pack["coords"][parent]
            box = item["box"]
            for name, states in point_presence.items():
                j = index[name]
                if state[j] != LABELED:
                    states["unknown"] += 1
                elif name in selected:
                    states["selected"] += 1
                elif (box["x0"] <= coords[j, 0] < box["x1"]
                      and box["y0"] <= coords[j, 1] < box["y1"]):
                    states["inside_unselected"] += 1
                else:
                    states["outside"] += 1
            centers = [f"l{i}_center" for i in range(1, 5)]
            if all(state[index[n]] == LABELED for n in centers):
                c = np.asarray([coords[index[n]] for n in centers], float)
                step = (c[3] - c[0]) / 3.0
                box = item["box"]
                x = float(c[:, 0].mean())
                try:
                    window_t.append((
                        _projection_t(np.array([x, box["y0"]]), c[0], step),
                        _projection_t(np.array([x, box["y1"]]), c[0], step),
                    ))
                except ValueError:
                    pass
        freq = {name: counts[name] / len(rows) for name in counts}
        output = [n for n in SPINE_NAMES if freq.get(n, 0.0) >= .90]
        fixed_coverage = {}
        for name in SPINE_NAMES:
            if name not in index:
                continue
            present = [i for i in spine_indices if pack["state"][i, index[name]] == LABELED]
            valid = 0
            for parent in present:
                h, w = pack["images"][parent].shape
                lo, hi = FIXED_BANDS[rank]
                y0, y1 = round(lo * h), round(hi * h)
                x, y = pack["coords"][parent, index[name]]
                valid += int(0 <= x < w and (y0 == 0 or y >= y0 + 10)
                             and (y1 == h or y < y1 - 10))
            fixed_coverage[name] = {"inside": valid, "labeled": len(present)}
        guaranteed_points = [n for n in output if fixed_coverage[n]["labeled"]
                             and fixed_coverage[n]["inside"] == fixed_coverage[n]["labeled"]]
        context = [n for n in SPINE_NAMES if .40 <= freq.get(n, 0.0) < .90]
        forbidden = [n for n in SPINE_NAMES if freq.get(n, 0.0) < .40]
        q = np.percentile(np.asarray(window_t), (25, 50, 75), axis=0)
        specs[spec_name] = {
            "rank": rank,
            "n_crops": len(rows),
            "fixed_band": list(FIXED_BANDS[rank]),
            "expected_centers": list(EXPECTED_CENTERS[rank]),
            "guaranteed_points": guaranteed_points,
            "fixed_coverage": fixed_coverage,
            "point_presence": point_presence,
            "output": output,
            "context": context,
            "forbidden": forbidden,
            "frequencies": {n: round(freq[n], 4) for n in SPINE_NAMES},
            "window_t": {
                "p25": [round(float(x), 3) for x in q[0]],
                "p50": [round(float(x), 3) for x in q[1]],
                "p75": [round(float(x), 3) for x in q[2]],
            },
        }
        all_output.update(output)

    missing_centers = [f"l{i}_center" for i in range(1, 5)
                       if f"l{i}_center" not in all_output]
    missing_edge_pairs = [f"l{i}" for i in range(1, 5)
                          if not {f"l{i}_edge_left", f"l{i}_edge_right"} <= all_output]
    if missing_centers:
        raise ValueError(f"центры не покрыты output-масками: {missing_centers}")
    if missing_edge_pairs:
        raise ValueError(f"пары краёв не покрыты output-масками: {missing_edge_pairs}")
    if any("th12_top" in spec["output"] for spec in specs.values()):
        raise ValueError("th12_top не должен входить в output кропов")
    for left, right in zip(SPEC_NAMES, SPEC_NAMES[1:]):
        common = set(specs[left]["output"]) & set(specs[right]["output"])
        if not common.intersection({f"l{i}_center" for i in range(1, 5)}):
            raise ValueError(f"соседние кропы {left} и {right} не перекрывают центр")
    return {
        "version": "semantic-v2",
        "source": {"format": "dxa-crops-v1", "confirmed_spine_crops":
                   sum(len(v) for v in grouped.values())},
        "specs": specs,
        "refine_radius_step": .5,
        "center_max_shift_step": .25,
        "edge_margin_px": 10,
        "min_crop_px": 96,
        "disagree_mm": 6.0,
        "anchor": {"step_ratio_max": 1.6, "chain_gap_min": 0.15},
    }


def presence_export(export: dict, pack: dict, source_sha256: str) -> dict:
    """Явное состояние каждой точки в каждом врачебном кропе, без CNN."""
    index = {name: i for i, name in enumerate(pack["names"])}
    by_id = {str(uid): i for i, uid in enumerate(pack["ids"])}
    by_image: dict[str, list[dict]] = defaultdict(list)
    for item in export.get("crops", []):
        uid = str(item.get("image_id", ""))
        if (item.get("status") == "confirmed" and item.get("region") == "spine"
                and uid in by_id):
            by_image[uid].append(item)
    records = []
    for uid, items in sorted(by_image.items()):
        ordered = sorted(items, key=lambda x: (x["box"]["y0"], x["box"]["y1"]))
        h = int(ordered[0]["image_size"]["h"])
        for ordinal, item in enumerate(ordered):
            if len(ordered) == 4:
                rank = ordinal
            else:
                cy = (item["box"]["y0"] + item["box"]["y1"]) / (2 * h)
                rank = int(np.argmin(np.abs(np.asarray((.19, .315, .48, .725)) - cy)))
            parent, box = by_id[uid], item["box"]
            selected = set(item["included_points"])
            states = {}
            for name in SPINE_NAMES:
                j = index.get(name)
                if j is None or pack["state"][parent, j] != LABELED:
                    state = "unknown"
                elif name in selected:
                    state = "selected"
                else:
                    x, y = pack["coords"][parent, j]
                    state = ("inside_unselected" if
                             box["x0"] <= x < box["x1"] and box["y0"] <= y < box["y1"]
                             else "outside")
                states[name] = state
            records.append({"crop_id": item["id"], "image_id": uid,
                            "type": SPEC_NAMES[rank], "box": box,
                            "point_state": states})
    return {"format": "dxa-crop-presence-v1", "source_sha256": source_sha256,
            "n_crops": len(records), "crops": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="Processing/dxa-crops.json")
    parser.add_argument("--pack", default="data/keypoints.npz")
    parser.add_argument("--out", default="configs/spine_multicrop.yaml")
    parser.add_argument("--presence-out", default="configs/spine_crop_presence.json")
    args = parser.parse_args()
    source_bytes = Path(args.crops).read_bytes()
    export = json.loads(source_bytes)
    pack = load_pack(args.pack)
    cfg = derive(export, pack)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    presence = presence_export(export, pack, hashlib.sha256(source_bytes).hexdigest())
    presence_path = Path(args.presence_out)
    presence_path.parent.mkdir(parents=True, exist_ok=True)
    presence_path.write_text(json.dumps(presence, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"{path}: {cfg['source']['confirmed_spine_crops']} spine-кропов")


if __name__ == "__main__":
    main()
