"""Получить семантические спецификации кропов позвоночника из dxa-crops-v1.

Runtime читает сгенерированный YAML и не зависит от врачебного JSON.
Частоты считаются только по подтверждённым spine-кропам, присутствующим в pack.
"""
from __future__ import annotations

import argparse
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
    for rank, spec_name in enumerate(SPEC_NAMES):
        rows = grouped[rank]
        if not rows:
            raise ValueError(f"нет кропов ранга {rank}")
        counts = {name: 0 for name in SPINE_NAMES if name in index}
        window_t: list[tuple[float, float]] = []
        for item, parent in rows:
            selected = set(item["included_points"])
            for name in counts:
                counts[name] += int(name in selected)
            state, coords = pack["state"][parent], pack["coords"][parent]
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
        context = [n for n in SPINE_NAMES if .40 <= freq.get(n, 0.0) < .90]
        forbidden = [n for n in SPINE_NAMES if freq.get(n, 0.0) < .40]
        q = np.percentile(np.asarray(window_t), (25, 50, 75), axis=0)
        specs[spec_name] = {
            "rank": rank,
            "n_crops": len(rows),
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
        "anchor": {"step_ratio_max": 1.6, "chain_gap_min": 0.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", default="Разметка.json")
    parser.add_argument("--pack", default="data/keypoints.npz")
    parser.add_argument("--out", default="configs/spine_multicrop.yaml")
    args = parser.parse_args()
    export = json.loads(Path(args.crops).read_text(encoding="utf-8"))
    cfg = derive(export, load_pack(args.pack))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    print(f"{path}: {cfg['source']['confirmed_spine_crops']} spine-кропов")


if __name__ == "__main__":
    main()
