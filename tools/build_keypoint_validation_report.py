"""Локальный интерактивный отчёт по сохранённому OOF модели точек.

PYTHONPATH=src python tools/build_keypoint_validation_report.py \
    --oof runs/keypoints/oof.npz --pack data/keypoints.npz

HTML содержит медицинские изображения, поэтому по умолчанию пишется в
игнорируемую Git папку Processing/. Он показывает сохранённые предсказания;
новый декодер нельзя проверить по старому OOF без его исходных тепловых карт.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dxa_qc.evaluate import level_errors
from dxa_qc.dataset import TruncatedDataset
from dxa_qc.points import HIP, SPINE_CORE, SPINE_EDGES


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("keypoint_validation_template.html")


def make_payload(oof_path: Path, pack_path: Path, folds_path: Path,
                 mm_per_px: float = 0.6) -> dict:
    with np.load(oof_path, allow_pickle=False) as z:
        oof = {k: z[k] for k in z.files}
    with np.load(pack_path, allow_pickle=True) as z:
        images = z["images"]
        pack_names = list(z["names"])
        pack_coords = z["coords"]
        pack_state = z["state"]
        pack_meta = z["meta"]
    names = list(oof["names"])
    if names != pack_names:
        raise ValueError("Порядок каналов в OOF и pack различается")
    with folds_path.open(newline="", encoding="utf-8") as f:
        fold_of = {r["study"]: int(r["fold"]) for r in csv.DictReader(f)}
    missing = set(map(str, oof["studies"])) - set(fold_of)
    if missing:
        raise ValueError(f"В folds.csv отсутствуют исследования: {sorted(missing)[:3]}")
    levels = level_errors(names, oof, np.asarray(pack_meta[:, 2]), mm_per_px)
    by_row = {int(x["row"]): x["shift"] for _, x in levels.iterrows()}
    threshold = np.asarray(oof["thresholds"], float)
    if threshold.shape not in ((len(names),), (len(oof["index"]), len(names))):
        raise ValueError(f"Неверная форма порогов: {threshold.shape}")
    rows = []
    for r, source_index in enumerate(oof["index"]):
        row_threshold = threshold if threshold.ndim == 1 else threshold[r]
        j = int(source_index)
        if j < 0 or j >= len(images) or str(pack_meta[j, 0]) != str(oof["ids"][r]):
            raise ValueError(f"OOF-строка {r} не совпадает со снимком pack[{j}]")
        region = str(oof["regions"][r])
        relevant = SPINE_CORE + SPINE_EDGES if region == "spine" else HIP
        buf = io.BytesIO()
        variant = int(oof["crop_variant"][r]) if "crop_variant" in oof else None
        if variant is None:
            image = np.asarray(images[j], dtype=np.uint8)
        else:
            n_crops = int(oof["crops_per_image"])
            one = TruncatedDataset({"images": images, "coords": pack_coords,
                                    "state": pack_state, "names": names},
                                   [j], crops_per_image=n_crops)
            clipped, _, _, _, _ = one.raw(variant)
            image = np.clip(clipped, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(buf, format="PNG")
        pts = []
        for name in relevant:
            if name not in names:
                continue
            c = names.index(name)
            labeled = bool(oof["labeled"][r, c])
            if not labeled:
                continue
            visible = bool(oof["visible"][r, c])
            found = bool(oof["conf"][r, c] >= row_threshold[c])
            true = [round(float(v), 2) for v in oof["true"][r, c]] if visible else None
            pred = [round(float(v), 2) for v in oof["pred"][r, c]] if found else None
            err = (round(float(np.linalg.norm(oof["pred"][r, c] - oof["true"][r, c])
                               * mm_per_px), 2) if visible and found else None)
            pts.append({"name": name, "true": true, "pred": pred,
                        "error": err, "visible": visible, "found": found,
                        "conf": round(float(oof["conf"][r, c]), 4),
                        "threshold": round(float(row_threshold[c]), 4)})
        shifts = by_row.get(r)
        centers_found = (region == "spine" and all(
            any(p["name"] == f"l{k}_center" and p["found"] for p in pts)
            for k in range(1, 5)))
        errors = [p["error"] for p in pts if p["error"] is not None]
        rows.append({"key": f"{j}:{variant}" if variant is not None else str(j),
                     "index": j, "crop_variant": variant, "id": str(oof["ids"][r]),
                     "study": str(oof["studies"][r]), "fold": fold_of[str(oof["studies"][r])],
                     "region": region, "width": int(image.shape[1]),
                     "height": int(image.shape[0]),
                     "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                     "points": pts, "median_error": round(float(np.median(errors)), 2) if errors else None,
                     "max_error": max(errors, default=None),
                     "missed": sum(p["visible"] and not p["found"] for p in pts),
                     "hallucinated": sum(not p["visible"] and p["found"] for p in pts),
                     "level_shifts": list(shifts) if shifts is not None else None,
                     "center4_found": centers_found,
                     "wrong_level": bool(any(shifts)) if shifts is not None else None})
    return {"source": str(oof_path),
            "protocol": str(oof["protocol"]) if "protocol" in oof else "saved_oof",
            "mm_per_px": mm_per_px, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=ROOT / "runs/keypoints/oof.npz")
    parser.add_argument("--pack", type=Path, default=ROOT / "data/keypoints.npz")
    parser.add_argument("--folds", type=Path, default=ROOT / "folds.csv")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "Processing/validation_report.html")
    parser.add_argument("--mm-per-px", type=float, default=0.6)
    args = parser.parse_args()
    payload = make_payload(args.oof, args.pack, args.folds, args.mm_per_px)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__PAYLOAD__", serialized)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Отчёт: {args.out} ({len(payload['rows'])} OOF-снимков)")


if __name__ == "__main__":
    main()
