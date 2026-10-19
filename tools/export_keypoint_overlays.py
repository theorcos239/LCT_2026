"""Прогон модели точек и ZIP с визуализацией разметки позвонков.

По умолчанию — полные снимки позвоночника из обучающего pack
(`data/keypoints.npz`, исходный датасет).

PYTHONPATH=src:. python tools/export_keypoint_overlays.py
PYTHONPATH=src:. python tools/export_keypoint_overlays.py --oof
PYTHONPATH=src:. python tools/export_keypoint_overlays.py --input Processing/demo_dicom
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from dxa_qc.data import guess_region, read_dicom
from dxa_qc.infer import Predictor, fold_checkpoints, load_predictor
from dxa_qc.pack import load as load_pack
from dxa_qc.points import SPINE_CORE, SPINE_EDGES
from dxa_qc.spine_shape import curve_descriptors, fused_tilt, source_variances, tilt_sources
from service.keypoints_backend import to_model_uint8


ROOT = Path(__file__).resolve().parents[1]
SPINE_NAMES = SPINE_CORE + SPINE_EDGES

COLORS = {
    "center": (0, 220, 90),
    "disc": (80, 180, 255),
    "edge": (255, 200, 60),
    "th12": (255, 120, 40),
    "iliac": (220, 80, 220),
}
GT_COLOR = (255, 255, 255)


def point_kind(name: str) -> str:
    if name.startswith("th12"):
        return "th12"
    if name.startswith("iliac"):
        return "iliac"
    if name.startswith("disc_"):
        return "disc"
    if "_edge_" in name:
        return "edge"
    return "center"


def short_label(name: str) -> str:
    return (name.replace("th12_", "T12 ")
                .replace("disc_", "")
                .replace("_center", "")
                .replace("_edge_right", "R")
                .replace("_edge_left", "L")
                .replace("iliac_right", "il R")
                .replace("iliac_left", "il L")
                .replace("_", "-"))


def to_rgb(px: np.ndarray) -> Image.Image:
    img, _ = to_model_uint8(px)
    return Image.fromarray(img).convert("RGB")


def scale_image(img: Image.Image, scale: int) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def draw_spine(img: Image.Image, names: list[str], coords: np.ndarray,
               visible: np.ndarray, conf: np.ndarray, scale: int = 3,
               gt_coords: np.ndarray | None = None,
               gt_labeled: np.ndarray | None = None) -> Image.Image:
    """Крупный оверлей: предсказание + опционально эксперт (белые точки)."""
    out = scale_image(img, scale)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    idx = {n: i for i, n in enumerate(names)}

    if gt_coords is not None and gt_labeled is not None:
        for name in SPINE_NAMES:
            i = idx.get(name)
            if i is None or not bool(gt_labeled[i]):
                continue
            x, y = float(gt_coords[i, 0]) * scale, float(gt_coords[i, 1]) * scale
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=GT_COLOR)

    pts = {}
    for name in SPINE_NAMES:
        i = idx.get(name)
        if i is None or not bool(visible[i]):
            continue
        x, y = float(coords[i, 0]) * scale, float(coords[i, 1]) * scale
        pts[name] = (x, y, float(conf[i]), point_kind(name))

    centers = [pts[f"l{k}_center"][:2] for k in range(1, 5) if f"l{k}_center" in pts]
    if len(centers) >= 2:
        draw.line(centers, fill=(0, 255, 140), width=2)

    for k in range(1, 5):
        left, right = f"l{k}_edge_left", f"l{k}_edge_right"
        if left in pts and right in pts:
            draw.line([pts[left][:2], pts[right][:2]], fill=(255, 210, 80), width=1)

    for name, (x, y, c, kind) in pts.items():
        color = COLORS[kind]
        r = 5 if kind == "center" else 4
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        draw.text((x + 6, y - 8), f"{short_label(name)} {c:.2f}", fill=color, font=font)

    caption = f"{img.width}x{img.height}px  visible={len(pts)}/{len(SPINE_NAMES)}"
    draw.rectangle((0, 0, 8 + 7 * len(caption), 18), fill=(0, 0, 0))
    draw.text((4, 2), caption, fill=(220, 220, 220), font=font)
    return out


def draw_multicrop_sources(
    img: Image.Image,
    pred: dict,
    scale: int = 3,
    gt_coords: np.ndarray | None = None,
    gt_labeled: np.ndarray | None = None,
) -> Image.Image:
    """Добавить рамки полос и сырые точки источников поверх итогового merge."""
    out = draw_spine(
        img, list(pred["names"]), np.asarray(pred["coords"]),
        np.asarray(pred["visible"]), np.asarray(pred["confidence"]), scale=scale,
        gt_coords=gt_coords, gt_labeled=gt_labeled,
    )
    draw = ImageDraw.Draw(out)
    colors = [(255, 90, 90), (70, 190, 255), (255, 170, 60),
              (180, 90, 255), (130, 130, 130)]
    meta = pred.get("multicrop") or {}
    sources = meta.get("sources") or []
    for source_no, source in enumerate(sources):
        color = colors[source_no % len(colors)]
        box = source["box"]
        if isinstance(box, (list, tuple)):
            x0, y0, x1, y1 = box
        else:
            x0, y0, x1, y1 = box.x0, box.y0, box.x1, box.y1
        draw.rectangle(
            (x0 * scale, y0 * scale, (x1 - 1) * scale, (y1 - 1) * scale),
            outline=color, width=1,
        )
        label = source.get("spec", f"crop-{source_no}")
        draw.text((x0 * scale + 2, y0 * scale + 2), label, fill=color)
        if "visible" in source:
            for i, is_visible in enumerate(source["visible"]):
                if not bool(is_visible):
                    continue
                x, y = source["coords"][i] * scale
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        else:
            idx = {n: i for i, n in enumerate(pred["names"])}
            for name in source.get("points", []):
                x, y = np.asarray(pred["coords"])[idx[name]] * scale
                draw.line((x, y, x0 * scale + 3, y0 * scale + 3), fill=color, width=1)

    # Геометрия исправленного угла: центры, пластинки и итоговая подпись.
    src = tilt_sources(np.asarray(pred["coords"]), np.asarray(pred["visible"]),
                       list(pred["names"]))
    curve = curve_descriptors(np.asarray(pred["coords"]), np.asarray(pred["visible"]),
                              list(pred["names"]))
    fused = fused_tilt(src, source_variances(src, curve))
    if fused["theta_fused_deg"] is not None:
        text = (f"axis={fused['theta_fused_deg']:.2f}+/-{fused['sigma_deg']:.2f} "
                f"Q={fused['Q']:.2f}")
        draw.rectangle((0, 19, 9 + 7 * len(text), 38), fill=(0, 0, 0))
        draw.text((4, 21), text, fill=(255, 255, 255))
    return out


def collect_dicoms(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".dcm", ".dicom"})


def predictors_by_fold(run: Path, device: str) -> dict[int, Predictor]:
    out = {}
    for path in fold_checkpoints(run):
        fold = int(path.stem.replace("fold", ""))
        out[fold] = Predictor([path], device)
    return out


def iter_pack_frames(pack_path: Path, folds_path: Path, region: str = "spine"):
    pack = load_pack(pack_path)
    fold_of = {}
    if folds_path.exists():
        fold_of = dict(zip(
            pd.read_csv(folds_path)["study"].astype(str),
            pd.read_csv(folds_path)["fold"].astype(int),
        ))
        # re-read once
        folds = pd.read_csv(folds_path)
        fold_of = dict(zip(folds.study.astype(str), folds.fold.astype(int)))
    for i, (image, region_i, study, uid) in enumerate(zip(
            pack["images"], pack["regions"], pack["studies"], pack["ids"])):
        if str(region_i) != region:
            continue
        yield {
            "index": i,
            "id": str(uid),
            "study": str(study),
            "fold": fold_of.get(str(study)),
            "image": np.asarray(image),
            "coords_gt": pack["coords"][i],
            "state": pack["state"][i],
            "names": pack["names"],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, default=ROOT / "data" / "keypoints.npz",
                    help="обучающий pack исходного датасета")
    ap.add_argument("--folds", type=Path, default=ROOT / "folds.csv")
    ap.add_argument("--input", type=Path, default=None,
                    help="если задан — брать DICOM из папки вместо pack")
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "keypoints")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "Processing" / "keypoints_dataset_spine.zip")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--oof", action="store_true",
                    help="честный OOF: кадр считает только модель своего фолда")
    ap.add_argument("--no-gt", action="store_true",
                    help="не рисовать экспертные точки")
    ap.add_argument("--multicrop", nargs="?", const="semantic",
                    choices=("v1", "semantic"), default=None,
                    help="v1: все каналы; semantic: разрешённые точки и якорная нумерация")
    ap.add_argument("--limit", type=int, default=None,
                    help="обработать только первые N spine-кадров (smoke/просмотр)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    kept = 0

    if args.input is not None:
        files = collect_dicoms(args.input)
        if not files:
            raise SystemExit(f"нет DICOM в {args.input}")
        print(f"загрузка ансамбля из {args.run} ...")
        predictor = load_predictor(args.run, device=args.device)
        frames = []
        for path in files:
            dicom = read_dicom(path)
            if guess_region(dicom.pixels) != "spine":
                continue
            frames.append({"path": path, "image": dicom.pixels,
                           "id": path.stem, "study": path.parts[-4] if len(path.parts) >= 4 else "",
                           "fold": None, "coords_gt": None, "state": None,
                           "names": predictor.names})
        get_pred = lambda _frame: predictor
        source = f"dicom:{args.input}"
    else:
        if not args.pack.exists():
            raise SystemExit(
                f"нет {args.pack}. Восстановите из backup/before-cleanup-2026-09-29 "
                "или соберите python -m dxa_qc.pack")
        frames = list(iter_pack_frames(args.pack, args.folds, region="spine"))
        if args.oof:
            print(f"загрузка OOF-фолдов из {args.run} ...")
            by_fold = predictors_by_fold(args.run, args.device)
            if not by_fold:
                raise SystemExit("нет fold*.pt для OOF")
            ensemble = load_predictor(args.run, device=args.device)

            def get_pred(frame):
                fold = frame["fold"]
                if fold in by_fold:
                    return by_fold[fold]
                return ensemble
        else:
            print(f"загрузка ансамбля из {args.run} ...")
            predictor = load_predictor(args.run, device=args.device)
            get_pred = lambda _frame: predictor
        source = f"pack:{args.pack}"

    if args.limit is not None:
        frames = frames[:max(0, args.limit)]
    print(f"источник: {source}, кадров позвоночника: {len(frames)}")

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for n, frame in enumerate(frames, 1):
            image = frame["image"]
            img_u8, flags = to_model_uint8(image)
            pred_model = get_pred(frame)
            pred = (pred_model.predict_semantic(img_u8)
                    if args.multicrop == "semantic"
                    else pred_model.predict_multicrop(img_u8, {"mode": "v1"})
                    if args.multicrop == "v1" else pred_model.predict(img_u8))
            names = list(pred["names"])
            spine_vis = sum(
                bool(pred["visible"][names.index(name)])
                for name in SPINE_NAMES if name in names
            )
            h, w = img_u8.shape
            row = {
                "index": frame.get("index", n - 1),
                "id": frame["id"],
                "study": frame["study"],
                "fold": frame.get("fold"),
                "height": h,
                "width": w,
                "model_region": pred["region"],
                "visible_spine": spine_vis,
                "flags": ";".join(flags),
                "oof": bool(args.oof and frame.get("fold") is not None),
                "multicrop": args.multicrop or "off",
                "point_quality": (pred.get("multicrop") or {}).get("point_quality"),
                "crop_abstain": bool((pred.get("multicrop") or {}).get("abstain")),
                "crop_conflicts": len(
                    (pred.get("multicrop") or {}).get("chain_conflicts", [])),
            }
            rows.append(row)
            print(f"[{n}/{len(frames)}] {h}x{w} fold={frame.get('fold')} "
                  f"{pred['region']} vis={spine_vis} {frame['id']}")

            gt_coords = None if args.no_gt else frame.get("coords_gt")
            gt_labeled = None
            if gt_coords is not None and frame.get("state") is not None:
                # LABELED = 2 в pack.py
                gt_labeled = np.asarray(frame["state"]) == 2

            rgb = to_rgb(image)
            original = scale_image(rgb, args.scale)
            if args.multicrop:
                overlay = draw_multicrop_sources(
                    rgb, pred, scale=args.scale,
                    gt_coords=gt_coords, gt_labeled=gt_labeled,
                )
            else:
                overlay = draw_spine(
                    rgb, names, pred["coords"], pred["visible"],
                    pred["confidence"], scale=args.scale,
                    gt_coords=gt_coords, gt_labeled=gt_labeled,
                )
            fold_tag = f"f{frame['fold']}" if frame.get("fold") is not None else "ens"
            stem = (f"{kept:03d}_{h}x{w}_{fold_tag}_{pred['region']}_"
                    f"v{spine_vis}_{frame['id'][:12]}")
            # Пара файлов рядом в сортировке: …_orig.png, …_pred.png
            zf.writestr(f"{stem}_orig.png", png_bytes(original))
            zf.writestr(f"{stem}_pred.png", png_bytes(overlay))
            kept += 1

        text = io.StringIO()
        w = csv.DictWriter(text, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        zf.writestr("summary.csv", text.getvalue())
        zf.writestr("README.txt",
                    "Пары PNG на каждый кадр позвоночника исходного датасета:\n"
                    "  *_orig.png — исходный снимок (без разметки)\n"
                    "  *_pred.png — тот же кадр с точками модели "
                    "(цветные) и эксперта (белые, если pack)\n"
                    f"source={source}\noof={args.oof}\npairs={kept}\n")

    print(f"готово: {args.out} ({kept} пар PNG + summary.csv)")


if __name__ == "__main__":
    main()
