"""Multi-crop инференс позвоночника с переносом точек в полный кадр.

Главная цель модуля — не само усреднение, а обнаружение несовместимой
нумерации позвонков. При конфликте цепочек вызывающая сторона должна
отказаться от точечной геометрии и оставить контурный метод.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


DEFAULT_BANDS = ((0.00, 0.38), (0.13, 0.50), (0.27, 0.69), (0.45, 1.00))
CENTERS = tuple(f"l{i}_center" for i in range(1, 5))
DEFAULT_SEMANTIC_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "spine_multicrop.yaml"
)


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int
    full_frame: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def spine_crop_boxes(
    height: int,
    width: int,
    bands=DEFAULT_BANDS,
    include_full_frame: bool = True,
) -> list[CropBox]:
    """Построить детерминированные полосы по статистике врачебной разметки."""
    if height <= 0 or width <= 0:
        raise ValueError("размер изображения должен быть положительным")
    boxes = []
    for lo, hi in bands:
        lo, hi = float(lo), float(hi)
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"неверная полоса: {(lo, hi)}")
        y0 = max(0, min(height - 1, int(round(lo * height))))
        y1 = max(y0 + 1, min(height, int(round(hi * height))))
        boxes.append(CropBox(0, y0, width, y1))
    if include_full_frame:
        boxes.append(CropBox(0, 0, width, height, full_frame=True))
    return boxes


def map_prediction(
    prediction: dict,
    box: CropBox,
    image_shape: tuple[int, int],
    edge_margin_px: int,
) -> dict:
    """Перенести локальный prediction в координаты исходного кадра."""
    names = list(prediction["names"])
    coords_local = np.asarray(prediction["coords"], dtype=float)
    coords = coords_local + np.array([box.x0, box.y0], dtype=float)
    visible = np.asarray(prediction["visible"], dtype=bool).copy()
    confidence = np.asarray(prediction["confidence"], dtype=float)

    if not box.full_frame and edge_margin_px > 0:
        h, w = box.shape
        x, y = coords_local[:, 0], coords_local[:, 1]
        # Отбрасываем только у искусственно созданных границ. Исходный край
        # кадра не означает, что точка сомнительна.
        if box.x0 > 0:
            visible &= x >= edge_margin_px
        if box.x1 < image_shape[1]:
            visible &= x < w - edge_margin_px
        if box.y0 > 0:
            visible &= y >= edge_margin_px
        if box.y1 < image_shape[0]:
            visible &= y < h - edge_margin_px

    return {
        "box": box,
        "names": names,
        "coords": coords,
        "confidence": confidence,
        "visible": visible,
        "region": prediction.get("region"),
        "chain_hypotheses": prediction.get("chain_hypotheses"),
    }


def _center_map(prediction: dict) -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(prediction["names"])}
    return {
        name: prediction["coords"][idx[name]]
        for name in CENTERS
        if name in idx and prediction["visible"][idx[name]]
    }


def chain_conflicts(predictions: list[dict], disagree_px: float) -> list[dict]:
    """Найти пары кропов с несовместимой нумерацией L1–L4.

    Помимо прямого расхождения одноимённых центров проверяется, не совпадает
    ли заметно лучше цепочка со сдвигом меток на один уровень.
    """
    conflicts = []
    maps = [_center_map(p) for p in predictions]
    for a in range(len(maps)):
        for b in range(a + 1, len(maps)):
            common = [name for name in CENTERS if name in maps[a] and name in maps[b]]
            if not common:
                continue
            direct = float(np.median([
                np.linalg.norm(maps[a][name] - maps[b][name]) for name in common
            ]))
            best_shift, best_distance, best_n = 0, direct, len(common)
            for shift in (-1, 1):
                pairs = []
                for i, name in enumerate(CENTERS):
                    j = i + shift
                    if 0 <= j < len(CENTERS):
                        other = CENTERS[j]
                        if name in maps[a] and other in maps[b]:
                            pairs.append(np.linalg.norm(maps[a][name] - maps[b][other]))
                if pairs and float(np.median(pairs)) < best_distance:
                    best_shift = shift
                    best_distance = float(np.median(pairs))
                    best_n = len(pairs)
            shifted_is_better = (
                best_shift != 0
                and best_n >= 1
                and best_distance <= disagree_px
                and best_distance + disagree_px * 0.5 < direct
            )
            direct_disagrees = len(common) >= 1 and direct > disagree_px
            if shifted_is_better or direct_disagrees:
                conflicts.append({
                    "crop_a": a,
                    "crop_b": b,
                    "direct_distance_px": round(direct, 2),
                    "best_shift": best_shift,
                    "best_shift_distance_px": round(best_distance, 2),
                    "common_centers": len(common),
                })
    return conflicts


def merge_predictions(
    predictions: list[dict],
    thresholds: np.ndarray,
    disagree_px: float,
) -> dict:
    """Согласовать и устойчиво усреднить предсказания нескольких кропов."""
    if not predictions:
        raise ValueError("нет предсказаний для объединения")
    names = list(predictions[0]["names"])
    if any(list(p["names"]) != names for p in predictions[1:]):
        raise ValueError("наборы каналов предсказаний различаются")

    conflicts = chain_conflicts(predictions, disagree_px)
    n = len(names)
    coords = np.zeros((n, 2), dtype=float)
    confidence = np.zeros(n, dtype=float)
    visible = np.zeros(n, dtype=bool)
    per_point = {}

    for i, name in enumerate(names):
        sources = [
            (p["coords"][i], float(p["confidence"][i]))
            for p in predictions if bool(p["visible"][i])
        ]
        if not sources:
            per_point[name] = {"n_sources": 0, "n_used": 0, "std_px": None}
            continue
        values = np.asarray([s[0] for s in sources], dtype=float)
        weights = np.asarray([max(s[1], 1e-6) for s in sources], dtype=float)
        median = np.median(values, axis=0)
        distances = np.linalg.norm(values - median, axis=1)
        keep = distances <= disagree_px
        if not keep.any():
            keep[np.argmin(distances)] = True
        values, weights = values[keep], weights[keep]
        coords[i] = np.average(values, axis=0, weights=weights)
        confidence[i] = float(np.average(weights))
        visible[i] = confidence[i] >= float(thresholds[i])
        std_px = float(np.sqrt(np.mean(np.sum((values - coords[i]) ** 2, axis=1))))
        per_point[name] = {
            "n_sources": len(sources),
            "n_used": int(keep.sum()),
            "std_px": round(std_px, 2),
        }

    return {
        "names": names,
        "coords": coords,
        "confidence": confidence,
        "visible": visible,
        "region": "spine" if any(visible) else "out_of_scope",
        "chain_hypotheses": None,
        "multicrop": {
            "n_crops": len(predictions),
            "per_point": per_point,
            "chain_conflicts": conflicts,
            "abstain": bool(conflicts),
        },
    }


def predict_multicrop(predictor, image: np.ndarray, config: dict | None = None) -> dict:
    """Запустить Predictor на полосах и объединить точки в полном кадре."""
    cfg = config or {}
    boxes = spine_crop_boxes(
        *image.shape,
        bands=cfg.get("bands", DEFAULT_BANDS),
        include_full_frame=bool(cfg.get("include_full_frame", True)),
    )
    edge_margin = int(cfg.get("edge_margin_px", 10))
    mm_per_px = float(predictor.cfg.get("data", {}).get("mm_per_px", 0.6))
    disagree_px = float(cfg.get("disagree_mm", 6.0)) / mm_per_px
    mapped = []
    for box in boxes:
        crop = np.ascontiguousarray(image[box.y0:box.y1, box.x0:box.x1])
        mapped.append(map_prediction(
            predictor.predict(crop), box, image.shape, edge_margin
        ))
    result = merge_predictions(mapped, predictor.thresholds, disagree_px)
    result["multicrop"]["boxes"] = [
        [b.x0, b.y0, b.x1, b.y1, b.full_frame] for b in boxes
    ]
    result["multicrop"]["sources"] = mapped
    return result


def load_semantic_config(config: dict | str | Path | None = None) -> dict:
    """Загрузить зафиксированные CropSpec; врачебный JSON в runtime не нужен."""
    if isinstance(config, dict):
        return config
    path = Path(config) if config is not None else DEFAULT_SEMANTIC_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("version") != "semantic-v2" or not cfg.get("specs"):
        raise ValueError(f"{path}: ожидается semantic-v2 со specs")
    return cfg


def anchor_quality(prediction: dict, config: dict) -> dict:
    """Проверить, можно ли использовать full-frame цепочку как нумерацию."""
    names = list(prediction["names"])
    idx = {n: i for i, n in enumerate(names)}
    reasons = []
    if not all(n in idx and prediction["visible"][idx[n]] for n in CENTERS):
        reasons.append("centers_missing")
        return {"reliable": False, "reasons": reasons, "step_px": None}
    centers = np.asarray([prediction["coords"][idx[n]] for n in CENTERS], float)
    step = np.mean(np.diff(centers, axis=0), axis=0)
    step_sq = float(step @ step)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    ratio = float(steps.max() / max(steps.min(), 1e-6))
    anchor_cfg = config.get("anchor", {})
    if step[1] <= 0 or step_sq < 1 or ratio > float(anchor_cfg.get("step_ratio_max", 1.6)):
        reasons.append("step_irregular")
    info = prediction.get("chain_hypotheses") or {}
    if not info.get("used"):
        reasons.append("chain_unavailable")
    gap = info.get("score_gap_top2")
    gap_min = float(anchor_cfg.get("chain_gap_min", 0.15))
    if gap_min <= 0:
        reasons.append("gap_uncalibrated")
    elif info.get("n_hypotheses", 1) > 1 and (gap is None or float(gap) < gap_min):
        reasons.append("chain_ambiguous")

    # Вся ровная цепочка может попасть на соседние позвонки. Проверяем
    # независимые анатомические ориентиры до запуска кропов.
    def along(name: str, origin: np.ndarray) -> float | None:
        j = idx.get(name)
        if j is None or not prediction["visible"][j] or step_sq < 1:
            return None
        return float((np.asarray(prediction["coords"][j]) - origin) @ step / step_sq)

    th12 = along("th12_bottom", centers[0])
    if th12 is not None and not -1.5 <= th12 <= -0.05:
        reasons.append("th12_inconsistent")
    disc = along("disc_th12_l1", centers[0])
    if disc is not None and not -1.5 <= disc <= 0.15:
        reasons.append("th12_inconsistent")
    top = along("th12_top", centers[0])
    if top is not None and th12 is not None and top >= th12 - 0.2:
        reasons.append("th12_inconsistent")
    pelvis = [along(name, centers[-1]) for name in ("iliac_right", "iliac_left")]
    if all(value is not None for value in pelvis):
        if not all(-0.7 <= value <= 1.8 for value in pelvis):
            reasons.append("pelvis_inconsistent")
        left = np.asarray(prediction["coords"][idx["iliac_left"]])
        right = np.asarray(prediction["coords"][idx["iliac_right"]])
        if not min(left[0], right[0]) < centers[-1, 0] < max(left[0], right[0]):
            reasons.append("pelvis_inconsistent")
    return {
        "reliable": not reasons,
        "reasons": sorted(set(reasons)),
        "step_px": float(np.median(steps)),
        "step_vector": np.mean(np.diff(centers, axis=0), axis=0).tolist(),
    }


def _soft_argmax_np(heat: np.ndarray, xy: np.ndarray, window: int) -> tuple[np.ndarray, float]:
    h, w = heat.shape
    x, y = np.asarray(xy, float)
    r = window // 2
    x0, x1 = max(0, int(round(x)) - r), min(w, int(round(x)) + r + 1)
    y0, y1 = max(0, int(round(y)) - r), min(h, int(round(y)) + r + 1)
    patch = np.clip(heat[y0:y1, x0:x1], 0, None)
    if not patch.size:
        return np.asarray(xy, float), 0.0
    py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
    confidence = float(patch[py, px])
    total = float(patch.sum())
    if total <= 1e-9:
        return np.asarray(xy, float), confidence
    ys = np.arange(y0, y1, dtype=float)[:, None]
    xs = np.arange(x0, x1, dtype=float)[None, :]
    return np.array([(patch * xs).sum() / total, (patch * ys).sum() / total]), confidence


def decode_allowed(
    heatmaps: np.ndarray,
    names: list[str],
    allowed: set[str],
    priors: dict[str, tuple[np.ndarray, float]],
    window: int = 11,
) -> dict:
    """Декодировать только allowed-каналы и только около якорного уровня."""
    coords = np.zeros((len(names), 2), dtype=float)
    confidence = np.zeros(len(names), dtype=float)
    in_prior = np.zeros(len(names), dtype=bool)
    idx = {name: i for i, name in enumerate(names)}
    for name in allowed:
        if name not in idx or name not in priors:
            continue
        col = idx[name]
        center, radius = priors[name]
        yy, xx = np.ogrid[:heatmaps.shape[1], :heatmaps.shape[2]]
        inside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
        masked = np.where(inside, heatmaps[col], 0.0)
        peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peak_xy = np.array([peak[1], peak[0]], dtype=float)
        coords[col], confidence[col] = _soft_argmax_np(masked, peak_xy, window)
        in_prior[col] = bool(masked[peak] > 0)
    return {"coords": coords, "confidence": confidence, "in_prior": in_prior}


def _semantic_boxes(shape: tuple[int, int], l1: np.ndarray, step: np.ndarray,
                    spec: dict, min_crop_px: int) -> list[CropBox]:
    h, w = shape
    boxes = []
    for key in ("p25", "p50", "p75"):
        top, bottom = spec["window_t"][key]
        ys = sorted((float((l1 + top * step)[1]), float((l1 + bottom * step)[1])))
        y0, y1 = max(0, int(np.floor(ys[0]))), min(h, int(np.ceil(ys[1])))
        if y1 - y0 < min_crop_px:
            pad = min_crop_px - (y1 - y0)
            y0 = max(0, y0 - pad // 2)
            y1 = min(h, y0 + min_crop_px)
            y0 = max(0, y1 - min_crop_px)
        box = CropBox(0, y0, w, y1)
        if box not in boxes:
            boxes.append(box)
    return boxes


def _candidate_score(decoded: dict, names: list[str], spec: dict,
                     box: CropBox, edge_margin: int) -> float:
    idx = {n: i for i, n in enumerate(names)}
    score = 0.0
    for group, weight in (("output", 1.0), ("context", .5)):
        for name in spec.get(group, []):
            i = idx[name]
            score += weight * np.log(max(float(decoded["confidence"][i]), 1e-8))
            y = decoded["coords"][i, 1]
            if group == "output" and (y < edge_margin or y >= box.shape[0] - edge_margin):
                score -= 4.0
    return float(score)


def predict_semantic(predictor, image: np.ndarray,
                     config: dict | str | Path | None = None) -> dict:
    """Full-frame задаёт уровни; семантические кропы только уточняют их."""
    cfg = load_semantic_config(config)
    anchor = predictor.predict(image)
    quality = anchor_quality(anchor, cfg)
    base_meta = {
        "mode": "semantic-v2", "anchor_quality": quality, "sources": [],
        "numbering_doubt": not quality["reliable"],
    }
    if not quality["reliable"]:
        anchor["multicrop"] = {**base_meta, "abstain": True,
                               "point_quality": "poor",
                               "quality_reasons": quality["reasons"]}
        return anchor

    names = list(anchor["names"])
    idx = {n: i for i, n in enumerate(names)}
    centers = np.asarray([anchor["coords"][idx[n]] for n in CENTERS], float)
    step = np.mean(np.diff(centers, axis=0), axis=0)
    step_len = float(np.linalg.norm(step))
    radius = float(cfg.get("refine_radius_step", .5)) * step_len
    edge_margin = int(cfg.get("edge_margin_px", 10))
    min_crop_px = int(cfg.get("min_crop_px", 96))
    window = int(predictor.cfg["heatmap"]["decode_window"])
    crop_thresholds = getattr(predictor, "crop_thresholds", predictor.thresholds)
    sources: dict[str, list[tuple[np.ndarray, float, str]]] = {
        n: [(np.asarray(anchor["coords"][i], float),
             float(anchor["confidence"][i]), "full")]
        for i, n in enumerate(names) if bool(anchor["visible"][i])
    }

    for spec_id, spec in cfg["specs"].items():
        candidates = []
        allowed = set(spec.get("output", [])) | set(spec.get("context", []))
        for box in _semantic_boxes(image.shape, centers[0], step, spec, min_crop_px):
            crop = np.ascontiguousarray(image[box.y0:box.y1, box.x0:box.x1])
            heat = predictor.heatmaps_tta(crop)
            priors = {
                name: (np.asarray(anchor["coords"][idx[name]], float)
                       - np.array([box.x0, box.y0]), radius)
                for name in allowed if name in idx
            }
            decoded = decode_allowed(heat, names, allowed, priors, window)
            decoded["box"] = box
            decoded["score"] = _candidate_score(decoded, names, spec, box, edge_margin)
            candidates.append(decoded)
        best = max(candidates, key=lambda x: x["score"])
        box = best["box"]
        source_meta = {"spec": spec_id, "box": [box.x0, box.y0, box.x1, box.y1],
                       "score": best["score"], "points": []}
        for name in spec.get("output", []):
            i = idx[name]
            xy = best["coords"][i] + np.array([box.x0, box.y0])
            y_local = best["coords"][i, 1]
            edge_ok = ((box.y0 == 0 or y_local >= edge_margin)
                       and (box.y1 == image.shape[0] or y_local < box.shape[0] - edge_margin))
            visible = (best["in_prior"][i] and edge_ok
                       and best["confidence"][i] >= crop_thresholds[i])
            if visible:
                sources.setdefault(name, []).append(
                    (xy, float(best["confidence"][i]), spec_id))
                source_meta["points"].append(name)
        base_meta["sources"].append(source_meta)

    coords = np.asarray(anchor["coords"], float).copy()
    conf = np.asarray(anchor["confidence"], float).copy()
    visible = np.asarray(anchor["visible"], bool).copy()
    per_point = {}
    disagree_px = float(cfg.get("disagree_mm", 6.0)) / float(
        predictor.cfg.get("data", {}).get("mm_per_px", .6))
    max_center_shift = float(cfg.get("center_max_shift_step", .25)) * step_len
    quality_reasons = []
    for name, vals in sources.items():
        i = idx[name]
        xy = np.asarray([v[0] for v in vals])
        weights = np.asarray([max(v[1], 1e-6) for v in vals])
        median = np.median(xy, axis=0)
        dist = np.linalg.norm(xy - median, axis=1)
        keep = dist <= disagree_px
        if not keep.any():
            keep[np.argmin(dist)] = True
        merged = np.average(xy[keep], axis=0, weights=weights[keep])
        rejected = False
        if name in CENTERS and np.linalg.norm(merged - anchor["coords"][i]) > max_center_shift:
            merged = np.asarray(anchor["coords"][i], float)
            rejected = True
            quality_reasons.append(f"{name}:center_refine_rejected")
        coords[i] = merged
        conf[i] = float(np.average(weights[keep]))
        visible[i] = conf[i] >= predictor.thresholds[i]
        spread = float(np.sqrt(np.mean(np.sum((xy[keep] - merged) ** 2, axis=1))))
        per_point[name] = {
            "n_sources": len(vals), "n_used": int(keep.sum()), "spread_px": spread,
            "source_ids": [vals[j][2] for j in np.flatnonzero(keep)],
            "center_refine_rejected": rejected,
        }
        if name in CENTERS and len(vals) < 2:
            quality_reasons.append(f"{name}:single_source")
        if name in CENTERS and spread > disagree_px:
            quality_reasons.append(f"{name}:spread")

    n_edges = sum(
        bool(visible[idx[f"l{i}_edge_left"]]) and bool(visible[idx[f"l{i}_edge_right"]])
        for i in range(1, 5)
    )
    if n_edges < 3:
        quality_reasons.append("edges_insufficient")
    point_quality = "ok" if not quality_reasons else "poor"
    return {
        **anchor, "coords": coords, "confidence": conf, "visible": visible,
        "multicrop": {
            **base_meta, "per_point": per_point, "point_quality": point_quality,
            "quality_reasons": quality_reasons, "abstain": point_quality != "ok",
        },
    }


def predict_oracle_semantic(predictor, image: np.ndarray, crop_items: list[dict],
                            config: dict | str | Path | None = None) -> dict:
    """Диагностический oracle: реальные box и included_points, нумерация от full-frame."""
    cfg = load_semantic_config(config)
    anchor = predictor.predict(image)
    quality = anchor_quality(anchor, cfg)
    if not quality["reliable"]:
        anchor["multicrop"] = {
            "mode": "oracle-semantic", "anchor_quality": quality,
            "numbering_doubt": True, "abstain": True, "sources": [],
        }
        return anchor
    names, mapped = list(anchor["names"]), []
    idx = {n: i for i, n in enumerate(names)}
    centers = np.asarray([anchor["coords"][idx[n]] for n in CENTERS], float)
    radius = float(cfg.get("refine_radius_step", .5)) * float(
        np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    window = int(predictor.cfg["heatmap"]["decode_window"])
    edge_margin = int(cfg.get("edge_margin_px", 10))
    for item in crop_items:
        raw = item["box"]
        box = CropBox(int(raw["x0"]), int(raw["y0"]), int(raw["x1"]), int(raw["y1"]))
        crop = np.ascontiguousarray(image[box.y0:box.y1, box.x0:box.x1])
        heat = predictor.heatmaps_tta(crop)
        allowed = set(item["included_points"])
        priors = {
            n: (np.asarray(anchor["coords"][idx[n]]) - np.array([box.x0, box.y0]), radius)
            for n in allowed if n in idx
        }
        decoded = decode_allowed(heat, names, allowed, priors, window)
        visible = np.array([
            n in allowed and decoded["in_prior"][i]
            and decoded["confidence"][i] >= predictor.crop_thresholds[i]
            for i, n in enumerate(names)
        ], bool)
        local = {"names": names, "coords": decoded["coords"],
                 "confidence": decoded["confidence"], "visible": visible,
                 "region": "spine"}
        mapped.append(map_prediction(local, box, image.shape, edge_margin))
    full = map_prediction(anchor, CropBox(0, 0, image.shape[1], image.shape[0], True),
                          image.shape, edge_margin)
    mm_per_px = float(predictor.cfg.get("data", {}).get("mm_per_px", .6))
    result = merge_predictions([full] + mapped, predictor.thresholds,
                               float(cfg.get("disagree_mm", 6.0)) / mm_per_px)
    result["chain_hypotheses"] = anchor.get("chain_hypotheses")
    result["multicrop"].update({
        "mode": "oracle-semantic", "anchor_quality": quality,
        "numbering_doubt": False, "sources": mapped,
    })
    return result
