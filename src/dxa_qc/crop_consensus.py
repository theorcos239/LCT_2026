"""Детерминированная нарезка позвоночника и согласование независимых пиков.

Маска каналов каждого фрагмента получена из врачебной разметки. CNN не
решает, какой позвонок физически должен находиться на фрагменте.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multicrop import CENTERS, CropBox, load_semantic_config, map_prediction


# Порог отсева кандидата у верифицированного соседа — треть среднего
# расстояния между центрами этой пары во врачебной разметке
# (data/keypoints.npz, 92 снимка позвоночника, 0.6 мм/px): L1–L2 31.184 мм,
# L2–L3 32.463 мм, L3–L4 33.846 мм. Дублируется в configs/spine_multicrop.yaml.
NEIGHBOR_EXCLUSION_MM = {
    ("l1_center", "l2_center"): 10.39,
    ("l2_center", "l3_center"): 10.82,
    ("l3_center", "l4_center"): 11.28,
}
_PAIR_KEYS = {
    "l1_l2": ("l1_center", "l2_center"),
    "l2_l3": ("l2_center", "l3_center"),
    "l3_l4": ("l3_center", "l4_center"),
}


def neighbor_exclusion_from_config(config: dict | None) -> dict:
    """Прочитать парные пороги из YAML; иначе — значения по умолчанию."""
    raw = (config or {}).get("neighbor_exclusion_mm") or {}
    out = dict(NEIGHBOR_EXCLUSION_MM)
    for key, pair in _PAIR_KEYS.items():
        if key in raw:
            out[pair] = float(raw[key])
    return out


@dataclass(frozen=True)
class Candidate:
    xy: np.ndarray
    count: int = 1
    confidence: float = 1.0
    sources: tuple[str, ...] = ()


def fixed_boxes(shape: tuple[int, int], config: dict) -> dict[str, CropBox]:
    """Четыре окна в долях *исходного* изображения, без координат модели."""
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("пустое изображение")
    result = {}
    for name, spec in config["specs"].items():
        lo, hi = map(float, spec["fixed_band"])
        if not 0 <= lo < hi <= 1:
            raise ValueError(f"{name}: неверная fixed_band")
        y0, y1 = round(lo * h), round(hi * h)
        result[name] = CropBox(0, int(y0), w, max(int(y0) + 1, int(y1)))
    return result


def adaptive_boxes(shape: tuple[int, int], config: dict,
                   prediction: dict) -> dict[str, CropBox]:
    """Одно медианное врачебное окно на тип по цепочке полного кадра."""
    names = list(prediction["names"])
    idx = {n: i for i, n in enumerate(names)}
    if any(n not in idx or not prediction["visible"][idx[n]] for n in CENTERS):
        return fixed_boxes(shape, config)
    centers = np.asarray([prediction["coords"][idx[n]] for n in CENTERS], float)
    step = (centers[-1] - centers[0]) / 3
    if step[1] <= 0:
        return fixed_boxes(shape, config)
    h, w = shape
    min_size = int(config.get("min_crop_px", 96))
    result = {}
    for name, spec in config["specs"].items():
        lo, hi = spec["window_t"]["p50"]
        y0 = max(0, int(np.floor(centers[0, 1] + lo * step[1])))
        y1 = min(h, int(np.ceil(centers[0, 1] + hi * step[1])))
        if y1 - y0 < min_size:
            mid = (y0 + y1) // 2
            y0 = max(0, min(h - min_size, mid - min_size // 2))
            y1 = min(h, y0 + min_size)
        result[name] = CropBox(0, y0, w, y1)
    return result


def merge_close(candidates: list[Candidate], radius_px: float) -> list[Candidate]:
    """Сливать первую подходящую пару и повторять до неподвижной точки.

Вес центроида равен числу *исходных* наблюдений, а не уверенности CNN.
Порядок стабилен: первый источник — полный кадр, затем кропы сверху вниз.
"""
    if radius_px <= 0:
        raise ValueError("радиус должен быть положительным")
    work = list(candidates)
    while True:
        pair = next(((i, j) for i in range(len(work)) for j in range(i + 1, len(work))
                     if np.linalg.norm(work[i].xy - work[j].xy) <= radius_px), None)
        if pair is None:
            return work
        i, j = pair
        a, b = work[i], work[j]
        count = a.count + b.count
        merged = Candidate((a.xy * a.count + b.xy * b.count) / count, count,
                           (a.confidence * a.count + b.confidence * b.count) / count,
                           a.sources + b.sources)
        work[i] = merged
        work.pop(j)


def resolve_centers(raw: dict[str, list[Candidate]], mm_per_px: float,
                    merge_mm: float = 10.0,
                    exclusion_mm: dict | None = None) -> dict:
    """Соседство задаёт анатомическая цепочка L1—L2—L3—L4.

Кандидат неверифицированного позвонка удаляется, если он ближе порога пары
к верифицированному соседу или лежит не с той стороны от него по вертикали
(ось y направлена к ногам: верхний сосед выше, нижний — ниже). Кандидаты
удаляются по одному; после каждого удаления позвонок с единственным
кандидатом сразу верифицируется и становится опорой для своих соседей.
Пропуск интерполируется только между двумя верифицированными соседями,
причём интерполированная точка не создаёт каскад фиктивных точек.
"""
    if mm_per_px <= 0:
        raise ValueError("mm_per_px должен быть положительным")
    limits = {**NEIGHBOR_EXCLUSION_MM, **(exclusion_mm or {})}
    clusters = {name: merge_close(raw.get(name, []), merge_mm / mm_per_px)
                for name in CENTERS}
    verified = {name: vals[0] for name, vals in clusters.items() if len(vals) == 1}
    removed = {name: 0 for name in CENTERS}
    removed_side = {name: 0 for name in CENTERS}

    def rejection(k: int, cand: Candidate) -> str | None:
        for j in (k - 1, k + 1):
            if not 0 <= j < len(CENTERS) or CENTERS[j] not in verified:
                continue
            other = verified[CENTERS[j]].xy
            pair = (CENTERS[min(j, k)], CENTERS[max(j, k)])
            if np.linalg.norm(cand.xy - other) * mm_per_px <= limits[pair]:
                return "near"
            if (cand.xy[1] - other[1]) * (k - j) <= 0:
                return "side"
        return None

    def remove_one() -> bool:
        for k, name in enumerate(CENTERS):
            if name in verified:
                continue
            for i, cand in enumerate(clusters[name]):
                reason = rejection(k, cand)
                if reason is None:
                    continue
                clusters[name] = clusters[name][:i] + clusters[name][i + 1:]
                (removed if reason == "near" else removed_side)[name] += 1
                if len(clusters[name]) == 1 and rejection(k, clusters[name][0]) is None:
                    verified[name] = clusters[name][0]
                return True
        return False

    while remove_one():
        pass
    interpolated = {}
    for k in (1, 2):
        name = CENTERS[k]
        left, right = CENTERS[k - 1], CENTERS[k + 1]
        if not clusters[name] and left in verified and right in verified:
            interpolated[name] = Candidate((verified[left].xy + verified[right].xy) / 2,
                                           0, min(verified[left].confidence,
                                                  verified[right].confidence),
                                           ("interpolated",))
    return {"clusters": clusters, "verified": verified,
            "interpolated": interpolated, "removed_near_neighbor": removed,
            "removed_wrong_side": removed_side}


def predict_consensus(predictor, image: np.ndarray, config: dict | None = None,
                      box_mode: str = "fixed") -> dict:
    """Полный кадр + 4 фиксированных кропа -> координаты исходного кадра."""
    cfg = load_semantic_config(config)
    mm_per_px = float(predictor.cfg["data"]["mm_per_px"])
    # Полный кадр использует штатное согласование последовательности уровней;
    # на частичном кадре оно недопустимо, потому что соседних уровней там нет.
    anchor = predictor.predict(image)
    names = list(anchor["names"])
    idx = {name: i for i, name in enumerate(names)}
    edge_margin = int(cfg.get("edge_margin_px", 10))
    source_points: dict[str, list[Candidate]] = {n: [] for n in names}
    for i, name in enumerate(names):
        if name in CENTERS or bool(anchor["visible"][i]):
            source_points[name].append(Candidate(np.asarray(anchor["coords"][i], float),
                                                  1, float(anchor["confidence"][i]), ("full",)))
    if box_mode not in ("fixed", "adaptive"):
        raise ValueError("box_mode должен быть fixed или adaptive")
    boxes = (fixed_boxes(image.shape, cfg) if box_mode == "fixed" else
             adaptive_boxes(image.shape, cfg, anchor))
    source_meta = []
    for spec_name, box in boxes.items():
        spec = cfg["specs"][spec_name]
        # Объединяем только центры позвонков: OOF показал, что независимые
        # пики краёв и дисков ухудшают геометрию. Остальные каналы остаются
        # из полного кадра; их присутствие всё равно сохранено в JSON аудита.
        allowed = set(spec["expected_centers"])
        crop = np.ascontiguousarray(image[box.y0:box.y1, box.x0:box.x1])
        local = predictor.predict(crop, chain=False)
        # Выбор видимости на кропе делается по его калиброванным порогам,
        # а не по порогам целого снимка.
        local["visible"] = np.ones(len(names), dtype=bool)
        mapped = map_prediction(local, box, image.shape, edge_margin)
        selected = []
        for name in allowed:
            i = idx.get(name)
            if i is None:
                continue
            source_points[name].append(Candidate(np.asarray(mapped["coords"][i], float),
                                                  1, float(mapped["confidence"][i]),
                                                  (spec_name,)))
            selected.append(name)
        source_meta.append({"spec": spec_name,
                            "box": [box.x0, box.y0, box.x1, box.y1],
                            "present_centers": list(spec["expected_centers"]),
                            "absent_centers": [n for n in CENTERS
                                               if n not in spec["expected_centers"]],
                            "points": sorted(selected)})

    resolved = resolve_centers(
        source_points, mm_per_px,
        exclusion_mm=neighbor_exclusion_from_config(cfg),
    )
    coords = np.asarray(anchor["coords"], float).copy()
    confidence = np.asarray(anchor["confidence"], float).copy()
    visible = np.asarray(anchor["visible"], bool).copy()
    per_point = {}
    for name in names:
        i = idx[name]
        vals = source_points[name]
        if name in CENTERS:
            winner = resolved["verified"].get(name) or resolved["interpolated"].get(name)
            if winner is None:
                # Координата здесь служит только совместимым заполнителем;
                # visible=False и confidence=0 запрещают её использовать.
                # Сохраняем full-frame значение для диагностической метрики.
                confidence[i] = 0
                visible[i] = False
            else:
                coords[i] = winner.xy
                confidence[i] = winner.confidence
                visible[i] = True
            per_point[name] = {
                "n_sources": len(vals), "n_clusters": len(resolved["clusters"][name]),
                "status": ("verified" if name in resolved["verified"] else
                           "interpolated" if name in resolved["interpolated"] else "missing"),
                "n_merged": winner.count if winner is not None else 0,
                "removed_near_neighbor": resolved["removed_near_neighbor"][name],
                "removed_wrong_side": resolved["removed_wrong_side"][name],
                "candidates_global": [{"xy": c.xy.tolist(), "sources": list(c.sources),
                                       "confidence": c.confidence} for c in vals],
                "clusters_global": [{"xy": c.xy.tolist(), "n_merged": c.count,
                                      "sources": list(c.sources)}
                                     for c in resolved["clusters"][name]],
            }
        else:
            per_point[name] = {"n_sources": len(vals), "status": "full_frame"}
    unresolved = [n for n in CENTERS if n not in resolved["verified"]]
    return {**anchor, "coords": coords, "confidence": confidence, "visible": visible,
            "region": "spine", "multicrop": {
            "mode": f"semantic-consensus-{box_mode}", "sources": source_meta,
                "per_point": per_point, "unresolved_centers": unresolved,
                "abstain": bool(unresolved),
                "numbering_doubt": bool(unresolved),
            }}
