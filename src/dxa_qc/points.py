"""Схемы ключевых точек: состав каналов, отражение, приведение бедра к левому."""
from __future__ import annotations

SPINE_CORE = [
    "th12_top", "th12_bottom",
    "l1_center", "l2_center", "l3_center", "l4_center",
    "iliac_right", "iliac_left",
    "disc_th12_l1", "disc_l1_l2", "disc_l2_l3", "disc_l3_l4", "disc_l4_l5",
]
SPINE_EDGES = [
    "l1_edge_right", "l1_edge_left", "l2_edge_right", "l2_edge_left",
    "l3_edge_right", "l3_edge_left", "l4_edge_right", "l4_edge_left",
]
HIP = [
    "trochanter_major", "trochanter_lateral", "trochanter_minor",
    "femoral_head_center", "neck_center", "ischium_bottom",
    "shaft_lateral", "shaft_medial",
]

# Наборы каналов. minimal — только те точки, что реально входят в критерии
# (ось, укладка, ROI, ротация); остальные при обучении лишь делят ёмкость.
SETS: dict[str, dict[str, list[str]]] = {
    "full": {"spine": SPINE_CORE + SPINE_EDGES, "hip": HIP},
    "core": {"spine": SPINE_CORE, "hip": HIP},
    "minimal": {
        "spine": ["th12_top", "th12_bottom", "l1_center", "l2_center", "l3_center",
                  "l4_center", "iliac_right", "iliac_left"],
        "hip": ["trochanter_major", "trochanter_lateral", "trochanter_minor",
                "ischium_bottom", "shaft_lateral", "shaft_medial"],
    },
}

# Пары, меняющиеся местами при отражении по горизонтали (только позвоночник).
FLIP_PAIRS = [("iliac_right", "iliac_left")] + [
    (f"l{i}_edge_right", f"l{i}_edge_left") for i in range(1, 5)
]


def resolve(points: str | None = None, spine_edges: bool | None = None) -> str:
    """Имя набора из конфига. spine_edges — прежний флаг, поддерживается для старых весов."""
    if points:
        if points not in SETS:
            raise ValueError(f"неизвестный набор точек {points!r}, есть: {sorted(SETS)}")
        return points
    return "full" if spine_edges in (None, True) else "core"


def channels(points: str | bool = "full", spine_edges: bool | None = None) -> list[str]:
    """Порядок каналов модели: позвоночник, затем бедро."""
    if isinstance(points, bool):                    # старый вызов channels(spine_edges)
        points, spine_edges = None, points
    s = SETS[resolve(points, spine_edges)]
    return s["spine"] + s["hip"]


def region_slices(points: str | bool = "full", spine_edges: bool | None = None) -> dict[str, slice]:
    if isinstance(points, bool):
        points, spine_edges = None, points
    s = SETS[resolve(points, spine_edges)]
    n = len(s["spine"])
    return {"spine": slice(0, n), "hip": slice(n, n + len(s["hip"]))}


def from_config(cfg: dict) -> tuple[list[str], dict[str, slice]]:
    """Каналы и границы областей по секции data конфига или чекпоинта."""
    d = cfg.get("data", cfg)
    name = resolve(d.get("points"), d.get("spine_edges"))
    return channels(name), region_slices(name)


def flip_permutation(names: list[str]) -> list[int]:
    """Индексы каналов после отражения по горизонтали (право/лево меняются)."""
    swap = {a: b for a, b in FLIP_PAIRS} | {b: a for a, b in FLIP_PAIRS}
    idx = {n: i for i, n in enumerate(names)}
    # пара, у которой второй точки нет в наборе, остаётся на месте
    return [idx[swap[n]] if swap.get(n) in idx else i for i, n in enumerate(names)]


def mirror_to_left(pixels, points: dict, side: str):
    """Правое бедро отражаем в левое: точки бедра стороны не содержат."""
    if side != "rhip":
        return pixels, points
    w = pixels.shape[1]
    flipped = pixels[:, ::-1].copy()
    return flipped, {k: (w - 1 - x, y) for k, (x, y) in points.items()}
