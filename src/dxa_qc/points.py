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

# Пары, меняющиеся местами при отражении по горизонтали (только позвоночник).
FLIP_PAIRS = [("iliac_right", "iliac_left")] + [
    (f"l{i}_edge_right", f"l{i}_edge_left") for i in range(1, 5)
]


def channels(spine_edges: bool = True) -> list[str]:
    """Порядок каналов модели: позвоночник, затем бедро."""
    spine = SPINE_CORE + (SPINE_EDGES if spine_edges else [])
    return spine + HIP


def region_slices(spine_edges: bool = True) -> dict[str, slice]:
    n_spine = len(SPINE_CORE) + (len(SPINE_EDGES) if spine_edges else 0)
    return {"spine": slice(0, n_spine), "hip": slice(n_spine, n_spine + len(HIP))}


def flip_permutation(names: list[str]) -> list[int]:
    """Индексы каналов после отражения по горизонтали (право/лево меняются)."""
    swap = {a: b for a, b in FLIP_PAIRS} | {b: a for a, b in FLIP_PAIRS}
    idx = {n: i for i, n in enumerate(names)}
    return [idx[swap.get(n, n)] for n in names]


def mirror_to_left(pixels, points: dict, side: str):
    """Правое бедро отражаем в левое: точки бедра стороны не содержат."""
    if side != "rhip":
        return pixels, points
    w = pixels.shape[1]
    flipped = pixels[:, ::-1].copy()
    return flipped, {k: (w - 1 - x, y) for k, (x, y) in points.items()}
