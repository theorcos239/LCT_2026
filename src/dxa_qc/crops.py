"""Врачебные кропы из dxa-crops-v1 как дополнительные обучающие примеры.

Исходный pack не меняется. Фрагменты наследуют study/side/id родительского
снимка и могут попадать только в его обучающую часть фолда. Правое бедро в
pack уже отражено в левое, поэтому рамка из JSON тоже отражается.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pack import ABSENT, LABELED, UNLABELED


def load(path: str | Path, source: dict) -> dict:
    export = json.loads(Path(path).read_text(encoding="utf-8"))
    if export.get("format") != "dxa-crops-v1" or export.get("schema_version") != 1:
        raise ValueError("Ожидается экспорт dxa-crops-v1, schema_version 1")
    if not isinstance(export.get("crops"), list):
        raise ValueError("В экспорте нет списка crops")

    by_id = {str(uid): i for i, uid in enumerate(source["ids"])}
    if len(by_id) != len(source["ids"]):
        raise ValueError("В pack повторяются image_id")
    names = list(source["names"])
    channel = {name: i for i, name in enumerate(names)}
    records, skipped, seen = [], 0, set()

    for ordinal, item in enumerate(export["crops"], 1):
        if item.get("status") != "confirmed":
            continue
        crop_id = item.get("id")
        if not isinstance(crop_id, str) or not crop_id or crop_id in seen:
            raise ValueError(f"Кроп {ordinal}: id отсутствует или повторяется")
        seen.add(crop_id)
        uid = str(item.get("image_id", ""))
        if uid not in by_id:
            # Некоторые из 252 снимков не вошли в pack (мало размеченных точек
            # или размер больше холста). Их кропы не становятся новой группой.
            skipped += 1
            continue
        parent = by_id[uid]
        pixels = source["images"][parent]
        h, w = pixels.shape
        if item.get("image_size") != {"w": w, "h": h}:
            raise ValueError(f"Кроп {ordinal}: размер исходного снимка не совпадает с pack")
        if item.get("region") != str(source["regions"][parent]):
            raise ValueError(f"Кроп {ordinal}: область не совпадает с pack")
        box = item.get("box", {})
        keys = ("x0", "y0", "x1", "y1")
        if not all(type(box.get(k)) is int for k in keys):
            raise ValueError(f"Кроп {ordinal}: границы должны быть целыми числами")
        x0, y0, x1, y1 = (box[k] for k in keys)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"Кроп {ordinal}: рамка вне исходного снимка")
        selected = item.get("included_points")
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
            raise ValueError(f"Кроп {ordinal}: нет подтверждённых точек или есть повторы")
        if any(name not in channel for name in selected):
            raise ValueError(f"Кроп {ordinal}: имя точки отсутствует в схеме pack")

        # JSON размечен на исходном изображении. В pack правое бедро зеркальное.
        if str(source["sides"][parent]) == "rhip":
            x0, x1 = w - x1, w - x0
        cropped = np.ascontiguousarray(pixels[y0:y1, x0:x1])
        coords = np.array(source["coords"][parent], copy=True)
        state = np.array(source["state"][parent], copy=True)
        if coords.shape != (len(names), 2) or state.shape != (len(names),):
            raise ValueError("Некорректная форма coords/state в исходном pack")

        inside = ((coords[:, 0] >= x0) & (coords[:, 0] < x1) &
                  (coords[:, 1] >= y0) & (coords[:, 1] < y1))
        for name in selected:
            i = channel[name]
            if state[i] != LABELED or not inside[i]:
                raise ValueError(f"Кроп {ordinal}: точка {name} не размечена или вне рамки")
            local = item.get("points_crop", {}).get(name)
            if not isinstance(local, dict):
                raise ValueError(f"Кроп {ordinal}: нет локальных координат {name}")
            # Экспорт округляет до 0.001 px. При отражении исходный x равен
            # w-1-x_pack, а локальная координата считается от исходного x0.
            original_x = w - 1 - coords[i, 0] if str(source["sides"][parent]) == "rhip" else coords[i, 0]
            if (abs(float(local["x"]) - (original_x - box["x0"])) > .002 or
                    abs(float(local["y"]) - (coords[i, 1] - box["y0"])) > .002):
                raise ValueError(f"Кроп {ordinal}: точка {name} не совпадает с исходной разметкой")

        # Отрезанные точки точно отсутствуют. Если прежняя точка геометрически
        # внутри, но врач её не выбрала, это неоднозначность, а не доказанное
        # отсутствие: не учим модель подавлять правильный пик.
        old_labeled = state == LABELED
        state[old_labeled & ~inside] = ABSENT
        state[old_labeled & inside] = UNLABELED
        for name in selected:
            state[channel[name]] = LABELED
        coords[:, 0] -= x0
        coords[:, 1] -= y0
        records.append((crop_id, uid, parent, cropped, coords, state))

    if not records:
        raise ValueError(f"{path}: нет подтверждённых фрагментов из снимков исходного pack")
    return {
        "names": names,
        "images": [r[3] for r in records],
        "coords": np.stack([r[4] for r in records]) if records else np.empty((0, len(names), 2), np.float32),
        "state": np.stack([r[5] for r in records]) if records else np.empty((0, len(names)), np.uint8),
        "ids": np.array([r[0] for r in records], dtype=str),
        "parent_ids": np.array([r[1] for r in records], dtype=str),
        "studies": np.array([source["studies"][r[2]] for r in records], dtype=str),
        "regions": np.array([source["regions"][r[2]] for r in records], dtype=str),
        "sides": np.array([source["sides"][r[2]] for r in records], dtype=str),
        "skipped_not_in_pack": skipped,
    }
