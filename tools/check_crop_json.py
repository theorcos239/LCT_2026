"""Проверить экспорт iPad по исходным id, границам и координатам точек.

Запуск: python3 tools/check_crop_json.py путь/к/dxa-crops-....json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_crop_annotator import ANNOTATIONS, source_html


def source_images() -> dict:
    import re

    html = source_html(None)
    match = re.search(r'<script id="dxa-data" type="application/json">(.*?)</script>', html, re.S)
    if match is None:
        raise ValueError("Не найден исходный список снимков")
    return {im["id"]: im for im in json.loads(match.group(1))["images"]}


def check(export: dict, images: dict, annotations: dict) -> tuple[int, int]:
    if export.get("format") != "dxa-crops-v1" or export.get("schema_version") != 1:
        raise ValueError("Ожидается формат dxa-crops-v1, schema_version 1")
    seen = set()
    confirmed = 0
    for n, c in enumerate(export.get("crops", []), 1):
        uid = c.get("image_id")
        if uid not in images or uid not in annotations:
            raise ValueError(f"Кроп {n}: неизвестный image_id {uid}")
        if c.get("id") in seen:
            raise ValueError(f"Кроп {n}: повторяется id")
        seen.add(c.get("id"))
        im = images[uid]
        if c.get("region") != im["region"] or c.get("image_size") != {"w": im["w"], "h": im["h"]}:
            raise ValueError(f"Кроп {n}: область или размер снимка не совпадают")
        b = c.get("box", {})
        if not all(isinstance(b.get(k), int) and not isinstance(b.get(k), bool) for k in ("x0", "y0", "x1", "y1")):
            raise ValueError(f"Кроп {n}: границы должны быть целыми числами")
        if not (0 <= b["x0"] < b["x1"] <= im["w"] and 0 <= b["y0"] < b["y1"] <= im["h"]):
            raise ValueError(f"Кроп {n}: рамка вне снимка")
        names = c.get("included_points")
        if not isinstance(names, list) or len(names) != len(set(names)):
            raise ValueError(f"Кроп {n}: неверный список точек")
        if c.get("status") == "confirmed":
            confirmed += 1
            if not names:
                raise ValueError(f"Кроп {n}: подтверждённый фрагмент без точек")
        elif c.get("status") != "draft":
            raise ValueError(f"Кроп {n}: неизвестный статус")
        points = annotations[uid]["points"]
        local = c.get("points_crop", {})
        if set(local) != set(names):
            raise ValueError(f"Кроп {n}: points_crop не соответствует included_points")
        for name in names:
            if name not in points:
                raise ValueError(f"Кроп {n}: точка {name} отсутствует в исходной разметке")
            p = points[name]
            if not (b["x0"] <= p["x"] < b["x1"] and b["y0"] <= p["y"] < b["y1"]):
                raise ValueError(f"Кроп {n}: точка {name} вне рамки")
            if abs(local[name]["x"] - (p["x"] - b["x0"])) > .0011 or abs(local[name]["y"] - (p["y"] - b["y0"])) > .0011:
                raise ValueError(f"Кроп {n}: неверные локальные координаты {name}")
    return len(export.get("crops", [])), confirmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    args = parser.parse_args()
    count, confirmed = check(
        json.loads(args.json.read_text(encoding="utf-8")),
        source_images(),
        json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["images"],
    )
    print(f"Проверено: {count} фрагментов, подтверждено {confirmed}")


if __name__ == "__main__":
    main()
