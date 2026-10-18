"""Собрать автономную страницу для разметки кропов из прежнего HTML.

Снимки и исходные точки встраиваются только в локальный файл Processing/.
Сборка не отправляет медицинские данные в сеть и не меняет исходную разметку.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_COMMIT = "b26b8d4"
OLD_HTML = "Processing/razmetka_all.html"
ANNOTATIONS = ROOT / "data/razmetka_aleksandra_2026-09-22_2241.json"
TEMPLATE = Path(__file__).with_name("crop_annotator_template.html")
OUTPUT = ROOT / "Processing/crop_annotator.html"


def source_html(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    local = ROOT / OLD_HTML
    if local.exists():
        return local.read_text(encoding="utf-8")
    return subprocess.check_output(
        ["git", "show", f"{OLD_COMMIT}:{OLD_HTML}"], cwd=ROOT, text=True
    )


def build(html: str, annotations: dict, template: str) -> str:
    match = re.search(
        r'<script id="dxa-data" type="application/json">(.*?)</script>', html, re.S
    )
    if match is None:
        raise ValueError("В исходном HTML не найден блок dxa-data")
    old = json.loads(match.group(1))
    images = old["images"]
    source = annotations["images"]
    missing = [im["id"] for im in images if im["id"] not in source]
    if missing:
        raise ValueError(f"Нет исходной разметки для {len(missing)} снимков")
    payload = {
        "schema": old["schema"],
        "images": images,
        "annotations": {im["id"]: source[im["id"]] for im in images},
        "source": {
            "html": OLD_HTML,
            "html_commit": OLD_COMMIT,
            "annotations": ANNOTATIONS.name,
            "annotation_saved_at": annotations.get("saved_at"),
        },
    }
    packed = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    packed = packed.replace("<", "\\u003c")
    return template.replace("__DXA_PAYLOAD__", packed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="локальный исходный razmetka_all.html")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    result = build(source_html(args.html), annotations, TEMPLATE.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result, encoding="utf-8")
    print(f"Готово: {args.out} ({len(result) / 1_000_000:.1f} МБ, только локально)")


if __name__ == "__main__":
    main()
