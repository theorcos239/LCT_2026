"""Извлечь JSON-экспорт dxa-crops-v1, сохранённый как DOCX.

python3 tools/docx_to_crop_json.py '/path/to/export.docx' \
    --out Processing/dxa-crops.json

Документ читается как данные: макросы и внешние ссылки не исполняются.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract(path: Path) -> dict:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(WORD + "body")
    if body is None:
        raise ValueError("В DOCX нет тела документа")
    lines = ["".join(part.text or "" for part in paragraph.iter(WORD + "t"))
             for paragraph in body.findall(WORD + "p")]
    data = json.loads("\n".join(lines))
    if not isinstance(data, dict) or data.get("format") != "dxa-crops-v1":
        raise ValueError("Документ не содержит экспорт dxa-crops-v1")
    if data.get("schema_version") != 1 or not isinstance(data.get("crops"), list):
        raise ValueError("Неизвестная версия JSON-разметки")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    parser.add_argument("--out", type=Path, default=Path("Processing/dxa-crops.json"))
    args = parser.parse_args()
    data = extract(args.docx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    confirmed = sum(c.get("status") == "confirmed" for c in data["crops"])
    print(f"{args.out}: {len(data['crops'])} фрагментов, {confirmed} подтверждено")


if __name__ == "__main__":
    main()
