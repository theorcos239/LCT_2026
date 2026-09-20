"""Конфиг в одном YAML; константы в коде не живут.

Переопределение с командной строки: --set train.batch_size=8 model.pretrained=false
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def save(cfg: dict, path: str | Path) -> None:
    """Копия конфига рядом с весами: без неё чекпоинт невоспроизводим."""
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                          encoding="utf-8")


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
