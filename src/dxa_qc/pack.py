"""Сборка обучающего набора: разметка + DICOM → один npz.

Связь id разметки со снимком восстанавливается сопоставлением пикселей: в HTML-сборке
для разметчика лежат те же изображения, что в DICOM, побайтово.

Состояние канала (почему не просто «есть точка / нет точки»):
    UNLABELED — точка не размечена (например, второй проход ещё не сделан) → вне потерь;
    ABSENT    — структуры нет в кадре, размечено явно → нулевая карта;
    LABELED   — есть координата;
    OTHER     — канал чужой области → нулевая карта с весом мягкой маски.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import points as P
from .points import HIP, SPINE_CORE
from .data import guess_region, read_dicom, scan_studies, unique_images

UNLABELED, ABSENT, LABELED, OTHER = 0, 1, 2, 3
SKIP_FLAGS = {"out_of_scope", "unreadable"}
# Флаг review сам по себе снимок не бракует: им помечают и полностью размеченные.
# Негодные отсеиваются по содержанию — по доле реально поставленных точек.
MIN_PLACED_FRACTION = 0.5
# p1 — закончен первый проход (13 основных точек), боковые края тел ещё не размечены:
# они остаются UNLABELED и в потерях не участвуют.
STATUSES = ("done", "p1")
_HTML_RE = re.compile(
    r'\{"id":"([0-9a-f]+)","w":(\d+),"h":(\d+),"region":"(\w+)","src":"data:image/png;base64,([^"]+)"'
)


def html_images(path: str | Path) -> dict[str, np.ndarray]:
    """id → пиксели из HTML-сборки для разметчика."""
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    return {
        m[0]: np.array(Image.open(io.BytesIO(base64.b64decode(m[4]))).convert("L"))
        for m in _HTML_RE.findall(html)
    }


def match_ids(html_px: dict[str, np.ndarray], images) -> dict[str, dict]:
    """Сопоставление id ↔ DICOM по точному совпадению пикселей."""
    by_shape: dict[tuple, list] = {}
    for row in images.itertuples():
        px = read_dicom(row.path).pixels
        by_shape.setdefault(px.shape, []).append((px, row))
    out = {}
    for uid, a in html_px.items():
        for px, row in by_shape.get(a.shape, []):
            if np.array_equal(px, a):
                out[uid] = {"study": row.study, "path": row.path, "pixels": px}
                break
        else:
            raise ValueError(f"снимок {uid} не найден среди DICOM")
    return out


def latest_annotations(annotations: list[str | Path]) -> dict[str, dict]:
    """Разметки из нескольких файлов; при повторе снимка берётся более поздняя версия."""
    out: dict[str, dict] = {}
    for path in annotations:
        ann = json.loads(Path(path).read_text(encoding="utf-8"))
        for uid, a in ann["images"].items():
            prev = out.get(uid)
            if prev is None or str(a.get("updated_at", "")) >= str(prev.get("updated_at", "")):
                out[uid] = {**a, "source": Path(path).name}
    return out


def build(annotations: list[str | Path], html: str | Path, studies_root: str | Path,
          spine_edges: bool = True, canvas: tuple[int, int] = (352, 320),
          statuses: tuple[str, ...] = STATUSES) -> dict:
    names = P.channels(spine_edges)
    idx = {n: i for i, n in enumerate(names)}
    regions = P.region_slices(spine_edges)
    matched = match_ids(html_images(html), unique_images(scan_studies(studies_root)))

    records, skipped = [], []
    for uid, a in latest_annotations(annotations).items():
        src = matched[uid]
        h, w = src["pixels"].shape
        side = guess_region(src["pixels"])                          # spine / rhip / lhip
        region = "spine" if side == "spine" else "hip"
        own = regions[region]
        core = [n for n in names[own]]

        if a["status"] not in statuses:
            skipped.append((uid, f"статус {a['status']}"))
            continue
        if SKIP_FLAGS & set(a["flags"]):
            skipped.append((uid, "флаг " + ",".join(sorted(SKIP_FLAGS & set(a["flags"])))))
            continue
        if h > canvas[0] or w > canvas[1]:
            # выбросы вне холста в обучение не идут, они уходят в смоук-тест
            # «x», а не «×»: причина печатается, а «×» нет в cp1251 — в консоли
            # Windows print падал уже после записи npz, и процесс выходил с 1.
            skipped.append((uid, f"размер {h}x{w} больше холста {canvas[0]}x{canvas[1]}"))
            continue
        # Долю считаем от основных точек: боковые края тел — отдельный проход,
        # их отсутствие не повод браковать снимок.
        essential = SPINE_CORE if region == "spine" else HIP
        placed = sum(1 for n in essential if n in a["points"])
        if placed < MIN_PLACED_FRACTION * len(essential):
            # Почти всё помечено отсутствующим: обычно разметчик не смог разобрать
            # анатомию, а не структур нет в кадре. Учить этому молчанию нельзя.
            skipped.append((uid, f"поставлено меньше половины основных точек ({placed}/{len(essential)})"))
            continue

        pts = {k: (v["x"], v["y"]) for k, v in a["points"].items()}
        px, pts = P.mirror_to_left(src["pixels"], pts, side)

        state = np.full(len(names), OTHER, dtype=np.uint8)
        coords = np.zeros((len(names), 2), dtype=np.float32)
        state[own] = UNLABELED
        for name in core:
            if name in pts:
                state[idx[name]] = LABELED
                coords[idx[name]] = pts[name]
            elif name in a["absent"]:
                state[idx[name]] = ABSENT
        records.append(dict(id=uid, study=src["study"], region=region, side=side,
                            image=px, coords=coords, state=state,
                            flags=",".join(a["flags"])))
    return {"names": names, "records": records, "skipped": skipped}


def save(pack: dict, path: str | Path) -> None:
    r = pack["records"]
    np.savez_compressed(
        path,
        names=np.array(pack["names"]),
        images=np.array([x["image"] for x in r], dtype=object),
        coords=np.stack([x["coords"] for x in r]),
        state=np.stack([x["state"] for x in r]),
        meta=np.array([[x["id"], x["study"], x["region"], x["side"], x["flags"]] for x in r]),
    )


def load(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=True)
    meta = z["meta"]
    return dict(names=list(z["names"]), images=list(z["images"]), coords=z["coords"],
                state=z["state"], ids=meta[:, 0], studies=meta[:, 1],
                regions=meta[:, 2], sides=meta[:, 3], flags=meta[:, 4])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--html", required=True, help="сборка со всеми снимками (razmetka_all.html)")
    p.add_argument("--studies", required=True, help="папка Исследования")
    p.add_argument("--out", required=True)
    p.add_argument("--no-spine-edges", action="store_true")
    p.add_argument("--canvas", nargs=2, type=int, default=[352, 320])
    p.add_argument("--statuses", nargs="+", default=list(STATUSES))
    a = p.parse_args()

    pack = build(a.annotations, a.html, a.studies,
                 spine_edges=not a.no_spine_edges, canvas=tuple(a.canvas),
                 statuses=tuple(a.statuses))
    save(pack, a.out)
    n = len(pack["records"])
    reg = [r["region"] for r in pack["records"]]
    print(f"{a.out}: {n} снимков (позвоночник {reg.count('spine')}, бедро {reg.count('hip')}), "
          f"каналов {len(pack['names'])}, пропущено {len(pack['skipped'])}")
    import collections
    for why, n in collections.Counter(w for _, w in pack["skipped"]).most_common():
        print(f"  пропущено {n:3d}: {why}")


if __name__ == "__main__":
    main()
