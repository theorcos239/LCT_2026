"""Чтение DICOM, дедупликация, определение области и загрузка разметки.

Подробности об устройстве датасета — в DATASET.md.
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

logging.getLogger("pydicom").setLevel(logging.ERROR)  # невалидные UID 1.2.643... засоряют лог

REGIONS = ("spine", "rhip", "lhip")
MM_PER_PX = 0.6  # PixelSpacing в файлах нет; оценка по ExposedArea, см. DATASET.md

LABEL_COLUMNS = [
    "n", "study",
    "spine_position", "spine_axis", "spine_artifacts",
    "rhip_rotation", "rhip_roi",
    "lhip_rotation", "lhip_roi",
    "spine_bad", "rhip_bad", "lhip_bad", "comment",
]
REGION_CRITERIA = {
    "spine": ["spine_position", "spine_axis", "spine_artifacts"],
    "rhip": ["rhip_rotation", "rhip_roi"],
    "lhip": ["lhip_rotation", "lhip_roi"],
}


@dataclass
class DicomImage:
    path: Path
    pixels: np.ndarray
    study_uid: str
    sop_uid: str
    instance: int


def read_dicom(path: str | Path) -> DicomImage:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = pydicom.dcmread(path)
        px = ds.pixel_array
    return DicomImage(
        path=Path(path),
        pixels=px,
        study_uid=str(ds.get("StudyInstanceUID", "")),
        sop_uid=str(ds.get("SOPInstanceUID", "")),
        instance=int(ds.get("InstanceNumber", -1) or -1),
    )


def pixel_hash(px: np.ndarray) -> str:
    return hashlib.md5(px.tobytes()).hexdigest()


def guess_region(px: np.ndarray) -> str:
    """Эвристика: ширина 300 → позвоночник; для бедра сторона по яркости верхней трети.

    Ориентация AP (PatientOrientation = L,F): головка левого бедра — слева на снимке.
    """
    h, w = px.shape
    if w >= 300:
        return "spine"
    top = px[: h // 3].astype(np.float32)
    return "lhip" if top[:, : w // 2].mean() > top[:, w // 2 :].mean() else "rhip"


def scan_studies(root: str | Path) -> pd.DataFrame:
    """Все .dcm под root: одна строка на файл. study = имя папки первого уровня.

    Нечитаемые файлы попадают в таблицу с error != None, а не роняют обход.
    """
    root = Path(root)
    rows = []
    for f in sorted(root.rglob("*.dcm")):
        rec = {"path": str(f), "study": f.relative_to(root).parts[0], "error": None}
        try:
            img = read_dicom(f)
            rec.update(
                study_uid=img.study_uid, sop_uid=img.sop_uid, instance=img.instance,
                rows=img.pixels.shape[0], cols=img.pixels.shape[1],
                px_hash=pixel_hash(img.pixels), region=guess_region(img.pixels),
            )
        except Exception as e:  # noqa: BLE001 — любая ошибка чтения фиксируется в отчёте
            rec["error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
    return pd.DataFrame(rows)


def unique_images(files: pd.DataFrame) -> pd.DataFrame:
    """Дедупликация по хешу пикселей внутри исследования; оставляем минимальный InstanceNumber."""
    ok = files[files.error.isna()]
    n_copies = ok.groupby(["study", "px_hash"]).size().rename("n_copies")
    return (ok.sort_values(["study", "instance"])
              .drop_duplicates(["study", "px_hash"])
              .join(n_copies, on=["study", "px_hash"])
              .reset_index(drop=True))


def load_labels(xlsx: str | Path) -> pd.DataFrame:
    """разметка.xlsx: двухстрочный заголовок, колонки 13+ — сводка эксперта (не нужна)."""
    lab = pd.read_excel(xlsx, header=None).iloc[2:, : len(LABEL_COLUMNS)].copy()
    lab.columns = LABEL_COLUMNS
    lab = lab.drop(columns="n").reset_index(drop=True)
    crit = [c for c in LABEL_COLUMNS if c not in ("n", "study", "comment")]
    lab[crit] = lab[crit].apply(pd.to_numeric, errors="coerce")
    lab["study"] = lab.study.astype(str)
    return lab
