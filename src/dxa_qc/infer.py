"""Инференс: ансамбль фолдов и TTA, усреднение карт.

Маскирование живёт только в функции потерь, поэтому здесь считаются все каналы:
какая группа заговорила — та и область. Отдельный классификатор области не нужен.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import points as P
from .dataset import normalize, pad_to_multiple
from .heatmaps import decode
from .model import KeypointNet

# Каталог прогона по умолчанию — тот же, что train.out_dir в configs/keypoints.yaml.
RUN_DIR = Path(__file__).resolve().parents[2] / "runs" / "keypoints"


def _rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    if degrees == 0:
        return x
    a = torch.tensor(np.deg2rad(degrees), dtype=x.dtype, device=x.device)
    m = torch.tensor([[torch.cos(a), -torch.sin(a), 0.0],
                      [torch.sin(a), torch.cos(a), 0.0]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(m[None].expand(x.shape[0], -1, -1), x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


class Predictor:
    def __init__(self, checkpoints: list[str | Path], device: str = "cpu"):
        self.device = torch.device(device)
        self.models, self.cfg, self.names = [], None, None
        for path in checkpoints:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = KeypointNet(len(ckpt["names"]),
                                tuple(ckpt["config"]["model"]["decoder_channels"]),
                                pretrained=False).to(self.device).eval()
            model.load_state_dict(ckpt["model"])
            self.models.append(model)
            self.cfg, self.names = ckpt["config"], ckpt["names"]
        self.flip_perm = P.flip_permutation(self.names)
        self.regions = P.region_slices(self.cfg["data"]["spine_edges"])
        self.thresholds = self._load_thresholds(checkpoints)

    def _load_thresholds(self, checkpoints) -> np.ndarray:
        """Пороги видимости, откалиброванные на OOF; иначе общий порог из конфига."""
        path = Path(checkpoints[0]).parent / "thresholds.csv"
        default = float(self.cfg["infer"]["visibility_threshold"])
        if not path.exists():
            return np.full(len(self.names), default)
        series = pd.read_csv(path, index_col=0).iloc[:, 0]
        return np.array([float(series.get(n, default)) for n in self.names])

    @torch.no_grad()
    def heatmaps(self, image: np.ndarray) -> torch.Tensor:
        """image: (H, W) uint8 в канонической ориентации. Возвращает (C, H, W) вероятности."""
        x = torch.from_numpy(normalize(pad_to_multiple(image)))[None, None].to(self.device)
        acc, n = 0.0, 0
        for model in self.models:
            for deg in self.cfg["infer"]["tta_rotations"]:
                acc = acc + _rotate(torch.sigmoid(model(_rotate(x, deg))), -deg)
                n += 1
            if self.cfg["infer"]["tta_flip_spine"]:
                flipped = torch.sigmoid(model(torch.flip(x, dims=[3])))
                acc = acc + torch.flip(flipped, dims=[3])[:, self.flip_perm]
                n += 1
        return (acc / n)[0]

    def predict(self, image: np.ndarray) -> dict:
        h = self.heatmaps(image)
        coords, conf = decode(h[None], self.cfg["heatmap"]["decode_window"])
        coords, conf = coords[0].cpu().numpy(), conf[0].cpu().numpy()
        visible = conf >= self.thresholds
        # какая группа каналов заговорила, та и область
        region = max(self.regions, key=lambda r: float(np.sort(conf[self.regions[r]])[-3:].mean()))
        return {
            "names": self.names,
            "coords": coords,                       # (C, 2) в координатах исходного снимка
            "confidence": conf,
            "visible": visible,
            "region": region if visible[self.regions[region]].any() else "out_of_scope",
        }


def fold_checkpoints(run_dir: str | Path | None = None) -> list[Path]:
    """Лучшие веса каждого фолда в каталоге прогона.

    `fold{N}_last.pt` — это состояние для продолжения обучения (оптимизатор,
    расписание, история), в ансамбль оно не идёт: там веса последней эпохи, а
    не лучшей.
    """
    d = Path(run_dir) if run_dir is not None else RUN_DIR
    return sorted(p for p in d.glob("fold*.pt") if not p.name.endswith("_last.pt"))


def load_predictor(run_dir: str | Path | None = None, device: str = "cpu") -> Predictor:
    """Ансамбль всех обученных фолдов из каталога прогона."""
    ckpts = fold_checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(
            f"в {Path(run_dir) if run_dir is not None else RUN_DIR} нет весов fold*.pt")
    return Predictor(ckpts, device)
