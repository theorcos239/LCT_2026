"""Инференс: ансамбль фолдов и TTA, усреднение карт.

Маскирование живёт только в функции потерь, поэтому здесь считаются все каналы:
какая группа заговорила — та и область. Отдельный классификатор области не нужен.
У позвоночника координаты затем согласуются по структуре (structure.py,
`infer.spine_chain`): каналы центров не прыгают на соседний позвонок.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import points as P
from .dataset import normalize, pad_to_multiple
from .heatmaps import decode, mask_outside
from .model import KeypointNet
from .structure import chain_decode, chain_decode_hypotheses

# Новый прогон с врачебными фрагментами предпочтителен. Старый каталог остаётся
# fallback, чтобы установка без новых весов продолжала работать.
_RUNS = Path(__file__).resolve().parents[2] / "runs"
RUN_DIR = _RUNS / "keypoints_crops"
LEGACY_RUN_DIR = _RUNS / "keypoints"


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
            numpy_types = [np._core.multiarray.scalar, np.dtype]
            numpy_types += [type(np.dtype(name)) for name in
                            ("float64", "int64", "str", "object", "bool")]
            with torch.serialization.safe_globals(numpy_types):
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model = KeypointNet(len(ckpt["names"]),
                                tuple(ckpt["config"]["model"]["decoder_channels"]),
                                pretrained=False).to(self.device).eval()
            model.load_state_dict(ckpt["model"])
            self.models.append(model)
            self.cfg, self.names = ckpt["config"], ckpt["names"]
        self.flip_perm = P.flip_permutation(self.names)
        self.regions = P.from_config(self.cfg)[1]
        self.thresholds = self._load_thresholds(checkpoints)
        self.crop_thresholds = self._load_crop_thresholds(checkpoints)
        self.anchor_gap_min = self._load_anchor_gap(checkpoints)

    @staticmethod
    def _load_anchor_gap(checkpoints) -> float:
        """OOF-фолд получает порог с других фолдов; ансамбль — общий."""
        path = Path(checkpoints[0]).parent / "thresholds_anchor.csv"
        if not path.exists():
            return .15
        table = pd.read_csv(path)
        fold = (int(Path(checkpoints[0]).stem.removeprefix("fold"))
                if len(checkpoints) == 1 else -1)
        row = table[table.fold == fold]
        if row.empty:
            raise ValueError(f"{path}: нет порога для fold={fold}")
        return float(row.iloc[0].gap_min)

    def _load_thresholds(self, checkpoints) -> np.ndarray:
        """Пороги видимости, откалиброванные на OOF; иначе общий порог из конфига."""
        path = Path(checkpoints[0]).parent / "thresholds.csv"
        default = float(self.cfg["infer"]["visibility_threshold"])
        if not path.exists():
            return np.full(len(self.names), default)
        series = pd.read_csv(path, index_col=0).iloc[:, 0]
        return np.array([float(series.get(n, default)) for n in self.names])

    def _load_crop_thresholds(self, checkpoints) -> np.ndarray:
        """Пороги семантических кропов; full-frame пороги — безопасный fallback."""
        path = Path(checkpoints[0]).parent / "thresholds_crop.csv"
        if not path.exists():
            return self.thresholds.copy()
        table = pd.read_csv(path)
        if {"name", "threshold"} <= set(table.columns):
            if "fold" in table.columns and len(checkpoints) == 1:
                fold = int(Path(checkpoints[0]).stem.removeprefix("fold"))
                own = table[table["fold"] == fold]
                if len(own):
                    table = own
            series = table.groupby("name")["threshold"].mean()
        else:
            series = pd.read_csv(path, index_col=0).iloc[:, 0]
        return np.array([float(series.get(n, self.thresholds[i]))
                         for i, n in enumerate(self.names)])

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

    def heatmaps_tta(self, image: np.ndarray) -> np.ndarray:
        """TTA-карты в координатах непаддированного входного изображения."""
        h = mask_outside(self.heatmaps(image)[None], image.shape)[0]
        return h[:, :image.shape[0], :image.shape[1]].cpu().numpy()

    def predict(self, image: np.ndarray, chain: bool | None = None) -> dict:
        """chain — согласовать позвоночник по структуре; None — как в конфиге (infer.spine_chain)."""
        window = self.cfg["heatmap"]["decode_window"]
        # Кадр дополнен нулями до кратности 32: точки в заполнении быть не может.
        h = mask_outside(self.heatmaps(image)[None], image.shape)
        coords, conf = decode(h, window)
        coords, conf = coords[0].cpu().numpy(), conf[0].cpu().numpy()
        visible = conf >= self.thresholds
        # какая группа каналов заговорила, та и область
        region = max(self.regions, key=lambda r: float(np.sort(conf[self.regions[r]])[-3:].mean()))
        if chain is None:
            chain = self.cfg["infer"].get("spine_chain", True)
        chain_info = None
        if region == "spine" and chain:
            maps = h[0].cpu().numpy()
            if self.cfg["infer"].get("spine_chain_mode", "hypotheses") == "legacy":
                coords = chain_decode(maps, self.names, coords, visible, window)
            else:
                coords, chain_info = chain_decode_hypotheses(
                    maps, self.names, coords, visible, window,
                    average=self.cfg["infer"].get("spine_chain_average", False),
                    temperature=self.cfg["infer"].get("spine_chain_temperature", 1.0),
                )
        return {
            "names": self.names,
            "coords": coords,                       # (C, 2) в координатах исходного снимка
            "confidence": conf,
            "visible": visible,
            "chain_hypotheses": chain_info,
            "region": region if visible[self.regions[region]].any() else "out_of_scope",
        }

    def predict_multicrop(self, image: np.ndarray, config: dict | None = None) -> dict:
        """Позвоночник по перекрывающимся полосам с проверкой цепочек."""
        from .multicrop import predict_multicrop, predict_semantic
        cfg = config
        if cfg is None:
            cfg = self.cfg.get("infer", {}).get("multicrop", {})
        if cfg.get("mode") == "consensus":
            return self.predict_consensus(image, cfg.get("config"))
        if cfg.get("mode") == "semantic" or cfg.get("version") == "semantic-v2":
            semantic_cfg = cfg.get("config") if isinstance(cfg, dict) else None
            return self.predict_semantic(image, semantic_cfg)
        return predict_multicrop(self, image, cfg)

    def predict_consensus(self, image: np.ndarray,
                          config: dict | str | Path | None = None,
                          box_mode: str = "fixed") -> dict:
        """Фиксированные семантические окна и согласование кандидатов в мм."""
        from .crop_consensus import predict_consensus
        return predict_consensus(self, image, config, box_mode)

    def predict_semantic(self, image: np.ndarray,
                         config: dict | str | Path | None = None) -> dict:
        from .multicrop import load_semantic_config, predict_semantic
        cfg = load_semantic_config(config)
        if config is None or isinstance(config, (str, Path)):
            cfg = {**cfg, "anchor": {**cfg.get("anchor", {}),
                                     "chain_gap_min": self.anchor_gap_min}}
        return predict_semantic(self, image, cfg)


def fold_checkpoints(run_dir: str | Path | None = None,
                     folds: list[int] | None = None) -> list[Path]:
    """Лучшие веса каждого фолда в каталоге прогона (или только перечисленных).

    `fold{N}_last.pt` — это состояние для продолжения обучения (оптимизатор,
    расписание, история), в ансамбль оно не идёт: там веса последней эпохи, а
    не лучшей.
    """
    if run_dir is not None:
        d = Path(run_dir)
    else:
        d = RUN_DIR if any(RUN_DIR.glob("fold[0-9].pt")) else LEGACY_RUN_DIR
    ckpts = sorted(p for p in d.glob("fold*.pt") if not p.name.endswith("_last.pt"))
    if folds is not None:
        ckpts = [p for p in ckpts if p.stem in {f"fold{k}" for k in folds}]
    return ckpts


def load_predictor(run_dir: str | Path | None = None, device: str = "cpu",
                   folds: list[int] | None = None) -> Predictor:
    """Ансамбль обученных фолдов из каталога прогона.

    `folds` — только эти фолды: модель одного фолда не видела в обучении его
    кадры, на этом строится честная out-of-fold оценка сервиса.
    """
    ckpts = fold_checkpoints(run_dir, folds)
    if not ckpts:
        raise FileNotFoundError(
            f"в {Path(run_dir) if run_dir is not None else RUN_DIR} нет весов fold*.pt")
    return Predictor(ckpts, device)
