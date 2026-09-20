"""Обучение модели точек по фолдам из folds.csv.

python -m dxa_qc.train --config configs/keypoints.yaml [--set train.epochs=50]
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as cfgmod
from . import folds as foldsmod
from . import metrics as M
from . import pack as packmod
from .dataset import KeypointDataset
from .heatmaps import decode
from .loss import focal_heatmap_loss
from .model import KeypointNet
from .transforms import AugmentConfig


class EMA:
    """Усреднение траектории обучения; буферы BatchNorm тоже, иначе модель поедет."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def set_seed(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lr_lambda(epoch: int, warmup: int, total: int):
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def predict(model, loader, device, window: int):
    model.eval()
    out = {k: [] for k in ("pred", "conf", "true", "visible", "labeled", "index")}
    for batch in loader:
        logits = model(batch["image"].to(device))
        coords, conf = decode(torch.sigmoid(logits), window)
        out["pred"].append(coords.cpu().numpy())
        out["conf"].append(conf.cpu().numpy())
        out["true"].append(batch["coords"].numpy())
        out["visible"].append(batch["visible"].numpy())
        out["labeled"].append(batch["labeled"].numpy())
        out["index"].append(batch["index"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run_fold(cfg: dict, data: dict, fold: int, device: torch.device) -> dict:
    names = data["names"]
    studies = np.array(data["studies"])
    folds = foldsmod.load(cfg["data"]["folds"], set(map(str, studies)))
    fold_of = dict(zip(folds.study, folds.fold))
    in_val = np.array([fold_of.get(s, -1) == fold for s in studies])
    train_idx, val_idx = np.where(~in_val)[0], np.where(in_val)[0]

    d, t = cfg["data"], cfg["train"]
    common = dict(canvas=d["canvas"], sigma=cfg["heatmap"]["sigma"],
                  soft_mask_weight=cfg["loss"]["soft_mask_weight"],
                  use_clahe=d["use_clahe"], seed=cfg["seed"])
    train_ds = KeypointDataset(data, train_idx, aug=AugmentConfig(**cfg["augment"]), **common)
    val_ds = KeypointDataset(data, val_idx, aug=AugmentConfig(enabled=False), **common)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                          num_workers=t["num_workers"], drop_last=len(train_ds) > t["batch_size"])
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"], num_workers=t["num_workers"])

    model = KeypointNet(len(names), tuple(cfg["model"]["decoder_channels"]),
                        cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.param_groups(t["lr_encoder"], t["lr_decoder"]),
                            weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_lambda(e, t["warmup_epochs"], t["epochs"]))
    ema = EMA(model, t["ema_decay"])

    best = {"median_mm": math.inf, "epoch": -1, "state": None, "oof": None}
    for epoch in range(t["epochs"]):
        model.train()
        train_ds.epoch = epoch
        for batch in train_dl:
            logits = model(batch["image"].to(device))
            loss = focal_heatmap_loss(logits, batch["target"].to(device),
                                      batch["weight"].to(device),
                                      cfg["loss"]["alpha"], cfg["loss"]["beta"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
        sched.step()

        oof = predict(ema.shadow, val_dl, device, cfg["heatmap"]["decode_window"])
        err = np.linalg.norm(oof["pred"] - oof["true"], axis=-1) * d["mm_per_px"]
        median = float(np.median(err[oof["visible"]])) if oof["visible"].any() else math.inf
        if median < best["median_mm"]:
            best = {"median_mm": median, "epoch": epoch,
                    "state": copy.deepcopy(ema.shadow.state_dict()), "oof": oof}
        if epoch - best["epoch"] >= t["patience"]:
            break
        print(f"fold {fold} epoch {epoch:3d} loss {loss.item():.4f} median {median:.2f} мм")

    out_dir = Path(t["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best["state"], "names": names, "config": cfg},
               out_dir / f"fold{fold}.pt")
    print(f"fold {fold}: лучшая медиана {best['median_mm']:.2f} мм (эпоха {best['epoch']})")
    return best


def train_all(cfg: dict) -> dict:
    """Обучение по всем фолдам конфига. Возвращает OOF-предсказания, метрики и пороги."""
    set_seed(cfg["seed"], cfg["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = packmod.load(cfg["data"]["pack"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgmod.save(cfg, out_dir / "config.yaml")

    oof = {fold: run_fold(cfg, data, fold, device)["oof"] for fold in cfg["train"]["folds"]}
    keys = ("pred", "conf", "true", "visible", "labeled", "index")
    merged = {k: np.concatenate([oof[f][k] for f in oof]) for k in keys}

    thresholds = M.calibrate_thresholds(data["names"], merged["conf"], merged["visible"],
                                        merged["labeled"])
    table = M.per_point(data["names"], merged["pred"], merged["conf"], merged["true"],
                        merged["visible"], merged["labeled"],
                        thresholds.to_numpy(), cfg["data"]["mm_per_px"])
    e2e = M.end_to_end(data["names"], merged["pred"], merged["true"], merged["visible"],
                       np.array(data["regions"])[merged["index"]])

    thresholds.to_csv(out_dir / "thresholds.csv")
    table.to_csv(out_dir / "metrics_per_point.csv", index=False)
    e2e.to_csv(out_dir / "metrics_end_to_end.csv", index=False)
    np.savez_compressed(out_dir / "oof.npz", **merged)
    return {"data": data, "oof": merged, "per_point": table, "end_to_end": e2e,
            "thresholds": thresholds, "out_dir": out_dir, "device": str(device)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/keypoints.yaml")
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    a = p.parse_args()

    result = train_all(cfgmod.load(a.config, a.overrides))
    pd.set_option("display.width", 200)
    print(result["per_point"].round(2).to_string(index=False))
    print(result["end_to_end"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
