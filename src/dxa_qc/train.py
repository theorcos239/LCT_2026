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


def save_atomic(obj: dict, path: Path) -> None:
    """Запись через временный файл: обрыв посреди сохранения не оставит битый чекпоинт."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


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


def run_fold(cfg: dict, data: dict, fold: int, device: torch.device,
             resume: bool = True, on_checkpoint=None) -> dict:
    """on_checkpoint(fold, epoch, out_dir) вызывается после записи fold{N}_last.pt —
    например, чтобы скопировать чекпоинты на Диск."""
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
    # persistent_workers: без него воркеры пересоздаются на каждой эпохе, и на
    # Windows (spawn) каждый заново получает весь pack — эпоха шла 19 с вместо
    # 2.3 с при GPU, загруженном на 2 %. Подготовка кадра стоит ~3 мс, так что
    # и num_workers=0 почти не проигрывает.
    loader = dict(num_workers=t["num_workers"], persistent_workers=t["num_workers"] > 0)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                          drop_last=len(train_ds) > t["batch_size"], **loader)
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"], **loader)

    model = KeypointNet(len(names), tuple(cfg["model"]["decoder_channels"]),
                        cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.param_groups(t["lr_encoder"], t["lr_decoder"]),
                            weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_lambda(e, t["warmup_epochs"], t["epochs"]))
    ema = EMA(model, t["ema_decay"])

    out_dir = Path(t["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out_dir / f"fold{fold}.pt", out_dir / f"fold{fold}_last.pt"
    history_path = out_dir / f"fold{fold}_history.csv"

    best = {"median_mm": math.inf, "epoch": -1, "state": None, "oof": None}
    history, start_epoch = [], 0
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        best.update(median_mm=ckpt["best_median_mm"], epoch=ckpt["best_epoch"])
        history, start_epoch = ckpt["history"], ckpt["epoch"] + 1
        print(f"fold {fold}: продолжаем с эпохи {start_epoch} "
              f"(лучшая медиана {best['median_mm']:.2f} мм)")

    for epoch in range(start_epoch, t["epochs"]):
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
        history.append({"epoch": epoch, "loss": float(loss), "median_mm": median})
        pd.DataFrame(history).to_csv(history_path, index=False)

        if median < best["median_mm"]:
            best = {"median_mm": median, "epoch": epoch}
            save_atomic({"model": ema.shadow.state_dict(), "names": names, "config": cfg,
                         "fold": fold, "epoch": epoch, "median_mm": median}, best_path)
        if epoch % t.get("checkpoint_every", 10) == 0 or epoch == t["epochs"] - 1:
            save_atomic({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                         "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                         "epoch": epoch, "history": history, "config": cfg, "fold": fold,
                         "best_median_mm": best["median_mm"], "best_epoch": best["epoch"]},
                        last_path)
            if on_checkpoint is not None:
                on_checkpoint(fold, epoch, out_dir)
        if epoch - best["epoch"] >= t["patience"]:
            break
        print(f"fold {fold} epoch {epoch:3d} loss {loss.item():.4f} median {median:.2f} мм")

    # OOF считаем от сохранённых лучших весов: так результат не зависит от того,
    # был ли прогон продолжен после обрыва.
    ema.shadow.load_state_dict(torch.load(best_path, map_location=device,
                                          weights_only=False)["model"])
    best["oof"] = predict(ema.shadow, val_dl, device, cfg["heatmap"]["decode_window"])
    np.savez_compressed(out_dir / f"oof_fold{fold}.npz", **best["oof"])
    print(f"fold {fold}: лучшая медиана {best['median_mm']:.2f} мм (эпоха {best['epoch']})")
    return best


def save_predictions(data: dict, oof: dict, thresholds: pd.Series, out_dir: Path) -> None:
    """OOF-предсказания в двух видах: npz для кода и csv для анализа.

    К координатам прикладываются идентификаторы кадра (id, исследование,
    область, сторона): без них предсказания невозможно связать ни с метками
    эксперта, ни с DICOM, а порядок строк в npz зависит от сборки датасета.
    """
    idx = oof["index"]
    ident = {k: np.asarray(data[k])[idx] for k in ("ids", "studies", "regions", "sides")}
    np.savez_compressed(out_dir / "oof.npz", names=np.array(data["names"]),
                        thresholds=thresholds.to_numpy(), **ident, **oof)

    rows = []
    for row, (name_i) in enumerate(idx):                     # кадр
        for c, point in enumerate(data["names"]):            # канал
            rows.append({
                "id": ident["ids"][row], "study": ident["studies"][row],
                "region": ident["regions"][row], "side": ident["sides"][row],
                "point": point,
                "x": round(float(oof["pred"][row, c, 0]), 2),
                "y": round(float(oof["pred"][row, c, 1]), 2),
                "confidence": round(float(oof["conf"][row, c]), 4),
                "predicted_visible": bool(oof["conf"][row, c] >= thresholds.iloc[c]),
                "x_true": round(float(oof["true"][row, c, 0]), 2) if oof["visible"][row, c] else None,
                "y_true": round(float(oof["true"][row, c, 1]), 2) if oof["visible"][row, c] else None,
                "true_visible": bool(oof["visible"][row, c]),
                "labeled": bool(oof["labeled"][row, c]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "oof_points.csv", index=False)


def train_all(cfg: dict, resume: bool = True, on_checkpoint=None) -> dict:
    """Обучение по всем фолдам конфига. Возвращает OOF-предсказания, метрики и пороги.

    С resume=True продолжает прерванный прогон: готовые фолды пропускает, незаконченный
    поднимает с последнего чекпоинта. on_checkpoint(fold, epoch, out_dir) вызывается
    после каждой периодической записи — например, для копирования на Диск.
    """
    set_seed(cfg["seed"], cfg["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = packmod.load(cfg["data"]["pack"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgmod.save(cfg, out_dir / "config.yaml")

    oof = {}
    for fold in cfg["train"]["folds"]:
        done = out_dir / f"oof_fold{fold}.npz"
        if resume and done.exists():
            print(f"fold {fold}: уже обучен, пропускаем ({done})")
            oof[fold] = dict(np.load(done))
        else:
            oof[fold] = run_fold(cfg, data, fold, device, resume, on_checkpoint)["oof"]
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
    save_predictions(data, merged, thresholds, out_dir)
    return {"data": data, "oof": merged, "per_point": table, "end_to_end": e2e,
            "thresholds": thresholds, "out_dir": out_dir, "device": str(device)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/keypoints.yaml")
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    p.add_argument("--no-resume", action="store_true",
                   help="начать заново, игнорируя чекпоинты в out_dir")
    a = p.parse_args()

    result = train_all(cfgmod.load(a.config, a.overrides), resume=not a.no_resume)
    pd.set_option("display.width", 200)
    print(result["per_point"].round(2).to_string(index=False))
    print(result["end_to_end"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
