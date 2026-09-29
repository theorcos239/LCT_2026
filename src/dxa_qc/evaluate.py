"""Честная оценка и калибровка модели точек на рабочем конвейере инференса.

Каждый фолд оценивается своими весами только на своих отложенных кадрах —
исходных и обрезанных (TruncatedDataset: край отрезан, часто вплотную к
размеченной точке, видимость после обрезания известна точно). Предсказания
делает тот же `Predictor`, что работает в сервисе: TTA, маска заполнения,
согласованная расшифровка позвоночника. Это важно: TTA усредняет чуть
смещённые пики и снижает их высоту (медиана −15 %, у каждой десятой точки
−33 % и больше), и порог, подобранный по одиночному проходу, в сервисе терял
15 % видимых точек вместо 6 %.

    python -m dxa_qc.evaluate --run runs/keypoints [--folds 0 1] [--calibrate]

--calibrate пересчитывает пороги видимости прогона (thresholds.csv) по этим
предсказаниям. Оценка видимости идёт с перекрёстными порогами: подобранными
на остальных фолдах и применёнными к этому, — иначе она подсматривает в ответ.

Печатает: локализацию (мм) с цепочкой позвоночника и без, ошибки уровня
позвонка, сквозные величины (угол оси, отступы ROI, выступ малого вертела),
долю выдуманных и потерянных точек.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import folds as foldsmod
from . import metrics as M
from . import pack as packmod
from .dataset import TruncatedDataset
from .infer import Predictor, fold_checkpoints
from .pack import ABSENT, LABELED
from .transforms import inside_frame

CENTERS = [f"l{i}_center" for i in range(1, 5)]
KEYS = ("pred", "conf", "true", "visible", "labeled", "index")


def fold_frames(data: dict, val_idx, crops: int = 6, seed: int = 0) -> tuple[list, list]:
    """Отложенные кадры фолда: исходные и обрезанные.

    Кадр — (снимок uint8, истинные точки, видимость, размечена ли, индекс в pack).
    Обрезы детерминированы seed: у всех прогонов одни и те же.
    """
    orig, trunc = [], []
    for j in val_idx:
        st = data["state"][j]
        orig.append((np.asarray(data["images"][j]), data["coords"][j], st == LABELED,
                     (st == LABELED) | (st == ABSENT), int(j)))
    td = TruncatedDataset(data, val_idx, crops, seed=seed)
    for i in range(len(td)):
        image, coords, st, valid, j = td.raw(i)
        trunc.append((image, coords, (st == LABELED) & inside_frame(coords, valid),
                      (st == LABELED) | (st == ABSENT), int(j)))
    return orig, trunc


def predict_frames(predictor: Predictor, frames: list, chain: bool = True) -> dict:
    out = {k: [] for k in KEYS}
    for image, coords, vis, lab, j in frames:
        r = predictor.predict(np.clip(image, 0, 255).astype(np.uint8), chain=chain)
        for k, v in (("pred", r["coords"]), ("conf", r["confidence"]), ("true", coords),
                     ("visible", vis), ("labeled", lab), ("index", j)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def fold_oof(run_dir: str | Path, data: dict, device: torch.device | str,
             crops: int = 6, folds: list[int] | None = None,
             thresholds: np.ndarray | None = None) -> list[dict]:
    """По каждому фолду — предсказания рабочего инференса на его отложенных кадрах.

    orig — исходные кадры, orig_indep — они же без цепочки позвоночника,
    trunc — обрезанные. thresholds — пороги видимости вместо thresholds.csv
    прогона (они решают, какие центры позвонков идут в цепочку).
    """
    studies = np.array(data["studies"])
    result = []
    for path in fold_checkpoints(run_dir, folds):
        predictor = Predictor([path], str(device))
        if thresholds is not None:
            predictor.thresholds = np.asarray(thresholds, float)
        cfg, fold = predictor.cfg, int(path.stem.removeprefix("fold"))
        split = foldsmod.load(cfg["data"]["folds"], set(map(str, studies)))
        fold_of = dict(zip(split.study, split.fold))
        val_idx = np.flatnonzero([fold_of.get(s, -1) == fold for s in studies])
        orig, trunc = fold_frames(data, val_idx, crops)
        result.append({"fold": fold,
                       "orig": predict_frames(predictor, orig),
                       "orig_indep": predict_frames(predictor, orig, chain=False),
                       "trunc": predict_frames(predictor, trunc, chain=False)})
    return result


def run_thresholds(run_dir: str | Path, names: list[str]) -> np.ndarray:
    path = Path(run_dir) / "thresholds.csv"
    if not path.exists():
        return np.full(len(names), 0.5)
    return pd.read_csv(path, index_col=0).iloc[:, 0].reindex(names).fillna(0.5).to_numpy()


def concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def pooled(folds: list[dict]) -> dict:
    """Исходные и обрезанные кадры всех фолдов вместе — материал для порога видимости."""
    return concat([f[k] for f in folds for k in ("orig", "trunc")])


def calibrate(names: list[str], folds: list[dict], objective: str = "balanced",
              default: float = 0.5) -> pd.Series:
    p = pooled(folds)
    return M.calibrate_thresholds(names, p["conf"], p["visible"], p["labeled"],
                                  objective=objective, default=default)


def level_errors(names: list[str], oof: dict, regions: np.ndarray,
                 mm_per_px: float = M.MM_PER_PX) -> pd.DataFrame:
    """Центр тела позвонка, поставленный на соседний позвонок.

    Для каждого видимого центра L1–L4 ищется ближайший истинный центр в цепочке
    Th12, L1–L4, L5 (Th12 — середина между её замыкательными пластинками, L5 —
    продолжение шага L3→L4). Сдвиг ≠ 0 — модель перепутала уровень.
    """
    idx = {n: i for i, n in enumerate(names)}
    c = [idx[n] for n in CENTERS]
    rows = []
    for r in range(len(oof["pred"])):
        if regions[oof["index"][r]] != "spine" or not oof["visible"][r, c].all():
            continue
        true = oof["true"][r, c]
        chain = [true[0] - (true[1] - true[0])]           # Th12 по шагу, если краёв нет
        if oof["visible"][r, [idx["th12_top"], idx["th12_bottom"]]].all():
            chain = [oof["true"][r, [idx["th12_top"], idx["th12_bottom"]]].mean(0)]
        chain = np.array(chain + list(true) + [true[3] + (true[3] - true[2])])
        shift = []
        for k in range(4):
            d = np.linalg.norm(chain - oof["pred"][r, c[k]], axis=1)
            shift.append(int(np.argmin(d)) - (k + 1))
        err = np.linalg.norm(oof["pred"][r, c] - true, axis=1) * mm_per_px
        rows.append(dict(index=int(oof["index"][r]), shift=tuple(shift),
                         any_shift=any(shift), max_err_mm=float(err.max())))
    return pd.DataFrame(rows, columns=["index", "shift", "any_shift", "max_err_mm"])


def crossfit_thresholds(names, folds: list[dict], objective="balanced",
                        default=0.5) -> list[np.ndarray]:
    """Порог для фолда k — по OOF (исходные + обрезанные) остальных фолдов."""
    if len(folds) < 2:
        raise ValueError("перекрёстный порог требует хотя бы двух фолдов")
    return [calibrate(names, [f for i, f in enumerate(folds) if i != k], objective,
                      default).to_numpy() for k in range(len(folds))]


def _visibility(names, parts: list[dict], thresholds: list[np.ndarray]) -> pd.DataFrame:
    """Сводная видимость, у каждого фолда — свой порог."""
    found = np.concatenate([p["conf"] >= t for p, t in zip(parts, thresholds)])
    pool = concat(parts)
    # visibility_table сравнивает conf с порогом: подставим 0/1 и порог 0.5
    return M.visibility_table(names, found.astype(float), pool["visible"], pool["labeled"], 0.5)


def summarize(run_dir: str | Path, data: dict, folds: list[dict]) -> dict:
    names, regions = list(data["names"]), np.array(data["regions"])
    out = {"run": str(run_dir), "folds": ",".join(str(f["fold"]) for f in folds)}
    lv = None
    for key in ("orig", "orig_indep"):
        orig = concat([f[key] for f in folds])
        err = np.linalg.norm(orig["pred"] - orig["true"], axis=-1) * M.MM_PER_PX
        vis = orig["visible"]
        spine = (regions[orig["index"]] == "spine")[:, None] & vis
        levels = level_errors(names, orig, regions)
        tag = "chain" if key == "orig" else "indep"
        out.update({f"median_mm_{tag}": float(np.median(err[vis])),
                    f"p90_mm_{tag}": float(np.percentile(err[vis], 90)),
                    f"spine_p90_mm_{tag}": float(np.percentile(err[spine], 90)),
                    f"spine_err_gt15mm_{tag}": float((err[spine] > 15).mean()),
                    f"level_err_frames_{tag}": int(levels.any_shift.sum())})
        if tag == "chain":
            lv = levels
            e2e = M.end_to_end(names, orig["pred"], orig["true"], vis, regions[orig["index"]])
            out.update({f"e2e_{r.kind}_median": r["median"] for _, r in e2e.iterrows()})
            out.update({f"e2e_{r.kind}_p90": r.p90 for _, r in e2e.iterrows()})
    out["spine_frames"] = len(lv)
    parts = [f["trunc"] for f in folds] + [f["orig"] for f in folds]
    tables = {}
    variants = [("own", [run_thresholds(run_dir, names)] * len(folds))]
    if len(folds) > 1:
        variants.append(("crossfit", crossfit_thresholds(names, folds)))
    for th_name, ths in variants:
        t = _visibility(names, parts, ths + ths)
        tables[th_name] = t
        neg, pos = t.n_absent.sum(), t.n_visible.sum()
        out[f"halluc_{th_name}"] = float(np.nansum(t.hallucination_rate * t.n_absent) / neg)
        out[f"miss_{th_name}"] = float(np.nansum(t.miss_rate * t.n_visible) / pos)
    return {"summary": out, "tables": tables, "levels": lv}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--pack", default="data/keypoints.npz")
    p.add_argument("--crops", type=int, default=6)
    p.add_argument("--folds", type=int, nargs="*", default=None,
                   help="только эти фолды — для сравнения с недообученным прогоном")
    p.add_argument("--calibrate", action="store_true",
                   help="пересчитать thresholds.csv прогона по рабочему инференсу")
    p.add_argument("--out", default=None, help="куда сохранить сводку и таблицы (csv)")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = packmod.load(a.pack)
    names = list(data["names"])
    rows, per_point = [], []
    for run in a.run:
        folds = fold_oof(run, data, device, a.crops, folds=a.folds)
        if a.calibrate:
            th = calibrate(names, folds)
            th.to_csv(Path(run) / "thresholds.csv")
            print(f"{run}: пороги видимости пересчитаны по рабочему инференсу")
            # цепочка позвоночника зависит от порогов — пересчитать с новыми
            folds = fold_oof(run, data, device, a.crops, folds=a.folds, thresholds=th.to_numpy())
        res = summarize(run, data, folds)
        rows.append(res["summary"])
        key = "crossfit" if "crossfit" in res["tables"] else "own"
        per_point.append(res["tables"][key].assign(run=run))
    pd.set_option("display.width", 220)
    summary = pd.DataFrame(rows)
    print(summary.round(3).T.to_string())
    table = pd.concat(per_point)
    print(table.pivot(index="point", columns="run",
                      values=["hallucination_rate", "miss_rate"]).round(3).to_string())
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        summary.to_csv(Path(a.out) / "metrics_eval_summary.csv", index=False)
        table.to_csv(Path(a.out) / "metrics_eval_visibility.csv", index=False)


if __name__ == "__main__":
    main()
