"""Локальный просмотр настоящих heatmap модели точек на OOF фолда 0.

    PYTHONPATH=src python tools/serve_keypoint_heatmaps.py

Откройте адрес из терминала на этом же Mac. Снимки и карты отдаются только
через localhost; HTML-шаблон не содержит медицинских изображений.
"""
from __future__ import annotations

import argparse
import base64
import csv
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dxa_qc.model import KeypointNet
from dxa_qc.points import flip_permutation


ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("keypoint_heatmap_viewer.html")


def choose_device(requested: str) -> str:
    mps = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps:
        raise RuntimeError("MPS недоступен в этом PyTorch; используйте --device cpu")
    return "mps" if requested == "auto" and mps else ("cpu" if requested == "auto" else requested)


def rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    """Та же TTA-ротация, что в dxa_qc.infer, без зависимостей тренировки."""
    if degrees == 0:
        return x
    a = torch.tensor(np.deg2rad(degrees), dtype=x.dtype, device=x.device)
    matrix = torch.tensor([[torch.cos(a), -torch.sin(a), 0.0],
                           [torch.sin(a), torch.cos(a), 0.0]],
                          dtype=x.dtype, device=x.device)
    grid = F.affine_grid(matrix[None].expand(x.shape[0], -1, -1), x.shape,
                         align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


class HeatmapModel:
    """Только вывод карт Predictor: те же веса, нормировка, ансамбль TTA."""
    def __init__(self, checkpoint: Path, device: str):
        self.device = torch.device(device)
        # Старый формат чекпоинта хранит names как np.str_. Разрешаем только
        # необходимые типы NumPy; произвольные pickle-объекты не загружаются.
        numpy_types = [np._core.multiarray.scalar, np.dtype]
        numpy_types += [type(np.dtype(name)) for name in
                        ("float64", "int64", "str", "object", "bool")]
        with torch.serialization.safe_globals(numpy_types):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.names = list(saved["names"])
        self.cfg = saved["config"]
        self.median_mm = float(saved.get("median_mm", float("nan")))
        self.net = KeypointNet(len(self.names),
                               tuple(self.cfg["model"]["decoder_channels"]),
                               pretrained=False)
        self.net.load_state_dict(saved["model"])
        self.net = self.net.to(self.device).eval()
        self.flip_perm = flip_permutation(self.names)
        default = float(self.cfg["infer"]["visibility_threshold"])
        csv_path = checkpoint.parent / "thresholds.csv"
        calibrated = {}
        if csv_path.exists():
            with csv_path.open(newline="") as f:
                calibrated = {row[""]: float(row["threshold"])
                              for row in csv.DictReader(f)}
        self.thresholds = np.array([calibrated.get(n, default) for n in self.names])

    @torch.no_grad()
    def heatmaps(self, image: np.ndarray) -> torch.Tensor:
        h, w = image.shape
        padded = np.zeros((int(np.ceil(h / 32)) * 32,
                           int(np.ceil(w / 32)) * 32), dtype=np.float32)
        padded[:h, :w] = image
        x = torch.from_numpy((padded / 255.0 - .449) / .226)[None, None].to(self.device)
        acc, n = 0.0, 0
        for deg in self.cfg["infer"]["tta_rotations"]:
            acc = acc + rotate(torch.sigmoid(self.net(rotate(x, deg))), -deg)
            n += 1
        if self.cfg["infer"]["tta_flip_spine"]:
            flipped = torch.sigmoid(self.net(torch.flip(x, dims=[3])))
            acc = acc + torch.flip(flipped, dims=[3])[:, self.flip_perm]
            n += 1
        return (acc / n)[0]


def png_url(pixels: np.ndarray, mode: str) -> str:
    with io.BytesIO() as b:
        Image.fromarray(pixels, mode=mode).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("ascii")


def colorize(values: np.ndarray, scale: str) -> np.ndarray:
    max_value = float(values.max())
    divisor = max(max_value, 1e-6) if scale == "relative" else 1.0
    v = np.clip(values / divisor, 0, 1)
    stops = np.array([0, .15, .4, .7, 1.0])
    colors = np.array([[0, 0, 0], [25, 42, 170], [0, 190, 240],
                       [255, 225, 0], [245, 40, 15]])
    rgb = np.stack([np.interp(v, stops, colors[:, k]) for k in range(3)], axis=-1)
    alpha = np.where(v < .02, 0, np.clip(v ** .7 * 240, 0, 240))
    return np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)


def find_peaks(h: np.ndarray, floor: float = .02, min_distance: float = 8,
               top_k: int = 30) -> list[dict]:
    local = F.max_pool2d(torch.from_numpy(h)[None, None], 9, stride=1,
                         padding=4)[0, 0].numpy()
    ys, xs = np.nonzero((h == local) & (h >= floor))
    result = []
    for i in np.argsort(-h[ys, xs]):
        x, y = int(xs[i]), int(ys[i])
        if all((x - p["x"]) ** 2 + (y - p["y"]) ** 2 > min_distance ** 2
               for p in result):
            result.append({"x": x, "y": y, "value": round(float(h[y, x]), 4)})
            if len(result) >= top_k:
                break
    return result


class Viewer:
    def __init__(self, checkpoint: Path, pack_path: Path, device: str):
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Нет весов {checkpoint}. Нужен обученный fold0.pt; "
                "по сохранённым OOF-координатам heatmap не восстановить."
            )
        if not pack_path.exists():
            raise FileNotFoundError(f"Нет набора снимков {pack_path}")
        self.device = choose_device(device)
        self.model = HeatmapModel(checkpoint, self.device)
        with np.load(pack_path, allow_pickle=True) as z:
            meta = z["meta"]
            self.data = {"names": list(z["names"]), "images": list(z["images"]),
                         "ids": meta[:, 0], "studies": meta[:, 1],
                         "regions": meta[:, 2]}
        self.names = list(self.model.names)
        if list(self.data["names"]) != self.names:
            raise ValueError("Порядок точек в pack и чекпоинте различается")
        fold = int(checkpoint.stem.removeprefix("fold"))
        folds_path = Path(self.model.cfg["data"]["folds"])
        if not folds_path.is_absolute():
            folds_path = ROOT / folds_path
        with folds_path.open(newline="") as f:
            fold_of = {row["study"]: int(row["fold"])
                       for row in csv.DictReader(f)}
        missing = set(map(str, self.data["studies"])) - set(fold_of)
        if missing:
            raise ValueError(f"В folds.csv нет исследований: {sorted(missing)[:3]}")
        self.indices = [j for j, study in enumerate(self.data["studies"])
                        if fold_of.get(str(study)) == fold]
        if not self.indices:
            raise ValueError(f"Нет отложенных кадров фолда {fold}")
        self.allowed = set(self.indices)
        self.lock = threading.Lock()
        self.run_warning = None
        oof_path = ROOT / "runs/keypoints/oof.npz"
        if oof_path.exists() and np.isfinite(self.model.median_mm):
            with np.load(oof_path, allow_pickle=False) as z:
                rows = np.isin(z["index"], self.indices)
                if rows.any():
                    err = np.linalg.norm(z["pred"][rows] - z["true"][rows], axis=-1)
                    oof_median = float(np.median(err[z["visible"][rows]]) *
                                       self.model.cfg["data"].get("mm_per_px", .6))
                    if abs(oof_median - self.model.median_mm) > .25:
                        self.run_warning = (
                            "Этот чекпоинт и локальный OOF относятся к разным прогонам: "
                            f"медиана весов {self.model.median_mm:.2f} мм, "
                            f"локального OOF {oof_median:.2f} мм. "
                            "Карты настоящие, но их нельзя напрямую сравнивать с локальными метриками."
                        )

    def items(self) -> dict:
        return {
            "device": self.device,
            "run_warning": self.run_warning,
            "channels": [{"index": i, "name": name,
                          "region": "spine" if i < 21 else "hip"}
                         for i, name in enumerate(self.names)],
            "images": [{"index": j, "id": str(self.data["ids"][j]),
                        "region": str(self.data["regions"][j]),
                        "size": list(np.asarray(self.data["images"][j]).shape)}
                       for j in self.indices],
        }

    def image(self, j: int) -> str:
        if j not in self.allowed:
            raise ValueError("Снимок не принадлежит отложенному фолду")
        pixels = np.asarray(self.data["images"][j], dtype=np.uint8)
        return png_url(pixels, "L")

    @lru_cache(maxsize=3)
    def heatmaps(self, j: int) -> np.ndarray:
        if j not in self.allowed:
            raise ValueError("Снимок не принадлежит отложенному фолду")
        pixels = np.asarray(self.data["images"][j], dtype=np.uint8)
        with self.lock:
            maps = self.model.heatmaps(pixels).cpu().numpy()
        return maps[:, :pixels.shape[0], :pixels.shape[1]]

    def view(self, j: int, channel: int, scale: str) -> dict:
        if not (0 <= channel < len(self.names)):
            raise ValueError("Неизвестный канал")
        if scale not in ("relative", "absolute"):
            raise ValueError("Неизвестный масштаб")
        h = self.heatmaps(j)[channel]
        peak_floor = min(.02, max(.001, float(h.max()) * .1))
        return {
            "image_index": j,
            "channel": channel,
            "name": self.names[channel],
            "heatmap": png_url(colorize(h, scale), "RGBA"),
            "max": round(float(h.max()), 5),
            "threshold": round(float(self.model.thresholds[channel]), 5),
            "peak_floor": round(peak_floor, 5),
            "peaks": find_peaks(h, floor=peak_floor),
            "size": [int(h.shape[0]), int(h.shape[1])],
        }


def handler_for(viewer: Viewer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    data = PAGE.read_bytes()
                    kind = "text/html; charset=utf-8"
                elif parsed.path == "/api/items":
                    data = json.dumps(viewer.items(), ensure_ascii=False).encode()
                    kind = "application/json; charset=utf-8"
                elif parsed.path in ("/api/image", "/api/view"):
                    q = parse_qs(parsed.query)
                    j = int(q["index"][0])
                    if parsed.path == "/api/image":
                        payload = {"image": viewer.image(j)}
                    else:
                        payload = viewer.view(j, int(q["channel"][0]),
                                              q.get("scale", ["relative"])[0])
                    data = json.dumps(payload, ensure_ascii=False).encode()
                    kind = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (KeyError, IndexError, ValueError) as e:
                self.send_error(400, str(e))
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "Processing/drive_models/fold0.pt")
    p.add_argument("--pack", type=Path, default=ROOT / "data/keypoints.npz")
    p.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    p.add_argument("--port", type=int, default=8767)
    a = p.parse_args()
    viewer = Viewer(a.checkpoint, a.pack, a.device)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler_for(viewer))
    print(f"Fold 0: {len(viewer.indices)} отложенных снимков; device={viewer.device}")
    print(f"Откройте http://127.0.0.1:{a.port}/ (Ctrl+C для остановки)")
    server.serve_forever()


if __name__ == "__main__":
    main()
