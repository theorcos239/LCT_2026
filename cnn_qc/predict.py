# -*- coding: utf-8 -*-
"""Инференс cnn_qc: onnxruntime на CPU, без torch.

    from cnn_qc import CnnQC
    cnn = CnnQC()                                   # читает cnn_qc/<критерий>/
    r = cnn.assess('rotation', px, 'lh', geometry=12.5)
    r['p_cnn'], r['probability'], r['violated'], r['heatmap']

Сеть отвечает на вопрос «похож ли кадр на нарушение», свод с геометрией — «есть
ли нарушение»: вероятность = логистическая регрессия по логиту сети и
геометрическому скору критерия, порог — из OOF (train.py, meta.json).

Детерминизм (ТЗ 2.7): один поток на сессию и последовательное исполнение —
многопоточная редукция в свёртках складывает в недетерминированном порядке.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .preprocess import SIZE, to_input, valid_box

HERE = Path(__file__).resolve().parent
CRITERIA = ('rotation', 'artifacts')


class Session:
    """Одна ONNX-модель: вход (B, 1, S, S) в [0, 1], выход — карта CAM (B, 1, h, w)."""

    def __init__(self, path: str | Path):
        import onnxruntime as ort
        try:
            ort.disable_telemetry_events()      # сервис локальный: наружу ничего не уходит
        except Exception:                       # noqa: BLE001 — старые сборки без этой функции
            pass
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), sess_options=so,
                                         providers=['CPUExecutionProvider'])

    def cam(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {'image': x[None, None].astype(np.float32)})[0][0, 0]

    def probability(self, x: np.ndarray, hflip: bool = False) -> float:
        p = _sigmoid(self.cam(x).mean())
        if hflip:
            p = (p + _sigmoid(self.cam(x[:, ::-1].copy()).mean())) / 2
        return float(p)


def _sigmoid(z) -> float:
    return float(1.0 / (1.0 + np.exp(-float(z))))


def _stack(meta: dict, p_cnn: float, geometry: float | None) -> float:
    s = meta['stack']
    p = min(max(p_cnn, 1e-6), 1 - 1e-6)
    g = geometry if geometry is not None and np.isfinite(geometry) else 0.0
    z = s['w_cnn_logit'] * np.log(p / (1 - p)) + s['w_geometry'] * g + s['b']
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def heatmap(cam: np.ndarray, shape: tuple[int, int], region: str) -> np.ndarray:
    """CAM -> карта в координатах исходного кадра, [0, 1] (0 — нет вклада «за нарушение»)."""
    from PIL import Image
    h, w = shape
    nh, nw = valid_box(shape, SIZE)
    c = np.clip(cam, 0.0, None).astype(np.float32)
    up = np.asarray(Image.fromarray(c, mode='F').resize((SIZE, SIZE), Image.BILINEAR))
    up = up[:nh, :nw]
    up = np.asarray(Image.fromarray(np.ascontiguousarray(up), mode='F')
                    .resize((w, h), Image.BILINEAR))
    if region == 'rh':
        up = up[:, ::-1]
    m = float(up.max())
    return (up / m).astype(np.float32) if m > 1e-9 else np.zeros((h, w), np.float32)


class CnnQC:
    """Модели критериев, загружаются лениво. Недоступность не роняет сервис."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else HERE
        self.meta, self.sessions, self.errors = {}, {}, {}
        for c in CRITERIA:
            mp = self.root / c / 'meta.json'
            if mp.exists():
                self.meta[c] = json.loads(mp.read_text(encoding='utf-8'))
            else:
                self.errors[c] = f'нет {mp.relative_to(self.root.parent)}'

    def _session(self, c: str) -> list[Session]:
        """Модели ансамбля критерия (по одной на сид обучения)."""
        if c not in self.sessions:
            files = self.meta[c]['onnx']['files']
            self.sessions[c] = [Session(self.root / c / f) for f in files]
        return self.sessions[c]

    def available(self, c: str) -> bool:
        if c not in self.meta:
            return False
        try:
            self._session(c)
            return True
        except Exception as e:                  # noqa: BLE001 — нет onnxruntime, битый файл
            self.errors[c] = f'{type(e).__name__}: {e}'
            return False

    def p_cnn(self, c: str, px: np.ndarray, region: str) -> tuple[float, np.ndarray]:
        """Среднее вероятностей моделей ансамбля и средняя карта CAM."""
        x = to_input(px, region)
        flip = self.meta[c]['input']['hflip_tta']
        ps, cams = [], []
        for sess in self._session(c):
            cam = sess.cam(x)
            cams.append(cam)
            p = _sigmoid(cam.mean())
            if flip:
                p = (p + _sigmoid(sess.cam(x[:, ::-1].copy()).mean())) / 2
            ps.append(p)
        return float(np.mean(ps)), np.mean(cams, axis=0)

    def assess(self, c: str, px: np.ndarray, region: str, geometry: float | None) -> dict:
        """Вероятность сети, свод с геометрией и вердикт по порогу из meta.json."""
        p, cam = self.p_cnn(c, px, region)
        meta = self.meta[c]
        prob = p if meta.get('verdict') == 'cnn' else _stack(meta, p, geometry)
        return {'p_cnn': round(p, 4), 'probability': round(prob, 4),
                'threshold': meta['threshold'], 'violated': bool(prob >= meta['threshold']),
                'heatmap': heatmap(cam, px.shape, region)}


class OofCnnQC(CnnQC):
    """То же, но вероятность сети — out-of-fold из oof.csv (для service.evaluate).

    Итоговая модель обучена на всех 252 кадрах, и её оценка на них ничего не
    значит. Здесь кадр получает предсказание модели того фолда, который его в
    обучении не видел; кадр ищется по MD5 пикселей, как при дедупликации.
    """

    def __init__(self, root: str | Path | None = None):
        super().__init__(root)
        import pandas as pd
        self.oof = {}
        for c in list(self.meta):
            f = self.root / c / 'oof.csv'
            if f.exists():
                d = pd.read_csv(f)
                self.oof[c] = dict(zip(d.px_hash, d.p_cnn))

    def available(self, c: str) -> bool:
        return c in self.oof

    def p_cnn(self, c: str, px: np.ndarray, region: str) -> tuple[float, np.ndarray]:
        h = hashlib.md5(np.ascontiguousarray(px).tobytes()).hexdigest()
        if h not in self.oof[c]:
            raise KeyError(f'кадр {h[:8]} не входит в обучающий набор {c}')
        return float(self.oof[c][h]), np.zeros((10, 10), np.float32)
