# -*- coding: utf-8 -*-
"""Обучение нейросетевой оценки вида кадра: ротация бедра, посторонние предметы.

    python -m cnn_qc.train --criterion rotation      # OOF 5 фолдов x 3 сида, ансамбль, ONNX
    python -m cnn_qc.train --criterion artifacts
    python -m cnn_qc.train --criterion rotation --seeds 0   # быстрее: одна модель

Нужен torch (requirements-train.txt) и onnx; GPU желателен, но не обязателен.
Рабочему сервису torch не нужен: модель уходит в ONNX и исполняется onnxruntime.

Порядок и почему он такой:

1. **Out-of-fold по `folds.csv`** (группа — исследование): каждая модель учится
   на четырёх фолдах и предсказывает пятый. Только эти предсказания идут в
   калибровку и в метрики — оценка на кадрах, которые модель видела, у сети
   на 150 кадрах близка к единице и ничего не значит.
   Сидов три, и поставляется **ансамбль из трёх моделей** того же рецепта:
   на 36 положительных примерах одиночная сеть заметно зависит от сида
   (OOF ROC-AUC 0.76-0.78), а среднее трёх — около 0.80. Свод поэтому
   калибруется по среднему OOF трёх сидов — аналогу поставляемого ансамбля.
2. **Свод с геометрией.** Сеть не заменяет измерение, а дополняет его: вероятность
   нарушения — логистическая регрессия по двум признакам, логиту сети и
   геометрическому скору критерия (выход за коридор выступа малого вертела,
   счёт top-hat). Коэффициентов три, подбираются на OOF.
3. **Порог** вердикта — максимум F1 на OOF-своде. Честная оценка — вложенная:
   для каждого фолда свод и порог подбираются по остальным четырём.
4. **Итоговые модели** — тот же рецепт на всех кадрах, по одной на сид, экспорт
   в ONNX (веса fp16, приводятся к fp32 при загрузке: файл вдвое меньше,
   расхождение с torch измеряется и пишется в meta.json).

Вход сети — весь кадр (preprocess.py), а не вырезка: вырезка потребовала бы
надёжного ориентира, а его ошибка на бедре как раз того же порядка, что и
разница между классами (см. hip_rotation/README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # детерминированный cuBLAS

import numpy as np
import pandas as pd

import stats
import trainset

from .preprocess import MEAN, SIZE, STD, to_input

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CRITERIA = {
    'rotation': {'region': 'hip', 'label': 'y_rotation', 'hflip': False,
                 'geometry': 'выход площади выступа малого вертела за коридор нормы, мм²'},
    'artifacts': {'region': 'spine', 'label': 'y_artifacts', 'hflip': True,
                  'geometry': 'счёт white top-hat ведущей структуры'},
}
RECIPE = {'arch': 'resnet34', 'size': SIZE, 'epochs': 40, 'batch_size': 16,
          'lr': 2e-4, 'weight_decay': 1e-3, 'pct_start': 0.15, 'dropout': 0.3,
          'aug': {'rotation_deg': 8.0, 'scale': 0.08, 'shift': 0.06,
                  'brightness': 0.15, 'contrast': 0.15}}


# --------------------------------------------------------------------------- #
#  Данные
# --------------------------------------------------------------------------- #
def geometry_scores(criterion: str, df: pd.DataFrame) -> np.ndarray:
    """Геометрический скор критерия для каждого кадра — из измерений калибровки."""
    if criterion == 'rotation':
        from hip_rotation import THRESHOLDS
        from hip_rotation.detect import corridor_distance
        m = pd.read_csv(ROOT / 'hip_rotation' / 'measurements.csv').set_index('rel_path')
        area = df.rel_path.map(m.area_mm2).values.astype(float)
        return np.array([corridor_distance(a, THRESHOLDS) if np.isfinite(a) else np.nan
                         for a in area])
    m = pd.read_csv(ROOT / 'spine_qc' / 'measurements.csv').set_index('rel_path')
    return df.rel_path.map(m.artifact_score).values.astype(float)


def load_frames(criterion: str, pixels: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
    cfg = CRITERIA[criterion]
    df = trainset.frames(cfg['region'])
    df = df[df[cfg['label']].notna()].reset_index(drop=True)
    X = np.stack([to_input(trainset.read(r.rel_path), r.label)
                  for r in df.itertuples()]) if pixels else None
    df['y'] = df[cfg['label']].astype(int)
    df['geometry'] = geometry_scores(criterion, df)
    return df, X


# --------------------------------------------------------------------------- #
#  Модель
# --------------------------------------------------------------------------- #
def build_model(pretrained: bool = True):
    import torch
    import torch.nn as nn
    import torchvision

    class CamNet(nn.Module):
        """ResNet-34 -> GAP -> линейный слой. Вход (B, 1, H, W) в [0, 1].

        Стандартизация и размножение канала — внутри сети, чтобы предобработка
        сервиса оставалась numpy-функцией. GAP + линейный слой дают карту
        активации класса (CAM) бесплатно: логит равен среднему по карте.
        """

        def __init__(self):
            super().__init__()
            w = 'IMAGENET1K_V1' if pretrained else None
            r = torchvision.models.resnet34(weights=w)
            self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                      r.layer1, r.layer2, r.layer3, r.layer4)
            self.drop = nn.Dropout(RECIPE['dropout'])
            self.fc = nn.Linear(512, 1)
            self.register_buffer('mean', torch.tensor(MEAN))
            self.register_buffer('std', torch.tensor(STD))

        def features(self, x):
            return self.body(((x - self.mean) / self.std).repeat(1, 3, 1, 1))

        def forward(self, x):
            return self.fc(self.drop(self.features(x).mean((2, 3))))[:, 0]

    return CamNet()


class CamExport(object):
    """Обёртка для экспорта: выход — карта CAM (B, 1, h, w), логит = её среднее."""

    @staticmethod
    def wrap(model):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.head = nn.Conv2d(512, 1, 1)
                self.head.weight.data = m.fc.weight.data.view(1, 512, 1, 1).clone()
                self.head.bias.data = m.fc.bias.data.clone()

            def forward(self, x):
                return self.head(self.m.features(x))

        return _M(model).eval()


def augment(x, g, hflip: bool):
    import torch
    import torch.nn.functional as F
    a = RECIPE['aug']
    B = x.shape[0]

    def u(*shape):
        return (torch.rand(*shape, generator=g) * 2 - 1)

    ang = u(B) * np.radians(a['rotation_deg'])
    sc = 1 + u(B) * a['scale']
    tx, ty = u(B) * a['shift'], u(B) * a['shift']
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    b = u(B, 1, 1, 1).to(x.device) * a['brightness']
    c = 1 + u(B, 1, 1, 1).to(x.device) * a['contrast']
    x = (x - 0.5) * c + 0.5 + b
    if hflip:
        f = torch.rand(B, generator=g) < 0.5
        x[f] = x[f].flip(-1)
    return x


def seed_everything(seed: int) -> None:
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit(X: np.ndarray, y: np.ndarray, idx: np.ndarray, seed: int, hflip: bool, device):
    """Одна модель на кадрах idx по рецепту RECIPE."""
    import torch
    import torch.nn.functional as F

    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    model = build_model().to(device)
    bs, ep = RECIPE['batch_size'], RECIPE['epochs']
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'],
                            weight_decay=RECIPE['weight_decay'])
    steps = ep * int(np.ceil(len(idx) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=RECIPE['lr'], total_steps=steps,
                                              pct_start=RECIPE['pct_start'])
    pw = torch.tensor(float((1 - y[idx].mean()) / y[idx].mean()), device=device)
    Xt = torch.from_numpy(X[:, None])
    for _ in range(ep):
        model.train()
        perm = idx[torch.randperm(len(idx), generator=g).numpy()]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = augment(Xt[b].to(device), g, hflip)
            yb = torch.from_numpy(y[b].astype(np.float32)).to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    return model.eval()


def predict_torch(model, X: np.ndarray, hflip: bool, device) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            xb = torch.from_numpy(X[i:i + 32, None]).to(device)
            p = torch.sigmoid(model(xb))
            if hflip:
                p = (p + torch.sigmoid(model(xb.flip(-1)))) / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out).astype(float)


# --------------------------------------------------------------------------- #
#  Свод с геометрией и порог
# --------------------------------------------------------------------------- #
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(p_cnn: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """[логит сети, геометрический скор]; неизмеренная геометрия -> 0 («в норме»)."""
    g = np.where(np.isfinite(geometry), geometry, 0.0)
    return np.column_stack([logit(p_cnn), g])


def fit_stack(p_cnn: np.ndarray, geometry: np.ndarray, y: np.ndarray) -> dict:
    """Логистическая регрессия на стандартизованных признаках -> коэффициенты в сырых."""
    from sklearn.linear_model import LogisticRegression
    F_ = stack_features(p_cnn, geometry)
    mu, sd = F_.mean(0), F_.std(0)
    sd[sd < 1e-12] = 1.0
    m = LogisticRegression(C=1.0, max_iter=1000).fit((F_ - mu) / sd, y.astype(int))
    w = m.coef_[0] / sd
    b = float(m.intercept_[0] - np.sum(m.coef_[0] * mu / sd))
    return {'w_cnn_logit': round(float(w[0]), 6), 'w_geometry': round(float(w[1]), 6),
            'b': round(b, 6)}


def apply_stack(s: dict, p_cnn, geometry) -> np.ndarray:
    F_ = stack_features(np.atleast_1d(p_cnn), np.atleast_1d(geometry))
    z = s['w_cnn_logit'] * F_[:, 0] + s['w_geometry'] * F_[:, 1] + s['b']
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Порог максимума F1; при равенстве — более строгий (меньше ложных срабатываний)."""
    grid = np.unique(np.round(p, 4))
    best, thr = -1.0, 0.5
    for t in grid:
        f1 = stats.confusion(y, (p >= t).astype(int))['f1']
        if f1 > best + 1e-9:
            best, thr = f1, float(t)
    return thr


def threshold(rule: str, y: np.ndarray, p: np.ndarray) -> float:
    """Порог вердикта по вероятности.

    prevalence — доля нарушений в обучающей выборке: для откалиброванной
    вероятности это правило «апостериорная выше априорной», оно уравнивает
    цену ошибок обоих классов и не имеет свободных параметров. f1 — максимум
    F1 на обучающей выборке. На 29-36 положительных примерах максимум F1
    неустойчив: во вложенной оценке порог, найденный на четырёх фолдах,
    плохо переносился на пятый.
    """
    return float(y.mean()) if rule == 'prevalence' else best_threshold(y, p)


def verdict_probability(kind: str, stack: dict | None, p_cnn, geometry) -> np.ndarray:
    """Вероятность, по которой выносится вердикт: свод с геометрией или одна сеть."""
    if kind == 'cnn':
        return np.asarray(p_cnn, float)
    return apply_stack(stack, p_cnn, geometry)


def nested(df: pd.DataFrame, p_cnn: np.ndarray, rule: str = 'prevalence',
           kind: str = 'stack') -> tuple[np.ndarray, np.ndarray]:
    """Вложенная оценка: для фолда k свод и порог учатся на остальных четырёх."""
    y, g = df.y.values, df.geometry.values
    prob, pred = np.zeros(len(df)), np.zeros(len(df), int)
    for k in sorted(df.fold.unique()):
        te = (df.fold == k).values
        s = fit_stack(p_cnn[~te], g[~te], y[~te]) if kind == 'stack' else None
        t = threshold(rule, y[~te], verdict_probability(kind, s, p_cnn[~te], g[~te]))
        prob[te] = verdict_probability(kind, s, p_cnn[te], g[te])
        pred[te] = (prob[te] >= t).astype(int)
    return prob, pred


# --------------------------------------------------------------------------- #
#  Экспорт
# --------------------------------------------------------------------------- #
def export_onnx(model, path: Path, size: int) -> None:
    """ONNX с весами fp16 и узлами Cast -> fp32 (onnxruntime сворачивает их при загрузке)."""
    import onnx
    import torch
    from onnx import TensorProto, helper, numpy_helper

    net = CamExport.wrap(model.cpu())
    dummy = torch.zeros(1, 1, size, size)
    torch.onnx.export(net, dummy, str(path), input_names=['image'], output_names=['cam'],
                      dynamic_axes={'image': {0: 'batch'}, 'cam': {0: 'batch'}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    m = onnx.load(str(path))
    g = m.graph
    inits, casts = [], []
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT and int(np.prod(t.dims or [1])) > 64:
            arr = numpy_helper.to_array(t).astype(np.float16)
            inits.append(numpy_helper.from_array(arr, t.name + '__fp16'))
            casts.append(helper.make_node('Cast', [t.name + '__fp16'], [t.name],
                                          to=TensorProto.FLOAT))
        else:
            inits.append(t)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    m.producer_name = 'dxa-qc cnn_qc'
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


# --------------------------------------------------------------------------- #
def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description='Обучение cnn_qc')
    ap.add_argument('--criterion', required=True, choices=sorted(CRITERIA))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                    help='сиды: по модели на сид и в OOF, и в поставляемом ансамбле')
    ap.add_argument('--no-final', action='store_true', help='только OOF, без итоговой модели')
    ap.add_argument('--calibrate-only', action='store_true',
                    help='не обучать: пересчитать свод, порог и метрики по oof.csv')
    ap.add_argument('--verdict', choices=('stack', 'cnn'), default='stack',
                    help='вердикт по своду сети с геометрией или по одной сети')
    ap.add_argument('--threshold-rule', choices=('prevalence', 'f1'), default='prevalence',
                    help='порог: доля нарушений (по умолчанию) или максимум F1')
    args = ap.parse_args()

    cfg = CRITERIA[args.criterion]
    out = HERE / args.criterion
    out.mkdir(exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    df, X = load_frames(args.criterion, pixels=not args.calibrate_only)
    y = df.y.values
    print(f'{args.criterion}: {len(df)} кадров, нарушений {int(y.sum())}, устройство {dev}',
          flush=True)

    oof = {}
    if args.calibrate_only:
        prev = pd.read_csv(out / 'oof.csv').set_index('rel_path')
        for seed in args.seeds:
            oof[seed] = df.rel_path.map(prev[f'p_cnn_seed{seed}']).values.astype(float)
        print('  OOF сети взяты из oof.csv, сеть не переобучается', flush=True)
    for seed in ([] if args.calibrate_only else args.seeds):
        p = np.zeros(len(df))
        for k in sorted(df.fold.unique()):
            tr, te = np.where(df.fold != k)[0], np.where(df.fold == k)[0]
            m = fit(X, y, tr, seed * 100 + int(k), cfg['hflip'], dev)
            p[te] = predict_torch(m, X[te], cfg['hflip'], dev)
            del m
        oof[seed] = p
        print(f'  сид {seed}: OOF ROC-AUC {stats.roc_auc(y, p):.3f}, '
              f'PR-AUC {stats.average_precision(y, p):.3f}  ({time.time() - t0:.0f} с)', flush=True)

    # ансамбль = среднее вероятностей моделей разных сидов; его OOF-аналог —
    # среднее OOF по сидам (каждый кадр предсказан моделями, его не видевшими)
    p_cnn = np.mean([oof[k] for k in args.seeds], axis=0)
    g = df.geometry.values
    g_ok = np.where(np.isfinite(g), g, 0.0)
    kind, rule = args.verdict, args.threshold_rule
    prob_n, pred_n = nested(df, p_cnn, rule, kind)
    stack = fit_stack(p_cnn, g, y)
    p_stack = apply_stack(stack, p_cnn, g)
    p_verdict = verdict_probability(kind, stack, p_cnn, g)
    thr = threshold(rule, y, p_verdict)

    # вердикт одной геометрии — для сравнения (как в сервисе до сети)
    if args.criterion == 'rotation':
        geo_pred = (g_ok > 0).astype(int)
    else:
        spt = json.loads((ROOT / 'spine_qc' / 'thresholds.json').read_text(encoding='utf-8'))
        geo_pred = (g_ok > spt['artifact_score']).astype(int)

    groups = df.study.values
    # verdict_oof — поставляемое правило (свод или сеть и порог) по всем OOF:
    # параметров 3-4, оценка слегка оптимистична. verdict_nested — то же, но
    # свод и порог подобраны без отложенного фолда: его чувствительность,
    # специфичность и F1 честные. AUC у nested ниже: вероятности пяти разных
    # сводов, склеенные в один ряд, ранжируются хуже любого из них.
    res = {
        'cnn_oof': stats.evaluate(y, (p_cnn >= threshold(rule, y, p_cnn)).astype(int),
                                  p_cnn, groups),
        'geometry': stats.evaluate(y, geo_pred, g_ok, groups),
        'stack_oof': stats.evaluate(y, (p_stack >= threshold(rule, y, p_stack)).astype(int),
                                    p_stack, groups),
        'verdict_oof': stats.evaluate(y, (p_verdict >= thr).astype(int), p_verdict, groups),
        'verdict_nested': stats.evaluate(y, pred_n, prob_n, groups),
    }
    res['cnn_oof_by_seed'] = {str(k): {'roc_auc': round(stats.roc_auc(y, v), 4),
                                       'pr_auc': round(stats.average_precision(y, v), 4)}
                              for k, v in oof.items()}
    for name, m in res.items():
        if 'f1' in m:
            print(f'  {name:15} {stats.fmt(m)}')

    meta = {
        'criterion': args.criterion, 'region': cfg['region'], 'recipe': RECIPE,
        'input': {'size': SIZE, 'mirror_rh': True, 'hflip_tta': cfg['hflip'],
                  'normalize': 'перцентили 0.5-99.5 -> [0,1], длинная сторона -> size'},
        'geometry': cfg['geometry'], 'verdict': kind, 'threshold_rule': rule,
        'stack': stack, 'threshold': round(thr, 4),
        'n': int(len(df)), 'positives': int(y.sum()), 'folds': 'folds.csv',
        'seeds': args.seeds,
        'metrics': {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                    if isinstance(v, dict) else v for k, v in res.items()},
    }
    pd.DataFrame({'rel_path': df.rel_path, 'study': df.study, 'px_hash': df.px_hash,
                  'label': df.label, 'fold': df.fold, 'y': y, 'geometry': g,
                  'p_cnn': np.round(p_cnn, 6),
                  **{f'p_cnn_seed{k}': np.round(v, 6) for k, v in oof.items()},
                  'p_stack': np.round(p_stack, 6), 'p_verdict': np.round(p_verdict, 6),
                  'p_verdict_nested': np.round(prob_n, 6), 'pred_nested': pred_n}
                 ).to_csv(out / 'oof.csv', index=False, encoding='utf-8')

    if args.calibrate_only:
        meta['onnx'] = json.loads((out / 'meta.json').read_text(encoding='utf-8')).get('onnx')
    if not (args.no_final or args.calibrate_only):
        from .predict import Session
        for stale in out.glob('model*.onnx'):
            stale.unlink()
        files, diffs = [], []
        for i, seed in enumerate(args.seeds):
            m = fit(X, y, np.arange(len(df)), seed * 100 + 99, cfg['hflip'], dev)
            p_torch = predict_torch(m, X, cfg['hflip'], dev)
            name = f'model_{i}.onnx'
            export_onnx(m, out / name, SIZE)
            sess = Session(out / name)
            p_onnx = np.array([sess.probability(x, cfg['hflip']) for x in X])
            files.append(name)
            diffs.append(float(np.max(np.abs(p_onnx - p_torch))))
            del m, sess
        meta['onnx'] = {'files': files,
                        'bytes': int(sum((out / f).stat().st_size for f in files)),
                        'max_abs_diff_vs_torch': round(max(diffs), 6)}
        print(f"  ONNX: {len(files)} x {meta['onnx']['bytes'] / len(files) / 1e6:.1f} МБ, "
              f"расхождение с torch {meta['onnx']['max_abs_diff_vs_torch']:.2e}")
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float),
                                   encoding='utf-8')
    print(f'сохранено в {out} за {time.time() - t0:.0f} с')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
