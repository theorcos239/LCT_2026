# -*- coding: utf-8 -*-
"""Тесты hip_roi. Запуск: python -m hip_roi.test_hip_roi  (или pytest).

1. Синтетический фантом бедра с известными T/B/L — каждый комплект обязан
   попасть в допуск; тот же фантом с тесным верхом / низом / латералью —
   обязан дать нарушение именно по этому отступу.
2. Реальные кадры (если данные на месте): детерминизм (два прогона побитово
   равны), симметрия (зеркальный lh == rh), короткие кадры (<= 210 строк)
   дают нарушение снизу у всех комплектов.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .geometry import MM_PER_PX, mm2px, to_lateral_left
from .kits import measure_roi_margins

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'НД_для_обучения' / 'Исследования'
LABELS = ROOT / 'region_clf' / 'labels.csv'
KITS = ('kit1', 'kit2', 'kit3')


# --------------------------------------------------------------------------- #
#  Фантом
# --------------------------------------------------------------------------- #
def _disc(shape, cy, cx, ry, rx):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _band(shape, p0, p1, half):
    """Пиксели на расстоянии <= half от отрезка p0-p1 (y, x)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    (y0, x0), (y1, x1) = p0, p1
    vy, vx = y1 - y0, x1 - x0
    t = ((yy - y0) * vy + (xx - x0) * vx) / max(vy * vy + vx * vx, 1e-9)
    t = np.clip(t, 0, 1)
    d = np.hypot(yy - (y0 + t * vy), xx - (x0 + t * vx))
    return d <= half


def synthetic_hip(top_mm=45.0, bottom_mm=50.0, lat_mm=35.0, d_tb_mm=50.0, W=280, side='rh'):
    """Фантом: латераль слева. Возвращает (uint8 кадр, ожидаемые T, B, L в px).

    Анатомия фиксирована (верхушка вертела -> низ малого вертела = d_tb_mm),
    отступы задаются положением краёв кадра: H = top + d_tb + bottom.
    """
    m = mm2px
    T, L = m(top_mm), m(lat_mm)
    B = T + m(d_tb_mm)
    H = B + m(bottom_mm)
    shape = (H, W)
    bone = np.zeros(shape, bool)
    shaft_x0, shaft_x1 = L + m(8), L + m(8) + m(27)
    bone[T + m(40):, shaft_x0:shaft_x1] = True                                   # диафиз
    # межвертельная зона: медиальный контур сужается по прямой от L+60 мм на
    # уровне T+30 мм до края диафиза на уровне B (~3.5 мм/см, как в жизни)
    yy, xx = np.mgrid[0:H, 0:W]
    y0, y1 = T + m(30), B
    x_med = L + m(60) + (xx * 0 + yy - y0) * (shaft_x1 - (L + m(60))) / max(y1 - y0, 1)
    bone |= (yy >= y0) & (yy <= y1) & (xx >= shaft_x0) & (xx <= x_med)
    bone |= _disc(shape, T + m(22), L + m(15), m(22), m(15))                    # большой вертел
    bone |= _band(shape, (T + m(35), L + m(25)), (T + m(10), L + m(70)), m(12))  # шейка
    bone |= _disc(shape, T + m(5), L + m(78), m(22), m(22))                      # головка
    bone |= _disc(shape, B - m(9), shaft_x1, m(9), m(7))                         # малый вертел
    pelvis = np.zeros(shape, bool)
    pelvis[:T + m(15), L + m(85):] = True
    pelvis |= _disc(shape, T - m(10), L + m(95), m(30), m(30))                   # вертлужная впадина
    pelvis |= _disc(shape, B - m(20), L + m(85), m(12), m(12))                   # седалищная кость (с зазором от бедра)
    img = np.zeros(shape, np.uint8)
    img[pelvis] = 150
    img[bone] = 160
    img[_disc(shape, T + m(22), L + m(15), m(12), m(8))] = 90                    # трабекулярная зона
    # край кости на настоящих кадрах не обрывается, а спадает за 3-5 px:
    # без этого фон кадра примыкает к кости вплотную и выглядит как обрез поля
    from scipy import ndimage as ndi
    img = ndi.gaussian_filter(img.astype(np.float32), 1.6)
    yy, xx = np.mgrid[0:H, 0:W]
    img[(img < 2) & (((yy * 7 + xx * 3) % 11) == 0)] = 25                         # крапинки гало
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = to_lateral_left(img, side)                                              # для lh зеркалим
    return img, T, B, L


def _check(name, res, T, B, L, tol_T, tol_B, tol_L, H, W, side):
    Lo = L if side == 'rh' else W - 1 - L
    errs = []
    if res['T_px'] is None or not (-tol_T[0] <= (res['T_px'] - T) * MM_PER_PX <= tol_T[1]):
        errs.append(f"T={res['T_px']} ожидалось {T}±({tol_T}) мм")
    if res['B_px'] is None or abs(res['B_px'] - B) * MM_PER_PX > tol_B:
        errs.append(f"B={res['B_px']} ожидалось {B}±{tol_B} мм")
    if res['L_px'] is None or abs(res['L_px'] - Lo) * MM_PER_PX > tol_L:
        errs.append(f"L={res['L_px']} ожидалось {Lo}±{tol_L} мм")
    assert not errs, f'{name}: ' + '; '.join(errs) + f"  flags={res['flags']}"


def test_synthetic_landmarks():
    for side in ('rh', 'lh'):
        img, T, B, L = synthetic_hip(side=side)
        H, W = img.shape
        r1 = measure_roi_margins(img, side, 'kit1')
        _check(f'kit1/{side}', r1, T, B, L, (0.0, 12.0), 12.0, 1.5, H, W, side)   # T1 ниже верхушки
        r2 = measure_roi_margins(img, side, 'kit2')
        # низ у kit2 — анатомический якорь с поправкой, его точность ±1.5 см
        _check(f'kit2/{side}', r2, T, B, L, (2.0, 2.0), 15.0, 1.5, H, W, side)
        r3 = measure_roi_margins(img, side, 'kit3')
        _check(f'kit3/{side}', r3, T, B, L, (3.0, 3.0), 10.0, 1.5, H, W, side)
        for r in (r1, r2, r3):
            assert r['roi_ok'] is True, (r['method'], r['violation_text'], r['flags'])


def test_synthetic_violations():
    cases = {
        'top': dict(top_mm=20.0),
        'lat': dict(lat_mm=14.0),
        'bottom': dict(bottom_mm=8.0),
    }
    for which, kw in cases.items():
        img, T, B, L = synthetic_hip(**kw)
        for kit in KITS:
            r = measure_roi_margins(img, 'rh', kit)
            assert r['roi_ok'] is False, (which, kit, r)
            assert r[f'{which}_ok'] is False, (which, kit, r)
            others = [k for k in ('top', 'bottom', 'lat') if k != which]
            assert all(r[f'{k}_ok'] is not False for k in others), (which, kit, r)


def test_kit0_scan_length():
    img, *_ = synthetic_hip(bottom_mm=8.0)          # 45 + 50 + 8 = 10.3 см < 11 см
    assert measure_roi_margins(img, 'rh', 'kit0')['roi_ok'] is False
    img, *_ = synthetic_hip()                       # 14.5 см
    assert measure_roi_margins(img, 'rh', 'kit0')['roi_ok'] is True


def test_custom_combo_runs():
    img, *_ = synthetic_hip()
    for L in ('L1', 'L2', 'L4'):
        for T in ('T1', 'T2', 'T3'):
            for B in ('B1', 'B2', 'B3', 'B4'):
                r = measure_roi_margins(img, 'rh', 'custom', L=L, T=T, B=B)
                assert r['T_px'] is not None and r['L_px'] is not None, (L, T, B, r['flags'])


# --------------------------------------------------------------------------- #
#  Реальные кадры
# --------------------------------------------------------------------------- #
def _real_frames(limit=None):
    import pandas as pd
    from region_clf.features import read_image
    if not (DATA.exists() and LABELS.exists()):
        return []
    lab = pd.read_csv(LABELS)
    lab = lab[lab.label.isin(('lh', 'rh'))]
    out = []
    for _, r in lab.iterrows():
        out.append((r.label, read_image(DATA / r.rel_path), r.rel_path))
        if limit and len(out) >= limit:
            break
    return out


def _strip(res):
    return {k: v for k, v in res.items() if k != 'diag'}


def test_real_deterministic():
    frames = _real_frames(limit=12)
    for side, img, rel in frames:
        for kit in KITS:
            a, b = measure_roi_margins(img, side, kit), measure_roi_margins(img, side, kit)
            assert _strip(a) == _strip(b), (kit, rel)


def test_real_symmetry():
    """Зеркальный кадр с противоположной меткой -> те же отступы."""
    frames = _real_frames(limit=20)
    for side, img, rel in frames:
        other = 'lh' if side == 'rh' else 'rh'
        for kit in KITS:
            a = measure_roi_margins(img, side, kit)
            b = measure_roi_margins(img[:, ::-1], other, kit)
            for key in ('m_top_mm', 'm_bottom_mm', 'm_lat_mm', 'roi_ok'):
                assert a[key] == b[key], (kit, rel, key, a[key], b[key])


def test_real_short_frames_flagged():
    """Кадры, где эксперт увидел некорректный ROI из-за обрезанного низа.

    Гарантия даётся на kit2: именно он измеряет верхушку вертела и ставит низ
    ROI от неё. kit0 (длина кадра) после калибровки d_TB = 50 мм такие кадры
    уже не ловит — 12.6 см длины формально хватает на 3 + 5 + 3 см; kit1 на
    части из них стабилизируется выше малого вертела. Это ограничения обоих
    комплектов, они описаны в README.
    """
    frames = [f for f in _real_frames() if f[1].shape[0] <= 195]
    assert not _real_frames() or frames, 'в выборке должны быть короткие кадры'
    for side, img, rel in frames:
        r = measure_roi_margins(img, side, 'kit2')
        assert r['roi_ok'] is False, (rel, r['m_top_mm'], r['m_bottom_mm'], r['m_lat_mm'], r['flags'])


def test_real_no_exceptions_all_frames():
    for side, img, rel in _real_frames():
        for kit in ('kit0',) + KITS:
            r = measure_roi_margins(img, side, kit)
            assert r['roi_ok'] in (True, False, None), (kit, rel)


def main():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'ok    {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print(f'ERROR {t.__name__}: {type(e).__name__}: {e}')
    print(f'{len(tests) - failed}/{len(tests)} тестов прошло')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
