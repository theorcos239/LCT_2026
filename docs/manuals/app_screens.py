# -*- coding: utf-8 -*-
"""Снимки экрана для инструкции к приложению.

    python docs/manuals/app_screens.py [--window] [--installer]

Интерфейс снимается в Chromium (playwright) с настоящего сервиса на
localhost: загрузка demo.zip, сводка, карточка снимка, фильтр, полноэкранный
просмотр, отметки, справка. Это та же страница, что открывается в окне
настольного приложения.

--window     дополнительно — окно установленного приложения (dist/DXA-QC/DXA-QC.exe)
--installer  дополнительно — окна мастера установки (dist/DXA-QC-Setup-*.exe)

Окна снимаются по одному через PrintWindow: в кадр не попадает ничего,
кроме самого окна.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'assets'
DEMO = ROOT / 'demo.zip'


def serve(port: int) -> subprocess.Popen:
    env = dict(os.environ, OMP_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    p = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'service.api:app', '--host', '127.0.0.1',
                          '--port', str(port)], cwd=str(ROOT), env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=60)
            return p
        except Exception:
            time.sleep(1)
    p.terminate()
    raise SystemExit('сервис не поднялся')


def ui_screens(port: int = 8792) -> None:
    from playwright.sync_api import sync_playwright
    proc = serve(port)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={'width': 1440, 'height': 900}, device_scale_factor=1.5)
            pg.goto(f'http://127.0.0.1:{port}/')
            pg.wait_for_function("document.querySelector('#health')?.textContent.includes('работает')",
                                 timeout=120000)
            pg.wait_for_timeout(500)
            pg.screenshot(path=str(OUT / 'app_start.png'))
            pg.locator('#fileInput').set_input_files(str(DEMO))
            try:
                pg.wait_for_function("!document.querySelector('#run').hidden", timeout=10000)
                pg.wait_for_timeout(1200)
                pg.screenshot(path=str(OUT / 'app_progress.png'))
            except Exception:
                pass
            pg.wait_for_function("document.querySelector('#stage')?.textContent === 'Готово'", timeout=600000)
            pg.wait_for_function("document.querySelectorAll('#rows tr').length > 3", timeout=60000)
            pg.wait_for_timeout(2500)
            pg.screenshot(path=str(OUT / 'app_results.png'))
            pg.locator('#side').screenshot(path=str(OUT / 'app_card.png'))
            pg.locator('#downloads').screenshot(path=str(OUT / 'app_downloads.png'))
            chip = pg.get_by_text('Ротация', exact=False).first
            if chip.count():
                chip.click()
                pg.wait_for_timeout(1500)
                pg.screenshot(path=str(OUT / 'app_filter.png'))
            pg.keyboard.press('a')                     # согласен
            pg.wait_for_timeout(600)
            pg.locator('#tableWrap').screenshot(path=str(OUT / 'app_marks.png'))
            pg.keyboard.press('f')                     # на весь экран
            pg.wait_for_timeout(1200)
            pg.screenshot(path=str(OUT / 'app_lightbox.png'))
            pg.keyboard.press('Escape')
            pg.wait_for_timeout(500)
            pg.keyboard.press('?')                     # справка
            pg.wait_for_timeout(800)
            pg.screenshot(path=str(OUT / 'app_help.png'))
            b.close()
    finally:
        proc.terminate()
    print('интерфейс снят')


# --------------------------------------------------------------------------- #
#  Окна Windows
# --------------------------------------------------------------------------- #
def _dpi_aware() -> None:
    """Без этого при масштабе Windows 125–150 % размер окна приходит в логических
    пикселях, а PrintWindow рисует в физических — снимок обрезается."""
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def capture_window(title_part: str, path: Path, timeout: float = 120) -> bool:
    """Снимок одного окна по части заголовка (PrintWindow с PW_RENDERFULLCONTENT)."""
    import ctypes

    import win32con
    import win32gui
    import win32ui
    from PIL import Image
    t0 = time.time()
    hwnd = 0
    while time.time() - t0 < timeout and not hwnd:
        found = []

        def cb(h, _):
            if win32gui.IsWindowVisible(h) and title_part in win32gui.GetWindowText(h):
                found.append(h)
        win32gui.EnumWindows(cb, None)
        hwnd = found[0] if found else 0
        if not hwnd:
            time.sleep(0.5)
    if not hwnd:
        return False
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception:
        pass
    time.sleep(1.0)
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetWindowDC(hwnd)
    mfc = win32ui.CreateDCFromHandle(hdc)
    save = mfc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc, w, h)
    save.SelectObject(bmp)
    ctypes.windll.user32.PrintWindow(hwnd, save.GetSafeHdc(), 2)
    info = bmp.GetInfo()
    img = Image.frombuffer('RGB', (info['bmWidth'], info['bmHeight']), bmp.GetBitmapBits(True), 'raw', 'BGRX', 0, 1)
    win32gui.DeleteObject(bmp.GetHandle())
    save.DeleteDC()
    mfc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hdc)
    img.save(path)
    print(path)
    return True


def window_screen() -> None:
    exe = ROOT / 'dist' / 'DXA-QC' / 'DXA-QC.exe'
    if not exe.exists():
        print('нет dist/DXA-QC — окно приложения не снято')
        return
    state = Path(os.environ.get('LOCALAPPDATA', '')) / 'DXA-QC' / 'instance.json'
    state.unlink(missing_ok=True)
    proc = subprocess.Popen([str(exe)])
    try:
        for _ in range(240):                                        # ждём загрузку моделей
            try:
                urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=60)
                break
            except Exception:
                time.sleep(1)
        time.sleep(4)
        capture_window('Контроль качества DXA', OUT / 'app_window.png')
    finally:
        subprocess.run(['taskkill', '/PID', str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def installer_screens() -> None:
    import win32con
    import win32gui
    setup = sorted((ROOT / 'dist').glob('DXA-QC-Setup-*.exe'))
    if not setup:
        print('нет установщика в dist/ — мастер не снят')
        return
    proc = subprocess.Popen([str(setup[-1]), '/LANG=ru', '/CURRENTUSER'])
    try:
        title = 'Установка'
        # экраны «Папка установки» и «Готово» не снимаем: на них путь с именем пользователя
        names = ['setup_info.png', None, 'setup_tasks.png']
        for k, name in enumerate(names):
            if name and not capture_window(title, OUT / name, timeout=60):
                break
            if not name:
                time.sleep(1.5)
            if k == len(names) - 1:
                break                         # «Установить» не нажимаем: только снимки мастера
            found = []
            win32gui.EnumWindows(lambda w, _: found.append(w) if title in win32gui.GetWindowText(w)
                                 and win32gui.IsWindowVisible(w) else None, None)
            if found:
                win32gui.PostMessage(found[0], win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
                win32gui.PostMessage(found[0], win32con.WM_KEYUP, win32con.VK_RETURN, 0)
            time.sleep(1.5)
    finally:
        proc.kill()                                                 # установка не выполняется
        subprocess.run(['taskkill', '/F', '/IM', 'DXA-QC-Setup-1.0.0.tmp'], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', action='store_true')
    ap.add_argument('--installer', action='store_true')
    ap.add_argument('--no-ui', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.no_ui:
        ui_screens()
    if a.window or a.installer:
        _dpi_aware()
    if a.window:
        window_screen()
    if a.installer:
        installer_screens()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
