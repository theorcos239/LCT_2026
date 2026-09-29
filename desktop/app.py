# -*- coding: utf-8 -*-
"""Настольное приложение: сервис внутри процесса и окно с веб-интерфейсом.

    python -m desktop              # из репозитория
    DXA-QC.exe                     # после установки

Как устроено. Сервис (тот же FastAPI, что в контейнере) поднимается в этом
процессе на свободном порту 127.0.0.1 — наружу он не виден, сеть не нужна.
Окно — WebView2 (движок Edge, есть в Windows 10/11 даже без браузера Edge)
через pywebview. Если WebView2 нет, интерфейс открывается в браузере в режиме
приложения, а маленькое окно управления держит сервис и завершает его.

Второй запуск не поднимает второй сервис: окно подключается к уже работающему.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

TITLE = 'Контроль качества DXA'
VERSION = '1.0.0'
log = logging.getLogger('dxa_qc.desktop')


def data_dir() -> Path:
    if os.name == 'nt':
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
        d = base / 'DXA-QC'
    else:
        d = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share') / 'dxa-qc'
    d.mkdir(parents=True, exist_ok=True)
    return d


def resource(*parts: str) -> Path:
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))
    return base.joinpath(*parts)


def build_info() -> dict:
    """Из какого коммита собрано приложение: build_info.json пишет
    desktop/build_windows.ps1 перед сборкой. Из исходников его нет."""
    try:
        return json.loads(resource('desktop', 'build_info.json').read_text(encoding='utf-8-sig'))
    except Exception:
        return {}


def version_text() -> str:
    b = build_info()
    if not b.get('commit'):
        return f'{VERSION} (из исходников)'
    dirty = ', с незакоммиченными изменениями' if b.get('dirty') else ''
    return f"{VERSION} (коммит {b['commit']}{dirty}, собрано {b.get('built', '?')})"


def setup_logging() -> Path:
    logs = data_dir() / 'logs'
    logs.mkdir(exist_ok=True)
    path = logs / 'desktop.log'
    handlers = [logging.FileHandler(path, encoding='utf-8')]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    wv = logging.getLogger('pywebview')          # у pywebview свой обработчик — пишем и в журнал
    wv.addHandler(handlers[0])
    # У оконного exe нет консоли: stdout/stderr = None, и любая печать упала бы.
    if sys.stdout is None or sys.stderr is None:
        stream = open(logs / 'console.log', 'a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    return path


# --------------------------------------------------------------------------- #
#  Сервис
# --------------------------------------------------------------------------- #
def free_port(preferred: int = 8765) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def healthy(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip('/') + '/health', timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(url: str, timeout: float = 600.0) -> bool:
    """Ждём, пока сервис загрузит модели (первый запуск после установки —
    до минуты: Windows проверяет новые файлы антивирусом)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if healthy(url, timeout=max(5.0, timeout)):
            return True
        time.sleep(0.5)
    return False


class Server:
    def __init__(self, port: int, host: str = '127.0.0.1'):
        import uvicorn

        from service import api
        self._cleanup_old_jobs(api.JOBS_DIR)
        cfg = uvicorn.Config(api.app, host=host, port=port, log_level='warning',
                             log_config=None, access_log=False)
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, name='uvicorn', daemon=True)
        self.url = f'http://127.0.0.1:{port}/'
        self.port = port

    @staticmethod
    def _cleanup_old_jobs(jobs_dir: Path, older_h: float = 24.0) -> None:
        # задания живут в памяти процесса; каталоги прошлых запусков — мусор
        if not jobs_dir.exists():
            return
        now = time.time()
        for d in jobs_dir.iterdir():
            try:
                if now - d.stat().st_mtime > older_h * 3600:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    def start(self) -> 'Server':
        self.thread.start()
        from service import api
        threading.Thread(target=api.analyzer, name='models', daemon=True).start()
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


STATE = 'instance.json'


def running_instance() -> str | None:
    try:
        st = json.loads((data_dir() / STATE).read_text(encoding='utf-8'))
        url = st.get('url')
        return url if url and healthy(url, timeout=1.5) else None
    except Exception:
        return None


def remember(url: str | None) -> None:
    p = data_dir() / STATE
    if url:
        p.write_text(json.dumps({'url': url, 'pid': os.getpid()}), encoding='utf-8')
    else:
        p.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
#  Окно
# --------------------------------------------------------------------------- #
LOADING = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
 html,body{{height:100%;margin:0;background:#0f1216;color:#e6edf3;font:15px/1.5 "Segoe UI",system-ui,sans-serif}}
 .c{{height:100%;display:grid;place-items:center;text-align:center}}
 h1{{font:600 22px "Segoe UI Semibold","Segoe UI",sans-serif;margin:18px 0 6px}}
 p{{color:#8b98a5;margin:0}}
 .s{{width:38px;height:38px;border:3px solid #26313b;border-top-color:#4493f8;border-radius:50%;
    margin:0 auto;animation:r 1s linear infinite}}
 @keyframes r{{to{{transform:rotate(360deg)}}}}
 @media (prefers-reduced-motion:reduce){{.s{{animation:none}}}}
</style>
<div class="c"><div><div class="s"></div><h1>{title}</h1>
<p>Загружаю модели. Первый запуск после установки занимает до минуты.</p></div></div>"""

FAILED = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{background:#0f1216;color:#e6edf3;font:15px/1.6 "Segoe UI",sans-serif;padding:48px}}
code{{background:#1b232c;padding:2px 6px;border-radius:4px}}</style>
<h1>Сервис не запустился</h1><p>Модели не загрузились за отведённое время.
Журнал работы: <code>{log}</code>. Закройте окно и запустите приложение ещё раз;
если не поможет — пришлите журнал команде проекта.</p>"""


def open_with_webview(url: str, server: Server | None, log_path: Path) -> None:
    import webview

    webview.settings['ALLOW_DOWNLOADS'] = True
    webview.settings['OPEN_EXTERNAL_LINKS_IN_BROWSER'] = True
    w = webview.create_window(TITLE, html=LOADING.format(title=TITLE), width=1440, height=900,
                              min_size=(1024, 640), background_color='#0f1216', text_select=True)

    def boot():
        if wait_ready(url):
            w.load_url(url)
        else:
            w.load_html(FAILED.format(title=TITLE, log=log_path))

    icon = resource('desktop', 'assets', 'icon.ico')
    webview.start(boot, gui='edgechromium' if os.name == 'nt' else None, private_mode=False,
                  storage_path=str(data_dir() / 'webview'),
                  icon=str(icon) if icon.exists() else None)
    log.info('окно закрыто')


def app_browser() -> str | None:
    """Браузер, умеющий --app (окно без адресной строки)."""
    cands = []
    if os.name == 'nt':
        pf = [os.environ.get(k) for k in ('PROGRAMFILES(X86)', 'PROGRAMFILES', 'LOCALAPPDATA')]
        rel = [r'Microsoft\Edge\Application\msedge.exe', r'Google\Chrome\Application\chrome.exe',
               r'Yandex\YandexBrowser\Application\browser.exe', r'Chromium\Application\chrome.exe']
        cands = [str(Path(b) / r) for b in pf if b for r in rel]
    else:
        cands = [shutil.which(n) or '' for n in ('microsoft-edge', 'google-chrome', 'chromium',
                                                 'chromium-browser', 'yandex-browser')]
        cands.append('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    return next((c for c in cands if c and Path(c).exists()), None)


def open_with_browser(url: str, server: Server | None) -> None:
    """Запасной путь без WebView2: браузер + окно управления."""
    ready = wait_ready(url)
    browser = app_browser()
    if ready:
        if browser:
            subprocess.Popen([browser, f'--app={url}', f'--user-data-dir={data_dir() / "browser"}',
                              '--no-first-run'])
        else:
            webbrowser.open(url)
    if server is None:
        return
    try:
        import tkinter as tk
    except Exception:
        log.info('tkinter недоступен: сервис работает до Ctrl+C')
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return
    root = tk.Tk()
    root.title(TITLE)
    root.geometry('460x170')
    root.resizable(False, False)
    msg = f'Сервис работает:\n{url}' if ready else 'Сервис не запустился — см. журнал работы.'
    tk.Label(root, text=msg, font=('Segoe UI', 11), pady=14).pack()
    row = tk.Frame(root)
    row.pack(pady=8)
    tk.Button(row, text='Открыть интерфейс', width=18,
              command=lambda: (subprocess.Popen([browser, f'--app={url}',
                                                 f'--user-data-dir={data_dir() / "browser"}'])
                               if browser else webbrowser.open(url))).pack(side='left', padx=6)
    tk.Button(row, text='Завершить', width=12, command=root.destroy).pack(side='left', padx=6)
    tk.Label(root, text='Закройте это окно, чтобы остановить сервис.', fg='#667').pack()
    root.mainloop()


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    log_path = setup_logging()
    log.info('%s %s, Python %s, frozen=%s', TITLE, version_text(), sys.version.split()[0],
             getattr(sys, 'frozen', False))
    url = running_instance()
    server = None
    if url:
        log.info('подключаюсь к уже работающему сервису %s', url)
    else:
        server = Server(free_port()).start()
        url = server.url
        remember(url)
        log.info('сервис: %s', url)
    try:
        try:
            open_with_webview(url, server, log_path)
        except Exception:
            log.exception('WebView2 недоступен — открываю в браузере')
            open_with_browser(url, server)
    finally:
        if server is not None:
            remember(None)
            server.stop()
            log.info('сервис остановлен')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
