"""Показать только локальную страницу разметки iPad в той же Wi-Fi сети.

Запуск: python3 tools/serve_crop_annotator.py
Откройте напечатанный URL в Safari iPad, подставив IP компьютера вместо HOST.
Ответы сервера не содержат других файлов репозитория; JSON сохраняется на iPad.
"""
from __future__ import annotations

import argparse
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_HTML = Path(__file__).resolve().parents[1] / "Processing/crop_annotator.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    page = args.html.read_bytes()
    token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != f"/{token}":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(page)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Откройте в Safari iPad: http://IP-КОМПЬЮТЕРА:{args.port}/{token}")
    print("Компьютер и iPad должны быть в одной Wi-Fi сети. Остановить: Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
