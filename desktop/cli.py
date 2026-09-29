# -*- coding: utf-8 -*-
"""Консольная точка входа установленного приложения: dxa-qc-cli.exe.

    dxa-qc-cli batch <папка|zip|файл.dcm> [--out report.xlsx] [--overlays o.zip] [--sr sr.zip]
    dxa-qc-cli serve [--host 0.0.0.0] [--port 8000]   веб-приложение и API для локальной сети
    dxa-qc-cli bot [--check]                          Telegram-бот
    dxa-qc-cli gui                                    окно приложения
    dxa-qc-cli version

Из репозитория то же самое: python -m desktop <команда>.
"""
from __future__ import annotations

import argparse
import sys

USAGE = __doc__.split('\n\n')[1]


def serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog='dxa-qc-cli serve',
                                 description='Веб-интерфейс и HTTP API сервиса')
    ap.add_argument('--host', default='0.0.0.0',
                    help='0.0.0.0 — доступен из локальной сети; 127.0.0.1 — только с этого компьютера')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args(argv)
    import uvicorn

    from service import api
    print(f'веб-интерфейс: http://127.0.0.1:{a.port}/   API: http://127.0.0.1:{a.port}/docs')
    if a.host == '0.0.0.0':
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('10.255.255.255', 1))
                print(f'из локальной сети:  http://{s.getsockname()[0]}:{a.port}/')
        except OSError:
            pass
    print('остановить — Ctrl+C')
    uvicorn.run(api.app, host=a.host, port=a.port, log_level='info')
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print('Контроль качества DXA\n\n' + USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'batch':
        from service.cli import main as cli
        return cli(rest)
    if cmd == 'serve':
        return serve(rest)
    if cmd == 'bot':
        from bot.app import main as bot
        return bot(rest)
    if cmd == 'gui':
        from desktop.app import main as gui
        return gui(rest)
    if cmd in ('version', '--version'):
        from desktop.app import VERSION
        print(f'Контроль качества DXA {VERSION}')
        return 0
    # dxa-qc-cli <путь> — сокращение для batch
    from pathlib import Path
    if Path(cmd).exists():
        from service.cli import main as cli
        return cli(argv)
    print(f'неизвестная команда: {cmd}\n\n{USAGE}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
