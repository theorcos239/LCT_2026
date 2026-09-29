# -*- coding: utf-8 -*-
"""Картинки ответов Telegram-бота для инструкции и презентации.

    python docs/manuals/bot_screens.py

Бот прогоняется по-настоящему — тот же конвейер и те же тексты, что уходят в
Telegram, — только сетевой клиент подменён записывающей заглушкой
(bot/test_bot.py). Записанные сообщения, снимки и кнопки рисуются в
нейтральном макете чата через Chromium (playwright). Это не скриншот
Telegram, а отрисовка настоящих ответов бота.

Нужны: обучающий набор (НД_для_обучения/) и playwright с chromium.
"""
from __future__ import annotations

import base64
import html
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from bot.app import ALBUM_LIMIT, HELP_TEXT, MAX_FILE_MB, RESULT_TTL_S, START_TEXT, Bot, Processor  # noqa: E402
from bot.test_bot import FakeAPI  # noqa: E402

OUT = HERE / 'assets'
STUDY = '2.25.102755089973625799055786462646268820450'   # норма, ось и ротация в одном исследовании
DATA = ROOT / 'НД_для_обучения' / 'Исследования'
ICON = ROOT / 'desktop' / 'assets' / 'icon.png'

CSS = """
*{box-sizing:border-box} body{margin:0;background:#EDE8F5;font:15px/1.38 "Segoe UI",system-ui,sans-serif;color:#1C1D22}
.phone{width:420px;background:#EDE8F5}
.head{display:flex;align-items:center;gap:10px;padding:12px 14px;background:#fff;border-bottom:1px solid #DCD5EA}
.head img{width:38px;height:38px;border-radius:50%}
.head b{display:block;font-size:15.5px} .head span{font-size:12.5px;color:#8B8599}
.chat{padding:12px 10px 16px;display:flex;flex-direction:column;gap:8px}
.msg{max-width:88%;padding:8px 11px 7px;border-radius:14px;background:#fff;box-shadow:0 1px 1px #0000000d;word-wrap:break-word}
.bot{align-self:flex-start;border-bottom-left-radius:4px}
.me{align-self:flex-end;background:#E3D8F6;border-bottom-right-radius:4px}
.msg code{font:13px Consolas,monospace;background:#F3EFF9;padding:0 3px;border-radius:3px}
.msg pre{font:12.5px Consolas,monospace;margin:4px 0;white-space:pre}
.kb{align-self:flex-start;width:88%;display:grid;gap:4px;margin-top:-4px}
.kb .row{display:grid;gap:4px} .kb .row.two{grid-template-columns:1fr 1fr}
.kb div.b{background:#5209771f;color:#520977;text-align:center;padding:8px 6px;border-radius:9px;font-size:13.5px;font-weight:600}
.album{align-self:flex-start;width:88%;display:grid;gap:3px;border-radius:14px;overflow:hidden;background:#0d0a12}
.album.n3{grid-template-columns:1fr 1fr 1fr} .album.n2{grid-template-columns:1fr 1fr} .album.n1{grid-template-columns:1fr}
.album img{width:100%;height:190px;object-fit:cover;display:block}
.cap{align-self:flex-start;max-width:88%;font-size:13.5px}
.doc{display:flex;gap:10px;align-items:center}
.doc .ic{width:42px;height:42px;border-radius:50%;background:#520977;color:#fff;display:grid;place-items:center;font-weight:700;font-size:12px}
.doc small{color:#8B8599;display:block}
.time{font-size:11px;color:#8B8599;text-align:right;margin-top:2px}
.sys{align-self:center;background:#0000001a;color:#fff;font-size:12.5px;padding:2px 10px;border-radius:10px}
"""


def _b64(data: bytes, mime='image/png') -> str:
    return f'data:{mime};base64,' + base64.b64encode(data).decode()


def _txt(s: str) -> str:
    return s.replace('\n', '<br>')


class Chat:
    def __init__(self):
        self.parts: list[str] = []

    def me(self, content: str, t='10:24'):
        self.parts.append(f'<div class="msg me">{content}<div class="time">{t}</div></div>')

    def bot(self, content_html: str, t='10:24'):
        self.parts.append(f'<div class="msg bot">{_txt(content_html)}<div class="time">{t}</div></div>')

    def buttons(self, kb: dict):
        rows = []
        for row in kb['inline_keyboard']:
            cls = 'row two' if len(row) == 2 else 'row'
            rows.append(f'<div class="{cls}">' + ''.join(f'<div class="b">{html.escape(b["text"])}</div>'
                                                        for b in row) + '</div>')
        self.parts.append('<div class="kb">' + ''.join(rows) + '</div>')

    def album(self, pngs: list[bytes], captions: list[str]):
        n = min(len(pngs), 3)
        imgs = ''.join(f'<img src="{_b64(p)}">' for p in pngs)
        self.parts.append(f'<div class="album n{n}">{imgs}</div>')

    def doc(self, name: str, size_kb: int, caption: str = '', me=False):
        cls = 'me' if me else 'bot'
        cap = f'<div style="margin-top:6px">{_txt(caption)}</div>' if caption else ''
        self.parts.append(f'<div class="msg {cls}"><div class="doc"><div class="ic">{Path(name).suffix.lstrip(".").upper()[:4]}</div>'
                          f'<div><b>{html.escape(name)}</b><small>{size_kb} КБ</small></div></div>{cap}'
                          f'<div class="time">10:24</div></div>')

    def html(self) -> str:
        icon = _b64(ICON.read_bytes())
        return (f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><div class="phone">'
                f'<div class="head"><img src="{icon}"><div><b>Контроль качества DXA</b>'
                f'<span>бот · @LCT_bone_density_research_bot</span></div></div>'
                f'<div class="chat">{"".join(self.parts)}</div></div>')


def shoot(page, chat: Chat, path: Path):
    page.set_content(chat.html())
    page.wait_for_timeout(200)
    page.locator('.phone').screenshot(path=str(path))
    print(path)


def main() -> int:
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='dxa_bot_screens_'))
    try:
        arc = tmp / 'исследование.zip'
        with zipfile.ZipFile(arc, 'w') as z:
            for f in sorted((DATA / STUDY).rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(DATA))
        api = FakeAPI({'f_zip': arc})
        bot = Bot(api, Processor(), tmp / 'work')
        bot.setup()

        # 1. /start
        c1 = Chat()
        c1.sys = None
        c1.me('/start', '10:21')
        c1.bot(START_TEXT.format(max_mb=MAX_FILE_MB, album=ALBUM_LIMIT, ttl=RESULT_TTL_S // 3600), '10:21')

        # 2. файл -> результат
        n0 = len(api.calls)
        bot.handle({'update_id': 1, 'message': {'message_id': 10, 'chat': {'id': 1}, 'from': {'id': 1},
                                                'document': {'file_id': 'f_zip', 'file_name': arc.name,
                                                             'file_size': arc.stat().st_size}}})
        bot._process(bot.tasks.get_nowait())
        calls = api.calls[n0:]
        c2 = Chat()
        c2.doc(arc.name, arc.stat().st_size // 1024, me=True)
        edit = next(c for c in calls if c[0] == 'editMessageText')
        c2.bot(edit[1]['text'].replace('\n\n<i>', '\n\n<i>'))
        c2.buttons(edit[1]['reply_markup'])
        album = next((c for c in calls if c[0] in ('sendMediaGroup', 'sendPhoto')), None)
        if album and album[0] == 'sendMediaGroup':
            pngs = [f[1] for f in album[2].values()]
            caps = [m['caption'] for m in album[1]['media']]
            c2.album(pngs, caps)
            c2.parts.append(f'<div class="msg bot cap">{_txt(caps[0])}</div>')
        report = next(c for c in calls if c[0] == 'sendDocument')
        name, data = report[2]['document']
        c2.doc(name, max(1, len(data) // 1024), report[1].get('caption', ''))

        # 3. кнопки: SR и JSON
        job = next(iter(bot.results))
        c3 = Chat()
        n1 = len(api.calls)
        for kind in ('sr', 'js'):
            bot.handle({'update_id': 2, 'callback_query': {'id': 'q', 'data': f'{kind}:{job}', 'from': {'id': 1},
                                                           'message': {'chat': {'id': 1}}}})
        for c in api.calls[n1:]:
            if c[0] == 'sendDocument':
                nm, dt = c[2]['document']
                c3.doc(nm, max(1, len(dt) // 1024), c[1].get('caption', ''))
        c4 = Chat()
        c4.me('/help', '10:30')
        c4.bot(HELP_TEXT.format(max_mb=MAX_FILE_MB, album=ALBUM_LIMIT, ttl=RESULT_TTL_S // 3600), '10:30')

        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={'width': 420, 'height': 900}, device_scale_factor=2)
            shoot(page, c1, OUT / 'bot_start.png')
            shoot(page, c2, OUT / 'bot_result.png')
            shoot(page, c3, OUT / 'bot_buttons.png')
            shoot(page, c4, OUT / 'bot_help.png')
            b.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
