# -*- coding: utf-8 -*-
"""Минимальный клиент Telegram Bot API на стандартной библиотеке.

Своя сотня строк вместо python-telegram-bot или aiogram: бот живёт в том же
образе и том же установщике, что и сервис, и не тянет за собой асинхронный
стек и ещё десяток пакетов. Нужно немного: long polling, отправка текста,
картинок и файлов, скачивание присланного файла.

Токен нигде не печатается: он входит только в URL запроса, а тексты ошибок
проходят через `_scrub`.
"""
from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API = 'https://api.telegram.org'


class TelegramError(RuntimeError):
    def __init__(self, description: str, code: int = 0, retry_after: float = 0.0):
        super().__init__(description)
        self.code = code
        self.retry_after = retry_after


class TelegramAPI:
    def __init__(self, token: str, timeout: float = 30.0):
        self._token = token
        self.base = f'{API}/bot{token}/'
        self.file_base = f'{API}/file/bot{token}/'
        self.timeout = timeout

    def _scrub(self, text: str) -> str:
        return str(text).replace(self._token, '<token>')

    # ------------------------------------------------------------------ #
    def call(self, method: str, params: dict | None = None, files: dict | None = None,
             timeout: float | None = None, retries: int = 3):
        """Вызов метода API. files: {поле: (имя, bytes)}. Возвращает result."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        for attempt in range(retries + 1):
            try:
                if files:
                    body, ctype = _multipart(params, files)
                else:
                    body, ctype = json.dumps(params, ensure_ascii=False).encode('utf-8'), 'application/json'
                req = urllib.request.Request(self.base + method, data=body,
                                             headers={'Content-Type': ctype})
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    payload = json.loads(r.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                try:
                    payload = json.loads(e.read().decode('utf-8'))
                except Exception:
                    payload = {'ok': False, 'error_code': e.code, 'description': f'HTTP {e.code}'}
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt >= retries:
                    raise TelegramError(self._scrub(f'сеть: {e}')) from None
                time.sleep(min(2 ** attempt, 10))
                continue
            if payload.get('ok'):
                return payload.get('result')
            code = int(payload.get('error_code') or 0)
            retry_after = float((payload.get('parameters') or {}).get('retry_after') or 0)
            if code == 429 and attempt < retries:
                time.sleep(retry_after or 3)
                continue
            if code >= 500 and attempt < retries:
                time.sleep(min(2 ** attempt, 10))
                continue
            raise TelegramError(self._scrub(payload.get('description') or 'ошибка API'), code, retry_after)
        raise TelegramError('исчерпаны попытки')

    # ------------------------------------------------------------------ #
    def get_updates(self, offset: int | None, timeout: int = 50) -> list[dict]:
        return self.call('getUpdates', {'offset': offset, 'timeout': timeout,
                                        'allowed_updates': ['message', 'callback_query']},
                         timeout=timeout + 15, retries=0) or []

    def send_message(self, chat_id, text: str, reply_markup: dict | None = None,
                     reply_to: int | None = None) -> dict:
        return self.call('sendMessage', {
            'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML',
            'link_preview_options': {'is_disabled': True},
            'reply_markup': reply_markup,
            'reply_parameters': {'message_id': reply_to, 'allow_sending_without_reply': True}
            if reply_to else None})

    def edit_message(self, chat_id, message_id: int, text: str, reply_markup: dict | None = None):
        try:
            return self.call('editMessageText', {'chat_id': chat_id, 'message_id': message_id,
                                                 'text': text, 'parse_mode': 'HTML',
                                                 'link_preview_options': {'is_disabled': True},
                                                 'reply_markup': reply_markup})
        except TelegramError as e:
            if 'not modified' in str(e):
                return None
            raise

    def send_chat_action(self, chat_id, action: str = 'typing') -> None:
        try:
            self.call('sendChatAction', {'chat_id': chat_id, 'action': action}, retries=0)
        except TelegramError:
            pass

    def send_document(self, chat_id, path: Path, caption: str = '', name: str | None = None):
        return self.call('sendDocument', {'chat_id': chat_id, 'caption': caption or None,
                                          'parse_mode': 'HTML'},
                         files={'document': (name or Path(path).name, Path(path).read_bytes())},
                         timeout=120)

    def send_photo(self, chat_id, png: bytes, caption: str = ''):
        return self.call('sendPhoto', {'chat_id': chat_id, 'caption': caption or None,
                                       'parse_mode': 'HTML'},
                         files={'photo': ('frame.png', png)}, timeout=120)

    def send_media_group(self, chat_id, photos: list[tuple[bytes, str]]):
        """До 10 картинок одним альбомом: [(png, подпись), ...]."""
        media, files = [], {}
        for i, (png, caption) in enumerate(photos[:10]):
            key = f'p{i}'
            files[key] = (f'{key}.png', png)
            media.append({'type': 'photo', 'media': f'attach://{key}',
                          'caption': caption or '', 'parse_mode': 'HTML'})
        return self.call('sendMediaGroup', {'chat_id': chat_id, 'media': media},
                         files=files, timeout=180)

    def answer_callback(self, callback_id: str, text: str = '') -> None:
        try:
            self.call('answerCallbackQuery', {'callback_query_id': callback_id,
                                              'text': text or None}, retries=0)
        except TelegramError:
            pass

    def download(self, file_id: str, dest: Path) -> Path:
        info = self.call('getFile', {'file_id': file_id})
        url = self.file_base + info['file_path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as r, dest.open('wb') as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as e:
            raise TelegramError(self._scrub(f'не удалось скачать файл: {e}')) from None
        return dest


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                f'{v}\r\n').encode('utf-8')
    for k, (name, data) in files.items():
        ctype = mimetypes.guess_type(name)[0] or 'application/octet-stream'
        safe = name.replace('"', '_')
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; '
                f'filename="{safe}"\r\nContent-Type: {ctype}\r\n\r\n').encode('utf-8')
        out += data + b'\r\n'
    out += f'--{boundary}--\r\n'.encode('utf-8')
    return bytes(out), f'multipart/form-data; boundary={boundary}'
