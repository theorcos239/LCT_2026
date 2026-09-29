# -*- coding: utf-8 -*-
"""Telegram-бот сервиса контроля качества DXA.

    python -m bot                  # токен из DXA_QC_TG_TOKEN или файла .env
    python -m bot --check          # проверить токен и выйти

Бот работает на том компьютере, где запущен (ноутбук, сервер, контейнер):
получает сообщения long polling'ом, поэтому ни белый IP, ни HTTPS-сертификат
не нужны. Снимки анализирует тот же конвейер, что CLI, API и настольное
приложение, — вердикты совпадают побитово.

Что умеет: принимает DICOM-файлы (по одному, пачкой) и zip-архивы с
исследованиями, отвечает сводкой по каждому снимку, снимками с разметкой и
таблицей отчёта в формате ТЗ 2.5; по кнопкам — DICOM SR, измерения в JSON и
архив визуализаций.

Файлы проходят через серверы Telegram, поэтому бот — интерфейс для
демонстрации и обезличенных данных. Для клинической работы — настольное
приложение или контейнер: они работают без сети.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .telegram import TelegramAPI, TelegramError

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_MB = 20                 # getFile у Bot API отдаёт файлы до 20 МБ
RESULT_TTL_S = 3 * 3600          # сколько хранить результаты для кнопок
ALBUM_LIMIT = 10                 # sendMediaGroup: до 10 картинок
ALL_FRAMES_UP_TO = 10            # столько снимков показываем все, дальше — только с нарушением
GROUP_WAIT_S = 2.5               # файлы одного альбома приходят отдельными сообщениями

log = logging.getLogger('dxa_qc.bot')

REGION_SHORT = {'spine': 'Позвоночник, L1–L4', 'lh': 'Левое бедро', 'rh': 'Правое бедро',
                'unknown': 'Область не определена'}
CRITERION_SHORT = {'spine_position': 'укладка', 'spine_axis': 'ось позвоночника',
                   'spine_artifacts': 'посторонние предметы', 'hip_rotation': 'ротация бедра',
                   'hip_roi': 'отступы области интереса', 'undetermined': 'область не определена'}
FLAG_TEXT = {
    'estimates_disagree': 'две оценки критерия разошлись — проверьте глазами',
    'cnn_vs_geometry_disagree': 'нейросеть и измерение разошлись — проверьте глазами',
    'axis:curved': 'позвоночник изогнут (сколиоз); эксперт сколиоз нарушением оси не считает',
    'orientation:mirrored_lr': 'снимок выгружен зеркально и приведён к стандартному виду',
    'I:anatomical_prior': 'седалищная кость не найдена, нижний отступ взят по анатомии',
    'I:strict_mask': 'седалищная кость найдена по строгой маске',
    'cnn:unavailable': 'нейросеть недоступна, оценка только геометрией',
}

COMMANDS = [
    ('start', 'как пользоваться'),
    ('demo', 'пример на демо-исследованиях'),
    ('about', 'что проверяется и насколько точно'),
    ('help', 'форматы, ограничения, конфиденциальность'),
]

SHORT_DESCRIPTION = ('Контроль качества денситометрии (DXA): укладка, ось, артефакты, '
                     'ротация бедра, отступы ROI. Пришлите DICOM или zip.')
DESCRIPTION = ('Бот сервиса контроля качества DXA-исследований (ЛЦТ 2026). Пришлите файлом '
               'DICOM-снимок позвоночника или бедра, несколько снимков или zip-архив с '
               'исследованиями: бот определит область, проверит пять критериев ТЗ и ответит '
               'вердиктом, снимками с разметкой и таблицей отчёта.\n\n'
               'Бот — необязательное дополнение к сервису, демонстрация альтернативного интерфейса. '
               'Файлы проходят через Telegram: присылайте только обезличенные данные.')

START_TEXT = """<b>Контроль качества денситометрии (DXA)</b>

Пришлите <b>файлом</b> DICOM-снимок поясничного отдела позвоночника или бедра, несколько снимков сразу или zip-архив с исследованиями. Я определю область, проверю пять критериев ТЗ и пришлю:
• вердикт по каждому снимку с объяснением в миллиметрах и градусах;
• снимки с разметкой: найденные ориентиры, пороги, тепловая карта нейросети;
• таблицу отчёта в формате ТЗ 2.5, по кнопкам — DICOM SR и все измерения.

Как отправить: скрепка → «Файл». До {max_mb} МБ за раз — ограничение Telegram для ботов; большие наборы обрабатывайте в настольном приложении или контейнере.

/demo — пример на демо-исследованиях
/about — что проверяется и насколько точно
/help — форматы и ограничения

<i>Бот — приятный бонус к сервису, а не обязательная его часть. Файлы проходят через серверы Telegram. Присылайте только обезличенные данные: для работы с реальными пациентами есть настольное приложение и контейнер — они работают без сети.</i>"""

HELP_TEXT = """<b>Что можно прислать</b>
• один или несколько DICOM-файлов (.dcm или без расширения) — они считаются одним исследованием;
• zip-архив: одно исследование или папка с исследованиями, вложенность любая; дубликаты снимков схлопываются.

Размер — до {max_mb} МБ за сообщение. Фото и скриншоты не подходят: нужен исходный DICOM, отправленный как файл.

<b>Что приходит в ответ</b>
Сводка по каждому снимку, снимки с разметкой (до {album} штук альбомом) и таблица report.xlsx — восемь колонок ТЗ 2.5 и дальше вероятности, флаги, описания. Кнопки под сводкой: DICOM SR, измерения в JSON, архив всех визуализаций. Результаты хранятся {ttl} ч.

<b>Как читать</b>
«Нарушение» — сработал хотя бы один критерий, все сработавшие перечислены. Флаг — не нарушение, а пометка «здесь измерение менее надёжно, посмотрите сами».

<b>Конфиденциальность</b>
Бот работает на компьютере команды, файлы удаляются сразу после обработки, результаты — через {ttl} ч. Но по пути файлы проходят через серверы Telegram, поэтому только обезличенные данные.

Сервис — поддержка решения, а не медицинское изделие: окончательное решение за специалистом."""


def _esc(s) -> str:
    return html.escape(str(s), quote=False)


def plural(n: int, one: str, few: str, many: str) -> str:
    k = n % 100
    w = one if k % 10 == 1 and k != 11 else few if 2 <= k % 10 <= 4 and not 12 <= k <= 14 else many
    return f'{n} {w}'


# --------------------------------------------------------------------------- #
#  Токен и настройки
# --------------------------------------------------------------------------- #
def config_dir() -> Path:
    if os.name == 'nt':
        return Path(os.environ.get('APPDATA') or Path.home()) / 'DXA-QC'
    return Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'dxa-qc'


def _app_dirs() -> list[Path]:
    dirs = [Path.cwd(), ROOT]
    if getattr(sys, 'frozen', False):
        dirs.insert(0, Path(sys.executable).resolve().parent)
    return dirs


def _read_env_file(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def load_settings() -> dict:
    """Переменные окружения важнее файлов; файлы — .env рядом с программой или
    в корне репозитория, затем bot.env в папке настроек пользователя."""
    merged: dict = {}
    for f in [config_dir() / 'bot.env'] + [d / '.env' for d in reversed(_app_dirs())]:
        merged.update(_read_env_file(f))
    merged.update({k: v for k, v in os.environ.items() if k.startswith('DXA_QC_')})
    return merged


def ask_token() -> str | None:
    """Первый запуск установленного бота: спросить токен и сохранить."""
    if not sys.stdin or not sys.stdin.isatty():
        return None
    print('Токен бота не найден. Получите его у @BotFather в Telegram (/newbot)\n'
          'и вставьте сюда (Enter — выход):')
    token = input('> ').strip()
    if not token:
        return None
    try:
        me = TelegramAPI(token).call('getMe')
    except TelegramError as e:
        print(f'Токен не подошёл: {e}')
        return None
    path = config_dir() / 'bot.env'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'DXA_QC_TG_TOKEN={token}\n', encoding='utf-8')
    if os.name != 'nt':
        path.chmod(0o600)
    print(f'Бот @{me.get("username")} подключён, токен сохранён в {path}')
    return token


def find_demo(settings: dict) -> Path | None:
    cands = [Path(settings['DXA_QC_DEMO'])] if settings.get('DXA_QC_DEMO') else []
    cands += [d / 'demo.zip' for d in _app_dirs()]
    return next((p for p in cands if p.is_file()), None)


# --------------------------------------------------------------------------- #
#  Обработка
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    chat_id: int
    reply_to: int | None
    docs: list[dict] = field(default_factory=list)        # {'file_id','name','size'}
    local: list[Path] = field(default_factory=list)       # уже на диске (демо)
    label: str = ''


@dataclass
class Result:
    job: str
    out: Path
    rows: list
    images: list          # [(png, подпись, есть_нарушение)]
    created: float = field(default_factory=time.time)


def _criterion_texts(row: dict) -> list[tuple[str, str]]:
    """[(критерий, объяснение)] для сработавших критериев снимка."""
    det = row.get('details') or {}
    spine = det.get('spine') or {}
    where = {'spine_position': (spine.get('position') or {}).get('text'),
             'spine_axis': (spine.get('axis') or {}).get('text'),
             'spine_artifacts': (spine.get('artifacts') or {}).get('text'),
             'hip_rotation': (det.get('hip_rotation') or {}).get('text'),
             'hip_roi': (det.get('hip_roi') or {}).get('violation_text')}
    from service.pipeline import VIOLATIONS
    out = []
    for code in [c for c in str(row.get('violation_type') or '').split(';') if c]:
        name = CRITERION_SHORT.get(code, code)
        t = str(where.get(code) or VIOLATIONS.get(code, code)).strip()
        # Тексты критериев бывают с собственным префиксом («ротация: …») или уже
        # начинаются с названия («ось позвоночника отклонена…») — без повторов.
        stem = name.split()[0][:6].lower()                   # «посторонн», «ротаци», «ось»…
        head, sep, rest = t.partition(': ')
        if sep and len(head) < 30 and head.split()[0][:6].lower() == stem:
            t = rest
        if t.lower().startswith(stem):
            name = ''
        out.append((name, t))
    return out


def _flag_texts(row: dict) -> list[str]:
    out = []
    for f in [f for f in str(row.get('flags') or '').split(';') if f]:
        key = next((k for k in FLAG_TEXT if f == k or f.endswith(':' + k) or f.startswith(k)), None)
        out.append(FLAG_TEXT[key] if key else f)
    return list(dict.fromkeys(out))


def frame_caption(i: int, row: dict, limit: int = 1000) -> str:
    region = REGION_SHORT.get(row.get('anatomical_region'), row.get('anatomical_region'))
    if row.get('processing_status') != 'Success':
        return f'<b>{i}. {_esc(row.get("file_name") or "файл")}</b> — не обработан'
    bad = bool(row.get('quality_class'))
    p = row.get('quality_probability')
    head = f'<b>{i}. {_esc(region)}</b>'
    if bad:
        head += ' — нарушение'
        lines = [head] + [f'• {_esc(c)}: {_esc(t)}' if c else f'• {_esc(t)}' for c, t in _criterion_texts(row)]
    else:
        head += ' — качественное'
        lines = [head + (f' (вероятность нарушения {float(p):.0%})' if p not in (None, '') else '')]
    s = '\n'.join(lines)
    return s if len(s) <= limit else s[:limit - 1] + '…'


def summary_text(rows: list[dict], label: str = '') -> str:
    ok = [r for r in rows if r.get('processing_status') == 'Success']
    bad = [r for r in ok if r.get('quality_class')]
    studies = len({r.get('path_to_study') for r in rows})
    head = f'<b>Готово{": " + _esc(label) if label else ""}</b>\n'
    head += (f'Исследований {studies}, снимков {len(rows)}: с нарушением {len(bad)}, '
             f'качественных {len(ok) - len(bad)}')
    failed = len(rows) - len(ok)
    if failed:
        head += f', не обработано {failed}'
    head += '.'
    parts, shown = [head], rows if len(rows) <= 25 else [r for r in rows if r.get('quality_class')]
    if len(shown) < len(rows):
        parts.append(f'Снимков много — ниже только {len(shown)} с нарушением или ошибкой; '
                     'остальные в таблице.')
    number = {id(r): i for i, r in enumerate(rows, 1)}     # те же номера, что в подписях снимков
    by_study: dict = {}
    for r in shown:
        by_study.setdefault(r.get('path_to_study'), []).append(r)
    for study, rs in by_study.items():
        if studies > 1:
            parts.append(f'<b>Исследование</b> <code>{_esc(str(study)[-40:])}</code>')
        for r in rs:
            block = [frame_caption(number[id(r)], r, limit=3000)]
            if r.get('processing_status') != 'Success' and r.get('error'):
                block.append(f'<i>{_esc(str(r.get("error"))[:300])}</i>')
            flags = _flag_texts(r)
            if flags and r.get('processing_status') == 'Success':
                block.append('<i>флаги: ' + _esc('; '.join(flags)) + '</i>')
            parts.append('\n'.join(block))
    return '\n\n'.join(parts)


def split_message(text: str, limit: int = 4000) -> list[str]:
    chunks, cur = [], ''
    for block in text.split('\n\n'):
        if len(cur) + len(block) + 2 > limit and cur:
            chunks.append(cur)
            cur = ''
        cur = f'{cur}\n\n{block}' if cur else block
    if cur:
        chunks.append(cur)
    return [c[:limit] for c in chunks]


class Processor:
    """Обёртка над конвейером сервиса: файлы -> отчёт, картинки, SR."""

    def __init__(self, analyzer=None):
        self._analyzer = analyzer

    @property
    def analyzer(self):
        if self._analyzer is None:
            from service.pipeline import Analyzer
            self._analyzer = Analyzer.load(
                cnn=os.environ.get('DXA_QC_CNN', '1') not in ('0', 'false'),
                roi_rule=os.environ.get('DXA_QC_ROI_RULE', 'scan_length'))
        return self._analyzer

    def run(self, files: list[Path], workdir: Path) -> Result:
        from service.dicom_io import extract_archive, study_dirs, unique_frames
        from service.overlay import render
        from service.pipeline import process_batch
        from service.report import write_details, write_overlays, write_sr, write_table

        data, out = workdir / 'data', workdir / 'out'
        data.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        loose = data / 'исследование'
        for i, f in enumerate(files):
            if f.suffix.lower() == '.zip':
                extract_archive(f, data / (f.stem or f'архив_{i}'))
            else:
                loose.mkdir(exist_ok=True)
                shutil.copy2(f, loose / f.name)
        rows = process_batch(data, self.analyzer)
        write_table(rows, out / 'report.xlsx')
        write_details(rows, out / 'details.json')
        frames = {}
        for s in study_dirs(data):
            for fr in unique_frames(s)[0]:
                frames[fr.image_uid] = fr
        write_overlays(rows, frames, out / 'overlays.zip', only_violations=False)
        write_sr(rows, frames, out / 'sr.zip')

        images = []
        ok = [r for r in rows if r.get('processing_status') == 'Success']
        pick = {id(r) for r in ok if len(ok) <= ALL_FRAMES_UP_TO or r.get('quality_class')}
        for i, r in enumerate(rows, 1):
            if id(r) not in pick or len(images) >= ALBUM_LIMIT:
                continue
            fr = frames.get(r.get('image_uid'))
            if fr is None:
                continue
            try:
                import io
                buf = io.BytesIO()
                render(fr.pixels, r, scale=3).save(buf, format='PNG')
                images.append((buf.getvalue(), frame_caption(i, r), bool(r.get('quality_class'))))
            except Exception:
                log.exception('визуализация не построена')
        shutil.rmtree(data, ignore_errors=True)       # пиксели больше не нужны
        return Result(job=workdir.name, out=out, rows=rows, images=images)


# --------------------------------------------------------------------------- #
#  Бот
# --------------------------------------------------------------------------- #
class Bot:
    def __init__(self, api: TelegramAPI, processor: Processor, workroot: Path,
                 allowed: set[str] | None = None, demo: Path | None = None):
        self.api = api
        self.proc = processor
        self.workroot = workroot
        self.allowed = allowed or set()
        self.demo = demo
        self.tasks: queue.Queue[Task] = queue.Queue()
        self.results: dict[str, Result] = {}
        self.groups: dict[tuple, dict] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.username = ''
        self.started = time.time()

    # ----------------------------------------------------------------- setup
    def setup(self) -> dict:
        me = self.api.call('getMe')
        self.username = me.get('username', '')
        for method, params in (
                ('setMyCommands', {'commands': [{'command': c, 'description': d} for c, d in COMMANDS]}),
                ('setMyShortDescription', {'short_description': SHORT_DESCRIPTION}),
                ('setMyDescription', {'description': DESCRIPTION})):
            try:
                self.api.call(method, params, retries=1)
            except TelegramError as e:
                log.warning('%s: %s', method, e)
        return me

    # --------------------------------------------------------------- polling
    def run(self) -> None:
        workers = [threading.Thread(target=self._worker, name='worker', daemon=True),
                   threading.Thread(target=self._flusher, name='albums', daemon=True)]
        for t in workers:
            t.start()
        offset = None
        log.info('бот @%s слушает сообщения (Ctrl+C — остановить)', self.username)
        while not self.stop.is_set():
            try:
                updates = self.api.get_updates(offset, timeout=50)
            except TelegramError as e:
                if e.code == 409:
                    log.error('этот токен уже слушает другой экземпляр бота: %s', e)
                    time.sleep(10)
                else:
                    log.warning('getUpdates: %s', e)
                    time.sleep(3)
                continue
            for u in updates:
                offset = u['update_id'] + 1
                try:
                    self.handle(u)
                except Exception:
                    log.exception('ошибка обработки сообщения')

    def handle(self, u: dict) -> None:
        if 'callback_query' in u:
            return self._on_callback(u['callback_query'])
        m = u.get('message')
        if not m:
            return
        chat, user = m['chat']['id'], m.get('from') or {}
        if not self._allowed(user):
            self.api.send_message(chat, 'Доступ к этому боту ограничен. Обратитесь к команде проекта.')
            return
        text = (m.get('text') or '').strip()
        if text.startswith('/'):
            cmd = text.split()[0][1:].split('@')[0].lower()
            return self._on_command(chat, cmd, m)
        if 'document' in m:
            return self._on_document(chat, m)
        if 'photo' in m:
            self.api.send_message(chat, 'Это фото — Telegram сжимает его, и DICOM-данных в нём нет. '
                                        'Пришлите исходный DICOM-файл или zip как <b>файл</b>: '
                                        'скрепка → «Файл».', reply_to=m['message_id'])
            return
        if text:
            self.api.send_message(chat, 'Пришлите DICOM-файл или zip-архив с исследованиями '
                                        '(скрепка → «Файл»). Подробнее — /help.')

    def _allowed(self, user: dict) -> bool:
        if not self.allowed:
            return True
        return (str(user.get('id')) in self.allowed
                or ('@' + str(user.get('username') or '').lower()) in self.allowed)

    def _on_command(self, chat: int, cmd: str, m: dict) -> None:
        if cmd in ('start', 'help'):
            tpl = START_TEXT if cmd == 'start' else HELP_TEXT
            self.api.send_message(chat, tpl.format(max_mb=MAX_FILE_MB, album=ALBUM_LIMIT,
                                                   ttl=RESULT_TTL_S // 3600))
        elif cmd == 'about':
            self.api.send_message(chat, about_text())
        elif cmd == 'demo':
            if self.demo is None:
                self.api.send_message(chat, 'Демо-архива на этом компьютере нет. Пришлите свои '
                                            'DICOM-файлы или zip — обработаю их.')
                return
            self._enqueue(Task(chat_id=chat, reply_to=m['message_id'], local=[self.demo],
                               label='демо-исследования'))
        elif cmd == 'status':
            up = int(time.time() - self.started)
            self.api.send_message(chat, f'Работаю {up // 3600} ч {up % 3600 // 60} мин, '
                                        f'в очереди {self.tasks.qsize()}.')
        else:
            self.api.send_message(chat, 'Не знаю такой команды. Список — /help.')

    def _on_document(self, chat: int, m: dict) -> None:
        d = m['document']
        name = d.get('file_name') or 'file.dcm'
        size = int(d.get('file_size') or 0)
        if size > MAX_FILE_MB * 1024 * 1024:
            self.api.send_message(
                chat, f'Файл {_esc(name)} весит {size / 2 ** 20:.0f} МБ, а бот может скачать не '
                      f'больше {MAX_FILE_MB} МБ — это ограничение Telegram. Разделите архив на части '
                      'или обработайте набор в настольном приложении или контейнере.',
                reply_to=m['message_id'])
            return
        doc = {'file_id': d['file_id'], 'name': name, 'size': size}
        mg = m.get('media_group_id')
        if not mg:
            self._enqueue(Task(chat_id=chat, reply_to=m['message_id'], docs=[doc], label=name))
            return
        with self.lock:                       # альбом: ждём остальные файлы
            g = self.groups.setdefault((chat, mg), {'docs': [], 'reply_to': m['message_id']})
            g['docs'].append(doc)
            g['deadline'] = time.time() + GROUP_WAIT_S

    def _flusher(self) -> None:
        while not self.stop.is_set():
            time.sleep(0.5)
            now, ready = time.time(), []
            with self.lock:
                for key, g in list(self.groups.items()):
                    if g['deadline'] <= now:
                        ready.append((key, self.groups.pop(key)))
            for (chat, _), g in ready:
                self._enqueue(Task(chat_id=chat, reply_to=g['reply_to'], docs=g['docs'],
                                   label=plural(len(g['docs']), 'файл', 'файла', 'файлов')))

    def _enqueue(self, task: Task) -> None:
        ahead = self.tasks.qsize()
        self.tasks.put(task)
        if ahead:
            self.api.send_message(task.chat_id, f'Принято. Перед вами в очереди: {ahead}.',
                                  reply_to=task.reply_to)

    # ---------------------------------------------------------------- worker
    def _worker(self) -> None:
        try:
            self.proc.analyzer                 # веса — до первого файла, а не на нём
            log.info('модели загружены')
        except Exception:
            log.exception('модели не загрузились')
        while not self.stop.is_set():
            try:
                task = self.tasks.get(timeout=1)
            except queue.Empty:
                self._cleanup()
                continue
            try:
                self._process(task)
            except Exception as e:
                log.exception('задание не выполнено')
                try:
                    self.api.send_message(task.chat_id, f'Не удалось обработать: {_esc(e)}',
                                          reply_to=task.reply_to)
                except TelegramError:
                    pass

    def _process(self, task: Task) -> None:
        job = uuid.uuid4().hex[:10]
        workdir = self.workroot / job
        inbox = workdir / 'in'
        inbox.mkdir(parents=True, exist_ok=True)
        n = len(task.docs) or len(task.local)
        label = task.label or plural(n, 'файл', 'файла', 'файлов')
        status = self.api.send_message(task.chat_id, f'Обрабатываю: {_esc(label)}…',
                                       reply_to=task.reply_to)
        self.api.send_chat_action(task.chat_id, 'typing')
        t0 = time.perf_counter()
        files = list(task.local)
        for i, d in enumerate(task.docs):
            safe = Path(d['name']).name or f'file_{i}.dcm'
            dest = inbox / f'{i:03d}_{safe}'
            files.append(self.api.download(d['file_id'], dest))
        try:
            res = self.proc.run(files, workdir)
        finally:
            shutil.rmtree(inbox, ignore_errors=True)
        res.job = job
        with self.lock:
            self.results[job] = res
        dt = time.perf_counter() - t0

        text = summary_text(res.rows, task.label) + f'\n\n<i>{dt:.1f} с</i>'
        chunks = split_message(text)
        kb = {'inline_keyboard': [
            [{'text': 'DICOM SR (.zip)', 'callback_data': f'sr:{job}'},
             {'text': 'Измерения (.json)', 'callback_data': f'js:{job}'}],
            [{'text': 'Все снимки с разметкой (.zip)', 'callback_data': f'ov:{job}'}]]}
        for i, c in enumerate(chunks):
            last = i == len(chunks) - 1
            if i == 0 and status:
                self.api.edit_message(task.chat_id, status['message_id'], c,
                                      reply_markup=kb if last else None)
            else:
                self.api.send_message(task.chat_id, c, reply_markup=kb if last else None)
        if res.images:
            self.api.send_chat_action(task.chat_id, 'upload_photo')
            if len(res.images) == 1:
                self.api.send_photo(task.chat_id, res.images[0][0], res.images[0][1])
            else:
                self.api.send_media_group(task.chat_id, [(png, cap) for png, cap, _ in res.images])
        self.api.send_chat_action(task.chat_id, 'upload_document')
        self.api.send_document(task.chat_id, res.out / 'report.xlsx',
                               caption='Отчёт: восемь колонок ТЗ 2.5, дальше вероятности, флаги и '
                                       'описания нарушений.', name=f'report_{job}.xlsx')
        log.info('задание %s: %d снимков за %.1f с', job, len(res.rows), dt)

    def _on_callback(self, q: dict) -> None:
        if not self._allowed(q.get('from') or {}):
            self.api.answer_callback(q['id'], 'Доступ ограничен.')
            return
        kind, _, job = (q.get('data') or '').partition(':')
        chat = (q.get('message') or {}).get('chat', {}).get('id')
        with self.lock:
            res = self.results.get(job)
        if res is None or chat is None:
            self.api.answer_callback(q['id'], 'Результаты уже удалены — пришлите файлы ещё раз.')
            return
        files = {'sr': ('sr.zip', f'sr_{job}.zip', 'DICOM SR: текстовое заключение по каждому снимку '
                                                   '(Basic Text SR), ссылается на исходный снимок.'),
                 'js': ('details.json', f'details_{job}.json', 'Все измерения: углы, миллиметры, '
                                                               'вероятности, флаги.'),
                 'ov': ('overlays.zip', f'overlays_{job}.zip', 'Все снимки с разметкой вердикта.')}
        if kind not in files:
            self.api.answer_callback(q['id'])
            return
        src, name, caption = files[kind]
        self.api.answer_callback(q['id'], 'Отправляю…')
        self.api.send_document(chat, res.out / src, caption=caption, name=name)

    def _cleanup(self) -> None:
        now = time.time()
        with self.lock:
            old = [j for j, r in self.results.items() if now - r.created > RESULT_TTL_S]
            for j in old:
                shutil.rmtree(self.workroot / j, ignore_errors=True)
                self.results.pop(j, None)


def about_text() -> str:
    lines = ['<b>Что проверяется</b> (ТЗ 2.3)',
             '• позвоночник: укладка (видны гребни подвздошных костей), ось не более 5°, '
             'посторонние предметы;',
             '• бедро: ротация (малый вертел), отступы области интереса до края поля.',
             'Область снимка определяется автоматически; всё, что не похоже на DXA, уходит '
             'на ручной разбор.', '',
             '<b>Насколько точно</b> — out-of-fold на 252 снимках обучающего набора, эталон — '
             'разметка эксперта:']
    try:
        m = json.loads((ROOT / 'service' / 'metrics.json').read_text(encoding='utf-8'))['honest_oof']
        names = [('spine_position', 'укладка'), ('spine_axis', 'ось'),
                 ('spine_artifacts', 'посторонние предметы'), ('hip_rotation', 'ротация'),
                 ('hip_roi', 'отступы ROI')]
        rows = [f'{n:<22}{m["per_violation"][k]["f1"]:>5.2f}{m["per_violation"][k]["roc_auc"]:>9.2f}'
                for k, n in names]
        o = m['overall']['binary_quality_class']
        rows.append(f'{"всего":<22}{o["f1"]:>5.2f}{o["roc_auc"]:>9.2f}')
        lines.append('<pre>' + _esc(f'{"":<22}{"F1":>5}{"ROC-AUC":>9}\n' + '\n'.join(rows)) + '</pre>')
    except Exception:
        lines.append('таблица метрик недоступна в этой сборке — см. README репозитория.')
    lines += ['Область снимка определяется верно на 252 из 252. Отступы ROI оцениваются так же, '
              'как это делает эксперт, — по длине поля сканирования.', '',
              'Сервис — поддержка решения, а не медицинское изделие.']
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='bot', description='Telegram-бот контроля качества DXA')
    ap.add_argument('--check', action='store_true', help='проверить токен и выйти')
    ap.add_argument('--workdir', type=Path, help='каталог для временных файлов')
    ap.add_argument('--log', type=Path, help='дописывать журнал ещё и в этот файл (UTF-8)')
    args = ap.parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding='utf-8'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=handlers)
    settings = load_settings()
    token = settings.get('DXA_QC_TG_TOKEN') or ask_token()
    if not token:
        print('Нет токена. Задайте DXA_QC_TG_TOKEN в окружении или в файле .env '
              '(шаблон — .env.example).', file=sys.stderr)
        return 2
    for k in ('DXA_QC_CNN', 'DXA_QC_ROI_RULE'):
        if k in settings:
            os.environ.setdefault(k, settings[k])

    api = TelegramAPI(token)
    try:
        me = api.call('getMe')
    except TelegramError as e:
        print(f'Токен не принят Telegram: {e}', file=sys.stderr)
        return 2
    if args.check:
        print(f'токен рабочий: @{me.get("username")}')
        return 0

    allowed = {a.strip().lower() for a in settings.get('DXA_QC_TG_ALLOWED', '').split(',') if a.strip()}
    workroot = args.workdir or Path(tempfile.gettempdir()) / 'dxa_qc_bot'
    shutil.rmtree(workroot, ignore_errors=True)
    workroot.mkdir(parents=True, exist_ok=True)
    bot = Bot(api, Processor(), workroot, allowed=allowed, demo=find_demo(settings))
    bot.setup()
    if bot.demo:
        log.info('демо-архив: %s', bot.demo)
    if allowed:
        log.info('доступ только для: %s', ', '.join(sorted(allowed)))
    try:
        bot.run()
    except KeyboardInterrupt:
        log.info('остановлен')
    finally:
        bot.stop.set()
        shutil.rmtree(workroot, ignore_errors=True)
    return 0
