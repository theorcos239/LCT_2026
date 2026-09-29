# -*- coding: utf-8 -*-
"""Тесты Telegram-бота без сети: Telegram подменён записывающей заглушкой.

    python -m bot.test_bot

Конвейер настоящий: снимки берутся из обучающего набора, если он лежит рядом
(НД_для_обучения/), иначе — синтетический DICOM.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.app import Bot, Processor, frame_caption, plural, split_message, summary_text  # noqa: E402
from bot.telegram import TelegramAPI, _multipart  # noqa: E402

DATA = ROOT / 'НД_для_обучения' / 'Исследования'
PASSED, FAILED = [], []


def check(name, cond, info=''):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{' — ' + str(info) if info else ''}")


class FakeAPI(TelegramAPI):
    """Записывает вызовы; download берёт файл с диска по file_id."""

    def __init__(self, files: dict[str, Path]):
        super().__init__('123:TEST')
        self.files = files
        self.calls: list[tuple[str, dict, dict]] = []
        self._mid = 100

    def call(self, method, params=None, files=None, timeout=None, retries=3):
        self.calls.append((method, dict(params or {}), dict(files or {})))
        if method == 'getMe':
            return {'id': 1, 'username': 'test_bot'}
        if method == 'getFile':
            return {'file_path': params['file_id']}
        self._mid += 1
        return {'message_id': self._mid}

    def download(self, file_id, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.files[file_id], dest)
        return dest

    def sent(self, method):
        return [c for c in self.calls if c[0] == method]

    def texts(self):
        return [c[1].get('text', '') for c in self.calls if c[0] in ('sendMessage', 'editMessageText')]


def sample_study() -> Path | None:
    if not DATA.exists():
        return None
    for s in sorted(DATA.iterdir()):
        if s.is_dir() and len(list(s.rglob('*.dcm'))) >= 3:
            return s
    return None


def synthetic_dicom(path: Path) -> Path:
    import numpy as np
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality, ds.PatientID = 'OT', 'TEST'
    px = (np.random.default_rng(0).random((200, 160)) * 255).astype('uint8')
    ds.Rows, ds.Columns = px.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = px.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


def update(uid, **msg):
    base = {'message_id': uid, 'chat': {'id': 42}, 'from': {'id': 7, 'username': 'tester'}}
    base.update(msg)
    return {'update_id': uid, 'message': base}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='dxa_bot_test_'))
    try:
        study = sample_study()
        if study is not None:
            dcms = sorted(study.rglob('*.dcm'))
        else:
            dcms = [synthetic_dicom(tmp / 'synthetic.dcm')]
        arc = tmp / 'study.zip'
        with zipfile.ZipFile(arc, 'w') as z:
            for f in dcms:
                z.write(f, f.relative_to(dcms[0].parents[1] if study else tmp))
        files = {'f_dcm': dcms[0], 'f_zip': arc, 'f_a': dcms[0], 'f_b': dcms[-1]}
        print(f'данные: {"обучающий набор, " + study.name if study else "синтетический DICOM"}')

        print('служебное')
        check('склонения', plural(1, 'файл', 'файла', 'файлов') == '1 файл'
              and plural(3, 'файл', 'файла', 'файлов') == '3 файла'
              and plural(11, 'файл', 'файла', 'файлов') == '11 файлов')
        body, ctype = _multipart({'chat_id': 1, 'media': [{'a': 1}]}, {'p': ('x.png', b'\x89PNG')})
        check('multipart собирается', b'name="chat_id"' in body and b'filename="x.png"' in body
              and 'boundary=' in ctype)
        api0 = TelegramAPI('123456:SECRET')
        check('токен вычищается из ошибок', 'SECRET' not in api0._scrub('url .../bot123456:SECRET/getMe'))
        check('длинная сводка режется на части', all(len(c) <= 4000 for c in split_message('x' * 3000 + '\n\n' + 'y' * 3000)))

        api = FakeAPI(files)
        bot = Bot(api, Processor(), tmp / 'work')
        me = bot.setup()
        check('setup: команды и описание', {'setMyCommands', 'setMyDescription'} <=
              {c[0] for c in api.calls} and me['username'] == 'test_bot')

        print('команды')
        bot.handle(update(1, text='/start'))
        check('/start отвечает инструкцией', 'Пришлите' in api.texts()[-1])
        bot.handle(update(2, text='/about'))
        check('/about показывает метрики', 'ROC-AUC' in api.texts()[-1], api.texts()[-1][:60])
        bot.handle(update(3, text='/help@test_bot'))
        check('команда с @именем бота', 'Что можно прислать' in api.texts()[-1])
        bot.handle(update(4, photo=[{'file_id': 'x'}]))
        check('фото — просьба прислать файлом', 'как <b>файл</b>' in api.texts()[-1])
        bot.handle(update(5, document={'file_id': 'big', 'file_name': 'big.zip',
                                       'file_size': 50 * 2 ** 20}))
        check('файл больше 20 МБ отклоняется', '20 МБ' in api.texts()[-1])

        print('обработка')
        n0 = len(api.calls)
        bot.handle(update(10, document={'file_id': 'f_dcm', 'file_name': dcms[0].name, 'file_size': 1}))
        task = bot.tasks.get_nowait()
        bot._process(task)
        new = api.calls[n0:]
        edits = [c for c in new if c[0] == 'editMessageText']
        check('сводка по снимку', edits and 'Готово' in edits[0][1]['text'], edits[0][1]['text'][:80] if edits else '')
        kb = edits[0][1].get('reply_markup') if edits else None
        check('кнопки SR / JSON / визуализации', kb and len(kb['inline_keyboard']) == 2)
        check('отчёт .xlsx отправлен', any(c[0] == 'sendDocument' and 'report_' in c[2]['document'][0]
                                          for c in new))
        pics = [c for c in new if c[0] in ('sendPhoto', 'sendMediaGroup')]
        check('снимок с разметкой отправлен', len(pics) == 1 or study is None)
        if pics and pics[0][0] == 'sendPhoto':
            png = pics[0][2]['photo'][1]
            check('картинка — PNG', png[:4] == b'\x89PNG', f'{len(png) // 1024} КБ')

        job = next(iter(bot.results))
        for kind in ('sr', 'js', 'ov'):
            n1 = len(api.calls)
            bot.handle({'update_id': 20, 'callback_query': {'id': 'q', 'data': f'{kind}:{job}',
                                                            'from': {'id': 7},
                                                            'message': {'chat': {'id': 42}}}})
            docs = [c for c in api.calls[n1:] if c[0] == 'sendDocument']
            check(f'кнопка {kind} присылает файл', docs and len(docs[0][2]['document'][1]) > 0)
        bot.handle({'update_id': 21, 'callback_query': {'id': 'q', 'data': 'sr:nope', 'from': {'id': 7},
                                                        'message': {'chat': {'id': 42}}}})
        check('устаревшая кнопка не падает', api.calls[-1][0] == 'answerCallbackQuery')

        n0 = len(api.calls)
        bot.handle(update(30, document={'file_id': 'f_zip', 'file_name': 'study.zip', 'file_size': 1}))
        bot._process(bot.tasks.get_nowait())
        res = list(bot.results.values())[-1]
        ok = [r for r in res.rows if r['processing_status'] == 'Success']
        check('zip: все снимки исследования', len(res.rows) >= 1 and len(ok) == len(res.rows),
              f'{len(res.rows)} снимков')
        album = [c for c in api.calls[n0:] if c[0] == 'sendMediaGroup']
        if len(ok) > 1:
            check('несколько снимков — одним альбомом', album and len(album[0][1]['media']) == len(res.images))

        print('альбом из нескольких файлов')
        for i, fid in enumerate(('f_a', 'f_b')):
            bot.handle(update(40 + i, document={'file_id': fid, 'file_name': f'{fid}.dcm', 'file_size': 1},
                              media_group_id='g1'))
        check('файлы альбома ждут друг друга', bot.tasks.qsize() == 0 and len(bot.groups) == 1)
        for g in bot.groups.values():
            g['deadline'] = 0
        import threading
        t = threading.Thread(target=bot._flusher, daemon=True)
        t.start()
        task = bot.tasks.get(timeout=5)
        bot.stop.set()
        check('альбом — одно задание из двух файлов', len(task.docs) == 2, task.label)

        print('доступ')
        api2 = FakeAPI(files)
        closed = Bot(api2, Processor(bot.proc.analyzer), tmp / 'work2', allowed={'@someone'})
        closed.handle(update(50, text='/start'))
        check('чужой пользователь получает отказ', 'ограничен' in api2.texts()[-1])
        closed.allowed = {'@tester'}
        closed.handle(update(51, text='/start'))
        check('разрешённый по @username проходит', 'Пришлите' in api2.texts()[-1])

        print('текст')
        row = {'anatomical_region': 'lh', 'processing_status': 'Success', 'quality_class': 1,
               'violation_type': 'hip_roi;hip_rotation', 'quality_probability': 0.9,
               'details': {'hip_roi': {'violation_text': 'поле 12.5 см < 12.9'},
                           'hip_rotation': {'text': 'выступ 118 мм²'}}, 'flags': ''}
        cap = frame_caption(1, row)
        check('подпись объясняет каждое нарушение', 'поле 12.5' in cap and 'выступ 118' in cap, cap)
        s = summary_text([row, dict(row, anatomical_region='spine', quality_class=0, violation_type='')])
        check('сводка считает нарушения', 'с нарушением 1, качественных 1' in s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'\nпройдено {len(PASSED)}, провалено {len(FAILED)}')
    return 1 if FAILED else 0


if __name__ == '__main__':
    raise SystemExit(main())
