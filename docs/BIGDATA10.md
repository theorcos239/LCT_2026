# Сервер bigdata10

Адрес приложения: **https://89.208.196.172/**. Пароля нет: стенд открыт для
демонстрации. Загружать на него можно только обезличенные снимки.

Вход по паролю снят 20.10.2026; прежний конфиг сохранён на сервере как
`/etc/nginx/dxa-qc.with-auth.bak`, файл `/etc/nginx/.dxa_qc_htpasswd` оставлен.
Вернуть пароль:
`sudo cp /etc/nginx/dxa-qc.with-auth.bak /etc/nginx/sites-available/dxa-qc && sudo nginx -t && sudo systemctl reload nginx`.
Файлу `.dxa_qc_htpasswd` нужны владелец `root`, группа `www-data` и права
`640`, иначе nginx отвечает 500 при попытке входа.

Контейнер `dxa-qc` слушает только `127.0.0.1:8000`; наружу его отдаёт nginx
через HTTPS. Сертификат Let's Encrypt для IP действует шесть дней;
таймер `dxa-qc-cert-renew.timer` проверяет продление дважды в сутки и обновляет
nginx после выдачи нового сертификата.

На сервере загружены четыре чекпоинта из `runs/keypoints_crops`: `fold0.pt` —
`fold3.pt`. Модель точек показывает диагностические координаты и участвует
в проверке верхней границы укладки по видимой части Th12. Она **не
подменяет итоговый вердикт по оси**: её новая нарезка и правило выбора
траектории пока не прошли контроль качества. ROI бедра проверяется по
отступам рисунка 6 ТЗ (`margins`). Рабочие CNN-модели ротации и предметов
также загружены. Исходные медицинские снимки на сервер не переносились.

Проверить состояние:

```bash
ssh -i ~/Downloads/bigdata10.pem ubuntu@89.208.196.172
sudo docker ps
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/version
sudo docker logs --tail 50 dxa-qc
systemctl status dxa-qc-cert-renew.timer
```

В `/health` должны быть `true` для `region_clf`, `spine_appearance`,
`keypoints`, `cnn_rotation`, `cnn_artifacts`. В `/version` должны быть
`roi_rule: margins`, `keypoints_available: true` и
`keypoints_axis_verdict_active: false`; `keypoints_error` должен быть `null`.

Для обновления кода и моделей есть отдельный выпуск
`/opt/dxa-qc/releases/20261020/`, каталог весов
`/opt/dxa-qc/models/keypoints_crops/` и конфигурация прокси
`deploy/bigdata10.nginx.conf`. После пересборки нового образа контейнер нужно
создать заново с теми же переменными и томом. Порт 8000 наружу не открывать.

Исходные снимки удаляются сразу после анализа. Результаты заданий (таблица,
снимки с разметкой, SR) хранятся в контейнере 1 час (`DXA_QC_JOB_TTL_HOURS=1`),
затем удаляются; кнопка «Удалить с сервера» стирает их сразу. В списке недавних
заданий каждый браузер видит только свои (заголовок `X-DXA-Client`). При
пересоздании контейнера все результаты пропадают.

Текущий выпуск — `/opt/dxa-qc/releases/20261020b/`; прежний образ сохранён как
`dxa-qc:keypoints-prev`. Контейнер запущен так:

```bash
sudo docker run -d --name dxa-qc --restart unless-stopped -p 127.0.0.1:8000:8000 \
  -v /opt/dxa-qc/models/keypoints_crops:/app/runs/keypoints_crops:ro \
  -e PYTHONPATH=/app/src -e DXA_QC_KEYPOINTS=1 -e DXA_QC_KEYPOINTS_VERDICT=0 \
  -e DXA_QC_KEYPOINTS_MULTICROP=0 -e DXA_QC_ROI_RULE=margins \
  -e DXA_QC_KEYPOINTS_DIR=/app/runs/keypoints_crops -e DXA_QC_JOB_TTL_HOURS=1 \
  dxa-qc:keypoints
```
Отметки оператора хранятся в браузере. Доступ к стенду сейчас открыт всем — для клинического пилота нужны
индивидуальные учётные записи и согласованный порядок хранения данных.
