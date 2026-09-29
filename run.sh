#!/usr/bin/env sh
# Запуск сервиса контроля качества DXA (ТЗ 3.2).
#
#   ./run.sh                              веб-интерфейс и API на localhost:8000
#   ./run.sh batch /путь/к/исследованиям  пакетная обработка, отчёт в ./out
#   ./run.sh bot                          Telegram-бот (токен — DXA_QC_TG_TOKEN или .env)
#   PORT=9000 ./run.sh                    другой порт
#   ROI_RULE=margins ./run.sh batch ...   отступы ROI строго по рисунку 6 ТЗ
#                                         (по умолчанию — по длине поля, как эксперт)
#
# Папка с исследованиями монтируется только на чтение: сервис не должен
# изменять исходные медицинские данные ни при каких обстоятельствах.
set -eu

IMAGE="${IMAGE:-dxa-qc}"
TAG="${TAG:-1.0.0}"
PORT="${PORT:-8000}"
OUT="${OUT:-$(pwd)/out}"
ROI_RULE="${ROI_RULE:-scan_length}"
MODE="${1:-serve}"
HERE=$(cd "$(dirname "$0")" && pwd)

# Git Bash / MSYS на Windows: без этого «/data» превращается в
# «C:/Program Files/Git/data», а путь хоста нужен в виде D:/..., а не /d/...
winpath() { cd "$1" && pwd; }
case "$(uname -s 2>/dev/null)" in
MINGW*|MSYS*|CYGWIN*)
    export MSYS_NO_PATHCONV=1
    winpath() { cd "$1" && pwd -W; }
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "docker не найден в PATH" >&2
    exit 1
fi
if ! docker image inspect "${IMAGE}:${TAG}" >/dev/null 2>&1; then
    echo "образ ${IMAGE}:${TAG} не найден, сначала: ./build.sh" >&2
    exit 1
fi

case "$MODE" in
serve)
    echo "Интерфейс:    http://localhost:${PORT}"
    echo "API:          http://localhost:${PORT}/docs  (проверка: /health)"
    exec docker run --rm -it -p "${PORT}:8000" -e "DXA_QC_ROI_RULE=${ROI_RULE}" \
        --name dxa-qc "${IMAGE}:${TAG}"
    ;;
batch)
    DATA="${2:-}"
    if [ -z "$DATA" ] || [ ! -d "$DATA" ]; then
        echo "укажите папку с исследованиями:  ./run.sh batch /путь/к/исследованиям" >&2
        exit 1
    fi
    DATA=$(winpath "$DATA")
    mkdir -p "$OUT"
    OUT=$(winpath "$OUT")
    FMT="${FMT:-xlsx}"
    echo "обработка $DATA -> $OUT/report.${FMT}"
    # Контейнер запускается от uid/gid вызывающего: на Linux каталог ./out
    # создан им с правами 755, и пользователь образа (uid 10001) писать туда
    # не смог бы. Заодно результаты принадлежат тому, кто их заказал, а не
    # безымянному uid.
    exec docker run --rm \
        --user "$(id -u):$(id -g)" \
        -v "${DATA}:/data:ro" \
        -v "${OUT}:/out" \
        "${IMAGE}:${TAG}" \
        python -m service.cli /data \
            --out "/out/report.${FMT}" \
            --roi-rule "${ROI_RULE}" \
            --details /out/details.json \
            --overlays /out/overlays.zip \
            --sr /out/sr.zip
    ;;
bot)
    # Токен — из окружения или из файла .env рядом со скриптом (в git не хранится).
    # Долго работающий бот на сервере: добавьте -d --restart unless-stopped.
    set -- docker run --rm -it --name dxa-qc-bot
    if [ -f "$HERE/.env" ]; then
        set -- "$@" --env-file "$(winpath "$HERE")/.env"
    fi
    if [ -n "${DXA_QC_TG_TOKEN:-}" ]; then
        set -- "$@" -e DXA_QC_TG_TOKEN
    fi
    if [ ! -f "$HERE/.env" ] && [ -z "${DXA_QC_TG_TOKEN:-}" ]; then
        echo "нет токена: задайте DXA_QC_TG_TOKEN или создайте .env по образцу .env.example" >&2
        exit 1
    fi
    exec "$@" "${IMAGE}:${TAG}" python -m bot
    ;;
*)
    echo "использование: $0 [serve|batch <папка>|bot]" >&2
    exit 1
    ;;
esac
