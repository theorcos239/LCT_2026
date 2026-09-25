#!/usr/bin/env sh
# Запуск сервиса контроля качества DXA (ТЗ 3.2).
#
#   ./run.sh                              поднять API на localhost:8000
#   ./run.sh batch /путь/к/исследованиям  пакетная обработка, отчёт в ./out
#   PORT=9000 ./run.sh                    другой порт
#
# Папка с исследованиями монтируется только на чтение: сервис не должен
# изменять исходные медицинские данные ни при каких обстоятельствах.
set -eu

IMAGE="${IMAGE:-dxa-qc}"
TAG="${TAG:-1.0.0}"
PORT="${PORT:-8000}"
OUT="${OUT:-$(pwd)/out}"
MODE="${1:-serve}"

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
    echo "API на http://localhost:${PORT}  (документация: /docs, проверка: /health)"
    exec docker run --rm -it -p "${PORT}:8000" --name dxa-qc "${IMAGE}:${TAG}"
    ;;
batch)
    DATA="${2:-}"
    if [ -z "$DATA" ] || [ ! -d "$DATA" ]; then
        echo "укажите папку с исследованиями:  ./run.sh batch /путь/к/исследованиям" >&2
        exit 1
    fi
    DATA=$(cd "$DATA" && pwd)
    mkdir -p "$OUT"
    FMT="${FMT:-xlsx}"
    echo "обработка $DATA -> $OUT/report.${FMT}"
    exec docker run --rm \
        -v "${DATA}:/data:ro" \
        -v "${OUT}:/out" \
        "${IMAGE}:${TAG}" \
        python -m service.cli /data \
            --out "/out/report.${FMT}" \
            --details /out/details.json \
            --overlays /out/overlays.zip
    ;;
*)
    echo "использование: $0 [serve|batch <папка>]" >&2
    exit 1
    ;;
esac
