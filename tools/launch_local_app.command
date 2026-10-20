#!/bin/sh
# Локальное веб-приложение с моделью точек. Снимки остаются на этом компьютере.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
WEIGHTS=${DXA_QC_KEYPOINTS_DIR:-$ROOT/runs/keypoints_crops}

if ! find "$WEIGHTS" -maxdepth 1 -name 'fold[0-9].pt' -print -quit 2>/dev/null | grep -q .; then
    echo "Не найдены веса fold*.pt в $WEIGHTS"
    echo "Скопируйте лучшие чекпоинты в runs/keypoints_crops/ и запустите снова."
    exit 1
fi

if [ -x "$ROOT/Processing/.venv-keypoints/bin/python" ]; then
    PY="$ROOT/Processing/.venv-keypoints/bin/python"
else
    PY="$ROOT/.venv-local/bin/python"
    if [ ! -x "$PY" ]; then
        if ! command -v uv >/dev/null 2>&1; then
            echo "Для первой установки нужен uv: https://docs.astral.sh/uv/"
            exit 1
        fi
        uv venv --python 3.11 "$ROOT/.venv-local"
    fi
fi

if ! "$PY" -c 'import fastapi, uvicorn, onnxruntime, pydicom, openpyxl' 2>/dev/null; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "Не хватает библиотек из requirements.lock; установите uv и запустите снова."
        exit 1
    fi
    echo "Первая установка библиотек приложения…"
    uv pip install --python "$PY" -r "$ROOT/requirements.lock"
fi
if ! "$PY" -c 'import torch, torchvision' 2>/dev/null; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "Не хватает PyTorch; установите uv и запустите снова."
        exit 1
    fi
    echo "Первая установка PyTorch для модели точек…"
    uv pip install --python "$PY" torch torchvision
fi

export DXA_QC_KEYPOINTS=1
export DXA_QC_KEYPOINTS_DIR="$WEIGHTS"
export DXA_QC_ROI_RULE=${DXA_QC_ROI_RULE:-margins}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PORT=${PORT:-8000}

echo "Приложение: http://127.0.0.1:$PORT"
echo "Модель точек: $WEIGHTS"
echo "Закройте окно терминала или нажмите Ctrl+C, чтобы остановить сервис."

"$PY" -m uvicorn service.api:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

if command -v curl >/dev/null 2>&1; then
    i=0
    READY=0
    while [ "$i" -lt 40 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID"
            exit 1
        fi
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            READY=1
            if [ "${NO_BROWSER:-0}" = 1 ]; then
                :
            elif command -v open >/dev/null 2>&1; then
                open "http://127.0.0.1:$PORT/"
            elif command -v xdg-open >/dev/null 2>&1; then
                xdg-open "http://127.0.0.1:$PORT/" >/dev/null 2>&1 || true
            fi
            break
        fi
        i=$((i + 1))
        sleep 1
    done
    if [ "$READY" -ne 1 ]; then
        echo "Сервис не ответил за 40 секунд. Проверьте сообщения выше." >&2
        exit 1
    fi
fi
wait "$SERVER_PID"
