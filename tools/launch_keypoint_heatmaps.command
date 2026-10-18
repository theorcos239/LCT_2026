#!/bin/bash
# Двойной щелчок в Finder: выбрать веса при первом запуске и открыть просмотрщик.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT_PATH="$PROJECT_ROOT/Processing/drive_models/fold0.pt"
PYTHON_PATH="$PROJECT_ROOT/Processing/.venv-keypoints/bin/python"
LOG_PATH="$PROJECT_ROOT/Processing/keypoint_viewer.log"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Не найдены веса модели точек fold0.pt."
  echo "Выберите этот файл в открывшемся окне, если он у вас есть."
  SELECTED_FILE="$(osascript -e 'POSIX path of (choose file with prompt "Выберите обученные веса fold0.pt модели точек")' 2>/dev/null || true)"
  if [[ -z "$SELECTED_FILE" ]]; then
    echo "Запуск остановлен: без fold0.pt настоящие тепловые карты получить нельзя."
    read -r -p "Нажмите Enter, чтобы закрыть окно…" _
    exit 1
  fi
  mkdir -p "$(dirname "$CHECKPOINT_PATH")"
  cp "$SELECTED_FILE" "$CHECKPOINT_PATH"
fi

if [[ ! -x "$PYTHON_PATH" ]]; then
  UV_BIN="$(command -v uv || true)"
  if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  fi
  if [[ -z "$UV_BIN" ]]; then
    echo "Не найден uv для установки Python-зависимостей."
    read -r -p "Нажмите Enter, чтобы закрыть окно…" _
    exit 1
  fi
  "$UV_BIN" venv --python 3.11 Processing/.venv-keypoints
  "$UV_BIN" pip install --python "$PYTHON_PATH" torch numpy pillow
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_PATH" tools/serve_keypoint_heatmaps.py --device auto > "$LOG_PATH" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for ((attempt=0; attempt<60; attempt++)); do
  if curl --silent --fail --max-time 1 http://127.0.0.1:8767/api/items > /dev/null; then
    open http://127.0.0.1:8767/
    echo "Просмотрщик открыт. Закройте это окно или нажмите Ctrl+C, чтобы остановить сервер."
    echo "Журнал: $LOG_PATH"
    wait "$SERVER_PID"
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.2
done

echo "Просмотрщик не запустился. Последние сообщения:"
tail -n 20 "$LOG_PATH"
read -r -p "Нажмите Enter, чтобы закрыть окно…" _
exit 1
