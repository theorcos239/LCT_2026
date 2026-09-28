#!/usr/bin/env sh
# Сборка контейнера сервиса контроля качества DXA (ТЗ 3.2).
#
#   ./build.sh                    собрать образ dxa-qc:1.0.0
#   IMAGE=my/dxa TAG=dev ./build.sh
#
# POSIX sh, без bash-измов: скрипт должен работать в Linux и UNIX-подобных
# системах, включая окружения, где /bin/sh — это dash или busybox.
set -eu

IMAGE="${IMAGE:-dxa-qc}"
TAG="${TAG:-1.0.0}"
HERE=$(cd "$(dirname "$0")" && pwd)

if ! command -v docker >/dev/null 2>&1; then
    echo "docker не найден в PATH" >&2
    exit 1
fi

# Веса обязаны лежать в репозитории: без них сервис поднимется, но будет
# работать на одной геометрии и молча потеряет часть критериев.
for f in region_clf/model.joblib spine_qc/model.joblib spine_qc/thresholds.json \
         hip_rotation/thresholds.json hip_roi/probability.json \
         cnn_qc/rotation/model.onnx cnn_qc/rotation/meta.json \
         cnn_qc/artifacts/model.onnx cnn_qc/artifacts/meta.json; do
    if [ ! -f "$HERE/$f" ]; then
        echo "не найден файл модели: $f" >&2
        echo "обучите заново, см. docs/TRAINING.md (region_clf.train, spine_qc.calibrate," >&2
        echo "hip_rotation.calibrate, hip_roi.probability, cnn_qc.train)" >&2
        exit 1
    fi
done

echo "сборка ${IMAGE}:${TAG}"
docker build -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" "$HERE"

echo
echo "готово: ${IMAGE}:${TAG}"
docker image inspect "${IMAGE}:${TAG}" --format 'размер образа: {{.Size}} байт' 2>/dev/null || true
echo "запуск:  ./run.sh"
