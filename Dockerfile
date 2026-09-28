# Сервис контроля качества денситометрических исследований (DXA).
#
# ТЗ 3.2 требует фиксировать в том числе версию базового контейнера. Тег
# 3.11-slim фиксирует минорную версию Python, но со временем переезжает на
# новую сборку, поэтому образ закреплён дайджестом (снят 2026-09-27):
#
#     docker pull python:3.11-slim
#     docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim
#
# Обновить — та же команда, подставить новый дайджест вместо строки FROM.
FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

LABEL org.opencontainers.image.title="DXA Quality Control" \
      org.opencontainers.image.description="Оценка качества исследований плотности костей" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Один поток на BLAS — не экономия, а условие воспроизводимости (ТЗ 2.7):
# многопоточная редукция складывает числа в недетерминированном порядке, и
# два прогона на одних данных могут разойтись в последнем знаке. Параллелизм
# берём процессами на уровне исследований, а не потоками внутри матриц.

WORKDIR /app

# Ставим замороженный набор с транзитивными зависимостями, а не
# requirements.txt: иначе starlette, pydantic и прочие подбирались бы заново
# при каждой сборке. pip check — страховка от неполного lock-файла.
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock && pip check

# Непривилегированный пользователь: сервис читает медицинские изображения,
# root внутри контейнера ему не нужен. Создаётся до копирования кода, а права
# выдаются через COPY --chown: `chown -R` отдельным слоем продублировал бы в
# образе все файлы, включая 255 МБ весов сети.
RUN useradd --create-home --uid 10001 dxa && \
    mkdir -p /data /out && chown dxa:dxa /app /data /out

# Код и веса. Обучающий набор в образ не кладётся: данные монтируются снаружи.
COPY --chown=dxa:dxa region_clf/ region_clf/
COPY --chown=dxa:dxa hip_roi/ hip_roi/
COPY --chown=dxa:dxa hip_rotation/ hip_rotation/
COPY --chown=dxa:dxa spine_qc/ spine_qc/
# Нейросетевое второе мнение (ротация, посторонние предметы): код, ONNX-веса
# (ансамбль 3 x 42.6 МБ на критерий, fp16) и калибровка свода. Обучающий
# скрипт в образе не исполняется (он требует torch), но весит килобайты.
COPY --chown=dxa:dxa cnn_qc/ cnn_qc/
COPY --chown=dxa:dxa service/ service/
COPY --chown=dxa:dxa stats.py trainset.py folds.csv ./

# Код модели ключевых точек (десятки КБ) — без него флаг --keypoints не
# заработал бы даже при смонтированных весах и torch. torch и веса
# (runs/keypoints, requirements-train.txt) в образ сознательно не входят:
# образ не растёт на 2 ГБ, флаг по умолчанию выключен и без них просто
# недоступен (см. README, «Модель ключевых точек»).
COPY --chown=dxa:dxa src/ src/

USER dxa

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

# По умолчанию — API. Пакетная обработка: docker run ... python -m service.cli ...
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
