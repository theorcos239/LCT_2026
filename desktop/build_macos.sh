#!/usr/bin/env sh
# Сборка настольного приложения для macOS (Apple Silicon или Intel — под
# архитектуру машины, на которой идёт сборка).
#
#   sh desktop/build_macos.sh            # версия 1.0.0
#   VERSION=1.1.0 sh desktop/build_macos.sh
#
# Что делает (из корня репозитория):
#   1. окружение Python 3.11 с теми же версиями пакетов, что в Docker-образе
#      (requirements.lock), плюс pywebview и PyInstaller;
#   2. PyInstaller -> dist/DXA-QC.app (окно и консольная dxa-qc-cli внутри);
#   3. проверка сборки: версия, загрузка моделей, отчёт на демо-архиве, если он есть;
#   4. образ диска dist/DXA-QC-<версия>-macos-<архитектура>.dmg.
#
# Нужен uv (https://docs.astral.sh/uv/). Интернет — только для установки пакетов.
set -eu
cd "$(dirname "$0")/.."
ROOT=$(pwd)
VERSION=${VERSION:-1.0.0}
ARCH=$(uname -m)
VENV=build/venv-macos
PY=$VENV/bin/python

step() { printf '\n== %s\n' "$1"; }

step "окружение Python 3.11"
command -v uv >/dev/null || { echo "нужен uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -x "$PY" ] || uv venv --python 3.11 "$VENV"
uv pip install --python "$PY" -r requirements.lock \
    pywebview==6.2.1 pyinstaller==6.22.3 pyinstaller-hooks-contrib==2026.7

step "иконка"
mkdir -p build/icon.iconset
for s in 16 32 128 256 512; do
    sips -z $s $s desktop/assets/icon.png --out build/icon.iconset/icon_${s}x${s}.png >/dev/null
    d=$((s * 2))
    sips -z $d $d desktop/assets/icon.png --out build/icon.iconset/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns build/icon.iconset -o build/icon.icns

step "версия сборки"
COMMIT=$(git rev-parse --short HEAD)
DIRTY=false
if [ -n "$(git status --porcelain -- desktop service region_clf spine_qc hip_roi hip_rotation cnn_qc requirements.lock)" ]; then
    echo "внимание: есть незакоммиченные изменения кода — сборка будет помечена как изменённая"
    DIRTY=true
fi
printf '{"version": "%s", "commit": "%s", "dirty": %s, "built": "%s"}\n' \
    "$VERSION" "$COMMIT" "$DIRTY" "$(date '+%Y-%m-%d %H:%M')" > build/build_info.json
echo "коммит $COMMIT"

step "PyInstaller"
"$PY" -m PyInstaller desktop/dxa_qc_macos.spec --noconfirm --distpath dist \
    --workpath build/pyinstaller-macos
APP=dist/DXA-QC.app
CLI=$APP/Contents/MacOS/dxa-qc-cli

step "проверка сборки"
VER=$("$CLI" version)
echo "$VER"
echo "$VER" | grep -q "$COMMIT" || { echo "в сборке не тот коммит"; exit 1; }
if [ -f demo.zip ]; then
    "$CLI" batch demo.zip --out build/smoke_report.csv --quiet
    echo "отчёт по demo.zip: build/smoke_report.csv"
else
    echo "demo.zip нет — пакетная проверка пропущена"
fi

step "образ диска"
DMG=dist/DXA-QC-$VERSION-macos-$ARCH.dmg
STAGE=build/dmg
rm -rf "$STAGE" "$DMG"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
[ -f docs/Инструкция_приложение.pdf ] && cp docs/Инструкция_приложение.pdf "$STAGE/Инструкция — приложение.pdf"
[ -f docs/Документация.pdf ] && cp docs/Документация.pdf "$STAGE/Документация.pdf"
hdiutil create -volname "Контроль качества DXA" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
ls -lh "$DMG"
printf '\nготово: %s\n' "$ROOT/$DMG"
