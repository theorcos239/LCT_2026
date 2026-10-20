#!/usr/bin/env sh
# Сборка docs/Документация.pdf из Документация.html через Chrome без окна.
#   sh docs/documentation/build_pdf.sh
set -eu
cd "$(dirname "$0")"
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
command -v "$CHROME" >/dev/null 2>&1 || [ -x "$CHROME" ] || CHROME=$(command -v google-chrome || command -v chromium)
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$(pwd)/../Документация.pdf" "file://$(pwd)/Документация.html" 2>/dev/null
ls -l ../Документация.pdf
