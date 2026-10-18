#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x Processing/.venv-keypoints/bin/python ]]; then
  PYTHON=Processing/.venv-keypoints/bin/python
else
  PYTHON=python3
fi

PYTHONPATH=src "$PYTHON" tools/build_keypoint_validation_report.py
open Processing/validation_report.html
