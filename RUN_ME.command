#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "$ROOT"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ).log"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.13 python3; do
  if ! command -v "$candidate" >/dev/null 2>&1; then
    continue
  fi
  if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)
PY
  then
    PYTHON_BIN="$candidate"
    break
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3.11–3.13 is required; Python 3.12 is recommended." >&2
  echo "On macOS with Homebrew: brew install python@3.12" >&2
  exit 2
fi
"$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.version.split()[0])
PY

if [ -x "$ROOT/.venv/bin/python" ]; then
  if ! "$ROOT/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)
PY
  then
    rm -rf "$ROOT/.venv"
  fi
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI is required. Install it and run: hf auth login" >&2
  exit 2
fi
hf auth whoami

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install --requirement "$ROOT/requirements-lock.txt"
"$ROOT/.venv/bin/python" -m pip install --no-deps --editable "$ROOT"

export PYTHONUNBUFFERED=1
"$ROOT/.venv/bin/python" -m pytest -q --disable-warnings

COMMAND=("$ROOT/.venv/bin/python" -m heocr_unified run --config "$ROOT/config.json")
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
else
  "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
fi
