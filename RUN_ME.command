#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "$ROOT"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 2
}

printf 'Hebrew OCR Unified Builder — macOS runtime release 15.0.0\n'
printf 'Dataset builder core: 15.0.0\n'
printf 'Log: %s\n\n' "$LOG_FILE"

OS="$(uname -s)"
PYTHON_BIN=""

if [ "$OS" = "Darwin" ]; then
  command -v brew >/dev/null 2>&1 || fail "Homebrew is required on macOS. Install Homebrew, then run this command again."

  echo "Installing/verifying the native RTL and image runtime..."
  brew install \
    python@3.12 \
    pkg-config \
    libraqm \
    harfbuzz \
    fribidi \
    freetype \
    jpeg-turbo \
    openjpeg \
    libtiff \
    webp \
    little-cms2

  PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12"
  [ -x "$PYTHON_BIN" ] || fail "Homebrew Python 3.12 was not found at $PYTHON_BIN"

  BREW_PREFIX="$(brew --prefix)"
  export CPPFLAGS="-I${BREW_PREFIX}/include -I$(brew --prefix freetype)/include -I$(brew --prefix jpeg-turbo)/include -I$(brew --prefix openjpeg)/include -I$(brew --prefix libtiff)/include -I$(brew --prefix webp)/include -I$(brew --prefix little-cms2)/include ${CPPFLAGS:-}"
  export LDFLAGS="-L${BREW_PREFIX}/lib -L$(brew --prefix freetype)/lib -L$(brew --prefix jpeg-turbo)/lib -L$(brew --prefix openjpeg)/lib -L$(brew --prefix libtiff)/lib -L$(brew --prefix webp)/lib -L$(brew --prefix little-cms2)/lib ${LDFLAGS:-}"
  export PKG_CONFIG_PATH="$(brew --prefix libraqm)/lib/pkgconfig:$(brew --prefix harfbuzz)/lib/pkgconfig:$(brew --prefix fribidi)/lib/pkgconfig:$(brew --prefix freetype)/lib/pkgconfig:$(brew --prefix jpeg-turbo)/lib/pkgconfig:$(brew --prefix openjpeg)/lib/pkgconfig:$(brew --prefix libtiff)/lib/pkgconfig:$(brew --prefix webp)/lib/pkgconfig:$(brew --prefix little-cms2)/lib/pkgconfig:${BREW_PREFIX}/lib/pkgconfig:${BREW_PREFIX}/share/pkgconfig:${PKG_CONFIG_PATH:-}"
  pkg-config --modversion raqm
else
  for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  [ -n "$PYTHON_BIN" ] || fail "Python 3.12.x is required."
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12); print("Python:", sys.version.split()[0])'

if [ -x "$ROOT/.venv/bin/python" ] && ! "$ROOT/.venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
  echo "Removing an incompatible virtual environment."
  rm -rf "$ROOT/.venv"
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
VENV_PY="$ROOT/.venv/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$VENV_PY" -m pip install --upgrade pip setuptools wheel

if [ "$OS" = "Darwin" ]; then
  REQUIREMENTS_NO_PILLOW="$ROOT/.requirements-without-pillow.txt"
  awk 'BEGIN{IGNORECASE=1} !/^Pillow==/' "$ROOT/requirements-lock.txt" > "$REQUIREMENTS_NO_PILLOW"
  "$VENV_PY" -m pip install --requirement "$REQUIREMENTS_NO_PILLOW"
  "$VENV_PY" -m pip uninstall -y Pillow >/dev/null 2>&1 || true
  "$VENV_PY" -m pip install --no-cache-dir --no-binary Pillow 'Pillow==12.2.0'
else
  "$VENV_PY" -m pip install --requirement "$ROOT/requirements-lock.txt"
fi

"$VENV_PY" -m pip install --no-deps --editable "$ROOT"

"$VENV_PY" - <<'PY'
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features
from heocr_unified.fonts import discover_fonts

assert features.check_feature("raqm"), (
    "Pillow was installed without RAQM. The builder will not render Hebrew without "
    "RAQM/FriBiDi/HarfBuzz."
)
print("RAQM:", features.version_feature("raqm"))
print("FriBiDi:", features.version_feature("fribidi"))
print("HarfBuzz:", features.version_feature("harfbuzz"))

text = "מסמך 2026 Section A שָׁלוֹם"
eligible = [
    info for info in discover_fonts()
    if info.supports(text, require_marks=True)
]
eligible.sort(key=lambda info: (
    "hebrew" not in f"{info.family} {info.path.name}".casefold(),
    info.family.casefold(),
    info.style.casefold(),
    str(info.path),
))
if not eligible:
    raise RuntimeError("No genuine macOS font with Hebrew, niqqud, digits, and Latin coverage was found")
font_path = eligible[0].path
font = ImageFont.truetype(str(font_path), 42, layout_engine=ImageFont.Layout.RAQM)
image = Image.new("RGB", (1400, 130), "white")
draw = ImageDraw.Draw(image)
draw.text((1360, 30), text, font=font, fill="black", direction="rtl", language="he", anchor="ra")
if image.getbbox() is None:
    raise RuntimeError("RAQM smoke render produced an empty image")
buffer = io.BytesIO()
image.save(buffer, format="WEBP", quality=92)
buffer.seek(0)
round_trip = Image.open(buffer)
round_trip.load()
assert round_trip.size == image.size
print("Hebrew/niqqud/mixed-BiDi render and WebP round-trip: PASS")
print("Smoke-test font:", font_path)
PY

"$ROOT/.venv/bin/hf" auth whoami

TEST_LOG="$LOG_DIR/tests-$(date -u +%Y%m%dT%H%M%SZ).txt"
set +e
"$VENV_PY" -m pytest -q --disable-warnings | tee "$TEST_LOG"
TEST_RC=${PIPESTATUS[0]}
set -e
[ "$TEST_RC" -eq 0 ] || fail "The builder test suite failed. See $TEST_LOG"
if grep -q 'skipped' "$TEST_LOG"; then
  fail "The test suite skipped tests; the runtime is not accepted. See $TEST_LOG"
fi

COMMAND=("$VENV_PY" -m heocr_unified run --config "$ROOT/config.json")
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu "${COMMAND[@]}"
else
  "${COMMAND[@]}"
fi
