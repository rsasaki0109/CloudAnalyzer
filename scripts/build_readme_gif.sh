#!/usr/bin/env bash
# Build docs/images/readme-demo.gif for the README first viewport.
# Skips gracefully when vhs is not installed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK=/tmp/cloudanalyzer-readme-gif
TAPE="$ROOT/scripts/build_readme_gif.tape"
OUT="$ROOT/docs/images/readme-demo.gif"

if ! command -v vhs >/dev/null 2>&1; then
  echo "vhs not installed; skipping README GIF generation." >&2
  echo "Install: https://github.com/charmbracelet/vhs" >&2
  exit 0
fi

if ! command -v ca >/dev/null 2>&1; then
  echo "ca not on PATH; install cloudanalyzer first (cd cloudanalyzer && pip install -e .)" >&2
  exit 1
fi

mkdir -p "$WORK" "$ROOT/docs/images"
if [[ ! -f "$WORK/map.pcd" ]]; then
  echo "Downloading hdl_localization sample map into $WORK ..."
  git clone --depth 1 https://github.com/koide3/hdl_localization "$WORK/hdl_localization"
  cp "$WORK/hdl_localization/data/map.pcd" "$WORK/map.pcd"
fi

cd "$ROOT"
vhs "$TAPE"
echo "Wrote $OUT"
