#!/usr/bin/env bash
# Wrapper for `python -m math_constraint sweep`. Same env scaffolding as eval.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

[ -d .venv ] && source .venv/bin/activate

if command -v module >/dev/null 2>&1; then
  for mod in boost/1.89.0-atomic boost/1.86.0-atomic boost/1.82.0-atomic; do
    module load "$mod" 2>/dev/null && break
  done
fi
if [ -n "${BOOST_ROOT:-}" ] && [ -d "$BOOST_ROOT/lib" ]; then
  export LD_LIBRARY_PATH="$BOOST_ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

exec python -m math_constraint sweep "$@"
