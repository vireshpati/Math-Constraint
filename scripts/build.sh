#!/usr/bin/env bash
# Wrapper for `python -m math_constraint build`. Activates the project venv,
# loads boost (so smsg subprocesses link), then execs the CLI.
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

export MCNST_SOLVER_TIMEOUT_SEC="${MCNST_SOLVER_TIMEOUT_SEC:-600}"
exec python -m math_constraint build "$@"
