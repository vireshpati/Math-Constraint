#!/usr/bin/env bash
# Assemble mathconstraint/ from mathconstraint_easy/ and pool/ via the two
# selection lists.
#
# Pipeline:
#   1. filter mathconstraint_easy/ with configs/selections/keptcore.txt
#      (80 instances → mathconstraint/)
#   2. append-filter pool/ with configs/selections/mathconstraint_hard.txt
#      (249 instances → mathconstraint/)
#
# Reproducibility note: the YAML profile `pool` describes the parameter space
# faithfully, but does not bit-reproduce the historical pool (which was a
# multi-iteration curation). Run `scripts/build.sh --profile pool --output pool/`
# to regenerate a similar-character pool, or use the shipped frozen
# mathconstraint/ artifact as ground truth.
#
# Usage:
#   scripts/build_mathconstraint.sh                  # default: filter into mathconstraint/
#   scripts/build_mathconstraint.sh --output OUT     # filter into OUT/
#   scripts/build_mathconstraint.sh --easy DIR --pool DIR --output OUT
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

EASY=mathconstraint_easy
POOL=pool
OUT=mathconstraint
while [ $# -gt 0 ]; do
  case "$1" in
    --easy)   EASY="$2"; shift 2 ;;
    --pool)   POOL="$2"; shift 2 ;;
    --output) OUT="$2";  shift 2 ;;
    -h|--help) sed -n '2,/^set/p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ ! -d "$EASY" ]; then
  echo "[build_mathconstraint] $EASY/ not found" >&2; exit 1
fi
if [ ! -d "$POOL" ]; then
  echo "[build_mathconstraint] $POOL/ not found — run scripts/build.sh --profile pool --output $POOL/ first" >&2
  exit 1
fi

rm -rf "$OUT"
echo "[build_mathconstraint] keptcore: $EASY/ → $OUT/"
python scripts/filter_pool.py \
  --source "$EASY" \
  --selection configs/selections/keptcore.txt \
  --output "$OUT"
echo "[build_mathconstraint] pool subset: $POOL/ → $OUT/ (append)"
python scripts/filter_pool.py --append \
  --source "$POOL" \
  --selection configs/selections/mathconstraint_hard.txt \
  --output "$OUT"

n=$(wc -l < "$OUT/problems.jsonl")
echo "[build_mathconstraint] wrote $n records to $OUT/"
