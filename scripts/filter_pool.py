#!/usr/bin/env python3
"""Filter a generated dataset down to a curated subset using a selection list.

Usage:
    python scripts/filter_pool.py --source <dir> \
        --selection <file> --output <dir> [--append]

Reads <source>/instances/<name>.json for each name in the selection list and
writes them to <output>/instances/, plus <output>/problems.jsonl (in selection
order). Errors (exit 1) if any name is not found.

With --append, instance files are added to <output>/instances/ without
clobbering existing ones, and <output>/problems.jsonl is appended to. Used by
build_datasets.sh to assemble mathconstraint/ from two sources.

Selection file: one instance name per line. Lines starting with '#' and blank
lines are ignored.
"""
import argparse, json, shutil, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", required=True, help="source dataset dir")
p.add_argument("--selection", required=True, help="selection file (one name per line; # comments allowed)")
p.add_argument("--output", required=True, help="destination dir")
p.add_argument("--append", action="store_true",
               help="add to existing output dir instead of replacing")
args = p.parse_args()

src = Path(args.source)
out = Path(args.output)

names = []
for line in Path(args.selection).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    names.append(line)

(out / "instances").mkdir(parents=True, exist_ok=True)

found, missing = [], []
for name in names:
    candidate = src / "instances" / f"{name}.json"
    if candidate.exists():
        shutil.copy2(candidate, out / "instances" / f"{name}.json")
        found.append(json.loads(candidate.read_text()))
    else:
        missing.append(name)

mode = "a" if args.append else "w"
with (out / "problems.jsonl").open(mode) as f:
    for inst in found:
        f.write(json.dumps(inst) + "\n")

print(f"[filter_pool] {len(found)}/{len(names)} from {src} -> {out}/ ({'append' if args.append else 'replace'})")
if missing:
    print(f"[filter_pool] MISSING ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)
    sys.exit(1)
