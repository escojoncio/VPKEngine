#!/bin/sh
# Read-only data in the executable segment, as PS4 images have it (-z noseparate-code): strings
# referenced with rip-relative lea must not become functions (their bytes decode as ins/outs).
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/data_in_text; mkdir -p "$B"
${CC:-gcc} -O2 -fpie -pie -static-libgcc -Wl,-z,noseparate-code "$D/prog.c" -o "$B/prog"
"$VPAOT" --elf "$B/prog" --stats "$B/stats.json" --out "$B/prog.c" 2> "$B/vpaot.txt"
python3 - "$B/stats.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
bad = {k: v for k, v in s["unsupported_by_mnemonic"].items() if k not in ("hlt", "ud2")}
print(f"data in text: {s['rejected_as_data']} candidates rejected as data, unsupported {bad}")
assert s["rejected_as_data"] > 0, "nothing rejected: strings were not seen as data"
assert sum(bad.values()) <= 2, f"data decoded as code is still translated: {bad}"
PY
