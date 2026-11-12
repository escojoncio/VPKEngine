#!/bin/sh
# --patch / --roots: a line longer than the read buffer must not be cut in two and read as two lines.
# Usage: patch_longline.sh PATH/TO/vpaot
set -e
VPAOT=${1:?usage: patch_longline.sh vpaot}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
python3 -I - "$T" <<'PY'
import sys
t = sys.argv[1]
open(t + "/img.bin", "wb").write(b"\x90" * 8000 + b"\xc3")
open(t + "/patch.txt", "w").write("0 " + "ab" * 5000 + "\n")
# a comment of 511+ characters ending in "10": the tail must not be read as the root 0x10
open(t + "/roots.txt", "w").write("# " + "x" * 509 + "10\n0\n")
PY
"$VPAOT" --raw "$T/img.bin" --base 0x400000 --entry 0x400000 --patch "$T/patch.txt" --out "$T/o.c" 2> "$T/err" || { cat "$T/err"; exit 1; }
grep -q "^vpaot: 5000 bytes changed" "$T/err" || { echo "FAIL: patch line cut:"; cat "$T/err"; exit 1; }
"$VPAOT" --raw "$T/img.bin" --base 0x400000 --entry 0x400000 --roots "$T/roots.txt" --out "$T/o.c" 2> "$T/err" || { cat "$T/err"; exit 1; }
grep -q "^vpaot: 1 extra roots" "$T/err" || { echo "FAIL: roots line cut:"; cat "$T/err"; exit 1; }
echo "patch_longline: ok"
