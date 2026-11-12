#!/bin/sh
# runtime/vp_cpu.h alone, in a folder with nothing else, compiles as C and as C++: an SDK that a
# client assembles with its own list of files (the app bundles vp_cpu.h, vp_emit.h, vp_regs.h, ...)
# must not need a header that list does not name.
#   cpu_selfcontained.sh [OUT_DIR]
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
OUT=${1:-${TMPDIR:-/tmp}/vp_cpu_selfcontained}
CC=${CC:-cc}; CXX=${CXX:-c++}
rm -rf "$OUT"; mkdir -p "$OUT" || exit 2
cp "$R/runtime/vp_cpu.h" "$OUT/" || exit 2
printf '#include "vp_cpu.h"\nint main(void) { return vp_rcp_f32(3.0f) > 0.0f && vp_rsqrt_f32(4.0f) > 0.0f ? 0 : 1; }\n' > "$OUT/t.c"
cp "$OUT/t.c" "$OUT/t.cpp"
bad=0
(cd "$OUT" && $CC -std=gnu11 -Wall -Wextra -o t_c t.c -lm) >"$OUT/c.log" 2>&1 || { echo "FAIL cpu_selfcontained (C): $(head -c 300 "$OUT/c.log")"; bad=1; }
(cd "$OUT" && $CXX -std=c++17 -Wall -Wextra -o t_cpp t.cpp -lm) >"$OUT/cpp.log" 2>&1 || { echo "FAIL cpu_selfcontained (C++): $(head -c 300 "$OUT/cpp.log")"; bad=1; }
[ "$bad" -eq 0 ] && echo "ok   cpu_selfcontained"
exit $bad
