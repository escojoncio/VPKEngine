#!/usr/bin/env bash
# libVPGLES.dylib (the shim, generated) + libEGL.dylib / libGLESv2.dylib (re-export it) for macOS arm64.
# Usage: build-shim.sh <headers dir> <out dir>. The install names are @executable_path/..., so the dylibs sit next to deqp-gles*.
set -euo pipefail
INC="$1"; OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 -I "$HERE/gen_shim.py" "$INC" "$OUT/vpgles_shim.c"
FLAGS=(-dynamiclib -arch arm64 -mmacosx-version-min=26.0 -O2 -Wall -Wno-unused-function)
clang "${FLAGS[@]}" "$OUT/vpgles_shim.c" -o "$OUT/libVPGLES.dylib" -install_name @executable_path/libVPGLES.dylib
echo 'int vpgles_shim_reexport;' > "$OUT/reexport.c"
for n in EGL GLESv2; do
  clang "${FLAGS[@]}" "$OUT/reexport.c" -o "$OUT/lib$n.dylib" -install_name "@executable_path/lib$n.dylib" \
    -Wl,-reexport_library,"$OUT/libVPGLES.dylib"
done
nm -gU "$OUT/libEGL.dylib" | grep -c " _egl" || true
nm -gU "$OUT/libGLESv2.dylib" | grep -c " _gl" || true
otool -L "$OUT/libEGL.dylib"
