#!/bin/sh
# Builds a VPEngine game pack (see make_game_pack.ps1, the same steps for Linux and macOS):
#   make_game_pack.sh GAME_DIR [OUT] [MISSING_LOG]
# Environment: VPAOT, CLANG (clang 17+ with visionOS), LD64 (ld64.lld), JOBS, SPLIT, TITLE, WORK,
# PLATFORM (xros: the headset; macos/ios only for tests), CLEAN=1 (delete WORK at the end).
set -e
GAME=$1; OUT=${2:-$1/vpengine.vpgame}; MISSING=${3:-}
[ -n "$GAME" ] && [ -d "$GAME" ] || { echo "usage: $0 GAME_DIR [OUT] [MISSING_LOG]" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
VPAOT=${VPAOT:-vpaot}; CLANG=${CLANG:-clang}; LD64=${LD64:-ld64.lld}
SPLIT=${SPLIT:-300}; TITLE=${TITLE:-$(basename "$(cd "$GAME" && pwd)")}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
PLATFORM=${PLATFORM:-xros}
WORK=${WORK:-${XDG_CACHE_HOME:-$HOME/.cache}/vpengine/$(printf '%s' "$TITLE" | tr -c 'A-Za-z0-9_.-' '_')}
RUNTIME=$HERE/runtime; [ -f "$RUNTIME/vp_emit.h" ] || RUNTIME=$HERE/../../runtime
SDK=$HERE/sdk; [ -f "$SDK/libVPRuntime.tbd" ] || SDK=$HERE/../sdk
case "$PLATFORM" in
    xros) TARGET=arm64-apple-xros2.0; PV="xros 2.0 26.0" ;;
    ios) TARGET=arm64-apple-ios17.0; PV="ios 17.0 17.0" ;;
    macos) TARGET=arm64-apple-macos11; PV="macos 11.0 11.0" ;;
    *) echo "PLATFORM must be xros, ios or macos" >&2; exit 2 ;;
esac
mkdir -p "$WORK"
start=$(date +%s)

# 1. Translate every module (seconds each) and list the pieces of C.
names=" "
pieces=""
for f in "$GAME/eboot.bin" "$GAME"/sce_module/*; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    case "$(printf '%s' "$base" | tr 'A-Z' 'a-z')" in
        eboot.bin) suffix="" ;;
        *.prx|*.sprx) suffix="_prx" ;;
        *) continue ;;
    esac
    stem=${base%.*}
    name=$(printf '%s' "$stem" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9_]/_/g; s/^\([0-9]\)/m_\1/')$suffix
    case "$names" in *" $name "*) echo "two modules map to the name $name (rename one of them)" >&2; exit 1 ;; esac
    rm -f "$WORK/$name.c" "$WORK/$name.h" "$WORK/${name}_decl.h" "$WORK/${name}_files.txt"
    for old in "$WORK/${name}"_[0-9]*.c; do [ -f "$old" ] && rm -f "$old"; done
    echo "== $base -> $name"
    if [ -n "$MISSING" ] && [ -f "$MISSING" ]; then
        "$VPAOT" --elf "$f" --pic --module "$name" --split "$SPLIT" --out "$WORK/$name.c" --stats "$WORK/$name.json" --roots "$MISSING"
    else
        "$VPAOT" --elf "$f" --pic --module "$name" --split "$SPLIT" --out "$WORK/$name.c" --stats "$WORK/$name.json"
    fi
    if [ -f "$WORK/${name}_files.txt" ]; then
        for p in $(cat "$WORK/${name}_files.txt"); do pieces="$pieces $WORK/$(basename "$p")"; done
    else
        pieces="$pieces $WORK/$name.c"
    fi
    names="$names$name "
done
[ "$names" != " " ] || { echo "no eboot.bin or sce_module/*.prx in $GAME" >&2; exit 1; }
# shellcheck disable=SC2086
"$VPAOT" --registry "$WORK/vpengine_registry.c" --pack "$TITLE" $names
pieces="$pieces $WORK/vpengine_registry.c"

# 2. Compile in parallel; unchanged pieces keep their object; each piece's C goes once compiled.
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
todo="$WORK/todo.txt"; : > "$todo"
objects=""
for c in $pieces; do
    o=${c%.c}.o
    objects="$objects $o"
    h=$(sha "$c")
    if [ -f "$o" ] && [ "$(cat "$o.sha256" 2>/dev/null)" = "$h" ]; then
        rm -f "$c"
    else
        echo "$c $o $h" >> "$todo"
    fi
done
for o in "$WORK"/*.o; do
    [ -f "$o" ] || continue
    case " $objects " in *" $o "*) ;; *) rm -f "$o" "$o.sha256" ;; esac
done
echo "Compiling $(wc -l < "$todo") of $(echo $pieces | wc -w) pieces with $JOBS jobs (the rest are unchanged)..."
export CLANG TARGET RUNTIME
# shellcheck disable=SC2016
[ -s "$todo" ] && xargs -P "$JOBS" -L 1 sh -c '
    "$CLANG" --target="$TARGET" -O2 -ffreestanding -fno-stack-protector -fno-math-errno -frounding-math -w \
        -isystem "$RUNTIME/freestanding" -I "$RUNTIME" -c "$0" -o "$1" && echo "$2" > "$1.sha256" && rm -f "$0"
' < "$todo"
rm -f "$todo"

# 3. Link: one library that only needs the app's runtime and the system.
# shellcheck disable=SC2086
"$LD64" -arch arm64 -platform_version $PV -dylib -adhoc_codesign -install_name @rpath/vpengine.vpgame \
    -rpath @executable_path/Frameworks -o "$OUT.tmp" $objects "$SDK/libVPRuntime.tbd" "$SDK/libSystem.tbd"
mv -f "$OUT.tmp" "$OUT"
[ "${CLEAN:-0}" = 1 ] && rm -rf "$WORK"
echo "Done in $(( $(date +%s) - start )) s: $OUT ($(( $(wc -c < "$OUT") / 1048576 )) MB, $(echo $names | wc -w) modules)."
