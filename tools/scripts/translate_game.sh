#!/bin/sh
# Translates a PS4 game dump (eboot.bin and sce_module/*.prx|*.sprx) for VPEngine's AOT guest CPU.
#   translate_game.sh GAME_DIR OUT_DIR [MISSING_LOG]      (VPAOT=path/to/vpaot, SPLIT=4000)
# Same output as translate_game.ps1. The C is derived from the game: keep it out of public repos.
set -e
GAME=$1; OUT=$2; MISSING=${3:-}
VPAOT=${VPAOT:-vpaot}; SPLIT=${SPLIT:-4000}
[ -n "$GAME" ] && [ -n "$OUT" ] || { echo "usage: $0 GAME_DIR OUT_DIR [MISSING_LOG]" >&2; exit 2; }
mkdir -p "$OUT"
names=" "
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
    # The previous translation of this module goes first (a different split leaves stale files).
    rm -f "$OUT/$name.c" "$OUT/$name".h "$OUT/${name}_decl.h" "$OUT/${name}_files.txt" "$OUT/$name.json"
    for old in "$OUT/${name}"_[0-9]*.c; do [ -f "$old" ] && rm -f "$old"; done
    echo "== $base -> $name"
    if [ -n "$MISSING" ] && [ -f "$MISSING" ]; then
        "$VPAOT" --elf "$f" --pic --module "$name" --split "$SPLIT" --out "$OUT/$name.c" --stats "$OUT/$name.json" --roots "$MISSING"
    else
        "$VPAOT" --elf "$f" --pic --module "$name" --split "$SPLIT" --out "$OUT/$name.c" --stats "$OUT/$name.json"
    fi
    names="$names$name "
done
[ "$names" != " " ] || { echo "no eboot.bin or sce_module/*.prx in $GAME" >&2; exit 1; }
# shellcheck disable=SC2086
"$VPAOT" --registry "$OUT/vpengine_registry.c" $names
