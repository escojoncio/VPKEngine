#!/bin/sh
# tools/scripts/make_game_pack.sh lists the pieces of C and the objects in words: a WORK folder with
# a space in its name must be refused at once (exit 2), not break the build half way.
# Usage: pack_spaces.sh
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/game" && : > "$T/game/eboot.bin"
printf '#!/bin/sh\nexit 0\n' > "$T/stub"; chmod +x "$T/stub"
WORK="$T/a folder" VPAOT=$T/stub CLANG=$T/stub LD64=$T/stub sh "$R/tools/scripts/make_game_pack.sh" "$T/game" "$T/out.vpgame" > "$T/log" 2>&1
rc=$?
if [ "$rc" -ne 2 ] || [ -e "$T/a folder" ]; then echo "FAIL pack_spaces: exit $rc"; cat "$T/log"; exit 1; fi
head -n1 "$T/log"
echo "pack_spaces: ok"
