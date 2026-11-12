#!/bin/sh
# Builds and runs every test in tests/rt with the sanitizer that suits it (gcc). Local only: nothing
# in .github/workflows calls this.
#   [VPAOT=path/to/vpaot] [CC=cc] [CXX=c++] run_local.sh [OUT_DIR]
# Without VPAOT the tests that need the translator are skipped (ubsan_aot alone takes ~15 minutes). Prints "FAIL ..." lines and ends
# with "run_local: N failed" (exit 1 if any). ASan cannot map at the loader tests' addresses
# (0x5_0000_0000 is in its shadow gap): those use UBSan; TSan runs host_races and loader_shared.
set -u
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
OUT=${1:-${TMPDIR:-/tmp}/vp_rt_local}
CC=${CC:-cc}; CXX=${CXX:-c++}
mkdir -p "$OUT"
CF="-O1 -g -std=gnu11 -frounding-math -I$R/runtime -I$D"
UB="-fsanitize=undefined -fno-sanitize-recover=all"
AS="-fsanitize=address,undefined -fno-sanitize-recover=all"
TS="-fsanitize=thread"
RT="$R/runtime/vp_host.c $R/runtime/vp_loader.c"
LIBS="-lm -latomic -lpthread"
bad=0
step() { # name, command...: a build or test that fails counts once
    name=$1; shift
    "$@" || { echo "FAIL $name"; bad=$((bad + 1)); }
}
build() { out=$1; shift; "$@" -o "$OUT/$out" 2>"$OUT/$out.cc.log" || { echo "FAIL $out (build: $(head -c 300 "$OUT/$out.cc.log"))"; bad=$((bad + 1)); return 1; }; }

build loader_bounds $CC $CF $UB "$D/loader_bounds.c" $RT $LIBS && step loader_bounds "$OUT/loader_bounds" "$OUT"
build loader_self $CC $CF $UB "$D/loader_self.c" $RT $LIBS && step loader_self "$OUT/loader_self" "$OUT"
build loader_stubs $CC $CF $UB "$D/loader_stubs.c" $RT $LIBS && step loader_stubs "$OUT/loader_stubs" "$OUT"
build loader_shared $CC $CF $UB "$D/loader_shared.c" $RT $LIBS && step loader_shared "$OUT/loader_shared" "$OUT"
build loader_shared_tsan $CC $CF $TS "$D/loader_shared.c" $RT $LIBS && step loader_shared_tsan "$OUT/loader_shared_tsan" "$OUT"
build loader_many $CC $CF $UB "$D/loader_many.c" $RT $LIBS && step loader_many "$OUT/loader_many"
build loader_rollback $CC $CF $UB "$D/loader_rollback.c" $RT $LIBS && step loader_rollback "$OUT/loader_rollback" "$OUT"
build loader_reload $CC $CF $UB "$D/loader_reload.c" $RT $LIBS && step loader_reload "$OUT/loader_reload" "$OUT"
build loader_stack $CC $CF $UB "$D/loader_stack.c" $RT $LIBS && step loader_stack "$OUT/loader_stack" "$OUT"
build loader_natives $CC $CF $UB "$D/loader_natives.c" $RT $LIBS && step loader_natives "$OUT/loader_natives" "$OUT"
build map_noreplace $CC $CF $UB "$D/map_noreplace.c" $RT $LIBS && step map_noreplace "$OUT/map_noreplace" "$OUT"
build host_races $CC $CF $TS "$D/host_races.c" $RT $LIBS && step host_races "$OUT/host_races" "$OUT"
build missing_dedupe $CC $CF $UB "$D/missing_dedupe.c" "$R/runtime/vp_host.c" $LIBS && step missing_dedupe "$OUT/missing_dedupe" "$OUT"
build cpu_ub $CC $CF -fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all "$D/cpu_ub.c" $RT $LIBS && step cpu_ub "$OUT/cpu_ub"
build x86_float $CC $CF $UB "$D/x86_float.c" $RT $LIBS && step x86_float "$OUT/x86_float"
build rcp_model $CC $CF $UB "$D/rcp_model.c" -lm && step rcp_model "$OUT/rcp_model"
build rcp_model_exact $CC $CF $UB -DVP_RCP_MODEL=VP_RCP_EXACT "$D/rcp_model.c" -lm && step rcp_model_exact "$OUT/rcp_model_exact"
build demo_elf $CC $CF $AS "$D/demo_elf.c" "$R/apps/Demo/App/Bridge/demo.c" $LIBS && step demo_elf "$OUT/demo_elf"
build demo_map $CC $CF $UB "$D/demo_map.c" "$R/apps/Demo/App/Bridge/demo.c" $LIBS && step demo_map "$OUT/demo_map"
build codesign_describe $CC $CF $AS "$D/codesign_describe.c" "$R/runtime/vp_codesign.c" && step codesign_describe "$OUT/codesign_describe" "$OUT"
build vpconvert_cache $CXX -std=c++17 -O1 -g $AS "$D/vpconvert_cache.cpp" -lpthread && step vpconvert_cache "$OUT/vpconvert_cache" "$OUT"
build vpconvert_cache_tsan $CXX -std=c++17 -O1 -g $TS "$D/vpconvert_cache.cpp" -lpthread && step vpconvert_cache_tsan "$OUT/vpconvert_cache_tsan" "$OUT"
step cpu_selfcontained sh "$D/cpu_selfcontained.sh" "$OUT/cpu_selfcontained"
step check_workflows python3 -I "$D/../ci/check_workflows.py"
step test_check_workflows python3 -I "$D/../ci/test_check_workflows.py"
step pack_spaces sh "$D/pack_spaces.sh"
step audit_scripts sh "$D/audit_scripts.sh"
step pkcs12_secrets sh "$D/pkcs12_secrets.sh" "$OUT/pkcs12"
if [ -n "${VPAOT:-}" ]; then
    step self_valid sh "$D/../aot/self_valid/run.sh" "$VPAOT" "$OUT/self_valid"
    step self_malformed sh "$D/../aot/self_malformed/run.sh" "$VPAOT" "$OUT/self_malformed"
    step patch_longline sh "$D/patch_longline.sh" "$VPAOT"
    step ubsan_aot sh "$D/ubsan_aot.sh" "$VPAOT" "$OUT/ubsan_aot"
else
    echo "SKIP self_valid, self_malformed, patch_longline, ubsan_aot (set VPAOT)"
fi
echo "run_local: $bad failed"
[ "$bad" -eq 0 ]
