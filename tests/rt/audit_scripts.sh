#!/bin/sh
# The test and tool scripts must fail when they have nothing to work with, not report success:
#  - tools/sdk/gen_runtime_tbd.sh on a library without runtime symbols wrote an empty stub and said "0 runtime symbols";
#  - tests/rt/ubsan_aot.sh counted a case vpaot could not translate (or the C that did not compile) as SKIP and exited 0;
#  - tests/aot/self_malformed/run.sh with a generator that made no cases ran no case and exited 0.
# Usage: audit_scripts.sh
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
bad=0

# gen_runtime_tbd.sh
printf 'int not_the_runtime(void) { return 0; }\n' > "$T/x.c"
${CC:-cc} -shared -fPIC -o "$T/x.so" "$T/x.c" || { echo "SKIP audit_scripts (no cc)"; exit 0; }
if sh "$R/tools/sdk/gen_runtime_tbd.sh" "$T/x.so" "$T/out.tbd" > "$T/gen.log" 2>&1 || [ -e "$T/out.tbd" ]; then
    echo "FAIL gen_runtime_tbd: a library without vp_* symbols produced a stub"; cat "$T/gen.log"; bad=1
fi

# ubsan_aot.sh: a case the translator cannot handle is a failure
mkdir -p "$T/cases"
printf '.text\n.globl _start\n_start:\n    ret\n' > "$T/cases/one.s"
printf '#!/bin/sh\nexit 1\n' > "$T/vpaot_fails"; chmod +x "$T/vpaot_fails"
if CASES_DIR="$T/cases" sh "$D/ubsan_aot.sh" "$T/vpaot_fails" "$T/ubsan" > "$T/ub.log" 2>&1; then
    echo "FAIL ubsan_aot: vpaot failing on every case still exited 0"; cat "$T/ub.log"; bad=1
fi

# self_malformed/run.sh: no cases is not a pass
mkdir -p "$T/sm"
cp "$R/tests/aot/self_malformed/run.sh" "$T/sm/run.sh"
printf 'import sys\nsys.exit(1)\n' > "$T/sm/gen.py"
if sh "$T/sm/run.sh" /bin/true "$T/sm/work" > "$T/sm.log" 2>&1; then
    echo "FAIL self_malformed: a generator that made no cases still exited 0"; cat "$T/sm.log"; bad=1
fi

[ "$bad" -eq 0 ] && echo "audit_scripts: ok"
exit $bad
