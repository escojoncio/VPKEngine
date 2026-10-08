#!/usr/bin/env python3
"""Differential tests for vpaot: tests/aot/cases/*.s -> native x86-64 vs translated C.

  python3 tests/aot/run.py [--vpaot build/vpaot/vpaot] [--write-golden] [case...]

On x86-64 every case is run natively and translated and the states are compared; with
--write-golden the native state is saved to tests/aot/golden/<case>.txt. On other hosts the
translated state is compared with the golden file.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests" / "aot" / "cases"
GOLDEN = ROOT / "tests" / "aot" / "golden"
BUILD = ROOT / "build" / "aot-tests"
CODE_BASE = 0x400000
CC = os.environ.get("CC", "cc")


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout}{r.stderr}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vpaot", default=str(ROOT / "build" / "vpaot" / "vpaot"))
    ap.add_argument("--write-golden", action="store_true")
    ap.add_argument("cases", nargs="*")
    args = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    x86 = platform.machine() in ("x86_64", "AMD64")
    cases = [CASES / f"{c}.s" if not c.endswith(".s") else Path(c) for c in args.cases] or sorted(CASES.glob("*.s"))
    failed = []
    for case in cases:
        name = case.stem
        out = BUILD / name
        out.mkdir(exist_ok=True)
        try:
            run(["as", "--64", "-o", out / "code.o", case])
            run(["objcopy", "-O", "binary", "-j", ".text", out / "code.o", out / "code.bin"])
            r = run([args.vpaot, "--raw", out / "code.bin", "--base", hex(CODE_BASE), "--entry", hex(CODE_BASE),
                     "--out", out / "code.c", "--stats", out / "stats.json"])
            sources = [ROOT / "runtime" / "vp_host.c", ROOT / "tests" / "aot" / "harness.c", out / "code.c"]
            if x86:
                sources.append(ROOT / "tests" / "aot" / "native_x86.c")
            run([CC, "-O2", "-g", "-std=gnu11", "-Wall", "-Wno-unused-variable", "-Wno-unused-but-set-variable",
                 "-Wno-unused-label", "-I", ROOT / "runtime", "-o", out / "harness", *sources, "-lm"])
            golden = GOLDEN / f"{name}.txt"
            cmd = [out / "harness", out / "code.bin"]
            if x86 and args.write_golden:
                cmd += ["--write-golden", golden]
            elif golden.exists():
                cmd += ["--golden", golden]
            elif not x86:
                raise RuntimeError("no golden file and no native x86 to compare with")
            run(cmd)
            print(f"ok   {name}")
        except RuntimeError as e:
            failed.append(name)
            print(f"FAIL {name}\n{e}")
    print(f"{len(cases) - len(failed)}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
