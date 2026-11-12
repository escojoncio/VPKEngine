#!/usr/bin/env python3
"""Randomised differential test of vpaot on an x86-64 host.

  python3 tests/aot/fuzz/fuzz.py --vpaot build/vpaot/vpaot --enumerate build/vpaot/isa_enumerate \
          [--seeds 200] [--seed0 1] [--profile 7] [--work DIR] [--filter TEXT] [--ext EXT[,EXT]]

Every instruction form of the Jaguar ISA that can run alone (tools/isa_audit, "test" forms: operands
rax/rcx/xmm0/xmm1/xmm14, memory [rdi+0x40]) is translated once; tests/aot/fuzz/fuzz_harness.c then runs
each from --seeds random start states, natively and translated, and compares registers, defined
flags, vector registers, MXCSR control bits, x87 state and the memory it can reach. Exits 1 on any
difference, printing the slot (instruction), seed and first differing field. Replay one with
--filter "<text>" --seed0 <seed> --seeds 1 (FUZZ_VERBOSE=1 also prints the start state).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = 0x400000
SHIFTS = {"shl", "shr", "sar", "rol", "ror", "rcl", "rcr", "shld", "shrd"}


FLOATOP = re.compile(r"^v?(add|sub|mul|div|sqrt|hadd|hsub|addsub|round|dp|cvt)[a-z0-9]*?(ps|pd|ss|sd)$")
APPROX = re.compile(r"^v?(rcp|rsqrt)(ps|ss)$")


def kind_of(mn):
    """How the harness compares a vector result: bit for bit (the translation follows Intel's NaN rules and
    tables); FUZZ_LOOSE=1 ignores NaN payloads and accepts a 12-bit error in rcp/rsqrt, for a host that is not Intel."""
    if not os.environ.get("FUZZ_LOOSE"):
        return 0
    if APPROX.match(mn):
        return -32
    if FLOATOP.match(mn) and not re.search(r"2(si|dq|pi)$", mn):
        return 32 if mn.endswith(("ps", "ss")) else 64
    return 0


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout}{r.stderr}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vpaot", default=str(ROOT / "build" / "vpaot" / "vpaot"))
    ap.add_argument("--enumerate", default=str(ROOT / "build" / "vpaot" / "isa_enumerate"))
    ap.add_argument("--work", default=str(ROOT / "build" / "aot-fuzz"))
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--profile", type=int, default=7, help="1 float-edge vectors, 2 x87 stack, 4 DF, 8 MXCSR DAZ/FTZ (not modelled by the translation)")
    ap.add_argument("--filter", default="")
    ap.add_argument("--ext", default="", help="comma list of ISA extensions (BASE, SSE2, AVX, X87, ...)")
    ap.add_argument("--extra", type=int, default=-1, help="instead of the ISA forms: the slots of forms_extra.py, from this seed")
    ap.add_argument("--sequences", type=int, default=400, help="with --extra: number of flag-sequence slots")
    ap.add_argument("--vpflags", default=os.environ.get("VPFLAGS", ""))
    ap.add_argument("--cflags", default=os.environ.get("VPCFLAGS", ""))
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    isa = work / "isa"
    isa.mkdir(exist_ok=True)
    slots = []
    if args.extra >= 0:
        # An image of our own: every slot is a few Intel-syntax instructions and a ret.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import forms_extra
        items = forms_extra.slots(args.extra, args.sequences)
        keep = list(range(len(items)))
        while True:
            src = [".intel_syntax noprefix", ".text", ".globl _start", "_start:"]
            first_line = {}
            for i in keep:
                text = items[i][0]
                if args.filter and args.filter not in text:
                    continue
                src += [".balign 128", f"slot_{i}:"]
                first_line[len(src)] = i
                src += [f"    {t.strip()}" for t in text.split(";") if t.strip()] + ["    ret"]
            (work / "extra.s").write_text("\n".join(src) + "\n")
            r = subprocess.run(["as", "--64", "-o", str(work / "extra.o"), str(work / "extra.s")], capture_output=True, text=True)
            if r.returncode == 0:
                break
            # drop the slots the assembler rejects (an operand mix that has no encoding, such as ah with r8b)
            bad = set()
            starts = sorted(first_line)
            for m in re.finditer(r"extra\.s:(\d+): Error", r.stderr):
                ln = int(m.group(1))
                owner = [l for l in starts if l <= ln]
                if owner:
                    bad.add(first_line[owner[-1]])
            if not bad:
                raise RuntimeError(r.stderr)
            keep = [i for i in keep if i not in bad]
        slots = [(i, items[i][1], items[i][2], items[i][0]) for i in keep if not args.filter or args.filter in items[i][0]]
        run(["ld", "-static", "-T", ROOT / "tests" / "aot" / "link.ld", "-o", work / "extra.elf", work / "extra.o"])
        run(["objcopy", "-O", "binary", work / "extra.elf", isa / "code.bin"])
        syms = {l.split()[2]: int(l.split()[0], 16) - BASE for l in run(["nm", work / "extra.elf"]).stdout.splitlines() if l.split()[-1].startswith("slot_")}
        slots = [(syms[f"slot_{i}"], undef, kind, f"EXTRA: {text}") for i, undef, kind, text in slots]
    else:
        if not (isa / "forms.txt").exists():
            run([args.enumerate, isa])
        exts = set(args.ext.upper().split(",")) - {""}
        for line in open(isa / "forms.txt"):
            off, ext, mnem, text, hexbytes, undef, test = line.rstrip("\n").split("\t")
            # x87 forms are not marked "test" (they need a populated stack); the harness gives them one.
            if (test != "test" and ext != "X87") or (exts and ext not in exts) or "[rcx]" in text:
                continue
            undef = int(undef, 16)
            mn = text.split()[0]
            if mn in SHIFTS and text.endswith(", 0x01"):
                undef &= ~0x800 # OF is defined for a count of 1 (the table cannot know the count)
            if mn in ("shld", "shrd") and text.endswith(", cl") and re.search(r"\b(ax|cx|dx|bx)\b|word ptr|\[rdi", text) and "dword" not in text and "qword" not in text and not re.search(r"\be[a-d]x\b|\br[a-d]x\b", text):
                continue # 16-bit double shift by a count above 16: undefined in the ISA
            if args.filter and args.filter not in text:
                continue
            slots.append((int(off, 16), undef, kind_of(mn), f"{ext}: {text}"))
    (work / "slots.txt").write_text("".join(f"{off} {undef:x} {kind} {text}\n" for off, undef, kind, text in slots))
    entries = []
    for off, _, _, _ in slots:
        entries += ["--entry", hex(BASE + off)]
    run([args.vpaot, "--raw", isa / "code.bin", "--base", hex(BASE), *entries, "--out", work / "code.c",
         "--stats", work / "stats.json", *args.vpflags.split()])
    srcs = [ROOT / "runtime" / "vp_host.c", ROOT / "tests" / "aot" / "fuzz" / "fuzz_harness.c",
            ROOT / "tests" / "aot" / "fuzz" / "fuzz_native.c", work / "code.c"]
    run([os.environ.get("CC", "cc"), "-O2", "-g", "-std=gnu11", "-frounding-math", "-Wall", "-Wno-unused-variable",
         "-Wno-unused-but-set-variable", "-Wno-unused-label", "-Wno-unused-function", "-I", ROOT / "runtime",
         *args.cflags.split(), "-o", work / "fuzz", *srcs, "-lm", "-latomic", "-lpthread"])
    r = subprocess.run([str(work / "fuzz"), str(isa / "code.bin"), str(work / "slots.txt"), str(args.seeds),
                        str(args.seed0), str(args.profile), ""], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
