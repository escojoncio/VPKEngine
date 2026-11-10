#!/usr/bin/env python3
"""Directed MUBUF fetch programs (programs_buf_fetch.txt) for the guarded buffer loads of the lifter: the multi-dword
loads (buffer_load_dwordx2/x3/x4) check the whole access once and read without a guard when it is inside the V# (else
dword by dword: out of range reads 0, an access that straddles the end keeps the dwords inside), the typed ones
(buffer_load_format_*) read through a clamped address and select the result. Same program layout and harness as
programs_buf.txt (gen_buf.py): gcn_wave_diff --buf compares registers, test buffer and LDS with the reference.

Extra lines understood by the harness: "ifbase" marks the program whose lifted module has no load (its count of `if`
statements is the baseline), "ifmax K" says the program may add at most K `if` statements to that baseline, "oobmax K"
that the lifted code may read at most K times (summed over the 64 lanes) past the 8 KiB of the test buffer (a read the
evaluator answers with 0, so only this count tells that a load of an out of range access was sent to an address of the
V# instead of being clamped to 0), "noref" that the registers are not compared with the reference (which does not wrap
the address at 2^32 as the 32-bit lifted code does), "dispatch" that the program must lift to the dispatch loop (its
branches do not nest) and the loads in it must not use the `if` / `else` form of the whole-access check (that form
stays out of the dispatch loop: the per-dword guards of VPGCN_FETCH_GUARDS=1 are used there).

usage: gen_buf_fetch.py   -> programs_buf_fetch.txt
"""
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('OUT', 'programs_buf_fetch.txt')

# (the prologue / epilogue / assembler of gen_buf.py)
PROLOGUE = [
    "v_lshlrev_b32 v10, 2, v0",
    "v_mov_b32 v11, v0",
    "v_and_b32 v13, 7, v0",
    "v_lshlrev_b32 v12, 2, v13",
    "v_and_b32 v13, 15, v0",
    "v_mov_b32 v26, v0",
    "v_lshlrev_b32 v27, 2, v0",
    "v_lshrrev_b32 v28, 1, v0",
    "v_and_b32 v29, 1, v0",
    "v_lshlrev_b32 v29, 3, v29",
    "v_lshlrev_b32 v32, 2, v0",
    "v_mov_b32 v33, 0",
    "v_lshlrev_b32 v34, 3, v0",
    "v_cmp_lt_u32 vcc, 59, v0",
    "v_mov_b32 v36, 1",
    "v_cndmask_b32 v35, 0, v36, vcc",
    "s_mov_b32 s21, 64",
    "s_mov_b32 s22, 0x100",
    "s_mov_b32 s23, 0x20c",
    "s_mov_b32 s20, 0x9e3779b1",
    "v_mul_lo_u32 v1, v0, s20",
    "v_xor_b32 v2, 0x5a5a1234, v1",
    "v_add_i32 v3, vcc, 0x7fff0000, v1",
    "v_lshrrev_b32 v4, 3, v2",
    "v_mul_lo_u32 v5, v3, 3",
    "v_mov_b32 v6, 0xffff0000",
    "v_add_i32 v7, vcc, v0, v6",
    "v_mul_f32 v8, 0x40490fdb, v0",
    "v_cvt_f32_u32 v9, v0",
    "v_and_b32 v14, 0x3f, v2",
    # (small values so inc / dec / compares hit; a quarter of the lanes get the loaded-data-like small ones)
    "v_cmp_gt_u32 vcc, 16, v13",
    "v_cndmask_b32 v5, v5, v13, vcc",
    "v_cndmask_b32 v6, v6, v13, vcc",
]
EPILOGUE = ["s_waitcnt vmcnt(0)", "s_barrier", "s_mov_b64 exec, -1", "v_lshlrev_b32 v30, 2, v0",
            "v_add_i32 v31, vcc, 0x1000, v30"] + \
    [f"buffer_store_dword v{i}, v{30 if i < 16 else 31}, s[12:15], 0 offen offset:{(i % 16) * 256}" for i in range(24)] + ["s_endpgm"]


def assemble(lines):
    plain = [l.replace("LDSBIT ", "") for l in lines]
    out = subprocess.run(["llvm-mc", "-arch=amdgcn", "-mcpu=bonaire", "-show-encoding"], input="\n".join(plain) + "\n",
                         capture_output=True, text=True)
    bad = {int(m.group(1)) - 1 for m in re.finditer(r"<stdin>:(\d+):\d+: error", out.stderr)}
    enc = [l for l in out.stdout.splitlines() if "; encoding:" in l]
    res, it = [], iter(enc)
    for i in range(len(lines)):
        if i in bad:
            res.append(None)
            continue
        e = next(it).split("; encoding:")[1].strip()[1:-1]
        b = [int(x, 16) for x in e.split(",")]
        w = [b[j] | b[j + 1] << 8 | b[j + 2] << 16 | b[j + 3] << 24 for j in range(0, len(b), 4)]
        if lines[i].startswith("LDSBIT "):
            w[0] |= 1 << 16  # MUBUF lds bit
        res.append(w)
    return res



def vsharp(stride=0, swz=False, tid=False, es=2, istr=1, num=8192):
    w1 = (stride << 16) | (0x80000000 if swz else 0)
    w3 = (4 | 5 << 3 | 6 << 6 | 7 << 9) | es << 19 | istr << 21 | (1 << 23 if tid else 0)
    return [0, w1, num, w3]


def load(op, nregs, addr, soff="0", inst_off=0):
    """addr: (mode, register(s))."""
    mode, reg = addr
    dst = f"v[1:{nregs}]" if nregs > 1 else "v1"
    if mode == "addr64":
        s = f"buffer_{op} {dst}, {reg}, s[16:19], {soff} addr64"
    elif mode == "none":
        s = f"buffer_{op} {dst}, off, s[16:19], {soff}"
    else:
        s = f"buffer_{op} {dst}, {reg}, s[16:19], {soff} " + {"offen": "offen", "idxen": "idxen", "both": "idxen offen"}[mode]
    return s + (f" offset:{inst_off}" if inst_off else "")


NONE, OFFEN, IDX = ("none", None), ("offen", "v10"), ("idxen", "v11")
OFFEN_SMALL, BOTH, BOTH2 = ("offen", "v12"), ("both", "v[26:27]"), ("both", "v[28:29]")
ADDR64, ADDR64_HI = ("addr64", "v[32:33]"), ("addr64", "v[34:35]")

cases = []  # (description, V#, body, extra ifs, reads past the buffer, compared with the reference)


def add(desc, vs, op, n, addr, soff="0", inst_off=0, ifs=1, oob=0, ref=True, setup=()):
    cases.append((desc, vs, list(setup) + [load(op, n, addr, soff, inst_off), "s_waitcnt vmcnt(0)", "s_barrier"], ifs, oob, ref))


# no load at all: the baseline
cases.append(("baseline", vsharp(), ["s_waitcnt vmcnt(0)", "s_barrier"], None, 0, True))
# an access that is entirely in / entirely out / straddles the end of the records (dwords 0-1 valid, 2-3 read 0)
for n, op in ((2, "load_dwordx2"), (3, "load_dwordx3"), (4, "load_dwordx4")):
    add(f"{op} inside", vsharp(num=8192), op, n, NONE, "0", 16)
    add(f"{op} outside", vsharp(num=0), op, n, NONE, "0", 16)
    add(f"{op} outside, first dword at the end", vsharp(num=16), op, n, NONE, "0", 16)
    for num in (20, 24, 28, 32):
        add(f"{op} straddling num_records={num}", vsharp(num=num), op, n, NONE, "0", 16)
    add(f"{op} unaligned straddling", vsharp(num=26), op, n, NONE, "0", 3 + 16)
    add(f"{op} unaligned address inside", vsharp(num=8192), op, n, NONE, "0", 6)
    add(f"{op} soffset", vsharp(num=100), op, n, NONE, "s22", 0)
    add(f"{op} per-lane offsets, some inside some straddling some outside", vsharp(num=136), op, n, OFFEN)
    add(f"{op} per-lane offsets with soffset and offset", vsharp(num=300), op, n, OFFEN, "s21", 8)
    add(f"{op} small offsets", vsharp(num=20), op, n, OFFEN_SMALL, "0", 4)
    add(f"{op} idxen stride 16", vsharp(stride=16, num=20), op, n, IDX)
    add(f"{op} idxen stride 8 (wider than the stride)", vsharp(stride=8, num=33), op, n, IDX)
    add(f"{op} idxen+offen", vsharp(stride=16, num=10), op, n, BOTH, "0", 4)
    add(f"{op} idxen+offen 2", vsharp(stride=32, num=20), op, n, BOTH2)
    add(f"{op} add_tid stride 16 offen", vsharp(stride=16, tid=True, num=40), op, n, OFFEN_SMALL)
    add(f"{op} add_tid stride 16 idxen", vsharp(stride=16, tid=True, num=40), op, n, IDX)
    add(f"{op} swizzled", vsharp(stride=32, swz=True, es=2, istr=1, num=40), op, n, OFFEN_SMALL)
    add(f"{op} swizzled idxen+offen", vsharp(stride=64, swz=True, es=3, istr=0, num=50), op, n, BOTH2)
    add(f"{op} add_tid swizzled", vsharp(stride=16, swz=True, tid=True, es=2, istr=2, num=70), op, n, OFFEN_SMALL)
    add(f"{op} addr64", vsharp(num=8192), op, n, ADDR64)
    add(f"{op} addr64 with high bits", vsharp(num=8192), op, n, ADDR64_HI, "s21", 4)
# an address far past the V# and the 8 KiB test buffer: the load of the out of range part goes to address 0, not there
FAR = ("s_mov_b32 s25, 0x10000",)
for n, op in ((2, "load_dwordx2"), (3, "load_dwordx3"), (4, "load_dwordx4")):
    add(f"{op} far outside", vsharp(num=16), op, n, NONE, "s25", 16, setup=FAR)
    add(f"{op} far outside per lane", vsharp(num=136), op, n, OFFEN, "s25", 0, setup=FAR)
# the address wraps past 2^32 inside the access (dwords at 0xfffffff8, 0xfffffffc, 0, 4): the access is not "whole" in
# range though its first and last dwords are, and the dword at 0xfffffffc (out of range) must not be read
WRAP = ("s_mov_b32 s24, 0xfffffff8",)
for n, op in ((3, "load_dwordx3"), (4, "load_dwordx4")):
    add(f"{op} wrapping address", vsharp(num=0xfffffffc), op, n, NONE, "s24", 0, oob=64, ref=False, setup=WRAP)
# typed loads of 32-bit floats (the V# names no format): `if`-free
for n, f in ((1, "x"), (2, "xy"), (3, "xyz"), (4, "xyzw")):
    op = f"load_format_{f}"
    add(f"{op} inside", vsharp(num=8192), op, n, NONE, "0", 16, 0)
    add(f"{op} straddling", vsharp(num=16 + 4 * n - 2), op, n, NONE, "0", 16, 0)
    add(f"{op} per-lane offsets", vsharp(num=136), op, n, OFFEN, "0", 0, 0)
    add(f"{op} idxen stride 16", vsharp(stride=16, num=20), op, n, IDX, "0", 0, 0)
    add(f"{op} add_tid swizzled", vsharp(stride=16, swz=True, tid=True, es=2, istr=2, num=70), op, n, OFFEN_SMALL, "0", 0, 0)
    add(f"{op} far outside", vsharp(num=16), op, n, NONE, "s25", 16, 0, setup=FAR)
    add(f"{op} far outside per lane", vsharp(num=136), op, n, OFFEN, "s25", 0, 0, setup=FAR)
# the same loads with a partial EXEC
for op, n in (("load_dwordx4", 4), ("load_format_xyzw", 4)):
    cases.append((f"{op} partial exec", vsharp(num=136),
                  ["v_cmp_gt_u32 vcc, 40, v0", "s_and_saveexec_b64 s[30:31], vcc", load(op, n, OFFEN), "s_mov_b64 exec, s[30:31]",
                   "s_waitcnt vmcnt(0)", "s_barrier"], 1 if op == "load_dwordx4" else 0, 0, True))

# the same loads inside the dispatch loop: a chain of conditional branches over the next block (each block reaches the one
# after the next or the next) does not nest, and duplicates more code than the structured form may, so the program lifts to
# the dispatch loop. The first program has no load (the baseline of the `if` statements of the dispatch loop).
CHAIN = 16


def chain(body, at, taken):
    """CHAIN blocks of `s_cmp_eq_u32 s21, K; s_cbranch_scc1 <skip the next block>` (s21 is 64: K = 64 takes the branch, 63 does
    not), `body` in block `at`; then a one-word block, then what follows."""
    blocks = [list(body) if i == at else [] for i in range(CHAIN)] + [["s_nop 0"]]
    words = [sum(2 if l.startswith("buffer_") else 1 for l in b) + 2 for b in blocks[:CHAIN]] + [1]
    out = []
    for i in range(CHAIN):
        out += blocks[i] + [f"s_cmp_eq_u32 s21, {64 if taken else 63}", f"s_cbranch_scc1 {words[i + 1]}"]
    return out + blocks[CHAIN]


def add_disp(desc, vs, op, n, addr, soff="0", inst_off=0, ifs=1, oob=0, ref=True, setup=()):
    ld = list(setup) + [load(op, n, addr, soff, inst_off), "s_waitcnt vmcnt(0)", "s_barrier"]
    cases.append((f"{desc} in the dispatch loop", vs, chain(ld, 6, True), ifs, oob, ref, True))
    cases.append((f"{desc} in the dispatch loop, every block run", vs, chain(ld, 9, False), ifs, oob, ref, True))


cases.append(("baseline of the dispatch loop", vsharp(), chain(["s_waitcnt vmcnt(0)", "s_barrier"], 6, True), None, 0, True, True))
# (one `if` per dword of the per-dword guards, one for a typed load)
for n, op in ((2, "load_dwordx2"), (3, "load_dwordx3"), (4, "load_dwordx4")):
    add_disp(f"{op} inside", vsharp(num=8192), op, n, NONE, "0", 16, ifs=n)
    add_disp(f"{op} outside", vsharp(num=0), op, n, NONE, "0", 16, ifs=n)
    add_disp(f"{op} straddling num_records=24", vsharp(num=24), op, n, NONE, "0", 16, ifs=n)
    add_disp(f"{op} per-lane offsets", vsharp(num=136), op, n, OFFEN, "0", 0, ifs=n)
    add_disp(f"{op} far outside per lane", vsharp(num=136), op, n, OFFEN, "s25", 0, ifs=n + 1, setup=FAR)  # (one `if` more than the dwords)
add_disp("load_dwordx4 idxen stride 16", vsharp(stride=16, num=20), "load_dwordx4", 4, IDX, "0", 0, ifs=4)
add_disp("load_format_xyzw per-lane offsets", vsharp(num=136), "load_format_xyzw", 4, OFFEN, "0", 0, ifs=1)
add_disp("load_format_xy straddling", vsharp(num=16 + 4 * 2 - 2), "load_format_xy", 2, NONE, "0", 16, ifs=1)
cases.append(("load_dwordx4 partial exec in the dispatch loop", vsharp(num=136),
              chain(["v_cmp_gt_u32 vcc, 40, v0", "s_and_saveexec_b64 s[30:31], vcc", load("load_dwordx4", 4, OFFEN), "s_mov_b64 exec, s[30:31]",
                     "s_waitcnt vmcnt(0)", "s_barrier"], 6, True), 4, 0, True, True))

with open(os.path.join(here, OUT), "w") as f:
    f.write("# generated by gen_buf_fetch.py: one program per 'prog' line; '#' lines show its source\n")
    for desc, vs, body, ifs, oob, ref, *disp in cases:
        lines = PROLOGUE + body + EPILOGUE
        words = assemble(lines)
        if any(w is None for w in words):
            sys.exit("cannot assemble: " + [l for l, w in zip(lines, words) if w is None][0])
        f.write(f"# {desc}\n")
        for l in body:
            f.write("# " + l + "\n")
        f.write("vsharp " + " ".join(f"{x:08x}" for x in vs) + "\n")
        f.write("ifbase\n" if ifs is None else f"ifmax {ifs}\noobmax {oob}\n")
        if not ref:
            f.write("noref\n")
        if disp:
            f.write("dispatch\n")
        f.write("prog " + " ".join(f"{x:08x}" for w in words for x in w) + "\n")
print(len(cases), "programs")
