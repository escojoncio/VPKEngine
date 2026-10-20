#!/usr/bin/env python3
"""Random LDS programs for the wave-mode differential test (gcn_wave_diff --lds 2048): one 64-lane wave, 2 KiB of LDS
initialised by the program, a body of DS operations (the whole Sea Islands DS set but GDS, GWS and f64), and an epilogue
that dumps the LDS and the registers to the buffer.

Operations whose result depends on the order of the lanes use per-lane addresses (v10 = lane * 4, v11 = lane * 8);
commutative ones may also share a small window (v12 = (lane & 7) * 4, v13 = (lane & 7) * 8, row 7). 64-bit read-modify-
write runs lane by lane in the lifted code, in lane order, as the reference does: any addresses.

usage: gen_ds.py [count] [seed] [src2]   -> programs_ds.txt (programs_ds_src2.txt)
"""
import os
import random
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 99)
SRC2 = len(sys.argv) > 3 and sys.argv[3] == "src2"

D32 = [f"v{i}" for i in range(1, 10)]
D64 = [f"v[{i}:{i + 1}]" for i in range(1, 9)]
D128 = ["v[1:4]", "v[5:8]"]


def d32():
    return rng.choice(D32)


def d64():
    return rng.choice(D64)


def row_off(rows=range(0, 7)):
    return 256 * rng.choice(list(rows))


def addr32(shared_ok):
    if shared_ok and rng.random() < 0.5:
        return "v12", 256 * rng.randrange(8) + rng.choice([0, 0, 4, 8])
    return "v10", row_off()


def addr64(shared_ok):
    if shared_ok and rng.random() < 0.5:
        return "v13", 256 * rng.randrange(7)
    return "v11", rng.choice([0, 512, 1024]) if rng.random() < 0.8 else 8 * rng.randrange(0, 32)


COMM32 = ["add_u32", "sub_u32", "min_i32", "max_i32", "min_u32", "max_u32", "and_b32", "or_b32", "xor_b32"]
ORD32 = ["rsub_u32", "inc_u32", "dec_u32", "mskor_b32", "write_b32", "cmpst_b32", "cmpst_f32", "min_f32", "max_f32"]
RTN32 = ["add", "sub", "rsub", "inc", "dec", "min_i32", "max_i32", "min_u32", "max_u32", "and", "or", "xor", "mskor",
         "wrxchg", "cmpst_b32", "cmpst_f32", "min_f32", "max_f32", "wrap"]


def op32():
    r = rng.random()
    if r < 0.25:
        o = rng.choice(COMM32)
        a, off = addr32(True)
        return f"ds_{o} {a}, {d32()} offset:{off}"
    if r < 0.45:
        o = rng.choice(ORD32)
        a, off = addr32(False)
        if o.startswith("mskor") or o.startswith("cmpst"):
            return f"ds_{o} {a}, {d32()}, {d32()} offset:{off}"
        return f"ds_{o} {a}, {d32()} offset:{off}"
    if r < 0.7:
        o = rng.choice(RTN32)
        a, off = addr32(False)
        name = {"add": "add_rtn_u32", "sub": "sub_rtn_u32", "rsub": "rsub_rtn_u32", "inc": "inc_rtn_u32", "dec": "dec_rtn_u32",
                "and": "and_rtn_b32", "or": "or_rtn_b32", "xor": "xor_rtn_b32", "mskor": "mskor_rtn_b32",
                "wrxchg": "wrxchg_rtn_b32", "cmpst_b32": "cmpst_rtn_b32", "cmpst_f32": "cmpst_rtn_f32", "wrap": "wrap_rtn_b32",
                "min_i32": "min_rtn_i32", "max_i32": "max_rtn_i32", "min_u32": "min_rtn_u32", "max_u32": "max_rtn_u32",
                "min_f32": "min_rtn_f32", "max_f32": "max_rtn_f32"}[o]
        if o in ("mskor", "cmpst_b32", "cmpst_f32", "wrap"):
            return f"ds_{name} {d32()}, {a}, {d32()}, {d32()} offset:{off}"
        return f"ds_{name} {d32()}, {a}, {d32()} offset:{off}"
    if r < 0.8:
        a, off = addr32(False)
        k = rng.choice(["read_b32", "read_i8", "read_u8", "read_i16", "read_u16"])
        extra = rng.choice([0, 1, 2, 3]) if k.endswith("8") else rng.choice([0, 2]) if k.endswith("16") else 0
        return f"ds_{k} {d32()}, {a} offset:{off + extra}"
    if r < 0.85:
        o0, o1 = rng.sample([0, 64, 128, 192], 2)
        return f"ds_write2_b32 v10, {d32()}, {d32()} offset0:{o0} offset1:{o1}"
    if r < 0.9:
        o0, o1 = rng.sample([0, 64, 128, 192], 2)
        return f"ds_read2_b32 v[1:2], v10 offset0:{o0} offset1:{o1}" if rng.random() < 0.5 else \
            f"ds_wrxchg2_rtn_b32 v[1:2], v10, {d32()}, {d32()} offset0:{o0} offset1:{o1}"
    if r < 0.93:
        o0, o1 = rng.sample([0, 1, 2], 2)
        k = rng.choice(["ds_write2st64_b32 v10, %s, %s" % (d32(), d32()), "ds_read2st64_b32 v[1:2], v10",
                        "ds_wrxchg2st64_rtn_b32 v[1:2], v10, %s, %s" % (d32(), d32())])
        return f"{k} offset0:{o0} offset1:{o1}"
    if r < 0.96:
        return rng.choice([f"ds_write_b8 v10, {d32()} offset:{row_off() + rng.randrange(4)}",
                           f"ds_write_b16 v10, {d32()} offset:{row_off() + rng.choice([0, 2])}",
                           f"ds_write_b96 v14, v[1:3] offset:{rng.choice([0, 1024])}", f"ds_write_b128 v14, v[1:4] offset:{rng.choice([0, 1024])}"])
    return rng.choice([f"ds_read_b96 v[1:3], v14 offset:{rng.choice([0, 1024])}", f"ds_read_b128 v[1:4], v14 offset:{rng.choice([0, 1024])}"])


COMM64 = ["add_u64", "sub_u64", "min_i64", "max_i64", "min_u64", "max_u64", "and_b64", "or_b64", "xor_b64",
          "rsub_u64", "inc_u64", "dec_u64"]


def op64():
    r = rng.random()
    a, off = addr64(True)
    if r < 0.3:
        return f"ds_{rng.choice(COMM64)} {a}, {d64()} offset:{off}"
    if r < 0.33:
        return f"ds_write_b64 v11, {d64()} offset:{rng.choice([0, 512, 1024])}"  # (plain stores: per-lane addresses only)
    if r < 0.4:
        return f"ds_{rng.choice(['mskor_b64', 'cmpst_b64'])} {a}, {d64()}, {d64()} offset:{off}"
    if r < 0.7:
        o = rng.choice(["add", "sub", "rsub", "inc", "dec", "min_i", "max_i", "min_u", "max_u", "and", "or", "xor", "wrxchg"])
        suffix = {"add": "add_rtn_u64", "sub": "sub_rtn_u64", "rsub": "rsub_rtn_u64", "inc": "inc_rtn_u64",
                  "dec": "dec_rtn_u64", "min_i": "min_rtn_i64", "max_i": "max_rtn_i64", "min_u": "min_rtn_u64",
                  "max_u": "max_rtn_u64", "and": "and_rtn_b64", "or": "or_rtn_b64", "xor": "xor_rtn_b64",
                  "wrxchg": "wrxchg_rtn_b64"}[o]
        return f"ds_{suffix} {d64()}, {a}, {d64()} offset:{off}"
    if r < 0.78:
        return f"ds_{rng.choice(['mskor_rtn_b64', 'cmpst_rtn_b64'])} {d64()}, {a}, {d64()}, {d64()} offset:{off}"
    if r < 0.88:
        return f"ds_read_b64 {d64()}, v11 offset:{rng.choice([0, 512, 1024])}"
    if r < 0.92:
        o0, o1 = rng.sample([0, 64, 128], 2)
        return rng.choice([f"ds_write2_b64 v11, {d64()}, {d64()} offset0:{o0} offset1:{o1}",
                           f"ds_read2_b64 v[1:4], v11 offset0:{o0} offset1:{o1}",
                           f"ds_wrxchg2_rtn_b64 v[1:4], v11, {d64()}, {d64()} offset0:{o0} offset1:{o1}"])
    if r < 0.95:
        o0, o1 = rng.sample([0, 1, 2], 2)
        return rng.choice([f"ds_write2st64_b64 v11, {d64()}, {d64()} offset0:{o0} offset1:{o1}",
                           f"ds_read2st64_b64 v[1:4], v11 offset0:{o0} offset1:{o1}",
                           f"ds_wrxchg2st64_rtn_b64 v[1:4], v11, {d64()}, {d64()} offset0:{o0} offset1:{o1}"])
    return f"ds_condxchg32_rtn_b64 {d64()}, v11, {d64()} offset:{rng.choice([0, 512])}"


SRC2_OPS = ["add_src2_u32", "sub_src2_u32", "rsub_src2_u32", "inc_src2_u32", "dec_src2_u32", "min_src2_i32", "max_src2_i32",
            "min_src2_u32", "max_src2_u32", "and_src2_b32", "or_src2_b32", "xor_src2_b32", "write_src2_b32", "min_src2_f32",
            "max_src2_f32"]


def op_src2():
    # A = v10 + (nothing): row 0 (written), B = A + 256 * k bytes (row k, read only: count 64 * k dwords)
    k = rng.choice([1, 2, 3])
    cnt = 64 * k
    return f"ds_{rng.choice(SRC2_OPS)} v10 offset:{cnt}"


def instruction():
    if SRC2:
        return op_src2()
    return op32() if rng.random() < 0.65 else op64()


PROLOGUE = [
    "v_lshlrev_b32 v10, 2, v0",
    "v_lshlrev_b32 v11, 3, v0",
    "v_and_b32 v12, 7, v0",
    "v_lshlrev_b32 v13, 3, v12",
    "v_lshlrev_b32 v12, 2, v12",
    "v_lshlrev_b32 v14, 4, v0",
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
] + [f"ds_write_b32 v10, v{1 + (r % 9)} offset:{256 * r}" for r in range(8)] + [
    "s_waitcnt lgkmcnt(0)",
    "s_barrier",
]
EPILOGUE = ["s_waitcnt lgkmcnt(0)", "s_barrier"] + [f"ds_read_b32 v{16 + r}, v10 offset:{256 * r}" for r in range(8)] + [
    "s_waitcnt lgkmcnt(0)",
    "v_lshlrev_b32 v30, 2, v0",
    "v_add_i32 v31, vcc, 0x1000, v30",  # (the offset field has 12 bits: registers 16-23 go through a second base)
] + [f"buffer_store_dword v{i}, v{30 if i < 16 else 31}, s[12:15], 0 offen offset:{(i % 16) * 256}" for i in range(24)] + ["s_endpgm"]


def assemble(lines):
    out = subprocess.run(["llvm-mc", "-arch=amdgcn", "-mcpu=bonaire", "-show-encoding"], input="\n".join(lines) + "\n",
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
        res.append([b[j] | b[j + 1] << 8 | b[j + 2] << 16 | b[j + 3] << 24 for j in range(0, len(b), 4)])
    return res


programs = []
while len(programs) < count:
    body = []
    for _ in range(rng.randrange(2, 10)):
        # (a barrier after each op: the test's threads only meet at barriers, a wave runs its instructions in lockstep)
        body += [instruction(), "s_waitcnt lgkmcnt(0)", "s_barrier"]
    lines = PROLOGUE + body + EPILOGUE
    words = assemble(lines)
    if any(w is None for w in words):
        bad = [l for l, w in zip(lines, words) if w is None]
        sys.exit("cannot assemble: " + bad[0])
    programs.append((body, [x for w in words for x in w]))

with open(os.path.join(here, "programs_ds_src2.txt" if SRC2 else "programs_ds.txt"), "w") as f:
    f.write("# generated by gen_ds.py: one program per 'prog' line; '#' lines show its source\n")
    for body, flat in programs:
        for l in body:
            f.write("# " + l + "\n")
        f.write("prog " + " ".join(f"{x:08x}" for x in flat) + "\n")
print(len(programs), "programs")
