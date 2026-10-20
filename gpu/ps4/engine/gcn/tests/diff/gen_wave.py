#!/usr/bin/env python3
"""Random compute programs for the wave-mode differential test (gcn_wave_diff.cpp), assembled with llvm-mc
(-mcpu=bonaire) into programs_wave.txt.

Each program is one 64-lane wave (workgroup 64x1x1): v0 = lane id, s0-s11 random user data, s[12:15] the V# of an
output buffer. A body of random scalar/vector ALU instructions, lane masks used as data (read and written as
dwords, bit counted, scanned), cross-lane ops (v_readlane / v_readfirstlane / v_writelane / v_mbcnt with real masks,
ds_swizzle_b32 patterns) and control flow on EXEC / VCC (saveexec if/else, divergent and uniform loops), then an
epilogue that stores v0-v23 (and the masks) to the buffer, one 256-byte row per register.

usage: gen_wave.py [count] [seed]
"""
import os
import random
import re
import subprocess
import sys

MC = ["llvm-mc", "-arch=amdgcn", "-mcpu=bonaire", "-show-encoding"]
here = os.path.dirname(os.path.abspath(__file__))
count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 4321)

V = [f"v{i}" for i in range(0, 12)]
SRC = [f"s{i}" for i in range(0, 12)] + ["s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23"]
DST = ["s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23"]
CONSTS = ["0", "1", "-1", "4", "63", "31", "32", "33", "-16", "5", "17"]
MASKS = ["vcc", "s[24:25]", "s[26:27]"]
MASK_LO = ["vcc_lo", "s24", "s26", "exec_lo"]
MASK_HI = ["vcc_hi", "s25", "s27", "exec_hi"]
PAIRS_DST = ["s[16:17]", "s[18:19]", "s[20:21]", "s[22:23]"]
CMP_I = ["lt", "eq", "le", "gt", "ne", "ge"]


def vdst():
    return rng.choice(V[1:])


def vsrc():
    return rng.choice(V)


def ssrc():
    r = rng.random()
    return rng.choice(SRC) if r < 0.5 else rng.choice(CONSTS)


def src0():
    return vsrc() if rng.random() < 0.6 else ssrc()


def lane_sel():
    # lane number for readlane / writelane: a constant, or an SGPR (uniform) masked to a lane
    return rng.choice(CONSTS) if rng.random() < 0.6 else rng.choice(SRC)


def swizzle_offset():
    if rng.random() < 0.4:
        sel = [rng.randrange(4) for _ in range(4)]
        return 0x8000 | sel[0] | sel[1] << 2 | sel[2] << 4 | sel[3] << 6
    return rng.randrange(0x8000) if rng.random() < 0.5 else (rng.randrange(32) | rng.randrange(32) << 5 | rng.randrange(32) << 10)


def valu():
    r = rng.random()
    if r < 0.2:
        return f"{rng.choice(['v_add_i32', 'v_sub_i32', 'v_and_b32', 'v_or_b32', 'v_xor_b32', 'v_min_u32', 'v_max_i32', 'v_lshlrev_b32', 'v_lshrrev_b32'])} {vdst()}, {src0()}, {vsrc()}"
    if r < 0.3:
        return f"{rng.choice(['v_mul_lo_u32', 'v_mul_hi_u32'])} {vdst()}, {vsrc()}, {src0()}"
    if r < 0.4:
        return f"v_cndmask_b32_e64 {vdst()}, {src0()}, {vsrc()}, {rng.choice(MASKS)}"
    if r < 0.5:
        return f"v_mad_u32_u24 {vdst()}, {vsrc()}, {vsrc()}, {vsrc()}"
    if r < 0.55:
        return f"v_{rng.choice(['add', 'mul', 'sub'])}_f32 {vdst()}, {vsrc()}, {vsrc()}"
    if r < 0.65:
        return f"v_mbcnt_lo_u32_b32_e64 {vdst()}, {rng.choice(MASK_LO + ['-1'])}, {vsrc()}"
    if r < 0.75:
        return f"v_mbcnt_hi_u32_b32_e64 {vdst()}, {rng.choice(MASK_HI + ['-1'])}, {vsrc()}"
    if r < 0.85:
        return f"v_mov_b32 {vdst()}, {rng.choice(MASK_LO + MASK_HI)}"
    return f"ds_swizzle_b32 {vdst()}, {vsrc()} offset:{swizzle_offset()}"


def lanes_op():
    r = rng.random()
    if r < 0.15:
        return f"v_readlane_b32 {rng.choice(DST)}, {vsrc()}, {lane_sel()}"
    if r < 0.28:
        return f"v_readfirstlane_b32 {rng.choice(DST)}, {vsrc()}"
    if r < 0.45:
        return f"v_writelane_b32 {vdst()}, {ssrc()}, {lane_sel()}"
    if r < 0.5:
        return f"v_writelane_b32 {vdst()}, {ssrc()}, {rng.choice(CONSTS)}"
    if r < 0.58:
        return f"s_bcnt1_i32_b64 {rng.choice(DST)}, {rng.choice(MASKS + ['exec'])}"
    if r < 0.63:
        return f"s_bcnt0_i32_b64 {rng.choice(DST)}, {rng.choice(MASKS + ['exec'])}"
    if r < 0.70:
        return f"{rng.choice(['s_ff1_i32_b64', 's_ff0_i32_b64', 's_flbit_i32_b64'])} {rng.choice(DST)}, {rng.choice(MASKS + ['exec'])}"
    if r < 0.75:
        return f"s_quadmask_b64 {rng.choice(PAIRS_DST)}, {rng.choice(MASKS + ['exec'])}"
    if r < 0.8:
        return f"s_quadmask_b32 {rng.choice(DST)}, {rng.choice(MASK_LO + MASK_HI)}"
    if r < 0.86:
        return f"s_mov_b32 {rng.choice(DST)}, {rng.choice(MASK_LO + MASK_HI)}"
    if r < 0.9:
        # a lane mask written as dwords
        return f"s_mov_b32 {rng.choice(['vcc_lo', 'vcc_hi', 's24', 's25', 's26', 's27'])}, {ssrc()}"
    if r < 0.93:
        return f"{rng.choice(['s_lshl_b64', 's_lshr_b64'])} {rng.choice(MASKS)}, {rng.choice(MASKS + ['exec'])}, {rng.choice(['1', '3', '8', '31', '32', '35'])}"
    if r < 0.96:
        return f"s_mov_b64 {rng.choice(MASKS)}, {rng.choice(PAIRS_DST + ['exec', '-1', '0', '1'])}"
    return f"{rng.choice(['s_bitset1_b64', 's_bitset0_b64'])} {rng.choice(MASKS)}, {rng.choice(['0', '5', '31', '32', '63', '17'])}"


def mask_op():
    r = rng.random()
    if r < 0.4:
        cmp = rng.choice(CMP_I)
        kind = rng.choice(["u32", "i32"])
        return f"v_cmp_{cmp}_{kind}_e64 {rng.choice(MASKS)}, {src0() if rng.random() < 0.7 else vsrc()}, {vsrc()}"
    if r < 0.5:
        return f"v_cmp_{rng.choice(CMP_I)}_u32_e32 vcc, {src0()}, {vsrc()}"
    if r < 0.7:
        return f"{rng.choice(['s_and_b64', 's_or_b64', 's_xor_b64', 's_andn2_b64', 's_orn2_b64', 's_nand_b64', 's_nor_b64', 's_xnor_b64'])} {rng.choice(MASKS)}, {rng.choice(MASKS + ['exec'])}, {rng.choice(MASKS + ['exec'])}"
    if r < 0.8:
        return f"s_not_b64 {rng.choice(MASKS)}, {rng.choice(MASKS)}"
    if r < 0.9:
        return f"s_wqm_b64 {rng.choice(MASKS)}, {rng.choice(MASKS)}"
    return f"s_cselect_b64 {rng.choice(MASKS)}, {rng.choice(MASKS)}, {rng.choice(MASKS)}"


def instruction():
    r = rng.random()
    if r < 0.4:
        return valu()
    if r < 0.7:
        return lanes_op()
    if r < 0.85:
        return mask_op()
    if r < 0.9:
        return f"{rng.choice(['s_add_u32', 's_and_b32', 's_xor_b32', 's_min_u32'])} {rng.choice(DST)}, {ssrc()}, {ssrc()}"
    return valu()


EPILOGUE_REGS = 24
PROLOGUE = [
    "v_cmp_lt_u32_e32 vcc, 31, v0",
    "v_cmp_gt_u32_e64 s[24:25], 48, v0",
    "v_cmp_ne_u32_e64 s[26:27], 0, v0",
    "v_mul_lo_u32 v1, v0, 7",
    "v_xor_b32 v2, 0x5a5a, v0",
    "v_add_i32 v3, vcc, s1, v0",
    "v_and_b32 v4, 3, v0",
    "v_mov_b32 v5, s2",
]
EPILOGUE = [
    "v_cndmask_b32_e64 v16, 0, 1, exec",
    "v_cndmask_b32_e64 v17, 0, 1, vcc",
    "v_cndmask_b32_e64 v18, 0, 1, s[24:25]",
    "v_cndmask_b32_e64 v19, 0, 1, s[26:27]",
    "s_mov_b64 exec, -1",
    "v_mov_b32 v12, s16", "v_mov_b32 v13, s17", "v_mov_b32 v14, s18", "v_mov_b32 v15, s19",
    "v_mov_b32 v20, s20", "v_mov_b32 v21, s21", "v_mov_b32 v22, s22", "v_mov_b32 v23, s23",
    "v_lshlrev_b32 v30, 2, v0",
] + [f"buffer_store_dword v{i}, v30, s[12:15], 0 offen offset:{i * 256}" for i in range(EPILOGUE_REGS)] + [
    "s_endpgm",
]

labels = [0]


def label(stem):
    labels[0] += 1
    return f"{stem}{labels[0]}"


def straight(n):
    return [instruction() for _ in range(n)]


def region(depth):
    out = straight(rng.randrange(1, 5))
    if depth < 2 and rng.random() < 0.3:
        out += if_else(depth + 1)
        out += straight(rng.randrange(0, 3))
    return out


def if_else(depth):
    save = ["s[28:29]", "s[30:31]", "s[32:33]"][depth]
    cond = "s[36:37]" if depth else "vcc"
    els, end = label("else"), label("endif")
    out = [f"v_cmp_{rng.choice(CMP_I)}_u32_e64 {cond}, {src0() if rng.random() < 0.5 else vsrc()}, {vsrc()}",
           f"s_and_saveexec_b64 {save}, {cond}",
           f"s_xor_b64 {save}, exec, {save}",
           f"s_cbranch_execz {els}"]
    out += region(depth)
    out += [f"{els}:", f"s_or_saveexec_b64 {save}, {save}", f"s_xor_b64 exec, exec, {save}", f"s_cbranch_execz {end}"]
    if rng.random() < 0.7:
        out += region(depth)
    out += [f"{end}:", f"s_or_b64 exec, exec, {save}"]
    return out


def uniform_loop():
    top = label("loop")
    trips = rng.randrange(1, 4)
    return ["s_mov_b32 s40, 0", f"{top}:"] + region(1) + ["s_add_u32 s40, s40, 1", f"s_cmp_lt_u32 s40, {trips}", f"s_cbranch_scc1 {top}"]


def divergent_loop():
    top = label("dloop")
    return ["v_and_b32 v25, 7, v0", "v_add_i32 v25, vcc, 1, v25", "v_mov_b32 v24, 0", "s_mov_b64 s[38:39], exec",
            f"{top}:"] + region(2) + ["v_add_i32_e64 v24, s[34:35], 1, v24", "v_cmp_ge_u32_e64 s[34:35], v24, v25",
                                    "s_andn2_b64 exec, exec, s[34:35]", f"s_cbranch_execnz {top}", "s_mov_b64 exec, s[38:39]"]


def vcc_branch():
    # a branch on VCC being zero / nonzero over a short region
    skip = label("skip")
    cond = rng.choice(["s_cbranch_vccz", "s_cbranch_vccnz"])
    return [f"v_cmp_{rng.choice(CMP_I)}_u32_e32 vcc, {rng.choice(CONSTS)}, {vsrc()}", f"{cond} {skip}"] + straight(rng.randrange(1, 4)) + [f"{skip}:"]


def body():
    out = straight(rng.randrange(3, 8))
    for _ in range(rng.randrange(1, 4)):
        k = rng.random()
        out += if_else(0) if k < 0.4 else uniform_loop() if k < 0.55 else divergent_loop() if k < 0.7 else vcc_branch()
        out += straight(rng.randrange(1, 5))
    return out


def assemble_obj(lines):
    import shutil, tempfile
    objcopy = shutil.which("llvm-objcopy") or "llvm-objcopy-18"
    with tempfile.TemporaryDirectory() as tmp:
        obj, raw = os.path.join(tmp, "p.o"), os.path.join(tmp, "p.bin")
        out = subprocess.run(["llvm-mc", "-arch=amdgcn", "-mcpu=bonaire", "-filetype=obj", "-o", obj],
                             input="\n".join(lines) + "\n", capture_output=True, text=True)
        if out.returncode or "error" in out.stderr:
            return None, out.stderr
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        data = open(raw, "rb").read()
    return [int.from_bytes(data[j:j + 4], "little") for j in range(0, len(data), 4)], ""


def assemble(lines):
    out = subprocess.run(MC, input="\n".join(lines) + "\n", capture_output=True, text=True)
    return {int(m.group(1)) - 1 for m in re.finditer(r"<stdin>:(\d+):\d+: error", out.stderr)}


programs = []
while len(programs) < count:
    b = body()
    plain = [l for l in b if not l.endswith(":")]
    bad = assemble(plain)
    badlines = {plain[i] for i in bad}
    b = [l for l in b if l not in badlines]
    flat, err = assemble_obj(PROLOGUE + b + EPILOGUE)
    if flat is None:
        continue
    programs.append((b, flat))

with open(os.path.join(here, "programs_wave.txt"), "w") as f:
    f.write("# generated by gen_wave.py: one program per 'prog' line; '#' lines show its source\n")
    for b, flat in programs:
        for l in b:
            f.write("# " + l + "\n")
        f.write("prog " + " ".join(f"{x:08x}" for x in flat) + "\n")
print(len(programs), "programs")
