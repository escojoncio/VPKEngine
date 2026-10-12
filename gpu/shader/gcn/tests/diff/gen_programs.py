#!/usr/bin/env python3
"""Random straight-line GCN programs for the lifter's differential test (gcn_diff.cpp), assembled with
llvm-mc (-mcpu=bonaire) into programs.txt. Run it after changing the op pools; the test reads
programs.txt only.

Each program: a vertex shader with 16 user SGPRs (random data) and v0 = lane index. A body of
random scalar/vector ALU instructions over small register pools (so results feed each other),
including EXEC/VCC mask manipulation, then an epilogue that restores EXEC, copies SGPRs and the
lane's VCC/EXEC bits into VGPRs and exports v0-v23 as param0-5. The reference interpreter (one
64-lane wave) and the lifted VPIR (64 invocations) must export the same bits.

Ops whose per-lane lifting is an approximation by design (cross-lane reads, lane counts, wave-wide
SCC from masks, transcendentals, f64, div_scale/fixup) are left out; see the pools below.

usage: gen_programs.py [count] [seed]       -> programs.txt next to this script
       gen_programs.py count seed cf       -> programs_cf.txt: the same with control flow (if/else in
       LLVM's saveexec idiom, nested; uniform loops on SCC; per-lane loops leaving through EXEC)
"""
import os
import random
import re
import subprocess
import sys

MC = ["llvm-mc", "-arch=amdgcn", "-mcpu=bonaire", "-show-encoding"]
here = os.path.dirname(os.path.abspath(__file__))
count = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1234)

V = [f"v{i}" for i in range(0, 12)]
S = [f"s{i}" for i in range(0, 24)]
CONSTS = ["0", "1", "-1", "4", "64", "-16", "0.5", "1.0", "-2.0", "4.0"]
LITS = ["0x3fc00000", "0x7fffffff", "0x80000000", "0x12345678", "0xbf000000", "0x7f800000", "0x00800000"]


def vdst():
    return rng.choice(V[1:])  # keep v0 (lane index) intact


def vsrc():
    return rng.choice(V)


def ssrc(literal=True):
    r = rng.random()
    if r < 0.5:
        return rng.choice(S)
    if r < 0.85 or not literal:
        return rng.choice(CONSTS)
    return rng.choice(LITS)


def src0(literal=True):
    r = rng.random()
    if r < 0.55:
        return vsrc()
    return ssrc(literal)


def mods(text):
    r = rng.random()
    if r < 0.15:
        return "-" + text
    if r < 0.25:
        return f"|{text}|"
    return text


FLOAT2 = ["v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_min_f32", "v_max_f32", "v_mul_legacy_f32",
          "v_min_legacy_f32", "v_max_legacy_f32", "v_mac_f32"]
INT2 = ["v_mul_i32_i24", "v_mul_hi_i32_i24", "v_mul_u32_u24", "v_mul_hi_u32_u24", "v_min_i32", "v_max_i32",
        "v_min_u32", "v_max_u32", "v_lshr_b32", "v_lshrrev_b32", "v_ashr_i32", "v_ashrrev_i32", "v_lshl_b32",
        "v_lshlrev_b32", "v_and_b32", "v_or_b32", "v_xor_b32", "v_bfm_b32", "v_bcnt_u32_b32", "v_cvt_pkrtz_f16_f32",
        "v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32", "v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32", "v_ldexp_f32"]
UN = ["v_mov_b32", "v_not_b32", "v_bfrev_b32", "v_ffbh_u32", "v_ffbl_b32", "v_ffbh_i32", "v_cvt_f32_i32",
      "v_cvt_f32_u32", "v_cvt_u32_f32", "v_cvt_i32_f32", "v_cvt_rpi_i32_f32", "v_cvt_flr_i32_f32", "v_cvt_f16_f32",
      "v_cvt_f32_f16", "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2", "v_cvt_f32_ubyte3",
      "v_cvt_off_f32_i4", "v_fract_f32", "v_trunc_f32", "v_ceil_f32", "v_rndne_f32", "v_floor_f32",
      "v_frexp_mant_f32", "v_frexp_exp_i32_f32"]
FLOAT_UN = {"v_cvt_f32_i32", "v_cvt_f32_u32", "v_cvt_u32_f32", "v_cvt_i32_f32", "v_cvt_rpi_i32_f32",
            "v_cvt_flr_i32_f32", "v_cvt_f16_f32", "v_fract_f32", "v_trunc_f32", "v_ceil_f32", "v_rndne_f32",
            "v_floor_f32", "v_frexp_mant_f32", "v_frexp_exp_i32_f32", "v_cvt_f32_f16"}
THREE = ["v_mad_f32", "v_fma_f32", "v_mad_legacy_f32", "v_min3_f32", "v_max3_f32", "v_med3_f32", "v_mad_i32_i24",
         "v_mad_u32_u24", "v_bfe_u32", "v_bfe_i32", "v_bfi_b32", "v_alignbit_b32", "v_alignbyte_b32",
         "v_min3_i32", "v_max3_i32", "v_min3_u32", "v_max3_u32", "v_med3_i32", "v_med3_u32", "v_lerp_u8",
         "v_sad_u32", "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32"]
FLOAT3 = {"v_mad_f32", "v_fma_f32", "v_mad_legacy_f32", "v_min3_f32", "v_max3_f32", "v_med3_f32", "v_cubeid_f32",
          "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32"}
VOP3_2 = ["v_mul_lo_u32", "v_mul_hi_u32", "v_mul_lo_i32", "v_mul_hi_i32"]
CMP_F = ["f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"]
CMP_I = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"]
SALU2 = ["s_add_u32", "s_sub_u32", "s_add_i32", "s_sub_i32", "s_min_i32", "s_min_u32", "s_max_i32", "s_max_u32",
         "s_and_b32", "s_or_b32", "s_xor_b32", "s_andn2_b32", "s_orn2_b32", "s_nand_b32", "s_nor_b32",
         "s_xnor_b32", "s_lshl_b32", "s_lshr_b32", "s_ashr_i32", "s_bfm_b32", "s_mul_i32", "s_bfe_u32",
         "s_bfe_i32", "s_absdiff_i32"]
SCC_USERS = ["s_addc_u32", "s_subb_u32", "s_cselect_b32"]
SALU1 = ["s_mov_b32", "s_not_b32", "s_brev_b32", "s_bcnt0_i32_b32", "s_bcnt1_i32_b32", "s_ff1_i32_b32",
         "s_flbit_i32_b32", "s_sext_i32_i8", "s_sext_i32_i16", "s_abs_i32"]
SCMP = ["s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
        "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
        "s_bitcmp0_b32", "s_bitcmp1_b32"]
SOPK = ["s_movk_i32", "s_addk_i32", "s_mulk_i32", "s_cmpk_eq_i32", "s_cmpk_lg_i32", "s_cmpk_gt_i32",
        "s_cmpk_lt_u32", "s_cmpk_ge_u32"]
MASKS = ["vcc", "s[24:25]", "s[26:27]"]


def instruction(state):
    """One random instruction (text); state tracks whether SCC holds a wave-uniform value."""
    r = rng.random()
    if r < 0.14:
        op = rng.choice(FLOAT2)
        if op == "v_mac_f32" or rng.random() < 0.5:
            return f"{op} {vdst()}, {src0()}, {vsrc()}"
        cl = " clamp" if rng.random() < 0.15 else ""
        om = rng.choice(["", "", "", " mul:2", " div:2"])
        return f"{op}_e64 {vdst()}, {mods(src0(False))}, {mods(src0(False))}{cl}{om}"
    if r < 0.27:
        op = rng.choice(INT2)
        return f"{op} {vdst()}, {src0()}, {vsrc()}"
    if r < 0.38:
        op = rng.choice(UN)
        if op in FLOAT_UN and op != "v_cvt_f32_i32" and op != "v_cvt_f32_u32" and rng.random() < 0.3:
            return f"{op}_e64 {vdst()}, {mods(src0(False))}"
        return f"{op} {vdst()}, {src0()}"
    if r < 0.48:
        op = rng.choice(THREE)
        a, b, c = src0(False), vsrc(), src0(False)
        if a in S and c in S and a != c:
            c = vsrc()  # one scalar value per instruction (constant bus)
        if op in FLOAT3 and rng.random() < 0.4:
            a, b, c = mods(a), mods(b), mods(c)
        return f"{op} {vdst()}, {a}, {b}, {c}"
    if r < 0.52:
        return f"{rng.choice(VOP3_2)} {vdst()}, {vsrc()}, {src0(False)}"
    if r < 0.57:
        # v_madmk / v_madak with a literal
        lit = rng.choice(LITS + ["0x40400000"])
        if rng.random() < 0.5:
            return f"v_madmk_f32 {vdst()}, {vsrc()}, {lit}, {vsrc()}"
        return f"v_madak_f32 {vdst()}, {vsrc()}, {vsrc()}, {lit}"
    if r < 0.63:
        # compares into VCC or an SGPR pair (e64), sometimes cmpx
        kind = rng.choice(["f32", "f32", "i32", "u32", "class"])
        x = "x" if rng.random() < 0.2 else ""
        if kind == "class":
            op = f"v_cmp{x}_class_f32"
        else:
            op = f"v_cmp{x}_{rng.choice(CMP_F if kind == 'f32' else CMP_I)}_{kind}"
        if rng.random() < 0.5:
            return f"{op}_e32 vcc, {src0()}, {vsrc()}"
        dst = rng.choice(MASKS)
        a, b = src0(False), src0(False)
        if a in S and b in S and a != b:
            b = vsrc()
        if kind == "f32":
            a, b = mods(a), mods(b)
        return f"{op}_e64 {dst}, {a}, {b}"
    if r < 0.67:
        # v_cndmask / carries reading and writing masks
        m = rng.choice(MASKS)
        k = rng.random()
        if k < 0.4:
            return f"v_cndmask_b32_e64 {vdst()}, {src0(False)}, {vsrc()}, {m}"
        if k < 0.6:
            return f"v_cndmask_b32_e32 {vdst()}, {src0()}, {vsrc()}, vcc"
        if k < 0.8:
            return f"v_add_i32_e64 {vdst()}, {m}, {vsrc()}, {src0(False)}"
        return f"v_addc_u32_e64 {vdst()}, {rng.choice(MASKS)}, {vsrc()}, {vsrc()}, {m}"
    if r < 0.72:
        # EXEC manipulation (never all lanes off for long: restored by the epilogue)
        k = rng.random()
        state["scc_uniform"] = False
        if k < 0.3:
            return f"s_and_saveexec_b64 {rng.choice(['s[28:29]', 's[30:31]'])}, {rng.choice(MASKS)}"
        if k < 0.5:
            return f"s_andn2_b64 exec, exec, {rng.choice(MASKS)}"
        if k < 0.7:
            return f"s_or_b64 exec, exec, {rng.choice(MASKS)}"
        if k < 0.85:
            return f"s_xor_b64 {rng.choice(MASKS)}, {rng.choice(MASKS)}, exec"
        return f"s_mov_b64 {rng.choice(MASKS)}, exec"
    if r < 0.84:
        op = rng.choice(SALU2)
        if op not in ("s_bfm_b32", "s_mul_i32"):
            state["scc_uniform"] = True  # (s_bfm_b32 and s_mul_i32 leave SCC alone)
        return f"{op} {rng.choice(S[16:])}, {ssrc()}, {ssrc(False)}"
    if r < 0.89:
        op = rng.choice(SALU1)
        if op not in ("s_mov_b32", "s_brev_b32", "s_sext_i32_i8", "s_sext_i32_i16", "s_ff1_i32_b32", "s_flbit_i32_b32"):
            state["scc_uniform"] = True  # (only these SOP1 ops leave SCC alone)
        return f"{op} {rng.choice(S[16:])}, {ssrc()}"
    if r < 0.93:
        state["scc_uniform"] = True
        return f"{rng.choice(SCMP)} {ssrc()}, {ssrc(False)}"
    if r < 0.96:
        op = rng.choice(SOPK)
        imm = rng.choice([0, 1, 7, 255, 0x7fff, 0x8000, 0xffff, rng.randrange(65536)])
        if op not in ("s_movk_i32", "s_mulk_i32"):  # (these two leave SCC alone)
            state["scc_uniform"] = True
        return f"{op} {rng.choice(S[16:])}, {hex(imm)}"
    if state.get("scc_uniform", True):
        op = rng.choice(SCC_USERS)
        return f"{op} {rng.choice(S[16:])}, {ssrc()}, {ssrc(False)}"
    return f"v_mov_b32 {vdst()}, {src0()}"


EPILOGUE = [
    "v_cndmask_b32_e64 v16, 0, 1, exec",
    "v_cndmask_b32_e64 v17, 0, 1, vcc",
    "v_cndmask_b32_e64 v18, 0, 1, s[24:25]",
    "v_cndmask_b32_e64 v19, 0, 1, s[26:27]",
    "s_mov_b64 exec, -1",
    "v_mov_b32 v12, s16", "v_mov_b32 v13, s17", "v_mov_b32 v14, s18", "v_mov_b32 v15, s19",
    "v_mov_b32 v20, s20", "v_mov_b32 v21, s21", "v_mov_b32 v22, s22", "v_mov_b32 v23, s23",
    "exp param0 v0, v1, v2, v3",
    "exp param1 v4, v5, v6, v7",
    "exp param2 v8, v9, v10, v11",
    "exp param3 v12, v13, v14, v15",
    "exp param4 v16, v17, v18, v19",
    "exp param5 v20, v21, v22, v23 done",
    "s_endpgm",
]
# Masks start as all lanes (VCC) or as lane patterns from compares: the prologue sets them up.
PROLOGUE = [
    "v_cmp_lt_u32_e32 vcc, 31, v0",
    "v_cmp_gt_u32_e64 s[24:25], 48, v0",
    "v_cmp_ne_u32_e64 s[26:27], 0, v0",
    "v_cvt_f32_u32 v1, v0",
    "v_mul_f32 v2, s0, v1",
    "v_add_i32 v3, vcc, s1, v0",
]


def assemble(lines):
    out = subprocess.run(MC, input="\n".join(lines) + "\n", capture_output=True, text=True)
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


def valu(state):
    """A vector ALU instruction for a divergent region: writes VGPRs only (no compares, no carry
    or mask destinations), so lanes outside EXEC are untouched in both models."""
    while True:
        line = instruction(dict(state))  # (a discarded SALU pick must not change what SCC holds)
        op = line.split()[0]
        if op.startswith("v_") and not op.startswith("v_cmp") and op not in ("v_add_i32_e64", "v_addc_u32_e64"):
            return line


def straight(state, n):
    return [instruction(state) for _ in range(n)]


def region(state, depth):
    """VALU-only body of a divergent region, maybe holding a nested if/else."""
    out = [valu(state) for _ in range(rng.randrange(1, 4))]
    if depth < 2 and rng.random() < 0.3:
        out += if_else(state, depth + 1, nested=True)
        out += [valu(state) for _ in range(rng.randrange(0, 3))]
    return out


label_counter = [0]


def label(stem):
    label_counter[0] += 1
    return f"{stem}{label_counter[0]}"


def if_else(state, depth, nested=False):
    # LLVM's structurized if/else on a per-lane compare (the saveexec idiom). Nested ones compare
    # into an SGPR pair: VCC is exported, and written in a divergent region it would keep only the
    # lanes active there.
    save = ["s[28:29]", "s[30:31]", "s[32:33]"][depth]
    cond = "s[36:37]" if nested else "vcc"
    kind = rng.choice(["f32", "i32", "u32"])
    cmp = rng.choice(CMP_F if kind == "f32" else CMP_I)
    a, b = rng.choice(V + CONSTS), rng.choice(V)
    els, end = label("else"), label("endif")
    state["scc_uniform"] = False  # saveexec and s_or_b64 set SCC from the wave's EXEC
    out = [f"v_cmp_{cmp}_{kind}_e64 {cond}, {a}, {b}",
           f"s_and_saveexec_b64 {save}, {cond}",
           f"s_xor_b64 {save}, exec, {save}",
           f"s_cbranch_execz {els}"]
    out += region(state, depth)
    out += [f"{els}:", f"s_or_saveexec_b64 {save}, {save}", f"s_xor_b64 exec, exec, {save}", f"s_cbranch_execz {end}"]
    if rng.random() < 0.7:
        out += region(state, depth)
    out += [f"{end}:", f"s_or_b64 exec, exec, {save}"]
    state["scc_uniform"] = False
    return out


def uniform_loop(state):
    top = label("loop")
    trips = rng.randrange(1, 5)
    out = ["s_mov_b32 s40, 0", f"{top}:"] + region(state, 1) + ["s_add_u32 s40, s40, 1", f"s_cmp_lt_u32 s40, {trips}",
                                                                f"s_cbranch_scc1 {top}"]
    state["scc_uniform"] = True
    return out


def divergent_loop(state):
    # Each lane leaves after (lane & 7) + 1 trips: v24 counts, v25 is the lane's limit (outside the
    # body's register pool), s[38:39] keeps the lanes to restore.
    top = label("dloop")
    out = ["v_and_b32 v25, 7, v0", "v_add_i32 v25, vcc, 1, v25", "v_mov_b32 v24, 0", "s_mov_b64 s[38:39], exec",
           f"{top}:"] + region(state, 2) + ["v_add_i32_e64 v24, s[34:35], 1, v24",
                                           "v_cmp_ge_u32_e64 s[34:35], v24, v25",
                                           "s_andn2_b64 exec, exec, s[34:35]",
                                           f"s_cbranch_execnz {top}",
                                           "s_mov_b64 exec, s[38:39]"]
    state["scc_uniform"] = False
    return out


def structured_body(state):
    out = straight(state, rng.randrange(2, 6))
    for _ in range(rng.randrange(2, 5)):
        k = rng.random()
        out += if_else(state, 0) if k < 0.45 else uniform_loop(state) if k < 0.75 else divergent_loop(state)
        out += straight(state, rng.randrange(1, 5))
    return out


def assemble_obj(lines):
    """Words of a program with labels (an object file: -show-encoding leaves branches unresolved)."""
    import shutil, tempfile
    objcopy = shutil.which("llvm-objcopy") or "llvm-objcopy-18"
    with tempfile.TemporaryDirectory() as tmp:
        obj, raw = os.path.join(tmp, "p.o"), os.path.join(tmp, "p.bin")
        out = subprocess.run(["llvm-mc", "-arch=amdgcn", "-mcpu=bonaire", "-filetype=obj", "-o", obj],
                             input="\n".join(lines) + "\n", capture_output=True, text=True)
        if out.returncode or "error" in out.stderr:
            return None
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        data = open(raw, "rb").read()
    return [int.from_bytes(data[j:j + 4], "little") for j in range(0, len(data), 4)]


if len(sys.argv) > 3 and sys.argv[3] == "cf":
    # Control flow: if/else (saveexec), uniform loops and per-lane loops; -> programs_cf.txt.
    programs = []
    while len(programs) < count:
        state = {}
        body = structured_body(state)
        # Drop what the assembler refuses (labels and branches assemble), then lay it out.
        plain = [l for l in body if not l.endswith(":")]
        words = assemble(plain)
        bad = {l for l, w in zip(plain, words) if w is None}
        body = [l for l in body if l not in bad]
        flat = assemble_obj(PROLOGUE + body + EPILOGUE)
        if flat is None:
            continue
        programs.append((body, flat))
    with open(os.path.join(here, "programs_cf.txt"), "w") as f:
        f.write("# generated by gen_programs.py (cf): one program per 'prog' line; '#' lines show its source\n")
        for body, flat in programs:
            for l in body:
                f.write("# " + l + "\n")
            f.write("prog " + " ".join(f"{x:08x}" for x in flat) + "\n")
    print(len(programs), "programs with control flow")
    sys.exit(0)

programs = []
attempts = 0
while len(programs) < count and attempts < count * 20:
    attempts += 1
    state = {}
    body = [instruction(state) for _ in range(14)]
    lines = PROLOGUE + body + EPILOGUE
    words = assemble(lines)
    if any(w is None for w in words):
        # Drop the instructions the assembler refuses and keep the rest.
        lines = [l for l, w in zip(lines, words) if w is not None]
        words = [w for w in words if w is not None]
        if len(lines) < len(PROLOGUE) + 8 + len(EPILOGUE):
            continue
    flat = [x for w in words for x in w]
    programs.append((lines, flat))

with open(os.path.join(here, "programs.txt"), "w") as f:
    f.write("# generated by gen_programs.py: one program per 'prog' line; '#' lines show its source\n")
    for lines, flat in programs:
        for l in lines[len(PROLOGUE):-len(EPILOGUE)]:
            f.write("# " + l + "\n")
        f.write("prog " + " ".join(f"{x:08x}" for x in flat) + "\n")
print(len(programs), "programs")
