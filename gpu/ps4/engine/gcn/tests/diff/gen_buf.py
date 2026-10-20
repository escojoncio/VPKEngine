#!/usr/bin/env python3
"""Random MUBUF programs for the buffer differential test (gcn_wave_diff --buf --lds 8192): one 64-lane wave, a test
buffer behind the V# in s[16:19] (stride / swizzle / add_tid / num_records drawn per program, "vsharp" line), a body of
buffer loads, stores and atomics (32 and 64 bit, all addressing modes, partial EXEC, lds=1, tfe) and an epilogue that
dumps the registers. The harness compares the registers, the test buffer and the LDS with the reference
(ref/gcn_ref_buf.cpp).

Instructions whose result depends on the order the lanes run (stores, returning or order dependent atomics) use
addresses that are distinct across the lanes: a python model of the addressing checks that and draws again. 64-bit
atomics may share addresses (the lifted code serialises the lanes in lane order, as the reference does).

usage: gen_buf.py [count] [seed]   -> programs_buf.txt
"""
import os
import random
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
count = int(sys.argv[1]) if len(sys.argv) > 1 else 400
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 7)

TEST_BYTES = 8192
BODY_MAX = int(os.environ.get('BODY_MAX', '0'))
OUT = os.environ.get('OUT', 'programs_buf.txt')


class VSharp:
    def __init__(self):
        r = rng.random()
        self.stride = 0
        self.swz = False
        self.es = rng.randrange(4)
        self.istr = rng.randrange(4)
        self.tid = rng.random() < 0.25
        if r < 0.35:
            self.num = rng.choice([8192, rng.randrange(64, 8192), rng.randrange(256, 2048)])
        elif r < 0.7:
            self.stride = rng.choice([4, 8, 12, 16, 20, 32, 48, 64])
            self.num = rng.randrange(8, 130)
        else:
            self.stride = rng.choice([16, 32, 64, 128])
            self.swz = True
            self.es = rng.randrange(1, 4)  # (elements of 4 bytes or more: dword accesses stay aligned)
            self.num = rng.randrange(8, 130)

    def words(self):
        w1 = (self.stride << 16) | (0x80000000 if self.swz else 0)
        w3 = (4 | 5 << 3 | 6 << 6 | 7 << 9) | self.es << 19 | self.istr << 21 | (1 << 23 if self.tid else 0)
        return [0, w1, self.num, w3]


# Register contents the model knows (lane -> value); pairs are (first, second) registers.
def regs(lane):
    return {
        "v10": lane * 4, "v11": lane, "v12": (lane & 7) * 4, "v13": lane & 15,
        "v[26:27]": (lane, lane * 4), "v[28:29]": (lane >> 1, (lane & 1) * 8),
        "v[32:33]": (lane * 4, 0), "v[34:35]": (lane * 8, 1 if lane >= 60 else 0),
    }


SOFFSETS = {"0": 0, "4": 4, "16": 16, "64": 64, "s21": 64, "s22": 0x100, "s23": 0x20c}


def address(vs, mode, regname, inst_off, soff, lane, offen, idxen):
    """(byte address, valid-index flag) of lane, or None when the model cannot say."""
    r = regs(lane)[regname] if regname else None
    if mode == "addr64":
        lo, hi = r
        a = lo + inst_off + soff
        return a & 0xffffffff, (a >> 32) == 0 and hi == 0, None
    index = off = 0
    if idxen and offen:
        index, off = r
    elif idxen:
        index = r
    elif offen:
        off = r
    off += inst_off
    if vs.tid:
        index += lane
    if vs.stride and vs.swz:
        elem, istr = 2 << vs.es, 8 << vs.istr
        buf = (index // istr * vs.stride + off // elem * elem) * istr + (index % istr) * elem + off % elem
    else:
        buf = index * vs.stride + off
    a = (buf + soff) & 0xffffffff
    idx_bad = index >= vs.num or ((idxen or vs.tid) and off >= vs.stride)
    return a, None, idx_bad


def inside(vs, mode, info, extra):
    a, hi_ok, idx_bad = info
    if mode == "addr64":
        return hi_ok
    if vs.stride:
        return not idx_bad
    return a + extra < vs.num


FORMATS = ["x", "xy", "xyz", "xyzw"]
ATOMS = ["swap", "cmpswap", "add", "sub", "smin", "umin", "smax", "umax", "and", "or", "xor", "inc", "dec", "fcmpswap", "fmin", "fmax"]
SHARED_OK = {"add", "sub", "smin", "umin", "smax", "umax", "and", "or", "xor"}
ADDR_REGS = {
    "offen": ["v10", "v12", "v10", "v10"],
    "idxen": ["v11", "v13", "v11"],
    "both": ["v[26:27]", "v[28:29]"],
    "addr64": ["v[32:33]", "v[34:35]"],
    "none": [None],
}


def draw_address(vs):
    mode = rng.choices(["offen", "idxen", "both", "addr64", "none"], [30, 20, 20, 8, 10])[0]
    reg = rng.choice(ADDR_REGS[mode])
    soff = rng.choice(list(SOFFSETS))
    inst_off = rng.choice([0, 0, 4, 8, 16, 64, 252, 1024, 4092])
    return mode, reg, soff, inst_off


def lanes_info(vs, mode, reg, inst_off, soff):
    return [address(vs, mode, reg, inst_off, SOFFSETS[soff], lane, mode in ("offen", "both"), mode in ("idxen", "both"))
            for lane in range(64)]


def disjoint(vs, mode, infos, size, exec_lanes, align):
    spans = []
    for lane in exec_lanes:
        info = infos[lane]
        ok = all(inside(vs, mode, info, e) for e in range(0, size, 4)) if size >= 4 else inside(vs, mode, info, 0)
        if not ok:
            continue
        a = info[0] & ~(align - 1)
        spans.append((a, a + size))
    spans.sort()
    return all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def mubuf_line(opname, data, mode, reg, soff, inst_off, glc=False, extra=""):
    parts = [f"buffer_{opname} {data}"]
    if mode == "addr64":
        parts.append(f", {reg}, s[16:19], {soff} addr64")
    elif mode == "none":
        parts.append(f", off, s[16:19], {soff}")
    else:
        mods = {"offen": "offen", "idxen": "idxen", "both": "idxen offen"}[mode]
        parts.append(f", {reg}, s[16:19], {soff} {mods}")
    s = "".join(parts)
    if inst_off:
        s += f" offset:{inst_off}"
    if glc:
        s += " glc"
    return s + extra


def dreg(n):
    base = rng.randrange(1, 11 - n)
    return f"v{base}" if n == 1 else f"v[{base}:{base + n - 1}]"


def make_instruction(vs):
    """(lines, exec_lanes) for one random buffer instruction."""
    kind = rng.choices(["load", "store", "atomic", "ldsload"], [30, 28, 38, 4])[0]
    for _ in range(200):
        mode, reg, soff, inst_off = draw_address(vs)
        infos = lanes_info(vs, mode, reg, inst_off, soff)
        # (an address in range of the V# but past the 8 KiB the test allocates is not something to compare)
        if any(inside(vs, mode, infos[l], 0) and infos[l][0] + 20 > TEST_BYTES for l in range(64)):
            continue
        # exec: all lanes, or a prefix selected by v_cmp
        lim = rng.choice([64, 64, 64, 40, 17, 5])
        lanes = range(lim)
        wrap = lim != 64
        glc = False
        extra = ""
        if kind == "load":
            op = rng.choice(["load_dword", "load_dwordx2", "load_dwordx3", "load_dwordx4", "load_ubyte", "load_sbyte",
                             "load_ushort", "load_sshort"] + [f"load_format_{f}" for f in FORMATS])
            n = {"dword": 1, "dwordx2": 2, "dwordx3": 3, "dwordx4": 4}.get(op[5:], 1)
            if op.startswith("load_format_"):
                n = len(op[12:])
            if mode == "addr64" and op.startswith("load_format"):
                continue
            tfe = rng.random() < 0.1 and mode != "none"
            if tfe:
                extra = " tfe"
                n += 1
            line = mubuf_line(op, dreg(n), mode, reg, soff, inst_off, False, extra)
            return line, wrap, lim
        if kind == "ldsload":
            line = mubuf_line("load_dword", "v1", mode, reg, soff, inst_off, False, "")
            if inst_off > 2048:
                continue
            return line + " LDS", wrap, lim
        if kind == "store":
            op = rng.choice(["store_dword", "store_dwordx2", "store_dwordx3", "store_dwordx4", "store_byte", "store_short"] +
                            [f"store_format_{f}" for f in FORMATS])
            if mode == "addr64" and op.startswith("store_format"):
                continue
            size = {"store_dword": 4, "store_dwordx2": 8, "store_dwordx3": 12, "store_dwordx4": 16, "store_byte": 1,
                    "store_short": 2}.get(op, 4 * len(op[13:]) if op.startswith("store_format") else 4)
            n = size // 4 if size >= 4 else 1
            if not disjoint(vs, mode, infos, size, lanes, 4 if size >= 4 else size):
                continue
            return mubuf_line(op, dreg(n), mode, reg, soff, inst_off), wrap, lim
        # atomic
        wide = rng.random() < 0.35
        a = rng.choice(ATOMS)
        glc = rng.random() < 0.5
        needs_unique = glc or a not in SHARED_OK
        if not wide and needs_unique and not disjoint(vs, mode, infos, 4, lanes, 4):
            continue
        cmp = a in ("cmpswap", "fcmpswap")
        n = (2 if wide else 1) * (2 if cmp else 1)
        name = f"atomic_{a}" + ("_x2" if wide else "")
        return mubuf_line(name, dreg(n), mode, reg, soff, inst_off, glc), wrap, lim
    return None


def build_body(vs):
    body = []
    for _ in range(rng.randrange(2, 9) if BODY_MAX == 0 else BODY_MAX):
        got = None
        while got is None:
            got = make_instruction(vs)
        line, wrap, lim = got
        lds = line.endswith(" LDS")
        if lds:
            line = line[:-4]
        if wrap:
            body += [f"v_cmp_gt_u32 vcc, {lim}, v0", "s_and_saveexec_b64 s[30:31], vcc"]
        if lds:
            body += [f"s_mov_b32 m0, {rng.randrange(0, 1024) * 4}"]
            body += ["LDSBIT " + line]
        else:
            body += [line]
        if wrap:
            body += ["s_mov_b64 exec, s[30:31]"]
        body += ["s_waitcnt vmcnt(0)", "s_barrier"]
    return body


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


programs = []
while len(programs) < count:
    vs = VSharp()
    body = build_body(vs)
    lines = PROLOGUE + body + EPILOGUE
    words = assemble(lines)
    if any(w is None for w in words):
        bad = [l for l, w in zip(lines, words) if w is None]
        sys.exit("cannot assemble: " + bad[0])
    programs.append((vs, body, [x for w in words for x in w]))

with open(os.path.join(here, OUT), "w") as f:
    f.write("# generated by gen_buf.py: one program per 'prog' line; '#' lines show its source\n")
    for vs, body, flat in programs:
        f.write(f"# V# stride={vs.stride} swizzle={int(vs.swz)} add_tid={int(vs.tid)} num_records={vs.num} es={vs.es} is={vs.istr}\n")
        for l in body:
            f.write("# " + l + "\n")
        f.write("vsharp " + " ".join(f"{x:08x}" for x in vs.words()) + "\n")
        f.write("prog " + " ".join(f"{x:08x}" for x in flat) + "\n")
print(len(programs), "programs")
