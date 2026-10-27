#!/usr/bin/env python3
"""Random MIMG programs for the image differential test (gcn_wave_diff --img): one 64-lane wave and one RGBA32 image
(1D / 2D / 3D / 2D array / cube / cube array, behind the T# in s[20:27]; the dimension comes from the "timage" line,
as the client's descriptor decoding would give it). A body of image_load / load_mip / store (any dmask) / atomics (all
16 ops, glc) / get_resinfo, with coordinates that are distinct per lane (or shared by few lanes for commutative
non-returning atomics) and partly outside the image. The harness compares the registers and every texel of the image
with the reference (ref/gcn_ref_img.cpp).

usage: gen_img.py [count] [seed]   -> programs_img.txt
"""
import os
import random
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 21)
BODY_MAX = int(os.environ.get("BODY_MAX", "0"))
OUT = os.environ.get("OUT", "programs_img.txt")

D1, D2, D3, D2A, CUBE, CUBEA = range(6)
NAMES = ["1d", "2d", "3d", "2d array", "cube", "cube array"]


class Img:
    def __init__(self, storage):
        self.dim = rng.randrange(6)
        d = self.dim
        self.w = rng.randrange(3, 40) if d == D1 else rng.randrange(3, 12)
        self.h = 1 if d == D1 else self.w if d in (CUBE, CUBEA) else rng.randrange(3, 12) if d != D1 else 1
        if d == D1:
            self.h = 1
        self.depth = {D1: 1, D2: 1, D3: rng.randrange(2, 6), D2A: rng.randrange(1, 5), CUBE: 6, CUBEA: 6 * rng.randrange(1, 3)}[d]
        self.levels = 1 if storage or rng.random() < 0.3 else rng.randrange(2, 4)

    @property
    def ncoord(self):
        return {D1: 1, D2: 2, D3: 3, D2A: 3, CUBE: 3, CUBEA: 3}[self.dim]


# per-lane coordinate registers prepared by the prologue
PROLOGUE = [
    "v_and_b32 v20, 7, v0",
    "v_lshrrev_b32 v21, 3, v0",
    "v_and_b32 v22, 3, v0",
    "v_lshrrev_b32 v23, 2, v0",
    "v_and_b32 v23, 3, v23",
    "v_lshrrev_b32 v24, 4, v0",
    "v_lshrrev_b32 v25, 3, v0",
    "v_and_b32 v25, 3, v25",
    "v_lshrrev_b32 v26, 5, v0",
    "v_and_b32 v28, 1, v0",
    "v_lshrrev_b32 v29, 1, v0",
    "v_and_b32 v29, 1, v29",
    "s_mov_b32 s30, 0x9e3779b1",
    "v_mul_lo_u32 v1, v0, s30",
    "v_xor_b32 v2, 0x5a5a1234, v1",
    "v_add_i32 v3, vcc, 0x7fff0000, v1",
    "v_lshrrev_b32 v4, 3, v2",
    "v_and_b32 v5, 7, v2",
    "v_mov_b32 v6, 0xffff0000",
    "v_add_i32 v7, vcc, v0, v6",
    "v_mul_f32 v8, 0x40490fdb, v0",
    "v_cvt_f32_u32 v9, v0",
    "v_and_b32 v10, 3, v1",
]
EPILOGUE = ["s_waitcnt vmcnt(0)", "s_barrier", "s_mov_b64 exec, -1", "v_lshlrev_b32 v30, 2, v0",
            "v_add_i32 v31, vcc, 0x1000, v30"] + \
    [f"buffer_store_dword v{i}, v{30 if i < 16 else 31}, s[12:15], 0 offen offset:{(i % 16) * 256}" for i in range(24)] + ["s_endpgm"]

SIGNED = {"smin", "smax"}
ATOMS = ["swap", "cmpswap", "add", "sub", "rsub", "smin", "umin", "smax", "umax", "and", "or", "xor", "inc", "dec", "fcmpswap", "fmin", "fmax"]
SHARED_OK = {"add", "sub", "smin", "umin", "smax", "umax", "and", "or", "xor"}


def coords(img, shared):
    """Lines that build the address window in v[40..]: x, y, z/layer (injective across lanes unless `shared`)."""
    base = {D1: ["v0"], D2: ["v20", "v21"], D3: ["v22", "v23", "v24"], D2A: ["v20", "v25", "v26"],
            CUBE: ["v20", "v25", "v26"], CUBEA: ["v20", "v25", "v26"]}[img.dim]
    if shared:
        base = ["v28", "v29", "v28"][:img.ncoord]
    lines = []
    for k, reg in enumerate(base):
        off = rng.choice([0, 0, 0, 1, 2, -1])
        # (a shift past the image edge puts some lanes outside)
        if off >= 0:
            lines.append(f"v_add_i32 v{40 + k}, vcc, {off}, {reg}")
        else:
            lines.append(f"v_subrev_i32 v{40 + k}, vcc, {-off}, {reg}")
    return lines


def make_op(img, allow_store):
    kind = rng.choices(["load", "load_mip", "store", "atomic", "resinfo"], [30, 14, 18 if allow_store else 0, 30 if allow_store else 0, 8])[0]
    n = img.ncoord
    if kind == "load":
        dmask = rng.randrange(1, 16)
        c = bin(dmask).count("1")
        tfe = rng.random() < 0.1
        regs = f"v[1:{c + (1 if tfe else 0)}]" if c + tfe > 1 else "v1"
        return coords(img, False) + [f"image_load {regs}, v[40:{39 + n}], s[20:27] dmask:{dmask:#x} unorm{' tfe' if tfe else ''}{' da' if n == 3 and img.dim != D3 else ''}".replace("v[40:40]", "v40")], kind
    if kind == "load_mip":
        dmask = rng.randrange(1, 16)
        c = bin(dmask).count("1")
        regs = f"v[1:{c}]" if c > 1 else "v1"
        return coords(img, False) + [f"v_and_b32 v{40 + n}, 3, v10", f"v_min_u32 v{40 + n}, {img.levels - 1 if (img.levels == 1 or True) else 3}, v{40 + n}", f"image_load_mip {regs}, v[40:{40 + n}], s[20:27] dmask:{dmask:#x} unorm{' da' if n == 3 and img.dim != D3 else ''}"], kind
    if kind == "store":
        dmask = rng.randrange(1, 16)
        c = bin(dmask).count("1")
        regs = f"v[1:{c}]" if c > 1 else "v1"
        return coords(img, False) + [f"image_store {regs}, v[40:{39 + n}], s[20:27] dmask:{dmask:#x} unorm{' da' if n == 3 and img.dim != D3 else ''}".replace("v[40:40]", "v40")], kind
    if kind == "resinfo":
        dmask = rng.randrange(1, 16)
        c = bin(dmask).count("1")
        regs = f"v[1:{c}]" if c > 1 else "v1"
        return [f"v_and_b32 v40, 3, v10", f"v_min_u32 v40, {img.levels - 1}, v40", f"image_get_resinfo {regs}, v40, s[20:27] dmask:{dmask:#x} unorm"], kind
    a = rng.choice(ATOMS)
    glc = rng.random() < 0.5
    shared = a in SHARED_OK and not glc and rng.random() < 0.5
    cmp = a in ("cmpswap", "fcmpswap")
    regs = "v[1:2]" if cmp else "v1"
    dmask = 3 if cmp else 1
    return coords(img, shared) + [f"image_atomic_{a} {regs}, v[40:{39 + n}], s[20:27] dmask:{dmask:#x} unorm{' glc' if glc else ''}{' da' if n == 3 and img.dim != D3 else ''}".replace("v[40:40]", "v40")], kind


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
    storage = rng.random() < 0.7
    img = Img(storage)
    body, kinds = [], set()
    for _ in range(rng.randrange(2, 8) if BODY_MAX == 0 else BODY_MAX):
        lines, kind = make_op(img, storage)
        kinds.add(kind)
        wrap = rng.random() < 0.2
        if wrap:
            lim = rng.choice([40, 17, 5])
            body += [f"v_cmp_gt_u32 vcc, {lim}, v0", "s_and_saveexec_b64 s[34:35], vcc"]
        body += lines
        if wrap:
            body += ["s_mov_b64 exec, s[34:35]"]
        body += ["s_waitcnt vmcnt(0)", "s_barrier"]
    lines = PROLOGUE + body + EPILOGUE
    words = assemble(lines)
    if any(w is None for w in words):
        bad = [l for l, w in zip(lines, words) if w is None]
        sys.exit("cannot assemble: " + bad[0])
    programs.append((img, body, [x for w in words for x in w]))

with open(os.path.join(here, OUT), "w") as f:
    f.write("# generated by gen_img.py: one program per 'prog' line; '#' lines show its source\n")
    for img, body, flat in programs:
        f.write(f"# image {NAMES[img.dim]} {img.w}x{img.h}x{img.depth} levels={img.levels}\n")
        for l in body:
            f.write("# " + l + "\n")
        f.write(f"timage {img.dim} {img.w} {img.h} {img.depth} {img.levels}\n")
        f.write("prog " + " ".join(f"{x:08x}" for x in flat) + "\n")
print(len(programs), "programs")
