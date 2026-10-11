#!/usr/bin/env python3
"""Builds the GCN 2 golden corpus (corpus.txt) using llvm-mc as a black-box oracle.

llvm-mc 18 cannot disassemble gfx7 (bonaire), so every entry comes from the assembler: each candidate
instruction word is built from its fields, spelled as assembly, and assembled with
`llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding`. The word is valid iff the assembler accepts the
text and produces exactly that word; its canonical text is the assembler's echo. Candidates are field
sweeps per opcode and operand slot plus random words per encoding.

Corpus line: `<hex dwords, comma separated> <text>` or `<hex dwords> !` for an invalid word.
Usage: gen_corpus.py <path to opcodes.h> <output corpus>
"""
import random
import re
import subprocess
import sys

LLVM_MC = ["llvm-mc", "-arch=amdgcn", "-mcpu=bonaire", "-show-encoding"]
LITERAL = 0x12345678
rng = random.Random(0x6C6E)


def assemble(lines):
    """Returns, per line, (text, [dwords]) or None if the assembler rejects it."""
    out = subprocess.run(LLVM_MC, input="\n".join(lines) + "\n", capture_output=True, text=True)
    bad = {int(m.group(1)) - 1 for m in re.finditer(r"<stdin>:(\d+):\d+: error", out.stderr)}
    encoded = [l for l in out.stdout.splitlines() if "; encoding:" in l]
    res, it = [], iter(encoded)
    for i in range(len(lines)):
        if i in bad:
            res.append(None)
            continue
        text, enc = next(it).split("; encoding:")
        b = [int(x, 16) for x in enc.strip()[1:-1].split(",")]
        words = [b[j] | b[j + 1] << 8 | b[j + 2] << 16 | b[j + 3] << 24 for j in range(0, len(b), 4)]
        res.append((text.strip(), words))
    return res


# ---- Operand spelling from the ISA's 8-bit scalar operand encoding --------------------------------

SPECIAL = {104: "flat_scratch", 106: "vcc", 108: "tba", 110: "tma", 126: "exec"}


def sreg(code, n):
    if code < 104 or 112 <= code < 124:
        p, i = ("s", code) if code < 104 else ("ttmp", code - 112)
        return f"{p}{i}" if n == 1 else f"{p}[{i}:{i + n - 1}]"
    if code == 124 and n == 1:
        return "m0"
    base = code & ~1
    if base in SPECIAL and code != 125:
        if n == 2:
            return SPECIAL[base] if code == base else None
        if n == 1:
            return SPECIAL[base] + ("_hi" if code & 1 else "_lo")
    return None


FLOATS = ["0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0", "4.0", "-4.0"]


def ssrc(code, n):
    """Spelling of a source operand code, and whether it consumes the literal dword."""
    if code < 128:
        return sreg(code, n), False
    if code <= 192:
        return str(code - 128), False
    if code <= 208:
        return str(192 - code), False
    if 240 <= code <= 247:
        return FLOATS[code - 240], False
    return {248: "0.15915494", 251: "src_vccz", 252: "src_execz", 253: "src_scc",
            254: "src_lds_direct", 255: hex(LITERAL)}.get(code), code == 255


# ---- Encodings: field layout, operand slots, and text templates -----------------------------------

class Op:
    def __init__(self, enc, name, opcode, slots):
        self.enc, self.name, self.opcode, self.slots = enc, name, opcode, slots


def build(enc, opcode, f):
    """Instruction word from fields: d (dst), a/b (sources), k (simm16), base, imm, off."""
    if enc == "SOP2":
        return 0x80000000 | opcode << 23 | f.get("d", 0) << 16 | f.get("b", 0) << 8 | f.get("a", 0)
    if enc == "SOP1":
        return 0xBE800000 | f.get("d", 0) << 16 | opcode << 8 | f.get("a", 0)
    if enc == "SOPC":
        return 0xBF000000 | opcode << 16 | f.get("b", 0) << 8 | f.get("a", 0)
    if enc == "SOPK":
        return 0xB0000000 | opcode << 23 | f.get("d", 0) << 16 | f.get("k", 0)
    if enc == "SOPP":
        return 0xBF800000 | opcode << 16 | f.get("k", 0)
    return (0xC0000000 | opcode << 22 | f.get("d", 0) << 15 | f.get("base", 0) << 9 |
            f.get("imm", 0) << 8 | f.get("off", 0))


def spell(op, f):
    """Assembly text for op with fields f, plus the literal dword if one is used (None = no text)."""
    parts, lit = [], False
    for kind, n in op.slots:
        if kind == "d":
            t = sreg(f["d"], n)
        elif kind in "ab":
            t, l = ssrc(f[kind], n)
            lit |= l
        elif kind == "base":
            t = sreg(f["base"] * 2, n)
        elif kind == "off":
            if f["imm"]:
                t = hex(f["off"])
            else:
                t, l = ssrc(f["off"], 1)
                lit |= l
                if f["off"] >= 128 and f["off"] not in (251, 252, 253, 255):
                    t = None  # no spelling encodes an inline constant here
        else:  # "k": simm16, spelled as a plain number whatever its meaning
            t = str(f["k"])
        if t is None:
            return None, False
        parts.append(t)
    if op.enc == "SOPK" and op.name == "s_setreg_imm32_b32":
        parts.append(hex(LITERAL))
        lit = True
    return (op.name + (" " + ", ".join(parts) if parts else "")), lit


DEFAULTS = {"d": 16, "a": 4, "b": 8, "k": 0, "base": 2, "imm": 1, "off": 1}

# Candidate slot shapes per encoding, tried in order until the assembler accepts the mnemonic.
SIZES = [1, 2, 4, 8, 16]
SHAPES = {
    "SOP2": [[("d", x), ("a", y), ("b", z)] for x in (1, 2) for y in (1, 2) for z in (1, 2)] +
            [[("a", 2), ("b", 2)]],
    "SOP1": [[("d", x), ("a", y)] for x in (1, 2) for y in (1, 2)] + [[("d", 2)], [("a", 2)], [("a", 1)]],
    "SOPC": [[("a", x), ("b", y)] for x in (1, 2) for y in (1, 2)],
    "SOPK": [[("d", 1), ("k", 1)], [("d", 2), ("k", 1)], [("k", 1), ("d", 1)], [("k", 1)]],
    "SOPP": [[("k", 1)], []],
    "SMRD": [[("d", x), ("base", y), ("off", 1)] for x in SIZES for y in (2, 4)] + [[("d", 2)], []],
}


def discover(enc, names):
    """Finds each mnemonic's slot shape and opcode by assembling probes."""
    ops = []
    for name in names:
        for slots in SHAPES[enc]:
            op = Op(enc, name, 0, slots)
            f = dict(DEFAULTS, k=1)
            text, _ = spell(op, f)
            r = assemble([text])[0]
            if r is None:
                continue
            w = r[1][0]
            op.opcode = {"SOP2": (w >> 23) & 0x7F, "SOP1": (w >> 8) & 0xFF, "SOPC": (w >> 16) & 0x7F,
                         "SOPK": (w >> 23) & 0x1F, "SOPP": (w >> 16) & 0x7F, "SMRD": (w >> 22) & 0x1F}[enc]
            # The shape must also place every operand in the right field.
            if build(enc, op.opcode, canonical(op, f)) != w:
                continue
            ops.append(op)
            break
        else:
            sys.exit(f"no shape accepted for {name}")
    return ops


def field_ranges(op):
    """All values of each field that varies for this op."""
    r = {}
    for kind, _ in op.slots:
        if kind == "d":
            r["d"] = range(128)
        elif kind in "ab":
            r[kind] = range(256)
        elif kind == "base":
            r["base"] = range(64)
        elif kind == "off":
            r["imm"], r["off"] = range(2), range(256)
        else:
            r["k"] = range(65536)
    return r


def candidates(op, full):
    """Field sweeps (one field at a time from the defaults) and random combinations. Register fields
    are swept fully for the first op of each operand shape and sampled for the rest."""
    out = []
    ranges = field_ranges(op)
    k_edges = [0, 1, 2, 7, 15, 16, 63, 64, 65, 0x70, 0xF7F, 0x7FFF, 0x8000, 0xFFFF]
    for field, values in ranges.items():
        if field == "k":
            values = k_edges + [rng.randrange(65536) for _ in range(150)]
        elif not full and field != "imm":
            values = sorted(set(rng.sample(list(values), min(len(values), 24))))
        for v in values:
            f = dict(DEFAULTS)
            f[field] = v
            if field == "off":
                out.append(dict(f, imm=0))
            out.append(f)
    for _ in range(60 if full else 12):
        out.append({k: rng.choice(v) for k, v in ranges.items()} | {"imm": rng.randrange(2)})
    return out


def canonical(op, f):
    """Clears fields the op does not use: the assembler always encodes them as zero."""
    used = {s[0] for s in op.slots}
    used |= {"imm"} if "off" in used else set()
    return {k: (v if k in used else 0) for k, v in f.items()}


def entry(words, text):
    return ",".join(f"{w:08x}" for w in words) + " " + (text if text else "!")


def run(ops_by_enc, out_path, src_header):
    lines_out = []
    for enc, ops in ops_by_enc.items():
        jobs = []  # (words, text or None)
        seen_shape = set()
        for op in ops:
            shape = tuple(op.slots)
            full = shape not in seen_shape
            seen_shape.add(shape)
            for f in candidates(op, full):
                f = canonical(op, dict(DEFAULTS) | f)
                w = build(enc, op.opcode, f)
                text, lit = spell(op, f)
                words = [w, LITERAL] if lit else [w]
                jobs.append((words, text))
        # Random words over the whole encoding space, including unassigned opcodes.
        by_opcode = {op.opcode: op for op in ops}
        # Opcode space of each encoding (high SOP2/SOPK opcodes are the other scalar encodings).
        opcodes = {"SOP2": 0x60, "SOP1": 256, "SOPC": 128, "SOPK": 29, "SOPP": 128, "SMRD": 32}[enc]
        for _ in range(1500):
            opcode = rng.randrange(opcodes)
            f = {"d": rng.randrange(128), "a": rng.randrange(256), "b": rng.randrange(256),
                 "k": rng.randrange(65536), "base": rng.randrange(64), "imm": rng.randrange(2),
                 "off": rng.randrange(256)}
            w = build(enc, opcode, f)
            op = by_opcode.get(opcode)
            if op is None:
                jobs.append(([w], None))
                continue
            f = canonical(op, f)
            w = build(enc, opcode, f)
            text, lit = spell(op, f)
            jobs.append(([w, LITERAL] if lit else [w], text))
        texts = [t for _, t in jobs if t]
        results = iter(assemble(texts))
        for words, text in jobs:
            r = next(results) if text else None
            ok = r is not None and r[1] == words
            lines_out.append(entry(words, r[0] if ok else None))
    # Unique by word: the first verdict wins (all verdicts for the same word agree).
    lines_out += vector_entries(src_header)
    lines_out += memory_entries(src_header)
    seen, uniq = set(), []
    for l in lines_out:
        key = l.split(" ", 1)[0]
        if key not in seen:
            seen.add(key)
            uniq.append(l)
    with open(out_path, "w") as f:
        f.write("# GCN 2 golden corpus, generated by gen_corpus.py with llvm-mc -mcpu=bonaire.\n")
        f.write("\n".join(uniq) + "\n")
    valid = sum(not l.endswith(" !") for l in uniq)
    print(f"{len(uniq)} entries, {valid} valid")


# ---- Vector ALU: VOP2, VOP1, VOPC and VOP3 ------------------------------------------------------
#
# Each op is tried in its 32-bit form and, when it has one, its VOP3 form (opcode = form base + op).
# Operands come from the table signature (see opcodes.h); modifiers are spelled from the candidate's
# fields whatever the table says, so the assembler alone decides which ones exist.

VOP3_BASE = {"VOP2": 256, "VOP1": 384, "VOPC": 0, "VOP3": 0}
OMOD = ["", " mul:2", " mul:4", " div:2"]


class VOp:
    def __init__(self, enc, name, opcode, sig):
        self.enc, self.name, self.opcode = enc, name, opcode
        self.ops, _, self.mods = sig.partition("/")
        self.has64 = "/" in sig or enc == "VOP3"
        self.has32 = enc != "VOP3"
        # VOP3b: a scalar carry-out besides a vector result.
        self.vop3b = enc != "VOPC" and "C" in self.ops
        # Source fields actually encoded, per form.
        self.nsrc32 = sum(c not in "-+MckC" for c in self.ops[1:])
        self.nsrc64 = sum(c not in "-+MkC" for c in self.ops[1:])


def vsrc(code, n):
    if code >= 256:
        i = code - 256
        return (f"v{i}" if n == 1 else f"v[{i}:{i + n - 1}]") if i + n <= 256 else None, False
    if code == 254:
        return "src_lds_direct", False
    return ssrc(code, n)


def with_mods(text, code, a, n):
    if a:
        text = f"|{text}|"
    if n:
        numeric = 128 <= code <= 255
        text = f"neg({text})" if numeric and not a else "-" + text
    return text


def vbuild(op, e64, f):
    if e64:
        mid = f["sdst"] << 8 if op.vop3b else f["abs"] << 8 | f["clamp"] << 11
        w0 = 0xD0000000 | (VOP3_BASE[op.enc] + op.opcode) << 17 | mid | f["vdst"]
        return [w0, f["neg"] << 29 | f["omod"] << 27 | f["s2"] << 18 | f["s1"] << 9 | f["s0"]]
    if op.enc == "VOP2":
        return [op.opcode << 25 | f["vdst"] << 17 | f["s1"] << 9 | f["s0"]]
    if op.enc == "VOP1":
        return [0x7E000000 | f["vdst"] << 17 | op.opcode << 9 | f["s0"]]
    return [0x7C000000 | op.opcode << 17 | f["s1"] << 9 | f["s0"]]


def vspell(op, e64, f):
    """Assembly text and whether a literal dword follows (None text = not spellable)."""
    parts, lit, field = [], False, 0
    for i, c in enumerate(op.ops):
        if c in "-+M":
            continue
        if i == 0:
            if c == "s":
                t = sreg(f["vdst"], 1) if f["vdst"] < 128 else None
            else:
                t, _ = vsrc(256 + f["vdst"], int(c))
        elif c == "C":
            t = sreg(f["sdst"] if op.vop3b else f["vdst"], 2) if e64 else "vcc"
        elif c == "k":
            t, lit = hex(LITERAL), True
        elif c == "c" and not e64:
            t = "vcc"
        else:
            key = "s%d" % field
            code = f[key]
            field += 1
            if c == "c":
                t = sreg(code, 2) if code < 128 else None
            elif not e64 and field == 2 and op.enc in ("VOP2", "VOPC") and c.isdigit():
                t, _ = vsrc(256 + code, int(c))  # 8-bit VGPR field
            else:
                t, l = vsrc(code, 1 if c in "svSV" else int(c))
                lit |= l
            if t is not None and e64:
                j = field - 1
                t = with_mods(t, code, f["abs"] >> j & 1 and not op.vop3b, f["neg"] >> j & 1)
        if t is None:
            return None, False
        parts.append(t)
    name = op.name
    if op.enc != "VOP3" and op.has64 and op.has32:
        name += "_e64" if e64 else "_e32"
    text = name + (" " + ", ".join(parts) if parts else "")
    if e64:
        text += (" clamp" if f["clamp"] and not op.vop3b else "") + OMOD[f["omod"]]
    return text, lit


def vcanonical(op, e64, f):
    f = dict(f)
    n = op.nsrc64 if e64 else op.nsrc32
    for j in range(n, 3):
        f["s%d" % j] = 0
    if op.ops[0] == "-" and not (e64 and op.enc == "VOPC"):
        f["vdst"] = 0
    if not e64:
        f.update(abs=0, neg=0, clamp=0, omod=0, sdst=0)
    if not op.vop3b:
        f["sdst"] = 0
    else:
        f.update(abs=0, clamp=0)
    return f


VDEF = {"vdst": 10, "sdst": 12, "s0": 258, "s1": 259, "s2": 260, "abs": 0, "neg": 0, "clamp": 0,
        "omod": 0}


def vgpr_sample(base, n):
    """VGPR codes: both ends of the file (alignment and range limits live there) plus a sample."""
    edges = [0, 1, 2, 3, 4, 252, 253, 254, 255]
    return sorted({base + v for v in edges + rng.sample(range(256), n)})


def vcandidates(op, e64, full):
    """Field sweeps from the defaults plus random combinations. Source sweeps cover every non-VGPR
    code on the first op of each shape; VGPR indices are sampled (decoded the same way)."""
    ranges = {"vdst": range(256), "s0": range(512), "s1": range(512), "s2": range(512)}
    if e64:
        ranges.update(sdst=range(128), abs=range(8), neg=range(8), clamp=range(2), omod=range(4))
    elif op.enc in ("VOP2", "VOPC"):
        ranges["s1"] = range(256)
    out = []
    for field, values in ranges.items():
        if len(values) == 512:
            values = (list(range(256)) if full else sorted(rng.sample(range(256), 12))) + vgpr_sample(256, 6)
        elif len(values) == 256 and field != "s1":
            values = vgpr_sample(0, 8)
        elif not full and len(values) > 8:
            values = sorted(set(rng.sample(list(values), 12)))
        for v in values:
            f = dict(VDEF)
            if not e64 and op.enc in ("VOP2", "VOPC"):
                f["s1"] = 3
            if e64 and "c" in op.ops:
                f["s2"] = 106  # the carry-in / select source must be an SGPR pair: vcc
            f[field] = v
            out.append(f)
    for _ in range(30 if full else 8):
        out.append({k: rng.choice(v) for k, v in ranges.items()})
    return out


def vector_entries(src):
    ops = []
    for enc in ["VOP2", "VOP1", "VOPC", "VOP3"]:
        body = re.search(rf"#define VPGCN_{enc}\(X\)(.*?)(?:\n\n|\Z)", src, re.S).group(1)
        ops += [VOp(enc, n, int(o), sg) for n, o, sg in re.findall(r'X\((\w+), (\d+), "([^"]*)"\)', body)]
    jobs, seen_shape = [], set()

    def job(op, e64, f):
        f = vcanonical(op, e64, dict(VDEF) | f)
        text, lit = vspell(op, e64, f)
        words = vbuild(op, e64, f) + ([LITERAL] if lit else [])
        jobs.append((words, text))

    for op in ops:
        for e64 in (False, True):
            if not (op.has64 if e64 else op.has32):
                continue
            shape = (e64, op.enc, op.ops)  # modifier fields are swept fully for every op anyway
            full = shape not in seen_shape
            seen_shape.add(shape)
            for f in vcandidates(op, e64, full):
                job(op, e64, f)
    # Random words over each encoding's opcode space, including unassigned opcodes.
    by32 = {(op.enc, op.opcode): op for op in ops if op.has32}
    by64 = {VOP3_BASE[op.enc] + op.opcode: op for op in ops if op.has64}
    for enc, space in (("VOP2", 62), ("VOP1", 256), ("VOPC", 256), ("VOP3", 512)):
        for _ in range(1000):
            opcode = rng.randrange(space)
            f = {"vdst": rng.randrange(256), "sdst": rng.randrange(128), "s0": rng.randrange(512),
                 "s1": rng.randrange(512), "s2": rng.randrange(512), "abs": rng.randrange(8),
                 "neg": rng.randrange(8), "clamp": rng.randrange(2), "omod": rng.randrange(4)}
            e64 = enc == "VOP3"
            op = by64.get(opcode) if e64 else by32.get((enc, opcode))
            if op is None:
                stub = VOp(enc, "", opcode, "")
                stub.vop3b = False
                jobs.append((vbuild(stub, e64, dict(f, s1=f["s1"] & 0xFF)), None))
                continue
            if not e64 and enc in ("VOP2", "VOPC"):
                f["s1"] &= 0xFF
            job(op, e64, f)
    texts = [t for _, t in jobs if t]
    results = iter(assemble(texts))
    out = []
    for words, text in jobs:
        r = next(results) if text else None
        # An echo carrying a diagnostic comment is an operand the hardware does not accept.
        ok = r is not None and r[1] == words and "/*" not in r[0]
        if not ok and text and assembler_only(text):
            out.append(",".join(f"{w:08x}" for w in words) + " ?")
            continue
        out.append(entry(words, r[0] if ok else None))
    return out


# Encodings the hardware defines but llvm-mc 18 refuses to assemble, so the oracle cannot judge them:
# wide literals and VOP3 inline constants for v_cvt_f32_f16, a non-VGPR v_mqsad_u32_u8 accumulator,
# lds_direct in the operand-reversed ops. Recorded as "?" and skipped by the test.
REV_OPS = re.compile(r"^v_(subrev|subbrev|lshrrev|ashrrev|lshlrev)_")


def assembler_only(text):
    args = text.split(" ", 1)[1] if " " in text else ""
    srcs = [a.strip(" -|") for a in args.split(",")]
    if text.startswith("v_cvt_f32_f16_e32"):
        return "0x" in text  # a literal wider than 16 bits
    if text.startswith("v_cvt_f32_f16_e64"):
        s = srcs[1]
        return s[0].isdigit() or s.startswith("neg(") and s[4].isdigit()
    if text.startswith("v_mqsad_u32_u8"):
        return not srcs[3].startswith("v")
    return bool(REV_OPS.match(text)) and "src_lds_direct" in args


# ---- Memory, export and interpolation: DS, MUBUF, MTBUF, MIMG, FLAT, EXP, VINTRP ----------------
#
# Words are built from fields (unused fields zero, as the assembler encodes them) and spelled from the
# table signature. Accepted and re-encoded identically: a valid entry with the assembler's echo. The
# assembler refuses some encodings the hardware runs (lds, r128, address counts it has no register
# class for, gds-only ops without gds...): those are "?" (unjudged). Unassigned opcodes and invalid
# export targets are "!".

DFMT = ["INVALID", "8", "16", "8_8", "32", "16_16", "10_11_11", "11_11_10", "10_10_10_2", "2_10_10_10",
        "8_8_8_8", "32_32", "16_16_16_16", "32_32_32", "32_32_32_32", "RESERVED_15"]
NFMT = ["UNORM", "SNORM", "USCALED", "SSCALED", "UINT", "SINT", "RESERVED_6", "FLOAT"]


def vreg(i, n):
    return None if n == 0 or i + n > 256 else (f"v{i}" if n == 1 else f"v[{i}:{i + n - 1}]")


def rsrc(field, n):
    return sreg(field * 4, n)


def popcount(x):
    return bin(x).count("1")


class MOp:
    def __init__(self, enc, name, opcode, sig):
        self.enc, self.name, self.opcode, self.sig = enc, name, opcode, sig


def mbuild(op, f):
    g = f.get
    if op.enc == "DS":
        return [0xD8000000 | op.opcode << 18 | g("gds", 0) << 17 | g("off1", 0) << 8 | g("off0", 0),
                g("vdst", 0) << 24 | g("d1", 0) << 16 | g("d0", 0) << 8 | g("addr", 0)]
    if op.enc in ("MUBUF", "MTBUF"):
        w0 = (g("addr64", 0) << 15 | g("glc", 0) << 14 | g("idxen", 0) << 13 | g("offen", 0) << 12 |
              g("offset", 0))
        if op.enc == "MUBUF":
            w0 |= 0xE0000000 | op.opcode << 18 | g("lds", 0) << 16
        else:
            w0 |= 0xE8000000 | g("nfmt", 0) << 23 | g("dfmt", 0) << 19 | op.opcode << 16
        return [w0, g("soff", 0) << 24 | g("tfe", 0) << 23 | g("slc", 0) << 22 | g("rsrc", 0) << 16 |
                g("vdata", 0) << 8 | g("vaddr", 0)]
    if op.enc == "MIMG":
        return [0xF0000000 | g("slc", 0) << 25 | op.opcode << 18 | g("lwe", 0) << 17 | g("tfe", 0) << 16 |
                g("r128", 0) << 15 | g("da", 0) << 14 | g("glc", 0) << 13 | g("unorm", 0) << 12 |
                g("dmask", 0) << 8,
                g("ssamp", 0) << 21 | g("rsrc", 0) << 16 | g("vdata", 0) << 8 | g("vaddr", 0)]
    if op.enc == "FLAT":
        return [0xDC000000 | op.opcode << 18 | g("slc", 0) << 17 | g("glc", 0) << 16,
                g("vdst", 0) << 24 | g("tfe", 0) << 23 | g("data", 0) << 8 | g("vaddr", 0)]
    if op.enc == "EXP":
        return [0xF8000000 | g("vm", 0) << 12 | g("done", 0) << 11 | g("compr", 0) << 10 | g("tgt", 0) << 4 |
                g("en", 0), g("s3", 0) << 24 | g("s2", 0) << 16 | g("s1", 0) << 8 | g("s0", 0)]
    return [0xC8000000 | g("vdst", 0) << 18 | op.opcode << 16 | g("attr", 0) << 10 | g("chan", 0) << 8 |
            g("vsrc", 0)]


def exp_target(t):
    if t < 8:
        return f"mrt{t}"
    if t in (8, 9):
        return ["mrtz", "null"][t - 8]
    if 12 <= t < 16:
        return f"pos{t - 12}"
    if t >= 32:
        return f"param{t - 32}"
    return None


def mspell(op, f):
    """(text, canonical fields) or (None, fields) when the fields have no spelling."""
    sig, f, parts, flags = op.sig, dict(f), [], []
    if op.enc == "DS":
        d, a, d0, d1, off = sig
        for key, c in (("vdst", d), ("addr", a), ("d0", d0), ("d1", d1)):
            if c == "-":
                f[key] = 0
            else:
                t = vreg(f[key], int(c))
                if t is None:
                    return None, f
                parts.append(t)
        if off == "2":
            flags += [f"offset0:{f['off0']}"] if f["off0"] else []
            flags += [f"offset1:{f['off1']}"] if f["off1"] else []
        elif f["off0"] or f["off1"]:
            flags.append(f"offset:{f['off1'] << 8 | f['off0']}")
        flags += ["gds"] if f["gds"] else []
    elif op.enc in ("MUBUF", "MTBUF"):
        kind, n = sig[0], int(sig[1])
        if kind == "N":
            f = {k: 0 for k in f}
            return op.name, f
        if op.enc == "MUBUF":
            f["dfmt"] = f["nfmt"] = 0
        else:
            f["lds"] = 0
        if f["addr64"] and (f["offen"] or f["idxen"]):
            return None, f
        naddr = 2 if f["addr64"] or (f["offen"] and f["idxen"]) else 1 if f["offen"] or f["idxen"] else 0
        if not naddr:
            f["vaddr"] = 0
        t = [vreg(f["vdata"], n + (f["tfe"] and kind == "L")),
             vreg(f["vaddr"], naddr) if naddr else "off", rsrc(f["rsrc"], 4), ssrc(f["soff"], 1)[0]]
        if None in t or f["soff"] in (251, 252, 253, 254, 255):
            return None, f
        parts += t
        if op.enc == "MTBUF" and (f["dfmt"], f["nfmt"]) != (1, 0):
            flags.append(f"format:[BUF_DATA_FORMAT_{DFMT[f['dfmt']]},BUF_NUM_FORMAT_{NFMT[f['nfmt']]}]")
        for k in ("idxen", "offen", "addr64"):
            flags += [k] if f[k] else []
        flags += [f"offset:{f['offset']}"] if f["offset"] else []
        for k in ("glc", "slc", "lds", "tfe"):
            flags += [k] if f[k] else []
    elif op.enc == "MIMG":
        kind, na, samp = sig[0], int(sig[1]), sig[2] == "s"
        nd = 4 if kind == "G" else max(1, popcount(f["dmask"]))
        nd += f["tfe"] or f["lwe"]
        if not samp:
            f["ssamp"] = 0
        t = [vreg(f["vdata"], nd), vreg(f["vaddr"], na), rsrc(f["rsrc"], 4 if f["r128"] else 8)]
        if samp:
            t.append(rsrc(f["ssamp"], 4))
        if None in t:
            return None, f
        parts += t
        flags += [f"dmask:{hex(f['dmask'])}"] if f["dmask"] else []
        for k in ("unorm", "glc", "slc", "r128", "tfe", "lwe", "da"):
            flags += [k] if f[k] else []
    elif op.enc == "FLAT":
        kind, n = sig[0], int(sig[1])
        ret = kind == "L" or (kind == "A" and f["glc"])
        nret = (n // 2 if "cmpswap" in op.name else n) + (f["tfe"] and kind == "L")
        if not ret:
            f["vdst"] = 0
        if kind == "L":
            f["data"] = 0
        t = ([vreg(f["vdst"], nret)] if ret else []) + [vreg(f["vaddr"], 2)] + \
            ([vreg(f["data"], n)] if kind != "L" else [])
        if None in t:
            return None, f
        parts += t
        for k in ("glc", "slc", "tfe"):
            flags += [k] if f[k] else []
    elif op.enc == "EXP":
        tgt = exp_target(f["tgt"])
        if tgt is None:
            return None, f
        srcs = []
        for i in range(4):
            on = f["en"] >> i & 1
            if f["compr"]:
                key = "s%d" % (i // 2)
                if i >= 2:
                    f["s%d" % i] = 0
                srcs.append(f"v{f[key]}" if on else "off")
            else:
                if not on:
                    f["s%d" % i] = 0
                srcs.append(f"v{f['s%d' % i]}" if on else "off")
        if f["compr"]:
            for i in (0, 1):  # a pair with neither half enabled encodes no source
                if not (f["en"] >> (2 * i)) & 3:
                    f["s%d" % i] = 0
        text = f"exp {tgt} " + ", ".join(srcs)
        for k in ("compr", "done", "vm"):
            text += f" {k}" if f[k] else ""
        return text, f
    else:
        if sig == "p":
            if f["vsrc"] > 2:
                return None, f
            src = ["p10", "p20", "p0"][f["vsrc"]]
        else:
            src = f"v{f['vsrc']}"
        return f"{op.name} v{f['vdst']}, {src}, attr{f['attr']}.{'xyzw'[f['chan']]}", f
    text = op.name + (" " + ", ".join(parts) if parts else "") + "".join(" " + x for x in flags)
    return text, f


MRANGES = {
    "DS": {"vdst": 256, "addr": 256, "d0": 256, "d1": 256, "off0": 256, "off1": 256, "gds": 2},
    "MUBUF": {"vdata": 256, "vaddr": 256, "rsrc": 32, "soff": 256, "offset": 4096, "offen": 2, "idxen": 2,
              "glc": 2, "addr64": 2, "slc": 2, "tfe": 2, "lds": 2},
    "MTBUF": {"vdata": 256, "vaddr": 256, "rsrc": 32, "soff": 256, "offset": 4096, "offen": 2, "idxen": 2,
              "glc": 2, "addr64": 2, "slc": 2, "tfe": 2, "dfmt": 16, "nfmt": 8},
    "MIMG": {"vdata": 256, "vaddr": 256, "rsrc": 32, "ssamp": 32, "dmask": 16, "unorm": 2, "glc": 2, "da": 2,
             "r128": 2, "tfe": 2, "lwe": 2, "slc": 2},
    "FLAT": {"vdst": 256, "vaddr": 256, "data": 256, "glc": 2, "slc": 2, "tfe": 2},
    "EXP": {"en": 16, "tgt": 64, "compr": 2, "done": 2, "vm": 2, "s0": 256, "s1": 256, "s2": 256, "s3": 256},
    "VINTRP": {"vdst": 256, "vsrc": 256, "attr": 64, "chan": 4},
}
MDEF = {"vdst": 10, "addr": 20, "d0": 30, "d1": 40, "off0": 0, "off1": 0, "gds": 0, "vdata": 10, "vaddr": 20,
        "rsrc": 2, "soff": 1, "offset": 0, "offen": 0, "idxen": 0, "glc": 0, "addr64": 0, "slc": 0, "tfe": 0,
        "lds": 0, "dfmt": 1, "nfmt": 0, "ssamp": 5, "dmask": 0xf, "unorm": 0, "da": 0, "r128": 0, "lwe": 0,
        "data": 30, "en": 0xf, "tgt": 0, "compr": 0, "done": 0, "vm": 0, "s0": 1, "s1": 2, "s2": 3, "s3": 4,
        "attr": 0, "chan": 0, "vsrc": 2}
OPCODE_SPACE = {"DS": 256, "MUBUF": 128, "MTBUF": 8, "MIMG": 128, "FLAT": 128, "EXP": 1, "VINTRP": 4}


def msweep(enc, full):
    out = []
    for field, n in MRANGES[enc].items():
        if n <= 64:
            values = range(n)
        elif field in ("soff",):
            values = range(256) if full else sorted(rng.sample(range(256), 16))
        elif field == "offset":
            values = [1, 2, 4, 255, 256, 2048, 4095] + rng.sample(range(4096), 8)
        else:
            values = sorted({0, 1, 2, 3, 4, 252, 253, 254, 255} | set(rng.sample(range(256), 6 if full else 3)))
        for v in values:
            f = dict(MDEF)
            if enc == "DS":
                f["off0"] = 0
            f[field] = v
            out.append(f)
    for _ in range(40 if full else 10):
        out.append(dict(MDEF) | {k: rng.randrange(n) for k, n in MRANGES[enc].items()})
    return out


def memory_entries(src):
    ops = []
    for enc in ["DS", "MUBUF", "MTBUF", "MIMG", "FLAT", "EXP", "VINTRP"]:
        body = re.search(rf"#define VPGCN_{enc}\(X\)(.*?)(?:\n\n|\n//|\Z)", src, re.S).group(1)
        ops += [MOp(enc, n, int(o), sg) for n, o, sg in re.findall(r'X\((\w+), (\d+), "([^"]*)"\)', body)]
    jobs, seen = [], set()
    for op in ops:
        shape = (op.enc, op.sig)
        full = shape not in seen
        seen.add(shape)
        for f in msweep(op.enc, full):
            text, f = mspell(op, f)
            words = mbuild(op, f)
            if text is None:
                # Export targets 10, 11 and 16-31 do not exist; everything else unspellable is unjudged.
                bad = op.enc == "EXP" and exp_target(f["tgt"]) is None
                jobs.append((words, None, "!" if bad else "?"))
            else:
                jobs.append((words, text, op.enc))
    # Unassigned opcodes: never valid.
    assigned = {(op.enc, op.opcode) for op in ops}
    for enc, space in OPCODE_SPACE.items():
        stub = MOp(enc, "", 0, "")
        for opcode in range(space):
            if (enc, opcode) in assigned:
                continue
            stub.opcode = opcode
            for _ in range(4):
                jobs.append((mbuild(stub, {k: rng.randrange(n) for k, n in MRANGES[enc].items()}), None, "!"))
    texts = [t for _, t, _ in jobs if t]
    results = iter(assemble(texts))
    out, stats = [], {}
    for words, text, verdict in jobs:
        enc = verdict
        if text is None:
            out.append(",".join(f"{w:08x}" for w in words) + " " + verdict)
            continue
        r = next(results)
        ok = r is not None and r[1] == words and "/*" not in r[0]
        stats.setdefault(enc, [0, 0])[0 if ok else 1] += 1
        out.append(",".join(f"{w:08x}" for w in words) + " " + (r[0] if ok else "?"))
        if not ok and r is not None and len(stats.setdefault("mismatch", [])) < 60:
            stats["mismatch"].append(f"{text} -> {r[0]} {[hex(x) for x in r[1]]} != {[hex(x) for x in words]}")
    for k, v in stats.items():
        if k != "mismatch":
            print(f"  {k}: {v[0]} judged valid, {v[1]} unjudged", file=sys.stderr)
    for m in [] if "-q" in sys.argv else stats.get("mismatch", []):
        print("  mismatch:", m, file=sys.stderr)
    return out


def main():
    header, out_path = sys.argv[1], sys.argv[2]
    src = open(header).read()
    ops_by_enc = {}
    for enc in ["SOP2", "SOP1", "SOPC", "SOPK", "SOPP", "SMRD"]:
        body = re.search(rf"#define VPGCN_{enc}\(X\)(.*?)(?:\n\n|\Z)", src, re.S).group(1)
        names = re.findall(r"X\((\w+),", body)
        ops_by_enc[enc] = discover(enc, names)
    run(ops_by_enc, out_path, src)


if __name__ == "__main__":
    main()
