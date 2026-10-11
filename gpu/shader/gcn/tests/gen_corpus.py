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


def run(ops_by_enc, out_path):
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


def main():
    header, out_path = sys.argv[1], sys.argv[2]
    src = open(header).read()
    ops_by_enc = {}
    for enc in ["SOP2", "SOP1", "SOPC", "SOPK", "SOPP", "SMRD"]:
        body = re.search(rf"#define VPGCN_{enc}\(X\)(.*?)(?:\n\n|\Z)", src, re.S).group(1)
        names = re.findall(r"X\((\w+),", body)
        ops_by_enc[enc] = discover(enc, names)
    run(ops_by_enc, out_path)


if __name__ == "__main__":
    main()
