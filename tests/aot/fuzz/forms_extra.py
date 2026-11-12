"""Slots the ISA enumeration (tools/isa_audit) cannot give the fuzzer: integer instructions with varied
operands, counts and immediates, sequences that exercise the lazy flags, string instructions, and the x87
instructions the enumeration lacks. Each slot is a few Intel-syntax instructions; `undef` is the set of
RFLAGS bits the slot leaves undefined (not compared), worked out from the instructions' flag effects.

  slots(rng, n_seq) -> [(text, undef, kind)]
"""
import itertools
import random

C, P, A, Z, S, O = 0x1, 0x4, 0x10, 0x40, 0x80, 0x800
ALL = C | P | A | Z | S | O
CC_FLAGS = {"o": O, "no": O, "b": C, "ae": C, "e": Z, "ne": Z, "be": C | Z, "a": C | Z, "s": S, "ns": S,
            "p": P, "np": P, "l": S | O, "ge": S | O, "le": Z | S | O, "g": Z | S | O}
CCS = list(CC_FLAGS)

# registers a slot may write / only read (rsi, rdi, rbp hold pointers the memory operands use)
W = {8: ["al", "cl", "dl", "bl", "r8b", "r9b", "r10b", "r11b"], 16: ["ax", "cx", "dx", "bx", "r8w", "r11w", "r15w"],
     32: ["eax", "ecx", "edx", "ebx", "r8d", "r9d", "r12d", "r15d"], 64: ["rax", "rcx", "rdx", "rbx", "r8", "r10", "r13", "r15"]}
HIGH8 = ["ah", "ch", "dh", "bh"]
RD = {8: ["sil", "dil"], 16: ["si", "di", "bp"], 32: ["esi", "edi", "ebp"], 64: ["rsi", "rdi", "rbp"]}
PTR = {8: "byte", 16: "word", 32: "dword", 64: "qword"}
MEMS = ["[rdi]", "[rdi+0x20]", "[rdi+0x41]", "[rsi+0x7f]", "[rsi-8]", "[rsi+0x333]"]
IMMS = {8: [0, 1, 2, 0x7f, 0x80, 0xff, 0x55], 16: [0, 1, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff, 0x1234],
        32: [0, 1, 0x7f, 0x80, 0xff, 0x7fffffff, 0x80000000, 0xffffffff, 0x12345678],
        64: [0, 1, 0x7f, 0x80, 0xffffffff, 0x7fffffff, 0xffffffffffffffff, 0xffffffff80000000, 0x7fffffffffffffff]}
COUNTS = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255]
MASK = {8: 0xff, 16: 0xffff, 32: 0xffffffff, 64: 0xffffffffffffffff}


class I:
    def __init__(self, text, reads=0, wdef=0, wundef=0):
        self.text, self.reads, self.wdef, self.wundef = text, reads, wdef, wundef


def imm_for(bits, v):
    """An immediate that fits the instruction's immediate field (64-bit ones take a sign-extended 32 bits)."""
    if bits == 64:
        v &= MASK[64]
        if v >= 0x80000000 and v < 0xffffffff80000000:
            v = 0x7fffffff
    return hex(v)


def regs(rng, bits, high=False):
    pool = W[bits]
    return rng.choice(pool + (HIGH8 if bits == 8 and high else []))


def rm(rng, bits, allow_high=False):
    """A register or memory operand of `bits` and whether it is memory."""
    if rng.random() < 0.4:
        return f"{PTR[bits]} ptr {rng.choice(MEMS)}", True
    if allow_high and bits == 8 and rng.random() < 0.3:
        return rng.choice(HIGH8), False
    return regs(rng, bits), False


def src(rng, bits, high_ok=True):
    r = rng.random()
    if r < 0.3:
        return rng.choice(W[bits] + RD[bits])
    return regs(rng, bits)


def rotate_effect(op, bits, cnt):
    """(reads, wdef, wundef) of a shift/rotate/double shift by the constant `cnt`."""
    m = cnt & (63 if bits == 64 else 31)
    if op in ("shld", "shrd") and bits == 16 and m > 16:
        return None
    if m == 0:
        return (C if op in ("rcl", "rcr") else 0, 0, 0)
    if op in ("shl", "shr"):
        return (0, P | Z | S | (C if m <= bits else 0) | (O if m == 1 else 0), A | (O if m > 1 else 0) | (C if m > bits else 0))
    if op == "sar":
        return (0, P | Z | S | C | (O if m == 1 else 0), A | (O if m > 1 else 0))
    if op in ("shld", "shrd"):
        return (0, P | Z | S | C | (O if m == 1 else 0), A | (O if m > 1 else 0))
    if op in ("rol", "ror"):
        return (0, C | (O if m == 1 else 0), O if m > 1 else 0)
    # rcl / rcr
    eff = m % (bits + 1) if bits < 32 else m
    if eff == 0:
        return (C, 0, 0)
    return (C, C | (O if m == 1 else 0), O if m > 1 else 0)


def gen_alu(rng):
    """One flag-writing instruction (a few of them read a flag), random operands."""
    kind = rng.choice(["alu", "alu", "alu", "incdec", "neg", "shift", "shift", "dshift", "mul", "imul", "bt", "bs", "popcnt", "xadd", "cmpxchg", "cmov", "setcc", "bmi", "movx"])
    bits = rng.choice([8, 16, 32, 64])
    if kind == "alu":
        op = rng.choice(["add", "adc", "sub", "sbb", "and", "or", "xor", "cmp", "test"])
        dst, dmem = rm(rng, bits, True)
        form = rng.choice(["rr", "ri", "mr", "rm"]) if not dmem else rng.choice(["mr", "mi"])
        if form in ("rm", "mr", "mi") and dmem is False and form == "rm":
            s = f"{PTR[bits]} ptr {rng.choice(MEMS)}"
            d = regs(rng, bits)
            text = f"{op} {d}, {s}"
        elif form in ("ri", "mi"):
            text = f"{op} {dst}, {imm_for(bits, rng.choice(IMMS[bits]))}"
        elif form == "mr" or form == "rr":
            sr = src(rng, bits)
            text = f"{op} {dst}, {sr}"
        else:
            text = f"{op} {dst}, {src(rng, bits)}"
        if text.split()[0] in ("test",) and "ptr" in text.split(",")[0] and False:
            pass
        reads = C if op in ("adc", "sbb") else 0
        if op in ("and", "or", "xor", "test"):
            return I(text, reads, C | P | Z | S | O, A)
        return I(text, reads, ALL, 0)
    if kind == "incdec":
        op = rng.choice(["inc", "dec"])
        d, _ = rm(rng, bits, True)
        return I(f"{op} {d}", 0, P | A | Z | S | O, 0)
    if kind == "neg":
        op = rng.choice(["neg", "not"])
        d, _ = rm(rng, bits, True)
        return I(f"{op} {d}", 0, ALL if op == "neg" else 0, 0)
    if kind == "shift":
        op = rng.choice(["shl", "shr", "sar", "rol", "ror", "rcl", "rcr"])
        d, dmem = rm(rng, bits, True)
        cnt = rng.choice(COUNTS)
        eff = rotate_effect(op, bits, cnt)
        form = rng.choice(["imm", "cl", "one"])
        if form == "one":
            cnt = 1
            eff = rotate_effect(op, bits, 1)
            return I(f"{op} {d}, 1", eff[0], eff[1], eff[2])
        if form == "imm":
            return I(f"{op} {d}, {cnt & 0xff}", eff[0], eff[1], eff[2])
        if "cl" in d.split() or d == "cl" or d.endswith("cl") and d in W[8]:
            d = rng.choice([x for x in W[8] if x != "cl"]) if bits == 8 else d
        if bits == 8 and d == "ch":
            d = "dh"
        return I(f"mov cl, {cnt & 0xff}; {op} {d}, cl", eff[0], eff[1], eff[2])
    if kind == "dshift":
        op = rng.choice(["shld", "shrd"])
        bits = rng.choice([16, 32, 64])
        d, _ = rm(rng, bits)
        s = regs(rng, bits)
        cnt = rng.choice(COUNTS)
        eff = rotate_effect(op, bits, cnt)
        if eff is None:
            cnt = rng.choice([1, 2, 7, 16])
            eff = rotate_effect(op, bits, cnt)
        if rng.random() < 0.5:
            return I(f"{op} {d}, {s}, {cnt}", eff[0], eff[1], eff[2])
        if d in ("cl", "cx", "ecx", "rcx") or s in ("cx", "ecx", "rcx"):
            d = {16: "ax", 32: "eax", 64: "rax"}[bits]
        if s in ("cx", "ecx", "rcx"):
            s = {16: "bx", 32: "ebx", 64: "rbx"}[bits]
        return I(f"mov cl, {cnt}; {op} {d}, {s}, cl", eff[0], eff[1], eff[2])
    if kind == "mul":
        op = rng.choice(["mul", "imul"])
        d, _ = rm(rng, bits)
        return I(f"{op} {d}", 0, C | O, P | A | Z | S)
    if kind == "imul":
        bits = rng.choice([16, 32, 64])
        d = regs(rng, bits)
        if rng.random() < 0.5:
            s, _ = rm(rng, bits)
            return I(f"imul {d}, {s}", 0, C | O, P | A | Z | S)
        s, _ = rm(rng, bits)
        return I(f"imul {d}, {s}, {imm_for(bits, rng.choice(IMMS[bits]))}", 0, C | O, P | A | Z | S)
    if kind == "bt":
        bits = rng.choice([16, 32, 64])
        op = rng.choice(["bt", "bts", "btr", "btc"])
        d, dmem = rm(rng, bits)
        if rng.random() < 0.5:
            return I(f"{op} {d}, {rng.choice([0, 1, 5, 15, 16, 31, 32, 63, 64, 100, 255])}", 0, C, O | S | A | P)
        if dmem:
            d = f"{PTR[bits]} ptr [rsi+0x300]"  # a register offset reaches below the operand: stay clear of the start of the data
        s = regs(rng, bits)
        if d == s:
            s = {16: "cx", 32: "ecx", 64: "rcx"}[bits] if d not in ("cx", "ecx", "rcx") else {16: "dx", 32: "edx", 64: "rdx"}[bits]
        # a memory operand with a register offset reaches outside the operand: keep the offset small
        ext = {16: f"movsx {s}, {s[:-1] + 'l' if s.endswith('x') else s}", 32: f"movsx {s}, {s[:-1] + 'l' if s.endswith('x') else s}"}
        small = {16: "movsx {r}, {l}", 32: "movsx {r}, {l}", 64: "movsx {r}, {l}"}
        low8 = {"ax": "al", "cx": "cl", "dx": "dl", "bx": "bl", "eax": "al", "ecx": "cl", "edx": "dl", "ebx": "bl", "rax": "al", "rcx": "cl", "rdx": "dl", "rbx": "bl"}
        if dmem and s in low8:
            return I(f"movsx {s}, {low8[s]}; {op} {d}, {s}", 0, C, O | S | A | P)
        if dmem:
            return None
        return I(f"{op} {d}, {s}", 0, C, O | S | A | P)
    if kind == "bs":
        bits = rng.choice([16, 32, 64])
        op = rng.choice(["bsf", "bsr", "tzcnt", "lzcnt"])
        d = regs(rng, bits)
        s, _ = rm(rng, bits)
        if op in ("bsf", "bsr"):
            return I(f"{op} {d}, {s}", 0, Z, C | P | A | S | O)
        return I(f"{op} {d}, {s}", 0, C | Z, P | A | S | O)
    if kind == "popcnt":
        bits = rng.choice([16, 32, 64])
        d = regs(rng, bits)
        s, _ = rm(rng, bits)
        return I(f"popcnt {d}, {s}", 0, ALL, 0)
    if kind == "xadd":
        d, dmem = rm(rng, bits)
        s = regs(rng, bits)
        lock = "lock " if dmem and rng.random() < 0.5 else ""
        return I(f"{lock}xadd {d}, {s}", 0, ALL, 0)
    if kind == "cmpxchg":
        d, dmem = rm(rng, bits)
        s = regs(rng, bits)
        if s in ("al", "ax", "eax", "rax"):
            s = {8: "bl", 16: "bx", 32: "ebx", 64: "rbx"}[bits]
        if d in ("al", "ax", "eax", "rax"):
            d = {8: "dl", 16: "dx", 32: "edx", 64: "rdx"}[bits]
        lock = "lock " if dmem and rng.random() < 0.5 else ""
        # half the time the accumulator equals the operand, so both outcomes are taken
        eq = ""
        if rng.random() < 0.5:
            acc = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[bits]
            eq = f"mov {acc}, {d}; "
        return I(f"{eq}{lock}cmpxchg {d}, {s}", 0, ALL, 0)
    if kind == "cmov":
        bits = rng.choice([16, 32, 64])
        cc = rng.choice(CCS)
        d = regs(rng, bits)
        s, _ = rm(rng, bits)
        return I(f"cmov{cc} {d}, {s}", CC_FLAGS[cc], 0, 0)
    if kind == "setcc":
        cc = rng.choice(CCS)
        d, _ = rm(rng, 8, True)
        return I(f"set{cc} {d}", CC_FLAGS[cc], 0, 0)
    if kind == "bmi":
        bits = rng.choice([32, 64])
        op = rng.choice(["andn", "blsi", "blsr", "blsmsk", "bextr", "bzhi"])
        d = regs(rng, bits)
        a, _ = rm(rng, bits)
        b = regs(rng, bits)
        if op == "andn":
            return I(f"andn {d}, {b}, {a}", 0, S | Z | C | O, A | P)
        if op in ("blsi", "blsr", "blsmsk"):
            return I(f"{op} {d}, {a}", 0, S | Z | C | O, A | P)
        if op == "bextr":
            return I(f"{op} {d}, {a}, {b}", 0, Z | C | O, A | S | P)
        return I(f"{op} {d}, {a}, {b}", 0, S | Z | C | O, A | P)
    # movzx / movsx / movsxd and friends: no flags
    sb = rng.choice([8, 16])
    db = rng.choice([b for b in (16, 32, 64) if b > sb])
    s, _ = rm(rng, sb, db != 64)
    op = rng.choice(["movzx", "movsx"])
    return I(f"{op} {regs(rng, db)}, {s}", 0, 0, 0)


def gen_consumer(rng):
    """A flag reader that writes no flag; a Jcc jumps over one instruction (a block boundary)."""
    r = rng.random()
    cc = rng.choice(CCS)
    if r < 0.3:
        bits = rng.choice([16, 32, 64])
        return I(f"cmov{cc} {regs(rng, bits)}, {regs(rng, bits)}", CC_FLAGS[cc], 0, 0)
    if r < 0.5:
        return I(f"set{cc} {regs(rng, 8, True)}", CC_FLAGS[cc], 0, 0)
    if r < 0.6:
        return I("lahf", C | P | A | Z | S, 0, 0)
    if r < 0.7:
        return I("pushfq; pop r11", ALL, 0, 0)
    if r < 0.75:
        return I("cmc", C, C, 0)
    if r < 0.8:
        return I("sahf", 0, C | P | A | Z | S, 0)
    if r < 0.85:
        return I(rng.choice(["clc", "stc"]), 0, C, 0)
    # a jump over one instruction
    inner = gen_alu(rng)
    while inner is None:
        inner = gen_alu(rng)
    # the jumped-over instruction may or may not run: its flag effects become undefined unless it writes none
    return I(f"j{cc} 1f; {inner.text}; 1:", CC_FLAGS[cc] | inner.reads, 0, inner.wdef | inner.wundef)


def combine(rng, n):
    """A sequence of n instructions whose results are all defined, or None."""
    undef = 0
    out = []
    for _ in range(n):
        ins = gen_alu(rng) if rng.random() < 0.65 else gen_consumer(rng)
        if ins is None:
            return None
        if ins.reads & undef:
            continue
        undef = (undef & ~ins.wdef) | ins.wundef
        out.append(ins)
    if len(out) < 2:
        return None
    return "; ".join(i.text for i in out), undef


def string_slots(rng):
    out = []
    for op, w in itertools.product(["movs", "stos", "lods", "scas", "cmps"], [8, 16, 32, 64]):
        sfx = {8: "b", 16: "w", 32: "d", 64: "q"}[w]
        for rep in ["", "rep ", "repe ", "repne "]:
            if rep in ("repe ", "repne ") and op not in ("scas", "cmps"):
                continue
            if rep == "rep " and op in ("scas", "cmps"):
                continue
            for n in ([1, 3, 16] if rep else [0]):
                pre = "add rdi, 0x400; "
                pre += f"mov ecx, {n}; " if rep else ""
                text = f"{pre}{rep}{op}{sfx}" if not (op == "movs" and w == 32) else f"{pre}{rep}movsd"
                if op == "stos":
                    text = f"{pre}{rep}stos{sfx}"
                if op == "lods":
                    text = f"{pre}{rep}lods{sfx}"
                if op == "scas":
                    text = f"{pre}{rep}scas{sfx}"
                if op == "cmps":
                    text = f"{pre}{rep}cmps{sfx}" if w != 32 else f"{pre}{rep}cmpsd"
                # all flags defined for scas/cmps; for the others no flag is touched
                out.append((text, 0, 0))
    return out


def misc_slots(rng):
    out = []
    for bits in (16, 32, 64):
        r = {16: "ax", 32: "eax", 64: "rax"}[bits]
        out.append(({16: "push ax; pop cx", 32: "push rax; pop rcx", 64: "push rax; pop rcx"}[bits], 0))
    out += [("push 0x7fffffff; pop rdx", 0), ("push -1; pop rdx", 0), ("push word ptr [rdi+0x20]; pop word ptr [rdi+0x40]", 0),
            ("push qword ptr [rdi+0x20]; pop qword ptr [rdi+0x48]", 0), ("push rsp; pop rax", 0),
            ("pushfq; pop rax", 0), ("and eax, 0xcd5; push rax; popfq; pushfq; pop rcx", 0),
            ("push rbp; mov rbp, rsp; push rax; push rbx; leave", 0), ("enter 16, 0; leave", 0), ("enter 0x20, 3; leave", 0), ("enter 8, 1; leave", 0),
            ("cbw", 0), ("cwde", 0), ("cdqe", 0), ("cwd", 0), ("cdq", 0), ("cqo", 0), ("bswap eax", 0), ("bswap rcx", 0),
            ("xchg al, ah", 0), ("xchg ax, bx", 0), ("xchg eax, ecx", 0), ("xchg rax, rdx", 0), ("xchg byte ptr [rdi+8], bl", 0),
            ("xchg qword ptr [rdi+8], rbx", 0), ("lock cmpxchg8b qword ptr [rdi+0x20]", Z | 0), ("cmpxchg8b qword ptr [rdi+0x20]", 0),
            ("mov rax, [rdi+0x20]; mov rdx, [rdi+0x28]; mov [rdi+0x40], rax; mov [rdi+0x48], rdx; lock cmpxchg16b xmmword ptr [rdi+0x40]", 0),
            ("lock cmpxchg16b xmmword ptr [rdi+0x40]", 0),
            ("lea eax, [ebx+ecx*2+0x7fffffff]", 0), ("lea rax, [rbx+rcx*8-1]", 0), ("lea ax, [rbx+rcx*4+2]", 0), ("lea eax, [rip+0x10]", 0),
            ("lea rax, [rip-0x100]", 0), ("lea r9d, [r8*4+0x80]", 0), ("lea rax, [rbx+rbx*2]", 0), ("lea eax, [eax*4+7]", 0),
            ("mov eax, [edi+0x20]", 0), ("mov eax, [edi+ecx]", None), ("mov al, [rdi+0x41]; mov ah, [rdi+0x42]", 0),
            ("xlat", None), ("movbe eax, [rdi+0x20]", 0), ("movbe [rdi+0x20], rax", 0), ("movbe cx, [rdi+0x20]", 0),
            ("crc32 eax, cl", 0), ("crc32 eax, dx", 0), ("crc32 eax, ecx", 0), ("crc32 rax, rcx", 0), ("crc32 eax, byte ptr [rdi+0x20]", 0),
            ("mulx rax, rcx, rbx", 0), ("mulx eax, ecx, ebx", 0), ("mulx rax, rax, rbx", 0), ("mulx rax, rbx, qword ptr [rdi]", 0),
            ("sarx eax, ecx, edx", 0), ("shlx rax, rcx, rdx", 0), ("shrx r8d, ecx, edx", 0), ("rorx eax, ecx, 5", 0), ("rorx rax, rcx, 63", 0),
            ("rorx eax, ecx, 37", 0), ("pdep rax, rcx, rdx", 0), ("pext rax, rcx, rdx", 0), ("pdep eax, ecx, edx", 0), ("pext eax, ecx, edx", 0),
            ("movzx eax, ah", 0), ("movsx ax, ch", 0), ("mov dl, ah", 0), ("mov ah, dl", 0), ("mov bh, ch", 0), ("add ah, bl", 0),
            ("movnti [rdi+0x20], eax", 0), ("movnti [rdi+0x20], rax", 0),
            ("mov ecx, 5; loop 1f; 1: add eax, 1", 0), ("mov ecx, 1; loop 1f; 1: add eax, 1", 0), ("xor ecx, ecx; loop 1f; 1: add eax, 1", 0),
            ("mov ecx, 3; cmp eax, eax; loope 1f; mov edx, 7; 1:", 0), ("mov ecx, 3; test eax, eax; loopne 1f; mov edx, 7; 1:", 0),
            ("jrcxz 1f; mov edx, 7; 1:", 0), ("xor ecx, ecx; jrcxz 1f; mov edx, 7; 1:", 0), ("jecxz 1f; mov edx, 7; 1:", 0),
            ("call 1f; jmp 2f; 1: add rax, 1; ret; 2: add rax, 2", 0), ("lea rcx, [rip+1f]; call rcx; jmp 2f; 1: ret; 2: add rax, 2", 0),
            ("lea rcx, [rip+1f]; jmp rcx; 1: add rax, 2", 0), ("lea rcx, [rip+1f]; push rcx; ret; 1: add rax, 2", 0),
            ("call 1f; 1: pop rax", 0), ("lea rdx, [rip+1f]; mov [rdi], rdx; jmp qword ptr [rdi]; 1: add rax, 2", 0),
            ("lea rdx, [rip+1f]; mov [rdi], rdx; call qword ptr [rdi]; jmp 2f; 1: ret; 2: add rbx, 2", 0),
            ("mov rax, 0x123456789abcdef0; mov rbx, rax; ror rbx, 8; bswap rax", 0x800), ("mov eax, 0x80000000; cdq; idiv ecx", None),
            ]
    return [(t, u) for t, u in out if u is not None]


def div_slots():
    out = []
    for bits in (8, 16, 32, 64):
        for op in ("div", "idiv"):
            for form in ("xor", "sext", "keep", "small"):
                r = {8: "cl", 16: "cx", 32: "ecx", 64: "rcx"}[bits]
                pre = ""
                a = {8: "al", 16: "ax", 32: "eax", 64: "rax"}[bits]
                if form == "xor":
                    pre = {8: "movzx eax, al; ", 16: "xor edx, edx; ", 32: "xor edx, edx; ", 64: "xor edx, edx; "}[bits]
                elif form == "sext":
                    pre = {8: "cbw; ", 16: "cwd; ", 32: "cdq; ", 64: "cqo; "}[bits]
                elif form == "small":
                    pre = {8: "movzx eax, al; shr ecx, 4; ", 16: "xor edx, edx; shr ecx, 9; ", 32: "xor edx, edx; shr ecx, 20; ", 64: "xor edx, edx; shr rcx, 40; "}[bits]
                out.append((f"{pre}{op} {r}", ALL, 0))
                if bits != 8:
                    out.append((f"{pre}{op} {PTR[bits]} ptr [rdi+0x20]", ALL, 0))
    return out


def x87_slots(rng):
    """x87 instructions without a register operand form in the enumeration, and the register forms with other indexes."""
    out = []
    for m in ["fchs", "fabs", "fsqrt", "frndint", "ftst", "fxam", "fprem", "fprem1", "fscale", "fxtract", "fsin", "fcos", "fsincos",
              "fptan", "fpatan", "f2xm1", "fyl2x", "fyl2xp1", "fld1", "fldz", "fldpi", "fldl2e", "fldl2t", "fldln2", "fldlg2",
              "fincstp", "fdecstp", "fnclex", "fninit", "fnop", "fcompp", "fucompp", "fxch", "fnstsw ax", "fstp st(0)", "fld st(0)"]:
        out.append((m, 0, 0))
    for i in (0, 1, 2, 3, 5, 7):
        for m in ["fld st({i})", "fxch st({i})", "fst st({i})", "fstp st({i})", "fadd st(0), st({i})", "fadd st({i}), st(0)", "faddp st({i}), st(0)",
                  "fsub st(0), st({i})", "fsub st({i}), st(0)", "fsubp st({i}), st(0)", "fsubr st(0), st({i})", "fsubr st({i}), st(0)", "fsubrp st({i}), st(0)",
                  "fmul st(0), st({i})", "fmul st({i}), st(0)", "fmulp st({i}), st(0)", "fdiv st(0), st({i})", "fdiv st({i}), st(0)", "fdivp st({i}), st(0)",
                  "fdivr st(0), st({i})", "fdivr st({i}), st(0)", "fdivrp st({i}), st(0)", "fcom st({i})", "fcomp st({i})", "fucom st({i})", "fucomp st({i})",
                  "fcmovb st(0), st({i})", "fcmovnbe st(0), st({i})", "fcmove st(0), st({i})", "fcmovu st(0), st({i})", "ffree st({i})"]:
            if i == 0 and ("fadd st(0), st(0)" in m.format(i=0) or True):
                pass
            out.append((m.format(i=i), 0, 0))
        out.append((f"fcomi st(0), st({i})", 0, 0))
        out.append((f"fucomip st(0), st({i})", 0, 0))
    for sz, ptr in (("dword", "dword"), ("qword", "qword"), ("tbyte", "tbyte")):
        out.append((f"fld {ptr} ptr [rdi+0x40]", 0, 0))
        out.append((f"fstp {ptr} ptr [rdi+0x40]", 0, 0))
        out.append((f"fst {ptr} ptr [rdi+0x40]" if sz != "tbyte" else "fstp tbyte ptr [rdi+0x40]", 0, 0))
    for mn in ["fadd", "fsub", "fsubr", "fmul", "fdiv", "fdivr", "fcom", "fcomp"]:
        out.append((f"{mn} dword ptr [rdi+0x40]", 0, 0))
        out.append((f"{mn} qword ptr [rdi+0x40]", 0, 0))
    for mn in ["fiadd", "fisub", "fisubr", "fimul", "fidiv", "fidivr", "ficom", "ficomp"]:
        out.append((f"{mn} word ptr [rdi+0x40]", 0, 0))
        out.append((f"{mn} dword ptr [rdi+0x40]", 0, 0))
    for mn in ["fild", "fist", "fistp", "fisttp"]:
        for p in ("word", "dword", "qword"):
            if mn in ("fist",) and p == "qword":
                continue
            out.append((f"{mn} {p} ptr [rdi+0x40]", 0, 0))
    out += [("fnstcw word ptr [rdi+0x40]", 0, 0), ("fnstsw word ptr [rdi+0x40]", 0, 0), ("fnstenv [rdi+0x40]; fldenv [rdi+0x40]", 0, 0),
            ("fnsave [rdi+0x40]; frstor [rdi+0x40]", 0, 0), ("fbstp tbyte ptr [rdi+0x40]; fbld tbyte ptr [rdi+0x40]", 0, 0),
            ("fldcw word ptr [rdi+0x20]", None, 0), ("mov word ptr [rdi+0x40], 0x37f; fldcw word ptr [rdi+0x40]", 0, 0),
            ("mov word ptr [rdi+0x40], 0x73f; fldcw word ptr [rdi+0x40]", 0, 0), ("mov word ptr [rdi+0x40], 0xb7f; fldcw word ptr [rdi+0x40]", 0, 0),
            ("mov word ptr [rdi+0x40], 0xf7f; fldcw word ptr [rdi+0x40]", 0, 0), ("mov word ptr [rdi+0x40], 0x27f; fldcw word ptr [rdi+0x40]", 0, 0),
            ("mov word ptr [rdi+0x40], 0x77f; fldcw word ptr [rdi+0x40]", 0, 0), ("fnstenv [rdi+0x40]", 0, 0),
            ("fxsave [rdi+0x40]", 0, 0), ("fxsave [rdi+0x40]; fxrstor [rdi+0x40]", 0, 0), ("fxsave [rdi+0x40]; fninit; fxrstor [rdi+0x40]", 0, 0)]
    return [(t, u, k) for t, u, k in out if u is not None]


def slots(seed, n_seq):
    rng = random.Random(seed)
    res = []
    for _ in range(3):
        for _ in range(500):
            ins = gen_alu(rng)
            if ins:
                res.append((ins.text, ins.wundef, 0))
    for _ in range(n_seq):
        c = combine(rng, rng.randint(2, 5))
        if c:
            res.append((c[0], c[1], 0))
    for t, u, k in string_slots(rng):
        res.append((t, 0, 0))
    for t, u in misc_slots(rng):
        res.append((t, u & ALL, 0))
    for t, u, k in div_slots():
        res.append((t, u, k))
    for t, u, k in x87_slots(rng):
        res.append((t, u, k))
    return res
