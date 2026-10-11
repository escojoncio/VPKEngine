#include "vpgcn/gcn.h"

#include <array>
#include <cstdio>

namespace vpgcn {
namespace {

struct OpInfo {
    std::string_view name;
    Encoding enc;
    uint16_t opcode;
    std::string_view sig;
};

constexpr OpInfo kOps[] = {
#define X_SOP2(n, o, s) {#n, Encoding::SOP2, o, s},
#define X_SOP1(n, o, s) {#n, Encoding::SOP1, o, s},
#define X_SOPC(n, o, s) {#n, Encoding::SOPC, o, s},
#define X_SOPP(n, o, s) {#n, Encoding::SOPP, o, s},
#define X_SOPK(n, o, s) {#n, Encoding::SOPK, o, s},
#define X_SMRD(n, o, s) {#n, Encoding::SMRD, o, s},
    VPGCN_SOP2(X_SOP2) VPGCN_SOP1(X_SOP1) VPGCN_SOPC(X_SOPC) VPGCN_SOPP(X_SOPP)
    VPGCN_SOPK(X_SOPK) VPGCN_SMRD(X_SMRD)
#undef X_SOP2
#undef X_SOP1
#undef X_SOPC
#undef X_SOPP
#undef X_SOPK
#undef X_SMRD
};
static_assert(std::size(kOps) == size_t(Op::Count));

constexpr size_t kEncodings = 6;
constexpr size_t kMaxOpcode = 128;

// Per-encoding opcode -> Op map (Op::Count = unassigned).
constexpr auto kOpcodeMap = [] {
    std::array<std::array<Op, kMaxOpcode>, kEncodings> m{};
    for (auto& row : m) row.fill(Op::Count);
    for (size_t i = 0; i < std::size(kOps); ++i)
        m[size_t(kOps[i].enc)][kOps[i].opcode] = Op(i);
    return m;
}();

// ---- Operand codes (the 8/9-bit scalar source encoding) -----------------------------------------

constexpr uint16_t kFlatScratch = 104, kVcc = 106, kTba = 108, kTma = 110, kTtmp0 = 112,
                   kM0 = 124, kExec = 126, kIntZero = 128, kIntMax = 192, kIntNegMax = 208,
                   kFloatFirst = 240, kFloatLast = 247, kVccz = 251, kExecz = 252, kScc = 253,
                   kLiteral = 255;
constexpr uint32_t kFloatConsts[] = {0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                                     0x40000000, 0xc0000000, 0x40800000, 0xc0800000};
constexpr const char* kFloatNames[] = {"0.5", "-0.5", "1.0", "-1.0", "2.0", "-2.0", "4.0", "-4.0"};

// A register operand spanning `count` dwords starting at `code`, or false if the hardware has no such
// register tuple (alignment, range, or a 32-bit-only register used as 64-bit).
bool regOperand(uint16_t code, unsigned count, Operand& o) {
    o.code = code;
    o.count = uint8_t(count);
    unsigned align = count >= 4 ? 4 : count;
    if (code < kFlatScratch) {
        if (code % align || code + count > kFlatScratch) return false;
        o.kind = OperandKind::Sgpr;
        o.index = code;
        return true;
    }
    if (code >= kTtmp0 && code < kM0) {
        unsigned t = code - kTtmp0;
        if (t % align || t + count > 12) return false;
        o.kind = OperandKind::Ttmp;
        o.index = uint16_t(t);
        return true;
    }
    bool pair = code == kFlatScratch || code == kVcc || code == kTba || code == kTma || code == kExec;
    bool single = pair || code == kFlatScratch + 1 || code == kVcc + 1 || code == kTba + 1 ||
                  code == kTma + 1 || code == kM0 || code == kExec + 1;
    if (!(count == 1 ? single : count == 2 && pair)) return false;
    o.kind = OperandKind::Special;
    return true;
}

// Source classes, from the opcode tables' signature characters.
enum class Src : uint8_t { Any, NoLiteral, RegOrCond, RegOnly };

// Decodes a scalar source operand; a literal is only flagged here and filled in by decode().
bool srcOperand(uint16_t code, unsigned count, Src cls, Operand& o, bool& usesLiteral) {
    if (code < 128) return regOperand(code, count, o);
    o = {};
    o.code = code;
    o.count = uint8_t(count);
    bool cond = code == kVccz || code == kExecz || code == kScc;
    if (cls == Src::RegOnly || (cls == Src::RegOrCond && !cond)) return false;
    if (cond) {
        o.kind = OperandKind::Special;
        return true;
    }
    if (code <= kIntNegMax) {
        o.kind = OperandKind::Const;
        o.value = code <= kIntMax ? uint32_t(code - kIntZero) : uint32_t(int32_t(kIntMax) - code);
        return true;
    }
    if (code >= kFloatFirst && code <= kFloatLast) {
        o.kind = OperandKind::Const;
        o.isFloat = true;
        o.value = kFloatConsts[code - kFloatFirst];
        return true;
    }
    if (code == kLiteral && cls == Src::Any) {
        o.kind = OperandKind::Literal;
        usesLiteral = true;
        return true;
    }
    return false;
}

unsigned sigCount(char c) {
    return c == 'G' ? 16 : c == 'k' || c == 'R' ? 2 : c == 'r' ? 1 : unsigned(c - '0');
}
Src sigClass(char c) {
    return c == 'k' ? Src::NoLiteral : c == 'r' ? Src::RegOrCond : c == 'R' ? Src::RegOnly : Src::Any;
}

// ---- Decoding per encoding -----------------------------------------------------------------------

bool decodeScalarAlu(uint32_t w, Instruction& in, bool& lit) {
    std::string_view sig = kOps[size_t(in.op)].sig;
    uint16_t sdst = (w >> 16) & 0x7f, s0 = w & 0xff, s1 = (w >> 8) & 0xff;
    if (sig[0] != '-' && !regOperand(sdst, sigCount(sig[0]), in.dst)) return false;
    if (sig[1] != '-' && !srcOperand(s0, sigCount(sig[1]), sigClass(sig[1]), in.src[0], lit))
        return false;
    if (sig.size() > 2 && sig[2] != '-' &&
        !srcOperand(s1, sigCount(sig[2]), sigClass(sig[2]), in.src[1], lit))
        return false;
    return true;
}

bool decodeSopk(uint32_t w, Instruction& in, bool& lit) {
    char k = kOps[size_t(in.op)].sig[0];
    if (k != 'I' && !regOperand((w >> 16) & 0x7f, k == 'b' ? 2 : 1, in.dst)) return false;
    if (k == 'I') {
        in.src[0].kind = OperandKind::Literal;
        in.src[0].code = kLiteral;
        in.src[0].count = 1;
        lit = true;
    }
    return true;
}

bool decodeSopp(Instruction& in) {
    char k = kOps[size_t(in.op)].sig[0];
    return k != '-' || in.simm16 == 0;
}

bool decodeSmrd(uint32_t w, Instruction& in, bool& lit) {
    std::string_view sig = kOps[size_t(in.op)].sig;
    // Loads cannot target m0 or exec.
    uint16_t sdst = (w >> 15) & 0x7f;
    if (sig[0] != '-' && (sdst >= kM0 || !regOperand(sdst, sigCount(sig[0]), in.dst))) return false;
    if (sig[1] == '-') return true;
    if (!regOperand(uint16_t(((w >> 9) & 0x3f) * 2), sigCount(sig[1]), in.src[0])) return false;
    Operand& off = in.src[1];
    if (w & 0x100) {
        off.kind = OperandKind::Imm;
        off.value = w & 0xff;
        return true;
    }
    uint16_t code = w & 0xff;
    if (code == kLiteral) {
        off.kind = OperandKind::Literal;
        off.code = code;
        off.count = 1;
        lit = true;
        return true;
    }
    if (code == kVccz || code == kExecz || code == kScc) {
        off = {OperandKind::Special, code, 0, 1, 0, false};
        return true;
    }
    return code < 128 && regOperand(code, 1, off);
}

// ---- Printing -----------------------------------------------------------------------------------

std::string hex(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%x", v);
    return b;
}

std::string regName(const Operand& o) {
    static constexpr const char* kSpecial[] = {"flat_scratch", "vcc", "tba", "tma"};
    std::string prefix = o.kind == OperandKind::Ttmp ? "ttmp" : "s";
    if (o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp) {
        if (o.count == 1) return prefix + std::to_string(o.index);
        return prefix + "[" + std::to_string(o.index) + ":" + std::to_string(o.index + o.count - 1) +
               "]";
    }
    switch (o.code) {
        case kM0: return "m0";
        case kVccz: return "src_vccz";
        case kExecz: return "src_execz";
        case kScc: return "src_scc";
        default: break;
    }
    unsigned base = o.code >= kExec ? 4 : (o.code - kFlatScratch) / 2;
    std::string name = base == 4 ? "exec" : kSpecial[base];
    if (o.count == 2) return name;
    return name + (o.code & 1 ? "_hi" : "_lo");
}

// A 32-bit value as llvm prints a literal: an inline-constant spelling when one exists, else hex.
std::string literalText(uint32_t v, bool wide) {
    int32_t s = int32_t(v);
    if (s >= -16 && s <= 64 && (!wide || s >= 0)) return std::to_string(s);
    if (!wide)
        for (size_t i = 0; i < std::size(kFloatConsts); ++i)
            if (kFloatConsts[i] == v) return kFloatNames[i];
    return hex(v);
}

std::string operandText(const Operand& o) {
    switch (o.kind) {
        case OperandKind::Const:
            return o.isFloat ? kFloatNames[o.code - kFloatFirst] : std::to_string(int32_t(o.value));
        case OperandKind::Literal: return literalText(o.value, o.count == 2);
        case OperandKind::Imm: return hex(o.value);
        default: return regName(o);
    }
}

std::string hwregText(uint16_t v) {
    static constexpr const char* kNames[] = {nullptr,          "HW_REG_MODE",     "HW_REG_STATUS",
                                             "HW_REG_TRAPSTS", "HW_REG_HW_ID",    "HW_REG_GPR_ALLOC",
                                             "HW_REG_LDS_ALLOC", "HW_REG_IB_STS"};
    unsigned id = v & 0x3f, offset = (v >> 6) & 0x1f, size = (v >> 11) + 1;
    std::string s = "hwreg(" + (id && id < 8 ? std::string(kNames[id]) : std::to_string(id));
    if (offset != 0 || size != 32) s += ", " + std::to_string(offset) + ", " + std::to_string(size);
    return s + ")";
}

// Symbolic when the message/operation pair is defined (stray bits are then ignored); numeric triple
// when every set bit belongs to a field; otherwise the raw value.
std::string sendmsgText(uint16_t v) {
    static constexpr const char* kGsOps[] = {"GS_OP_NOP", "GS_OP_CUT", "GS_OP_EMIT", "GS_OP_EMIT_CUT"};
    static constexpr const char* kSysOps[] = {nullptr, "SYSMSG_OP_ECC_ERR_INTERRUPT",
                                              "SYSMSG_OP_REG_RD", "SYSMSG_OP_HOST_TRAP_ACK",
                                              "SYSMSG_OP_TTRACE_PC"};
    unsigned id = v & 0xf, op = (v >> 4) & 7, stream = (v >> 8) & 3;
    if (id == 1 && op == 0 && stream == 0) return "sendmsg(MSG_INTERRUPT)";
    if ((id == 2 || id == 3) && op >= 1 && op <= 3)
        return std::string("sendmsg(") + (id == 2 ? "MSG_GS, " : "MSG_GS_DONE, ") + kGsOps[op] +
               ", " + std::to_string(stream) + ")";
    if (id == 3 && op == 0 && stream == 0) return "sendmsg(MSG_GS_DONE, GS_OP_NOP)";
    if (id == 15 && op >= 1 && op <= 4 && stream == 0)
        return std::string("sendmsg(MSG_SYSMSG, ") + kSysOps[op] + ")";
    if (v & ~0x37fu) return std::to_string(v);
    return "sendmsg(" + std::to_string(id) + ", " + std::to_string(op) + ", " +
           std::to_string(stream) + ")";
}

// Counters at their maximum (no wait) are omitted unless all are.
std::string waitcntText(uint16_t v) {
    struct Field { const char* name; unsigned value, max; };
    Field f[] = {{"vmcnt", v & 0xfu, 15}, {"expcnt", (v >> 4) & 7u, 7}, {"lgkmcnt", (v >> 8) & 0xfu, 15}};
    bool all = true;
    for (auto& x : f) all &= x.value == x.max;
    std::string s;
    for (auto& x : f) {
        if (!all && x.value == x.max) continue;
        if (!s.empty()) s += ' ';
        s += std::string(x.name) + "(" + std::to_string(x.value) + ")";
    }
    return s;
}

std::string soppOperand(char k, uint16_t v) {
    switch (k) {
        case 'e': return v ? std::to_string(v) : "";
        case 'd': return v <= 64 ? std::to_string(v) : hex(v);
        case 'j': return std::to_string(v);
        case 'w': return waitcntText(v);
        case 'm': return sendmsgText(v);
        default: return "";
    }
}

}  // namespace

std::string_view mnemonic(Op op) { return op < Op::Count ? kOps[size_t(op)].name : "<invalid>"; }

bool decode(std::span<const uint32_t> words, Instruction& out) {
    out = {};
    if (words.empty()) return false;
    uint32_t w = words[0];
    Encoding enc;
    unsigned opcode;
    if ((w >> 23) == 0x17d) enc = Encoding::SOP1, opcode = (w >> 8) & 0xff;
    else if ((w >> 23) == 0x17e) enc = Encoding::SOPC, opcode = (w >> 16) & 0x7f;
    else if ((w >> 23) == 0x17f) enc = Encoding::SOPP, opcode = (w >> 16) & 0x7f;
    else if ((w >> 28) == 0xb) enc = Encoding::SOPK, opcode = (w >> 23) & 0x1f;
    else if ((w >> 30) == 0x2) enc = Encoding::SOP2, opcode = (w >> 23) & 0x7f;
    else if ((w >> 27) == 0x18) enc = Encoding::SMRD, opcode = (w >> 22) & 0x1f;
    else return false;
    if (opcode >= kMaxOpcode) return false;
    Op op = kOpcodeMap[size_t(enc)][opcode];
    if (op == Op::Count) return false;

    out.encoding = enc;
    out.op = op;
    out.opcode = uint16_t(opcode);
    out.simm16 = uint16_t(w);
    bool lit = false, ok = false;
    switch (enc) {
        case Encoding::SOP2:
        case Encoding::SOP1:
        case Encoding::SOPC: ok = decodeScalarAlu(w, out, lit); break;
        case Encoding::SOPK: ok = decodeSopk(w, out, lit); break;
        case Encoding::SOPP: ok = decodeSopp(out); break;
        case Encoding::SMRD: ok = decodeSmrd(w, out, lit); break;
    }
    if (!ok) return false;
    out.size = 4;
    if (lit) {
        if (words.size() < 2) return false;
        out.size = 8;
        for (Operand* o : {&out.src[0], &out.src[1]})
            if (o->kind == OperandKind::Literal) o->value = words[1];
    }
    return true;
}

std::string disassemble(const Instruction& in) {
    if (in.op >= Op::Count) return "<invalid>";
    const OpInfo& info = kOps[size_t(in.op)];
    std::string s(info.name);
    std::string args;
    auto add = [&](const std::string& a) {
        if (a.empty()) return;
        args += args.empty() ? " " : ", ";
        args += a;
    };
    auto addOp = [&](const Operand& o) {
        if (o.kind != OperandKind::None) add(operandText(o));
    };
    switch (in.encoding) {
        case Encoding::SOPP: add(soppOperand(info.sig[0], in.simm16)); break;
        case Encoding::SMRD:
            addOp(in.dst);
            addOp(in.src[0]);
            // Offsets print in hex whether immediate or literal.
            if (in.src[1].kind == OperandKind::Imm || in.src[1].kind == OperandKind::Literal)
                add(hex(in.src[1].value));
            else addOp(in.src[1]);
            break;
        case Encoding::SOPK: {
            char k = info.sig[0];
            if (k == 'H' || k == 'I') add(hwregText(in.simm16));
            addOp(in.dst);
            if (k == 'h') add(hwregText(in.simm16));
            else if (k == 'b') add(std::to_string(in.simm16));
            else if (k == 'm' || k == 'k') add(hex(in.simm16));
            addOp(in.src[0]);
            break;
        }
        default:
            addOp(in.dst);
            addOp(in.src[0]);
            addOp(in.src[1]);
            break;
    }
    return s + args;
}

}  // namespace vpgcn
