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
#define X_VOP2(n, o, s) {#n, Encoding::VOP2, o, s},
#define X_VOP1(n, o, s) {#n, Encoding::VOP1, o, s},
#define X_VOPC(n, o, s) {#n, Encoding::VOPC, o, s},
#define X_VOP3(n, o, s) {#n, Encoding::VOP3, o, s},
    VPGCN_SOP2(X_SOP2) VPGCN_SOP1(X_SOP1) VPGCN_SOPC(X_SOPC) VPGCN_SOPP(X_SOPP)
    VPGCN_SOPK(X_SOPK) VPGCN_SMRD(X_SMRD) VPGCN_VOP2(X_VOP2) VPGCN_VOP1(X_VOP1)
    VPGCN_VOPC(X_VOPC) VPGCN_VOP3(X_VOP3)
#undef X_SOP2
#undef X_SOP1
#undef X_SOPC
#undef X_SOPP
#undef X_SOPK
#undef X_SMRD
#undef X_VOP2
#undef X_VOP1
#undef X_VOPC
#undef X_VOP3
};
static_assert(std::size(kOps) == size_t(Op::Count));

constexpr size_t kEncodings = 10;
constexpr size_t kMaxOpcode = 512;

// Signature parts of vector ops: operands before '/', VOP3 modifiers after it (see opcodes.h).
constexpr std::string_view sigOperands(std::string_view sig) { return sig.substr(0, sig.find('/')); }
constexpr bool hasVop3Form(std::string_view sig) { return sig.find('/') != sig.npos; }
constexpr std::string_view sigMods(std::string_view sig) {
    return hasVop3Form(sig) ? sig.substr(sig.find('/') + 1) : std::string_view{};
}

// Where the VOP3 opcode space places the ops of each 32-bit vector encoding.
constexpr unsigned vop3Base(Encoding e) {
    return e == Encoding::VOP2 ? 256 : e == Encoding::VOP1 ? 384 : 0;
}

// Per-encoding opcode -> Op map (Op::Count = unassigned). The VOP3 row also holds the VOP3 form of
// every VOP1/VOP2/VOPC op that has one.
constexpr auto kOpcodeMap = [] {
    std::array<std::array<Op, kMaxOpcode>, kEncodings> m{};
    for (auto& row : m) row.fill(Op::Count);
    for (size_t i = 0; i < std::size(kOps); ++i) {
        const OpInfo& o = kOps[i];
        m[size_t(o.enc)][o.opcode] = Op(i);
        if ((o.enc == Encoding::VOP2 || o.enc == Encoding::VOP1 || o.enc == Encoding::VOPC) &&
            hasVop3Form(o.sig))
            m[size_t(Encoding::VOP3)][vop3Base(o.enc) + o.opcode] = Op(i);
    }
    return m;
}();

// ---- Operand codes (the 8/9-bit scalar source encoding) -----------------------------------------

constexpr uint16_t kFlatScratch = 104, kVcc = 106, kTba = 108, kTma = 110, kTtmp0 = 112,
                   kM0 = 124, kExec = 126, kIntZero = 128, kIntMax = 192, kIntNegMax = 208,
                   kFloatFirst = 240, kFloatLast = 247, kVccz = 251, kExecz = 252, kScc = 253,
                   kLdsDirect = 254, kLiteral = 255, kVgpr0 = 256;
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

// ---- Vector ALU ---------------------------------------------------------------------------------

Operand vgpr(unsigned index, unsigned count) {
    Operand o;
    o.kind = OperandKind::Vgpr;
    o.code = uint16_t(kVgpr0 + index);
    o.index = uint16_t(index);
    o.count = uint8_t(count);
    return o;
}

Operand vcc64() { return {OperandKind::Special, kVcc, 0, 2, 0, false}; }

bool vgprOperand(unsigned index, unsigned count, Operand& o) {
    if (index + count > 256) return false;
    o = vgpr(index, count);
    return true;
}

// A 9-bit vector source: VGPRs, everything a scalar source takes, and lds_direct (32-bit, src0 only).
bool vsrcOperand(uint16_t code, unsigned count, bool literalOk, bool ldsOk, Operand& o, bool& lit) {
    if (code >= kVgpr0) return vgprOperand(code - kVgpr0, count, o);
    if (code == kLdsDirect) {
        if (!ldsOk || count != 1) return false;
        o = {OperandKind::Special, code, 0, 1, 0, false};
        return true;
    }
    return srcOperand(code, count, literalOk ? Src::Any : Src::NoLiteral, o, lit);
}

// Fields of a vector instruction, already split out of its encoding.
struct VectorFields {
    bool vop3 = false, vop3b = false;
    uint16_t vdst = 0, sdst = 0, src[3] = {}, abs = 0, neg = 0, omod = 0;
    bool clamp = false;
    uint8_t srcBits[3] = {9, 9, 9};  // VOP2/VOPC vsrc1 is an 8-bit VGPR index
};

bool decodeVector(const VectorFields& f, Instruction& in, bool& lit) {
    std::string_view sig = kOps[size_t(in.op)].sig, ops = sigOperands(sig), mods = sigMods(sig);
    // Scalar values read through the constant bus: at most one distinct (code, size) per instruction.
    uint32_t busKey = 0;
    auto bus = [&](uint32_t key) {
        if (busKey && busKey != key) return false;
        busKey = key;
        return true;
    };
    unsigned slot = 0, field = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
        char c = ops[i];
        if (i == 0) {
            if (c == 's') {
                if (f.vdst >= 128 || !regOperand(f.vdst, 1, in.dst)) return false;
            } else if (c != '-' && !vgprOperand(f.vdst, unsigned(c - '0'), in.dst)) {
                return false;
            }
            continue;
        }
        if (c == 'C') {
            if (!f.vop3) in.sdst = vcc64();
            else if (!regOperand(f.vop3b ? f.sdst : f.vdst, 2, in.sdst)) return false;
            continue;
        }
        if (c == '+' || c == 'M') {
            if (!bus(c == '+' ? kVcc << 8 | 2 : kM0 << 8 | 1)) return false;
            continue;
        }
        Operand& o = in.src[slot];
        if (c == 'k') {
            o = {OperandKind::Literal, kLiteral, 0, 1, 0, false};
            lit = true;
            if (!bus(kLiteral << 8 | 1)) return false;  // shares the dword with a src0 literal
            ++slot;
            continue;
        }
        if (c == 'c' && !f.vop3) {
            o = vcc64();
            if (!bus((kVcc << 8) | 2)) return false;
            ++slot;
            continue;
        }
        uint16_t code = f.src[field];
        bool narrow = f.srcBits[field] == 8;
        ++field;
        bool ok;
        if (c == 'c') ok = code < 128 && code != kExec && regOperand(code, 2, o);  // XEXEC class
        else if (c == 's') ok = srcOperand(code, 1, Src::NoLiteral, o, lit);
        else if (c == 'S') ok = code < kVgpr0 && vsrcOperand(code, 1, true, true, o, lit);
        else if (c == 'V') ok = code >= kVgpr0 && vgprOperand(code - kVgpr0, 1, o);
        else if (c == 'v')
            ok = code >= kVgpr0 ? vgprOperand(code - kVgpr0, 1, o)
                                : code == kLdsDirect && vsrcOperand(code, 1, false, true, o, lit);
        else if (narrow) ok = vgprOperand(code, unsigned(c - '0'), o);
        else ok = vsrcOperand(code, unsigned(c - '0'), !f.vop3, field == 1, o, lit);
        if (!ok) return false;
        bool scalar = o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp ||
                      o.kind == OperandKind::Literal ||
                      (o.kind == OperandKind::Special && o.code != kLdsDirect);
        if (scalar && !bus(uint32_t(o.code) << 8 | o.count)) return false;
        ++slot;
    }
    if (!f.vop3) return true;
    // Modifiers: only where the op takes them (the assembler cannot encode the others).
    for (unsigned j = 0; j < 3; ++j) {
        char m = j < mods.size() ? mods[j] : '-';
        bool hasSrc = j < slot && m != 'c' && m != 'o';
        if ((f.abs >> j & 1) && !(hasSrc && m == 'a')) return false;
        if ((f.neg >> j & 1) && !(hasSrc && (m == 'a' || m == 'n'))) return false;
    }
    std::string_view tail = mods.size() > slot ? mods.substr(slot) : std::string_view{};
    if (f.clamp && tail.find('c') == tail.npos) return false;
    if (f.omod && tail.find('o') == tail.npos) return false;
    in.abs = uint8_t(f.abs);
    in.neg = uint8_t(f.neg);
    in.omod = uint8_t(f.omod);
    in.clamp = f.clamp;
    return true;
}

// ---- Printing -----------------------------------------------------------------------------------

std::string hex(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%x", v);
    return b;
}

std::string regName(const Operand& o) {
    static constexpr const char* kSpecial[] = {"flat_scratch", "vcc", "tba", "tma"};
    std::string prefix = o.kind == OperandKind::Ttmp ? "ttmp" : o.kind == OperandKind::Vgpr ? "v" : "s";
    if (o.kind == OperandKind::Sgpr || o.kind == OperandKind::Ttmp || o.kind == OperandKind::Vgpr) {
        if (o.count == 1) return prefix + std::to_string(o.index);
        return prefix + "[" + std::to_string(o.index) + ":" + std::to_string(o.index + o.count - 1) +
               "]";
    }
    switch (o.code) {
        case kM0: return "m0";
        case kVccz: return "src_vccz";
        case kExecz: return "src_execz";
        case kScc: return "src_scc";
        case kLdsDirect: return "src_lds_direct";
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

// Source with VOP3 input modifiers. A negated numeric constant needs neg() so it is not read back
// as a different inline constant.
std::string modifiedSource(const Operand& o, bool abs, bool neg) {
    std::string t = operandText(o);
    if (abs) t = "|" + t + "|";
    if (!neg) return t;
    bool numeric = o.kind == OperandKind::Const || o.kind == OperandKind::Literal;
    return numeric && !abs ? "neg(" + t + ")" : "-" + t;
}

std::string vectorText(const Instruction& in, const OpInfo& info) {
    std::string_view ops = sigOperands(info.sig);
    std::string s(info.name);
    // Ops with both encodings name the one in use; ones without operands never do.
    if (hasVop3Form(info.sig) && info.enc != Encoding::VOP3 && ops != "-")
        s += in.encoding == Encoding::VOP3 ? "_e64" : "_e32";
    std::string args;
    auto add = [&](const std::string& a) {
        args += args.empty() ? " " : ", ";
        args += a;
    };
    unsigned slot = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
        char c = ops[i];
        if (c == '-' || c == '+' || c == 'M') continue;
        if (i == 0) add(operandText(in.dst));
        else if (c == 'C') add(operandText(in.sdst));
        else {
            add(modifiedSource(in.src[slot], in.abs >> slot & 1, in.neg >> slot & 1));
            ++slot;
        }
    }
    if (in.clamp) args += " clamp";
    static constexpr const char* kOmod[] = {"", " mul:2", " mul:4", " div:2"};
    args += kOmod[in.omod & 3];
    return s + args;
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
    else if ((w >> 25) == 0x3f) enc = Encoding::VOP1, opcode = (w >> 9) & 0xff;
    else if ((w >> 25) == 0x3e) enc = Encoding::VOPC, opcode = (w >> 17) & 0xff;
    else if ((w >> 31) == 0) enc = Encoding::VOP2, opcode = (w >> 25) & 0x3f;
    else if ((w >> 26) == 0x34) enc = Encoding::VOP3, opcode = (w >> 17) & 0x1ff;
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
        case Encoding::VOP2:
        case Encoding::VOP1:
        case Encoding::VOPC: {
            VectorFields f;
            f.vdst = (w >> 17) & 0xff;
            f.src[0] = w & 0x1ff;
            f.src[1] = (w >> 9) & 0xff;
            f.srcBits[1] = 8;
            ok = decodeVector(f, out, lit);
            break;
        }
        case Encoding::VOP3: {
            if (words.size() < 2) return false;
            uint32_t w1 = words[1];
            VectorFields f;
            f.vop3 = true;
            // VOP3b: ops with a scalar carry-out besides a vector result keep sdst where abs/clamp go.
            Encoding native = kOps[size_t(op)].enc;
            f.vop3b = native != Encoding::VOPC &&
                      sigOperands(kOps[size_t(op)].sig).find('C') != std::string_view::npos;
            f.vdst = w & 0xff;
            f.sdst = (w >> 8) & 0x7f;
            if (!f.vop3b) f.abs = (w >> 8) & 7, f.clamp = (w >> 11) & 1;
            f.src[0] = w1 & 0x1ff;
            f.src[1] = (w1 >> 9) & 0x1ff;
            f.src[2] = (w1 >> 18) & 0x1ff;
            f.omod = (w1 >> 27) & 3;
            f.neg = w1 >> 29;
            ok = decodeVector(f, out, lit);
            break;
        }
    }
    if (!ok) return false;
    out.size = enc == Encoding::VOP3 ? 8 : 4;
    if (lit) {
        if (words.size() < 2) return false;
        out.size = 8;
        for (Operand* o : {&out.src[0], &out.src[1], &out.src[2]})
            if (o->kind == OperandKind::Literal) o->value = words[1];
    }
    return true;
}

std::string disassemble(const Instruction& in) {
    if (in.op >= Op::Count) return "<invalid>";
    const OpInfo& info = kOps[size_t(in.op)];
    if (in.encoding >= Encoding::VOP2) return vectorText(in, info);
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
