#include "vpgcn/gcn.h"

#include <array>
#include <bit>
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
#define X_VINTRP(n, o, s) {#n, Encoding::VINTRP, o, s},
#define X_DS(n, o, s) {#n, Encoding::DS, o, s},
#define X_MUBUF(n, o, s) {#n, Encoding::MUBUF, o, s},
#define X_MTBUF(n, o, s) {#n, Encoding::MTBUF, o, s},
#define X_MIMG(n, o, s) {#n, Encoding::MIMG, o, s},
#define X_FLAT(n, o, s) {#n, Encoding::FLAT, o, s},
#define X_EXP(n, o, s) {#n, Encoding::EXP, o, s},
    VPGCN_SOP2(X_SOP2) VPGCN_SOP1(X_SOP1) VPGCN_SOPC(X_SOPC) VPGCN_SOPP(X_SOPP)
    VPGCN_SOPK(X_SOPK) VPGCN_SMRD(X_SMRD) VPGCN_VOP2(X_VOP2) VPGCN_VOP1(X_VOP1)
    VPGCN_VOPC(X_VOPC) VPGCN_VOP3(X_VOP3) VPGCN_VINTRP(X_VINTRP) VPGCN_DS(X_DS)
    VPGCN_MUBUF(X_MUBUF) VPGCN_MTBUF(X_MTBUF) VPGCN_MIMG(X_MIMG) VPGCN_FLAT(X_FLAT) VPGCN_EXP(X_EXP)
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
#undef X_VINTRP
#undef X_DS
#undef X_MUBUF
#undef X_MTBUF
#undef X_MIMG
#undef X_FLAT
#undef X_EXP
};
static_assert(std::size(kOps) == size_t(Op::Count));

constexpr size_t kEncodings = size_t(Encoding::EXP) + 1;
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
        if (c == 'c') ok = code < 128 && regOperand(code, 2, o);  // an SGPR pair, VCC or EXEC
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

// ---- Memory, export and interpolation -----------------------------------------------------------

// A buffer/image descriptor: 4 or 8 SGPRs (or trap temporaries) named by a 5-bit field in units of 4.
bool resourceOperand(unsigned field, unsigned count, Operand& o) {
    return regOperand(uint16_t(field * 4), count, o);
}

// MUBUF/MTBUF soffset: an SGPR, m0 or an inline constant (no literal, no condition sources).
bool soffsetOperand(uint16_t code, Operand& o) {
    bool unused = false;
    if (code == kVccz || code == kExecz || code == kScc || code == kLdsDirect || code == kLiteral)
        return false;
    return srcOperand(code, 1, Src::NoLiteral, o, unused);
}

unsigned sigDigit(char c) { return c == '-' ? 0 : unsigned(c - '0'); }

bool isCmpswap(Op op) { return kOps[size_t(op)].name.find("cmpswap") != std::string_view::npos; }

// The registers a load or returning atomic writes, from its vdata range: compare-swap returns one
// value, half of what it sends (the status dword of tfe/lwe stays at the end).
Operand returned(const Instruction& in, const Operand& vdata) {
    if (!isCmpswap(in.op)) return vdata;
    unsigned extra = in.mem.tfe || in.mem.lwe;
    return vgpr(vdata.index, (vdata.count - extra) / 2 + extra);
}

bool decodeBuffer(uint32_t w0, uint32_t w1, Instruction& in) {
    std::string_view sig = kOps[size_t(in.op)].sig;
    MemoryFields& m = in.mem;
    m.offset = w0 & 0xfff;
    m.offen = w0 >> 12 & 1;
    m.idxen = w0 >> 13 & 1;
    m.glc = w0 >> 14 & 1;
    m.addr64 = w0 >> 15 & 1;
    if (in.encoding == Encoding::MUBUF) m.lds = w0 >> 16 & 1;
    else m.dfmt = w0 >> 19 & 0xf, m.nfmt = w0 >> 23 & 7;
    m.slc = w1 >> 22 & 1;
    m.tfe = w1 >> 23 & 1;
    if (sig[0] == 'N') return true;
    unsigned data = sigDigit(sig[1]) + (m.tfe && sig[0] == 'L');
    if (!vgprOperand((w1 >> 8) & 0xff, data, in.src[0])) return false;
    // addr64 takes a 64-bit address in place of index/offset: offen and idxen must be clear.
    if (m.addr64 && (m.offen || m.idxen)) return false;
    unsigned addr = m.addr64 || (m.offen && m.idxen) ? 2 : m.offen || m.idxen ? 1 : 0;
    if (addr && !vgprOperand(w1 & 0xff, addr, in.src[1])) return false;
    if (!resourceOperand((w1 >> 16) & 0x1f, 4, in.src[2])) return false;
    if (!soffsetOperand(w1 >> 24, in.src[3])) return false;
    // Loads into LDS (lds) write no VGPR; a returning compare-swap gives back half its data.
    if ((sig[0] == 'L' && !m.lds) || (sig[0] == 'A' && m.glc)) in.dst = returned(in, in.src[0]);
    return true;
}

bool decodeImage(uint32_t w0, uint32_t w1, Instruction& in) {
    std::string_view sig = kOps[size_t(in.op)].sig;
    MemoryFields& m = in.mem;
    m.dmask = (w0 >> 8) & 0xf;
    m.unorm = w0 >> 12 & 1;
    m.glc = w0 >> 13 & 1;
    m.da = w0 >> 14 & 1;
    m.r128 = w0 >> 15 & 1;
    m.tfe = w0 >> 16 & 1;
    m.lwe = w0 >> 17 & 1;
    m.slc = w0 >> 25 & 1;
    unsigned data = sig[0] == 'G' ? 4u : std::max(1u, unsigned(std::popcount(unsigned(m.dmask))));
    // tfe or lwe: one more dword comes back (the status). Stores carry it too (the assembler
    // counts it; the hardware ignores it).
    data += m.tfe || m.lwe;
    if (!vgprOperand((w1 >> 8) & 0xff, data, in.src[0])) return false;
    if (!vgprOperand(w1 & 0xff, sigDigit(sig[1]), in.src[1])) return false;
    if (!resourceOperand((w1 >> 16) & 0x1f, m.r128 ? 4 : 8, in.src[2])) return false;
    if (sig[2] == 's' && !resourceOperand((w1 >> 21) & 0x1f, 4, in.src[3])) return false;
    if (sig[0] != 'W' && (sig[0] != 'A' || m.glc)) in.dst = returned(in, in.src[0]);
    return true;
}

bool decodeDs(uint32_t w0, uint32_t w1, Instruction& in) {
    std::string_view sig = kOps[size_t(in.op)].sig;
    MemoryFields& m = in.mem;
    m.offset = w0 & 0xffff;
    m.gds = w0 >> 17 & 1;
    if (sig[0] != '-' && !vgprOperand(w1 >> 24, sigDigit(sig[0]), in.dst)) return false;
    unsigned slot = 0;
    if (sig[1] != '-' && !vgprOperand(w1 & 0xff, 1, in.src[slot++])) return false;
    if (sig[2] != '-' && !vgprOperand((w1 >> 8) & 0xff, sigDigit(sig[2]), in.src[slot++])) return false;
    if (sig[3] != '-' && !vgprOperand((w1 >> 16) & 0xff, sigDigit(sig[3]), in.src[slot++])) return false;
    return true;
}

bool decodeFlat(uint32_t w0, uint32_t w1, Instruction& in) {
    std::string_view sig = kOps[size_t(in.op)].sig;
    MemoryFields& m = in.mem;
    m.glc = w0 >> 16 & 1;
    m.slc = w0 >> 17 & 1;
    m.tfe = w1 >> 23 & 1;
    unsigned data = sigDigit(sig[1]);
    if (!vgprOperand(w1 & 0xff, 2, in.src[0])) return false;
    if (sig[0] != 'L' && !vgprOperand((w1 >> 8) & 0xff, data, in.src[1])) return false;
    if (sig[0] == 'L' || (sig[0] == 'A' && m.glc)) {
        // A returning compare-swap gives back one value: half its data.
        unsigned ret = isCmpswap(in.op) ? data / 2 : data;
        if (!vgprOperand(w1 >> 24, ret + (m.tfe && sig[0] == 'L'), in.dst)) return false;
    }
    return true;
}

bool decodeExp(uint32_t w0, uint32_t w1, Instruction& in) {
    MemoryFields& m = in.mem;
    m.enable = w0 & 0xf;
    m.target = (w0 >> 4) & 0x3f;
    m.compr = w0 >> 10 & 1;
    m.done = w0 >> 11 & 1;
    m.vm = w0 >> 12 & 1;
    if (m.target == 10 || m.target == 11 || (m.target >= 16 && m.target < 32)) return false;
    // Disabled sources are None. Compressed: src[0] = vsrc0 (packed x,y), src[1] = vsrc1 (z,w).
    for (unsigned i = 0; i < 4; ++i) {
        bool on = m.compr ? i < 2 && (m.enable >> (2 * i) & 3) : m.enable >> i & 1;
        if (on) in.src[i] = vgpr((w1 >> (8 * i)) & 0xff, 1);
    }
    return true;
}

bool decodeInterp(uint32_t w, Instruction& in) {
    MemoryFields& m = in.mem;
    m.chan = (w >> 8) & 3;
    m.attr = (w >> 10) & 0x3f;
    in.dst = vgpr((w >> 18) & 0xff, 1);
    if (kOps[size_t(in.op)].sig[0] == 'v') {
        in.src[0] = vgpr(w & 0xff, 1);
        return true;
    }
    if ((w & 0xff) > 2) return false;
    in.src[0].kind = OperandKind::Imm;
    in.src[0].value = w & 0xff;
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

std::string expTarget(unsigned t) {
    if (t < 8) return "mrt" + std::to_string(t);
    if (t == 8) return "mrtz";
    if (t == 9) return "null";
    if (t < 16) return "pos" + std::to_string(t - 12);
    return "param" + std::to_string(t - 32);
}

std::string bufferFormat(unsigned dfmt, unsigned nfmt) {
    static constexpr const char* kData[] = {
        "INVALID", "8", "16", "8_8", "32", "16_16", "10_11_11", "11_11_10", "10_10_10_2",
        "2_10_10_10", "8_8_8_8", "32_32", "16_16_16_16", "32_32_32", "32_32_32_32", "RESERVED_15"};
    static constexpr const char* kNum[] = {"UNORM", "SNORM", "USCALED", "SSCALED",
                                           "UINT",  "SINT",  "RESERVED_6", "FLOAT"};
    // Only the parts that differ from the default (8, UNORM).
    if (dfmt == 1 && nfmt == 0) return "";
    std::string s = " format:[";
    if (dfmt != 1) s += std::string("BUF_DATA_FORMAT_") + kData[dfmt];
    if (nfmt != 0) s += std::string(dfmt != 1 ? "," : "") + "BUF_NUM_FORMAT_" + kNum[nfmt];
    return s + "]";
}

// ds_swizzle_b32 pattern: quad permute (bit 15 and nothing above the 8-bit pattern), else the
// bit-mask mode, where each lane-id bit is and/or/xor-ed: printed per bit (MSB first) as p (kept),
// i (inverted), 0 or 1 (forced).
std::string swizzleText(uint16_t v) {
    if ((v & 0xff00) == 0x8000) {
        std::string s = "swizzle(QUAD_PERM";
        for (unsigned i = 0; i < 4; ++i) s += "," + std::to_string((v >> (2 * i)) & 3);
        return s + ")";
    }
    if (v & 0x8000) return std::to_string(v);
    unsigned andMask = v & 0x1f, orMask = (v >> 5) & 0x1f, xorMask = (v >> 10) & 0x1f;
    auto pow2 = [](unsigned x) { return x && !(x & (x - 1)); };
    // Named forms: swap groups of n lanes, reverse groups of n, broadcast lane l of groups of n.
    if (andMask == 0x1f && orMask == 0 && pow2(xorMask))
        return "swizzle(SWAP," + std::to_string(xorMask) + ")";
    if (andMask == 0x1f && orMask == 0 && xorMask && pow2(xorMask + 1))
        return "swizzle(REVERSE," + std::to_string(xorMask + 1) + ")";
    if (xorMask == 0 && pow2(32 - andMask) && orMask < 32 - andMask)
        return "swizzle(BROADCAST," + std::to_string(32 - andMask) + "," + std::to_string(orMask) + ")";
    std::string mask(5, '0');
    for (unsigned b = 0; b < 5; ++b) {
        unsigned bit = 1u << (4 - b);
        bool keep = andMask & bit, set = orMask & bit, flip = xorMask & bit;
        mask[b] = keep && !set ? (flip ? 'i' : 'p') : ((keep ? 1 : set) ^ flip) ? '1' : '0';
    }
    return "swizzle(BITMASK_PERM,\"" + mask + "\")";
}

std::string memoryText(const Instruction& in, const OpInfo& info) {
    const MemoryFields& m = in.mem;
    std::string s(info.name), args;
    auto add = [&](const std::string& a) {
        args += args.empty() ? " " : ", ";
        args += a;
    };
    auto flag = [&](bool on, const char* name) {
        if (on) args += std::string(" ") + name;
    };
    std::string_view sig = info.sig;
    switch (in.encoding) {
        case Encoding::MUBUF:
        case Encoding::MTBUF:
            if (sig[0] == 'N') break;
            add(operandText(in.src[0]));
            add(in.src[1].kind == OperandKind::None ? "off" : operandText(in.src[1]));
            add(operandText(in.src[2]));
            add(operandText(in.src[3]));
            if (in.encoding == Encoding::MTBUF) args += bufferFormat(m.dfmt, m.nfmt);
            flag(m.idxen, "idxen");
            flag(m.offen, "offen");
            flag(m.addr64, "addr64");
            if (m.offset) args += " offset:" + std::to_string(m.offset);
            flag(m.glc, "glc");
            flag(m.slc, "slc");
            flag(m.lds, "lds");
            flag(m.tfe, "tfe");
            break;
        case Encoding::MIMG:
            for (unsigned i = 0; i < 4; ++i)
                if (in.src[i].kind != OperandKind::None) add(operandText(in.src[i]));
            if (m.dmask) args += " dmask:" + hex(m.dmask);
            flag(m.unorm, "unorm");
            flag(m.glc, "glc");
            flag(m.slc, "slc");
            flag(m.r128, "r128");
            flag(m.tfe, "tfe");
            flag(m.lwe, "lwe");
            flag(m.da, "da");
            break;
        case Encoding::DS:
            if (in.dst.kind != OperandKind::None) add(operandText(in.dst));
            for (const Operand& o : in.src)
                if (o.kind != OperandKind::None) add(operandText(o));
            if (sig[4] == '2') {
                if (m.offset & 0xff) args += " offset0:" + std::to_string(m.offset & 0xff);
                if (m.offset >> 8) args += " offset1:" + std::to_string(m.offset >> 8);
            } else if (sig[4] == 'z') {
                if (m.offset) args += " offset:" + swizzleText(m.offset);
            } else if (m.offset) {
                args += " offset:" + std::to_string(m.offset);
            }
            flag(m.gds, "gds");
            break;
        case Encoding::FLAT:
            if (in.dst.kind != OperandKind::None) add(operandText(in.dst));
            add(operandText(in.src[0]));
            if (in.src[1].kind != OperandKind::None) add(operandText(in.src[1]));
            flag(m.glc, "glc");
            flag(m.slc, "slc");
            flag(m.tfe, "tfe");
            break;
        case Encoding::EXP: {
            args = " " + expTarget(m.target);
            for (unsigned i = 0; i < 4; ++i) {
                // Compressed exports carry two packed sources (vsrc0, vsrc1), each shown twice.
                const Operand& o = in.src[m.compr ? i / 2 : i];
                args += i ? ", " : " ";
                args += o.kind != OperandKind::None ? operandText(o) : "off";
            }
            flag(m.done, "done");
            flag(m.compr, "compr");
            flag(m.vm, "vm");
            break;
        }
        case Encoding::VINTRP: {
            static constexpr const char* kParams[] = {"p10", "p20", "p0"};
            add(operandText(in.dst));
            add(in.src[0].kind == OperandKind::Imm ? kParams[in.src[0].value] : operandText(in.src[0]));
            add("attr" + std::to_string(m.attr) + "." + "xyzw"[m.chan]);
            break;
        }
        default: break;
    }
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
    else if ((w >> 26) == 0x32) enc = Encoding::VINTRP, opcode = (w >> 16) & 3;
    else if ((w >> 26) == 0x36) enc = Encoding::DS, opcode = (w >> 18) & 0xff;
    else if ((w >> 26) == 0x38) enc = Encoding::MUBUF, opcode = (w >> 18) & 0x7f;
    else if ((w >> 26) == 0x3a) enc = Encoding::MTBUF, opcode = (w >> 16) & 7;
    else if ((w >> 26) == 0x3c) enc = Encoding::MIMG, opcode = (w >> 18) & 0x7f;
    else if ((w >> 26) == 0x37) enc = Encoding::FLAT, opcode = (w >> 18) & 0x7f;
    else if ((w >> 26) == 0x3e) enc = Encoding::EXP, opcode = 0;
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
        case Encoding::VINTRP: ok = decodeInterp(w, out); break;
        case Encoding::DS:
        case Encoding::MUBUF:
        case Encoding::MTBUF:
        case Encoding::MIMG:
        case Encoding::FLAT:
        case Encoding::EXP: {
            if (words.size() < 2) return false;
            uint32_t w1 = words[1];
            ok = enc == Encoding::DS      ? decodeDs(w, w1, out)
                 : enc == Encoding::MIMG  ? decodeImage(w, w1, out)
                 : enc == Encoding::FLAT  ? decodeFlat(w, w1, out)
                 : enc == Encoding::EXP   ? decodeExp(w, w1, out)
                                          : decodeBuffer(w, w1, out);
            break;
        }
    }
    if (!ok) return false;
    out.size = enc == Encoding::VOP3 || enc >= Encoding::DS ? 8 : 4;
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
    if (in.encoding >= Encoding::VINTRP) return memoryText(in, info);
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
