// GCN 2 (Sea Islands / gfx7, the PS4 GPU) instruction decoder and disassembler.
//
// decode() turns a dword stream into a structured Instruction meant for lifting: a single opcode
// enum shared by every encoding (the VOP3 form of a VOP2 op has the same Op), typed operands with
// register ranges, and every modifier field. disassemble() prints it in llvm-mc's syntax for
// -mcpu=bonaire.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "vpgcn/opcodes.h"

namespace vpgcn {

enum class Encoding : uint8_t {
    SOP2, SOP1, SOPC, SOPP, SOPK, SMRD, VOP2, VOP1, VOPC, VOP3,
    VINTRP, DS, MUBUF, MTBUF, MIMG, FLAT, EXP
};

enum class Op : uint16_t {
#define VPGCN_ENUM(name, opc, sig) name,
    VPGCN_SOP2(VPGCN_ENUM) VPGCN_SOP1(VPGCN_ENUM) VPGCN_SOPC(VPGCN_ENUM) VPGCN_SOPP(VPGCN_ENUM)
    VPGCN_SOPK(VPGCN_ENUM) VPGCN_SMRD(VPGCN_ENUM) VPGCN_VOP2(VPGCN_ENUM) VPGCN_VOP1(VPGCN_ENUM)
    VPGCN_VOPC(VPGCN_ENUM) VPGCN_VOP3(VPGCN_ENUM) VPGCN_VINTRP(VPGCN_ENUM) VPGCN_DS(VPGCN_ENUM)
    VPGCN_MUBUF(VPGCN_ENUM) VPGCN_MTBUF(VPGCN_ENUM) VPGCN_MIMG(VPGCN_ENUM) VPGCN_FLAT(VPGCN_ENUM)
    VPGCN_EXP(VPGCN_ENUM)
#undef VPGCN_ENUM
    Count
};

std::string_view mnemonic(Op op);

enum class OperandKind : uint8_t {
    None,
    Sgpr,     // s[index .. index+count-1]
    Ttmp,     // trap temporaries ttmp[index ..]
    Vgpr,     // v[index ..]
    Special,  // vcc, exec, m0, flat_scratch, tba, tma, scc/vccz/execz/lds_direct: see `code`
    Const,    // inline constant (integer or float); `value` holds its 32-bit pattern
    Literal,  // 32-bit literal dword following the instruction; `value` holds it
    Imm,      // an encoding's immediate field (simm16, SMRD offset, ...)
};

// `code` is the raw 9-bit source operand encoding (0..511) for register/constant operands, which
// identifies specials uniquely (106 = vcc_lo, 124 = m0, 253 = scc, 254 = lds_direct, 256+n = vn).
struct Operand {
    OperandKind kind = OperandKind::None;
    uint16_t code = 0;
    uint16_t index = 0;
    uint8_t count = 0;   // dwords
    uint32_t value = 0;  // Const/Literal bit pattern, Imm value
    bool isFloat = false;  // Const: one of the float inline constants (0.5, -4.0, 1/(2pi), ...)
};

// Fields of the memory, export and interpolation encodings (zero elsewhere).
struct MemoryFields {
    uint16_t offset = 0;  // MUBUF/MTBUF 12-bit, DS 16-bit (offset0 | offset1 << 8 for two offsets)
    uint8_t dmask = 0;    // MIMG
    uint8_t dfmt = 0, nfmt = 0;  // MTBUF data and number format
    bool offen = false, idxen = false, addr64 = false;  // MUBUF/MTBUF address mode
    bool glc = false, slc = false, tfe = false, lds = false, gds = false;
    bool unorm = false, da = false, r128 = false, lwe = false;  // MIMG
    // EXP: target (0-7 mrt, 8 mrtz, 9 null, 12-15 pos, 32-63 param), enabled components, flags.
    uint8_t target = 0, enable = 0;
    bool compr = false, done = false, vm = false;
    // VINTRP: attribute and channel (x..w).
    uint8_t attr = 0, chan = 0;
};

struct Instruction {
    Encoding encoding{};
    Op op = Op::Count;
    uint16_t opcode = 0;  // raw opcode field
    uint8_t size = 0;     // bytes: 4 or 8, plus 4 with a literal
    Operand dst;
    Operand sdst;  // vector ops' scalar result: carry-out or compare mask (vcc in the 32-bit forms)
    // In printed order; implicit vcc reads of the 32-bit forms are explicit here. Memory forms:
    // MUBUF/MTBUF vdata, vaddr (None = off), srsrc, soffset; MIMG vdata, vaddr, srsrc, ssamp;
    // DS addr, data0, data1; FLAT vaddr, data; EXP the four sources (None = off); VINTRP the
    // barycentric VGPR or the parameter (Imm 0 p10, 1 p20, 2 p0). Results that come back are in dst
    // (DS/FLAT returns; MUBUF/MTBUF/MIMG loads and returning atomics also mirror vdata there).
    Operand src[4];
    uint16_t simm16 = 0;  // SOPP/SOPK immediate (also the hwreg descriptor of getreg/setreg)
    // VOP3 modifiers: per-source bit masks (bit i = src[i]), clamp, output modifier
    // (1 = *2, 2 = *4, 3 = /2).
    uint8_t abs = 0, neg = 0, omod = 0;
    bool clamp = false;
    MemoryFields mem;
};

// Decodes the instruction at the start of `words`. Returns false on an invalid or truncated word.
// Memory encodings decode whatever the hardware executes: fields an op ignores are not checked.
bool decode(std::span<const uint32_t> words, Instruction& out);

// llvm-mc text, without leading whitespace or trailing comment.
std::string disassemble(const Instruction& inst);

}  // namespace vpgcn
