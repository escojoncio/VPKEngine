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

enum class Encoding : uint8_t { SOP2, SOP1, SOPC, SOPP, SOPK, SMRD, VOP2, VOP1, VOPC, VOP3 };

enum class Op : uint16_t {
#define VPGCN_ENUM(name, opc, sig) name,
    VPGCN_SOP2(VPGCN_ENUM) VPGCN_SOP1(VPGCN_ENUM) VPGCN_SOPC(VPGCN_ENUM) VPGCN_SOPP(VPGCN_ENUM)
    VPGCN_SOPK(VPGCN_ENUM) VPGCN_SMRD(VPGCN_ENUM) VPGCN_VOP2(VPGCN_ENUM) VPGCN_VOP1(VPGCN_ENUM)
    VPGCN_VOPC(VPGCN_ENUM) VPGCN_VOP3(VPGCN_ENUM)
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

struct Instruction {
    Encoding encoding{};
    Op op = Op::Count;
    uint16_t opcode = 0;  // raw opcode field
    uint8_t size = 0;     // bytes: 4 or 8, plus 4 with a literal
    Operand dst;
    Operand sdst;  // vector ops' scalar result: carry-out or compare mask (vcc in the 32-bit forms)
    Operand src[3];  // in printed order; implicit vcc reads of the 32-bit forms are explicit here
    uint16_t simm16 = 0;  // SOPP/SOPK immediate (also the hwreg descriptor of getreg/setreg)
    // VOP3 modifiers: per-source bit masks (bit i = src[i]), clamp, output modifier
    // (1 = *2, 2 = *4, 3 = /2).
    uint8_t abs = 0, neg = 0, omod = 0;
    bool clamp = false;
};

// Decodes the instruction at the start of `words`. Returns false on an invalid or truncated word.
bool decode(std::span<const uint32_t> words, Instruction& out);

// llvm-mc text, without leading whitespace or trailing comment.
std::string disassemble(const Instruction& inst);

}  // namespace vpgcn
