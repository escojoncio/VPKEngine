// Internal helpers shared by the VPIR printer and parser.
#pragma once

#include <cstdint>
#include <string>

#include "vpir/ir.h"

namespace vpir {

struct ImageOperandSpec {
    ImageOperand bit;
    const char* name;
    unsigned count;  // number of values in the group
};

// Canonical order: matches the bit order, which is also the order of the values in Inst::args.
inline constexpr ImageOperandSpec kImageOperands[] = {
    {kImageLod, "lod", 1},       {kImageBias, "bias", 1},       {kImageGrad, "grad", 2},
    {kImageOffset, "offset", 1}, {kImageCompare, "compare", 1},
};

// Shortest text that parses back to exactly `bits`; non-finite floats print as hex bits.
std::string formatLiteral(Scalar s, uint32_t bits);

}  // namespace vpir
