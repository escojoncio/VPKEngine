// Liverpool (gfx7) data / number formats: element layouts, a CPU reference of every conversion and
// the VPIR emitters that implement the same conversions in shaders (typed buffer access, texture
// results, colour exports). The CPU reference is the specification the emitters are tested against.
//
// Component naming follows the hardware: a format name lists its fields from the most significant
// bit, the last one named being component x at bit 0 (10_11_11 = z10 y11 x11, 2_10_10_10 =
// w2 z10 y10 x10, 5_5_5_1 = w5 z5 y5 x1).
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "vpir/builder.h"

namespace vpgcn {

// Number formats (IMG_NUM_FORMAT / BUF_NUM_FORMAT; the CB uses 6 for sRGB, callers map it to kSrgb).
enum NumFmt : uint8_t {
    kUnorm = 0, kSnorm = 1, kUscaled = 2, kSscaled = 3, kUint = 4, kSint = 5, kSnormNz = 6, kFloat = 7,
    kSrgb = 9, kUbnorm = 10,
};

// IMG_DATA_FORMAT numbering (BUF_DATA_FORMAT is the same up to 15).
enum DataFmt : uint8_t {
    kFmtInvalid = 0, kFmt8 = 1, kFmt16 = 2, kFmt8_8 = 3, kFmt32 = 4, kFmt16_16 = 5, kFmt10_11_11 = 6,
    kFmt11_11_10 = 7, kFmt10_10_10_2 = 8, kFmt2_10_10_10 = 9, kFmt8_8_8_8 = 10, kFmt32_32 = 11,
    kFmt16_16_16_16 = 12, kFmt32_32_32 = 13, kFmt32_32_32_32 = 14, kFmt5_6_5 = 16, kFmt1_5_5_5 = 17,
    kFmt5_5_5_1 = 18, kFmt4_4_4_4 = 19, kFmt8_24 = 20, kFmt24_8 = 21, kFmtX24_8_32 = 22, kFmtGB_GR = 32,
    kFmtBG_RG = 33, kFmt5_9_9_9 = 34, kFmtBC1 = 35, kFmtBC7 = 41, kFmtFmask8_1 = 47, kFmtFmask64_8 = 56,
    kFmt4_4 = 57, kFmt6_5_5 = 58, kFmt1 = 59, kFmt1Reversed = 60, kFmt32As8 = 61, kFmt32As8_8 = 62,
    kFmt32As32x4 = 63,
};

struct FormatLayout {
    // Plain: fields of a texel, one element per texel. SharedExp: 5_9_9_9. Block: BC1-7 (compressed).
    // Fmask: an opaque FMASK word. Subsampled: GB_GR / BG_RG (two texels share an element).
    // OneBit: 1 / 1_REVERSED (eight texels per byte). None: no element (invalid, reserved, 32_AS_*).
    enum Kind : uint8_t { None, Plain, SharedExp, Block, Fmask, Subsampled, OneBit } kind = None;
    uint8_t comps = 0;           // fields (Plain)
    uint8_t bits[4] = {};        // width of each field
    uint8_t shift[4] = {};       // bit offset of each field within the element (up to 127)
    uint8_t bytes = 0;           // bytes per element (BC: per block)
    uint8_t texelsPerElement = 1;  // 1, 2 (Subsampled) or 8 (OneBit)
    bool reversed = false;       // OneBit: texel 0 is the most significant bit
    uint8_t words() const { return uint8_t((bytes + 3) / 4); }
};
const FormatLayout& layoutOf(unsigned dfmt);  // dfmt 0..63

// How one field is interpreted under a number format: floats exist for widths 32, 16, 11 and 10
// only; any other width asked as float is an unsigned integer.
enum class FieldKind : uint8_t { Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, SnormNz, Float, Ubnorm };
FieldKind fieldKind(unsigned width, unsigned nfmt);
bool isIntegerNumFmt(unsigned nfmt);          // uint / sint: the components are integers
bool numFmtValid(unsigned nfmt);              // 0..7, 9 and 10 (the others have no defined conversion)

// ---- CPU reference ------------------------------------------------------------------------------

float srgbToLinear(float c);
float linearToSrgb(float c);
float smallFloatToF32(uint32_t field, unsigned mantissaBits);   // unsigned 5-bit-exponent float (11: 6, 10: 5)
uint32_t f32ToSmallFloat(float v, unsigned mantissaBits);       // nearest-even, negative and NaN-less clamp, see formats.cpp
float halfToF32(uint32_t h);
uint32_t f32ToHalf(float v);

// Decodes texel `sub` (0 unless Subsampled / OneBit) of the element `words` into 4 component bit
// patterns (f32 bits for float-class numbers, u32 / i32 for integers); components the format lacks
// are 0, w 1 (1.0f / 1). False when the combination has no conversion.
bool decodeTexel(unsigned dfmt, unsigned nfmt, const uint32_t words[4], unsigned sub, uint32_t out[4]);
// Encodes components (bit patterns as above) into an element. `have` is a mask of the components
// supplied; `words` / `masks` get the bits written (masks: the bits owned by the supplied fields).
// saturate: integer fields clamp to their range (colour exports) instead of keeping the low bits.
bool encodeTexel(unsigned dfmt, unsigned nfmt, const uint32_t in[4], unsigned have, bool saturate, uint32_t words[4],
                 uint32_t masks[4]);

// ---- VPIR emitters ------------------------------------------------------------------------------

class FormatEmitter {
public:
    using ValueId = vpir::ValueId;
    explicit FormatEmitter(vpir::FunctionBuilder& b) : b_(b) {}

    // Same contract as decodeTexel; `words` are u32 values (words().size() of the layout, missing
    // ones read as 0), `sub` a u32 value or kNoValue. Returns 4 bit patterns (u32 typed).
    // Invalid combinations return false and emit nothing.
    bool decode(unsigned dfmt, unsigned nfmt, std::span<const ValueId> words, ValueId sub, ValueId out[4]);

    struct Encoded {
        ValueId word[4] = {};
        uint32_t mask[4] = {};
        unsigned count = 0;
    };
    // Same contract as encodeTexel; comps are u32-typed bit patterns.
    bool encode(unsigned dfmt, unsigned nfmt, std::span<const ValueId> comps, unsigned have, bool saturate, Encoded& out);

    ValueId srgbToLinear(ValueId f32value);
    ValueId linearToSrgb(ValueId f32value);
    ValueId halfToF(ValueId bits);       // low 16 bits of a u32 -> f32
    ValueId halfBits(ValueId f32value);  // f32 -> f16 bits (low half of a u32)
    ValueId smallFloatToF(ValueId field, unsigned mantissaBits);  // returns the f32 bits (u32)
    ValueId fToSmallFloat(ValueId f32value, unsigned mantissaBits);

private:
    ValueId u(uint32_t v) { return b_.constU32(v); }
    ValueId f(float v) { return b_.constF32(v); }
    ValueId bitsOf(ValueId x, vpir::Type t) { return b_.unary(vpir::Op::Bitcast, x, t); }
    ValueId field(ValueId word, unsigned shift, unsigned width, bool sign);
    ValueId encodeField(ValueId v, unsigned width, FieldKind k, bool saturate);
    vpir::FunctionBuilder& b_;
};

}  // namespace vpgcn
