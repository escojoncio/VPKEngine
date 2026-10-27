// VPIR emitters of the Liverpool format conversions (the same arithmetic as the CPU reference in
// formats.cpp, op for op, so shader and reference agree bit for bit).
#include <bit>

#include "vpgcn/formats.h"

namespace vpgcn {

namespace V = vpir;
using V::ValueId;

namespace {
constexpr V::Type U = V::kU32;
constexpr V::Type F = V::kF32;
constexpr V::Type I = V::kI32;
constexpr V::Type B = V::kBool;
uint32_t maskOf(unsigned w) { return w >= 32 ? 0xffffffffu : (1u << w) - 1; }
float maxUnsigned(unsigned w) { return float((uint64_t(1) << w) - 1); }
float maxSigned(unsigned w) { return float((uint64_t(1) << (w - 1)) - 1); }
}  // namespace

ValueId FormatEmitter::halfToF(ValueId bits) {
    ValueId pair = bitsOf(bits, V::vec(V::kF16, 2));
    return b_.unary(V::Op::Convert, b_.extract(pair, 0, V::kF16), F);
}

ValueId FormatEmitter::halfBits(ValueId fv) {
    ValueId h = b_.unary(V::Op::Convert, fv, V::kF16);
    ValueId pair = b_.construct(std::array<ValueId, 2>{h, b_.constant(V::kF16, {0})}, V::vec(V::kF16, 2));
    return bitsOf(pair, U);
}

ValueId FormatEmitter::srgbToLinear(ValueId c) {
    ValueId a = b_.binary(V::Op::Pow, b_.binary(V::Op::Div, b_.binary(V::Op::Add, c, f(0.055f), F), f(1.055f), F), f(2.4f), F);
    ValueId lo = b_.binary(V::Op::Div, c, f(12.92f), F);
    return b_.ternary(V::Op::Select, b_.binary(V::Op::Le, c, f(0.04045f), B), lo, a, F);
}

ValueId FormatEmitter::linearToSrgb(ValueId c) {
    ValueId p = b_.binary(V::Op::Pow, c, f(1.0f / 2.4f), F);
    ValueId a = b_.binary(V::Op::Sub, b_.binary(V::Op::Mul, f(1.055f), p, F), f(0.055f), F);
    ValueId lo = b_.binary(V::Op::Mul, c, f(12.92f), F);
    return b_.ternary(V::Op::Select, b_.binary(V::Op::Le, c, f(0.0031308f), B), lo, a, F);
}

ValueId FormatEmitter::smallFloatToF(ValueId fieldBits, unsigned mant) {
    ValueId m = b_.binary(V::Op::And, fieldBits, u((1u << mant) - 1), U);
    ValueId e = b_.binary(V::Op::Shr, fieldBits, u(mant), U);
    ValueId mShifted = b_.binary(V::Op::Shl, m, u(23 - mant), U);
    ValueId normal = b_.binary(V::Op::Or, b_.binary(V::Op::Shl, b_.binary(V::Op::Add, e, u(112), U), u(23), U), mShifted, U);
    ValueId special = b_.binary(V::Op::Or, u(0x7f800000u), mShifted, U);
    ValueId denorm = bitsOf(b_.binary(V::Op::Mul, b_.unary(V::Op::Convert, m, F), f(std::bit_cast<float>(uint32_t(113 - mant) << 23)), F), U);
    ValueId r = b_.ternary(V::Op::Select, b_.binary(V::Op::Eq, e, u(31), B), special, normal, U);
    return b_.ternary(V::Op::Select, b_.binary(V::Op::Eq, e, u(0), B), denorm, r, U);
}

ValueId FormatEmitter::fToSmallFloat(ValueId fv, unsigned mant) {
    const uint32_t maxPat = (30u << mant) | ((1u << mant) - 1);
    ValueId bits = bitsOf(fv, U);
    ValueId e = b_.binary(V::Op::And, b_.binary(V::Op::Shr, bits, u(23), U), u(0xff), U);
    ValueId m = b_.binary(V::Op::And, bits, u(0x7fffff), U);
    ValueId E = b_.binary(V::Op::Sub, bitsOf(e, I), b_.constI32(112), I);
    const unsigned drop = 23 - mant;
    ValueId mq = b_.binary(V::Op::Shr, m, u(drop), U);
    ValueId round = b_.binary(V::Op::And, b_.binary(V::Op::Shr, m, u(drop - 1), U), u(1), U);
    ValueId sticky = b_.ternary(V::Op::Select, b_.binary(V::Op::Ne, b_.binary(V::Op::And, m, u((1u << (drop - 1)) - 1), U), u(0), B), u(1), u(0), U);
    ValueId inc = b_.binary(V::Op::And, round, b_.binary(V::Op::Or, sticky, b_.binary(V::Op::And, mq, u(1), U), U), U);
    ValueId normal = b_.binary(V::Op::Add, b_.binary(V::Op::Add, b_.binary(V::Op::Shl, bitsOf(E, U), u(mant), U), mq, U), inc, U);
    normal = b_.binary(V::Op::Min, normal, u(maxPat), U);
    ValueId tiny = b_.unary(V::Op::Convert,
                            b_.unary(V::Op::Round, b_.binary(V::Op::Mul, fv, f(std::bit_cast<float>(uint32_t(127 + 14 + mant) << 23)), F), F), U);
    ValueId r = b_.ternary(V::Op::Select, b_.binary(V::Op::Ge, E, b_.constI32(1), B), normal, tiny, U);
    r = b_.ternary(V::Op::Select, b_.unary(V::Op::IsInf, fv, B), u(31u << mant), r, U);
    r = b_.ternary(V::Op::Select, b_.binary(V::Op::Gt, fv, f(0.0f), B), r, u(0), U);
    r = b_.ternary(V::Op::Select, b_.unary(V::Op::IsNan, fv, B), u((31u << mant) | (1u << (mant - 1))), r, U);
    return r;
}

// `shift` / `width` of a field in one word; sign: sign-extended (I32), else zero-extended (U32).
ValueId FormatEmitter::field(ValueId word, unsigned shift, unsigned width, bool sign) {
    if (width >= 32) return sign ? bitsOf(word, I) : word;
    if (!sign) return shift == 0 ? b_.binary(V::Op::And, word, u(maskOf(width)), U)
                                 : b_.ternary(V::Op::BitfieldExtract, word, u(shift), u(width), U);
    ValueId up = shift + width == 32 ? word : b_.binary(V::Op::Shl, word, u(32 - shift - width), U);
    return b_.binary(V::Op::Shr, bitsOf(up, I), b_.constI32(int32_t(32 - width)), I);
}

bool FormatEmitter::decode(unsigned dfmt, unsigned nfmt, std::span<const ValueId> words, ValueId sub, ValueId out[4]) {
    if (!numFmtValid(nfmt)) return false;
    const FormatLayout& l = layoutOf(dfmt);
    const bool integer = isIntegerNumFmt(nfmt);
    auto word = [&](unsigned i) { return i < words.size() ? words[i] : u(0); };
    auto toU = [&](ValueId fv) { return bitsOf(fv, U); };
    out[0] = out[1] = out[2] = u(0);
    out[3] = integer ? u(1) : toU(f(1.0f));
    switch (l.kind) {
        case FormatLayout::Plain:
            for (unsigned c = 0; c < l.comps; ++c) {
                const unsigned w = l.bits[c], at = l.shift[c];
                const FieldKind k = fieldKind(w, nfmt);
                ValueId src = word(at / 32);
                const unsigned off = at % 32;
                ValueId v = u(0);
                switch (k) {
                    case FieldKind::Unorm: {
                        ValueId raw = field(src, off, w, false);
                        v = toU(b_.binary(V::Op::Div, b_.unary(V::Op::Convert, raw, F), f(maxUnsigned(w)), F));
                        break;
                    }
                    case FieldKind::Snorm: {
                        ValueId s = field(src, off, w, true);
                        ValueId q = b_.binary(V::Op::Div, b_.unary(V::Op::Convert, s, F), f(maxSigned(w)), F);
                        v = toU(b_.binary(V::Op::Max, q, f(-1.0f), F));
                        break;
                    }
                    case FieldKind::Uscaled: v = toU(b_.unary(V::Op::Convert, field(src, off, w, false), F)); break;
                    case FieldKind::Sscaled: v = toU(b_.unary(V::Op::Convert, field(src, off, w, true), F)); break;
                    case FieldKind::Uint: v = field(src, off, w, false); break;
                    case FieldKind::Sint: v = bitsOf(field(src, off, w, true), U); break;
                    case FieldKind::SnormNz: {
                        ValueId s = b_.unary(V::Op::Convert, field(src, off, w, true), F);
                        ValueId t = b_.binary(V::Op::Add, b_.binary(V::Op::Mul, s, f(2.0f), F), f(1.0f), F);
                        v = toU(b_.binary(V::Op::Div, t, f(maxUnsigned(w)), F));
                        break;
                    }
                    case FieldKind::Ubnorm: {
                        ValueId raw = field(src, off, w, false);
                        ValueId q = b_.binary(V::Op::Div, b_.unary(V::Op::Convert, raw, F), f(maxUnsigned(w)), F);
                        v = toU(b_.binary(V::Op::Sub, b_.binary(V::Op::Mul, q, f(2.0f), F), f(1.0f), F));
                        break;
                    }
                    case FieldKind::Float: {
                        ValueId raw = field(src, off, w, false);
                        v = w == 32 ? raw : w == 16 ? toU(halfToF(raw)) : smallFloatToF(raw, w - 5);
                        break;
                    }
                }
                if (nfmt == kSrgb && c < 3) v = toU(srgbToLinear(bitsOf(v, F)));
                out[c] = v;
            }
            return true;
        case FormatLayout::SharedExp: {
            ValueId w = word(0);
            ValueId e = b_.binary(V::Op::Shr, w, u(27), U);
            ValueId scale = bitsOf(b_.binary(V::Op::Shl, b_.binary(V::Op::Add, e, u(103), U), u(23), U), F);
            for (unsigned c = 0; c < 3; ++c) {
                ValueId m = b_.ternary(V::Op::BitfieldExtract, w, u(9 * c), u(9), U);
                out[c] = toU(b_.binary(V::Op::Mul, b_.unary(V::Op::Convert, m, F), scale, F));
            }
            out[3] = toU(f(1.0f));
            return true;
        }
        case FormatLayout::Fmask:
            out[0] = word(0);
            out[1] = l.bytes > 4 ? word(1) : u(0);
            out[3] = u(1);
            return true;
        case FormatLayout::Subsampled: {
            if (nfmt != kUnorm && nfmt != kSrgb) return false;
            ValueId w = word(0);
            auto byte = [&](unsigned k) { return b_.ternary(V::Op::BitfieldExtract, w, u(8 * k), u(8), U); };
            ValueId odd = b_.binary(V::Op::Ne, b_.binary(V::Op::And, sub == V::kNoValue ? u(0) : sub, u(1), U), u(0), B);
            auto pick = [&](ValueId a, ValueId c2) { return b_.ternary(V::Op::Select, odd, c2, a, U); };
            ValueId r, g, bl;
            if (dfmt == kFmtGB_GR) r = byte(0), g = pick(byte(1), byte(3)), bl = byte(2);
            else r = byte(1), g = pick(byte(0), byte(2)), bl = byte(3);
            ValueId c3[3] = {r, g, bl};
            for (unsigned c = 0; c < 3; ++c) {
                ValueId v = b_.binary(V::Op::Div, b_.unary(V::Op::Convert, c3[c], F), f(255.0f), F);
                if (nfmt == kSrgb) v = srgbToLinear(v);
                out[c] = toU(v);
            }
            out[3] = toU(f(1.0f));
            return true;
        }
        case FormatLayout::OneBit: {
            if (nfmt != kUnorm && nfmt != kUint) return false;
            ValueId t = b_.binary(V::Op::And, sub == V::kNoValue ? u(0) : sub, u(7), U);
            if (l.reversed) t = b_.binary(V::Op::Sub, u(7), t, U);
            ValueId bit = b_.binary(V::Op::And, b_.binary(V::Op::Shr, word(0), t, U), u(1), U);
            out[0] = integer ? bit : toU(b_.unary(V::Op::Convert, bit, F));
            return true;
        }
        default: return false;
    }
}

ValueId FormatEmitter::encodeField(ValueId raw, unsigned w, FieldKind k, bool saturate) {
    const uint32_t mask = maskOf(w);
    ValueId fv = bitsOf(raw, F);
    auto clean = [&](ValueId x) { return b_.ternary(V::Op::Select, b_.unary(V::Op::IsNan, x, B), f(0.0f), x, F); };
    auto masked = [&](ValueId x) { return w >= 32 ? x : b_.binary(V::Op::And, x, u(mask), U); };
    auto toIntBits = [&](ValueId x) { return bitsOf(b_.unary(V::Op::Convert, x, I), U); };
    switch (k) {
        case FieldKind::Unorm: {
            ValueId s = b_.unary(V::Op::Saturate, clean(fv), F);
            if (w == 32) {
                ValueId r = b_.unary(V::Op::Convert, b_.unary(V::Op::Round, b_.binary(V::Op::Mul, s, f(4294967040.0f), F), F), U);
                return b_.ternary(V::Op::Select, b_.binary(V::Op::Ge, s, f(1.0f), B), u(0xffffffffu), r, U);
            }
            return b_.unary(V::Op::Convert, b_.unary(V::Op::Round, b_.binary(V::Op::Mul, s, f(maxUnsigned(w)), F), F), U);
        }
        case FieldKind::Snorm: {
            ValueId c = b_.ternary(V::Op::Clamp, clean(fv), f(-1.0f), f(1.0f), F);
            const float m = w == 32 ? 2147483520.0f : maxSigned(w);
            return masked(toIntBits(b_.unary(V::Op::Round, b_.binary(V::Op::Mul, c, f(m), F), F)));
        }
        case FieldKind::Uscaled: {
            const float m = w == 32 ? 4294967040.0f : maxUnsigned(w);
            return b_.unary(V::Op::Convert, b_.ternary(V::Op::Clamp, clean(fv), f(0.0f), f(m), F), U);
        }
        case FieldKind::Sscaled: {
            const float hi = w == 32 ? 2147483520.0f : maxSigned(w);
            const float lo = -maxSigned(w) - (w == 32 ? 128.0f : 1.0f);
            return masked(toIntBits(b_.ternary(V::Op::Clamp, clean(fv), f(lo), f(hi), F)));
        }
        case FieldKind::Uint:
            if (!saturate) return masked(raw);
            return b_.binary(V::Op::Min, raw, u(mask), U);
        case FieldKind::Sint:
            if (!saturate || w >= 32) return masked(raw);
            return masked(bitsOf(b_.binary(V::Op::Min, b_.binary(V::Op::Max, bitsOf(raw, I), b_.constI32(-(1 << (w - 1))), I),
                                           b_.constI32((1 << (w - 1)) - 1), I), U));
        case FieldKind::SnormNz: {
            ValueId s = b_.ternary(V::Op::Clamp, clean(fv), f(-1.0f), f(1.0f), F);
            ValueId t = b_.binary(V::Op::Div, b_.binary(V::Op::Sub, b_.binary(V::Op::Mul, s, f(maxUnsigned(w)), F), f(1.0f), F), f(2.0f), F);
            t = b_.unary(V::Op::Round, t, F);
            t = b_.binary(V::Op::Min, b_.binary(V::Op::Max, t, f(-float(uint64_t(1) << (w - 1))), F), f(maxSigned(w)), F);
            return masked(toIntBits(t));
        }
        case FieldKind::Ubnorm: {
            ValueId q = b_.unary(V::Op::Saturate, b_.binary(V::Op::Div, b_.binary(V::Op::Add, fv, f(1.0f), F), f(2.0f), F), F);
            q = b_.ternary(V::Op::Select, b_.unary(V::Op::IsNan, fv, B), f(0.0f), q, F);
            return b_.unary(V::Op::Convert, b_.unary(V::Op::Round, b_.binary(V::Op::Mul, q, f(maxUnsigned(w)), F), F), U);
        }
        case FieldKind::Float:
            if (w == 32) return raw;
            if (w == 16) return halfBits(fv);
            return fToSmallFloat(fv, w - 5);
    }
    return u(0);
}

bool FormatEmitter::encode(unsigned dfmt, unsigned nfmt, std::span<const ValueId> comps, unsigned have, bool saturate,
                           Encoded& out) {
    if (!numFmtValid(nfmt)) return false;
    const FormatLayout& l = layoutOf(dfmt);
    if (l.kind != FormatLayout::Plain) return false;
    out = Encoded{};
    out.count = l.words();
    for (unsigned i = 0; i < out.count; ++i) out.word[i] = u(0);
    for (unsigned c = 0; c < l.comps && c < comps.size(); ++c) {
        if (!(have >> c & 1)) continue;
        const unsigned w = l.bits[c], at = l.shift[c];
        const FieldKind k = fieldKind(w, nfmt);
        ValueId v = comps[c];
        if (nfmt == kSrgb && c < 3) {
            ValueId fv = bitsOf(v, F);
            fv = b_.ternary(V::Op::Select, b_.unary(V::Op::IsNan, fv, B), f(0.0f), fv, F);
            v = bitsOf(linearToSrgb(b_.unary(V::Op::Saturate, fv, F)), U);
        }
        ValueId e = encodeField(v, w, k, saturate);
        ValueId placed = at % 32 ? b_.binary(V::Op::Shl, e, u(at % 32), U) : e;
        ValueId& dst = out.word[at / 32];
        dst = b_.binary(V::Op::Or, dst, placed, U);
        out.mask[at / 32] |= maskOf(w) << (at % 32);
    }
    return true;
}

}  // namespace vpgcn
