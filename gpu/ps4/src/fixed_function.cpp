#include "vpps4/fixed_function.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <utility>

#include "vpps4/formats.h"

namespace vpps4 {

namespace {

bool isTriangles(PrimitiveType p) {
    switch (p) {
        case PrimitiveType::TriangleList:
        case PrimitiveType::TriangleFan:
        case PrimitiveType::TriangleStrip:
        case PrimitiveType::RectList:
        case PrimitiveType::QuadList:
        case PrimitiveType::QuadStrip:
        case PrimitiveType::Polygon: return true;
        default: return false;
    }
}

}  // namespace

DepthBias resolveDepthBias(const Raster& r, PrimitiveType prim, uint8_t zFormat, bool hasStencil) {
    DepthBias out;
    const bool frontVisible = !r.cullFront, backVisible = !r.cullBack;
    float scale = 0, offset = 0;
    if (!isTriangles(prim)) {
        // Points and lines: the "para" enable, with the front parameters.
        if (!r.offsetPara) return out;
        scale = r.frontScale, offset = r.frontOffset;
    } else {
        const bool useFront = frontVisible && r.offsetFront;
        const bool useBack = backVisible && r.offsetBack;
        if (!useFront && !useBack) return out;
        if (useFront && useBack) {
            out.approximated = r.frontScale != r.backScale || r.frontOffset != r.backOffset;
            scale = r.frontScale, offset = r.frontOffset;
        } else if (useFront) {
            // Only front faces offset: when back faces are drawn without, they are approximated.
            out.approximated = backVisible && !r.offsetBack;
            scale = r.frontScale, offset = r.frontOffset;
        } else {
            out.approximated = frontVisible && !r.offsetFront;
            scale = r.backScale, offset = r.backOffset;
        }
    }
    if (scale == 0 && offset == 0) return out;
    out.enabled = true;
    // Z_16 beside stencil is a 32-bit float plane here: its unit is 2^-24 instead of 2^-16.
    if (r.dbFmtValid) {
        // The register names the guest's unit. The host plane is a 16-bit unorm one for Z_16 alone, else
        // a 32-bit float one whose unit is taken as 2^-24; a fixed-point guest unit of 2^-N converts by
        // 2^(hostBits - N), a float guest depth already counts in the host's float units.
        const int hostBits = (zFormat == 1 && !hasStencil) ? 16 : 24;
        if (!r.dbIsFloat) offset *= std::ldexp(1.0f, hostBits - int(r.dbBits));
    } else if (zFormat == 1 && hasStencil) {
        offset *= 256.0f;
    }
    out.constant = offset;
    out.slope = scale / 16.0f;
    out.clamp = r.offsetClamp;
    return out;
}

uint8_t permuteWriteMask(uint8_t mask, const uint8_t select[4]) {
    uint8_t out = 0;
    for (unsigned j = 0; j < 4; ++j) {
        const uint8_t c = select[j];
        if (c < 4 ? ((mask >> c) & 1) : (mask & 0xF) != 0) out |= uint8_t(1u << j);
    }
    return out;
}

BlendResolved resolveBlend(const Blend& b, const ColorTarget& c, bool integer) {
    BlendResolved r;
    vp4gpu::BlendState& bs = r.state;
    bs.writeMask = c.writeMask;
    bs.enabled = b.enable && !integer && !c.blendBypass;
    if (!bs.enabled) return r;
    auto factor = [&](uint8_t g) {
        if (c.forceDstAlpha1) {
            if (g == 6) g = 1;       // DST_ALPHA -> ONE
            else if (g == 7) g = 0;  // ONE_MINUS_DST_ALPHA -> ZERO
        }
        if (g >= 15 && g <= 18) r.dualSource = true;
        return blendFactor(g, r.exact);
    };
    bs.srcColor = factor(b.colorSrc);
    bs.dstColor = factor(b.colorDst);
    bs.colorOp = blendOp(b.colorFunc);
    bs.srcAlpha = b.separateAlpha ? factor(b.alphaSrc) : bs.srcColor;
    bs.dstAlpha = b.separateAlpha ? factor(b.alphaDst) : bs.dstColor;
    bs.alphaOp = b.separateAlpha ? blendOp(b.alphaFunc) : bs.colorOp;
    return r;
}

namespace {

struct Converted {
    vp4gpu::StencilOp op = vp4gpu::StencilOp::Keep;
    int replace = -1;  // the value a Replace writes
    bool exact = true;
};

Converted convertOp(uint8_t g, uint8_t ref, uint8_t opVal) {
    using S = vp4gpu::StencilOp;
    Converted c;
    auto replace = [&](int v) { c.op = S::Replace, c.replace = v; };
    // The add/subtract operations step by STENCILOPVAL; the logic ones combine it bitwise.
    auto step = [&](S one) {
        if (opVal == 0) c.op = S::Keep;
        else c.op = one, c.exact = opVal == 1;
    };
    auto logic = [&](S whenOnes, int replaceOnes, S whenZero, int replaceZero) {
        auto pick = [&](S op, int rep) { if (rep >= 0) replace(rep); else c.op = op; };
        if (opVal == 0xFF) pick(whenOnes, replaceOnes);
        else if (opVal == 0) pick(whenZero, replaceZero);
        else c.exact = false;
    };
    switch (g) {
        case 0: break;
        case 1: c.op = S::Zero; break;
        case 2: replace(0xFF); break;
        case 3: replace(ref); break;
        case 4: replace(opVal); break;
        case 5: step(S::IncrementClamp); break;
        case 6: step(S::DecrementClamp); break;
        case 7: c.op = S::Invert; break;
        case 8: step(S::IncrementWrap); break;
        case 9: step(S::DecrementWrap); break;
        case 10: logic(S::Keep, -1, S::Zero, -1); break;          // AND
        case 11: logic(S::Replace, 0xFF, S::Keep, -1); break;     // OR
        case 12: logic(S::Invert, -1, S::Keep, -1); break;        // XOR
        case 13: logic(S::Invert, -1, S::Replace, 0xFF); break;   // NAND
        case 14: logic(S::Zero, -1, S::Invert, -1); break;        // NOR
        case 15: logic(S::Keep, -1, S::Invert, -1); break;        // XNOR
        default: c.exact = false; break;
    }
    return c;
}

const char* const kStencilNames[16] = {"keep", "zero", "ones", "replace_test", "replace_op", "add_clamp", "sub_clamp",
                                       "invert", "add_wrap", "sub_wrap", "and", "or", "xor", "nand", "nor", "xnor"};

}  // namespace

const char* stencilOpName(uint8_t gcn) { return kStencilNames[gcn & 15]; }

StencilResolved resolveStencil(const DepthStencil& ds, bool readOnly) {
    StencilResolved r;
    auto face = [&](const DepthStencil::Face& f, vp4gpu::StencilFace& out, uint8_t& ref) {
        const Converted fail = convertOp(f.fail, f.ref, f.opVal);
        const Converted pass = convertOp(f.zPass, f.ref, f.opVal);
        const Converted zfail = convertOp(f.zFail, f.ref, f.opVal);
        out.compare = compareOp(f.func);
        out.fail = fail.op;
        out.pass = pass.op;
        out.depthFail = zfail.op;
        out.readMask = f.readMask;
        out.writeMask = readOnly ? 0 : f.writeMask;
        r.exact = r.exact && fail.exact && pass.exact && zfail.exact;
        // Metal has one reference per face, for the test and for Replace. A test that never looks at
        // the reference (always / never) lends it to the replace value.
        int want = -1;
        for (const Converted* c : {&fail, &pass, &zfail})
            if (c->replace >= 0) {
                if (want >= 0 && want != c->replace) r.exact = false;
                else want = c->replace;
            }
        ref = f.ref;
        if (want >= 0 && want != f.ref) {
            if (f.func == 0 || f.func == 7) ref = uint8_t(want);
            else r.exact = false;
        }
    };
    face(ds.front, r.front, r.refFront);
    if (ds.backfaceEnable) {
        face(ds.back, r.back, r.refBack);
        r.backDiffers = true;
    } else {
        r.back = r.front;
        r.refBack = r.refFront;
    }
    return r;
}

Scissor combineScissor(const ScissorRegs& s) {
    const int32_t wx = s.windowOffsetDisable ? 0 : s.windowOffsetX;
    const int32_t wy = s.windowOffsetDisable ? 0 : s.windowOffsetY;
    const int32_t gx = s.genericOffsetDisable ? 0 : s.windowOffsetX;
    const int32_t gy = s.genericOffsetDisable ? 0 : s.windowOffsetY;
    int32_t l = std::max({s.screen.left, s.window.left + wx, s.generic.left + gx});
    int32_t t = std::max({s.screen.top, s.window.top + wy, s.generic.top + gy});
    int32_t r = std::min({s.screen.right, s.window.right + wx, s.generic.right + gx});
    int32_t b = std::min({s.screen.bottom, s.window.bottom + wy, s.generic.bottom + gy});
    if (s.viewportScissorEnable) {
        l = std::max(l, s.viewportScissor.left);
        t = std::max(t, s.viewportScissor.top);
        r = std::min(r, s.viewportScissor.right);
        b = std::min(b, s.viewportScissor.bottom);
    }
    l = std::max(l, 0);
    t = std::max(t, 0);
    if (r < l) r = l;  // (corners the wrong way round reach as far as the top left one)
    if (b < t) b = t;
    return {l, t, r, b};
}

ViewportRect hostViewportRect(const Viewport& in, uint32_t width, uint32_t height) {
    Viewport v = in;
    ViewportRect r;
    if (v.xScreen) v.xScale = float(width) / 2, v.xOffset = float(width) / 2;
    if (v.yScreen) v.yScale = -float(height) / 2, v.yOffset = float(height) / 2;
    r.wholeTarget = v.xScreen || v.yScreen;
    if (v.xScale == 0 || v.yScale == 0) return r;
    r.valid = true;
    r.flipped = v.yScale > 0;
    r.x = v.xOffset - std::abs(v.xScale);
    r.y = v.yOffset - std::abs(v.yScale);
    r.width = 2 * std::abs(v.xScale);
    r.height = 2 * std::abs(v.yScale);
    return r;
}

ViewportZ viewportDepthRange(const Viewport& v, bool zeroToW) {
    ViewportZ z;
    if (zeroToW) {
        z.znear = v.zOffset;
        z.zfar = v.zOffset + v.zScale;
    } else {
        z.znear = v.zOffset - v.zScale;
        z.zfar = v.zOffset + v.zScale;
    }
    z.znear = std::clamp(z.znear, 0.0f, 1.0f);
    z.zfar = std::clamp(z.zfar, 0.0f, 1.0f);
    return z;
}

bool depthClamp(const Raster& r, bool& approximated) {
    approximated = r.zclipNearDisable != r.zclipFarDisable;
    return r.clipDisable || r.zclipNearDisable || r.zclipFarDisable;
}

PolygonMode resolvePolygonMode(const Raster& r, PrimitiveType prim) {
    PolygonMode m;
    if (!r.polygonModeEnable || !isTriangles(prim) || prim == PrimitiveType::RectList) return m;
    uint8_t mode = 2;
    if (!r.cullFront) mode = r.polygonModeFront;
    else if (!r.cullBack) mode = r.polygonModeBack;
    if (!r.cullFront && !r.cullBack && r.polygonModeFront != r.polygonModeBack) mode = r.polygonModeFront;
    m.wireframe = mode == 1;
    m.pointMode = mode == 0;
    return m;
}

namespace {
bool isStrip(PrimitiveType p) { return p == PrimitiveType::LineStrip || p == PrimitiveType::TriangleStrip; }
}  // namespace

bool indicesNeedPrep(const DrawState& d) {
    if (!d.indexAddress) return false;
    if (d.indexSwap != 0) return true;
    const uint32_t mask = d.index32 ? 0xFFFFFFFFu : 0xFFFFu;
    // Strips: the host always restarts at the all-ones index. That must be the guest's restart index, and
    // only when its restart is on; any other draw with the value in its data needs a look at the data.
    return isStrip(d.primitive) && (!d.raster.primitiveRestart || (d.raster.restartIndex & mask) != mask);
}

void prepareIndices(const DrawState& d, std::span<const uint8_t> in, std::vector<uint8_t>& out, bool* widened) {
    if (widened) *widened = false;
    out.assign(in.begin(), in.end());
    const size_t n4 = out.size() / 4 * 4;
    switch (d.indexSwap) {
        case 1:
            for (size_t i = 0; i + 1 < out.size(); i += 2) std::swap(out[i], out[i + 1]);
            break;
        case 2:
            for (size_t i = 0; i < n4; i += 4) {
                std::swap(out[i], out[i + 3]);
                std::swap(out[i + 1], out[i + 2]);
            }
            break;
        case 3:
            for (size_t i = 0; i < n4; i += 4) {
                std::swap(out[i], out[i + 2]);
                std::swap(out[i + 1], out[i + 3]);
            }
            break;
        default: break;
    }
    if (!isStrip(d.primitive)) return;
    const bool restartOn = d.raster.primitiveRestart;
    const uint32_t mask = d.index32 ? 0xFFFFFFFFu : 0xFFFFu;
    const uint32_t restart = d.raster.restartIndex & mask;
    const size_t width = d.index32 ? 4 : 2;
    auto at = [&](size_t i) {
        uint32_t v = 0;
        std::memcpy(&v, &out[i * width], width);
        return v;
    };
    const size_t n = out.size() / width;
    // A literal all-ones index that is not a restart for the guest would restart on the host: 16-bit
    // data goes to 32-bit indices, where it is an ordinary vertex.
    bool literalMax = false;
    if (!restartOn || restart != mask)
        for (size_t i = 0; i < n && !literalMax; ++i) literalMax = at(i) == mask && !(restartOn && restart == mask);
    if (literalMax && !d.index32 && widened) {
        std::vector<uint8_t> wide(n * 4);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t v = at(i);
            const uint32_t w = restartOn && v == restart ? 0xFFFFFFFFu : v;
            std::memcpy(&wide[i * 4], &w, 4);
        }
        out.swap(wide);
        *widened = true;
        return;
    }
    if (!restartOn) return;  // (all-ones 32-bit literals cannot be told apart: they stay)
    for (size_t i = 0; i < n; ++i)
        if (at(i) == restart) {
            const uint32_t v = mask;
            std::memcpy(&out[i * width], &v, width);
        }
}

}  // namespace vpps4
