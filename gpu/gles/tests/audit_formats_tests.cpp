// Audit tests for gpu/gles/src/formats.cpp and xfb.cpp (no GPU): every (internal format, format, type) the
// transfer validation accepts must have an upload conversion, and every colour-renderable format must be readable
// in the (format, type) pairs ES 3.x requires. Usage: vpgles_audit_formats_tests
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "formats.h"
#include "gl_enums.h"
#include "xfb.h"

using namespace vpgles;
namespace gl = vpgles::gl;

namespace {
int failures = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

const GLenum kInternals[] = {
    gl::R8, gl::RG8, gl::RGB8, gl::RGBA8, gl::SRGB8, gl::SRGB8_ALPHA8, gl::RGB565, gl::RGBA4, gl::RGB5_A1, gl::RGB10_A2,
    gl::RGB10_A2UI, gl::R8_SNORM, gl::RG8_SNORM, gl::RGB8_SNORM, gl::RGBA8_SNORM, gl::R16F, gl::RG16F, gl::RGB16F,
    gl::RGBA16F, gl::R32F, gl::RG32F, gl::RGB32F, gl::RGBA32F, gl::R11F_G11F_B10F, gl::RGB9_E5, gl::R8I, gl::R8UI,
    gl::R16I, gl::R16UI, gl::R32I, gl::R32UI, gl::RG8I, gl::RG8UI, gl::RG16I, gl::RG16UI, gl::RG32I, gl::RG32UI,
    gl::RGBA8I, gl::RGBA8UI, gl::RGBA16I, gl::RGBA16UI, gl::RGBA32I, gl::RGBA32UI, gl::RGB8I, gl::RGB8UI, gl::RGB16I,
    gl::RGB16UI, gl::RGB32I, gl::RGB32UI, gl::DEPTH_COMPONENT16, gl::DEPTH_COMPONENT24, gl::DEPTH_COMPONENT32F,
    gl::DEPTH24_STENCIL8, gl::DEPTH32F_STENCIL8, gl::STENCIL_INDEX8,
};
const GLenum kFormats[] = {gl::RED, gl::RG, gl::RGB, gl::RGBA, gl::RED_INTEGER, gl::RG_INTEGER, gl::RGB_INTEGER,
                           gl::RGBA_INTEGER, gl::DEPTH_COMPONENT, gl::DEPTH_STENCIL};
const GLenum kTypes[] = {gl::UNSIGNED_BYTE, gl::BYTE, gl::UNSIGNED_SHORT, gl::SHORT, gl::UNSIGNED_INT, gl::INT,
                         gl::HALF_FLOAT, gl::FLOAT, gl::UNSIGNED_SHORT_5_6_5, gl::UNSIGNED_SHORT_4_4_4_4,
                         gl::UNSIGNED_SHORT_5_5_5_1, gl::UNSIGNED_INT_2_10_10_10_REV, gl::UNSIGNED_INT_10F_11F_11F_REV,
                         gl::UNSIGNED_INT_5_9_9_9_REV, gl::UNSIGNED_INT_24_8, gl::FLOAT_32_UNSIGNED_INT_24_8_REV};

// A: ES 3.0 table 3.2 pairs the validation accepts must convert (a false from convertUpload is GL_INVALID_OPERATION to the app).
void testUploadCoverage() {
    int missing = 0;
    for (GLenum internal : kInternals)
        for (GLenum format : kFormats)
            for (GLenum type : kTypes) {
                const FormatInfo* f = formatFor(internal, format, type);
                if (!f || f->internal != internal) continue;
                if (transferError(*f, format, type, false) != 0) continue;
                const uint32_t px = clientPixelBytes(format, type);
                CHECK(px != 0);
                std::vector<uint8_t> src(size_t(px) * 4, 0x11), out;
                if (!convertUpload(*f, format, type, src.data(), 4, out)) {
                    std::fprintf(stderr, "  no upload conversion: internal=0x%x format=0x%x type=0x%x\n", internal, format, type);
                    ++missing;
                    continue;
                }
                CHECK(out.size() == size_t(4) * f->texelBytes);
            }
    CHECK(missing == 0);
}

uint16_t half(float f) {
    // exact for the values used below
    uint32_t b;
    std::memcpy(&b, &f, 4);
    if (!(b & 0x7FFFFFFF)) return uint16_t(b >> 16);
    const int e = int((b >> 23) & 0xFF) - 127 + 15;
    return uint16_t((b >> 16 & 0x8000) | uint32_t(e) << 10 | (b >> 13 & 0x3FF));
}

// B: values of the conversions that exist but are easy to get wrong.
void testUploadValues() {
    {  // RGBA/FLOAT into RGBA16F: half floats, not the raw 32-bit pattern
        const float in[4] = {1.0f, 0.5f, 2.0f, 0.0f};
        std::vector<uint8_t> out;
        const FormatInfo* f = formatFor(gl::RGBA16F);
        CHECK(convertUpload(*f, gl::RGBA, gl::FLOAT, reinterpret_cast<const uint8_t*>(in), 1, out) && out.size() == 8);
        if (out.size() == 8) {
            uint16_t h[4];
            std::memcpy(h, out.data(), 8);
            CHECK(h[0] == half(1.0f) && h[1] == half(0.5f) && h[2] == half(2.0f) && h[3] == 0);
        }
    }
    {  // RGB/UNSIGNED_BYTE into RGB5_A1 (valid in ES 3.0 table 3.2)? checked in the coverage test; here RGBA bytes into RGB5_A1
        const uint8_t in[4] = {255, 0, 255, 255};
        std::vector<uint8_t> out;
        const FormatInfo* f = formatFor(gl::RGB5_A1);
        if (convertUpload(*f, gl::RGBA, gl::UNSIGNED_BYTE, in, 1, out) && out.size() == 2) {
            uint16_t q;
            std::memcpy(&q, out.data(), 2);
            CHECK(q == uint16_t(1u << 15 | 31u << 10 | 31u));  // A1R5G5B5: A=1 R=31 G=0 B=31
        } else {
            CHECK(false);
        }
    }
}

// B1: 8-bit data into 4/5/6-bit targets rounds to nearest (ES 3.0 2.1.6.1 / 3.8.2: unorm conversion), it does not truncate.
void testPackedRounding() {
    {  // RGBA4: 8 -> 0 (0.47), 9 -> 1 (0.53: a truncation gives 0), 255 -> 15, 136 -> 8, 247 -> 15 (14.53), 120 -> 7, 128 -> 8
        const uint8_t in[8] = {8, 9, 255, 136, 247, 0, 120, 128};
        std::vector<uint8_t> out;
        const FormatInfo* f = formatFor(gl::RGBA4);
        CHECK(convertUpload(*f, gl::RGBA, gl::UNSIGNED_BYTE, in, 2, out) && out.size() == 4);
        if (out.size() == 4) {
            uint16_t q[2];
            std::memcpy(q, out.data(), 4);
            CHECK(q[0] == uint16_t(0 << 12 | 1 << 8 | 15 << 4 | 8));
            CHECK(q[1] == uint16_t(15 << 12 | 0 << 8 | 7 << 4 | 8));
        }
    }
    {  // RGB565: R 4 -> 0, 5 -> 1 (0.61); G 3 -> 1 (0.74); B 5 -> 1, 130 -> 16 (15.8); a truncation gives 0 for R 5 and G 3
        const uint8_t in[6] = {4, 3, 5, 5, 2, 130};
        std::vector<uint8_t> out;
        const FormatInfo* f = formatFor(gl::RGB565);
        CHECK(convertUpload(*f, gl::RGB, gl::UNSIGNED_BYTE, in, 2, out) && out.size() == 4);
        if (out.size() == 4) {
            uint16_t q[2];
            std::memcpy(q, out.data(), 4);
            CHECK(q[0] == uint16_t(0 << 11 | 1 << 5 | 1));
            CHECK(q[1] == uint16_t(1 << 11 | 0 << 5 | 16));
        }
    }
    {  // RGB565 readback to bytes rounds too: 3/31 -> 24.68 = 25, 1/63 -> 4.05 = 4, 62/63 -> 250.95 = 251
        const uint16_t p = uint16_t(3 << 11 | 62 << 5 | 1);
        std::vector<uint8_t> out;
        const FormatInfo* f = formatFor(gl::RGB565);
        uint8_t px[4] = {0, 0, 0, 0};
        if (convertReadback(*f, gl::RGBA, gl::UNSIGNED_BYTE, reinterpret_cast<const uint8_t*>(&p), 1, px)) {
            CHECK(px[0] == 25 && px[1] == 251 && px[2] == 8 && px[3] == 255);
        } else {
            CHECK(false);
        }
    }
}

// B2: packed float formats from FLOAT / HALF_FLOAT data.
void testPackedFloat() {
    const float in[3] = {1.0f, 2.0f, 0.5f};
    std::vector<uint8_t> out;
    CHECK(convertUpload(*formatFor(gl::R11F_G11F_B10F), gl::RGB, gl::FLOAT, reinterpret_cast<const uint8_t*>(in), 1, out) && out.size() == 4);
    if (out.size() == 4) {
        uint32_t q;
        std::memcpy(&q, out.data(), 4);
        CHECK((q & 0x7FF) == 0x3C0 && ((q >> 11) & 0x7FF) == 0x400 && (q >> 22) == 0x1C0);  // 1.0, 2.0, 0.5 (10f: exponent 14)
    }
    const uint16_t hin[3] = {0x3C00, 0x4000, 0x3800};
    out.clear();
    CHECK(convertUpload(*formatFor(gl::RGB9_E5), gl::RGB, gl::HALF_FLOAT, reinterpret_cast<const uint8_t*>(hin), 1, out) && out.size() == 4);
    if (out.size() == 4) {
        uint32_t q;
        std::memcpy(&q, out.data(), 4);
        const double scale = std::ldexp(1.0, int(q >> 27) - 15 - 9);
        CHECK((q & 511) * scale == 1.0 && ((q >> 9) & 511) * scale == 2.0 && ((q >> 18) & 511) * scale == 0.5);
    }
    {  // 2-10-10-10 into RGB5_A1: full white, alpha 3 -> 1
        const uint32_t px = 1023u | 1023u << 10 | 1023u << 20 | 3u << 30;
        out.clear();
        CHECK(convertUpload(*formatFor(gl::RGB5_A1), gl::RGBA, gl::UNSIGNED_INT_2_10_10_10_REV, reinterpret_cast<const uint8_t*>(&px), 1, out) && out.size() == 2);
        uint16_t q = 0;
        if (out.size() == 2) std::memcpy(&q, out.data(), 2);
        CHECK(q == 0xFFFF);
    }
    {  // half rounds to nearest even: 1 + 2^-11 is the tie between 1.0 and 1 + 2^-10
        const float v[2] = {1.0f + 1.0f / 2048.0f, 1.0f + 3.0f / 2048.0f};
        out.clear();
        CHECK(convertUpload(*formatFor(gl::RG16F), gl::RG, gl::FLOAT, reinterpret_cast<const uint8_t*>(v), 1, out) && out.size() == 4);
        uint16_t h[2] = {};
        if (out.size() == 4) std::memcpy(h, out.data(), 4);
        CHECK(h[0] == 0x3C00 && h[1] == 0x3C02);
    }
}

// B3: luminance / alpha textures with FLOAT data (GL_OES_texture_float, advertised by the layer).
void testLuminanceFloat() {
    const GLenum internals[] = {gl::LUMINANCE, gl::ALPHA, gl::LUMINANCE_ALPHA};
    for (GLenum internal : internals) {
        const FormatInfo* f = formatFor(internal, internal, gl::FLOAT);
        CHECK(f != nullptr);
        if (!f) continue;
        CHECK(transferError(*f, internal, gl::FLOAT) == 0);
        const float in[2] = {0.25f, 0.75f};
        std::vector<uint8_t> out;
        CHECK(convertUpload(*f, internal, gl::FLOAT, reinterpret_cast<const uint8_t*>(in), 1, out));
        CHECK(out.size() == f->texelBytes && std::memcmp(out.data(), in, out.size()) == 0);
        CHECK(f->vk == (internal == gl::LUMINANCE_ALPHA ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R32_SFLOAT));
        // the 8-bit and half float rows are still found
        CHECK(formatFor(internal, internal, gl::UNSIGNED_BYTE)->texelBytes == (internal == gl::LUMINANCE_ALPHA ? 2 : 1));
        CHECK(formatFor(internal, internal, gl::HALF_FLOAT_OES)->texelBytes == (internal == gl::LUMINANCE_ALPHA ? 4 : 2));
    }
}

// C: glReadPixels pairs ES 3.x requires for each colour-renderable format.
void testReadback() {
    int bad = 0;
    for (GLenum internal : kInternals) {
        const FormatInfo* f = formatFor(internal);
        if (!f || !f->renderable || f->depth || f->stencil) continue;
        std::vector<uint8_t> in(32, 0x22), out(64);
        GLenum fmt, type;
        if (f->integer) {
            fmt = gl::RGBA_INTEGER;
            type = f->componentType == gl::INT ? gl::INT : gl::UNSIGNED_INT;
        } else if (f->componentType == gl::FLOAT) {
            fmt = gl::RGBA, type = gl::FLOAT;
        } else {
            fmt = gl::RGBA, type = gl::UNSIGNED_BYTE;
        }
        if (!convertReadback(*f, fmt, type, in.data(), 1, out.data())) {
            std::fprintf(stderr, "  no readback: internal=0x%x -> format=0x%x type=0x%x\n", internal, fmt, type);
            ++bad;
        }
        // RGBA/UNSIGNED_BYTE is accepted for every normalised format
        if (!f->integer && f->componentType != gl::FLOAT && !convertReadback(*f, gl::RGBA, gl::UNSIGNED_BYTE, in.data(), 1, out.data())) ++bad;
    }
    CHECK(bad == 0);
    {  // RGB10_A2UI as RGBA_INTEGER/UNSIGNED_INT: the integer values, with the 2-bit alpha intact
        const FormatInfo* f = formatFor(gl::RGB10_A2UI);
        uint32_t px = 5u | 7u << 10 | 1023u << 20 | 3u << 30, o[4] = {};
        CHECK(convertReadback(*f, gl::RGBA_INTEGER, gl::UNSIGNED_INT, reinterpret_cast<uint8_t*>(&px), 1, reinterpret_cast<uint8_t*>(o)));
        CHECK(o[0] == 5 && o[1] == 7 && o[2] == 1023 && o[3] == 3);
    }
    {  // RGBA8I: negative values survive RGBA_INTEGER/INT
        const FormatInfo* f = formatFor(gl::RGBA8I);
        int8_t px[4] = {-1, -128, 127, 5};
        int32_t o[4] = {};
        CHECK(convertReadback(*f, gl::RGBA_INTEGER, gl::INT, reinterpret_cast<uint8_t*>(px), 1, reinterpret_cast<uint8_t*>(o)));
        CHECK(o[0] == -1 && o[1] == -128 && o[2] == 127 && o[3] == 5);
    }
    {  // RGBA16F NaN reads as 0 in RGBA/UNSIGNED_BYTE (like the generic path), never 255
        const FormatInfo* f = formatFor(gl::RGBA16F);
        uint16_t px[4] = {0x7E00, 0x3C00, 0x0000, 0x3C00};
        uint8_t o[4] = {};
        CHECK(convertReadback(*f, gl::RGBA, gl::UNSIGNED_BYTE, reinterpret_cast<uint8_t*>(px), 1, o));
        CHECK(o[0] == 0 && o[1] == 255);
    }
    {  // RGBA32F NaN must not be UB/garbage: 0
        const FormatInfo* f = formatFor(gl::RGBA32F);
        float px[4] = {__builtin_nanf(""), 1.0f, -1.0f, 0.5f};
        uint8_t o[4] = {9, 9, 9, 9};
        CHECK(convertReadback(*f, gl::RGBA, gl::UNSIGNED_BYTE, reinterpret_cast<uint8_t*>(px), 1, o));
        CHECK(o[0] == 0 && o[1] == 255 && o[2] == 0 && o[3] == 128);
    }
}

// D: size helpers.
void testSizes() {
    const FormatInfo* e = formatFor(gl::COMPRESSED_RGB8_ETC2);
    CHECK(e && compressedSize(*e, 5, 5) == 4 * 8);
    CHECK(compressedSize(*e, 0, 4) == 0);
    // large dimensions must not wrap 32 bits (GL_MAX_TEXTURE_SIZE can be 16384, 3D 2048; but w*h blocks overflow at 2^28 blocks)
    CHECK(clientPixelBytes(gl::RGBA, gl::UNSIGNED_BYTE) == 4);
    CHECK(clientPixelBytes(gl::DEPTH_STENCIL, gl::UNSIGNED_INT_24_8) == 4);
    CHECK(clientPixelBytes(gl::RGB, gl::UNSIGNED_SHORT_4_4_4_4) == 2);  // wrong format for the type (validated elsewhere)
}

// E: transform feedback assembly.
void testXfb() {
    std::vector<uint32_t> v = {0, 1, 2, 3, 4}, out;
    xfbAssemble(5, v.data(), v.size(), false, 0, out);  // strip
    CHECK((out == std::vector<uint32_t>{0, 1, 2, 2, 1, 3, 2, 3, 4}));
    out.clear();
    xfbAssemble(6, v.data(), v.size(), false, 0, out);  // fan: triangle k is (v[0], v[k+1], v[k+2]) (GL's order; Vulkan's is (v[k+1], v[k+2], v[0]))
    CHECK((out == std::vector<uint32_t>{0, 1, 2, 0, 2, 3, 0, 3, 4}));
    out.clear();
    xfbAssemble(6, v.data(), 2, false, 0, out);  // a fan of 2 vertices: no triangle
    CHECK(out.empty());
    out.clear();
    xfbAssemble(6, v.data(), 3, false, 0, out);
    CHECK((out == std::vector<uint32_t>{0, 1, 2}));
    out.clear();
    xfbAssemble(2, v.data(), 3, false, 0, out);  // line loop of 3: 0-1 1-2 2-0
    CHECK((out == std::vector<uint32_t>{0, 1, 1, 2, 2, 0}));
    out.clear();
    xfbAssemble(2, v.data(), 2, false, 0, out);  // loop of 2
    CHECK((out == std::vector<uint32_t>{0, 1, 1, 0}));
    out.clear();
    xfbAssemble(2, v.data(), 1, false, 0, out);  // loop of 1: nothing
    CHECK(out.empty());
    out.clear();
    // restart: two separate strips; an index equal to the restart value at the start/end/twice in a row
    std::vector<uint32_t> r = {9, 0, 1, 2, 9, 9, 3, 4, 5, 9};
    xfbAssemble(4, r.data(), r.size(), true, 9, out);  // triangles per sequence
    CHECK((out == std::vector<uint32_t>{0, 1, 2, 3, 4, 5}));
    out.clear();
    xfbAssemble(2, r.data(), r.size(), true, 9, out);  // loops per sequence
    CHECK((out == std::vector<uint32_t>{0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3}));
    // writes: a primitive is written whole or not at all; partial parts
    std::vector<uint32_t> rec = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
    uint8_t buf[64];
    std::memset(buf, 0xEE, sizeof buf);
    std::vector<XfbOut> outs(1);
    outs[0].dst = buf, outs[0].room = 3 * 8 + 4, outs[0].stride = 8, outs[0].parts = {{1, 2}};
    const uint64_t n = xfbWrite({0, 1, 2, 0, 1, 2}, 3, rec.data(), 4, outs);
    CHECK(n == 1 && outs[0].used == 24);
    uint32_t w[2];
    std::memcpy(w, buf + 8, 8);
    CHECK(w[0] == 21 && w[1] == 22);
    CHECK(buf[24] == 0xEE);
    // stride larger than the captured words: the gap is untouched, the next vertex starts at the stride
    outs[0] = XfbOut{};
    outs[0].dst = buf, outs[0].room = 64, outs[0].stride = 16, outs[0].parts = {{0, 1}};
    std::memset(buf, 0xEE, sizeof buf);
    CHECK(xfbWrite({2, 1}, 1, rec.data(), 4, outs) == 2 && outs[0].used == 32);
    std::memcpy(w, buf + 16, 4);
    CHECK(w[0] == 20);
    // 32-bit overflow of used + prim * stride must not wrap
    outs[0] = XfbOut{};
    outs[0].dst = buf, outs[0].room = 64, outs[0].used = 0, outs[0].stride = 0x80000000u, outs[0].parts = {{0, 1}};
    CHECK(xfbWrite({0, 1, 2}, 3, rec.data(), 4, outs) == 0);
    // zero vertices per primitive: nothing to write (and the loop must end)
    outs[0] = XfbOut{};
    outs[0].dst = buf, outs[0].room = 64, outs[0].stride = 4, outs[0].parts = {{0, 1}};
    CHECK(xfbWrite({0, 1}, 0, rec.data(), 4, outs) == 0);
    // restart value 0xFFFF / 0xFF for the index types is the caller's; here a restart value that is a real vertex id of a non-indexed draw
    out.clear();
    xfbAssemble(0, v.data(), 3, true, 1, out);  // points with restart 1: vertex 1 dropped
    CHECK((out == std::vector<uint32_t>{0, 2}));
}
}  // namespace

int main() {
    testUploadCoverage();
    testUploadValues();
    testPackedRounding();
    testPackedFloat();
    testLuminanceFloat();
    testReadback();
    testSizes();
    testXfb();
    std::printf("audit_formats: %s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
