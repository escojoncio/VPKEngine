// The GLES 2.0 / 3.0 / 3.1 (+ OES/EXT/OVR extensions games use) entry points.
//
// ABI: the guest is AAPCS64 and the visor build is Darwin arm64. They agree on x0-x7/v0-v7 and on
// 8-byte stack arguments, and disagree on narrower stack arguments (AAPCS64: one 8-byte slot
// each; Darwin: packed to their size). Every integer parameter past the eighth is therefore
// declared 64-bit here (`S`) and truncated, so both read the slot the caller wrote.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "context.h"
#include "egl_internal.h"
#include "vpgles/vpgles.h"

namespace vpgles {

void setDevice(Device* d);
bool eglImageToTexture(void* image, Texture& t);
void* eglProc(const char* name);

namespace {

using S = int64_t;  // stack-passed integer argument (see the header comment)

struct Lock {
    std::lock_guard<std::recursive_mutex> l;
    Lock() : l(device()->lock) {}
};
#define CTX(...)                 \
    Context* c = current();      \
    if (!c) return __VA_ARGS__;  \
    Lock lock_;

// ---- errors, state queries ----------------------------------------------------------------------------
GLenum GetError() {
    CTX(gl::NO_ERROR)
    GLenum e = c->error;
    c->error = gl::NO_ERROR;
    return e;
}

const char* kExtensions[] = {
    "GL_OES_element_index_uint", "GL_OES_packed_depth_stencil", "GL_OES_depth24", "GL_OES_depth32",
    "GL_OES_rgb8_rgba8", "GL_OES_standard_derivatives", "GL_OES_texture_npot", "GL_OES_vertex_array_object",
    "GL_OES_EGL_image", "GL_OES_EGL_image_external", "GL_OES_EGL_image_external_essl3", "GL_OES_texture_half_float",
    "GL_OES_texture_half_float_linear", "GL_OES_texture_float", "GL_OES_texture_float_linear",
    "GL_OES_compressed_ETC1_RGB8_texture", "GL_OES_vertex_half_float", "GL_OES_texture_3D", "GL_OES_surfaceless_context",
    "GL_OES_mapbuffer", "GL_OES_get_program_binary", "GL_EXT_texture_format_BGRA8888", "GL_EXT_read_format_bgra",
    "GL_EXT_texture_filter_anisotropic", "GL_EXT_shader_texture_lod", "GL_EXT_color_buffer_float",
    "GL_EXT_color_buffer_half_float", "GL_EXT_sRGB", "GL_EXT_texture_sRGB_decode", "GL_EXT_discard_framebuffer",
    "GL_EXT_multisampled_render_to_texture", "GL_EXT_multisampled_render_to_texture2", "GL_EXT_draw_buffers",
    "GL_EXT_frag_depth", "GL_EXT_shadow_samplers", "GL_EXT_texture_storage", "GL_EXT_map_buffer_range",
    "GL_EXT_occlusion_query_boolean", "GL_EXT_instanced_arrays", "GL_EXT_draw_instanced",
    "GL_EXT_texture_rg", "GL_EXT_unpack_subimage", "GL_EXT_debug_marker", "GL_EXT_robustness",
    "GL_KHR_texture_compression_astc_ldr", "GL_KHR_debug", "GL_OVR_multiview", "GL_OVR_multiview2",
    "GL_OVR_multiview_multisampled_render_to_texture", "GL_ANGLE_instanced_arrays", "GL_NV_draw_buffers",
};
std::string extensionString() {
    std::string s;
    for (const char* e : kExtensions) s += std::string(e) + " ";
    return s;
}

const GLubyte* GetString(GLenum name) {
    CTX(nullptr)
    static const std::string ext = extensionString();
    switch (name) {
        case gl::VENDOR: return reinterpret_cast<const GLubyte*>("Qualcomm");
        case gl::RENDERER: return reinterpret_cast<const GLubyte*>("Adreno (TM) 740");
        case gl::VERSION: return reinterpret_cast<const GLubyte*>("OpenGL ES 3.2 V@0615.0 (VPKEngine)");
        case gl::SHADING_LANGUAGE_VERSION: return reinterpret_cast<const GLubyte*>("OpenGL ES GLSL ES 3.20");
        case gl::EXTENSIONS: return reinterpret_cast<const GLubyte*>(ext.c_str());
        default: c->setError(gl::INVALID_ENUM); return nullptr;
    }
}
const GLubyte* GetStringi(GLenum name, GLuint i) {
    CTX(nullptr)
    if (name != gl::EXTENSIONS || i >= sizeof kExtensions / sizeof kExtensions[0]) {
        c->setError(name == gl::EXTENSIONS ? gl::INVALID_VALUE : gl::INVALID_ENUM);
        return nullptr;
    }
    return reinterpret_cast<const GLubyte*>(kExtensions[i]);
}

// Fills `out` (up to 16 values) for `pname`; returns the count, 0 for unknown pnames. `isFloat`
// marks values that are naturally floats (converted per caller).
int queryState(Context* c, GLenum p, double* out) {
    auto one = [&](double v) { out[0] = v; return 1; };
    auto fbBits = [&](int which) -> double {
        Context::FbInfo fb = c->resolveFramebuffer(c->drawFbo, true);
        const FormatInfo* f = nullptr;
        if (which < 4) f = fb.color[0].image ? fb.color[0].image->format : nullptr;
        else if (which == 4) f = fb.depth.image ? fb.depth.image->format : nullptr;
        else f = fb.stencil.image ? fb.stencil.image->format : nullptr;
        if (!f) return 0;
        switch (which) {
            case 0: return f->redBits;
            case 1: return f->greenBits;
            case 2: return f->blueBits;
            case 3: return f->alphaBits;
            case 4: return f->depthBits;
            default: return f->stencilBits;
        }
    };
    static const GLenum kCompressed[] = {gl::ETC1_RGB8_OES, gl::COMPRESSED_RGB8_ETC2, gl::COMPRESSED_SRGB8_ETC2,
                                         gl::COMPRESSED_RGBA8_ETC2_EAC, gl::COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
                                         gl::COMPRESSED_R11_EAC, gl::COMPRESSED_RG11_EAC, gl::COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
                                         0x93B0, 0x93B4, 0x93B7, 0x93D0, 0x93D4, 0x93D7};
    switch (p) {
        case gl::MAX_TEXTURE_SIZE: case gl::MAX_RENDERBUFFER_SIZE: case gl::MAX_CUBE_MAP_TEXTURE_SIZE: return one(16384);
        case gl::MAX_3D_TEXTURE_SIZE: return one(2048);
        case gl::MAX_ARRAY_TEXTURE_LAYERS: return one(2048);
        case gl::MAX_VIEWPORT_DIMS: out[0] = out[1] = 16384; return 2;
        case gl::MAX_VERTEX_ATTRIBS: return one(kMaxVertexAttribs);
        case gl::MAX_VERTEX_UNIFORM_VECTORS: case gl::MAX_FRAGMENT_UNIFORM_VECTORS: return one(256);
        case gl::MAX_VERTEX_UNIFORM_COMPONENTS: case gl::MAX_FRAGMENT_UNIFORM_COMPONENTS: return one(1024);
        case gl::MAX_VARYING_VECTORS: return one(15);
        case gl::MAX_VARYING_COMPONENTS: return one(60);
        case gl::MAX_VERTEX_OUTPUT_COMPONENTS: return one(64);
        case gl::MAX_FRAGMENT_INPUT_COMPONENTS: return one(60);
        case gl::MAX_COMBINED_TEXTURE_IMAGE_UNITS: return one(kMaxTextureUnits);
        case gl::MAX_VERTEX_TEXTURE_IMAGE_UNITS: case gl::MAX_TEXTURE_IMAGE_UNITS: return one(16);
        case gl::MAX_DRAW_BUFFERS: case gl::MAX_COLOR_ATTACHMENTS: return one(kMaxDrawBuffers);
        case gl::MAX_SAMPLES: return one(4);
        case gl::MAX_UNIFORM_BUFFER_BINDINGS: return one(kMaxUniformBufferBindings);
        case gl::MAX_UNIFORM_BLOCK_SIZE: return one(65536);
        case gl::MAX_VERTEX_UNIFORM_BLOCKS: case gl::MAX_FRAGMENT_UNIFORM_BLOCKS: return one(14);
        case gl::MAX_COMBINED_UNIFORM_BLOCKS: return one(28);
        case gl::UNIFORM_BUFFER_OFFSET_ALIGNMENT: return one(64);
        case gl::MAX_ELEMENT_INDEX: return one(4294967295.0);
        case gl::MAX_ELEMENTS_INDICES: case gl::MAX_ELEMENTS_VERTICES: return one(1 << 20);
        case gl::MAX_TEXTURE_LOD_BIAS: return one(16);
        case gl::MIN_PROGRAM_TEXEL_OFFSET: return one(-8);
        case gl::MAX_PROGRAM_TEXEL_OFFSET: return one(7);
        case gl::MAX_TEXTURE_MAX_ANISOTROPY_EXT: return one(16);
        case gl::MAX_VIEWS_OVR: return one(2);
        case gl::MAX_SERVER_WAIT_TIMEOUT: return one(0);
        case gl::MAX_VERTEX_ATTRIB_BINDINGS: return one(16);
        case gl::MAX_VERTEX_ATTRIB_STRIDE: return one(2048);
        case gl::MAX_SAMPLE_MASK_WORDS: return one(1);
        case gl::MAX_COMPUTE_WORK_GROUP_INVOCATIONS: return one(1024);
        case gl::ALIASED_LINE_WIDTH_RANGE: out[0] = out[1] = 1; return 2;
        case gl::ALIASED_POINT_SIZE_RANGE: out[0] = 1, out[1] = 511; return 2;
        case gl::SUBPIXEL_BITS: return one(8);
        case gl::MAJOR_VERSION: return one(3);
        case gl::MINOR_VERSION: return one(2);
        case gl::NUM_EXTENSIONS: return one(double(sizeof kExtensions / sizeof kExtensions[0]));
        case gl::NUM_COMPRESSED_TEXTURE_FORMATS: return one(double(sizeof kCompressed / sizeof kCompressed[0]));
        case gl::COMPRESSED_TEXTURE_FORMATS: {
            int n = 0;
            for (GLenum f : kCompressed) out[n++] = f;
            return n;
        }
        case gl::NUM_SHADER_BINARY_FORMATS: case gl::NUM_PROGRAM_BINARY_FORMATS: return one(0);
        case gl::SHADER_COMPILER: return one(1);
        case gl::RED_BITS: return one(fbBits(0));
        case gl::GREEN_BITS: return one(fbBits(1));
        case gl::BLUE_BITS: return one(fbBits(2));
        case gl::ALPHA_BITS: return one(fbBits(3));
        case gl::DEPTH_BITS: return one(fbBits(4));
        case gl::STENCIL_BITS: return one(fbBits(5));
        case gl::SAMPLE_BUFFERS: return one(0);
        case gl::SAMPLES: return one(0);
        case gl::IMPLEMENTATION_COLOR_READ_FORMAT: return one(gl::RGBA);
        case gl::IMPLEMENTATION_COLOR_READ_TYPE: return one(gl::UNSIGNED_BYTE);
        case gl::VIEWPORT: for (int i = 0; i < 4; ++i) out[i] = c->viewport[i]; return 4;
        case gl::SCISSOR_BOX: for (int i = 0; i < 4; ++i) out[i] = c->scissor[i]; return 4;
        case gl::DEPTH_RANGE: out[0] = c->depthRange[0], out[1] = c->depthRange[1]; return 2;
        case gl::COLOR_CLEAR_VALUE: for (int i = 0; i < 4; ++i) out[i] = c->clearColor[i]; return 4;
        case gl::COLOR_WRITEMASK: for (int i = 0; i < 4; ++i) out[i] = c->blend[0].mask[i]; return 4;
        case gl::BLEND_COLOR: for (int i = 0; i < 4; ++i) out[i] = c->blendColor[i]; return 4;
        case gl::DEPTH_CLEAR_VALUE: return one(c->clearDepth);
        case gl::DEPTH_FUNC: return one(c->depthFunc);
        case gl::DEPTH_WRITEMASK: return one(c->depthMask);
        case gl::STENCIL_CLEAR_VALUE: return one(c->clearStencil);
        case gl::STENCIL_FUNC: return one(c->stencil[0].func);
        case gl::STENCIL_REF: return one(c->stencil[0].ref);
        case gl::STENCIL_VALUE_MASK: return one(c->stencil[0].mask);
        case gl::STENCIL_WRITEMASK: return one(c->stencil[0].writeMask);
        case gl::STENCIL_FAIL: return one(c->stencil[0].fail);
        case gl::STENCIL_PASS_DEPTH_FAIL: return one(c->stencil[0].zfail);
        case gl::STENCIL_PASS_DEPTH_PASS: return one(c->stencil[0].zpass);
        case gl::STENCIL_BACK_FUNC: return one(c->stencil[1].func);
        case gl::STENCIL_BACK_REF: return one(c->stencil[1].ref);
        case gl::STENCIL_BACK_VALUE_MASK: return one(c->stencil[1].mask);
        case gl::STENCIL_BACK_WRITEMASK: return one(c->stencil[1].writeMask);
        case gl::STENCIL_BACK_FAIL: return one(c->stencil[1].fail);
        case gl::STENCIL_BACK_PASS_DEPTH_FAIL: return one(c->stencil[1].zfail);
        case gl::STENCIL_BACK_PASS_DEPTH_PASS: return one(c->stencil[1].zpass);
        case gl::CULL_FACE_MODE: return one(c->cullMode);
        case gl::FRONT_FACE: return one(c->frontFace);
        case gl::LINE_WIDTH: return one(c->lineWidth);
        case gl::POLYGON_OFFSET_FACTOR: return one(c->polygonOffset[0]);
        case gl::POLYGON_OFFSET_UNITS: return one(c->polygonOffset[1]);
        case gl::BLEND_SRC_RGB: return one(c->blend[0].srcRGB);
        case gl::BLEND_DST_RGB: return one(c->blend[0].dstRGB);
        case gl::BLEND_SRC_ALPHA: return one(c->blend[0].srcA);
        case gl::BLEND_DST_ALPHA: return one(c->blend[0].dstA);
        case gl::BLEND_EQUATION_RGB: return one(c->blend[0].eqRGB);
        case gl::BLEND_EQUATION_ALPHA: return one(c->blend[0].eqA);
        case gl::SAMPLE_COVERAGE_VALUE: return one(c->sampleCoverageValue);
        case gl::SAMPLE_COVERAGE_INVERT: return one(c->sampleCoverageInvert);
        case gl::ACTIVE_TEXTURE: return one(gl::TEXTURE0 + c->activeUnit);
        case gl::TEXTURE_BINDING_2D: return one(c->units[size_t(c->activeUnit)].tex2D);
        case gl::TEXTURE_BINDING_CUBE_MAP: return one(c->units[size_t(c->activeUnit)].texCube);
        case gl::TEXTURE_BINDING_3D: return one(c->units[size_t(c->activeUnit)].tex3D);
        case gl::TEXTURE_BINDING_2D_ARRAY: return one(c->units[size_t(c->activeUnit)].tex2DArray);
        case gl::TEXTURE_BINDING_EXTERNAL_OES: return one(c->units[size_t(c->activeUnit)].texExternal);
        case gl::SAMPLER_BINDING: return one(c->units[size_t(c->activeUnit)].sampler);
        case gl::ARRAY_BUFFER_BINDING: return one(c->arrayBuffer);
        case gl::ELEMENT_ARRAY_BUFFER_BINDING: return one(c->currentVao()->elementBuffer);
        case gl::UNIFORM_BUFFER_BINDING: return one(c->uniformBuffer);
        case gl::PIXEL_PACK_BUFFER_BINDING: return one(c->pixelPack);
        case gl::PIXEL_UNPACK_BUFFER_BINDING: return one(c->pixelUnpack);
        case 0x8F36: return one(c->copyRead);
        case 0x8F37: return one(c->copyWrite);
        case gl::VERTEX_ARRAY_BINDING: return one(c->vao);
        case gl::FRAMEBUFFER_BINDING: return one(c->drawFbo);
        case gl::READ_FRAMEBUFFER_BINDING: return one(c->readFbo);
        case gl::RENDERBUFFER_BINDING: return one(c->renderbuffer);
        case gl::CURRENT_PROGRAM: return one(c->program);
        case gl::UNPACK_ALIGNMENT: return one(c->unpackAlignment);
        case gl::PACK_ALIGNMENT: return one(c->packAlignment);
        case gl::UNPACK_ROW_LENGTH: return one(c->unpackRowLength);
        case gl::UNPACK_SKIP_ROWS: return one(c->unpackSkipRows);
        case gl::UNPACK_SKIP_PIXELS: return one(c->unpackSkipPixels);
        case gl::UNPACK_IMAGE_HEIGHT: return one(c->unpackImageHeight);
        case gl::UNPACK_SKIP_IMAGES: return one(c->unpackSkipImages);
        case gl::PACK_ROW_LENGTH: return one(c->packRowLength);
        case gl::PACK_SKIP_ROWS: return one(c->packSkipRows);
        case gl::PACK_SKIP_PIXELS: return one(c->packSkipPixels);
        case gl::READ_BUFFER: {
            const Framebuffer* f = c->fbo(c->readFbo);
            return one(f ? f->readBuffer : gl::BACK);
        }
        case gl::GENERATE_MIPMAP_HINT: return one(c->hints[0]);
        case gl::FRAGMENT_SHADER_DERIVATIVE_HINT: return one(c->hints[1]);
        case gl::CONTEXT_FLAGS: return one(0);
        case gl::RESET_NOTIFICATION_STRATEGY: return one(gl::NO_RESET_NOTIFICATION);
        case gl::GPU_DISJOINT_EXT: return one(0);
        case gl::MAX_DEBUG_MESSAGE_LENGTH: return one(1024);
        default: break;
    }
    if (p >= gl::DRAW_BUFFER0 && p < gl::DRAW_BUFFER0 + kMaxDrawBuffers) {
        const Framebuffer* f = c->fbo(c->drawFbo);
        return one(f ? f->drawBuffers[p - gl::DRAW_BUFFER0] : gl::NONE);
    }
    return 0;
}

bool IsEnabledImpl(Context* c, GLenum cap, bool& ok) {
    ok = true;
    switch (cap) {
        case gl::CULL_FACE: return c->cullFace;
        case gl::DEPTH_TEST: return c->depthTest;
        case gl::STENCIL_TEST: return c->stencilTest;
        case gl::DITHER: return c->dither;
        case gl::BLEND: return c->blend[0].enabled;
        case gl::SCISSOR_TEST: return c->scissorTest;
        case gl::POLYGON_OFFSET_FILL: return c->polygonOffsetFill;
        case gl::SAMPLE_ALPHA_TO_COVERAGE: return c->sampleAlphaToCoverage;
        case gl::SAMPLE_COVERAGE: return c->sampleCoverage;
        case gl::PRIMITIVE_RESTART_FIXED_INDEX: return c->primitiveRestart;
        case gl::RASTERIZER_DISCARD: return c->rasterizerDiscard;
        case gl::SAMPLE_MASK: case gl::DEBUG_OUTPUT: case gl::DEBUG_OUTPUT_SYNCHRONOUS: case gl::FRAMEBUFFER_SRGB_EXT: return false;
        default: ok = false; return false;
    }
}

void GetIntegerv(GLenum p, GLint* v) {
    CTX()
    double tmp[64];
    int n = queryState(c, p, tmp);
    if (!n) {
        bool ok;
        bool e = IsEnabledImpl(c, p, ok);
        if (ok) {
            v[0] = e;
            return;
        }
        return c->setError(gl::INVALID_ENUM);
    }
    const bool floatColor = p == gl::COLOR_CLEAR_VALUE || p == gl::BLEND_COLOR || p == gl::DEPTH_RANGE ||
                            p == gl::DEPTH_CLEAR_VALUE;
    for (int i = 0; i < n; ++i)
        v[i] = floatColor ? GLint(std::clamp(tmp[i], -1.0, 1.0) * 2147483647.0) : GLint(int64_t(tmp[i]));
}
void GetInteger64v(GLenum p, GLint64* v) {
    CTX()
    double tmp[64];
    int n = queryState(c, p, tmp);
    if (!n) return c->setError(gl::INVALID_ENUM);
    for (int i = 0; i < n; ++i) v[i] = GLint64(tmp[i]);
}
void GetFloatv(GLenum p, GLfloat* v) {
    CTX()
    double tmp[64];
    int n = queryState(c, p, tmp);
    if (!n) {
        bool ok;
        bool e = IsEnabledImpl(c, p, ok);
        if (ok) {
            v[0] = e;
            return;
        }
        return c->setError(gl::INVALID_ENUM);
    }
    for (int i = 0; i < n; ++i) v[i] = GLfloat(tmp[i]);
}
void GetBooleanv(GLenum p, GLboolean* v) {
    CTX()
    double tmp[64];
    int n = queryState(c, p, tmp);
    if (!n) {
        bool ok;
        bool e = IsEnabledImpl(c, p, ok);
        if (ok) {
            v[0] = e;
            return;
        }
        return c->setError(gl::INVALID_ENUM);
    }
    for (int i = 0; i < n; ++i) v[i] = tmp[i] != 0.0;
}
void GetIntegeri_v(GLenum target, GLuint index, GLint* v) {
    CTX()
    if (target == gl::UNIFORM_BUFFER_BINDING && index < kMaxUniformBufferBindings) {
        v[0] = GLint(c->uniformBindings[index].buffer);
        return;
    }
    if (target == 0x8A29 /* UNIFORM_BUFFER_START */ && index < kMaxUniformBufferBindings) {
        v[0] = GLint(c->uniformBindings[index].offset);
        return;
    }
    if (target == 0x8A2A /* UNIFORM_BUFFER_SIZE */ && index < kMaxUniformBufferBindings) {
        v[0] = GLint(c->uniformBindings[index].size);
        return;
    }
    if (target == gl::MAX_COMPUTE_WORK_GROUP_COUNT) {
        v[0] = 65535;
        return;
    }
    if (target == gl::MAX_COMPUTE_WORK_GROUP_SIZE) {
        v[0] = index == 2 ? 64 : 1024;
        return;
    }
    c->setError(gl::INVALID_ENUM);
}
void GetInteger64i_v(GLenum target, GLuint index, GLint64* v) {
    GLint x = 0;
    GetIntegeri_v(target, index, &x);
    *v = x;
}
void GetBooleani_v(GLenum, GLuint, GLboolean* v) { *v = 0; }

void Enable_(GLenum cap, bool on) {
    CTX()
    switch (cap) {
        case gl::CULL_FACE: c->cullFace = on; break;
        case gl::DEPTH_TEST: c->depthTest = on; break;
        case gl::STENCIL_TEST: c->stencilTest = on; break;
        case gl::DITHER: c->dither = on; break;
        case gl::BLEND: for (auto& b : c->blend) b.enabled = on; break;
        case gl::SCISSOR_TEST: c->scissorTest = on; break;
        case gl::POLYGON_OFFSET_FILL: c->polygonOffsetFill = on; break;
        case gl::SAMPLE_ALPHA_TO_COVERAGE: c->sampleAlphaToCoverage = on; break;
        case gl::SAMPLE_COVERAGE: c->sampleCoverage = on; break;
        case gl::PRIMITIVE_RESTART_FIXED_INDEX: c->primitiveRestart = on; break;
        case gl::RASTERIZER_DISCARD: c->rasterizerDiscard = on; break;
        case gl::SAMPLE_MASK: case gl::DEBUG_OUTPUT: case gl::DEBUG_OUTPUT_SYNCHRONOUS: case gl::FRAMEBUFFER_SRGB_EXT: break;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void Enable(GLenum cap) { Enable_(cap, true); }
void Disable(GLenum cap) { Enable_(cap, false); }
GLboolean IsEnabled(GLenum cap) {
    CTX(0)
    bool ok;
    bool e = IsEnabledImpl(c, cap, ok);
    if (!ok) c->setError(gl::INVALID_ENUM);
    return e;
}
void Enablei(GLenum cap, GLuint i) {
    CTX()
    if (cap == gl::BLEND && i < kMaxDrawBuffers) c->blend[i].enabled = true;
}
void Disablei(GLenum cap, GLuint i) {
    CTX()
    if (cap == gl::BLEND && i < kMaxDrawBuffers) c->blend[i].enabled = false;
}

// ---- fixed function ------------------------------------------------------------------------------------
void Viewport(GLint x, GLint y, GLsizei w, GLsizei h) {
    CTX()
    if (w < 0 || h < 0) return c->setError(gl::INVALID_VALUE);
    c->viewport[0] = x, c->viewport[1] = y, c->viewport[2] = std::min(w, 16384), c->viewport[3] = std::min(h, 16384);
}
void Scissor(GLint x, GLint y, GLsizei w, GLsizei h) {
    CTX()
    if (w < 0 || h < 0) return c->setError(gl::INVALID_VALUE);
    c->scissor[0] = x, c->scissor[1] = y, c->scissor[2] = w, c->scissor[3] = h;
}
void DepthRangef(GLfloat n, GLfloat f) {
    CTX()
    c->depthRange[0] = std::clamp(n, 0.0f, 1.0f), c->depthRange[1] = std::clamp(f, 0.0f, 1.0f);
}
void ClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    CTX()
    c->clearColor[0] = r, c->clearColor[1] = g, c->clearColor[2] = b, c->clearColor[3] = a;
}
void ClearDepthf(GLfloat d) {
    CTX()
    c->clearDepth = std::clamp(d, 0.0f, 1.0f);
}
void ClearStencil(GLint s) {
    CTX()
    c->clearStencil = s;
}
void Clear(GLbitfield m) {
    CTX()
    c->clear(m);
}
void ClearBufferfv(GLenum b, GLint i, const GLfloat* v) {
    CTX()
    if (b == gl::DEPTH) return c->clearBuffer(b, i, nullptr, gl::FLOAT, v[0], 0);
    c->clearBuffer(b, i, v, gl::FLOAT, 0, 0);
}
void ClearBufferiv(GLenum b, GLint i, const GLint* v) {
    CTX()
    if (b == gl::STENCIL) return c->clearBuffer(b, i, nullptr, gl::INT, 0, v[0]);
    c->clearBuffer(b, i, v, gl::INT, 0, 0);
}
void ClearBufferuiv(GLenum b, GLint i, const GLuint* v) {
    CTX()
    c->clearBuffer(b, i, v, gl::UNSIGNED_INT, 0, 0);
}
void ClearBufferfi(GLenum b, GLint i, GLfloat d, GLint s) {
    CTX()
    c->clearBuffer(b, i, nullptr, 0, d, s);
}
void ColorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
    CTX()
    for (auto& x : c->blend) x.mask[0] = r, x.mask[1] = g, x.mask[2] = b, x.mask[3] = a;
}
void ColorMaski(GLuint i, GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
    CTX()
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE);
    auto& x = c->blend[i];
    x.mask[0] = r, x.mask[1] = g, x.mask[2] = b, x.mask[3] = a;
}
void DepthMask(GLboolean m) {
    CTX()
    c->depthMask = m;
}
void DepthFunc(GLenum f) {
    CTX()
    if (f < gl::NEVER || f > gl::ALWAYS) return c->setError(gl::INVALID_ENUM);
    c->depthFunc = f;
}
void CullFace(GLenum m) {
    CTX()
    if (m != gl::FRONT && m != gl::BACK && m != gl::FRONT_AND_BACK) return c->setError(gl::INVALID_ENUM);
    c->cullMode = m;
}
void FrontFace(GLenum m) {
    CTX()
    if (m != gl::CW && m != gl::CCW) return c->setError(gl::INVALID_ENUM);
    c->frontFace = m;
}
void LineWidth(GLfloat w) {
    CTX()
    if (w <= 0) return c->setError(gl::INVALID_VALUE);
    c->lineWidth = w;
}
void PolygonOffset(GLfloat f, GLfloat u) {
    CTX()
    c->polygonOffset[0] = f, c->polygonOffset[1] = u;
}
void SampleCoverage(GLfloat v, GLboolean inv) {
    CTX()
    c->sampleCoverageValue = std::clamp(v, 0.0f, 1.0f), c->sampleCoverageInvert = inv;
}
void Hint(GLenum t, GLenum m) {
    CTX()
    if (t == gl::GENERATE_MIPMAP_HINT) c->hints[0] = m;
    else if (t == gl::FRAGMENT_SHADER_DERIVATIVE_HINT) c->hints[1] = m;
}
void BlendFunc(GLenum s, GLenum d) {
    CTX()
    for (auto& b : c->blend) b.srcRGB = b.srcA = s, b.dstRGB = b.dstA = d;
}
void BlendFuncSeparate(GLenum sr, GLenum dr, GLenum sa, GLenum da) {
    CTX()
    for (auto& b : c->blend) b.srcRGB = sr, b.dstRGB = dr, b.srcA = sa, b.dstA = da;
}
void BlendFunci(GLuint i, GLenum s, GLenum d) {
    CTX()
    if (i < kMaxDrawBuffers) c->blend[i].srcRGB = c->blend[i].srcA = s, c->blend[i].dstRGB = c->blend[i].dstA = d;
}
void BlendFuncSeparatei(GLuint i, GLenum sr, GLenum dr, GLenum sa, GLenum da) {
    CTX()
    if (i < kMaxDrawBuffers) c->blend[i].srcRGB = sr, c->blend[i].dstRGB = dr, c->blend[i].srcA = sa, c->blend[i].dstA = da;
}
void BlendEquation(GLenum e) {
    CTX()
    for (auto& b : c->blend) b.eqRGB = b.eqA = e;
}
void BlendEquationSeparate(GLenum r, GLenum a) {
    CTX()
    for (auto& b : c->blend) b.eqRGB = r, b.eqA = a;
}
void BlendEquationi(GLuint i, GLenum e) {
    CTX()
    if (i < kMaxDrawBuffers) c->blend[i].eqRGB = c->blend[i].eqA = e;
}
void BlendEquationSeparatei(GLuint i, GLenum r, GLenum a) {
    CTX()
    if (i < kMaxDrawBuffers) c->blend[i].eqRGB = r, c->blend[i].eqA = a;
}
void BlendColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    CTX()
    c->blendColor[0] = r, c->blendColor[1] = g, c->blendColor[2] = b, c->blendColor[3] = a;
}
void StencilFuncSeparate(GLenum face, GLenum f, GLint ref, GLuint mask) {
    CTX()
    for (int i = 0; i < 2; ++i)
        if (face == gl::FRONT_AND_BACK || (face == gl::FRONT) == (i == 0)) c->stencil[i].func = f, c->stencil[i].ref = ref, c->stencil[i].mask = mask;
}
void StencilFunc(GLenum f, GLint ref, GLuint mask) { StencilFuncSeparate(gl::FRONT_AND_BACK, f, ref, mask); }
void StencilMaskSeparate(GLenum face, GLuint m) {
    CTX()
    for (int i = 0; i < 2; ++i)
        if (face == gl::FRONT_AND_BACK || (face == gl::FRONT) == (i == 0)) c->stencil[i].writeMask = m;
}
void StencilMask(GLuint m) { StencilMaskSeparate(gl::FRONT_AND_BACK, m); }
void StencilOpSeparate(GLenum face, GLenum sf, GLenum zf, GLenum zp) {
    CTX()
    for (int i = 0; i < 2; ++i)
        if (face == gl::FRONT_AND_BACK || (face == gl::FRONT) == (i == 0)) c->stencil[i].fail = sf, c->stencil[i].zfail = zf, c->stencil[i].zpass = zp;
}
void StencilOp(GLenum sf, GLenum zf, GLenum zp) { StencilOpSeparate(gl::FRONT_AND_BACK, sf, zf, zp); }
void PixelStorei(GLenum p, GLint v) {
    CTX()
    switch (p) {
        case gl::UNPACK_ALIGNMENT:
        case gl::PACK_ALIGNMENT:
            if (v != 1 && v != 2 && v != 4 && v != 8) return c->setError(gl::INVALID_VALUE);
            (p == gl::UNPACK_ALIGNMENT ? c->unpackAlignment : c->packAlignment) = v;
            return;
        case gl::UNPACK_ROW_LENGTH: c->unpackRowLength = v; return;
        case gl::UNPACK_SKIP_ROWS: c->unpackSkipRows = v; return;
        case gl::UNPACK_SKIP_PIXELS: c->unpackSkipPixels = v; return;
        case gl::UNPACK_IMAGE_HEIGHT: c->unpackImageHeight = v; return;
        case gl::UNPACK_SKIP_IMAGES: c->unpackSkipImages = v; return;
        case gl::PACK_ROW_LENGTH: c->packRowLength = v; return;
        case gl::PACK_SKIP_ROWS: c->packSkipRows = v; return;
        case gl::PACK_SKIP_PIXELS: c->packSkipPixels = v; return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void Flush() {
    CTX()
    c->flush(false);
}
void Finish() {
    CTX()
    c->flush(true);
}

// ---- buffers --------------------------------------------------------------------------------------------
void GenBuffers(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->genName();
        auto b = std::make_unique<Buffer>();
        b->name = name;
        c->share->buffers[name] = std::move(b);
        out[i] = name;
    }
}
void DeleteBuffers(GLsizei n, const GLuint* names) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint x = names[i];
        if (!x) continue;
        auto it = c->share->buffers.find(x);
        if (it == c->share->buffers.end()) continue;
        if (it->second->storage) c->keep(it->second->storage);
        // Unbind from this context.
        for (GLuint* b : {&c->arrayBuffer, &c->copyRead, &c->copyWrite, &c->pixelPack, &c->pixelUnpack, &c->uniformBuffer})
            if (*b == x) *b = 0;
        for (auto& ub : c->uniformBindings)
            if (ub.buffer == x) ub = {};
        for (auto& [k, va] : c->vaos) {
            if (va->elementBuffer == x) va->elementBuffer = 0;
            for (auto& a : va->attribs)
                if (a.buffer == x) a.buffer = 0;
        }
        c->share->buffers.erase(it);
    }
}
GLboolean IsBuffer(GLuint n) {
    CTX(0)
    return c->buffer(n) && c->buffer(n)->storage ? 1 : 0;
}
Buffer* bindNamed(Context* c, GLuint name) {
    if (!name) return nullptr;
    Buffer* b = c->buffer(name);
    if (!b) {
        // GLES 2 allows binding names never generated.
        auto nb = std::make_unique<Buffer>();
        nb->name = name;
        b = nb.get();
        c->share->buffers[name] = std::move(nb);
        c->share->nextName = std::max(c->share->nextName, name + 1);
    }
    return b;
}
void BindBuffer(GLenum target, GLuint name) {
    CTX()
    GLuint* slot = c->bufferBinding(target);
    if (!slot) return c->setError(gl::INVALID_ENUM);
    bindNamed(c, name);
    *slot = name;
}
void BindBufferRange(GLenum target, GLuint index, GLuint name, GLintptr off, GLsizeiptr size) {
    CTX()
    if (target != gl::UNIFORM_BUFFER || index >= kMaxUniformBufferBindings) return c->setError(gl::INVALID_VALUE);
    if (name && (off < 0 || size <= 0 || off % 64)) return c->setError(gl::INVALID_VALUE);
    bindNamed(c, name);
    c->uniformBindings[index] = {name, off, size};
    c->uniformBuffer = name;
}
void BindBufferBase(GLenum target, GLuint index, GLuint name) {
    CTX()
    if (target != gl::UNIFORM_BUFFER || index >= kMaxUniformBufferBindings) return c->setError(gl::INVALID_VALUE);
    bindNamed(c, name);
    c->uniformBindings[index] = {name, 0, 0};
    c->uniformBuffer = name;
}
Buffer* targetBuffer(Context* c, GLenum target) {
    GLuint* slot = c->bufferBinding(target);
    if (!slot) {
        c->setError(gl::INVALID_ENUM);
        return nullptr;
    }
    Buffer* b = c->buffer(*slot);
    if (!b) c->setError(gl::INVALID_OPERATION);
    return b;
}
constexpr VkBufferUsageFlags kAllUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
void BufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    CTX()
    if (size < 0) return c->setError(gl::INVALID_VALUE);
    Buffer* b = targetBuffer(c, target);
    if (!b) return;
    if (b->mapped) return c->setError(gl::INVALID_OPERATION);
    // New storage every time (the old one stays alive with any work that reads it).
    if (b->storage) c->keep(b->storage);
    b->storage = c->dev->newStorage(VkDeviceSize(std::max<GLsizeiptr>(size, 1)), kAllUsage);
    if (!b->storage) return c->setError(gl::OUT_OF_MEMORY);
    b->size = size, b->usage = usage;
    if (data) std::memcpy(b->storage->host, data, size_t(size));
    else std::memset(b->storage->host, 0, size_t(size));
}
void BufferSubData(GLenum target, GLintptr off, GLsizeiptr size, const void* data) {
    CTX()
    Buffer* b = targetBuffer(c, target);
    if (!b) return;
    if (off < 0 || size < 0 || off + size > b->size) return c->setError(gl::INVALID_VALUE);
    if (b->mapped) return c->setError(gl::INVALID_OPERATION);
    if (!data || !size) return;
    c->prepareWrite(*b, off == 0 && size == b->size);
    std::memcpy(b->storage->host + off, data, size_t(size));
}
void* MapBufferRange(GLenum target, GLintptr off, GLsizeiptr len, GLbitfield access) {
    CTX(nullptr)
    Buffer* b = targetBuffer(c, target);
    if (!b) return nullptr;
    if (off < 0 || len < 0 || off + len > b->size || !(access & (gl::MAP_READ_BIT | gl::MAP_WRITE_BIT))) {
        c->setError(gl::INVALID_VALUE);
        return nullptr;
    }
    if (b->mapped) {
        c->setError(gl::INVALID_OPERATION);
        return nullptr;
    }
    if (access & gl::MAP_READ_BIT) {
        if (b->storage->lastUse > c->dev->completed) c->flush(true);  // GPU writes visible
    } else if (!(access & gl::MAP_UNSYNCHRONIZED_BIT)) {
        c->prepareWrite(*b, (access & gl::MAP_INVALIDATE_BUFFER_BIT) != 0);
    }
    b->mapped = true, b->mapAccess = access, b->mapOffset = off, b->mapLength = len;
    return b->storage->host + off;
}
void* MapBufferOES(GLenum target, GLenum) {
    CTX(nullptr)
    Buffer* b = targetBuffer(c, target);
    if (!b) return nullptr;
    return MapBufferRange(target, 0, b->size, gl::MAP_WRITE_BIT);
}
GLboolean UnmapBuffer(GLenum target) {
    CTX(0)
    Buffer* b = targetBuffer(c, target);
    if (!b) return 0;
    if (!b->mapped) {
        c->setError(gl::INVALID_OPERATION);
        return 0;
    }
    b->mapped = false;
    return 1;
}
void FlushMappedBufferRange(GLenum, GLintptr, GLsizeiptr) {}  // coherent memory
void GetBufferParameteriv(GLenum target, GLenum p, GLint* v) {
    CTX()
    Buffer* b = targetBuffer(c, target);
    if (!b) return;
    switch (p) {
        case gl::BUFFER_SIZE: *v = GLint(b->size); return;
        case gl::BUFFER_USAGE: *v = GLint(b->usage); return;
        case gl::BUFFER_MAPPED: *v = b->mapped; return;
        case gl::BUFFER_ACCESS_FLAGS: *v = GLint(b->mapAccess); return;
        case gl::BUFFER_MAP_LENGTH: *v = GLint(b->mapLength); return;
        case gl::BUFFER_MAP_OFFSET: *v = GLint(b->mapOffset); return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void GetBufferParameteri64v(GLenum target, GLenum p, GLint64* v) {
    GLint x = 0;
    GetBufferParameteriv(target, p, &x);
    *v = x;
}
void GetBufferPointerv(GLenum target, GLenum, void** v) {
    CTX()
    Buffer* b = targetBuffer(c, target);
    *v = b && b->mapped ? b->storage->host + b->mapOffset : nullptr;
}
void CopyBufferSubData(GLenum rt, GLenum wt, GLintptr ro, GLintptr wo, GLsizeiptr size) {
    CTX()
    Buffer* r = targetBuffer(c, rt);
    Buffer* w = targetBuffer(c, wt);
    if (!r || !w) return;
    if (ro < 0 || wo < 0 || size < 0 || ro + size > r->size || wo + size > w->size) return c->setError(gl::INVALID_VALUE);
    if (!size) return;
    c->useStorage(r->storage);
    c->useStorage(w->storage);
    c->barrier();
    VkBufferCopy bc{VkDeviceSize(ro), VkDeviceSize(wo), VkDeviceSize(size)};
    c->dev->vk.vkCmdCopyBuffer(c->cmd(), r->storage->buffer, w->storage->buffer, 1, &bc);
    c->barrier();
}

// ---- vertex arrays ----------------------------------------------------------------------------------------
void GenVertexArrays(GLsizei n, GLuint* out) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->nextVao++;
        c->vaos[name] = std::make_unique<VertexArray>();
        c->vaos[name]->name = name;
        out[i] = name;
    }
}
void DeleteVertexArrays(GLsizei n, const GLuint* names) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        if (!names[i]) continue;
        if (c->vao == names[i]) c->vao = 0;
        c->vaos.erase(names[i]);
    }
}
void BindVertexArray(GLuint n) {
    CTX()
    if (!c->vaos.count(n)) return c->setError(gl::INVALID_OPERATION);
    c->vao = n;
}
GLboolean IsVertexArray(GLuint n) {
    CTX(0)
    return n && c->vaos.count(n) ? 1 : 0;
}
void EnableVertexAttribArray(GLuint i) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    c->currentVao()->attribs[i].enabled = true;
}
void DisableVertexAttribArray(GLuint i) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    c->currentVao()->attribs[i].enabled = false;
}
void attribPointer(Context* c, GLuint i, GLint size, GLenum type, bool norm, bool integer, GLsizei stride, const void* ptr) {
    if (i >= kMaxVertexAttribs || size < 1 || size > 4 || stride < 0) return c->setError(gl::INVALID_VALUE);
    if (c->vao && !c->arrayBuffer && ptr) return c->setError(gl::INVALID_OPERATION);
    VertexAttrib& a = c->currentVao()->attribs[i];
    a.size = size, a.type = type, a.normalized = norm, a.integer = integer, a.stride = stride;
    a.buffer = c->arrayBuffer, a.pointer = ptr;
}
void VertexAttribPointer(GLuint i, GLint size, GLenum type, GLboolean norm, GLsizei stride, const void* p) {
    CTX()
    attribPointer(c, i, size, type, norm, false, stride, p);
}
void VertexAttribIPointer(GLuint i, GLint size, GLenum type, GLsizei stride, const void* p) {
    CTX()
    attribPointer(c, i, size, type, false, true, stride, p);
}
void VertexAttribDivisor(GLuint i, GLuint d) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    c->currentVao()->attribs[i].divisor = d;
}
void setCurrentAttrib(GLuint i, float x, float y, float z, float w) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    c->currentAttrib[i] = {x, y, z, w};
    c->currentAttribKind[i] = 0;
}
void VertexAttrib1f(GLuint i, GLfloat x) { setCurrentAttrib(i, x, 0, 0, 1); }
void VertexAttrib2f(GLuint i, GLfloat x, GLfloat y) { setCurrentAttrib(i, x, y, 0, 1); }
void VertexAttrib3f(GLuint i, GLfloat x, GLfloat y, GLfloat z) { setCurrentAttrib(i, x, y, z, 1); }
void VertexAttrib4f(GLuint i, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { setCurrentAttrib(i, x, y, z, w); }
void VertexAttrib1fv(GLuint i, const GLfloat* v) { setCurrentAttrib(i, v[0], 0, 0, 1); }
void VertexAttrib2fv(GLuint i, const GLfloat* v) { setCurrentAttrib(i, v[0], v[1], 0, 1); }
void VertexAttrib3fv(GLuint i, const GLfloat* v) { setCurrentAttrib(i, v[0], v[1], v[2], 1); }
void VertexAttrib4fv(GLuint i, const GLfloat* v) { setCurrentAttrib(i, v[0], v[1], v[2], v[3]); }
void VertexAttribI4i(GLuint i, GLint x, GLint y, GLint z, GLint w) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    GLint v[4] = {x, y, z, w};
    std::memcpy(c->currentAttrib[i].data(), v, 16);
    c->currentAttribKind[i] = 1;
}
void VertexAttribI4ui(GLuint i, GLuint x, GLuint y, GLuint z, GLuint w) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    GLuint v[4] = {x, y, z, w};
    std::memcpy(c->currentAttrib[i].data(), v, 16);
    c->currentAttribKind[i] = 2;
}
void VertexAttribI4iv(GLuint i, const GLint* v) { VertexAttribI4i(i, v[0], v[1], v[2], v[3]); }
void VertexAttribI4uiv(GLuint i, const GLuint* v) { VertexAttribI4ui(i, v[0], v[1], v[2], v[3]); }
void GetVertexAttribiv(GLuint i, GLenum p, GLint* v) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    const VertexAttrib& a = c->currentVao()->attribs[i];
    switch (p) {
        case gl::VERTEX_ATTRIB_ARRAY_ENABLED: *v = a.enabled; return;
        case gl::VERTEX_ATTRIB_ARRAY_SIZE: *v = a.size; return;
        case gl::VERTEX_ATTRIB_ARRAY_STRIDE: *v = a.stride; return;
        case gl::VERTEX_ATTRIB_ARRAY_TYPE: *v = GLint(a.type); return;
        case gl::VERTEX_ATTRIB_ARRAY_NORMALIZED: *v = a.normalized; return;
        case gl::VERTEX_ATTRIB_ARRAY_INTEGER: *v = a.integer; return;
        case gl::VERTEX_ATTRIB_ARRAY_DIVISOR: *v = GLint(a.divisor); return;
        case gl::VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: *v = GLint(a.buffer); return;
        case gl::CURRENT_VERTEX_ATTRIB: for (int k = 0; k < 4; ++k) v[k] = GLint(c->currentAttrib[i][size_t(k)]); return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void GetVertexAttribfv(GLuint i, GLenum p, GLfloat* v) {
    CTX()
    if (p == gl::CURRENT_VERTEX_ATTRIB) {
        if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
        for (int k = 0; k < 4; ++k) v[k] = c->currentAttrib[i][size_t(k)];
        return;
    }
    GLint x[4] = {};
    GetVertexAttribiv(i, p, x);
    v[0] = GLfloat(x[0]);
}
void GetVertexAttribPointerv(GLuint i, GLenum, void** v) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    *v = const_cast<void*>(c->currentVao()->attribs[i].pointer);
}

// ---- draws ----------------------------------------------------------------------------------------------
void DrawArrays(GLenum m, GLint f, GLsizei n) {
    CTX()
    c->drawArrays(m, f, n, 1);
}
void DrawArraysInstanced(GLenum m, GLint f, GLsizei n, GLsizei inst) {
    CTX()
    c->drawArrays(m, f, n, inst);
}
void DrawElements(GLenum m, GLsizei n, GLenum t, const void* idx) {
    CTX()
    c->drawElements(m, n, t, idx, 1, 0);
}
void DrawElementsInstanced(GLenum m, GLsizei n, GLenum t, const void* idx, GLsizei inst) {
    CTX()
    c->drawElements(m, n, t, idx, inst, 0);
}
void DrawRangeElements(GLenum m, GLuint, GLuint, GLsizei n, GLenum t, const void* idx) {
    CTX()
    c->drawElements(m, n, t, idx, 1, 0);
}
void DrawElementsBaseVertex(GLenum m, GLsizei n, GLenum t, const void* idx, GLint bv) {
    CTX()
    c->drawElements(m, n, t, idx, 1, bv);
}
void DrawElementsInstancedBaseVertex(GLenum m, GLsizei n, GLenum t, const void* idx, GLsizei inst, GLint bv) {
    CTX()
    c->drawElements(m, n, t, idx, inst, bv);
}
void DrawRangeElementsBaseVertex(GLenum m, GLuint, GLuint, GLsizei n, GLenum t, const void* idx, GLint bv) {
    CTX()
    c->drawElements(m, n, t, idx, 1, bv);
}
void DispatchCompute(GLuint x, GLuint y, GLuint z) {
    CTX()
    c->dispatch(x, y, z);
}
void MemoryBarrier(GLbitfield) {
    CTX()
    c->barrier();
}

// ---- textures -------------------------------------------------------------------------------------------
void GenTextures(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->genName();
        auto t = std::make_unique<Texture>();
        t->name = name;
        c->share->textures[name] = std::move(t);
        out[i] = name;
    }
}
void DeleteTextures(GLsizei n, const GLuint* names) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint x = names[i];
        if (!x) continue;
        auto it = c->share->textures.find(x);
        if (it == c->share->textures.end()) continue;
        if (it->second->image) c->keep(it->second->image);
        for (auto& u : c->units)
            for (GLuint* b : {&u.tex2D, &u.texCube, &u.tex3D, &u.tex2DArray, &u.texExternal, &u.tex2DMS})
                if (*b == x) *b = 0;
        for (auto& [k, f] : c->fbos) {
            for (auto& a : f->color)
                if (a.kind == Attachment::Tex && a.name == x) a = {};
            if (f->depth.kind == Attachment::Tex && f->depth.name == x) f->depth = {};
            if (f->stencil.kind == Attachment::Tex && f->stencil.name == x) f->stencil = {};
        }
        c->share->textures.erase(it);
    }
}
GLboolean IsTexture(GLuint n) {
    CTX(0)
    Texture* t = c->texture(n);
    return t && t->target ? 1 : 0;
}
void ActiveTexture(GLenum u) {
    CTX()
    if (u < gl::TEXTURE0 || u >= gl::TEXTURE0 + kMaxTextureUnits) return c->setError(gl::INVALID_ENUM);
    c->activeUnit = int(u - gl::TEXTURE0);
}
void BindTexture(GLenum target, GLuint name) {
    CTX()
    GLuint* slot = c->textureBinding(target, c->activeUnit);
    if (!slot) return c->setError(gl::INVALID_ENUM);
    if (name) {
        Texture* t = c->texture(name);
        if (!t) {
            auto nt = std::make_unique<Texture>();
            nt->name = name;
            t = nt.get();
            c->share->textures[name] = std::move(nt);
            c->share->nextName = std::max(c->share->nextName, name + 1);
        }
        if (t->target && t->target != target) return c->setError(gl::INVALID_OPERATION);
        if (!t->target) {
            t->target = target;
            if (target == gl::TEXTURE_EXTERNAL_OES) {
                t->sampler.minFilter = gl::LINEAR;
                t->sampler.wrapS = t->sampler.wrapT = gl::CLAMP_TO_EDGE;
            }
        }
    }
    *slot = name;
}
void TexImage2D(GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLint border, GLenum fmt, GLenum type, const void* px) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE);
    c->texImage(target, level, GLenum(ifmt), w, h, 1, fmt, type, px, false, 0, 0, 0, false, 0);
}
void TexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLenum type, const void* px) {
    CTX()
    c->texImage(target, level, 0, w, h, 1, fmt, type, px, true, x, y, 0, false, 0);
}
void TexImage3D(GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLsizei d, GLint border, GLenum fmt, S type, const void* px) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE);
    c->texImage(target, level, GLenum(ifmt), w, h, d, fmt, GLenum(type), px, false, 0, 0, 0, false, 0);
}
void TexSubImage3D(GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, S fmt, S type, const void* px) {
    CTX()
    c->texImage(target, level, 0, w, h, d, GLenum(fmt), GLenum(type), px, true, x, y, z, false, 0);
}
void CompressedTexImage2D(GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLint border, GLsizei size, const void* data) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE);
    c->texImage(target, level, ifmt, w, h, 1, 0, 0, data, false, 0, 0, 0, true, size);
}
void CompressedTexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLsizei size, S data) {
    CTX()
    c->texImage(target, level, fmt, w, h, 1, 0, 0, reinterpret_cast<const void*>(data), true, x, y, 0, true, size);
}
void CompressedTexImage3D(GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d, GLint border, GLsizei size, S data) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE);
    c->texImage(target, level, ifmt, w, h, d, 0, 0, reinterpret_cast<const void*>(data), false, 0, 0, 0, true, size);
}
void CompressedTexSubImage3D(GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, S fmt, S size, S data) {
    CTX()
    c->texImage(target, level, GLenum(fmt), w, h, d, 0, 0, reinterpret_cast<const void*>(data), true, x, y, z, true, GLsizei(size));
}
void TexStorage2D(GLenum target, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h) {
    CTX()
    c->texStorage(target, levels, ifmt, w, h, target == gl::TEXTURE_CUBE_MAP ? 6 : 1, 0);
}
void TexStorage3D(GLenum target, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d) {
    CTX()
    c->texStorage(target, levels, ifmt, w, h, d, 0);
}
void TexStorage2DMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h, GLboolean) {
    CTX()
    c->texStorage(target == gl::TEXTURE_2D_MULTISAMPLE ? gl::TEXTURE_2D_MULTISAMPLE : target, 1, ifmt, w, h, 1, samples);
}
void TexStorage3DMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d, GLboolean) {
    CTX()
    c->texStorage(target, 1, ifmt, w, h, d, samples);
}
void CopyTexImage2D(GLenum target, GLint level, GLenum ifmt, GLint x, GLint y, GLsizei w, GLsizei h, GLint border) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE);
    c->copyTexImage(target, level, ifmt, x, y, w, h, false, 0, 0, 0);
}
void CopyTexSubImage2D(GLenum target, GLint level, GLint xo, GLint yo, GLint x, GLint y, GLsizei w, GLsizei h) {
    CTX()
    c->copyTexImage(target, level, 0, x, y, w, h, true, xo, yo, 0);
}
void CopyTexSubImage3D(GLenum target, GLint level, GLint xo, GLint yo, GLint zo, GLint x, GLint y, GLsizei w, S h) {
    CTX()
    c->copyTexImage(target, level, 0, x, y, w, GLsizei(h), true, xo, yo, zo);
}
void GenerateMipmap(GLenum target) {
    CTX()
    c->generateMipmap(target);
}
void texParam(Context* c, Texture* t, SamplerParams& s, GLenum p, double v, const double* vv) {
    switch (p) {
        case gl::TEXTURE_MIN_FILTER: s.minFilter = GLenum(v); return;
        case gl::TEXTURE_MAG_FILTER: s.magFilter = GLenum(v); return;
        case gl::TEXTURE_WRAP_S: s.wrapS = GLenum(v); return;
        case gl::TEXTURE_WRAP_T: s.wrapT = GLenum(v); return;
        case gl::TEXTURE_WRAP_R: s.wrapR = GLenum(v); return;
        case gl::TEXTURE_MIN_LOD: s.minLod = float(v); return;
        case gl::TEXTURE_MAX_LOD: s.maxLod = float(v); return;
        case gl::TEXTURE_COMPARE_MODE: s.compareMode = GLenum(v); return;
        case gl::TEXTURE_COMPARE_FUNC: s.compareFunc = GLenum(v); return;
        case gl::TEXTURE_MAX_ANISOTROPY_EXT: s.maxAnisotropy = std::clamp(float(v), 1.0f, 16.0f); return;
        case gl::TEXTURE_BORDER_COLOR: if (vv) for (int i = 0; i < 4; ++i) s.border[i] = float(vv[i]); return;
        default: break;
    }
    if (!t) return c->setError(gl::INVALID_ENUM);
    switch (p) {
        case gl::TEXTURE_BASE_LEVEL: t->baseLevel = GLint(v); return;
        case gl::TEXTURE_MAX_LEVEL: t->maxLevel = GLint(v); return;
        case gl::TEXTURE_SWIZZLE_R: case gl::TEXTURE_SWIZZLE_G: case gl::TEXTURE_SWIZZLE_B: case gl::TEXTURE_SWIZZLE_A:
            t->swizzle[p - gl::TEXTURE_SWIZZLE_R] = GLenum(v);
            return;
        case gl::DEPTH_STENCIL_TEXTURE_MODE: t->depthStencilMode = GLenum(v); return;
        case gl::TEXTURE_SRGB_DECODE_EXT: t->srgbDecode = GLenum(v) == gl::DECODE_EXT; return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void TexParameteri(GLenum target, GLenum p, GLint v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM);
    texParam(c, t, t->sampler, p, v, nullptr);
}
void TexParameterf(GLenum target, GLenum p, GLfloat v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM);
    texParam(c, t, t->sampler, p, v, nullptr);
}
void TexParameteriv(GLenum target, GLenum p, const GLint* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM);
    double vv[4] = {double(v[0]), 0, 0, 0};
    if (p == gl::TEXTURE_BORDER_COLOR)
        for (int i = 0; i < 4; ++i) vv[i] = v[i] / 2147483647.0;
    texParam(c, t, t->sampler, p, vv[0], vv);
}
void TexParameterfv(GLenum target, GLenum p, const GLfloat* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM);
    double vv[4] = {v[0], 0, 0, 0};
    if (p == gl::TEXTURE_BORDER_COLOR)
        for (int i = 0; i < 4; ++i) vv[i] = v[i];
    texParam(c, t, t->sampler, p, vv[0], vv);
}
void GetTexParameteriv(GLenum target, GLenum p, GLint* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM);
    const SamplerParams& s = t->sampler;
    switch (p) {
        case gl::TEXTURE_MIN_FILTER: *v = GLint(s.minFilter); return;
        case gl::TEXTURE_MAG_FILTER: *v = GLint(s.magFilter); return;
        case gl::TEXTURE_WRAP_S: *v = GLint(s.wrapS); return;
        case gl::TEXTURE_WRAP_T: *v = GLint(s.wrapT); return;
        case gl::TEXTURE_WRAP_R: *v = GLint(s.wrapR); return;
        case gl::TEXTURE_MIN_LOD: *v = GLint(s.minLod); return;
        case gl::TEXTURE_MAX_LOD: *v = GLint(s.maxLod); return;
        case gl::TEXTURE_COMPARE_MODE: *v = GLint(s.compareMode); return;
        case gl::TEXTURE_COMPARE_FUNC: *v = GLint(s.compareFunc); return;
        case gl::TEXTURE_BASE_LEVEL: *v = t->baseLevel; return;
        case gl::TEXTURE_MAX_LEVEL: *v = t->maxLevel; return;
        case gl::TEXTURE_IMMUTABLE_FORMAT: *v = t->immutable; return;
        case gl::TEXTURE_IMMUTABLE_LEVELS: *v = t->immutableLevels; return;
        case gl::TEXTURE_SWIZZLE_R: case gl::TEXTURE_SWIZZLE_G: case gl::TEXTURE_SWIZZLE_B: case gl::TEXTURE_SWIZZLE_A:
            *v = GLint(t->swizzle[p - gl::TEXTURE_SWIZZLE_R]);
            return;
        case gl::DEPTH_STENCIL_TEXTURE_MODE: *v = GLint(t->depthStencilMode); return;
        case gl::TEXTURE_MAX_ANISOTROPY_EXT: *v = GLint(s.maxAnisotropy); return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void GetTexParameterfv(GLenum target, GLenum p, GLfloat* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM);
    if (p == gl::TEXTURE_MIN_LOD) { *v = t->sampler.minLod; return; }
    if (p == gl::TEXTURE_MAX_LOD) { *v = t->sampler.maxLod; return; }
    if (p == gl::TEXTURE_MAX_ANISOTROPY_EXT) { *v = t->sampler.maxAnisotropy; return; }
    GLint x = 0;
    GetTexParameteriv(target, p, &x);
    *v = GLfloat(x);
}
void EGLImageTargetTexture2DOES(GLenum target, void* image) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t || (target != gl::TEXTURE_2D && target != gl::TEXTURE_EXTERNAL_OES)) return c->setError(gl::INVALID_ENUM);
    if (t->image) c->keep(t->image);
    if (!eglImageToTexture(image, *t)) c->setError(gl::INVALID_OPERATION);
}

// Sampler objects
void GenSamplers(GLsizei n, GLuint* out) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->genName();
        c->share->samplers[name] = std::make_unique<SamplerObj>();
        c->share->samplers[name]->name = name;
        out[i] = name;
    }
}
void DeleteSamplers(GLsizei n, const GLuint* names) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        for (auto& u : c->units)
            if (u.sampler == names[i]) u.sampler = 0;
        c->share->samplers.erase(names[i]);
    }
}
GLboolean IsSampler(GLuint n) {
    CTX(0)
    return c->share->samplers.count(n) ? 1 : 0;
}
void BindSampler(GLuint unit, GLuint s) {
    CTX()
    if (unit >= kMaxTextureUnits) return c->setError(gl::INVALID_VALUE);
    if (s && !c->share->samplers.count(s)) return c->setError(gl::INVALID_OPERATION);
    c->units[unit].sampler = s;
}
SamplerObj* samplerObj(Context* c, GLuint s) {
    auto it = c->share->samplers.find(s);
    if (it == c->share->samplers.end()) {
        c->setError(gl::INVALID_OPERATION);
        return nullptr;
    }
    return it->second.get();
}
void SamplerParameteri(GLuint s, GLenum p, GLint v) {
    CTX()
    if (SamplerObj* o = samplerObj(c, s)) texParam(c, nullptr, o->params, p, v, nullptr);
}
void SamplerParameterf(GLuint s, GLenum p, GLfloat v) {
    CTX()
    if (SamplerObj* o = samplerObj(c, s)) texParam(c, nullptr, o->params, p, v, nullptr);
}
void SamplerParameteriv(GLuint s, GLenum p, const GLint* v) { SamplerParameteri(s, p, v[0]); }
void SamplerParameterfv(GLuint s, GLenum p, const GLfloat* v) {
    CTX()
    double vv[4] = {v[0], 0, 0, 0};
    if (p == gl::TEXTURE_BORDER_COLOR)
        for (int i = 0; i < 4; ++i) vv[i] = v[i];
    if (SamplerObj* o = samplerObj(c, s)) texParam(c, nullptr, o->params, p, vv[0], vv);
}
void GetSamplerParameteriv(GLuint s, GLenum p, GLint* v) {
    CTX()
    SamplerObj* o = samplerObj(c, s);
    if (!o) return;
    switch (p) {
        case gl::TEXTURE_MIN_FILTER: *v = GLint(o->params.minFilter); return;
        case gl::TEXTURE_MAG_FILTER: *v = GLint(o->params.magFilter); return;
        case gl::TEXTURE_WRAP_S: *v = GLint(o->params.wrapS); return;
        case gl::TEXTURE_WRAP_T: *v = GLint(o->params.wrapT); return;
        case gl::TEXTURE_WRAP_R: *v = GLint(o->params.wrapR); return;
        case gl::TEXTURE_COMPARE_MODE: *v = GLint(o->params.compareMode); return;
        case gl::TEXTURE_COMPARE_FUNC: *v = GLint(o->params.compareFunc); return;
        case gl::TEXTURE_MIN_LOD: *v = GLint(o->params.minLod); return;
        case gl::TEXTURE_MAX_LOD: *v = GLint(o->params.maxLod); return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void GetSamplerParameterfv(GLuint s, GLenum p, GLfloat* v) {
    CTX()
    SamplerObj* o = samplerObj(c, s);
    if (!o) return;
    if (p == gl::TEXTURE_MIN_LOD) { *v = o->params.minLod; return; }
    if (p == gl::TEXTURE_MAX_LOD) { *v = o->params.maxLod; return; }
    GLint x = 0;
    GetSamplerParameteriv(s, p, &x);
    *v = GLfloat(x);
}

// ---- renderbuffers / framebuffers ------------------------------------------------------------------------
void GenRenderbuffers(GLsizei n, GLuint* out) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->genName();
        c->share->renderbuffers[name] = std::make_unique<Renderbuffer>();
        c->share->renderbuffers[name]->name = name;
        out[i] = name;
    }
}
void DeleteRenderbuffers(GLsizei n, const GLuint* names) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint x = names[i];
        auto it = c->share->renderbuffers.find(x);
        if (!x || it == c->share->renderbuffers.end()) continue;
        if (it->second->image) c->keep(it->second->image);
        if (c->renderbuffer == x) c->renderbuffer = 0;
        for (auto& [k, f] : c->fbos) {
            for (auto& a : f->color)
                if (a.kind == Attachment::Rbo && a.name == x) a = {};
            if (f->depth.kind == Attachment::Rbo && f->depth.name == x) f->depth = {};
            if (f->stencil.kind == Attachment::Rbo && f->stencil.name == x) f->stencil = {};
        }
        c->share->renderbuffers.erase(it);
    }
}
GLboolean IsRenderbuffer(GLuint n) {
    CTX(0)
    return c->rbo(n) && c->rbo(n)->image ? 1 : 0;
}
void BindRenderbuffer(GLenum target, GLuint n) {
    CTX()
    if (target != gl::RENDERBUFFER) return c->setError(gl::INVALID_ENUM);
    if (n && !c->rbo(n)) {
        c->share->renderbuffers[n] = std::make_unique<Renderbuffer>();
        c->share->renderbuffers[n]->name = n;
    }
    c->renderbuffer = n;
}
void RenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h) {
    CTX()
    if (target != gl::RENDERBUFFER) return c->setError(gl::INVALID_ENUM);
    Renderbuffer* r = c->rbo(c->renderbuffer);
    if (!r) return c->setError(gl::INVALID_OPERATION);
    if (w < 0 || h < 0 || w > 16384 || h > 16384 || samples < 0 || samples > 4) return c->setError(gl::INVALID_VALUE);
    const FormatInfo* f = formatFor(ifmt);
    if (!f || f->compressed || (!f->renderable && !(f->depth || f->stencil))) return c->setError(gl::INVALID_ENUM);
    if (r->image) c->keep(r->image);
    // Rendered 1x (VPKTAA does the AA); the renderbuffer reports the samples asked for.
    auto img = std::make_shared<Image>();
    img->dev = c->dev;
    img->format = f;
    img->width = uint32_t(std::max(w, 1)), img->height = uint32_t(std::max(h, 1));
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = f->vk, ci.extent = {img->width, img->height, 1};
    ci.mipLevels = 1, ci.arrayLayers = 1, ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = (f->depth || f->stencil ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (c->dev->vk.vkCreateImage(c->dev->device, &ci, nullptr, &img->image) != VK_SUCCESS) return c->setError(gl::OUT_OF_MEMORY);
    VkMemoryRequirements req;
    c->dev->vk.vkGetImageMemoryRequirements(c->dev->device, img->image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = uint32_t(c->dev->memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    c->dev->vk.vkAllocateMemory(c->dev->device, &mai, nullptr, &img->memory);
    c->dev->vk.vkBindImageMemory(c->dev->device, img->image, img->memory, 0);
    c->imageInit(*img);
    r->image = img;
    r->internalFormat = ifmt, r->width = w, r->height = h, r->samples = samples;
}
void RenderbufferStorage(GLenum target, GLenum ifmt, GLsizei w, GLsizei h) { RenderbufferStorageMultisample(target, 0, ifmt, w, h); }
void GetRenderbufferParameteriv(GLenum, GLenum p, GLint* v) {
    CTX()
    Renderbuffer* r = c->rbo(c->renderbuffer);
    if (!r) return c->setError(gl::INVALID_OPERATION);
    const FormatInfo* f = r->image ? r->image->format : nullptr;
    switch (p) {
        case gl::RENDERBUFFER_WIDTH: *v = r->width; return;
        case gl::RENDERBUFFER_HEIGHT: *v = r->height; return;
        case gl::RENDERBUFFER_INTERNAL_FORMAT: *v = GLint(r->internalFormat); return;
        case gl::RENDERBUFFER_SAMPLES: *v = r->samples; return;
        case 0x8D50: *v = f ? f->redBits : 0; return;
        case 0x8D51: *v = f ? f->greenBits : 0; return;
        case 0x8D52: *v = f ? f->blueBits : 0; return;
        case 0x8D53: *v = f ? f->alphaBits : 0; return;
        case 0x8D54: *v = f ? f->depthBits : 0; return;
        case 0x8D55: *v = f ? f->stencilBits : 0; return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void GetInternalformativ(GLenum, GLenum ifmt, GLenum p, GLsizei n, GLint* v) {
    CTX()
    if (n < 1) return;
    if (p == 0x80A9 /* SAMPLES */) {
        const FormatInfo* f = formatFor(ifmt);
        v[0] = f && !f->integer ? 4 : 0;
        if (n > 1) v[1] = f && !f->integer ? 2 : 0;
        return;
    }
    if (p == 0x9380 /* NUM_SAMPLE_COUNTS */) {
        const FormatInfo* f = formatFor(ifmt);
        v[0] = f && !f->integer ? 2 : 0;
        return;
    }
    c->setError(gl::INVALID_ENUM);
}
void GenFramebuffers(GLsizei n, GLuint* out) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->nextFbo++;
        c->fbos[name] = std::make_unique<Framebuffer>();
        c->fbos[name]->name = name;
        out[i] = name;
    }
}
void DeleteFramebuffers(GLsizei n, const GLuint* names) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        if (!names[i] || !c->fbos.count(names[i])) continue;
        c->endPass();
        if (c->drawFbo == names[i]) c->drawFbo = 0;
        if (c->readFbo == names[i]) c->readFbo = 0;
        c->fbos.erase(names[i]);
    }
}
GLboolean IsFramebuffer(GLuint n) {
    CTX(0)
    return n && c->fbos.count(n) ? 1 : 0;
}
void BindFramebuffer(GLenum target, GLuint n) {
    CTX()
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER)
        return c->setError(gl::INVALID_ENUM);
    if (n && !c->fbos.count(n)) {
        c->fbos[n] = std::make_unique<Framebuffer>();
        c->fbos[n]->name = n;
        c->nextFbo = std::max(c->nextFbo, n + 1);
    }
    if (target != gl::READ_FRAMEBUFFER) c->drawFbo = n;
    if (target != gl::DRAW_FRAMEBUFFER) c->readFbo = n;
}
Framebuffer* targetFbo(Context* c, GLenum target) {
    GLuint n = target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo;
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER) {
        c->setError(gl::INVALID_ENUM);
        return nullptr;
    }
    if (!n) {
        c->setError(gl::INVALID_OPERATION);
        return nullptr;
    }
    return c->fbo(n);
}
bool setAttachment(Context* c, Framebuffer* f, GLenum att, const Attachment& a) {
    c->endPass();
    if (att >= gl::COLOR_ATTACHMENT0 && att < gl::COLOR_ATTACHMENT0 + kMaxDrawBuffers) {
        f->color[att - gl::COLOR_ATTACHMENT0] = a;
        return true;
    }
    if (att == gl::DEPTH_ATTACHMENT) f->depth = a;
    else if (att == gl::STENCIL_ATTACHMENT) f->stencil = a;
    else if (att == gl::DEPTH_STENCIL_ATTACHMENT) f->depth = f->stencil = a;
    else {
        c->setError(gl::INVALID_ENUM);
        return false;
    }
    return true;
}
void FramebufferTexture2D(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    Attachment a;
    if (tex) {
        Texture* t = c->texture(tex);
        if (!t) return c->setError(gl::INVALID_OPERATION);
        a.kind = Attachment::Tex, a.name = tex, a.level = level;
        if (textarget >= gl::TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= gl::TEXTURE_CUBE_MAP_NEGATIVE_Z)
            a.layer = GLint(textarget - gl::TEXTURE_CUBE_MAP_POSITIVE_X);
    }
    setAttachment(c, f, att, a);
}
void FramebufferTextureLayer(GLenum target, GLenum att, GLuint tex, GLint level, GLint layer) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    Attachment a;
    if (tex) a.kind = Attachment::Tex, a.name = tex, a.level = level, a.layer = layer;
    setAttachment(c, f, att, a);
}
void FramebufferTexture(GLenum target, GLenum att, GLuint tex, GLint level) {
    FramebufferTextureLayer(target, att, tex, level, 0);
}
void FramebufferTextureMultiviewOVR(GLenum target, GLenum att, GLuint tex, GLint level, GLint baseView, GLsizei numViews) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    if (tex && (numViews < 1 || numViews > 2)) return c->setError(gl::INVALID_VALUE);
    Attachment a;
    if (tex) a.kind = Attachment::Tex, a.name = tex, a.level = level, a.baseView = baseView, a.numViews = numViews;
    setAttachment(c, f, att, a);
}
void FramebufferTextureMultisampleMultiviewOVR(GLenum target, GLenum att, GLuint tex, GLint level, GLsizei, GLint baseView, GLsizei numViews) {
    FramebufferTextureMultiviewOVR(target, att, tex, level, baseView, numViews);
}
void FramebufferTexture2DMultisampleEXT(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level, GLsizei) {
    FramebufferTexture2D(target, att, textarget, tex, level);  // 1x + VPKTAA
}
void FramebufferRenderbuffer(GLenum target, GLenum att, GLenum rbtarget, GLuint rb) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    if (rb && rbtarget != gl::RENDERBUFFER) return c->setError(gl::INVALID_ENUM);
    Attachment a;
    if (rb) {
        if (!c->rbo(rb)) return c->setError(gl::INVALID_OPERATION);
        a.kind = Attachment::Rbo, a.name = rb;
    }
    setAttachment(c, f, att, a);
}
GLenum CheckFramebufferStatus(GLenum target) {
    CTX(0)
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER) {
        c->setError(gl::INVALID_ENUM);
        return 0;
    }
    GLuint n = target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo;
    return c->resolveFramebuffer(n, target != gl::READ_FRAMEBUFFER).status;
}
void GetFramebufferAttachmentParameteriv(GLenum target, GLenum att, GLenum p, GLint* v) {
    CTX()
    GLuint n = target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo;
    if (n == 0) {
        // Default framebuffer: BACK / DEPTH / STENCIL.
        if (p == gl::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) { *v = GLint(gl::FRAMEBUFFER_DEFAULT); return; }
        Context::FbInfo fb = c->resolveFramebuffer(0, true);
        const Image* img = att == gl::BACK || att == gl::COLOR ? fb.color[0].image : fb.depth.image;
        const FormatInfo* fi = img ? img->format : nullptr;
        switch (p) {
            case gl::FRAMEBUFFER_ATTACHMENT_RED_SIZE: *v = fi ? fi->redBits : 0; return;
            case 0x8213: *v = fi ? fi->greenBits : 0; return;
            case 0x8214: *v = fi ? fi->blueBits : 0; return;
            case gl::FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: *v = fi ? fi->alphaBits : 0; return;
            case gl::FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: *v = fi ? fi->depthBits : 0; return;
            case gl::FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: *v = fi ? fi->stencilBits : 0; return;
            case gl::FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING: *v = GLint(gl::LINEAR_ENC); return;
            case gl::FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE: *v = fi ? GLint(fi->componentType) : 0; return;
            default: return c->setError(gl::INVALID_ENUM);
        }
    }
    Framebuffer* f = c->fbo(n);
    const Attachment* a = nullptr;
    if (att >= gl::COLOR_ATTACHMENT0 && att < gl::COLOR_ATTACHMENT0 + kMaxDrawBuffers) a = &f->color[att - gl::COLOR_ATTACHMENT0];
    else if (att == gl::DEPTH_ATTACHMENT || att == gl::DEPTH_STENCIL_ATTACHMENT) a = &f->depth;
    else if (att == gl::STENCIL_ATTACHMENT) a = &f->stencil;
    else return c->setError(gl::INVALID_ENUM);
    if (p == gl::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) {
        *v = a->kind == Attachment::None ? 0 : a->kind == Attachment::Tex ? GLint(gl::TEXTURE) : GLint(gl::RENDERBUFFER);
        return;
    }
    if (a->kind == Attachment::None) return c->setError(gl::INVALID_OPERATION);
    Context::FbTarget t;
    uint32_t w, h;
    const FormatInfo* fi = c->targetOf(*a, t, w, h) ? t.image->format : nullptr;
    switch (p) {
        case gl::FRAMEBUFFER_ATTACHMENT_OBJECT_NAME: *v = GLint(a->name); return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL: *v = a->level; return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE: *v = GLint(gl::TEXTURE_CUBE_MAP_POSITIVE_X) + a->layer; return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER: *v = a->layer; return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR: *v = a->numViews; return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR: *v = a->baseView; return;
        case gl::FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING: *v = fi && fi->srgb ? GLint(gl::SRGB) : GLint(gl::LINEAR_ENC); return;
        case gl::FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE: *v = fi ? GLint(fi->componentType) : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_RED_SIZE: *v = fi ? fi->redBits : 0; return;
        case 0x8213: *v = fi ? fi->greenBits : 0; return;
        case 0x8214: *v = fi ? fi->blueBits : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: *v = fi ? fi->alphaBits : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: *v = fi ? fi->depthBits : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: *v = fi ? fi->stencilBits : 0; return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void DrawBuffers(GLsizei n, const GLenum* bufs) {
    CTX()
    if (n < 0 || n > kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE);
    Framebuffer* f = c->fbo(c->drawFbo);
    c->endPass();
    if (c->drawFbo == 0) {
        if (n != 1 || (bufs[0] != gl::BACK && bufs[0] != gl::NONE)) return c->setError(gl::INVALID_OPERATION);
        f->drawBuffers[0] = bufs[0];
        return;
    }
    for (GLsizei i = 0; i < n; ++i)
        if (bufs[i] != gl::NONE && bufs[i] != gl::COLOR_ATTACHMENT0 + GLenum(i)) return c->setError(gl::INVALID_OPERATION);
    f->drawBuffers.fill(gl::NONE);
    for (GLsizei i = 0; i < n; ++i) f->drawBuffers[size_t(i)] = bufs[i];
}
void ReadBuffer(GLenum b) {
    CTX()
    Framebuffer* f = c->fbo(c->readFbo);
    if (c->readFbo == 0) {
        if (b != gl::BACK && b != gl::NONE) return c->setError(gl::INVALID_OPERATION);
    } else if (b != gl::NONE && (b < gl::COLOR_ATTACHMENT0 || b >= gl::COLOR_ATTACHMENT0 + kMaxDrawBuffers)) {
        return c->setError(gl::INVALID_OPERATION);
    }
    f->readBuffer = b;
}
void InvalidateFramebuffer(GLenum, GLsizei, const GLenum*) {}  // contents kept (load/store): valid, just slower
void InvalidateSubFramebuffer(GLenum, GLsizei, const GLenum*, GLint, GLint, GLsizei, GLsizei) {}
void DiscardFramebufferEXT(GLenum, GLsizei, const GLenum*) {}
void BlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, S mask, S filter) {
    CTX()
    c->blitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, GLbitfield(mask), GLenum(filter));
}
void ReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLenum type, void* px) {
    CTX()
    c->readPixels(x, y, w, h, fmt, type, px);
}
void ReadnPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLenum type, GLsizei, void* px) { ReadPixels(x, y, w, h, fmt, type, px); }

// ---- shaders / programs ------------------------------------------------------------------------------
GLuint CreateShader(GLenum type) {
    CTX(0)
    if (type != gl::VERTEX_SHADER && type != gl::FRAGMENT_SHADER && type != gl::COMPUTE_SHADER) {
        c->setError(gl::INVALID_ENUM);
        return 0;
    }
    GLuint n = c->genName();
    auto s = std::make_unique<Shader>();
    s->name = n, s->type = type;
    c->share->shaders[n] = std::move(s);
    return n;
}
Shader* shaderOrError(Context* c, GLuint n) {
    Shader* s = c->shader(n);
    if (!s) c->setError(c->prog(n) ? gl::INVALID_OPERATION : gl::INVALID_VALUE);
    return s;
}
Program* programOrError(Context* c, GLuint n) {
    Program* p = c->prog(n);
    if (!p) c->setError(c->shader(n) ? gl::INVALID_OPERATION : gl::INVALID_VALUE);
    return p;
}
void DeleteShader(GLuint n) {
    CTX()
    if (!n) return;
    Shader* s = shaderOrError(c, n);
    if (!s) return;
    if (s->attachedTo) s->deletePending = true;
    else c->share->shaders.erase(n);
}
GLboolean IsShader(GLuint n) {
    CTX(0)
    return c->shader(n) ? 1 : 0;
}
void ShaderSource(GLuint n, GLsizei count, const GLchar* const* str, const GLint* len) {
    CTX()
    Shader* s = shaderOrError(c, n);
    if (!s) return;
    if (count < 0) return c->setError(gl::INVALID_VALUE);
    s->source.clear();
    for (GLsizei i = 0; i < count; ++i) {
        if (!str[i]) continue;
        if (len && len[i] >= 0) s->source.append(str[i], size_t(len[i]));
        else s->source.append(str[i]);
    }
}
void CompileShader(GLuint n) {
    CTX()
    Shader* s = shaderOrError(c, n);
    if (!s) return;
    vpglsl::Stage st = s->type == gl::VERTEX_SHADER ? vpglsl::Stage::Vertex : s->type == gl::FRAGMENT_SHADER ? vpglsl::Stage::Fragment : vpglsl::Stage::Compute;
    vpglsl::ParseResult r = vpglsl::parse(st, s->source);
    s->parsed = r.shader;
    s->compiled = r.shader != nullptr;
    s->log = r.log;
    if (!s->compiled) logf("gles.shader compile_failed name=%u log=%s", n, r.log.c_str());
}
void GetShaderiv(GLuint n, GLenum p, GLint* v) {
    CTX()
    Shader* s = shaderOrError(c, n);
    if (!s) return;
    switch (p) {
        case gl::SHADER_TYPE: *v = GLint(s->type); return;
        case gl::DELETE_STATUS: *v = s->deletePending; return;
        case gl::COMPILE_STATUS: *v = s->compiled; return;
        case gl::INFO_LOG_LENGTH: *v = s->log.empty() ? 0 : GLint(s->log.size() + 1); return;
        case gl::SHADER_SOURCE_LENGTH: *v = s->source.empty() ? 0 : GLint(s->source.size() + 1); return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void copyString(const std::string& s, GLsizei max, GLsizei* len, GLchar* out) {
    if (max <= 0 || !out) {
        if (len) *len = 0;
        return;
    }
    GLsizei n = std::min<GLsizei>(max - 1, GLsizei(s.size()));
    std::memcpy(out, s.data(), size_t(n));
    out[n] = 0;
    if (len) *len = n;
}
void GetShaderInfoLog(GLuint n, GLsizei max, GLsizei* len, GLchar* log) {
    CTX()
    if (Shader* s = shaderOrError(c, n)) copyString(s->log, max, len, log);
}
void GetShaderSource(GLuint n, GLsizei max, GLsizei* len, GLchar* src) {
    CTX()
    if (Shader* s = shaderOrError(c, n)) copyString(s->source, max, len, src);
}
void GetShaderPrecisionFormat(GLenum, GLenum prec, GLint* range, GLint* precision) {
    if (prec >= gl::LOW_INT) range[0] = range[1] = 31, *precision = 0;
    else range[0] = range[1] = 127, *precision = 23;
}
void ReleaseShaderCompiler() {}
void ShaderBinary(GLsizei, const GLuint*, GLenum, const void*, GLsizei) {
    CTX()
    c->setError(gl::INVALID_ENUM);
}
GLuint CreateProgram() {
    CTX(0)
    GLuint n = c->genName();
    auto p = std::make_unique<Program>();
    p->name = n;
    c->share->programs[n] = std::move(p);
    return n;
}
void detachAll(Context* c, Program* p) {
    for (GLuint* s : {&p->vs, &p->fs, &p->cs})
        if (*s) {
            if (Shader* sh = c->shader(*s)) {
                if (--sh->attachedTo == 0 && sh->deletePending) c->share->shaders.erase(*s);
            }
            *s = 0;
        }
}
void DeleteProgram(GLuint n) {
    CTX()
    if (!n) return;
    Program* p = programOrError(c, n);
    if (!p) return;
    if (c->program == n) {
        p->deletePending = true;
        return;
    }
    c->flush(true);  // its pipelines may be in recorded work
    detachAll(c, p);
    c->share->programs.erase(n);
}
GLboolean IsProgram(GLuint n) {
    CTX(0)
    return c->prog(n) ? 1 : 0;
}
void AttachShader(GLuint pn, GLuint sn) {
    CTX()
    Program* p = programOrError(c, pn);
    Shader* s = shaderOrError(c, sn);
    if (!p || !s) return;
    GLuint* slot = s->type == gl::VERTEX_SHADER ? &p->vs : s->type == gl::FRAGMENT_SHADER ? &p->fs : &p->cs;
    if (*slot) return c->setError(gl::INVALID_OPERATION);
    *slot = sn;
    ++s->attachedTo;
}
void DetachShader(GLuint pn, GLuint sn) {
    CTX()
    Program* p = programOrError(c, pn);
    Shader* s = shaderOrError(c, sn);
    if (!p || !s) return;
    GLuint* slot = s->type == gl::VERTEX_SHADER ? &p->vs : s->type == gl::FRAGMENT_SHADER ? &p->fs : &p->cs;
    if (*slot != sn) return c->setError(gl::INVALID_OPERATION);
    *slot = 0;
    if (--s->attachedTo == 0 && s->deletePending) c->share->shaders.erase(sn);
}
void GetAttachedShaders(GLuint pn, GLsizei max, GLsizei* count, GLuint* out) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    GLsizei n = 0;
    for (GLuint s : {p->vs, p->fs, p->cs})
        if (s && n < max) out[n++] = s;
    if (count) *count = n;
}
void BindAttribLocation(GLuint pn, GLuint index, const GLchar* name) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (index >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    if (!std::strncmp(name, "gl_", 3)) return c->setError(gl::INVALID_OPERATION);
    p->attribBindings[name] = int(index);
}
void LinkProgram(GLuint pn) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (!c->linkProgram(*p)) logf("gles.program link_failed name=%u log=%s", pn, p->log.c_str());
}
void UseProgram(GLuint pn) {
    CTX()
    if (pn) {
        Program* p = programOrError(c, pn);
        if (!p) return;
        if (!p->linked) return c->setError(gl::INVALID_OPERATION);
    }
    Program* old = c->prog(c->program);
    c->program = pn;
    if (old && old->deletePending && old->name != pn) {
        c->flush(true);
        detachAll(c, old);
        c->share->programs.erase(old->name);
    }
}
void ValidateProgram(GLuint pn) {
    CTX()
    if (Program* p = programOrError(c, pn)) p->validated = p->linked;
}
void GetProgramiv(GLuint pn, GLenum q, GLint* v) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    auto maxLen = [](const auto& list) {
        GLint m = 0;
        for (const auto& x : list) m = std::max(m, GLint(x.name.size() + 1));
        return m;
    };
    switch (q) {
        case gl::DELETE_STATUS: *v = p->deletePending; return;
        case gl::LINK_STATUS: *v = p->linked; return;
        case gl::VALIDATE_STATUS: *v = p->validated; return;
        case gl::INFO_LOG_LENGTH: *v = p->log.empty() ? 0 : GLint(p->log.size() + 1); return;
        case gl::ATTACHED_SHADERS: *v = (p->vs ? 1 : 0) + (p->fs ? 1 : 0) + (p->cs ? 1 : 0); return;
        case gl::ACTIVE_ATTRIBUTES: *v = GLint(p->refl.attributes.size()); return;
        case gl::ACTIVE_ATTRIBUTE_MAX_LENGTH: *v = maxLen(p->refl.attributes); return;
        case gl::ACTIVE_UNIFORMS: {
            GLint n = GLint(p->refl.uniforms.size());
            for (const auto& b : p->refl.blocks)
                if (b.element == 0) n += GLint(b.members.size());
            *v = n;
            return;
        }
        case gl::ACTIVE_UNIFORM_MAX_LENGTH: {
            GLint m = maxLen(p->refl.uniforms);
            for (const auto& b : p->refl.blocks) m = std::max(m, maxLen(b.members));
            *v = m;
            return;
        }
        case gl::ACTIVE_UNIFORM_BLOCKS: *v = GLint(p->refl.blocks.size()); return;
        case gl::ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH: *v = maxLen(p->refl.blocks); return;
        case gl::PROGRAM_BINARY_LENGTH: *v = 0; return;
        case gl::PROGRAM_BINARY_RETRIEVABLE_HINT: *v = 0; return;
        case gl::TRANSFORM_FEEDBACK_VARYINGS: case gl::TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH: *v = 0; return;
        case gl::TRANSFORM_FEEDBACK_BUFFER_MODE: *v = 0x8C8D; return;
        case gl::COMPUTE_WORK_GROUP_SIZE: for (int i = 0; i < 3; ++i) v[i] = GLint(p->refl.localSize[size_t(i)]); return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void GetProgramInfoLog(GLuint pn, GLsizei max, GLsizei* len, GLchar* log) {
    CTX()
    if (Program* p = programOrError(c, pn)) copyString(p->log, max, len, log);
}
void ProgramParameteri(GLuint, GLenum, GLint) {}
void GetProgramBinary(GLuint, GLsizei, GLsizei* len, GLenum*, void*) {
    if (len) *len = 0;
}
void ProgramBinary(GLuint pn, GLenum, const void*, GLsizei) {
    CTX()
    // No binary formats: the program is left unlinked and the game recompiles from source.
    if (Program* p = programOrError(c, pn)) p->linked = false, p->log = "program binaries are not supported";
}
GLint GetAttribLocation(GLuint pn, const GLchar* name) {
    CTX(-1)
    Program* p = programOrError(c, pn);
    if (!p) return -1;
    if (!p->linked) {
        c->setError(gl::INVALID_OPERATION);
        return -1;
    }
    for (const auto& a : p->refl.attributes)
        if (a.name == name) return a.location;
    return -1;
}
GLint GetFragDataLocation(GLuint pn, const GLchar* name) {
    CTX(-1)
    Program* p = programOrError(c, pn);
    if (!p || !p->linked) return -1;
    for (const auto& o : p->refl.outputs)
        if (o.name == name) return o.location;
    return -1;
}
void GetActiveAttrib(GLuint pn, GLuint i, GLsizei max, GLsizei* len, GLint* size, GLenum* type, GLchar* name) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (i >= p->refl.attributes.size()) return c->setError(gl::INVALID_VALUE);
    const auto& a = p->refl.attributes[i];
    copyString(a.name, max, len, name);
    if (size) *size = a.arraySize;
    if (type) *type = a.glType;
}
// Active uniform list: default-block/sampler uniforms, then uniform block members.
const vpglsl::Variable* activeUniform(const Program& p, GLuint i, int* block) {
    if (i < p.refl.uniforms.size()) {
        *block = -1;
        return &p.refl.uniforms[i];
    }
    GLuint k = GLuint(p.refl.uniforms.size());
    for (size_t b = 0; b < p.refl.blocks.size(); ++b) {
        if (p.refl.blocks[b].element != 0) continue;
        for (const auto& m : p.refl.blocks[b].members) {
            if (k == i) {
                *block = int(b);
                return &m;
            }
            ++k;
        }
    }
    return nullptr;
}
void GetActiveUniform(GLuint pn, GLuint i, GLsizei max, GLsizei* len, GLint* size, GLenum* type, GLchar* name) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    int block;
    const vpglsl::Variable* u = activeUniform(*p, i, &block);
    if (!u) return c->setError(gl::INVALID_VALUE);
    copyString(u->name, max, len, name);
    if (size) *size = u->arraySize;
    if (type) *type = u->glType;
}
void GetActiveUniformsiv(GLuint pn, GLsizei n, const GLuint* idx, GLenum q, GLint* v) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    for (GLsizei k = 0; k < n; ++k) {
        int block;
        const vpglsl::Variable* u = activeUniform(*p, idx[k], &block);
        if (!u) return c->setError(gl::INVALID_VALUE);
        switch (q) {
            case gl::UNIFORM_TYPE: v[k] = GLint(u->glType); break;
            case gl::UNIFORM_SIZE: v[k] = u->arraySize; break;
            case gl::UNIFORM_NAME_LENGTH: v[k] = GLint(u->name.size() + 1); break;
            case gl::UNIFORM_BLOCK_INDEX: v[k] = block; break;
            case gl::UNIFORM_OFFSET: v[k] = block < 0 ? -1 : GLint(u->offset); break;
            case gl::UNIFORM_ARRAY_STRIDE: v[k] = block < 0 ? -1 : GLint(u->arrayStride); break;
            case gl::UNIFORM_MATRIX_STRIDE: v[k] = block < 0 ? -1 : GLint(u->matrixStride); break;
            case gl::UNIFORM_IS_ROW_MAJOR: v[k] = u->rowMajor; break;
            default: return c->setError(gl::INVALID_ENUM);
        }
    }
}
void GetUniformIndices(GLuint pn, GLsizei n, const GLchar* const* names, GLuint* out) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    for (GLsizei k = 0; k < n; ++k) {
        out[k] = gl::INVALID_INDEX;
        std::string want = names[k];
        for (GLuint i = 0;; ++i) {
            int block;
            const vpglsl::Variable* u = activeUniform(*p, i, &block);
            if (!u) break;
            std::string un = u->name;
            if (un == want || (u->isArray && un.size() > 3 && un.substr(0, un.size() - 3) == want)) {
                out[k] = i;
                break;
            }
        }
    }
}
GLint GetUniformLocation(GLuint pn, const GLchar* name) {
    CTX(-1)
    Program* p = programOrError(c, pn);
    if (!p) return -1;
    if (!p->linked) {
        c->setError(gl::INVALID_OPERATION);
        return -1;
    }
    return uniformLocation(*p, name);
}
GLuint GetUniformBlockIndex(GLuint pn, const GLchar* name) {
    CTX(gl::INVALID_INDEX)
    Program* p = programOrError(c, pn);
    if (!p) return gl::INVALID_INDEX;
    for (size_t i = 0; i < p->refl.blocks.size(); ++i)
        if (p->refl.blocks[i].name == name || (p->refl.blocks[i].element == 0 && p->refl.blocks[i].name + "[0]" == name))
            return GLuint(i);
    return gl::INVALID_INDEX;
}
void GetActiveUniformBlockiv(GLuint pn, GLuint i, GLenum q, GLint* v) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (i >= p->refl.blocks.size()) return c->setError(gl::INVALID_VALUE);
    const auto& b = p->refl.blocks[i];
    switch (q) {
        case gl::UNIFORM_BLOCK_BINDING: *v = GLint(p->blockBindings[i]); return;
        case gl::UNIFORM_BLOCK_DATA_SIZE: *v = GLint(b.size); return;
        case gl::UNIFORM_BLOCK_NAME_LENGTH: *v = GLint(b.name.size() + 1); return;
        case gl::UNIFORM_BLOCK_ACTIVE_UNIFORMS: *v = GLint(b.members.size()); return;
        case gl::UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES: {
            GLuint k = GLuint(p->refl.uniforms.size());
            for (size_t x = 0; x < i; ++x)
                if (p->refl.blocks[x].element == 0) k += GLuint(p->refl.blocks[x].members.size());
            for (size_t m = 0; m < b.members.size(); ++m) v[m] = GLint(k + m);
            return;
        }
        case gl::UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER: *v = (b.stages & 1) != 0; return;
        case gl::UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: *v = (b.stages & 2) != 0; return;
        default: c->setError(gl::INVALID_ENUM);
    }
}
void GetActiveUniformBlockName(GLuint pn, GLuint i, GLsizei max, GLsizei* len, GLchar* name) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (i >= p->refl.blocks.size()) return c->setError(gl::INVALID_VALUE);
    copyString(p->refl.blocks[i].name, max, len, name);
}
void UniformBlockBinding(GLuint pn, GLuint i, GLuint binding) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (i >= p->refl.blocks.size() || binding >= kMaxUniformBufferBindings) return c->setError(gl::INVALID_VALUE);
    p->blockBindings[i] = binding;
}

using K = Context::Kind;
#define UNI(name, comps, kind, T)                                                          \
    void Uniform##name(GLint loc, GLsizei n, const T* v) {                                 \
        CTX()                                                                              \
        c->setUniform(loc, comps, kind, n, v, 0, false);                                   \
    }
UNI(1fv, 1, K::Float, GLfloat) UNI(2fv, 2, K::Float, GLfloat) UNI(3fv, 3, K::Float, GLfloat) UNI(4fv, 4, K::Float, GLfloat)
UNI(1iv, 1, K::Int, GLint) UNI(2iv, 2, K::Int, GLint) UNI(3iv, 3, K::Int, GLint) UNI(4iv, 4, K::Int, GLint)
UNI(1uiv, 1, K::Uint, GLuint) UNI(2uiv, 2, K::Uint, GLuint) UNI(3uiv, 3, K::Uint, GLuint) UNI(4uiv, 4, K::Uint, GLuint)
#undef UNI
void Uniform1f(GLint l, GLfloat x) { Uniform1fv(l, 1, &x); }
void Uniform2f(GLint l, GLfloat x, GLfloat y) { GLfloat v[] = {x, y}; Uniform2fv(l, 1, v); }
void Uniform3f(GLint l, GLfloat x, GLfloat y, GLfloat z) { GLfloat v[] = {x, y, z}; Uniform3fv(l, 1, v); }
void Uniform4f(GLint l, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { GLfloat v[] = {x, y, z, w}; Uniform4fv(l, 1, v); }
void Uniform1i(GLint l, GLint x) { Uniform1iv(l, 1, &x); }
void Uniform2i(GLint l, GLint x, GLint y) { GLint v[] = {x, y}; Uniform2iv(l, 1, v); }
void Uniform3i(GLint l, GLint x, GLint y, GLint z) { GLint v[] = {x, y, z}; Uniform3iv(l, 1, v); }
void Uniform4i(GLint l, GLint x, GLint y, GLint z, GLint w) { GLint v[] = {x, y, z, w}; Uniform4iv(l, 1, v); }
void Uniform1ui(GLint l, GLuint x) { Uniform1uiv(l, 1, &x); }
void Uniform2ui(GLint l, GLuint x, GLuint y) { GLuint v[] = {x, y}; Uniform2uiv(l, 1, v); }
void Uniform3ui(GLint l, GLuint x, GLuint y, GLuint z) { GLuint v[] = {x, y, z}; Uniform3uiv(l, 1, v); }
void Uniform4ui(GLint l, GLuint x, GLuint y, GLuint z, GLuint w) { GLuint v[] = {x, y, z, w}; Uniform4uiv(l, 1, v); }
#define UMAT(name, cols, rows)                                                         \
    void UniformMatrix##name##fv(GLint loc, GLsizei n, GLboolean t, const GLfloat* v) { \
        CTX()                                                                          \
        c->setUniform(loc, rows, K::Float, n, v, cols, t);                             \
    }
UMAT(2, 2, 2) UMAT(3, 3, 3) UMAT(4, 4, 4) UMAT(2x3, 2, 3) UMAT(3x2, 3, 2) UMAT(2x4, 2, 4) UMAT(4x2, 4, 2) UMAT(3x4, 3, 4) UMAT(4x3, 4, 3)
#undef UMAT
void GetUniformfv(GLuint pn, GLint loc, GLfloat* v) {
    CTX()
    if (Program* p = programOrError(c, pn)) c->getUniform(*p, loc, v, K::Float);
}
void GetUniformiv(GLuint pn, GLint loc, GLint* v) {
    CTX()
    if (Program* p = programOrError(c, pn)) c->getUniform(*p, loc, v, K::Int);
}
void GetUniformuiv(GLuint pn, GLint loc, GLuint* v) {
    CTX()
    if (Program* p = programOrError(c, pn)) c->getUniform(*p, loc, v, K::Uint);
}

// ---- queries / sync ---------------------------------------------------------------------------------------
void GenQueries(GLsizei n, GLuint* out) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->nextQuery++;
        c->queries[name] = std::make_unique<Query>();
        c->queries[name]->name = name;
        out[i] = name;
    }
}
void DeleteQueries(GLsizei n, const GLuint* names) {
    CTX()
    for (GLsizei i = 0; i < n; ++i) c->queries.erase(names[i]);
}
GLboolean IsQuery(GLuint n) {
    CTX(0)
    return c->queries.count(n) ? 1 : 0;
}
void BeginQuery(GLenum target, GLuint n) {
    CTX()
    auto it = c->queries.find(n);
    if (it == c->queries.end()) return c->setError(gl::INVALID_OPERATION);
    // Occlusion queries answer "samples passed" (conservatively true): culling code stays correct.
    it->second->target = target, it->second->active = true, it->second->result = 1;
}
void EndQuery(GLenum) {
    CTX()
    for (auto& [k, q] : c->queries) q->active = false;
}
void GetQueryiv(GLenum, GLenum, GLint* v) { *v = 0; }
void GetQueryObjectuiv(GLuint n, GLenum p, GLuint* v) {
    CTX()
    auto it = c->queries.find(n);
    if (it == c->queries.end()) return c->setError(gl::INVALID_OPERATION);
    *v = p == gl::QUERY_RESULT_AVAILABLE ? 1 : GLuint(it->second->result);
}
void GetQueryObjectui64vEXT(GLuint n, GLenum p, GLuint64* v) {
    GLuint x = 0;
    GetQueryObjectuiv(n, p, &x);
    *v = x;
}
GLsync FenceSync(GLenum cond, GLbitfield) {
    CTX(nullptr)
    if (cond != gl::SYNC_GPU_COMMANDS_COMPLETE) {
        c->setError(gl::INVALID_ENUM);
        return nullptr;
    }
    auto s = std::make_unique<Sync>();
    s->serial = c->flush(false);
    if (!s->serial) s->serial = c->dev->submitted;
    GLsync h = s.get();
    c->share->syncs[h] = std::move(s);
    return h;
}
GLboolean IsSync(GLsync s) {
    CTX(0)
    return c->share->syncs.count(s) ? 1 : 0;
}
void DeleteSync(GLsync s) {
    CTX()
    c->share->syncs.erase(s);
}
GLenum ClientWaitSync(GLsync s, GLbitfield flags, GLuint64 timeout) {
    CTX(gl::WAIT_FAILED)
    auto it = c->share->syncs.find(s);
    if (it == c->share->syncs.end()) {
        c->setError(gl::INVALID_VALUE);
        return gl::WAIT_FAILED;
    }
    if (flags & gl::SYNC_FLUSH_COMMANDS_BIT) c->flush(false);
    c->dev->retire(false);
    if (c->dev->completed >= it->second->serial) return gl::ALREADY_SIGNALED;
    if (!timeout) return gl::TIMEOUT_EXPIRED;
    c->dev->waitSerial(it->second->serial);
    return gl::CONDITION_SATISFIED;
}
void WaitSync(GLsync, GLbitfield, GLuint64) {}  // one queue: already ordered
void GetSynciv(GLsync s, GLenum p, GLsizei, GLsizei* len, GLint* v) {
    CTX()
    auto it = c->share->syncs.find(s);
    if (it == c->share->syncs.end()) return c->setError(gl::INVALID_VALUE);
    c->dev->retire(false);
    switch (p) {
        case gl::OBJECT_TYPE: *v = GLint(gl::SYNC_FENCE); break;
        case gl::SYNC_STATUS: *v = GLint(c->dev->completed >= it->second->serial ? gl::SIGNALED : gl::UNSIGNALED); break;
        case gl::SYNC_CONDITION: *v = GLint(gl::SYNC_GPU_COMMANDS_COMPLETE); break;
        case gl::SYNC_FLAGS: *v = 0; break;
        default: return c->setError(gl::INVALID_ENUM);
    }
    if (len) *len = 1;
}

// ---- debug / no-ops -----------------------------------------------------------------------------------------
void DebugMessageCallback(void*, const void*) {}
void DebugMessageControl(GLenum, GLenum, GLenum, GLsizei, const GLuint*, GLboolean) {}
void DebugMessageInsert(GLenum, GLenum, GLuint, GLenum, GLsizei, const GLchar*) {}
void PushDebugGroup(GLenum, GLuint, GLsizei, const GLchar*) {}
void PopDebugGroup() {}
void ObjectLabel(GLenum, GLuint, GLsizei, const GLchar*) {}
void InsertEventMarkerEXT(GLsizei, const GLchar*) {}
void PushGroupMarkerEXT(GLsizei, const GLchar*) {}
void PopGroupMarkerEXT() {}
GLenum GetGraphicsResetStatus() { return gl::NO_ERROR; }
void PauseTransformFeedback() {}
void ResumeTransformFeedback() {}
void BeginTransformFeedback(GLenum) {
    CTX()
    c->setError(gl::INVALID_OPERATION);
}
void EndTransformFeedback() {}
void GenTransformFeedbacks(GLsizei n, GLuint* out) {
    for (GLsizei i = 0; i < n; ++i) out[i] = GLuint(i + 1);
}
void DeleteTransformFeedbacks(GLsizei, const GLuint*) {}
void BindTransformFeedback(GLenum, GLuint) {}
void TransformFeedbackVaryings(GLuint, GLsizei, const GLchar* const*, GLenum) {}

struct Entry {
    const char* name;
    void* fn;
};
#define E(n) {"gl" #n, reinterpret_cast<void*>(&n)}
const Entry kEntries[] = {
    E(GetError), E(GetString), E(GetStringi), E(GetIntegerv), E(GetInteger64v), E(GetFloatv), E(GetBooleanv),
    E(GetIntegeri_v), E(GetInteger64i_v), E(GetBooleani_v), E(Enable), E(Disable), E(IsEnabled), E(Enablei),
    E(Disablei), E(Viewport), E(Scissor), E(DepthRangef), E(ClearColor), E(ClearDepthf), E(ClearStencil), E(Clear),
    E(ClearBufferfv), E(ClearBufferiv), E(ClearBufferuiv), E(ClearBufferfi), E(ColorMask), E(ColorMaski),
    E(DepthMask), E(DepthFunc), E(CullFace), E(FrontFace), E(LineWidth), E(PolygonOffset), E(SampleCoverage),
    E(Hint), E(BlendFunc), E(BlendFuncSeparate), E(BlendFunci), E(BlendFuncSeparatei), E(BlendEquation),
    E(BlendEquationSeparate), E(BlendEquationi), E(BlendEquationSeparatei), E(BlendColor), E(StencilFunc),
    E(StencilFuncSeparate), E(StencilMask), E(StencilMaskSeparate), E(StencilOp), E(StencilOpSeparate),
    E(PixelStorei), E(Flush), E(Finish), E(GenBuffers), E(DeleteBuffers), E(IsBuffer), E(BindBuffer),
    E(BindBufferRange), E(BindBufferBase), E(BufferData), E(BufferSubData), E(MapBufferRange), E(MapBufferOES),
    E(UnmapBuffer), E(FlushMappedBufferRange), E(GetBufferParameteriv), E(GetBufferParameteri64v),
    E(GetBufferPointerv), E(CopyBufferSubData), E(GenVertexArrays), E(DeleteVertexArrays), E(BindVertexArray),
    E(IsVertexArray), E(EnableVertexAttribArray), E(DisableVertexAttribArray), E(VertexAttribPointer),
    E(VertexAttribIPointer), E(VertexAttribDivisor), E(VertexAttrib1f), E(VertexAttrib2f), E(VertexAttrib3f),
    E(VertexAttrib4f), E(VertexAttrib1fv), E(VertexAttrib2fv), E(VertexAttrib3fv), E(VertexAttrib4fv),
    E(VertexAttribI4i), E(VertexAttribI4ui), E(VertexAttribI4iv), E(VertexAttribI4uiv), E(GetVertexAttribiv),
    E(GetVertexAttribfv), E(GetVertexAttribPointerv), E(DrawArrays), E(DrawArraysInstanced), E(DrawElements),
    E(DrawElementsInstanced), E(DrawRangeElements), E(DrawElementsBaseVertex), E(DrawElementsInstancedBaseVertex),
    E(DrawRangeElementsBaseVertex), E(DispatchCompute), E(MemoryBarrier), E(GenTextures), E(DeleteTextures),
    E(IsTexture), E(ActiveTexture), E(BindTexture), E(TexImage2D), E(TexSubImage2D), E(TexImage3D),
    E(TexSubImage3D), E(CompressedTexImage2D), E(CompressedTexSubImage2D), E(CompressedTexImage3D),
    E(CompressedTexSubImage3D), E(TexStorage2D), E(TexStorage3D), E(TexStorage2DMultisample),
    E(TexStorage3DMultisample), E(CopyTexImage2D), E(CopyTexSubImage2D), E(CopyTexSubImage3D), E(GenerateMipmap),
    E(TexParameteri), E(TexParameterf), E(TexParameteriv), E(TexParameterfv), E(GetTexParameteriv),
    E(GetTexParameterfv), E(EGLImageTargetTexture2DOES), E(GenSamplers), E(DeleteSamplers), E(IsSampler),
    E(BindSampler), E(SamplerParameteri), E(SamplerParameterf), E(SamplerParameteriv), E(SamplerParameterfv),
    E(GetSamplerParameteriv), E(GetSamplerParameterfv), E(GenRenderbuffers), E(DeleteRenderbuffers),
    E(IsRenderbuffer), E(BindRenderbuffer), E(RenderbufferStorage), E(RenderbufferStorageMultisample),
    E(GetRenderbufferParameteriv), E(GetInternalformativ), E(GenFramebuffers), E(DeleteFramebuffers),
    E(IsFramebuffer), E(BindFramebuffer), E(FramebufferTexture2D), E(FramebufferTextureLayer), E(FramebufferTexture),
    E(FramebufferTextureMultiviewOVR), E(FramebufferTextureMultisampleMultiviewOVR),
    E(FramebufferTexture2DMultisampleEXT), E(FramebufferRenderbuffer), E(CheckFramebufferStatus),
    E(GetFramebufferAttachmentParameteriv), E(DrawBuffers), E(ReadBuffer), E(InvalidateFramebuffer),
    E(InvalidateSubFramebuffer), E(DiscardFramebufferEXT), E(BlitFramebuffer), E(ReadPixels), E(ReadnPixels),
    E(CreateShader), E(DeleteShader), E(IsShader), E(ShaderSource), E(CompileShader), E(GetShaderiv),
    E(GetShaderInfoLog), E(GetShaderSource), E(GetShaderPrecisionFormat), E(ReleaseShaderCompiler), E(ShaderBinary),
    E(CreateProgram), E(DeleteProgram), E(IsProgram), E(AttachShader), E(DetachShader), E(GetAttachedShaders),
    E(BindAttribLocation), E(LinkProgram), E(UseProgram), E(ValidateProgram), E(GetProgramiv), E(GetProgramInfoLog),
    E(ProgramParameteri), E(GetProgramBinary), E(ProgramBinary), E(GetAttribLocation), E(GetFragDataLocation),
    E(GetActiveAttrib), E(GetActiveUniform), E(GetActiveUniformsiv), E(GetUniformIndices), E(GetUniformLocation),
    E(GetUniformBlockIndex), E(GetActiveUniformBlockiv), E(GetActiveUniformBlockName), E(UniformBlockBinding),
    E(Uniform1f), E(Uniform2f), E(Uniform3f), E(Uniform4f), E(Uniform1i), E(Uniform2i), E(Uniform3i), E(Uniform4i),
    E(Uniform1ui), E(Uniform2ui), E(Uniform3ui), E(Uniform4ui), E(Uniform1fv), E(Uniform2fv), E(Uniform3fv),
    E(Uniform4fv), E(Uniform1iv), E(Uniform2iv), E(Uniform3iv), E(Uniform4iv), E(Uniform1uiv), E(Uniform2uiv),
    E(Uniform3uiv), E(Uniform4uiv), E(UniformMatrix2fv), E(UniformMatrix3fv), E(UniformMatrix4fv),
    E(UniformMatrix2x3fv), E(UniformMatrix3x2fv), E(UniformMatrix2x4fv), E(UniformMatrix4x2fv),
    E(UniformMatrix3x4fv), E(UniformMatrix4x3fv), E(GetUniformfv), E(GetUniformiv), E(GetUniformuiv),
    E(GenQueries), E(DeleteQueries), E(IsQuery), E(BeginQuery), E(EndQuery), E(GetQueryiv), E(GetQueryObjectuiv),
    E(GetQueryObjectui64vEXT), E(FenceSync), E(IsSync), E(DeleteSync), E(ClientWaitSync), E(WaitSync), E(GetSynciv),
    E(DebugMessageCallback), E(DebugMessageControl), E(DebugMessageInsert), E(PushDebugGroup), E(PopDebugGroup),
    E(ObjectLabel), E(InsertEventMarkerEXT), E(PushGroupMarkerEXT), E(PopGroupMarkerEXT), E(GetGraphicsResetStatus),
    E(PauseTransformFeedback), E(ResumeTransformFeedback), E(BeginTransformFeedback), E(EndTransformFeedback),
    E(GenTransformFeedbacks), E(DeleteTransformFeedbacks), E(BindTransformFeedback), E(TransformFeedbackVaryings),
};
#undef E

}  // namespace

// Extension-suffixed names (OES/EXT/KHR/ANGLE/NV) resolve to the core function of the base name, except
// those whose suffix is part of their only name (in the table above).
void* glProc(const char* name) {
    for (const Entry& e : kEntries)
        if (!std::strcmp(e.name, name)) return e.fn;
    static const char* const kSuffixes[] = {"OES", "EXT", "KHR", "ANGLE", "NV", "ARB"};
    const size_t len = std::strlen(name);
    for (const char* s : kSuffixes) {
        const size_t sl = std::strlen(s);
        if (len > sl && !std::strcmp(name + len - sl, s)) {
            std::string base(name, len - sl);
            for (const Entry& e : kEntries)
                if (base == e.name) return e.fn;
        }
    }
    if (!std::strcmp(name, "glGetQueryObjectuivEXT")) return reinterpret_cast<void*>(&GetQueryObjectuiv);
    return nullptr;
}

size_t glEntryCount() { return sizeof kEntries / sizeof kEntries[0]; }

}  // namespace vpgles

// ---- host API --------------------------------------------------------------------------------------------
using namespace vpgles;

namespace {
Device* gDev = nullptr;
int gInitResult = -1;
}  // namespace

extern "C" int vpgles_init(void* gipa) {
    if (gInitResult >= 0) return gInitResult;
    auto* d = new Device;
    if (!d->init(reinterpret_cast<PFN_vkGetInstanceProcAddr>(gipa))) {
        logf("gles.err what=init (Vulkan device)");
        delete d;
        gInitResult = 0;
        return 0;
    }
    gDev = d;
    setDevice(d);
    gInitResult = 1;
    return 1;
}

extern "C" void vpgles_shutdown(void) {
    if (!gDev) return;
    gDev->shutdown();
    gInitResult = -1;
}

extern "C" void* vpgles_proc(const char* name) {
    if (!name) return nullptr;
    if (!std::strncmp(name, "egl", 3)) return eglProc(name);
    if (!std::strncmp(name, "gl", 2)) return glProc(name);
    return nullptr;
}

extern "C" void vpgles_device(void** instance, void** phys, void** dev, void** queue) {
    Device* d = device();
    if (instance) *instance = d ? d->instance : nullptr;
    if (phys) *phys = d ? d->phys : nullptr;
    if (dev) *dev = d ? d->device : nullptr;
    if (queue) *queue = d ? d->queue : nullptr;
}

static uint32_t wrapImageAs(uint32_t forceName, void* vkImage, uint32_t vkFormat, uint32_t target, uint32_t w, uint32_t h,
                            uint32_t layers, uint32_t levels) {
    if (!device() || !vkImage) return 0;
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    Context* c = currentOrLast();
    if (!c) return 0;
    const FormatInfo* f = formatByVk(VkFormat(vkFormat));
    if (!f) {
        logf("gles.err what=wrap_image format=%u", vkFormat);
        return 0;
    }
    auto img = std::make_shared<Image>();
    img->dev = c->dev;
    img->image = reinterpret_cast<VkImage>(vkImage);
    img->external = true;
    img->format = f;
    img->width = w, img->height = h, img->layers = std::max(1u, layers), img->levels = std::max(1u, levels);
    GLuint name = forceName ? forceName : c->genName();
    auto t = std::make_unique<Texture>();
    t->name = name;
    t->target = target;
    t->image = img;
    t->immutable = true;
    t->immutableLevels = GLint(img->levels);
    for (uint32_t lv = 0; lv < img->levels; ++lv)
        t->levels.push_back({std::max(1u, w >> lv), std::max(1u, h >> lv), target == gl::TEXTURE_2D_ARRAY ? img->layers : 1, f});
    t->sampler.minFilter = gl::LINEAR;
    c->share->textures[name] = std::move(t);
    c->imageInit(*img);
    if (forceName) logf("gles legacy eye texture name=%u wrapped %ux%u layers=%u target=%#x", name, w, h, img->layers, target);
    return name;
}

extern "C" uint32_t vpgles_wrap_image(void* vkImage, uint32_t vkFormat, uint32_t target, uint32_t w, uint32_t h,
                                      uint32_t layers, uint32_t levels) {
    return wrapImageAs(0, vkImage, vkFormat, target, w, h, layers, levels);
}

// Same as vpgles_wrap_image, but the storage replaces that of the app's own texture `name` (OVRPlugin's
// legacy SetupEyeTexture2: Unity owns the name, the plugin provides the storage).
extern "C" uint32_t vpgles_wrap_image_as(uint32_t name, void* vkImage, uint32_t vkFormat, uint32_t target, uint32_t w,
                                         uint32_t h, uint32_t layers, uint32_t levels) {
    return name ? wrapImageAs(name, vkImage, vkFormat, target, w, h, layers, levels) : 0;
}

extern "C" void vpgles_flush(int wait) {
    if (!device()) return;
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    Context* c = currentOrLast();
    if (!c) return;
    c->flush(wait != 0);
}
