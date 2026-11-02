// The GLES 2.0 / 3.0 / 3.1 (+ OES/EXT/OVR extensions games use) entry points.
//
// ABI: the guest is AAPCS64 and the visor build is Darwin arm64. They agree on x0-x7/v0-v7 and on
// 8-byte stack arguments, and disagree on narrower stack arguments (AAPCS64: one 8-byte slot
// each; Darwin: packed to their size). Every integer parameter past the eighth is therefore
// declared 64-bit here (`S`) and truncated, so both read the slot the caller wrote.
#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <chrono>
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
    "GL_OES_element_index_uint", "GL_OES_packed_depth_stencil", "GL_OES_depth_texture", "GL_OES_depth24", "GL_OES_depth32",
    "GL_OES_rgb8_rgba8", "GL_OES_standard_derivatives", "GL_OES_texture_npot", "GL_OES_vertex_array_object",
    "GL_OES_EGL_image", "GL_OES_EGL_image_external", "GL_OES_EGL_image_external_essl3", "GL_OES_texture_half_float",
    "GL_OES_texture_half_float_linear", "GL_OES_texture_float", "GL_OES_texture_float_linear",
    "GL_OES_compressed_ETC1_RGB8_texture", "GL_OES_vertex_half_float", "GL_OES_texture_3D", "GL_OES_surfaceless_context",
    "GL_OES_mapbuffer", "GL_EXT_texture_format_BGRA8888", "GL_EXT_read_format_bgra",
    "GL_EXT_texture_filter_anisotropic", "GL_EXT_shader_texture_lod", "GL_EXT_color_buffer_float",
    "GL_EXT_color_buffer_half_float", "GL_EXT_sRGB", "GL_EXT_texture_sRGB_decode", "GL_EXT_discard_framebuffer",
    "GL_EXT_multisampled_render_to_texture", "GL_EXT_multisampled_render_to_texture2", "GL_EXT_draw_buffers",
    "GL_EXT_frag_depth", "GL_EXT_shadow_samplers", "GL_EXT_texture_storage", "GL_EXT_map_buffer_range",
    "GL_EXT_occlusion_query_boolean", "GL_EXT_instanced_arrays", "GL_EXT_draw_instanced",
    "GL_EXT_texture_rg", "GL_EXT_unpack_subimage", "GL_EXT_debug_marker", "GL_EXT_debug_label", "GL_EXT_robustness",
    "GL_KHR_texture_compression_astc_ldr", "GL_KHR_debug", "GL_OVR_multiview", "GL_OVR_multiview2",
    "GL_OVR_multiview_multisampled_render_to_texture", "GL_ANGLE_instanced_arrays", "GL_NV_draw_buffers",
    "GL_EXT_texture_buffer", "GL_EXT_copy_image", "GL_EXT_draw_buffers_indexed", "GL_OES_get_program_binary",
};
std::string extensionString() {
    std::string s;
    for (const char* e : kExtensions) s += std::string(e) + " ";
    return s;
}

// "OpenGL ES 3.1 ..." / "OpenGL ES GLSL ES 3.10", from the advertised level (gl_enums31.h).
const std::string& versionString() {
    static const std::string v =
        "OpenGL ES " + std::to_string(gl::kAdvertisedMajor) + "." + std::to_string(gl::kAdvertisedMinor) + " V@0615.0 (VPKEngine)";
    return v;
}
const std::string& glslVersionString() {
    static const std::string v = std::string("OpenGL ES GLSL ES ") + std::to_string(gl::kAdvertisedMajor) + "." +
                                 std::to_string(gl::kAdvertisedMinor) + "0";
    return v;
}

// GL_VENDOR / GL_RENDERER follow the Vulkan device's identity (Adreno 740 with VPVulkan's Quest identity, else the Apple
// GPU's own name), asked on every call: the host may choose the identity after vpgles_init. Lock held.
const GLubyte* identityString(bool vendor) {
    Device* d = device();
    VkPhysicalDeviceProperties p{};
    d->vk.vkGetPhysicalDeviceProperties(d->phys, &p);
    if (p.vendorID == 0x5143) return reinterpret_cast<const GLubyte*>(vendor ? "Qualcomm" : "Adreno (TM) 740");
    static const std::string apple = p.deviceName;  // fixed for the process
    return reinterpret_cast<const GLubyte*>(vendor ? "Apple" : apple.c_str());
}

const GLubyte* GetString(GLenum name) {
    CTX(nullptr)
    static const std::string ext = extensionString();
    switch (name) {
        case gl::VENDOR: return identityString(true);
        case gl::RENDERER: return identityString(false);
        case gl::VERSION: return reinterpret_cast<const GLubyte*>(versionString().c_str());
        case gl::SHADING_LANGUAGE_VERSION: return reinterpret_cast<const GLubyte*>(glslVersionString().c_str());
        case gl::EXTENSIONS: return reinterpret_cast<const GLubyte*>(ext.c_str());
        default: c->setError(gl::INVALID_ENUM, name); return nullptr;
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

// The ES 3.0 / 3.1 / 3.2 implementation-dependent and dynamic state that queryState's switch does not hold.
// Values describe this implementation (what it really supports); limits of stages it does not have
// (geometry, tessellation) are 0, so a guest probing them gets a defined answer instead of INVALID_ENUM.
constexpr int kCombinedUniformComponents = 14 * (65536 / 4) + 1024;  // blocks * block size / 4 + default-block components
int queryStateExtra(Context* c, GLenum p, double* out) {
    auto one = [&](double v) { out[0] = v; return 1; };
    static const struct {
        GLenum p;
        double v;
    } kLimits[] = {
        // ES 3.0
        {0x8A31, kCombinedUniformComponents},  // MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS
        {0x8A33, kCombinedUniformComponents},  // MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS
        {0x8C8A, 64},                          // MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS
        {0x8C8B, 4},                           // MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS
        {0x8C80, 4},                           // MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS
        // ES 3.1 compute / atomic counters / images / shader storage
        {0x8262, 32768},                       // MAX_COMPUTE_SHARED_MEMORY_SIZE
        {0x8263, 1024},                        // MAX_COMPUTE_UNIFORM_COMPONENTS
        {0x8264, kMaxAtomicCounterBindings},   // MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS
        {0x8265, 8},                           // MAX_COMPUTE_ATOMIC_COUNTERS
        {0x8266, kCombinedUniformComponents},  // MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS
        {0x90EB, 1024},                        // MAX_COMPUTE_WORK_GROUP_INVOCATIONS
        {0x91BB, 14},                          // MAX_COMPUTE_UNIFORM_BLOCKS
        {0x91BC, 16},                          // MAX_COMPUTE_TEXTURE_IMAGE_UNITS
        {0x91BD, kMaxImageUnits},              // MAX_COMPUTE_IMAGE_UNIFORMS
        {0x90DB, kMaxShaderStorageBindings},   // MAX_COMPUTE_SHADER_STORAGE_BLOCKS
        {0x92CC, 0},                           // MAX_VERTEX_ATOMIC_COUNTER_BUFFERS
        {0x92D0, 1},                           // MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS (descriptors are bound for both draw stages)
        {0x92D1, kMaxAtomicCounterBindings},   // MAX_COMBINED_ATOMIC_COUNTER_BUFFERS
        {0x92D2, 0},                           // MAX_VERTEX_ATOMIC_COUNTERS
        {0x92D6, 8},                           // MAX_FRAGMENT_ATOMIC_COUNTERS
        {0x92D7, 8},                           // MAX_COMBINED_ATOMIC_COUNTERS
        {0x92D8, 32},                          // MAX_ATOMIC_COUNTER_BUFFER_SIZE
        {0x92DC, kMaxAtomicCounterBindings},   // MAX_ATOMIC_COUNTER_BUFFER_BINDINGS
        {0x8F38, kMaxImageUnits},              // MAX_IMAGE_UNITS
        {0x90CA, 0},                           // MAX_VERTEX_IMAGE_UNIFORMS (vertexPipelineStoresAndAtomics is off in VPVulkan)
        {0x90CE, kMaxImageUnits},              // MAX_FRAGMENT_IMAGE_UNIFORMS
        {0x90CF, 24},                          // MAX_COMBINED_IMAGE_UNIFORMS (8 per stage)
        {0x90D6, 0},                           // MAX_VERTEX_SHADER_STORAGE_BLOCKS (vertexPipelineStoresAndAtomics is off in VPVulkan)
        {0x90DA, 4},                           // MAX_FRAGMENT_SHADER_STORAGE_BLOCKS
        {0x90DC, kMaxShaderStorageBindings},   // MAX_COMBINED_SHADER_STORAGE_BLOCKS
        {0x90DD, kMaxShaderStorageBindings},   // MAX_SHADER_STORAGE_BUFFER_BINDINGS
        {0x90DE, double(1 << 27)},             // MAX_SHADER_STORAGE_BLOCK_SIZE
        {0x90DF, 256},                         // SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT
        {0x8F39, 16},                          // MAX_COMBINED_SHADER_OUTPUT_RESOURCES
        {0x910E, 4},                           // MAX_COLOR_TEXTURE_SAMPLES
        {0x910F, 4},                           // MAX_DEPTH_TEXTURE_SAMPLES
        {0x9110, 4},                           // MAX_INTEGER_SAMPLES
        {0x9315, 16384},                       // MAX_FRAMEBUFFER_WIDTH
        {0x9316, 16384},                       // MAX_FRAMEBUFFER_HEIGHT
        {0x9317, 2048},                        // MAX_FRAMEBUFFER_LAYERS
        {0x9318, 4},                           // MAX_FRAMEBUFFER_SAMPLES
        {0x8E5E, -32},                         // MIN_PROGRAM_TEXTURE_GATHER_OFFSET (Quest 3; the compiler moves the coordinate beyond -8..7)
        {0x8E5F, 31},                          // MAX_PROGRAM_TEXTURE_GATHER_OFFSET
        {0x826E, 1048576},                     // MAX_UNIFORM_LOCATIONS
        {gl::MAX_VERTEX_ATTRIB_RELATIVE_OFFSET, 2047},
        {gl::TEXTURE_BUFFER_OFFSET_ALIGNMENT, 256},
        // ES 3.2 core / KHR_debug / robustness / OES_sample_variables: this implementation has none of the
        // optional stages, so their limits are 0 and their state is the default.
        {0x826C, 64},                          // MAX_DEBUG_GROUP_STACK_DEPTH
        {0x9144, 128},                         // MAX_DEBUG_LOGGED_MESSAGES
        {0x9143, 1024},                        // MAX_DEBUG_MESSAGE_LENGTH
        {0x82E8, 256},                         // MAX_LABEL_LENGTH
        {0x90F3, 0},                           // CONTEXT_ROBUST_ACCESS
        {0x8E5B, -0.5},                        // MIN_FRAGMENT_INTERPOLATION_OFFSET
        {0x8E5C, 0.5},                         // MAX_FRAGMENT_INTERPOLATION_OFFSET
        {0x8E5D, 4},                           // FRAGMENT_INTERPOLATION_OFFSET_BITS
        {0x8C37, 0},                           // MIN_SAMPLE_SHADING_VALUE
        {0x8E72, 0},                           // PATCH_VERTICES
        {0x825E, 0x8E4D},                      // LAYER_PROVOKING_VERTEX (FIRST_VERTEX_CONVENTION)
        {0x8E7D, 0}, {0x8E7E, 0}, {0x8E84, 0}, {0x8E7F, 0},              // MAX_PATCH_VERTICES, MAX_TESS_GEN_LEVEL, ...
        {0x8E80, 0}, {0x8E81, 0}, {0x8E82, 0}, {0x8E83, 0}, {0x8E85, 0}, {0x8E86, 0}, {0x886C, 0}, {0x886D, 0},
        {0x8E89, 0}, {0x8E8A, 0}, {0x8E1E, 0}, {0x8E1F, 0}, {0x92D3, 0}, {0x92D4, 0}, {0x92CD, 0}, {0x92CE, 0},
        {0x90D8, 0}, {0x90D9, 0}, {0x90CB, 0}, {0x90CC, 0},               // tessellation stage limits
        {0x8DE0, 0}, {0x8DE1, 0}, {0x8DDF, 0}, {0x8C29, 0}, {0x8A2C, 0}, {0x8A32, 0}, {0x9123, 0}, {0x9124, 0},
        {0x8E5A, 0}, {0x92CF, 0}, {0x92D5, 0}, {0x90CD, 0}, {0x90D7, 0},  // geometry stage limits
    };
    for (const auto& e : kLimits)
        if (e.p == p) return one(e.v);
    if (p == 0x8E23) return one(c->curTf().paused);              // TRANSFORM_FEEDBACK_PAUSED
    if (p == 0x8E24) return one(c->curTf().active);              // TRANSFORM_FEEDBACK_ACTIVE
    if (p == 0x8E25) return one(c->transformFeedback);           // TRANSFORM_FEEDBACK_BINDING
    switch (p) {
        case 0x825A: return one(double(c->programPipeline));          // PROGRAM_PIPELINE_BINDING
        case gl::MAX_TEXTURE_BUFFER_SIZE: return one(std::min<double>(c->dev->props.limits.maxTexelBufferElements, double(1 << 27)));
        case gl::DEBUG_GROUP_STACK_DEPTH: return one(double(c->debugGroups.size() + 1));
        case gl::DEBUG_LOGGED_MESSAGES: return one(double(c->debugLog.size()));
        case gl::DEBUG_NEXT_LOGGED_MESSAGE_LENGTH: return one(c->debugLog.empty() ? 0.0 : double(c->debugLog.front().text.size() + 1));
        case gl::DEBUG_CALLBACK_USER_PARAM: return one(double(reinterpret_cast<uintptr_t>(c->debugUser)));
        case 0x8E28: /* TIMESTAMP_EXT */ {
            using namespace std::chrono;
            return one(double(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()));
        }
        case 0x9381: out[0] = out[1] = 1; return 2;                  // MULTISAMPLE_LINE_WIDTH_RANGE
        case 0x9382: return one(0);                                  // MULTISAMPLE_LINE_WIDTH_GRANULARITY
        case 0x92BE: {                                               // PRIMITIVE_BOUNDING_BOX (default box)
            const double box[8] = {-1, -1, -1, 1, 1, 1, 1, 1};
            for (int i = 0; i < 8; ++i) out[i] = box[i];
            return 8;
        }
        case 0x87FF: out[0] = kProgramBinaryFormat; return 1;        // PROGRAM_BINARY_FORMATS
        case 0x8DF8: return 0;                                       // SHADER_BINARY_FORMATS: none
        default: return -1;
    }
}

// Fills `out` (up to 16 values) for `pname`; returns the count (0 for a known pname with no values,
// e.g. the program binary format list), -1 for unknown pnames.
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
    // Every format formats.cpp can create: ETC1, ETC2/EAC, and ASTC LDR (14 block sizes, RGBA then sRGB).
    static const GLenum kCompressed[] = {
        gl::ETC1_RGB8_OES, gl::COMPRESSED_RGB8_ETC2, gl::COMPRESSED_SRGB8_ETC2, gl::COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
        gl::COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, gl::COMPRESSED_RGBA8_ETC2_EAC, gl::COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
        gl::COMPRESSED_R11_EAC, gl::COMPRESSED_SIGNED_R11_EAC, gl::COMPRESSED_RG11_EAC, gl::COMPRESSED_SIGNED_RG11_EAC,
        0x93B0, 0x93B1, 0x93B2, 0x93B3, 0x93B4, 0x93B5, 0x93B6, 0x93B7, 0x93B8, 0x93B9, 0x93BA, 0x93BB, 0x93BC, 0x93BD,
        0x93D0, 0x93D1, 0x93D2, 0x93D3, 0x93D4, 0x93D5, 0x93D6, 0x93D7, 0x93D8, 0x93D9, 0x93DA, 0x93DB, 0x93DC, 0x93DD};
    switch (p) {
        case gl::MAX_TEXTURE_SIZE: case gl::MAX_RENDERBUFFER_SIZE: case gl::MAX_CUBE_MAP_TEXTURE_SIZE: return one(16384);
        case gl::MAX_3D_TEXTURE_SIZE: return one(2048);
        case gl::MAX_ARRAY_TEXTURE_LAYERS: return one(2048);
        case gl::MAX_VIEWPORT_DIMS: out[0] = out[1] = 16384; return 2;
        case gl::MAX_VERTEX_ATTRIBS: return one(kMaxVertexAttribs);
        case gl::MAX_VERTEX_UNIFORM_VECTORS: case gl::MAX_FRAGMENT_UNIFORM_VECTORS: return one(256);
        case gl::MAX_VERTEX_UNIFORM_COMPONENTS: case gl::MAX_FRAGMENT_UNIFORM_COMPONENTS: return one(1024);
        case gl::MAX_VARYING_VECTORS: return one(31);
        case gl::MAX_VARYING_COMPONENTS: return one(124);
        case gl::MAX_VERTEX_OUTPUT_COMPONENTS: return one(128);
        case gl::MAX_FRAGMENT_INPUT_COMPONENTS: return one(128);
        case gl::MAX_COMBINED_TEXTURE_IMAGE_UNITS: return one(kMaxTextureUnits);
        case gl::MAX_VERTEX_TEXTURE_IMAGE_UNITS: case gl::MAX_TEXTURE_IMAGE_UNITS: return one(16);
        case gl::MAX_DRAW_BUFFERS: case gl::MAX_COLOR_ATTACHMENTS: return one(kMaxDrawBuffers);
        case gl::MAX_SAMPLES: return one(4);
        case gl::MAX_UNIFORM_BUFFER_BINDINGS: return one(kMaxUniformBufferBindings);
        case gl::MAX_UNIFORM_BLOCK_SIZE: return one(65536);
        case gl::MAX_VERTEX_UNIFORM_BLOCKS: case gl::MAX_FRAGMENT_UNIFORM_BLOCKS: return one(14);
        case gl::MAX_COMBINED_UNIFORM_BLOCKS: return one(84);
        case gl::UNIFORM_BUFFER_OFFSET_ALIGNMENT: return one(64);
        case gl::MAX_ELEMENT_INDEX: return one(4294967295.0);
        case gl::MAX_ELEMENTS_INDICES: case gl::MAX_ELEMENTS_VERTICES: return one(1 << 20);
        case gl::MAX_TEXTURE_LOD_BIAS: return one(16);
        case gl::MIN_PROGRAM_TEXEL_OFFSET: return one(-8);
        case gl::MAX_PROGRAM_TEXEL_OFFSET: return one(7);
        case gl::MAX_TEXTURE_MAX_ANISOTROPY_EXT: return one(16);
        case gl::MAX_VIEWS_OVR: return one(2);
        case gl::MAX_SERVER_WAIT_TIMEOUT: return one(0);
        case gl::MAX_VERTEX_ATTRIB_BINDINGS: return one(kMaxVertexBindings);
        case gl::MAX_VERTEX_ATTRIB_STRIDE: return one(2048);
        case gl::MAX_SAMPLE_MASK_WORDS: return one(1);
        case gl::ALIASED_LINE_WIDTH_RANGE: out[0] = out[1] = 1; return 2;
        case gl::ALIASED_POINT_SIZE_RANGE: out[0] = 1, out[1] = 511; return 2;
        case gl::SUBPIXEL_BITS: return one(8);
        case gl::MAJOR_VERSION: return one(gl::kAdvertisedMajor);
        case gl::MINOR_VERSION: return one(gl::kAdvertisedMinor);
        case gl::NUM_EXTENSIONS: return one(double(sizeof kExtensions / sizeof kExtensions[0]));
        case gl::NUM_COMPRESSED_TEXTURE_FORMATS: return one(double(sizeof kCompressed / sizeof kCompressed[0]));
        case gl::COMPRESSED_TEXTURE_FORMATS: {
            int n = 0;
            for (GLenum f : kCompressed) out[n++] = f;
            return n;
        }
        case gl::NUM_SHADER_BINARY_FORMATS: return one(0);
        case gl::NUM_PROGRAM_BINARY_FORMATS: return one(1);
        case gl::SHADER_COMPILER: return one(1);
        case gl::RED_BITS: return one(fbBits(0));
        case gl::GREEN_BITS: return one(fbBits(1));
        case gl::BLUE_BITS: return one(fbBits(2));
        case gl::ALPHA_BITS: return one(fbBits(3));
        case gl::DEPTH_BITS: return one(fbBits(4));
        case gl::STENCIL_BITS: return one(fbBits(5));
        case gl::SAMPLE_BUFFERS: case gl::SAMPLES: {
            const auto fb = c->resolveFramebuffer(c->drawFbo, true);
            const GLint n = fb.status == gl::FRAMEBUFFER_COMPLETE && fb.samples > 1 ? GLint(fb.samples) : 0;
            return one(p == gl::SAMPLES ? n : (n ? 1 : 0));
        }
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
        case gl::TEXTURE_BINDING_2D_MULTISAMPLE: return one(c->units[size_t(c->activeUnit)].tex2DMS);
        case gl::TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY: return one(c->units[size_t(c->activeUnit)].tex2DMSArray);
        case gl::TEXTURE_BINDING_CUBE_MAP_ARRAY: return one(c->units[size_t(c->activeUnit)].texCubeArray);
        case gl::TEXTURE_BINDING_BUFFER: return one(c->units[size_t(c->activeUnit)].texBuffer);
        case gl::SAMPLER_BINDING: return one(c->units[size_t(c->activeUnit)].sampler);
        case gl::ARRAY_BUFFER_BINDING: return one(c->arrayBuffer);
        case gl::ELEMENT_ARRAY_BUFFER_BINDING: return one(c->currentVao()->elementBuffer);
        case gl::UNIFORM_BUFFER_BINDING: return one(c->uniformBuffer);
        case gl::PIXEL_PACK_BUFFER_BINDING: return one(c->pixelPack);
        case gl::PIXEL_UNPACK_BUFFER_BINDING: return one(c->pixelUnpack);
        case gl::SHADER_STORAGE_BUFFER_BINDING: return one(c->inertBuffer[0]);
        case gl::ATOMIC_COUNTER_BUFFER_BINDING: return one(c->inertBuffer[1]);
        case gl::DRAW_INDIRECT_BUFFER_BINDING: return one(c->inertBuffer[2]);
        case gl::DISPATCH_INDIRECT_BUFFER_BINDING: return one(c->inertBuffer[3]);
        case gl::TRANSFORM_FEEDBACK_BUFFER_BINDING: return one(c->inertBuffer[4]);
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
        default: break;
    }
    if (p >= gl::DRAW_BUFFER0 && p < gl::DRAW_BUFFER0 + kMaxDrawBuffers) {
        const Framebuffer* f = c->fbo(c->drawFbo);
        return one(f ? f->drawBuffers[p - gl::DRAW_BUFFER0] : gl::NONE);
    }
    return queryStateExtra(c, p, out);
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
        case gl::DEBUG_OUTPUT: return c->debugOutput;
        case gl::DEBUG_OUTPUT_SYNCHRONOUS: return c->debugSync;
        case gl::SAMPLE_MASK: return c->sampleMaskEnable;
        case 0x8C36 /* SAMPLE_SHADING */: case gl::FRAMEBUFFER_SRGB_EXT: return false;
        default: ok = false; return false;
    }
}

void GetIntegerv(GLenum p, GLint* v) {
    CTX()
    double tmp[64];
    int n = queryState(c, p, tmp);
    if (n < 0) {
        bool ok;
        bool e = IsEnabledImpl(c, p, ok);
        if (ok) {
            v[0] = e;
            return;
        }
        return c->setError(gl::INVALID_ENUM, p);
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
    if (n < 0) {
        bool ok;
        bool e = IsEnabledImpl(c, p, ok);
        if (ok) {
            v[0] = e;
            return;
        }
        return c->setError(gl::INVALID_ENUM, p);
    }
    for (int i = 0; i < n; ++i) v[i] = GLint64(tmp[i]);
}
void GetFloatv(GLenum p, GLfloat* v) {
    CTX()
    double tmp[64];
    int n = queryState(c, p, tmp);
    if (n < 0) {
        bool ok;
        bool e = IsEnabledImpl(c, p, ok);
        if (ok) {
            v[0] = e;
            return;
        }
        return c->setError(gl::INVALID_ENUM, p);
    }
    for (int i = 0; i < n; ++i) v[i] = GLfloat(tmp[i]);
}
void GetBooleanv(GLenum p, GLboolean* v) {
    CTX()
    double tmp[64];
    int n = queryState(c, p, tmp);
    if (n < 0) {
        bool ok;
        bool e = IsEnabledImpl(c, p, ok);
        if (ok) {
            v[0] = e;
            return;
        }
        return c->setError(gl::INVALID_ENUM, p);
    }
    for (int i = 0; i < n; ++i) v[i] = tmp[i] != 0.0;
}
// Indexed state: uniform / shader storage / atomic counter / transform feedback buffer bindings, vertex
// buffer bindings, image units, compute limits, sample mask words. Returns the values written, -1 for an
// unknown target, -2 for an index out of range.
int queryIndexed(Context* c, GLenum target, GLuint index, int64_t* out) {
    auto one = [&](int64_t v) { out[0] = v; return 1; };
    auto bufferState = [&](const Context::IndexedBuffer* arr, size_t n, int which) {
        if (index >= n) return -2;
        return one(which == 0 ? arr[index].buffer : which == 1 ? arr[index].offset : arr[index].size);
    };
    switch (target) {
        case gl::UNIFORM_BUFFER_BINDING: return bufferState(c->uniformBindings.data(), c->uniformBindings.size(), 0);
        case 0x8A29: /* UNIFORM_BUFFER_START */ return bufferState(c->uniformBindings.data(), c->uniformBindings.size(), 1);
        case 0x8A2A: /* UNIFORM_BUFFER_SIZE */ return bufferState(c->uniformBindings.data(), c->uniformBindings.size(), 2);
        case gl::SHADER_STORAGE_BUFFER_BINDING: return bufferState(c->ssboBindings.data(), c->ssboBindings.size(), 0);
        case 0x90D4: /* SHADER_STORAGE_BUFFER_START */ return bufferState(c->ssboBindings.data(), c->ssboBindings.size(), 1);
        case 0x90D5: /* SHADER_STORAGE_BUFFER_SIZE */ return bufferState(c->ssboBindings.data(), c->ssboBindings.size(), 2);
        case gl::ATOMIC_COUNTER_BUFFER_BINDING: return bufferState(c->atomicBindings.data(), c->atomicBindings.size(), 0);
        case 0x92C2: /* ATOMIC_COUNTER_BUFFER_START */ return bufferState(c->atomicBindings.data(), c->atomicBindings.size(), 1);
        case 0x92C3: /* ATOMIC_COUNTER_BUFFER_SIZE */ return bufferState(c->atomicBindings.data(), c->atomicBindings.size(), 2);
        case gl::TRANSFORM_FEEDBACK_BUFFER_BINDING: return bufferState(c->curTf().buffers.data(), c->curTf().buffers.size(), 0);
        case 0x8C84: /* TRANSFORM_FEEDBACK_BUFFER_START */ return bufferState(c->curTf().buffers.data(), c->curTf().buffers.size(), 1);
        case 0x8C85: /* TRANSFORM_FEEDBACK_BUFFER_SIZE */ return bufferState(c->curTf().buffers.data(), c->curTf().buffers.size(), 2);
        case gl::MAX_COMPUTE_WORK_GROUP_COUNT:  // 0x91BE
            if (index > 2) return -2;
            return one(65535);
        case gl::MAX_COMPUTE_WORK_GROUP_SIZE:  // 0x91BF
            if (index > 2) return -2;
            return one(index == 2 ? 64 : 1024);
        case gl::VERTEX_BINDING_DIVISOR: case gl::VERTEX_BINDING_OFFSET: case gl::VERTEX_BINDING_STRIDE:
        case gl::VERTEX_BINDING_BUFFER: {
            if (index >= kMaxVertexBindings) return -2;
            const VertexBinding& b = c->currentVao()->bindings[index];
            return one(target == gl::VERTEX_BINDING_DIVISOR ? int64_t(b.divisor)
                       : target == gl::VERTEX_BINDING_OFFSET ? int64_t(b.offset)
                       : target == gl::VERTEX_BINDING_STRIDE ? int64_t(b.stride) : int64_t(b.buffer));
        }
        case gl::IMAGE_BINDING_NAME: case gl::IMAGE_BINDING_LEVEL: case gl::IMAGE_BINDING_LAYERED: case gl::IMAGE_BINDING_LAYER:
        case gl::IMAGE_BINDING_ACCESS: case gl::IMAGE_BINDING_FORMAT: {
            if (index >= kMaxImageUnits) return -2;
            const Context::ImageUnit& u = c->images[index];
            return one(target == gl::IMAGE_BINDING_NAME ? int64_t(u.texture)
                       : target == gl::IMAGE_BINDING_LEVEL ? int64_t(u.level)
                       : target == gl::IMAGE_BINDING_LAYERED ? int64_t(u.layered)
                       : target == gl::IMAGE_BINDING_LAYER ? int64_t(u.layer)
                       : target == gl::IMAGE_BINDING_ACCESS ? int64_t(u.access) : int64_t(u.format));
        }
        case 0x8E52: /* SAMPLE_MASK_VALUE */
            if (index >= 1) return -2;
            return one(int64_t(int32_t(c->sampleMask)));
        // GL_EXT_draw_buffers_indexed: per draw buffer blend state and colour write mask.
        case gl::BLEND_SRC_RGB: case gl::BLEND_DST_RGB: case gl::BLEND_SRC_ALPHA: case gl::BLEND_DST_ALPHA:
        case gl::BLEND_EQUATION_RGB: case gl::BLEND_EQUATION_ALPHA: {
            if (index >= kMaxDrawBuffers) return -2;
            const BlendState& b = c->blend[index];
            return one(target == gl::BLEND_SRC_RGB ? b.srcRGB : target == gl::BLEND_DST_RGB ? b.dstRGB
                       : target == gl::BLEND_SRC_ALPHA ? b.srcA : target == gl::BLEND_DST_ALPHA ? b.dstA
                       : target == gl::BLEND_EQUATION_RGB ? b.eqRGB : b.eqA);
        }
        case gl::COLOR_WRITEMASK:
            if (index >= kMaxDrawBuffers) return -2;
            for (int i = 0; i < 4; ++i) out[i] = c->blend[index].mask[i];
            return 4;
        default: return -1;
    }
}
void GetIntegeri_v(GLenum target, GLuint index, GLint* v) {
    CTX()
    int64_t t[4] = {};
    const int n = queryIndexed(c, target, index, t);
    if (n == -1) return c->setError(gl::INVALID_ENUM, target);
    if (n < 0) return c->setError(gl::INVALID_VALUE, target);
    for (int i = 0; i < n; ++i) v[i] = GLint(t[i]);
}
void GetInteger64i_v(GLenum target, GLuint index, GLint64* v) {
    CTX()
    int64_t t[4] = {};
    const int n = queryIndexed(c, target, index, t);
    if (n == -1) return c->setError(gl::INVALID_ENUM, target);
    if (n < 0) return c->setError(gl::INVALID_VALUE, target);
    for (int i = 0; i < n; ++i) v[i] = t[i];
}
void GetBooleani_v(GLenum target, GLuint index, GLboolean* v) {
    CTX()
    if (target == gl::COLOR_WRITEMASK) {
        if (index >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE, target);
        for (int i = 0; i < 4; ++i) v[i] = c->blend[index].mask[i];
        return;
    }
    int64_t t[4] = {};
    const int n = queryIndexed(c, target, index, t);
    if (n == -1) return c->setError(gl::INVALID_ENUM, target);
    if (n < 0) return c->setError(gl::INVALID_VALUE, target);
    for (int i = 0; i < n; ++i) v[i] = t[i] != 0;
}

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
        case gl::DEBUG_OUTPUT: c->debugOutput = on; break;
        case gl::DEBUG_OUTPUT_SYNCHRONOUS: c->debugSync = on; break;
        case gl::SAMPLE_MASK: c->sampleMaskEnable = on; break;
        case 0x8C36 /* SAMPLE_SHADING */: case gl::FRAMEBUFFER_SRGB_EXT: break;
        default: c->setError(gl::INVALID_ENUM, cap);
    }
}
void Enable(GLenum cap) { Enable_(cap, true); }
void Disable(GLenum cap) { Enable_(cap, false); }
GLboolean IsEnabled(GLenum cap) {
    CTX(0)
    bool ok;
    bool e = IsEnabledImpl(c, cap, ok);
    if (!ok) c->setError(gl::INVALID_ENUM, cap);
    return e;
}
void Enablei(GLenum cap, GLuint i) {
    CTX()
    if (cap != gl::BLEND) return c->setError(gl::INVALID_ENUM, cap);
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE, cap);
    c->blend[i].enabled = true;
}
void Disablei(GLenum cap, GLuint i) {
    CTX()
    if (cap != gl::BLEND) return c->setError(gl::INVALID_ENUM, cap);
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE, cap);
    c->blend[i].enabled = false;
}
GLboolean IsEnabledi(GLenum cap, GLuint i) {
    CTX(0)
    if (cap != gl::BLEND) return c->setError(gl::INVALID_ENUM, cap), 0;
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE, cap), 0;
    return c->blend[i].enabled;
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
// glClearBuffer*: the buffer must suit the type (ENUM) and the draw buffer be valid for it (VALUE) before anything else.
static bool validClearBuffer(Context* c, GLenum b, GLint i, bool ok) {
    if (!ok) return c->setError(gl::INVALID_ENUM, b), false;
    if (i < 0 || (b == gl::COLOR ? i >= kMaxDrawBuffers : i != 0)) return c->setError(gl::INVALID_VALUE, b), false;
    return true;
}
void ClearBufferfv(GLenum b, GLint i, const GLfloat* v) {
    CTX()
    if (!validClearBuffer(c, b, i, b == gl::COLOR || b == gl::DEPTH)) return;
    if (b == gl::DEPTH) return c->clearBuffer(b, i, nullptr, gl::FLOAT, v[0], 0);
    c->clearBuffer(b, i, v, gl::FLOAT, 0, 0);
}
void ClearBufferiv(GLenum b, GLint i, const GLint* v) {
    CTX()
    if (!validClearBuffer(c, b, i, b == gl::COLOR || b == gl::STENCIL)) return;
    if (b == gl::STENCIL) return c->clearBuffer(b, i, nullptr, gl::INT, 0, v[0]);
    c->clearBuffer(b, i, v, gl::INT, 0, 0);
}
void ClearBufferuiv(GLenum b, GLint i, const GLuint* v) {
    CTX()
    if (!validClearBuffer(c, b, i, b == gl::COLOR)) return;
    c->clearBuffer(b, i, v, gl::UNSIGNED_INT, 0, 0);
}
void ClearBufferfi(GLenum b, GLint i, GLfloat d, GLint s) {
    CTX()
    if (!validClearBuffer(c, b, i, b == gl::DEPTH_STENCIL)) return;
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
    if (f < gl::NEVER || f > gl::ALWAYS) return c->setError(gl::INVALID_ENUM, f);
    c->depthFunc = f;
}
void CullFace(GLenum m) {
    CTX()
    if (m != gl::FRONT && m != gl::BACK && m != gl::FRONT_AND_BACK) return c->setError(gl::INVALID_ENUM, m);
    c->cullMode = m;
}
void FrontFace(GLenum m) {
    CTX()
    if (m != gl::CW && m != gl::CCW) return c->setError(gl::INVALID_ENUM, m);
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
// Enum validation of the state setters (INVALID_ENUM before any state changes).
bool validBlendFactor(GLenum f) { return f == 0 || f == 1 || (f >= 0x300 && f <= 0x308) || (f >= 0x8001 && f <= 0x8004); }
bool validBlendEq(GLenum e) { return e == 0x8006 || e == 0x8007 || e == 0x8008 || e == 0x800A || e == 0x800B; }
bool validStencilOp(GLenum o) { return o == 0 || (o >= 0x1E00 && o <= 0x1E03) || o == 0x150A || o == 0x8507 || o == 0x8508; }
bool validCompare(GLenum f) { return f >= 0x200 && f <= 0x207; }
bool validFace(GLenum f) { return f == 0x404 || f == 0x405 || f == 0x408; }
bool validUsage(GLenum u) { return u >= 0x88E0 && u <= 0x88EA && u != 0x88E3 && u != 0x88E7; }
void Hint(GLenum t, GLenum m) {
    CTX()
    if ((t != gl::GENERATE_MIPMAP_HINT && t != gl::FRAGMENT_SHADER_DERIVATIVE_HINT) || m < 0x1100 || m > 0x1102)
        return c->setError(gl::INVALID_ENUM, t);
    if (t == gl::GENERATE_MIPMAP_HINT) c->hints[0] = m;
    else if (t == gl::FRAGMENT_SHADER_DERIVATIVE_HINT) c->hints[1] = m;
}
void BlendFunc(GLenum s, GLenum d) {
    CTX()
    if (!validBlendFactor(s) || !validBlendFactor(d)) return c->setError(gl::INVALID_ENUM, s);
    for (auto& b : c->blend) b.srcRGB = b.srcA = s, b.dstRGB = b.dstA = d;
}
void BlendFuncSeparate(GLenum sr, GLenum dr, GLenum sa, GLenum da) {
    CTX()
    if (!validBlendFactor(sr) || !validBlendFactor(dr) || !validBlendFactor(sa) || !validBlendFactor(da))
        return c->setError(gl::INVALID_ENUM, sr);
    for (auto& b : c->blend) b.srcRGB = sr, b.dstRGB = dr, b.srcA = sa, b.dstA = da;
}
void BlendFunci(GLuint i, GLenum s, GLenum d) {
    CTX()
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE);
    if (!validBlendFactor(s) || !validBlendFactor(d)) return c->setError(gl::INVALID_ENUM, s);
    c->blend[i].srcRGB = c->blend[i].srcA = s, c->blend[i].dstRGB = c->blend[i].dstA = d;
}
void BlendFuncSeparatei(GLuint i, GLenum sr, GLenum dr, GLenum sa, GLenum da) {
    CTX()
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE);
    if (!validBlendFactor(sr) || !validBlendFactor(dr) || !validBlendFactor(sa) || !validBlendFactor(da))
        return c->setError(gl::INVALID_ENUM, sr);
    c->blend[i].srcRGB = sr, c->blend[i].dstRGB = dr, c->blend[i].srcA = sa, c->blend[i].dstA = da;
}
void BlendEquation(GLenum e) {
    CTX()
    if (!validBlendEq(e)) return c->setError(gl::INVALID_ENUM, e);
    for (auto& b : c->blend) b.eqRGB = b.eqA = e;
}
void BlendEquationSeparate(GLenum r, GLenum a) {
    CTX()
    if (!validBlendEq(r) || !validBlendEq(a)) return c->setError(gl::INVALID_ENUM, r);
    for (auto& b : c->blend) b.eqRGB = r, b.eqA = a;
}
void BlendEquationi(GLuint i, GLenum e) {
    CTX()
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE);
    if (!validBlendEq(e)) return c->setError(gl::INVALID_ENUM, e);
    c->blend[i].eqRGB = c->blend[i].eqA = e;
}
void BlendEquationSeparatei(GLuint i, GLenum r, GLenum a) {
    CTX()
    if (i >= kMaxDrawBuffers) return c->setError(gl::INVALID_VALUE);
    if (!validBlendEq(r) || !validBlendEq(a)) return c->setError(gl::INVALID_ENUM, r);
    c->blend[i].eqRGB = r, c->blend[i].eqA = a;
}
void BlendColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    CTX()
    c->blendColor[0] = r, c->blendColor[1] = g, c->blendColor[2] = b, c->blendColor[3] = a;
}
void StencilFuncSeparate(GLenum face, GLenum f, GLint ref, GLuint mask) {
    CTX()
    if (!validFace(face) || !validCompare(f)) return c->setError(gl::INVALID_ENUM, face);
    for (int i = 0; i < 2; ++i)
        if (face == gl::FRONT_AND_BACK || (face == gl::FRONT) == (i == 0)) c->stencil[i].func = f, c->stencil[i].ref = ref, c->stencil[i].mask = mask;
}
void StencilFunc(GLenum f, GLint ref, GLuint mask) { StencilFuncSeparate(gl::FRONT_AND_BACK, f, ref, mask); }
void StencilMaskSeparate(GLenum face, GLuint m) {
    CTX()
    if (!validFace(face)) return c->setError(gl::INVALID_ENUM, face);
    for (int i = 0; i < 2; ++i)
        if (face == gl::FRONT_AND_BACK || (face == gl::FRONT) == (i == 0)) c->stencil[i].writeMask = m;
}
void StencilMask(GLuint m) { StencilMaskSeparate(gl::FRONT_AND_BACK, m); }
void StencilOpSeparate(GLenum face, GLenum sf, GLenum zf, GLenum zp) {
    CTX()
    if (!validFace(face) || !validStencilOp(sf) || !validStencilOp(zf) || !validStencilOp(zp)) return c->setError(gl::INVALID_ENUM, face);
    for (int i = 0; i < 2; ++i)
        if (face == gl::FRONT_AND_BACK || (face == gl::FRONT) == (i == 0)) c->stencil[i].fail = sf, c->stencil[i].zfail = zf, c->stencil[i].zpass = zp;
}
void StencilOp(GLenum sf, GLenum zf, GLenum zp) { StencilOpSeparate(gl::FRONT_AND_BACK, sf, zf, zp); }
void PixelStorei(GLenum p, GLint v) {
    CTX()
    switch (p) {
        case gl::UNPACK_ALIGNMENT:
        case gl::PACK_ALIGNMENT:
            if (v != 1 && v != 2 && v != 4 && v != 8) return c->setError(gl::INVALID_VALUE, p);
            (p == gl::UNPACK_ALIGNMENT ? c->unpackAlignment : c->packAlignment) = v;
            return;
        case gl::UNPACK_ROW_LENGTH: case gl::UNPACK_SKIP_ROWS: case gl::UNPACK_SKIP_PIXELS: case gl::UNPACK_IMAGE_HEIGHT:
        case gl::UNPACK_SKIP_IMAGES: case gl::PACK_ROW_LENGTH: case gl::PACK_SKIP_ROWS: case gl::PACK_SKIP_PIXELS:
            if (v < 0) return c->setError(gl::INVALID_VALUE, p);
            break;
        default: break;
    }
    switch (p) {
        case gl::UNPACK_ROW_LENGTH: c->unpackRowLength = v; return;
        case gl::UNPACK_SKIP_ROWS: c->unpackSkipRows = v; return;
        case gl::UNPACK_SKIP_PIXELS: c->unpackSkipPixels = v; return;
        case gl::UNPACK_IMAGE_HEIGHT: c->unpackImageHeight = v; return;
        case gl::UNPACK_SKIP_IMAGES: c->unpackSkipImages = v; return;
        case gl::PACK_ROW_LENGTH: c->packRowLength = v; return;
        case gl::PACK_SKIP_ROWS: c->packSkipRows = v; return;
        case gl::PACK_SKIP_PIXELS: c->packSkipPixels = v; return;
        default: c->setError(gl::INVALID_ENUM, p);
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
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint x = names[i];
        if (!x) continue;
        auto it = c->share->buffers.find(x);
        if (it == c->share->buffers.end()) continue;
        if (it->second->storage) c->keep(it->second->storage);
        // Unbind from this context.
        for (GLuint* b : {&c->arrayBuffer, &c->copyRead, &c->copyWrite, &c->pixelPack, &c->pixelUnpack, &c->uniformBuffer})
            if (*b == x) *b = 0;
        for (GLuint& b : c->inertBuffer)
            if (b == x) b = 0;
        for (auto& ub : c->uniformBindings)
            if (ub.buffer == x) ub = {};
        for (auto& ub : c->ssboBindings)
            if (ub.buffer == x) ub = {};
        for (auto& ub : c->atomicBindings)
            if (ub.buffer == x) ub = {};
        for (auto& [k, tf] : c->transformFeedbacks)
            for (auto& ub : tf->buffers)
                if (ub.buffer == x) ub = {};
        for (auto& [k, va] : c->vaos) {
            if (va->elementBuffer == x) va->elementBuffer = 0;
            for (auto& vb : va->bindings)
                if (vb.buffer == x) vb.buffer = 0, vb.offset = 0;  // the divisor and stride belong to the binding, not to the buffer
        }
        // A buffer texture whose buffer is deleted loses its data store.
        for (auto& [tn, t] : c->share->textures)
            if (t->bufferName == x) t->bufferName = 0, t->bufferSize = 0, t->bufferOffset = 0;
        c->share->buffers.erase(it);
    }
}
GLboolean IsBuffer(GLuint n) {
    CTX(0)
    return c->buffer(n) && c->buffer(n)->bound ? 1 : 0;
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
    b->bound = true;
    return b;
}
void BindBuffer(GLenum target, GLuint name) {
    CTX()
    GLuint* slot = c->bufferBinding(target);
    if (!slot) return c->setError(gl::INVALID_ENUM, target);
    bindNamed(c, name);
    *slot = name;
}
// Indexed binds of the shader storage / atomic counter / transform feedback targets: the indexed binding
// point and the generic one (uniform buffers have their own path below). `range` = glBindBufferRange.
bool bindIndexedTarget(Context* c, GLenum target, GLuint index, GLuint name, GLintptr off, GLsizeiptr size, bool range) {
    Context::IndexedBuffer* arr = nullptr;
    size_t n = 0;
    GLintptr align = 4;
    switch (target) {
        case gl::SHADER_STORAGE_BUFFER: arr = c->ssboBindings.data(), n = c->ssboBindings.size(), align = 256; break;
        case gl::ATOMIC_COUNTER_BUFFER: arr = c->atomicBindings.data(), n = c->atomicBindings.size(); break;
        case gl::TRANSFORM_FEEDBACK_BUFFER: arr = c->curTf().buffers.data(), n = c->curTf().buffers.size(); break;
        default: return false;
    }
    const bool xfb = target == gl::TRANSFORM_FEEDBACK_BUFFER;
    if (index >= n) c->setError(gl::INVALID_VALUE, target);
    else if (xfb && c->curTf().active && !c->curTf().paused) c->setError(gl::INVALID_OPERATION, target);
    else if (range && name && (off < 0 || size <= 0 || off % align || (xfb && size % 4))) c->setError(gl::INVALID_VALUE, target);
    else {
        bindNamed(c, name);
        arr[index] = name ? Context::IndexedBuffer{name, range ? off : 0, range ? size : 0} : Context::IndexedBuffer{};
        *c->bufferBinding(target) = name;
    }
    return true;
}
void BindBufferRange(GLenum target, GLuint index, GLuint name, GLintptr off, GLsizeiptr size) {
    CTX()
    if (bindIndexedTarget(c, target, index, name, off, size, true)) return;
    if (target != gl::UNIFORM_BUFFER) return c->setError(gl::INVALID_ENUM, target);
    if (index >= kMaxUniformBufferBindings) return c->setError(gl::INVALID_VALUE, target);
    if (name && (off < 0 || size <= 0 || off % 64)) return c->setError(gl::INVALID_VALUE, target);
    bindNamed(c, name);
    c->uniformBindings[index] = {name, off, size};
    c->uniformBuffer = name;
}
void BindBufferBase(GLenum target, GLuint index, GLuint name) {
    CTX()
    if (bindIndexedTarget(c, target, index, name, 0, 0, false)) return;
    if (target != gl::UNIFORM_BUFFER) return c->setError(gl::INVALID_ENUM, target);
    if (index >= kMaxUniformBufferBindings) return c->setError(gl::INVALID_VALUE, target);
    bindNamed(c, name);
    c->uniformBindings[index] = {name, 0, 0};
    c->uniformBuffer = name;
}
Buffer* targetBuffer(Context* c, GLenum target) {
    GLuint* slot = c->bufferBinding(target);
    if (!slot) {
        c->setError(gl::INVALID_ENUM, target);
        return nullptr;
    }
    Buffer* b = c->buffer(*slot);
    if (!b) c->setError(gl::INVALID_OPERATION, target);
    return b;
}
constexpr VkBufferUsageFlags kAllUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                                         VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
void BufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    CTX()
    if (!validUsage(usage)) return c->setError(gl::INVALID_ENUM, target);
    if (size < 0) return c->setError(gl::INVALID_VALUE, target);
    Buffer* b = targetBuffer(c, target);
    if (!b) return;
    if (b->mapped) return c->setError(gl::INVALID_OPERATION, target);
    // New storage every time (the old one stays alive with any work that reads it).
    if (b->storage) c->keep(b->storage);
    b->storage = c->dev->newStorage(VkDeviceSize(std::max<GLsizeiptr>(size, 1)), kAllUsage);
    if (!b->storage) return c->setError(gl::OUT_OF_MEMORY, target);
    b->size = size, b->usage = usage;
    if (data) std::memcpy(b->storage->host, data, size_t(size));
    else std::memset(b->storage->host, 0, size_t(size));
}
void BufferSubData(GLenum target, GLintptr off, GLsizeiptr size, const void* data) {
    CTX()
    Buffer* b = targetBuffer(c, target);
    if (!b) return;
    if (off < 0 || size < 0 || off + size > b->size) return c->setError(gl::INVALID_VALUE, target);
    if (b->mapped) return c->setError(gl::INVALID_OPERATION, target);
    if (!data || !size) return;
    c->prepareWrite(*b, off == 0 && size == b->size);
    std::memcpy(b->storage->host + off, data, size_t(size));
}
void* MapBufferRange(GLenum target, GLintptr off, GLsizeiptr len, GLbitfield access) {
    CTX(nullptr)
    Buffer* b = targetBuffer(c, target);
    if (!b) return nullptr;
    if (off < 0 || len < 0 || off + len > b->size || (access & ~0x3Fu)) {
        c->setError(gl::INVALID_VALUE, target);
        return nullptr;
    }
    const GLbitfield invalidating = gl::MAP_INVALIDATE_RANGE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT | gl::MAP_UNSYNCHRONIZED_BIT;
    if (!(access & (gl::MAP_READ_BIT | gl::MAP_WRITE_BIT)) || ((access & gl::MAP_READ_BIT) && (access & invalidating)) ||
        ((access & gl::MAP_FLUSH_EXPLICIT_BIT) && !(access & gl::MAP_WRITE_BIT))) {
        c->setError(gl::INVALID_OPERATION, target);
        return nullptr;
    }
    if (b->mapped) {
        c->setError(gl::INVALID_OPERATION, target);
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
        c->setError(gl::INVALID_OPERATION, target);
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
        default: c->setError(gl::INVALID_ENUM, target);
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
    if (ro < 0 || wo < 0 || size < 0 || ro + size > r->size || wo + size > w->size) return c->setError(gl::INVALID_VALUE, rt);
    if (r == w && ro < wo + size && wo < ro + size) return c->setError(gl::INVALID_VALUE, rt);  // the ranges overlap
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
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->nextVao++;
        c->vaos[name] = std::make_unique<VertexArray>();
        c->vaos[name]->name = name;
        out[i] = name;
    }
}
void DeleteVertexArrays(GLsizei n, const GLuint* names) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
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
// Vertex attribute types: the integer ones for glVertexAttribIPointer / IFormat, those and the float-convertible ones otherwise.
bool attribTypeOk(GLenum type, bool integer) {
    switch (type) {
        case gl::BYTE: case gl::UNSIGNED_BYTE: case gl::SHORT: case gl::UNSIGNED_SHORT: case gl::INT: case gl::UNSIGNED_INT: return true;
        case gl::FLOAT: case gl::HALF_FLOAT: case gl::HALF_FLOAT_OES: case gl::FIXED:
        case gl::INT_2_10_10_10_REV: case gl::UNSIGNED_INT_2_10_10_10_REV: return !integer;
        default: return false;
    }
}
void attribPointer(Context* c, GLuint i, GLint size, GLenum type, bool norm, bool integer, GLsizei stride, const void* ptr) {
    if (!attribTypeOk(type, integer)) return c->setError(gl::INVALID_ENUM, type);
    if (i >= kMaxVertexAttribs || size < 1 || size > 4 || stride < 0) return c->setError(gl::INVALID_VALUE, type);
    if ((type == gl::INT_2_10_10_10_REV || type == gl::UNSIGNED_INT_2_10_10_10_REV) && size != 4) return c->setError(gl::INVALID_OPERATION, type);
    if (c->vao && !c->arrayBuffer && ptr) return c->setError(gl::INVALID_OPERATION, type);
    VertexArray* va = c->currentVao();
    VertexAttrib& a = va->attribs[i];
    a.size = size, a.type = type, a.normalized = norm, a.integer = integer;
    a.relativeOffset = 0, a.binding = i, a.userStride = stride;
    VertexBinding& b = va->bindings[i];
    b.buffer = c->arrayBuffer, b.offset = reinterpret_cast<uintptr_t>(ptr);
    b.stride = stride ? stride : GLsizei(vertexAttribBytes(a));  // 0 = tightly packed
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
    VertexArray* va = c->currentVao();
    va->attribs[i].binding = i;  // VertexAttribDivisor = VertexAttribBinding(i, i) + VertexBindingDivisor(i, d)
    va->bindings[i].divisor = d;
}
// ARB_vertex_attrib_binding (ES 3.1): separate format and buffer binding.
bool attribFormatCheck(Context* c, GLuint i, GLint size, GLenum type, GLuint relOffset, bool integer) {
    if (!c->vao) {  // the attribute binding state is not available on the default vertex array object
        c->setError(gl::INVALID_OPERATION, type);
        return false;
    }
    if (i >= kMaxVertexAttribs || relOffset > 2047) {
        c->setError(gl::INVALID_VALUE, type);
        return false;
    }
    if (size < 1 || size > 4) {
        c->setError(gl::INVALID_VALUE, type);
        return false;
    }
    if (!attribTypeOk(type, integer)) {
        c->setError(gl::INVALID_ENUM, type);
        return false;
    }
    if ((type == gl::INT_2_10_10_10_REV || type == gl::UNSIGNED_INT_2_10_10_10_REV) && size != 4) {
        c->setError(gl::INVALID_OPERATION, type);
        return false;
    }
    return true;
}
void VertexAttribFormat(GLuint i, GLint size, GLenum type, GLboolean norm, GLuint relOffset) {
    CTX()
    if (!attribFormatCheck(c, i, size, type, relOffset, false)) return;
    VertexAttrib& a = c->currentVao()->attribs[i];
    a.size = size, a.type = type, a.normalized = norm, a.integer = false, a.relativeOffset = relOffset;
}
void VertexAttribIFormat(GLuint i, GLint size, GLenum type, GLuint relOffset) {
    CTX()
    if (!attribFormatCheck(c, i, size, type, relOffset, true)) return;
    VertexAttrib& a = c->currentVao()->attribs[i];
    a.size = size, a.type = type, a.normalized = false, a.integer = true, a.relativeOffset = relOffset;
}
void VertexAttribBinding(GLuint i, GLuint binding) {
    CTX()
    if (!c->vao) return c->setError(gl::INVALID_OPERATION);
    if (i >= kMaxVertexAttribs || binding >= kMaxVertexBindings) return c->setError(gl::INVALID_VALUE);
    c->currentVao()->attribs[i].binding = binding;
}
void BindVertexBuffer(GLuint binding, GLuint buffer, GLintptr offset, GLsizei stride) {
    CTX()
    if (!c->vao) return c->setError(gl::INVALID_OPERATION);
    if (binding >= kMaxVertexBindings || offset < 0 || stride < 0 || stride > 2048) return c->setError(gl::INVALID_VALUE);
    if (buffer && !c->buffer(buffer)) return c->setError(gl::INVALID_OPERATION);
    VertexBinding& b = c->currentVao()->bindings[binding];
    b.buffer = buffer, b.offset = uintptr_t(offset), b.stride = stride;
}
void VertexBindingDivisor(GLuint binding, GLuint divisor) {
    CTX()
    if (!c->vao) return c->setError(gl::INVALID_OPERATION);
    if (binding >= kMaxVertexBindings) return c->setError(gl::INVALID_VALUE);
    c->currentVao()->bindings[binding].divisor = divisor;
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
// Common reader of the per-attribute state. kind: 0 int, 1 float, 2 uint (CURRENT_VERTEX_ATTRIB of an
// integer attribute is read back as integers).
void getVertexAttrib(GLuint i, GLenum p, GLint* v, GLfloat* fv, bool* isFloat) {
    CTX()
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE, p);
    VertexArray* va = c->currentVao();
    const VertexAttrib& a = va->attribs[i];
    const VertexBinding& b = va->bindings[a.binding % kMaxVertexBindings];
    switch (p) {
        case gl::VERTEX_ATTRIB_ARRAY_ENABLED: *v = a.enabled; return;
        case gl::VERTEX_ATTRIB_ARRAY_SIZE: *v = a.size; return;
        case gl::VERTEX_ATTRIB_ARRAY_STRIDE: *v = a.userStride; return;
        case gl::VERTEX_ATTRIB_ARRAY_TYPE: *v = GLint(a.type); return;
        case gl::VERTEX_ATTRIB_ARRAY_NORMALIZED: *v = a.normalized; return;
        case gl::VERTEX_ATTRIB_ARRAY_INTEGER: *v = a.integer; return;
        case gl::VERTEX_ATTRIB_ARRAY_DIVISOR: *v = GLint(b.divisor); return;
        case gl::VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: *v = GLint(b.buffer); return;
        case gl::VERTEX_ATTRIB_BINDING: *v = GLint(a.binding); return;
        case gl::VERTEX_ATTRIB_RELATIVE_OFFSET: *v = GLint(a.relativeOffset); return;
        case gl::CURRENT_VERTEX_ATTRIB:
            if (c->currentAttribKind[i] == 0) {
                *isFloat = true;
                for (int k = 0; k < 4; ++k) fv[k] = c->currentAttrib[i][size_t(k)];
            } else {
                std::memcpy(v, c->currentAttrib[i].data(), 16);
            }
            return;
        default: c->setError(gl::INVALID_ENUM, p);
    }
}
void GetVertexAttribiv(GLuint i, GLenum p, GLint* v) {
    GLfloat f[4] = {};
    bool isFloat = false;
    getVertexAttrib(i, p, v, f, &isFloat);
    if (isFloat)
        for (int k = 0; k < 4; ++k) v[k] = GLint(f[k]);
}
void GetVertexAttribIiv(GLuint i, GLenum p, GLint* v) {
    GLfloat f[4] = {};
    bool isFloat = false;
    getVertexAttrib(i, p, v, f, &isFloat);
    if (isFloat)
        for (int k = 0; k < 4; ++k) v[k] = GLint(f[k]);
}
void GetVertexAttribIuiv(GLuint i, GLenum p, GLuint* v) { GetVertexAttribIiv(i, p, reinterpret_cast<GLint*>(v)); }
void GetVertexAttribfv(GLuint i, GLenum p, GLfloat* v) {
    GLint x[4] = {};
    GLfloat f[4] = {};
    bool isFloat = false;
    getVertexAttrib(i, p, x, f, &isFloat);
    if (isFloat) {
        for (int k = 0; k < 4; ++k) v[k] = f[k];
    } else if (p == gl::CURRENT_VERTEX_ATTRIB) {
        for (int k = 0; k < 4; ++k) v[k] = GLfloat(x[k]);
    } else {
        v[0] = GLfloat(x[0]);
    }
}
void GetVertexAttribPointerv(GLuint i, GLenum pname, void** v) {
    CTX()
    if (pname != gl::VERTEX_ATTRIB_ARRAY_POINTER) return c->setError(gl::INVALID_ENUM, pname);
    if (i >= kMaxVertexAttribs) return c->setError(gl::INVALID_VALUE);
    VertexArray* va = c->currentVao();
    const VertexAttrib& a = va->attribs[i];
    *v = reinterpret_cast<void*>(va->bindings[a.binding % kMaxVertexBindings].offset + a.relativeOffset);
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
void DrawRangeElements(GLenum m, GLuint start, GLuint end, GLsizei n, GLenum t, const void* idx) {
    CTX()
    if (end < start) return c->setError(gl::INVALID_VALUE);
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
void DrawRangeElementsBaseVertex(GLenum m, GLuint start, GLuint end, GLsizei n, GLenum t, const void* idx, GLint bv) {
    CTX()
    if (end < start) return c->setError(gl::INVALID_VALUE);
    c->drawElements(m, n, t, idx, 1, bv);
}
void DrawArraysIndirect(GLenum m, const void* indirect) {
    CTX()
    c->drawIndirect(m, 0, indirect);
}
void DrawElementsIndirect(GLenum m, GLenum t, const void* indirect) {
    CTX()
    if (!t) return c->setError(gl::INVALID_ENUM, t);
    c->drawIndirect(m, t, indirect);
}
void DispatchCompute(GLuint x, GLuint y, GLuint z) {
    CTX()
    c->dispatch(x, y, z);
}
void DispatchComputeIndirect(GLintptr offset) {
    CTX()
    c->dispatchIndirect(offset);
}
void BlendBarrier() {}  // coherent blending: no advanced blend equations are offered, nothing to order
void PatchParameteri(GLenum, GLint) {}  // no tessellation stage: accepted and ignored
void MemoryBarrier(GLbitfield) {
    CTX()
    c->barrier();
}
void MemoryBarrierByRegion(GLbitfield b) { MemoryBarrier(b); }

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
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint x = names[i];
        if (!x) continue;
        auto it = c->share->textures.find(x);
        if (it == c->share->textures.end()) continue;
        if (it->second->image) c->keep(it->second->image);
        for (auto& u : c->units)
            for (GLuint* b : {&u.tex2D, &u.texCube, &u.tex3D, &u.tex2DArray, &u.texExternal, &u.tex2DMS, &u.texCubeArray, &u.tex2DMSArray, &u.texBuffer})
                if (*b == x) *b = 0;
        // Only the bound framebuffers (draw / read) lose the attachment; the others keep the name (they are incomplete until
        // it is replaced, and still answer attachment queries with it).
        for (auto& [k, f] : c->fbos) {
            if (k != c->drawFbo && k != c->readFbo) continue;
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
    if (u < gl::TEXTURE0 || u >= gl::TEXTURE0 + kMaxTextureUnits) return c->setError(gl::INVALID_ENUM, u);
    c->activeUnit = int(u - gl::TEXTURE0);
}
void BindTexture(GLenum target, GLuint name) {
    CTX()
    GLuint* slot = c->textureBinding(target, c->activeUnit);
    if (!slot) return c->setError(gl::INVALID_ENUM, target);
    if (name) {
        Texture* t = c->texture(name);
        if (!t) {
            auto nt = std::make_unique<Texture>();
            nt->name = name;
            t = nt.get();
            c->share->textures[name] = std::move(nt);
            c->share->nextName = std::max(c->share->nextName, name + 1);
        }
        if (t->target && t->target != target) {
            // Legitimate GL ES error (a name keeps the target of its first bind), but the usual
            // cause is a guest binding a SurfaceTexture name (EXTERNAL_OES) as TEXTURE_2D: say which.
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1) < 16)
                logf("gles.diag BindTexture name=%u target=0x%X previous_target=0x%X unit=%d thread=%zx", name, target,
                     t->target, c->activeUnit, std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xFFFFFF);
            return c->setError(gl::INVALID_OPERATION, target);
        }
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
    if (border) return c->setError(gl::INVALID_VALUE, target);
    c->texImage(target, level, GLenum(ifmt), w, h, 1, fmt, type, px, false, 0, 0, 0, false, 0);
}
void TexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLenum type, const void* px) {
    CTX()
    c->texImage(target, level, 0, w, h, 1, fmt, type, px, true, x, y, 0, false, 0);
}
void TexImage3D(GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLsizei d, GLint border, GLenum fmt, S type, const void* px) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE, target);
    c->texImage(target, level, GLenum(ifmt), w, h, d, fmt, GLenum(type), px, false, 0, 0, 0, false, 0);
}
void TexSubImage3D(GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, S fmt, S type, const void* px) {
    CTX()
    c->texImage(target, level, 0, w, h, d, GLenum(fmt), GLenum(type), px, true, x, y, z, false, 0);
}
void CompressedTexImage2D(GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLint border, GLsizei size, const void* data) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE, target);
    c->texImage(target, level, ifmt, w, h, 1, 0, 0, data, false, 0, 0, 0, true, size);
}
void CompressedTexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLsizei size, S data) {
    CTX()
    c->texImage(target, level, fmt, w, h, 1, 0, 0, reinterpret_cast<const void*>(data), true, x, y, 0, true, size);
}
void CompressedTexImage3D(GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d, GLint border, GLsizei size, S data) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE, target);
    c->texImage(target, level, ifmt, w, h, d, 0, 0, reinterpret_cast<const void*>(data), false, 0, 0, 0, true, size);
}
void CompressedTexSubImage3D(GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, S fmt, S size, S data) {
    CTX()
    c->texImage(target, level, GLenum(fmt), w, h, d, 0, 0, reinterpret_cast<const void*>(data), true, x, y, z, true, GLsizei(size));
}
void TexStorage2D(GLenum target, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h) {
    CTX()
    if (target != gl::TEXTURE_2D && target != gl::TEXTURE_CUBE_MAP) return c->setError(gl::INVALID_ENUM, target);
    c->texStorage(target, levels, ifmt, w, h, target == gl::TEXTURE_CUBE_MAP ? 6 : 1, 0);
}
void TexStorage3D(GLenum target, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d) {
    CTX()
    if (target != gl::TEXTURE_3D && target != gl::TEXTURE_2D_ARRAY && target != gl::TEXTURE_CUBE_MAP_ARRAY) return c->setError(gl::INVALID_ENUM, target);
    c->texStorage(target, levels, ifmt, w, h, d, 0);
}
void CopyTexImage2D(GLenum target, GLint level, GLenum ifmt, GLint x, GLint y, GLsizei w, GLsizei h, GLint border) {
    CTX()
    if (border) return c->setError(gl::INVALID_VALUE, target);
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
// The enumerated texture / sampler parameters take only their own values (INVALID_ENUM otherwise).
bool validTexParamEnum(GLenum p, double v) {
    switch (p) {
        case gl::TEXTURE_MIN_FILTER: case gl::TEXTURE_MAG_FILTER: case gl::TEXTURE_WRAP_S: case gl::TEXTURE_WRAP_T:
        case gl::TEXTURE_WRAP_R: case gl::TEXTURE_COMPARE_MODE: case gl::TEXTURE_COMPARE_FUNC: break;
        default: return true;  // not an enumerated parameter (LOD, levels, swizzle...)
    }
    if (v < 0 || v > 4294967295.0 || v != double(GLenum(v))) return false;
    const GLenum e = GLenum(v);
    switch (p) {
        case gl::TEXTURE_MIN_FILTER:
            return e == gl::NEAREST || e == gl::LINEAR || e == gl::NEAREST_MIPMAP_NEAREST || e == gl::LINEAR_MIPMAP_NEAREST ||
                   e == gl::NEAREST_MIPMAP_LINEAR || e == gl::LINEAR_MIPMAP_LINEAR;
        case gl::TEXTURE_MAG_FILTER: return e == gl::NEAREST || e == gl::LINEAR;
        case gl::TEXTURE_WRAP_S: case gl::TEXTURE_WRAP_T: case gl::TEXTURE_WRAP_R:
            return e == gl::REPEAT || e == gl::CLAMP_TO_EDGE || e == gl::MIRRORED_REPEAT || e == gl::CLAMP_TO_BORDER;
        case gl::TEXTURE_COMPARE_MODE: return e == gl::NONE || e == gl::COMPARE_REF_TO_TEXTURE;
        case gl::TEXTURE_COMPARE_FUNC: return e >= gl::NEVER && e <= gl::ALWAYS;
        default: return true;
    }
}
void texParam(Context* c, Texture* t, SamplerParams& s, GLenum p, double v, const double* vv) {
    if (!validTexParamEnum(p, v)) return c->setError(gl::INVALID_ENUM, p);
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
        case gl::TEXTURE_BORDER_COLOR:
            if (vv) {
                for (int i = 0; i < 4; ++i) s.border[i] = float(vv[i]);
                s.borderInt = false;
            }
            return;
        default: break;
    }
    if (!t) return c->setError(gl::INVALID_ENUM, p);
    switch (p) {
        case gl::TEXTURE_BASE_LEVEL: t->baseLevel = GLint(v); return;
        case gl::TEXTURE_MAX_LEVEL: t->maxLevel = GLint(v); return;
        case gl::TEXTURE_SWIZZLE_R: case gl::TEXTURE_SWIZZLE_G: case gl::TEXTURE_SWIZZLE_B: case gl::TEXTURE_SWIZZLE_A:
            t->swizzle[p - gl::TEXTURE_SWIZZLE_R] = GLenum(v);
            return;
        case gl::DEPTH_STENCIL_TEXTURE_MODE: t->depthStencilMode = GLenum(v); return;
        case gl::TEXTURE_SRGB_DECODE_EXT: t->srgbDecode = GLenum(v) == gl::DECODE_EXT; return;
        default: c->setError(gl::INVALID_ENUM, p);
    }
}
// Only the border colour differs from the plain forms: its four integers are kept exactly.
void setBorderInt(SamplerParams& s, const GLint* v) {
    for (int i = 0; i < 4; ++i) s.borderI[i] = v[i], s.border[i] = float(v[i]);
    s.borderInt = true;
}
// The border colour as four GLint: the stored integers, or the float colour converted with the glGet*iv rule (-1..1 spans the range).
void borderAsInt(const SamplerParams& s, GLint* v) {
    for (int i = 0; i < 4; ++i)
        v[i] = s.borderInt ? s.borderI[i] : GLint(std::clamp(double(s.border[i]) * 2147483647.0, -2147483648.0, 2147483647.0));
}
void TexParameteri(GLenum target, GLenum p, GLint v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
    texParam(c, t, t->sampler, p, v, nullptr);
}
void TexParameterf(GLenum target, GLenum p, GLfloat v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
    texParam(c, t, t->sampler, p, v, nullptr);
}
void TexParameteriv(GLenum target, GLenum p, const GLint* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
    double vv[4] = {double(v[0]), 0, 0, 0};
    if (p == gl::TEXTURE_BORDER_COLOR)
        for (int i = 0; i < 4; ++i) vv[i] = v[i] / 2147483647.0;
    texParam(c, t, t->sampler, p, vv[0], vv);
}
void TexParameterfv(GLenum target, GLenum p, const GLfloat* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
    double vv[4] = {v[0], 0, 0, 0};
    if (p == gl::TEXTURE_BORDER_COLOR)
        for (int i = 0; i < 4; ++i) vv[i] = v[i];
    texParam(c, t, t->sampler, p, vv[0], vv);
}
void GetTexParameteriv(GLenum target, GLenum p, GLint* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
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
        case gl::TEXTURE_BORDER_COLOR: borderAsInt(s, v); return;
        default: c->setError(gl::INVALID_ENUM, target);
    }
}
void GetTexParameterfv(GLenum target, GLenum p, GLfloat* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
    if (p == gl::TEXTURE_MIN_LOD) { *v = t->sampler.minLod; return; }
    if (p == gl::TEXTURE_MAX_LOD) { *v = t->sampler.maxLod; return; }
    if (p == gl::TEXTURE_MAX_ANISOTROPY_EXT) { *v = t->sampler.maxAnisotropy; return; }
    if (p == gl::TEXTURE_BORDER_COLOR) {
        for (int i = 0; i < 4; ++i) v[i] = t->sampler.borderInt ? float(t->sampler.borderI[i]) : t->sampler.border[i];
        return;
    }
    GLint x = 0;
    GetTexParameteriv(target, p, &x);
    *v = GLfloat(x);
}
// ---- pure integer parameters (ES 3.2 / EXT_texture_border_clamp): glTexParameterI[u]iv, glSamplerParameterI[u]iv ------------
void TexParameterIiv(GLenum target, GLenum p, const GLint* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
    if (p == gl::TEXTURE_BORDER_COLOR) return setBorderInt(t->sampler, v);
    texParam(c, t, t->sampler, p, v[0], nullptr);
}
void TexParameterIuiv(GLenum target, GLenum p, const GLuint* v) { TexParameterIiv(target, p, reinterpret_cast<const GLint*>(v)); }
void GetTexParameterIiv(GLenum target, GLenum p, GLint* v) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_ENUM, target);
    if (p == gl::TEXTURE_BORDER_COLOR) return borderAsInt(t->sampler, v);
    GetTexParameteriv(target, p, v);
}
void GetTexParameterIuiv(GLenum target, GLenum p, GLuint* v) { GetTexParameterIiv(target, p, reinterpret_cast<GLint*>(v)); }

void EGLImageTargetTexture2DOES(GLenum target, void* image) {
    CTX()
    Texture* t = c->boundTexture(target);
    if (!t || (target != gl::TEXTURE_2D && target != gl::TEXTURE_EXTERNAL_OES)) return c->setError(gl::INVALID_ENUM, target);
    if (t->image) c->keep(t->image);
    if (!eglImageToTexture(image, *t)) c->setError(gl::INVALID_OPERATION, target);
}

// ---- buffer textures (glTexBuffer, ES 3.1 / OES_texture_buffer) --------------------------------------
// The data store is a range of a buffer object, read through a texel buffer descriptor (VkBufferView) by
// samplerBuffer uniforms. Formats are the ones whose Vulkan texel is exactly the GL texel: the 96-bit
// RGB32F/I/UI are widened to 128 bits by this layer's format table, so they cannot view a tightly packed
// buffer and are rejected with INVALID_ENUM.
// Known gap: GL_EXT_texture_buffer requires RGB32F/I/UI; supporting them needs 96-bit texel buffer formats (12-byte
// stride views) in VPVulkan's format table, which it does not have.
const FormatInfo* texelBufferFormat(GLenum ifmt) {
    static const GLenum ok[] = {gl::R8, gl::R16F, gl::R32F, gl::R8I, gl::R16I, gl::R32I, gl::R8UI, gl::R16UI, gl::R32UI, gl::RG8,
                                gl::RG16F, gl::RG32F, gl::RG8I, gl::RG16I, gl::RG32I, gl::RG8UI, gl::RG16UI, gl::RG32UI, gl::RGBA8,
                                gl::RGBA16F, gl::RGBA32F, gl::RGBA8I, gl::RGBA16I, gl::RGBA32I, gl::RGBA8UI, gl::RGBA16UI, gl::RGBA32UI};
    for (GLenum f : ok)
        if (f == ifmt) return formatFor(ifmt);
    return nullptr;
}
void texBufferImpl(Context* c, GLenum target, GLenum ifmt, GLuint buf, GLintptr off, GLsizeiptr size, bool whole) {
    if (target != gl::TEXTURE_BUFFER) return c->setError(gl::INVALID_ENUM, target);
    Texture* t = c->texture(c->units[size_t(c->activeUnit)].texBuffer);
    if (!t) return c->setError(gl::INVALID_OPERATION, target);
    const FormatInfo* fi = texelBufferFormat(ifmt);
    if (!fi) return c->setError(gl::INVALID_ENUM, ifmt);
    if (!buf) {
        t->bufferName = 0, t->bufferOffset = 0, t->bufferSize = 0, t->bufferFormat = fi;
        return;
    }
    Buffer* b = c->buffer(buf);
    if (!b) return c->setError(gl::INVALID_OPERATION, target);
    if (whole) off = 0, size = b->size;
    else if (off < 0 || size <= 0 || off + size > b->size) return c->setError(gl::INVALID_VALUE, target);
    else if (off % 256) return c->setError(gl::INVALID_VALUE, target);
    const GLsizeiptr maxBytes = GLsizeiptr(std::min<uint64_t>(c->dev->props.limits.maxTexelBufferElements, 1u << 27)) * fi->texelBytes;
    size = std::min<GLsizeiptr>(size - size % fi->texelBytes, maxBytes);  // whole texels only
    t->bufferName = buf, t->bufferOffset = off, t->bufferSize = size, t->bufferFormat = fi;
    if (t->bufferView) c->keep(t->bufferView);
    t->bufferView.reset();
}
void TexBuffer(GLenum target, GLenum ifmt, GLuint buf) {
    CTX()
    texBufferImpl(c, target, ifmt, buf, 0, 0, true);
}
void TexBufferRange(GLenum target, GLenum ifmt, GLuint buf, GLintptr off, GLsizeiptr size) {
    CTX()
    texBufferImpl(c, target, ifmt, buf, off, size, false);
}

// ---- glGetTexLevelParameter ----------------------------------------------------------------------------
// Fills `*out` for the level of the texture bound to `target`; false after raising the error.
bool texLevelParameter(Context* c, GLenum target, GLint level, GLenum p, double* out) {
    const bool face = target >= gl::TEXTURE_CUBE_MAP_POSITIVE_X && target <= gl::TEXTURE_CUBE_MAP_NEGATIVE_Z;
    const bool known = face || target == gl::TEXTURE_2D || target == gl::TEXTURE_3D || target == gl::TEXTURE_2D_ARRAY ||
                       target == gl::TEXTURE_2D_MULTISAMPLE || target == gl::TEXTURE_2D_MULTISAMPLE_ARRAY ||
                       target == gl::TEXTURE_CUBE_MAP_ARRAY || target == gl::TEXTURE_BUFFER;
    if (!known) return c->setError(gl::INVALID_ENUM, target), false;
    Texture* t = c->boundTexture(face ? gl::TEXTURE_CUBE_MAP : target);
    if (!t) return c->setError(gl::INVALID_OPERATION, target), false;
    if (level < 0 || level > 15) return c->setError(gl::INVALID_VALUE, target), false;
    if (target == gl::TEXTURE_BUFFER) {
        if (level != 0) return c->setError(gl::INVALID_VALUE, target), false;
        const FormatInfo* bf = t->bufferFormat;
        switch (p) {
            case gl::TEXTURE_WIDTH: *out = bf && t->bufferName ? double(t->bufferSize / bf->texelBytes) : 0; return true;
            case gl::TEXTURE_HEIGHT: case gl::TEXTURE_DEPTH: *out = 1; return true;
            case gl::TEXTURE_INTERNAL_FORMAT: *out = bf ? bf->internal : gl::R8; return true;
            case gl::TEXTURE_BUFFER_OFFSET: *out = double(t->bufferOffset); return true;
            case gl::TEXTURE_BUFFER_SIZE: *out = double(t->bufferSize); return true;
            case gl::TEXTURE_BUFFER_DATA_STORE_BINDING: *out = t->bufferName; return true;
            default: break;
        }
    }
    const Texture::Level* lv = size_t(level) < t->levels.size() && t->levels[size_t(level)].format ? &t->levels[size_t(level)] : nullptr;
    const FormatInfo* f = lv ? lv->format : nullptr;
    switch (p) {
        case gl::TEXTURE_WIDTH: *out = lv ? lv->w : 0; return true;
        case gl::TEXTURE_HEIGHT: *out = lv ? lv->h : 0; return true;
        case gl::TEXTURE_DEPTH:
            *out = lv ? (face || target == gl::TEXTURE_2D || target == gl::TEXTURE_2D_MULTISAMPLE ? 1 : lv->d) : 0;
            return true;
        case gl::TEXTURE_INTERNAL_FORMAT: *out = f ? f->internal : gl::RGBA; return true;
        case gl::TEXTURE_RED_SIZE: *out = f ? f->redBits : 0; return true;
        case gl::TEXTURE_GREEN_SIZE: *out = f ? f->greenBits : 0; return true;
        case gl::TEXTURE_BLUE_SIZE: *out = f ? f->blueBits : 0; return true;
        case gl::TEXTURE_ALPHA_SIZE: *out = f ? f->alphaBits : 0; return true;
        case gl::TEXTURE_DEPTH_SIZE: *out = f ? f->depthBits : 0; return true;
        case gl::TEXTURE_STENCIL_SIZE: *out = f ? f->stencilBits : 0; return true;
        case gl::TEXTURE_SHARED_SIZE: *out = f && f->internal == gl::RGB9_E5 ? 5 : 0; return true;
        case gl::TEXTURE_RED_TYPE: *out = f && f->redBits ? f->componentType : 0; return true;
        case gl::TEXTURE_GREEN_TYPE: *out = f && f->greenBits ? f->componentType : 0; return true;
        case gl::TEXTURE_BLUE_TYPE: *out = f && f->blueBits ? f->componentType : 0; return true;
        case gl::TEXTURE_ALPHA_TYPE: *out = f && f->alphaBits ? f->componentType : 0; return true;
        case gl::TEXTURE_DEPTH_TYPE: *out = f && f->depthBits ? f->componentType : 0; return true;
        case gl::TEXTURE_COMPRESSED: *out = f && f->compressed ? 1 : 0; return true;
        case gl::TEXTURE_SAMPLES: *out = lv ? t->samples : 0; return true;
        case gl::TEXTURE_FIXED_SAMPLE_LOCATIONS: *out = t->samples ? (t->fixedSampleLocations ? 1 : 0) : 1; return true;
        case gl::TEXTURE_BUFFER_OFFSET: case gl::TEXTURE_BUFFER_SIZE: case gl::TEXTURE_BUFFER_DATA_STORE_BINDING:
        default: break;
    }
    return c->setError(gl::INVALID_ENUM, p), false;
}
void GetTexLevelParameteriv(GLenum target, GLint level, GLenum p, GLint* v) {
    CTX()
    double d = 0;
    if (texLevelParameter(c, target, level, p, &d)) *v = GLint(d);
}
void GetTexLevelParameterfv(GLenum target, GLint level, GLenum p, GLfloat* v) {
    CTX()
    double d = 0;
    if (texLevelParameter(c, target, level, p, &d)) *v = GLfloat(d);
}

// ---- multisample textures --------------------------------------------------------------------------------
// The image has renderSamples(samples) samples; the texture reports the count it was given and its fixed-location flag.
void TexStorage2DMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h, GLboolean fixed) {
    CTX()
    if (target != gl::TEXTURE_2D_MULTISAMPLE) return c->setError(gl::INVALID_ENUM, target);
    if (samples < 1 || w < 1 || h < 1) return c->setError(gl::INVALID_VALUE, target);
    if (samples > 4) return c->setError(gl::INVALID_OPERATION, target);
    c->texStorage(target, 1, ifmt, w, h, 1, samples);
    if (Texture* t = c->boundTexture(target)) t->fixedSampleLocations = fixed != 0, t->samples = samples;
}
void TexImage2DMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h, GLboolean fixed) {
    CTX()
    if (target != gl::TEXTURE_2D_MULTISAMPLE) return c->setError(gl::INVALID_ENUM, target);
    if (samples < 1 || w < 1 || h < 1) return c->setError(gl::INVALID_VALUE, target);
    if (samples > 4) return c->setError(gl::INVALID_OPERATION, target);
    Texture* t = c->boundTexture(target);
    if (!t) return c->setError(gl::INVALID_OPERATION, target);
    t->immutable = false;  // the mutable form may be re-specified
    c->texStorage(target, 1, ifmt, w, h, 1, samples);
    t->immutable = false, t->fixedSampleLocations = fixed != 0, t->samples = samples;
}
void TexStorage3DMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d, GLboolean fixed) {
    CTX()
    c->texStorage(target, 1, ifmt, w, h, d, samples);
    if (Texture* t = c->boundTexture(target)) t->fixedSampleLocations = fixed != 0;
}

// ---- images ---------------------------------------------------------------------------------------------------
void CopyImageSubData(GLuint sn, GLenum st, GLint sl, GLint sx, GLint sy, GLint sz, GLuint dn, GLenum dt, GLint dl, GLint dx, GLint dy,
                      GLint dz, GLsizei w, GLsizei h, S d) {
    CTX()
    c->copyImageSubData(sn, st, sl, sx, sy, sz, dn, dt, dl, dx, dy, dz, w, h, GLsizei(d));
}
void BindImageTexture(GLuint unit, GLuint tex, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format) {
    CTX()
    if (unit >= kMaxImageUnits || level < 0 || layer < 0) return c->setError(gl::INVALID_VALUE);
    if (access != gl::READ_ONLY && access != gl::WRITE_ONLY && access != gl::READ_WRITE) return c->setError(gl::INVALID_ENUM, access);
    if (tex && !c->texture(tex)) return c->setError(gl::INVALID_VALUE);
    if (tex && !formatFor(format)) return c->setError(gl::INVALID_VALUE, format);
    if (tex) {
        Texture* t = c->texture(tex);
        t->wantStorage = true;  // an image defined later is created with STORAGE_BIT
        c->makeStorage(*t);
    }
    c->images[unit] = {tex, level, layer, layered, access, format};
}
// Sampler objects
void GenSamplers(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->genName();
        c->share->samplers[name] = std::make_unique<SamplerObj>();
        c->share->samplers[name]->name = name;
        out[i] = name;
    }
}
void DeleteSamplers(GLsizei n, const GLuint* names) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
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
void SamplerParameteriv(GLuint s, GLenum p, const GLint* v) {
    CTX()
    double vv[4] = {double(v[0]), 0, 0, 0};
    if (p == gl::TEXTURE_BORDER_COLOR)
        for (int i = 0; i < 4; ++i) vv[i] = v[i] / 2147483647.0;
    if (SamplerObj* o = samplerObj(c, s)) texParam(c, nullptr, o->params, p, vv[0], vv);
}
void SamplerParameterIiv(GLuint s, GLenum p, const GLint* v) {
    CTX()
    SamplerObj* o = samplerObj(c, s);
    if (!o) return;
    if (p == gl::TEXTURE_BORDER_COLOR) return setBorderInt(o->params, v);
    texParam(c, nullptr, o->params, p, v[0], nullptr);
}
void SamplerParameterIuiv(GLuint s, GLenum p, const GLuint* v) { SamplerParameterIiv(s, p, reinterpret_cast<const GLint*>(v)); }
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
        case gl::TEXTURE_BORDER_COLOR: borderAsInt(o->params, v); return;
        default: c->setError(gl::INVALID_ENUM, p);
    }
}
void GetSamplerParameterIiv(GLuint s, GLenum p, GLint* v) { GetSamplerParameteriv(s, p, v); }
void GetSamplerParameterIuiv(GLuint s, GLenum p, GLuint* v) { GetSamplerParameteriv(s, p, reinterpret_cast<GLint*>(v)); }
void GetSamplerParameterfv(GLuint s, GLenum p, GLfloat* v) {
    CTX()
    SamplerObj* o = samplerObj(c, s);
    if (!o) return;
    if (p == gl::TEXTURE_MIN_LOD) { *v = o->params.minLod; return; }
    if (p == gl::TEXTURE_MAX_LOD) { *v = o->params.maxLod; return; }
    if (p == gl::TEXTURE_BORDER_COLOR) {
        for (int i = 0; i < 4; ++i) v[i] = o->params.borderInt ? float(o->params.borderI[i]) : o->params.border[i];
        return;
    }
    GLint x = 0;
    GetSamplerParameteriv(s, p, &x);
    *v = GLfloat(x);
}

// ---- renderbuffers / framebuffers ------------------------------------------------------------------------
void GenRenderbuffers(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->genName();
        c->share->renderbuffers[name] = std::make_unique<Renderbuffer>();
        c->share->renderbuffers[name]->name = name;
        out[i] = name;
    }
}
void DeleteRenderbuffers(GLsizei n, const GLuint* names) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint x = names[i];
        auto it = c->share->renderbuffers.find(x);
        if (!x || it == c->share->renderbuffers.end()) continue;
        if (it->second->image) c->keep(it->second->image);
        if (c->renderbuffer == x) c->renderbuffer = 0;
        for (auto& [k, f] : c->fbos) {
            if (k != c->drawFbo && k != c->readFbo) continue;  // see DeleteTextures
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
    return c->rbo(n) && c->rbo(n)->bound ? 1 : 0;
}
void BindRenderbuffer(GLenum target, GLuint n) {
    CTX()
    if (target != gl::RENDERBUFFER) return c->setError(gl::INVALID_ENUM, target);
    if (n && !c->rbo(n)) {
        c->share->renderbuffers[n] = std::make_unique<Renderbuffer>();
        c->share->renderbuffers[n]->name = n;
    }
    if (n) c->rbo(n)->bound = true;
    c->renderbuffer = n;
}
void RenderbufferStorageMultisample(GLenum target, GLsizei samples, GLenum ifmt, GLsizei w, GLsizei h) {
    CTX()
    if (target != gl::RENDERBUFFER) return c->setError(gl::INVALID_ENUM, target);
    Renderbuffer* r = c->rbo(c->renderbuffer);
    if (!r) return c->setError(gl::INVALID_OPERATION, target);
    if (w < 0 || h < 0 || w > 16384 || h > 16384 || samples < 0) return c->setError(gl::INVALID_VALUE, target);
    if (samples > 4) return c->setError(gl::INVALID_OPERATION, target);  // above MAX_SAMPLES / the format's maximum
    // Unsized base formats (GL_RGB, GL_DEPTH_COMPONENT, ...) are texture formats, not renderbuffer ones.
    if (ifmt == gl::RGB || ifmt == gl::RGBA || ifmt == 0x1906 /*ALPHA*/ || ifmt == 0x1909 /*LUMINANCE*/ || ifmt == 0x190A /*LUMINANCE_ALPHA*/ ||
        ifmt == gl::DEPTH_COMPONENT || ifmt == 0x84F9 /*DEPTH_STENCIL*/)
        return c->setError(gl::INVALID_ENUM, target);
    const FormatInfo* f = formatFor(ifmt);
    if (!f || f->compressed || (!f->renderable && !(f->depth || f->stencil))) return c->setError(gl::INVALID_ENUM, target);
    if (r->image) c->keep(r->image);
    const uint32_t n = samples > 0 ? renderSamples(uint32_t(samples)) : 1;
    // Game MSAA is memoryless: a multisample renderbuffer stores only this single-sample image. Its passes render in a transient
    // multisample twin (Context::msTwin) that resolves into it and is seeded from it (FbTarget::msSamples); the stencil plane of
    // the seed is read through a stencil view (format views).
    ImageP img = createImage2D(c->dev, f, uint32_t(std::max(w, 1)), uint32_t(std::max(h, 1)), 1, 1,
                               (f->depth || f->stencil ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               f->stencil && n > 1 ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0);
    if (!img) return c->setError(gl::OUT_OF_MEMORY, target);
    c->imageInit(*img);
    r->image = img;
    r->internalFormat = ifmt, r->width = w, r->height = h, r->samples = GLsizei(n > 1 ? n : 0);
}
void RenderbufferStorage(GLenum target, GLenum ifmt, GLsizei w, GLsizei h) { RenderbufferStorageMultisample(target, 0, ifmt, w, h); }
void GetRenderbufferParameteriv(GLenum target, GLenum p, GLint* v) {
    CTX()
    if (target != gl::RENDERBUFFER) return c->setError(gl::INVALID_ENUM, target);
    Renderbuffer* r = c->rbo(c->renderbuffer);
    if (!r) return c->setError(gl::INVALID_OPERATION, p);
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
        default: c->setError(gl::INVALID_ENUM, p);
    }
}
void GetInternalformativ(GLenum target, GLenum ifmt, GLenum p, GLsizei n, GLint* v) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    if (target != 0x8D41 /* RENDERBUFFER */ && target != 0x9100 /* TEXTURE_2D_MULTISAMPLE */ && target != 0x9102 /* TEXTURE_2D_MULTISAMPLE_ARRAY */)
        return c->setError(gl::INVALID_ENUM, target);
    if (p != 0x80A9 && p != 0x9380) return c->setError(gl::INVALID_ENUM, p);
    {
        const FormatInfo* rf = formatFor(ifmt);
        const bool unsized = ifmt == 0x1902 || (ifmt >= 0x1906 && ifmt <= 0x190A) || ifmt == 0x84F9;
        if (unsized || !rf || !(rf->renderable || rf->depth || rf->stencil)) return c->setError(gl::INVALID_ENUM, ifmt);
    }
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
    c->setError(gl::INVALID_ENUM, p);
}
void GenFramebuffers(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->nextFbo++;
        c->fbos[name] = std::make_unique<Framebuffer>();
        c->fbos[name]->name = name;
        out[i] = name;
    }
}
void DeleteFramebuffers(GLsizei n, const GLuint* names) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
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
    return n && c->fbos.count(n) && c->fbos[n]->bound ? 1 : 0;
}
void BindFramebuffer(GLenum target, GLuint n) {
    CTX()
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER)
        return c->setError(gl::INVALID_ENUM, target);
    if (n && !c->fbos.count(n)) {
        c->fbos[n] = std::make_unique<Framebuffer>();
        c->fbos[n]->name = n;
        c->nextFbo = std::max(c->nextFbo, n + 1);
    }
    if (n) c->fbos[n]->bound = true;
    if (target != gl::READ_FRAMEBUFFER) c->drawFbo = n;
    if (target != gl::DRAW_FRAMEBUFFER) c->readFbo = n;
}
Framebuffer* targetFbo(Context* c, GLenum target) {
    GLuint n = target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo;
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER) {
        c->setError(gl::INVALID_ENUM, target);
        return nullptr;
    }
    if (!n) {
        c->setError(gl::INVALID_OPERATION, target);
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
        c->setError(gl::INVALID_ENUM, att);
        return false;
    }
    return true;
}
// `samples` > 0: EXT_multisampled_render_to_texture (the single-sample texture is rendered with that many samples).
static void framebufferTexture2D(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level, GLsizei samples) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    if (samples < 0 || samples > 4) return c->setError(gl::INVALID_VALUE, target);  // above MAX_SAMPLES_EXT
    Attachment a;
    if (tex) {
        const bool cube = textarget >= gl::TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= gl::TEXTURE_CUBE_MAP_NEGATIVE_Z;
        if (textarget != gl::TEXTURE_2D && textarget != gl::TEXTURE_2D_MULTISAMPLE && !cube) return c->setError(gl::INVALID_ENUM, textarget);
        Texture* t = c->texture(tex);
        if (!t) return c->setError(gl::INVALID_OPERATION, target);
        if (level < 0 || level > 14) return c->setError(gl::INVALID_VALUE, target);  // log2(MAX_TEXTURE_SIZE)
        // The target must suit the texture (once it has one: a texture name that was never bound has none yet).
        if (t->target && (cube ? t->target != gl::TEXTURE_CUBE_MAP : t->target != textarget)) return c->setError(gl::INVALID_OPERATION, target);
        a.kind = Attachment::Tex, a.name = tex, a.level = level, a.samples = samples;
        if (textarget >= gl::TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= gl::TEXTURE_CUBE_MAP_NEGATIVE_Z)
            a.layer = GLint(textarget - gl::TEXTURE_CUBE_MAP_POSITIVE_X);
    }
    setAttachment(c, f, att, a);
}
void FramebufferTexture2D(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level) {
    framebufferTexture2D(target, att, textarget, tex, level, 0);
}
void FramebufferTextureLayer(GLenum target, GLenum att, GLuint tex, GLint level, GLint layer) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    Attachment a;
    if (tex) {
        Texture* t = c->texture(tex);
        if (!t) return c->setError(gl::INVALID_OPERATION, target);
        // A layer of a 3D, 2D array (or cube array / multisample array) texture.
        if (t->target && t->target != gl::TEXTURE_3D && t->target != gl::TEXTURE_2D_ARRAY && t->target != gl::TEXTURE_CUBE_MAP_ARRAY &&
            t->target != gl::TEXTURE_2D_MULTISAMPLE_ARRAY)
            return c->setError(gl::INVALID_OPERATION, target);
        if (layer < 0 || layer > 2047 || level < 0 || level > 14) return c->setError(gl::INVALID_VALUE, target);  // MAX_3D_TEXTURE_SIZE / MAX_ARRAY_TEXTURE_LAYERS - 1
        a.kind = Attachment::Tex, a.name = tex, a.level = level, a.layer = layer;
    }
    setAttachment(c, f, att, a);
}
void FramebufferTexture(GLenum target, GLenum att, GLuint tex, GLint level) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    Attachment a;
    if (tex) {
        if (!c->texture(tex)) return c->setError(gl::INVALID_OPERATION, target);
        a.kind = Attachment::Tex, a.name = tex, a.level = level;
    }
    setAttachment(c, f, att, a);
}
void FramebufferTexture3D(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level, GLint zoffset) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    if (tex && textarget != gl::TEXTURE_3D) return c->setError(gl::INVALID_ENUM, textarget);
    Attachment a;
    if (tex) {
        Texture* t = c->texture(tex);
        if (!t || t->target != gl::TEXTURE_3D) return c->setError(gl::INVALID_OPERATION, target);
        if (zoffset < 0 || zoffset >= 2048) return c->setError(gl::INVALID_VALUE, target);  // MAX_3D_TEXTURE_SIZE
        a.kind = Attachment::Tex, a.name = tex, a.level = level, a.layer = zoffset;
    }
    setAttachment(c, f, att, a);
}
void FramebufferTextureMultiviewOVR(GLenum target, GLenum att, GLuint tex, GLint level, GLint baseView, GLsizei numViews) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    if (tex && (numViews < 1 || numViews > 2)) return c->setError(gl::INVALID_VALUE, target);
    Attachment a;
    if (tex) a.kind = Attachment::Tex, a.name = tex, a.level = level, a.baseView = baseView, a.numViews = numViews;
    setAttachment(c, f, att, a);
}
void FramebufferTextureMultisampleMultiviewOVR(GLenum target, GLenum att, GLuint tex, GLint level, GLsizei, GLint baseView, GLsizei numViews) {
    FramebufferTextureMultiviewOVR(target, att, tex, level, baseView, numViews);
}
void FramebufferTexture2DMultisampleEXT(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level, GLsizei samples) {
    framebufferTexture2D(target, att, textarget, tex, level, samples);
}
void FramebufferRenderbuffer(GLenum target, GLenum att, GLenum rbtarget, GLuint rb) {
    CTX()
    Framebuffer* f = targetFbo(c, target);
    if (!f) return;
    if (rb && rbtarget != gl::RENDERBUFFER) return c->setError(gl::INVALID_ENUM, target);
    Attachment a;
    if (rb) {
        if (!c->rbo(rb)) return c->setError(gl::INVALID_OPERATION, target);
        a.kind = Attachment::Rbo, a.name = rb;
    }
    setAttachment(c, f, att, a);
}
GLenum CheckFramebufferStatus(GLenum target) {
    CTX(0)
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER) {
        c->setError(gl::INVALID_ENUM, target);
        return 0;
    }
    GLuint n = target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo;
    return c->resolveFramebuffer(n, target != gl::READ_FRAMEBUFFER).status;
}
void GetFramebufferAttachmentParameteriv(GLenum target, GLenum att, GLenum p, GLint* v) {
    CTX()
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER) return c->setError(gl::INVALID_ENUM, target);
    GLuint n = target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo;
    if (n == 0) {
        // Default framebuffer: BACK / DEPTH / STENCIL.
        if (att != gl::BACK && att != gl::COLOR && att != gl::DEPTH && att != gl::STENCIL) {
            const bool known = att == gl::DEPTH_ATTACHMENT || att == gl::STENCIL_ATTACHMENT || att == gl::DEPTH_STENCIL_ATTACHMENT ||
                               (att >= gl::COLOR_ATTACHMENT0 && att < gl::COLOR_ATTACHMENT0 + kMaxDrawBuffers);
            return c->setError(known ? gl::INVALID_OPERATION : gl::INVALID_ENUM, att);
        }
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
            default: return c->setError(gl::INVALID_ENUM, target);
        }
    }
    Framebuffer* f = c->fbo(n);
    const Attachment* a = nullptr;
    if (att >= gl::COLOR_ATTACHMENT0 && att < gl::COLOR_ATTACHMENT0 + kMaxDrawBuffers) a = &f->color[att - gl::COLOR_ATTACHMENT0];
    else if (att == gl::DEPTH_ATTACHMENT || att == gl::DEPTH_STENCIL_ATTACHMENT) a = &f->depth;
    else if (att == gl::STENCIL_ATTACHMENT) a = &f->stencil;
    else return c->setError(gl::INVALID_ENUM, target);
    if (p == gl::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) {
        *v = a->kind == Attachment::None ? 0 : a->kind == Attachment::Tex ? GLint(gl::TEXTURE) : GLint(gl::RENDERBUFFER);
        return;
    }
    if (a->kind == Attachment::None) {
        if (p == gl::FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) { *v = 0; return; }
        return c->setError(gl::INVALID_OPERATION, target);
    }
    const Texture* tex = a->kind == Attachment::Tex ? c->texture(a->name) : nullptr;
    if (a->kind == Attachment::Rbo &&
        (p == gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL || p == gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE ||
         p == gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER || p == gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR ||
         p == gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR))
        return c->setError(gl::INVALID_ENUM, p);
    const bool layered = tex && (tex->target == gl::TEXTURE_3D || tex->target == gl::TEXTURE_2D_ARRAY);
    Context::FbTarget t;
    uint32_t w, h;
    const FormatInfo* fi = c->targetOf(*a, t, w, h) ? t.image->format : nullptr;
    switch (p) {
        case gl::FRAMEBUFFER_ATTACHMENT_OBJECT_NAME: *v = GLint(a->name); return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL: *v = a->level; return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
            *v = tex && tex->target == gl::TEXTURE_CUBE_MAP ? GLint(gl::TEXTURE_CUBE_MAP_POSITIVE_X) + a->layer : 0;
            return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER: *v = layered ? a->layer : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR: *v = a->numViews; return;
        case 0x8D6C: *v = a->samples; return;  // FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT
        case gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR: *v = a->baseView; return;
        case gl::FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING: *v = fi && fi->srgb ? GLint(gl::SRGB) : GLint(gl::LINEAR_ENC); return;
        case gl::FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE: *v = fi ? GLint(fi->componentType) : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_RED_SIZE: *v = fi ? fi->redBits : 0; return;
        case 0x8213: *v = fi ? fi->greenBits : 0; return;
        case 0x8214: *v = fi ? fi->blueBits : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: *v = fi ? fi->alphaBits : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: *v = fi ? fi->depthBits : 0; return;
        case gl::FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: *v = fi ? fi->stencilBits : 0; return;
        default: c->setError(gl::INVALID_ENUM, target);
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
        if (b != gl::BACK && b != gl::NONE) return c->setError(gl::INVALID_OPERATION, b);
    } else if (b != gl::NONE && (b < gl::COLOR_ATTACHMENT0 || b >= gl::COLOR_ATTACHMENT0 + kMaxDrawBuffers)) {
        return c->setError(gl::INVALID_OPERATION, b);
    }
    f->readBuffer = b;
}
void InvalidateFramebuffer(GLenum target, GLsizei n, const GLenum* atts) {
    CTX()
    c->invalidateFramebuffer(target, n, atts);
}
// A sub-rectangle is invalidated only when it covers the whole framebuffer (then the contents may be dropped).
void InvalidateSubFramebuffer(GLenum target, GLsizei n, const GLenum* atts, GLint x, GLint y, GLsizei w, GLsizei h) {
    CTX()
    if (w < 0 || h < 0) return c->setError(gl::INVALID_VALUE);
    Context::FbInfo fb = c->resolveFramebuffer(target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo, target != gl::READ_FRAMEBUFFER);
    if (x > 0 || y > 0 || int64_t(x) + w < int64_t(fb.width) || int64_t(y) + h < int64_t(fb.height)) return;
    c->invalidateFramebuffer(target, n, atts);
}
void DiscardFramebufferEXT(GLenum target, GLsizei n, const GLenum* atts) {
    CTX()
    c->invalidateFramebuffer(target, n, atts);
}
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
        c->setError(gl::INVALID_ENUM, type);
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
    std::string text;
    for (GLsizei i = 0; i < count; ++i) {
        if (!str[i]) continue;
        if (len && len[i] >= 0) text.append(str[i], size_t(len[i]));
        else text.append(str[i]);
    }
    s->source = std::make_shared<const std::string>(std::move(text));
}
void CompileShader(GLuint n) {
    CTX()
    Shader* s = shaderOrError(c, n);
    if (!s) return;
    vpglsl::Stage st = s->type == gl::VERTEX_SHADER ? vpglsl::Stage::Vertex : s->type == gl::FRAGMENT_SHADER ? vpglsl::Stage::Fragment : vpglsl::Stage::Compute;
    vpglsl::ParseResult r = vpglsl::parse(st, s->sourceText());
    s->parsed = r.shader;
    s->compiled = r.shader != nullptr;
    if (s->compiled) s->compiledSource = s->source;  // the same object: no second copy of the text
    s->log = r.log;
    logLenient(r.lenient, "shader", n);
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
        case gl::SHADER_SOURCE_LENGTH: *v = s->sourceText().empty() ? 0 : GLint(s->sourceText().size() + 1); return;
        default: c->setError(gl::INVALID_ENUM, p);
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
    Shader* s = shaderOrError(c, n);
    if (!s) return;
    if (max < 0) return c->setError(gl::INVALID_VALUE);
    copyString(s->log, max, len, log);
}
void GetShaderSource(GLuint n, GLsizei max, GLsizei* len, GLchar* src) {
    CTX()
    Shader* s = shaderOrError(c, n);
    if (!s) return;
    if (max < 0) return c->setError(gl::INVALID_VALUE);
    copyString(s->sourceText(), max, len, src);
}
void GetShaderPrecisionFormat(GLenum shader, GLenum prec, GLint* range, GLint* precision) {
    CTX()
    if (shader != gl::VERTEX_SHADER && shader != gl::FRAGMENT_SHADER) return c->setError(gl::INVALID_ENUM, shader);
    if (prec < gl::LOW_FLOAT || prec > gl::HIGH_INT) return c->setError(gl::INVALID_ENUM, prec);
    // Adreno: lowp and mediump are 16-bit (fp16: 10 bits of mantissa, range 2^15; int16), highp is 32-bit.
    const bool high = prec == gl::HIGH_FLOAT || prec == gl::HIGH_INT;
    if (prec >= gl::LOW_INT) range[0] = high ? 31 : 15, range[1] = high ? 30 : 14, *precision = 0;
    else if (high) range[0] = range[1] = 127, *precision = 23;
    else range[0] = range[1] = 15, *precision = 10;
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
    if (max < 0) return c->setError(gl::INVALID_VALUE);
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
    if (c->curTf().active && pn == c->program) return c->setError(gl::INVALID_OPERATION);  // the program being captured
    static int unwrittenLines = 0, failedLines = 0;  // varying diagnostics per session: at most 64 lines of each kind
    const bool ok = c->linkProgram(*p);
    if (!ok) logf("gles.program link_failed name=%u log=%s", pn, p->log.c_str());
    if (!p->varyingDiag.empty() && (ok ? unwrittenLines : failedLines) < 64) {
        ++(ok ? unwrittenLines : failedLines);
        logf("gles.program %s name=%u %s", ok ? "varying_unwritten" : "link_failed_varyings", pn, p->varyingDiag.c_str());
    }
}
void UseProgram(GLuint pn) {
    CTX()
    if (c->curTf().active && !c->curTf().paused) return c->setError(gl::INVALID_OPERATION);
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
        case 0x92D9: *v = GLint(p->refl.atomicBuffers.size()); return;  // ACTIVE_ATOMIC_COUNTER_BUFFERS
        case gl::PROGRAM_BINARY_LENGTH: *v = GLint(Context::programBinary(*p).size()); return;
        case gl::PROGRAM_BINARY_RETRIEVABLE_HINT: *v = p->binaryRetrievableHint; return;
        case gl::PROGRAM_SEPARABLE: *v = p->separable; return;
        case gl::TRANSFORM_FEEDBACK_VARYINGS: *v = GLint(p->refl.xfb.size()); return;
        case gl::TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH: {
            size_t m = 0;
            for (const auto& x : p->refl.xfb) m = std::max(m, x.name.size() + 1);
            *v = GLint(m);
            return;
        }
        case gl::TRANSFORM_FEEDBACK_BUFFER_MODE: *v = GLint(p->xfbLinkedMode); return;
        case gl::COMPUTE_WORK_GROUP_SIZE: for (int i = 0; i < 3; ++i) v[i] = GLint(p->refl.localSize[size_t(i)]); return;
        default: c->setError(gl::INVALID_ENUM, q);
    }
}
void GetProgramInfoLog(GLuint pn, GLsizei max, GLsizei* len, GLchar* log) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (max < 0) return c->setError(gl::INVALID_VALUE);
    copyString(p->log, max, len, log);
}
void ProgramParameteri(GLuint pn, GLenum pname, GLint value) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (value != 0 && value != 1) return c->setError(gl::INVALID_VALUE, pname);
    if (pname == 0x8257 /* PROGRAM_BINARY_RETRIEVABLE_HINT */) p->binaryRetrievableHint = value != 0;
    else if (pname == gl::PROGRAM_SEPARABLE) p->separable = value != 0;
    else c->setError(gl::INVALID_ENUM, pname);
}
// One format (kProgramBinaryFormat): the blob holds the shader sources and bindings, so glProgramBinary is the
// same link as glLinkProgram. A blob that is not ours is a failed link (the app recompiles), not a GL error.
void GetProgramBinary(GLuint pn, GLsizei bufSize, GLsizei* len, GLenum* format, void* binary) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (!p->linked) return c->setError(gl::INVALID_OPERATION);  // not successfully linked: no binary, nothing written
    const std::vector<uint8_t> blob = Context::programBinary(*p);  // built now, freed on return (not kept per program)
    if (GLsizei(blob.size()) > bufSize) return c->setError(gl::INVALID_OPERATION);
    if (binary && !blob.empty()) std::memcpy(binary, blob.data(), blob.size());  // NULL buffer: the app's bug, not a crash
    if (len) *len = GLsizei(blob.size());
    if (format) *format = kProgramBinaryFormat;
}
void ProgramBinary(GLuint pn, GLenum fmt, const void* binary, GLsizei length) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (!c->loadProgramBinary(*p, fmt, binary, length > 0 ? size_t(length) : 0))
        logf("gles.program binary_rejected name=%u log=%s", pn, p->log.c_str());
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
    if (!p) return -1;
    if (!p->linked) return c->setError(gl::INVALID_OPERATION), -1;
    for (const auto& o : p->refl.outputs)
        if (o.name == name) return o.location;
    return -1;
}
void GetActiveAttrib(GLuint pn, GLuint i, GLsizei max, GLsizei* len, GLint* size, GLenum* type, GLchar* name) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (max < 0 || i >= p->refl.attributes.size()) return c->setError(gl::INVALID_VALUE);
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
    const vpglsl::Variable* u = max < 0 ? nullptr : activeUniform(*p, i, &block);
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
        if (!u) return c->setError(gl::INVALID_VALUE, q);
        switch (q) {
            case gl::UNIFORM_TYPE: v[k] = GLint(u->glType); break;
            case gl::UNIFORM_SIZE: v[k] = u->arraySize; break;
            case gl::UNIFORM_NAME_LENGTH: v[k] = GLint(u->name.size() + 1); break;
            case gl::UNIFORM_BLOCK_INDEX: v[k] = block; break;
            case gl::UNIFORM_OFFSET: v[k] = block < 0 ? -1 : GLint(u->offset); break;
            case gl::UNIFORM_ARRAY_STRIDE: v[k] = block < 0 ? -1 : GLint(u->arrayStride); break;
            case gl::UNIFORM_MATRIX_STRIDE: v[k] = block < 0 ? -1 : GLint(u->matrixStride); break;
            case gl::UNIFORM_IS_ROW_MAJOR: v[k] = u->rowMajor; break;
            default: return c->setError(gl::INVALID_ENUM, q);
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
    const GLint ret = uniformLocation(*p, name);
    // Diagnostics: the packed-uniform names ([vp]c<N>_[hiu]) the translated shaders use, once per (program, name) (1024 per process).
    if (name && (name[0] == 'v' || name[0] == 'p') && name[1] == 'c' && name[2] >= '0' && name[2] <= '9') {
        const char* q = name + 2;
        while (*q >= '0' && *q <= '9') ++q;
        if (q[0] == '_' && (q[1] == 'h' || q[1] == 'i' || q[1] == 'u')) {
            static std::mutex mu;
            static std::set<std::string> seen;
            std::lock_guard<std::mutex> g(mu);
            if (seen.size() < 1024 && seen.insert(std::to_string(pn) + ":" + name).second)
                logf("gles.uloc prog=%u name=%s ret=%d", unsigned(pn), name, int(ret));
        }
    }
    return ret;
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
    if (i >= p->refl.blocks.size()) return c->setError(gl::INVALID_VALUE, q);
    const auto& b = p->refl.blocks[i];
    switch (q) {
        case gl::UNIFORM_BLOCK_BINDING: *v = GLint(p->blockBindings[i]); return;
        case gl::UNIFORM_BLOCK_DATA_SIZE: *v = GLint(b.size); return;
        case gl::UNIFORM_BLOCK_NAME_LENGTH: *v = GLint(b.name.size() + 1); return;
        case gl::UNIFORM_BLOCK_ACTIVE_UNIFORMS: *v = GLint(b.members.size()); return;
        case gl::UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES: {
            // The members are listed once per declaration: every element of a block array shares the indices of element 0.
            size_t first = i;
            while (p->refl.blocks[first].element != 0 && first > 0) --first;
            GLuint k = GLuint(p->refl.uniforms.size());
            for (size_t x = 0; x < first; ++x)
                if (p->refl.blocks[x].element == 0) k += GLuint(p->refl.blocks[x].members.size());
            for (size_t m = 0; m < b.members.size(); ++m) v[m] = GLint(k + m);
            return;
        }
        case gl::UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER: *v = (b.stages & 1) != 0; return;
        case gl::UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: *v = (b.stages & 2) != 0; return;
        default: c->setError(gl::INVALID_ENUM, q);
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
void GetnUniformfv(GLuint pn, GLint loc, GLsizei, GLfloat* v) { GetUniformfv(pn, loc, v); }
void GetnUniformiv(GLuint pn, GLint loc, GLsizei, GLint* v) { GetUniformiv(pn, loc, v); }
void GetnUniformuiv(GLuint pn, GLint loc, GLsizei, GLuint* v) { GetUniformuiv(pn, loc, v); }

// glProgramUniform*: the glUniform* family addressed to a program instead of the current one.
#define PUNI(name, comps, kind, T)                                                         \
    void ProgramUniform##name(GLuint pn, GLint loc, GLsizei n, const T* v) {               \
        CTX()                                                                              \
        Program* p = programOrError(c, pn);                                                \
        if (p) c->setUniform(loc, comps, kind, n, v, 0, false, p);                         \
    }
PUNI(1fv, 1, K::Float, GLfloat) PUNI(2fv, 2, K::Float, GLfloat) PUNI(3fv, 3, K::Float, GLfloat) PUNI(4fv, 4, K::Float, GLfloat)
PUNI(1iv, 1, K::Int, GLint) PUNI(2iv, 2, K::Int, GLint) PUNI(3iv, 3, K::Int, GLint) PUNI(4iv, 4, K::Int, GLint)
PUNI(1uiv, 1, K::Uint, GLuint) PUNI(2uiv, 2, K::Uint, GLuint) PUNI(3uiv, 3, K::Uint, GLuint) PUNI(4uiv, 4, K::Uint, GLuint)
#undef PUNI
void ProgramUniform1f(GLuint p, GLint l, GLfloat x) { ProgramUniform1fv(p, l, 1, &x); }
void ProgramUniform2f(GLuint p, GLint l, GLfloat x, GLfloat y) { GLfloat v[] = {x, y}; ProgramUniform2fv(p, l, 1, v); }
void ProgramUniform3f(GLuint p, GLint l, GLfloat x, GLfloat y, GLfloat z) { GLfloat v[] = {x, y, z}; ProgramUniform3fv(p, l, 1, v); }
void ProgramUniform4f(GLuint p, GLint l, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { GLfloat v[] = {x, y, z, w}; ProgramUniform4fv(p, l, 1, v); }
void ProgramUniform1i(GLuint p, GLint l, GLint x) { ProgramUniform1iv(p, l, 1, &x); }
void ProgramUniform2i(GLuint p, GLint l, GLint x, GLint y) { GLint v[] = {x, y}; ProgramUniform2iv(p, l, 1, v); }
void ProgramUniform3i(GLuint p, GLint l, GLint x, GLint y, GLint z) { GLint v[] = {x, y, z}; ProgramUniform3iv(p, l, 1, v); }
void ProgramUniform4i(GLuint p, GLint l, GLint x, GLint y, GLint z, GLint w) { GLint v[] = {x, y, z, w}; ProgramUniform4iv(p, l, 1, v); }
void ProgramUniform1ui(GLuint p, GLint l, GLuint x) { ProgramUniform1uiv(p, l, 1, &x); }
void ProgramUniform2ui(GLuint p, GLint l, GLuint x, GLuint y) { GLuint v[] = {x, y}; ProgramUniform2uiv(p, l, 1, v); }
void ProgramUniform3ui(GLuint p, GLint l, GLuint x, GLuint y, GLuint z) { GLuint v[] = {x, y, z}; ProgramUniform3uiv(p, l, 1, v); }
void ProgramUniform4ui(GLuint p, GLint l, GLuint x, GLuint y, GLuint z, GLuint w) { GLuint v[] = {x, y, z, w}; ProgramUniform4uiv(p, l, 1, v); }
#define PUMAT(name, cols, rows)                                                                          \
    void ProgramUniformMatrix##name##fv(GLuint pn, GLint loc, GLsizei n, GLboolean t, const GLfloat* v) { \
        CTX()                                                                                            \
        Program* p = programOrError(c, pn);                                                              \
        if (p) c->setUniform(loc, rows, K::Float, n, v, cols, t, p);                                     \
    }
PUMAT(2, 2, 2) PUMAT(3, 3, 3) PUMAT(4, 4, 4) PUMAT(2x3, 2, 3) PUMAT(3x2, 3, 2) PUMAT(2x4, 2, 4) PUMAT(4x2, 4, 2) PUMAT(3x4, 3, 4)
PUMAT(4x3, 4, 3)
#undef PUMAT

// ---- program interface queries (ES 3.1) ----------------------------------------------------------------
// Backed by the reflection of the linked program. Interfaces with resources in this layer: UNIFORM (the
// default-block uniforms and samplers, then the members of the uniform blocks: the order of
// GL_ACTIVE_UNIFORMS), UNIFORM_BLOCK, PROGRAM_INPUT (vertex attributes), PROGRAM_OUTPUT (fragment outputs).
// SHADER_STORAGE_BLOCK / BUFFER_VARIABLE / ATOMIC_COUNTER_BUFFER / TRANSFORM_FEEDBACK_VARYING have no
// resources until the GLSL compiler reflects them (vpglsl::Reflection has no such lists yet).
bool validInterface(GLenum i) {
    return i == gl::UNIFORM_IF || i == gl::UNIFORM_BLOCK_IF || i == gl::PROGRAM_INPUT || i == gl::PROGRAM_OUTPUT ||
           i == gl::BUFFER_VARIABLE || i == gl::SHADER_STORAGE_BLOCK_IF || i == gl::ATOMIC_COUNTER_BUFFER_IF ||
           i == gl::TRANSFORM_FEEDBACK_VARYING;
}
// Buffer variables: the members of every shader storage block, block after block.
const vpglsl::Variable* bufferVariable(const Program& p, GLuint i, int* block) {
    for (size_t b = 0; b < p.refl.storageBlocks.size(); ++b) {
        if (p.refl.storageBlocks[b].element != 0) continue;  // block arrays list their members once (element 0)
        const auto& m = p.refl.storageBlocks[b].members;
        if (i < m.size()) {
            *block = int(b);
            return &m[i];
        }
        i -= GLuint(m.size());
    }
    return nullptr;
}
GLuint bufferVariableCount(const Program& p) {
    GLuint n = 0;
    for (const auto& b : p.refl.storageBlocks)
        if (b.element == 0) n += GLuint(b.members.size());
    return n;
}
GLuint resourceCount(const Program& p, GLenum iface) {
    if (!p.linked) return 0;
    switch (iface) {
        case gl::UNIFORM_IF: {
            GLuint n = GLuint(p.refl.uniforms.size());
            for (const auto& b : p.refl.blocks)
                if (b.element == 0) n += GLuint(b.members.size());
            return n;
        }
        case gl::UNIFORM_BLOCK_IF: return GLuint(p.refl.blocks.size());
        case gl::SHADER_STORAGE_BLOCK_IF: return GLuint(p.refl.storageBlocks.size());
        case gl::BUFFER_VARIABLE: return bufferVariableCount(p);
        case gl::ATOMIC_COUNTER_BUFFER_IF: return GLuint(p.refl.atomicBuffers.size());
        case gl::PROGRAM_INPUT: return p.csModule ? 0 : GLuint(p.refl.attributes.size());
        case gl::PROGRAM_OUTPUT: return p.csModule ? 0 : GLuint(p.refl.outputs.size());
        default: return 0;
    }
}
// Name of resource `i` of the interface, null when out of range (or the interface has no names).
const std::string* resourceName(const Program& p, GLenum iface, GLuint i) {
    if (i >= resourceCount(p, iface)) return nullptr;
    int block;
    switch (iface) {
        case gl::UNIFORM_IF: return &activeUniform(p, i, &block)->name;
        case gl::UNIFORM_BLOCK_IF: return &p.refl.blocks[i].name;
        case gl::SHADER_STORAGE_BLOCK_IF: return &p.refl.storageBlocks[i].name;
        case gl::BUFFER_VARIABLE: return &bufferVariable(p, i, &block)->name;
        case gl::PROGRAM_INPUT: return &p.refl.attributes[i].name;
        case gl::PROGRAM_OUTPUT: return &p.refl.outputs[i].name;
        default: return nullptr;
    }
}
GLint locationBase(const Program& p, GLuint uniformIndex) {  // first location of default-block uniform `uniformIndex`
    return p.refl.uniforms[uniformIndex].uniformLocation;
}
std::string stripSubscripts(std::string n) {
    while (!n.empty() && n.back() == ']') {
        size_t lb = n.rfind('[');
        if (lb == std::string::npos) break;
        n.resize(lb);
    }
    return n;
}
GLuint resourceIndex(const Program& p, GLenum iface, const std::string& want) {
    const GLuint n = resourceCount(p, iface);
    for (GLuint i = 0; i < n; ++i) {
        const std::string& have = *resourceName(p, iface, i);
        if (have == want) return i;
    }
    // "name" / "name[0]" / "name[2]" all name the array resource "name[0]" (or "name").
    const std::string base = stripSubscripts(want);
    for (GLuint i = 0; i < n; ++i)
        if (stripSubscripts(*resourceName(p, iface, i)) == base) return i;
    return gl::INVALID_INDEX;
}

void GetProgramInterfaceiv(GLuint pn, GLenum iface, GLenum pname, GLint* v) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (!validInterface(iface)) return c->setError(gl::INVALID_ENUM, iface);
    switch (pname) {
        case gl::ACTIVE_RESOURCES: *v = GLint(resourceCount(*p, iface)); return;
        case gl::MAX_NAME_LENGTH: {
            if (iface == gl::ATOMIC_COUNTER_BUFFER_IF) return c->setError(gl::INVALID_OPERATION, pname);
            GLint m = 0;
            for (GLuint i = 0, n = resourceCount(*p, iface); i < n; ++i) m = std::max(m, GLint(resourceName(*p, iface, i)->size() + 1));
            *v = m;
            return;
        }
        case gl::MAX_NUM_ACTIVE_VARIABLES: {
            if (iface != gl::UNIFORM_BLOCK_IF && iface != gl::SHADER_STORAGE_BLOCK_IF && iface != gl::ATOMIC_COUNTER_BUFFER_IF)
                return c->setError(gl::INVALID_OPERATION, pname);
            GLint m = 0;
            if (iface == gl::UNIFORM_BLOCK_IF)
                for (const auto& b : p->refl.blocks) m = std::max(m, GLint(b.members.size()));
            else if (iface == gl::SHADER_STORAGE_BLOCK_IF)
                for (const auto& b : p->refl.storageBlocks) m = std::max(m, GLint(b.members.size()));
            else
                for (const auto& u : p->refl.uniforms)  // atomic counters per buffer
                    if (u.atomicBinding >= 0) {
                        GLint n = 0;
                        for (const auto& v : p->refl.uniforms) n += v.atomicBinding == u.atomicBinding;
                        m = std::max(m, n);
                    }
            *v = m;
            return;
        }
        default: c->setError(gl::INVALID_ENUM, pname);
    }
}
GLuint GetProgramResourceIndex(GLuint pn, GLenum iface, const GLchar* name) {
    CTX(gl::INVALID_INDEX)
    Program* p = programOrError(c, pn);
    if (!p) return gl::INVALID_INDEX;
    if (!validInterface(iface) || iface == gl::ATOMIC_COUNTER_BUFFER_IF) {
        c->setError(gl::INVALID_ENUM, iface);
        return gl::INVALID_INDEX;
    }
    return resourceIndex(*p, iface, name ? name : "");
}
void GetProgramResourceName(GLuint pn, GLenum iface, GLuint index, GLsizei bufSize, GLsizei* len, GLchar* name) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (!validInterface(iface) || iface == gl::ATOMIC_COUNTER_BUFFER_IF) return c->setError(gl::INVALID_ENUM, iface);
    const std::string* n = resourceName(*p, iface, index);
    if (!n) return c->setError(gl::INVALID_VALUE, iface);
    copyString(*n, bufSize, len, name);
}
GLint GetProgramResourceLocation(GLuint pn, GLenum iface, const GLchar* name) {
    CTX(-1)
    Program* p = programOrError(c, pn);
    if (!p) return -1;
    if (iface != gl::UNIFORM_IF && iface != gl::PROGRAM_INPUT && iface != gl::PROGRAM_OUTPUT) {
        c->setError(gl::INVALID_ENUM, iface);
        return -1;
    }
    if (!p->linked) {
        c->setError(gl::INVALID_OPERATION, iface);
        return -1;
    }
    const std::string want = name ? name : "";
    if (iface == gl::UNIFORM_IF) return uniformLocation(*p, want.c_str());
    const auto& list = iface == gl::PROGRAM_INPUT ? p->refl.attributes : p->refl.outputs;
    for (const auto& v : list) {
        std::string vn = v.name;
        if (vn == want) return v.location;
        if (v.isArray && vn.size() > 3 && vn.compare(vn.size() - 3, 3, "[0]") == 0 && vn.substr(0, vn.size() - 3) == want) return v.location;
    }
    return -1;
}
void GetProgramResourceiv(GLuint pn, GLenum iface, GLuint index, GLsizei propCount, const GLenum* props, GLsizei bufSize, GLsizei* len,
                          GLint* params) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (!validInterface(iface)) return c->setError(gl::INVALID_ENUM, iface);
    if (propCount < 1 || bufSize < 0) return c->setError(gl::INVALID_VALUE);
    if (index >= resourceCount(*p, iface)) return c->setError(gl::INVALID_VALUE, iface);
    const bool uniform = iface == gl::UNIFORM_IF, bufvar = iface == gl::BUFFER_VARIABLE, ubo = iface == gl::UNIFORM_BLOCK_IF,
               ssbo = iface == gl::SHADER_STORAGE_BLOCK_IF, atomic = iface == gl::ATOMIC_COUNTER_BUFFER_IF,
               input = iface == gl::PROGRAM_INPUT, block = ubo || ssbo;
    int owner = -1;  // block of a uniform / buffer variable
    const vpglsl::Variable* var = nullptr;
    const vpglsl::UniformBlock* ub = nullptr;
    const vpglsl::Reflection::AtomicBuffer* ab = nullptr;
    if (uniform) var = activeUniform(*p, index, &owner);
    else if (bufvar) var = bufferVariable(*p, index, &owner);
    else if (ubo) ub = &p->refl.blocks[index];
    else if (ssbo) ub = &p->refl.storageBlocks[index];
    else if (atomic) ab = &p->refl.atomicBuffers[index];
    else var = input ? &p->refl.attributes[index] : &p->refl.outputs[index];
    const bool isAtomicVar = var && var->atomicBinding >= 0;
    // Index (in the UNIFORM interface) of the counters at an atomic binding point.
    auto atomicCounters = [&](int binding) {
        std::vector<GLint> r;
        for (size_t u = 0; u < p->refl.uniforms.size(); ++u)
            if (p->refl.uniforms[u].atomicBinding == binding) r.push_back(GLint(u));
        return r;
    };
    GLsizei written = 0;
    for (GLsizei k = 0; k < propCount; ++k) {
        const GLenum pr = props[k];
        std::vector<GLint> vals;  // one or more values for this property
        bool ok = true;
        const bool inBlock = (uniform && owner >= 0) || bufvar;
        switch (pr) {
            case gl::NAME_LENGTH:
                ok = !atomic;
                if (ok) vals = {GLint((var ? var->name : ub->name).size() + 1)};
                break;
            case gl::TYPE: ok = var != nullptr; if (ok) vals = {GLint(var->glType)}; break;
            case gl::ARRAY_SIZE: ok = var != nullptr; if (ok) vals = {var->arraySize}; break;
            case gl::OFFSET: ok = uniform || bufvar; if (ok) vals = {(inBlock || isAtomicVar) ? GLint(var->offset) : -1}; break;
            case gl::BLOCK_INDEX: ok = uniform || bufvar; if (ok) vals = {inBlock ? owner : -1}; break;
            case gl::ARRAY_STRIDE: ok = uniform || bufvar; if (ok) vals = {(inBlock || isAtomicVar) ? GLint(var->arrayStride) : -1}; break;
            case gl::MATRIX_STRIDE: ok = uniform || bufvar; if (ok) vals = {inBlock ? GLint(var->matrixStride) : -1}; break;
            case gl::IS_ROW_MAJOR: ok = uniform || bufvar; if (ok) vals = {var->rowMajor ? 1 : 0}; break;
            case gl::ATOMIC_COUNTER_BUFFER_INDEX:
                ok = uniform;
                if (ok) {
                    GLint idx = -1;
                    if (isAtomicVar)
                        for (size_t a = 0; a < p->refl.atomicBuffers.size(); ++a)
                            if (p->refl.atomicBuffers[a].glBinding == var->atomicBinding) idx = GLint(a);
                    vals = {idx};
                }
                break;
            case gl::BUFFER_BINDING:
                ok = block || atomic;
                if (ok) vals = {atomic ? ab->glBinding : ubo ? GLint(p->blockBindings[index]) : GLint(p->storageBindings[index])};
                break;
            case gl::BUFFER_DATA_SIZE: ok = block || atomic; if (ok) vals = {atomic ? GLint(ab->size) : GLint(ub->size)}; break;
            case gl::NUM_ACTIVE_VARIABLES:
                ok = block || atomic;
                if (ok) vals = {atomic ? GLint(atomicCounters(ab->glBinding).size()) : GLint(ub->members.size())};
                break;
            case gl::ACTIVE_VARIABLES:
                ok = block || atomic;
                if (ok && atomic) vals = atomicCounters(ab->glBinding);
                else if (ok) {
                    GLuint first = 0;
                    if (ubo) {
                        first = GLuint(p->refl.uniforms.size());
                        for (GLuint x = 0; x < index; ++x)
                            if (p->refl.blocks[x].element == 0) first += GLuint(p->refl.blocks[x].members.size());
                        if (ub->element != 0) {  // members of an array element resolve to element 0's uniforms
                            GLuint e0 = index - ub->element;
                            first = GLuint(p->refl.uniforms.size());
                            for (GLuint x = 0; x < e0; ++x)
                                if (p->refl.blocks[x].element == 0) first += GLuint(p->refl.blocks[x].members.size());
                        }
                    } else {
                        const GLuint e0 = index - ub->element;  // elements of a block array share element 0's variables
                        for (GLuint x = 0; x < e0; ++x)
                            if (p->refl.storageBlocks[x].element == 0) first += GLuint(p->refl.storageBlocks[x].members.size());
                    }
                    for (size_t m = 0; m < ub->members.size(); ++m) vals.push_back(GLint(first + GLuint(m)));
                }
                break;
            case gl::REFERENCED_BY_VERTEX_SHADER: case gl::REFERENCED_BY_FRAGMENT_SHADER: case gl::REFERENCED_BY_COMPUTE_SHADER: {
                const uint32_t bit = pr == gl::REFERENCED_BY_VERTEX_SHADER ? 1u : pr == gl::REFERENCED_BY_FRAGMENT_SHADER ? 2u : 4u;
                uint32_t stages;
                if (atomic) stages = ab->stages;
                else if (block) stages = ub->stages;
                else if (bufvar) stages = p->refl.storageBlocks[size_t(owner)].stages;
                else if (uniform) stages = owner >= 0 ? p->refl.blocks[size_t(owner)].stages : var->stages;
                else stages = input ? 1u : 2u;  // inputs: the vertex shader; outputs: the fragment shader
                vals = {(stages & bit) ? 1 : 0};
                break;
            }
            case gl::TOP_LEVEL_ARRAY_SIZE:
                ok = bufvar;
                if (ok) vals = {var->topLevelArraySize};
                break;
            case gl::TOP_LEVEL_ARRAY_STRIDE:
                ok = bufvar;
                if (ok) vals = {GLint(var->topLevelArrayStride)};
                break;
            case gl::LOCATION:
                ok = uniform || input || iface == gl::PROGRAM_OUTPUT;
                if (ok) vals = {uniform ? ((owner >= 0 || isAtomicVar) ? -1 : locationBase(*p, index)) : var->location};
                break;
            default: return c->setError(gl::INVALID_ENUM, pr);
        }
        if (!ok) return c->setError(gl::INVALID_OPERATION, pr);
        for (GLint x : vals) {
            if (written >= bufSize) goto done;
            params[written++] = x;
        }
    }
done:
    if (len) *len = written;
}

// glCreateShaderProgramv: one shader of `type`, compiled and linked into its own (separable) program.
// A vertex / fragment one is a single-stage program, used through glUseProgramStages pipelines.
GLuint CreateShaderProgramv(GLenum type, GLsizei count, const GLchar* const* strings) {
    CTX(0)
    GLuint sh = CreateShader(type);
    if (!sh) return 0;
    ShaderSource(sh, count, strings, nullptr);
    CompileShader(sh);
    GLuint pr = CreateProgram();
    Program* p = c->prog(pr);
    p->separable = true;
    Shader* s = c->shader(sh);
    if (s && s->compiled) {
        AttachShader(pr, sh);
        LinkProgram(pr);
        DetachShader(pr, sh);
    } else if (s) {
        p->log = s->log;
    }
    DeleteShader(sh);
    return pr;
}

// ---- program pipelines, framebuffer parameters, sample positions / mask (ES 3.1) ----------------------------
constexpr GLbitfield kVertexShaderBit = 0x1, kFragmentShaderBit = 0x2, kComputeShaderBit = 0x20, kAllShaderBits = 0xFFFFFFFFu;

ProgramPipeline* pipelineOrError(Context* c, GLuint n) {
    auto it = c->programPipelines.find(n);
    if (n == 0 || it == c->programPipelines.end()) {
        c->setError(gl::INVALID_OPERATION);
        return nullptr;
    }
    return it->second.get();
}
void GenProgramPipelines(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        auto pl = std::make_unique<ProgramPipeline>();
        pl->name = out[i] = c->nextProgramPipeline++;
        c->programPipelines[pl->name] = std::move(pl);
    }
}
void DeleteProgramPipelines(GLsizei n, const GLuint* names) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        if (!names[i] || !c->programPipelines.count(names[i])) continue;
        if (c->programPipeline == names[i]) c->programPipeline = 0;
        c->programPipelines.erase(names[i]);
    }
}
GLboolean IsProgramPipeline(GLuint n) {
    CTX(0)
    auto it = c->programPipelines.find(n);
    return n && it != c->programPipelines.end() && it->second->bound;
}
void BindProgramPipeline(GLuint n) {
    CTX()
    if (n) {
        ProgramPipeline* pl = pipelineOrError(c, n);
        if (!pl) return;
        pl->bound = true;
    }
    c->programPipeline = n;
}
void UseProgramStages(GLuint pipeline, GLbitfield stages, GLuint program) {
    CTX()
    ProgramPipeline* pl = pipelineOrError(c, pipeline);
    if (!pl) return;
    if (stages != kAllShaderBits && (stages & ~(kVertexShaderBit | kFragmentShaderBit | kComputeShaderBit)))
        return c->setError(gl::INVALID_VALUE);
    Program* p = nullptr;
    if (program) {
        p = c->prog(program);
        if (!p) return c->setError(c->shader(program) ? gl::INVALID_OPERATION : gl::INVALID_VALUE);
        if (!p->linked || !p->separable) return c->setError(gl::INVALID_OPERATION);
    }
    // A stage the program does not hold is cleared, as the spec says for a program without that executable.
    if (stages & kVertexShaderBit) pl->vs = p && p->stageVs ? program : 0;
    if (stages & kFragmentShaderBit) pl->fs = p && p->stageFs ? program : 0;
    if (stages & kComputeShaderBit) pl->cs = p && p->csModule ? program : 0;
}
void ActiveShaderProgram(GLuint pipeline, GLuint program) {
    CTX()
    ProgramPipeline* pl = pipelineOrError(c, pipeline);
    if (!pl) return;
    if (program) {
        Program* p = programOrError(c, program);
        if (!p) return;
        if (!p->linked) return c->setError(gl::INVALID_OPERATION);
    }
    pl->active = program;
}
// Valid when something can run: a compute stage alone, or a vertex and a fragment stage that link together.
void ValidateProgramPipeline(GLuint pipeline) {
    CTX()
    ProgramPipeline* pl = pipelineOrError(c, pipeline);
    if (!pl) return;
    const bool compute = pl->cs && !pl->vs && !pl->fs;
    const bool graphics = pl->vs && pl->fs && !pl->cs && (pl->vs == pl->fs || c->pipelineProgram(*pl));
    pl->validated = compute || graphics;
    pl->log = pl->validated ? "" : "program pipeline: the vertex and fragment stages are missing or do not link together\n";
}
void GetProgramPipelineiv(GLuint pipeline, GLenum pname, GLint* v) {
    CTX()
    ProgramPipeline* pl = pipelineOrError(c, pipeline);
    if (!pl) return;
    switch (pname) {
        case 0x8259: *v = GLint(pl->active); return;                  // ACTIVE_PROGRAM
        case gl::VERTEX_SHADER: *v = GLint(pl->vs); return;
        case gl::FRAGMENT_SHADER: *v = GLint(pl->fs); return;
        case gl::COMPUTE_SHADER: *v = GLint(pl->cs); return;
        case gl::VALIDATE_STATUS: *v = pl->validated; return;
        case 0x8B84: *v = pl->log.empty() ? 0 : GLint(pl->log.size() + 1); return;  // INFO_LOG_LENGTH
        default: c->setError(gl::INVALID_ENUM, pname);
    }
}
void GetProgramPipelineInfoLog(GLuint pipeline, GLsizei max, GLsizei* len, GLchar* log) {
    CTX()
    if (max < 0) return c->setError(gl::INVALID_VALUE);
    if (ProgramPipeline* pl = pipelineOrError(c, pipeline)) copyString(pl->log, max, len, log);
}

Framebuffer* framebufferForParameters(Context* c, GLenum target) {
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER) {
        c->setError(gl::INVALID_ENUM, target);
        return nullptr;
    }
    return c->fbo(target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo);
}
void FramebufferParameteri(GLenum target, GLenum pname, GLint param) {
    CTX()
    Framebuffer* f = framebufferForParameters(c, target);
    if (!f) return;
    if ((target == gl::READ_FRAMEBUFFER ? c->readFbo : c->drawFbo) == 0) return c->setError(gl::INVALID_OPERATION, target);
    switch (pname) {
        case 0x9310: case 0x9311:  // FRAMEBUFFER_DEFAULT_WIDTH / HEIGHT
            if (param < 0 || param > 16384) return c->setError(gl::INVALID_VALUE, pname);
            (pname == 0x9310 ? f->defaultWidth : f->defaultHeight) = param;
            return;
        case 0x9312:  // FRAMEBUFFER_DEFAULT_LAYERS
            if (param < 0 || param > 2048) return c->setError(gl::INVALID_VALUE, pname);
            f->defaultLayers = param;
            return;
        case 0x9313:  // FRAMEBUFFER_DEFAULT_SAMPLES
            if (param < 0 || param > 4) return c->setError(gl::INVALID_VALUE, pname);
            f->defaultSamples = param;
            return;
        case 0x9314: f->defaultFixedSampleLocations = param != 0; return;  // FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS
        default: c->setError(gl::INVALID_ENUM, pname);
    }
}
void GetFramebufferParameteriv(GLenum target, GLenum pname, GLint* v) {
    CTX()
    Framebuffer* f = framebufferForParameters(c, target);
    if (!f) return;
    switch (pname) {
        case 0x9310: *v = f->defaultWidth; return;
        case 0x9311: *v = f->defaultHeight; return;
        case 0x9312: *v = f->defaultLayers; return;
        case 0x9313: *v = f->defaultSamples; return;
        case 0x9314: *v = f->defaultFixedSampleLocations; return;
        default: c->setError(gl::INVALID_ENUM, pname);
    }
}

// Standard sample positions (Vulkan / D3D) for 1, 2, 4 and 8 samples, in [0,1] from the pixel's corner.
void GetMultisamplefv(GLenum pname, GLuint index, GLfloat* val) {
    CTX()
    if (pname != 0x8E50) return c->setError(gl::INVALID_ENUM, pname);  // SAMPLE_POSITION
    // Samples of the draw framebuffer: what its attachments were created with (the default one has none).
    GLint samples = 0;
    if (c->drawFbo) {
        const Framebuffer* f = c->fbo(c->drawFbo);
        for (const Attachment& a : f->color) {
            if (a.kind == Attachment::Rbo) { if (Renderbuffer* r = c->rbo(a.name)) samples = std::max(samples, r->samples); }
            else if (a.kind == Attachment::Tex) { if (Texture* t = c->texture(a.name)) samples = std::max(samples, t->samples); }
        }
        samples = std::max(samples, f->defaultSamples);
    }
    const GLint n = samples <= 1 ? 1 : samples <= 2 ? 2 : samples <= 4 ? 4 : 8;
    if (index >= GLuint(n)) return c->setError(gl::INVALID_VALUE);
    static const float kPos[][2] = {
        {0.5f, 0.5f},
        {0.75f, 0.75f}, {0.25f, 0.25f},
        {0.375f, 0.125f}, {0.875f, 0.375f}, {0.125f, 0.625f}, {0.625f, 0.875f},
        {0.5625f, 0.3125f}, {0.4375f, 0.6875f}, {0.8125f, 0.5625f}, {0.3125f, 0.1875f},
        {0.1875f, 0.8125f}, {0.0625f, 0.4375f}, {0.6875f, 0.9375f}, {0.9375f, 0.0625f},
    };
    const int base = n == 1 ? 0 : n == 2 ? 1 : n == 4 ? 3 : 7;
    val[0] = kPos[base + int(index)][0], val[1] = kPos[base + int(index)][1];
}
void SampleMaski(GLuint maskNumber, GLbitfield mask) {
    CTX()
    if (maskNumber >= 1) return c->setError(gl::INVALID_VALUE);  // MAX_SAMPLE_MASK_WORDS = 1
    c->sampleMask = mask;
}
void GetTransformFeedbackVarying(GLuint pn, GLuint index, GLsizei max, GLsizei* len, GLsizei* size, GLenum* type, GLchar* name) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (max < 0 || index >= p->refl.xfb.size()) return c->setError(gl::INVALID_VALUE);
    const auto& x = p->refl.xfb[index];
    if (size) *size = x.arraySize;
    if (type) *type = x.glType;
    const GLsizei n = max > 0 ? std::min<GLsizei>(max - 1, GLsizei(x.name.size())) : 0;
    if (max > 0 && name) std::memcpy(name, x.name.data(), size_t(n)), name[n] = 0;
    if (len) *len = n;
}
GLboolean IsTransformFeedback(GLuint n) {
    CTX(0)
    auto it = c->transformFeedbacks.find(n);
    return n && it != c->transformFeedbacks.end() && it->second->bound ? 1 : 0;
}

// ---- queries / sync ---------------------------------------------------------------------------------------
// Queries are "coherent", not GPU-measured: the occlusion result is 1 when a draw was issued between Begin
// and End (0 when none was), TIME_ELAPSED is the host time between Begin and End, TIMESTAMP the host clock.
// Occlusion queries with a GPU counter (see queryValue) are available when the GPU has finished them.
uint64_t nowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
bool isOcclusion(GLenum t) { return t == gl::ANY_SAMPLES_PASSED || t == gl::ANY_SAMPLES_PASSED_CONSERVATIVE; }
bool validQueryTarget(GLenum t) {
    return t == gl::ANY_SAMPLES_PASSED || t == gl::ANY_SAMPLES_PASSED_CONSERVATIVE || t == gl::TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN ||
           t == gl::TIME_ELAPSED_EXT;
}
void GenQueries(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = c->nextQuery++;
        c->queries[name] = std::make_unique<Query>();
        c->queries[name]->name = name;
        out[i] = name;
    }
}
void DeleteQueries(GLsizei n, const GLuint* names) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        auto it = c->queries.find(names[i]);
        if (it == c->queries.end()) continue;
        if (it->second->active) c->occEnd(*it->second), c->activeQuery.erase(it->second->target);
        if (it->second->slot != UINT32_MAX) c->occBusy_[it->second->slot] = false;
        c->queries.erase(it);
    }
}
GLboolean IsQuery(GLuint n) {
    CTX(0)
    auto it = c->queries.find(n);
    return it != c->queries.end() && it->second->target != 0 ? 1 : 0;
}
void BeginQuery(GLenum target, GLuint n) {
    CTX()
    if (!validQueryTarget(target)) return c->setError(gl::INVALID_ENUM, target);
    if (n == 0) return c->setError(gl::INVALID_OPERATION, target);
    auto it = c->queries.find(n);
    if (it == c->queries.end()) return c->setError(gl::INVALID_OPERATION, target);
    Query& q = *it->second;
    if (c->activeQuery.count(target) || q.active || (q.target && q.target != target)) return c->setError(gl::INVALID_OPERATION, target);
    q.target = target, q.active = true, q.result = 0, q.beginNs = nowNs(), q.beginDraws = c->drawCount, q.beginXfb = c->xfbPrimitivesWritten;
    c->activeQuery[target] = n;
    if (isOcclusion(target)) c->occBegin(q);
}
void EndQuery(GLenum target) {
    CTX()
    if (!validQueryTarget(target)) return c->setError(gl::INVALID_ENUM, target);
    auto at = c->activeQuery.find(target);
    if (at == c->activeQuery.end()) return c->setError(gl::INVALID_OPERATION, target);
    Query& q = *c->queries[at->second];
    q.active = false;
    if (target == gl::TIME_ELAPSED_EXT) q.result = nowNs() - q.beginNs;
    else if (target == gl::TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN) q.result = c->xfbPrimitivesWritten - q.beginXfb;
    else q.result = c->drawCount > q.beginDraws ? 1 : 0;
    if (isOcclusion(target)) c->occEnd(q);
    c->activeQuery.erase(at);
}
void QueryCounter(GLuint n, GLenum target) {
    CTX()
    if (target != gl::TIMESTAMP_EXT) return c->setError(gl::INVALID_ENUM, target);
    auto it = c->queries.find(n);
    if (it == c->queries.end() || it->second->active || (it->second->target && it->second->target != target))
        return c->setError(gl::INVALID_OPERATION, target);
    it->second->target = target, it->second->result = nowNs();
}
void GetQueryiv(GLenum target, GLenum pname, GLint* v) {
    CTX()
    if (!validQueryTarget(target) && target != gl::TIMESTAMP_EXT) return c->setError(gl::INVALID_ENUM, target);
    if (pname == gl::CURRENT_QUERY) {
        auto at = c->activeQuery.find(target);
        *v = at == c->activeQuery.end() ? 0 : GLint(at->second);
    } else if (pname == gl::QUERY_COUNTER_BITS_EXT) {
        *v = (target == gl::TIME_ELAPSED_EXT || target == gl::TIMESTAMP_EXT) ? 64 : 0;
    } else {
        c->setError(gl::INVALID_ENUM, pname);
    }
}
Query* queryObject(Context* c, GLuint n, GLenum pname) {
    auto it = c->queries.find(n);
    if (it == c->queries.end() || it->second->active || it->second->target == 0) {
        c->setError(gl::INVALID_OPERATION, pname);
        return nullptr;
    }
    if (pname != gl::QUERY_RESULT && pname != gl::QUERY_RESULT_AVAILABLE) {
        c->setError(gl::INVALID_ENUM, pname);
        return nullptr;
    }
    return it->second.get();
}
// vk.query_stats: what the game reads from the occlusion queries, every ~5 s while it reads them (720 lines at most).
struct OccStats {
    std::atomic<uint64_t> reads{0}, notReady{0}, zero{0}, positive{0}, expired{0}, skipped{0};
    std::atomic<uint64_t> lastNs{0};
    std::atomic<uint32_t> lines{0};
    void tick() {
        constexpr uint64_t kEveryNs = 5'000'000'000ull;
        constexpr uint32_t kMaxLines = 720;
        const uint64_t now = nowNs();
        uint64_t last = lastNs.load(std::memory_order_relaxed);
        if (!last) {
            lastNs.compare_exchange_strong(last, now);
            return;
        }
        if (now - last < kEveryNs || !lastNs.compare_exchange_strong(last, now)) return;
        if (lines.fetch_add(1) >= kMaxLines) return;
        logf("vk.query_stats api=gles reads=%llu not_ready=%llu zero=%llu positive=%llu wait_expired=%llu wait_skipped=%llu",
             static_cast<unsigned long long>(reads.load()), static_cast<unsigned long long>(notReady.load()),
             static_cast<unsigned long long>(zero.load()), static_cast<unsigned long long>(positive.load()),
             static_cast<unsigned long long>(expired.load()), static_cast<unsigned long long>(skipped.load()));
    }
} occStats;
// Occlusion: the draws were counted by the GPU (VPVulkan's visibility results) and the layer saw at least one draw. A query
// that got no counter (none free, or another one counting) keeps the coherent answer. QUERY_RESULT waits for the count
// (within Context::kOccWaitNs per serial, then it reads visible without waiting while the GPU is behind): a not-ready count must never read 0, Unreal culls what reads 0.
uint64_t queryValue(Context* c, Query& q, GLenum pname) {
    if (isOcclusion(q.target) && q.slot != UINT32_MAX && q.endSerial && c->occPool_) {
        uint64_t v = 0;
        Context::OccWait how = Context::OccWait::Done;
        const bool ready = c->occResult(q, pname == gl::QUERY_RESULT, v, &how);
        occStats.reads.fetch_add(1, std::memory_order_relaxed);
        if (how == Context::OccWait::Expired) occStats.expired.fetch_add(1, std::memory_order_relaxed);
        else if (how == Context::OccWait::Skipped) occStats.skipped.fetch_add(1, std::memory_order_relaxed);
        else if (!ready) occStats.notReady.fetch_add(1, std::memory_order_relaxed);
        if (pname == gl::QUERY_RESULT) {
            const uint64_t visible = how != Context::OccWait::Done || (q.result && v) ? 1 : 0;
            (visible ? occStats.positive : occStats.zero).fetch_add(1, std::memory_order_relaxed);
            occStats.tick();
            return visible;
        }
        occStats.tick();
        return ready ? 1 : 0;
    }
    return pname == gl::QUERY_RESULT_AVAILABLE ? 1 : q.result;
}
void GetQueryObjectuiv(GLuint n, GLenum p, GLuint* v) {
    CTX()
    if (Query* q = queryObject(c, n, p)) *v = GLuint(std::min<uint64_t>(queryValue(c, *q, p), 0xFFFFFFFFu));
}
void GetQueryObjectiv(GLuint n, GLenum p, GLint* v) {
    CTX()
    if (Query* q = queryObject(c, n, p)) *v = GLint(std::min<uint64_t>(queryValue(c, *q, p), 0x7FFFFFFFu));
}
void GetQueryObjectui64v(GLuint n, GLenum p, GLuint64* v) {
    CTX()
    if (Query* q = queryObject(c, n, p)) *v = queryValue(c, *q, p);
}
void GetQueryObjecti64v(GLuint n, GLenum p, GLint64* v) {
    CTX()
    if (Query* q = queryObject(c, n, p)) *v = GLint64(std::min<uint64_t>(queryValue(c, *q, p), uint64_t(INT64_MAX)));
}
GLsync FenceSync(GLenum cond, GLbitfield) {
    CTX(nullptr)
    if (cond != gl::SYNC_GPU_COMMANDS_COMPLETE) {
        c->setError(gl::INVALID_ENUM, cond);
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
void WaitSync(GLsync s, GLbitfield flags, GLuint64 timeout) {  // one queue: already ordered
    CTX()
    if (!c->share->syncs.count(s) || flags != 0 || timeout != ~0ull) c->setError(gl::INVALID_VALUE);
}
void GetSynciv(GLsync s, GLenum p, GLsizei, GLsizei* len, GLint* v) {
    CTX()
    auto it = c->share->syncs.find(s);
    if (it == c->share->syncs.end()) return c->setError(gl::INVALID_VALUE, p);
    c->dev->retire(false);
    switch (p) {
        case gl::OBJECT_TYPE: *v = GLint(gl::SYNC_FENCE); break;
        case gl::SYNC_STATUS: *v = GLint(c->dev->completed >= it->second->serial ? gl::SIGNALED : gl::UNSIGNALED); break;
        case gl::SYNC_CONDITION: *v = GLint(gl::SYNC_GPU_COMMANDS_COMPLETE); break;
        case gl::SYNC_FLAGS: *v = 0; break;
        default: return c->setError(gl::INVALID_ENUM, p);
    }
    if (len) *len = 1;
}

// ---- debug / no-ops -----------------------------------------------------------------------------------------
void DebugMessageCallback(void* cb, const void* user) {
    CTX()
    c->debugCb = reinterpret_cast<Context::DebugProc>(cb), c->debugUser = user;
}
bool validSource(GLenum e) { return e == gl::DONT_CARE_ENUM || (e >= gl::DEBUG_SOURCE_API && e <= gl::DEBUG_SOURCE_OTHER); }
bool validType(GLenum e) {
    return e == gl::DONT_CARE_ENUM || (e >= gl::DEBUG_TYPE_ERROR && e <= gl::DEBUG_TYPE_ERROR + 5) || e == gl::DEBUG_TYPE_MARKER ||
           e == gl::DEBUG_TYPE_PUSH_GROUP || e == gl::DEBUG_TYPE_POP_GROUP || e == 0x8251 /* OTHER */;
}
bool validSeverity(GLenum e) { return e == gl::DONT_CARE_ENUM || (e >= gl::DEBUG_SEVERITY_HIGH && e <= gl::DEBUG_SEVERITY_HIGH + 2) || e == gl::DEBUG_SEVERITY_NOTIFICATION; }
void DebugMessageControl(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint* ids, GLboolean enabled) {
    CTX()
    if (count < 0) return c->setError(gl::INVALID_VALUE);
    if (!validSource(source) || !validType(type) || !validSeverity(severity)) return c->setError(gl::INVALID_ENUM);
    if (count > 0 && (source == gl::DONT_CARE_ENUM || type == gl::DONT_CARE_ENUM || severity != gl::DONT_CARE_ENUM))
        return c->setError(gl::INVALID_OPERATION);
    c->debugRules.push_back({source, type, severity, std::vector<GLuint>(ids, ids + (count > 0 ? count : 0)), enabled != 0});
}
void DebugMessageInsert(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* buf) {
    CTX()
    if (source != gl::DEBUG_SOURCE_APPLICATION && source != gl::DEBUG_SOURCE_THIRD_PARTY) return c->setError(gl::INVALID_ENUM, source);
    if (!validType(type) || type == gl::DONT_CARE_ENUM || !validSeverity(severity) || severity == gl::DONT_CARE_ENUM)
        return c->setError(gl::INVALID_ENUM);
    if (length >= 256) return c->setError(gl::INVALID_VALUE);
    if (!c->debugOutput) return;
    c->debugMessage(source, type, id, severity, length < 0 ? std::string(buf ? buf : "") : std::string(buf ? buf : "", size_t(length)));
}
GLuint GetDebugMessageLog(GLuint count, GLsizei bufSize, GLenum* sources, GLenum* types, GLuint* ids, GLenum* severities, GLsizei* lengths,
                          GLchar* log) {
    CTX(0)
    GLuint got = 0;
    GLsizei used = 0;
    while (got < count && !c->debugLog.empty()) {
        const auto& m = c->debugLog.front();
        const GLsizei need = GLsizei(m.text.size() + 1);
        if (log && used + need > bufSize) break;
        if (sources) sources[got] = m.source;
        if (types) types[got] = m.type;
        if (ids) ids[got] = m.id;
        if (severities) severities[got] = m.severity;
        if (lengths) lengths[got] = need;
        if (log) std::memcpy(log + used, m.text.c_str(), size_t(need)), used += need;
        c->debugLog.erase(c->debugLog.begin());
        ++got;
    }
    return got;
}
void PushDebugGroup(GLenum source, GLuint id, GLsizei length, const GLchar* msg) {
    CTX()
    if (source != gl::DEBUG_SOURCE_APPLICATION && source != gl::DEBUG_SOURCE_THIRD_PARTY) return c->setError(gl::INVALID_ENUM, source);
    if (length >= 256) return c->setError(gl::INVALID_VALUE);
    if (int(c->debugGroups.size()) + 1 >= gl::kMaxDebugGroupDepth) return c->setError(gl::STACK_OVERFLOW);
    std::string text = length < 0 ? std::string(msg ? msg : "") : std::string(msg ? msg : "", size_t(length));
    if (c->debugOutput) c->debugMessage(source, gl::DEBUG_TYPE_PUSH_GROUP, id, gl::DEBUG_SEVERITY_NOTIFICATION, text);
    c->debugGroups.push_back(text);
}
void PopDebugGroup() {
    CTX()
    if (c->debugGroups.empty()) return c->setError(gl::STACK_UNDERFLOW);
    std::string text = c->debugGroups.back();
    c->debugGroups.pop_back();
    if (c->debugOutput) c->debugMessage(gl::DEBUG_SOURCE_APPLICATION, gl::DEBUG_TYPE_POP_GROUP, 0, gl::DEBUG_SEVERITY_NOTIFICATION, text);
}
// glObjectLabel identifiers -> the namespace of the object (checked for existence).
bool labelObjectExists(Context* c, GLenum id, GLuint name) {
    switch (id) {
        case gl::BUFFER_ID: return c->share->buffers.count(name) != 0;
        case gl::SHADER_ID: { auto it = c->share->shaders.find(name); return it != c->share->shaders.end(); }
        case gl::PROGRAM_ID: return c->share->programs.count(name) != 0;
        case gl::VERTEX_ARRAY_ID: return c->vaos.count(name) != 0;
        case gl::QUERY_ID: return c->queries.count(name) != 0;
        case gl::SAMPLER_ID: return c->share->samplers.count(name) != 0;
        case gl::TEXTURE_ID: return c->share->textures.count(name) != 0;
        case gl::RENDERBUFFER_ID: return c->share->renderbuffers.count(name) != 0;
        case gl::FRAMEBUFFER_ID: return c->fbos.count(name) != 0 || name == 0;
        case gl::PROGRAM_PIPELINE_ID: case gl::TRANSFORM_FEEDBACK_ID: return false;
        default: return false;
    }
}
GLenum extLabelId(GLenum t) {  // EXT_debug_label type -> KHR_debug identifier
    switch (t) {
        case gl::BUFFER_OBJECT_EXT: return gl::BUFFER_ID;
        case gl::SHADER_OBJECT_EXT: return gl::SHADER_ID;
        case gl::PROGRAM_OBJECT_EXT: return gl::PROGRAM_ID;
        case gl::VERTEX_ARRAY_OBJECT_EXT: return gl::VERTEX_ARRAY_ID;
        case gl::QUERY_OBJECT_EXT: return gl::QUERY_ID;
        case gl::TEXTURE_ID: case gl::RENDERBUFFER_ID: case gl::FRAMEBUFFER_ID: return t;
        default: return 0;
    }
}
void ObjectLabel(GLenum id, GLuint name, GLsizei length, const GLchar* label) {
    CTX()
    if (!labelObjectExists(c, id, name)) return c->setError(id >= gl::BUFFER_ID || id == gl::TEXTURE_ID ? gl::INVALID_VALUE : gl::INVALID_ENUM, id);
    if (length >= gl::kMaxLabelLength) return c->setError(gl::INVALID_VALUE);
    if (!label) return void(c->share->labels.erase({id, name}));
    c->share->labels[{id, name}] = length < 0 ? std::string(label) : std::string(label, size_t(length));
}
void GetObjectLabel(GLenum id, GLuint name, GLsizei bufSize, GLsizei* length, GLchar* label) {
    CTX()
    if (bufSize < 0) return c->setError(gl::INVALID_VALUE);
    if (!labelObjectExists(c, id, name)) return c->setError(gl::INVALID_VALUE, id);
    auto it = c->share->labels.find({id, name});
    copyString(it == c->share->labels.end() ? std::string() : it->second, bufSize, length, label);
}
void ObjectPtrLabel(const void* ptr, GLsizei length, const GLchar* label) {
    CTX()
    auto it = c->share->syncs.find(const_cast<GLsync>(ptr));
    if (it == c->share->syncs.end()) return c->setError(gl::INVALID_VALUE);
    if (length >= gl::kMaxLabelLength) return c->setError(gl::INVALID_VALUE);
    if (!label) return void(c->share->syncLabels.erase(const_cast<GLsync>(ptr)));
    c->share->syncLabels[const_cast<GLsync>(ptr)] = length < 0 ? std::string(label) : std::string(label, size_t(length));
}
void GetObjectPtrLabel(const void* ptr, GLsizei bufSize, GLsizei* length, GLchar* label) {
    CTX()
    if (bufSize < 0) return c->setError(gl::INVALID_VALUE);
    if (!c->share->syncs.count(const_cast<GLsync>(ptr))) return c->setError(gl::INVALID_VALUE);
    auto it = c->share->syncLabels.find(const_cast<GLsync>(ptr));
    copyString(it == c->share->syncLabels.end() ? std::string() : it->second, bufSize, length, label);
}
void LabelObjectEXT(GLenum type, GLuint name, GLsizei length, const GLchar* label) {
    const GLenum id = extLabelId(type);
    CTX()
    if (!id) return c->setError(gl::INVALID_ENUM, type);
    ObjectLabel(id, name, length, label);
}
void GetObjectLabelEXT(GLenum type, GLuint name, GLsizei bufSize, GLsizei* length, GLchar* label) {
    const GLenum id = extLabelId(type);
    CTX()
    if (!id) return c->setError(gl::INVALID_ENUM, type);
    GetObjectLabel(id, name, bufSize, length, label);
}
void GetPointerv(GLenum pname, void** params) {
    CTX()
    if (pname == gl::DEBUG_CALLBACK_FUNCTION) *params = reinterpret_cast<void*>(c->debugCb);
    else if (pname == gl::DEBUG_CALLBACK_USER_PARAM) *params = const_cast<void*>(c->debugUser);
    else c->setError(gl::INVALID_ENUM, pname);
}
void InsertEventMarkerEXT(GLsizei, const GLchar*) {}
void PushGroupMarkerEXT(GLsizei, const GLchar*) {}
void PopGroupMarkerEXT() {}
GLenum GetGraphicsResetStatus() { return gl::NO_ERROR; }
void PauseTransformFeedback() {
    CTX()
    Context::TransformFeedback& tf = c->curTf();
    if (!tf.active || tf.paused) return c->setError(gl::INVALID_OPERATION);
    tf.paused = true;
}
void ResumeTransformFeedback() {
    CTX()
    Context::TransformFeedback& tf = c->curTf();
    if (!tf.active || !tf.paused) return c->setError(gl::INVALID_OPERATION);
    tf.paused = false;
}
void BeginTransformFeedback(GLenum mode) {
    CTX()
    if (mode != gl::POINTS && mode != gl::LINES && mode != gl::TRIANGLES) return c->setError(gl::INVALID_ENUM, mode);
    Context::TransformFeedback& tf = c->curTf();
    Program* p = c->prog(c->program);
    if (tf.active || !p || !p->linked || p->refl.xfb.empty()) return c->setError(gl::INVALID_OPERATION);
    const size_t bindings = p->xfbLinkedMode == 0x8C8D ? p->refl.xfb.size() : 1;
    for (size_t i = 0; i < bindings; ++i) {
        Buffer* b = tf.buffers[i].buffer ? c->buffer(tf.buffers[i].buffer) : nullptr;
        if (!b || !b->storage) return c->setError(gl::INVALID_OPERATION);  // every binding the varyings use needs a buffer
    }
    tf.active = true, tf.paused = false, tf.mode = mode, tf.program = p->serial;
    tf.used = {};
}
void EndTransformFeedback() {
    CTX()
    Context::TransformFeedback& tf = c->curTf();
    if (!tf.active) return c->setError(gl::INVALID_OPERATION);
    tf.active = tf.paused = false;
}
void GenTransformFeedbacks(GLsizei n, GLuint* out) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        const GLuint name = c->nextTransformFeedback++;
        auto o = std::make_unique<Context::TransformFeedback>();
        o->name = name;
        c->transformFeedbacks[name] = std::move(o);
        out[i] = name;
    }
}
void DeleteTransformFeedbacks(GLsizei n, const GLuint* names) {
    CTX()
    if (n < 0) return c->setError(gl::INVALID_VALUE);
    for (GLsizei i = 0; i < n; ++i) {
        auto it = c->transformFeedbacks.find(names[i]);
        if (!names[i] || it == c->transformFeedbacks.end()) continue;
        if (it->second->active) return c->setError(gl::INVALID_OPERATION);
        if (c->transformFeedback == names[i]) c->transformFeedback = 0;
        c->transformFeedbacks.erase(it);
    }
}
void BindTransformFeedback(GLenum target, GLuint name) {
    CTX()
    if (target != gl::TRANSFORM_FEEDBACK) return c->setError(gl::INVALID_ENUM, target);
    if (c->curTf().active && !c->curTf().paused) return c->setError(gl::INVALID_OPERATION);
    if (name && !c->transformFeedbacks.count(name)) return c->setError(gl::INVALID_OPERATION);
    c->transformFeedback = name;
    c->curTf().bound = true;
}
void TransformFeedbackVaryings(GLuint pn, GLsizei count, const GLchar* const* varyings, GLenum mode) {
    CTX()
    Program* p = programOrError(c, pn);
    if (!p) return;
    if (mode != 0x8C8C && mode != 0x8C8D) return c->setError(gl::INVALID_ENUM, mode);
    if (count < 0 || (mode == 0x8C8D && count > 4)) return c->setError(gl::INVALID_VALUE);
    p->xfbVaryings.clear();
    for (GLsizei i = 0; i < count; ++i) p->xfbVaryings.push_back(varyings[i]);
    p->xfbMode = mode;
}

struct Entry {
    const char* name;
    void* fn;
};
#define E(n) {"gl" #n, reinterpret_cast<void*>(&n)}
const Entry kEntries[] = {
    E(GetError), E(GetString), E(GetStringi), E(GetIntegerv), E(GetInteger64v), E(GetFloatv), E(GetBooleanv),
    E(GetIntegeri_v), E(GetInteger64i_v), E(GetBooleani_v), E(Enable), E(Disable), E(IsEnabled), E(Enablei),
    E(Disablei), E(IsEnabledi), E(Viewport), E(Scissor), E(DepthRangef), E(ClearColor), E(ClearDepthf), E(ClearStencil), E(Clear),
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
    E(DrawRangeElementsBaseVertex), E(DispatchCompute), E(MemoryBarrier), E(MemoryBarrierByRegion), E(GenTextures), E(DeleteTextures),
    E(IsTexture), E(ActiveTexture), E(BindTexture), E(TexImage2D), E(TexSubImage2D), E(TexImage3D),
    E(TexSubImage3D), E(CompressedTexImage2D), E(CompressedTexSubImage2D), E(CompressedTexImage3D),
    E(CompressedTexSubImage3D), E(TexStorage2D), E(TexStorage3D), E(TexStorage2DMultisample),
    E(TexStorage3DMultisample), E(CopyTexImage2D), E(CopyTexSubImage2D), E(CopyTexSubImage3D), E(GenerateMipmap),
    E(TexParameteri), E(TexParameterf), E(TexParameteriv), E(TexParameterfv), E(GetTexParameteriv),
    E(GetTexParameterfv), E(EGLImageTargetTexture2DOES), E(GenSamplers), E(DeleteSamplers), E(IsSampler),
    E(BindSampler), E(SamplerParameteri), E(SamplerParameterf), E(SamplerParameteriv), E(SamplerParameterfv),
    E(GetSamplerParameteriv), E(GetSamplerParameterfv), E(TexParameterIiv), E(TexParameterIuiv), E(GetTexParameterIiv),
    E(GetTexParameterIuiv), E(SamplerParameterIiv), E(SamplerParameterIuiv), E(GetSamplerParameterIiv), E(GetSamplerParameterIuiv), E(GenRenderbuffers), E(DeleteRenderbuffers),
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
    E(QueryCounter), E(GetQueryObjectiv), E(GetQueryObjectui64v), E(GetQueryObjecti64v), E(FenceSync), E(IsSync), E(DeleteSync), E(ClientWaitSync), E(WaitSync), E(GetSynciv),
    E(DebugMessageCallback), E(DebugMessageControl), E(DebugMessageInsert), E(GetDebugMessageLog), E(PushDebugGroup),
    E(PopDebugGroup), E(ObjectLabel), E(GetObjectLabel), E(ObjectPtrLabel), E(GetObjectPtrLabel), E(LabelObjectEXT),
    E(GetObjectLabelEXT), E(GetPointerv), E(InsertEventMarkerEXT), E(PushGroupMarkerEXT), E(PopGroupMarkerEXT), E(GetGraphicsResetStatus),
    E(PauseTransformFeedback), E(ResumeTransformFeedback), E(BeginTransformFeedback), E(EndTransformFeedback),
    E(GenTransformFeedbacks), E(DeleteTransformFeedbacks), E(BindTransformFeedback), E(TransformFeedbackVaryings),
    // ES 3.1 / 3.2 additions.
    E(VertexAttribFormat), E(VertexAttribIFormat), E(VertexAttribBinding), E(BindVertexBuffer), E(VertexBindingDivisor),
    E(DrawArraysIndirect), E(DrawElementsIndirect), E(DispatchComputeIndirect), E(BlendBarrier), E(PatchParameteri),
    E(TexBuffer), E(TexBufferRange), E(TexImage2DMultisample), E(GetTexLevelParameteriv), E(GetTexLevelParameterfv),
    E(CopyImageSubData), E(BindImageTexture), E(FramebufferTexture3D), E(GetVertexAttribIiv), E(GetVertexAttribIuiv),
    E(GetnUniformfv), E(GetnUniformiv), E(GetnUniformuiv), E(GetProgramInterfaceiv), E(GetProgramResourceIndex),
    E(GetProgramResourceName), E(GetProgramResourceiv), E(GetProgramResourceLocation), E(CreateShaderProgramv),
    E(ProgramUniform1f), E(ProgramUniform2f), E(ProgramUniform3f), E(ProgramUniform4f), E(ProgramUniform1i),
    E(ProgramUniform2i), E(ProgramUniform3i), E(ProgramUniform4i), E(ProgramUniform1ui), E(ProgramUniform2ui),
    E(ProgramUniform3ui), E(ProgramUniform4ui), E(ProgramUniform1fv), E(ProgramUniform2fv), E(ProgramUniform3fv),
    E(ProgramUniform4fv), E(ProgramUniform1iv), E(ProgramUniform2iv), E(ProgramUniform3iv), E(ProgramUniform4iv),
    E(ProgramUniform1uiv), E(ProgramUniform2uiv), E(ProgramUniform3uiv), E(ProgramUniform4uiv),
    E(ProgramUniformMatrix2fv), E(ProgramUniformMatrix3fv), E(ProgramUniformMatrix4fv), E(ProgramUniformMatrix2x3fv),
    E(ProgramUniformMatrix3x2fv), E(ProgramUniformMatrix2x4fv), E(ProgramUniformMatrix4x2fv), E(ProgramUniformMatrix3x4fv),
    E(ProgramUniformMatrix4x3fv), E(GenProgramPipelines), E(DeleteProgramPipelines), E(IsProgramPipeline), E(BindProgramPipeline),
    E(UseProgramStages), E(ActiveShaderProgram), E(ValidateProgramPipeline), E(GetProgramPipelineiv),
    E(GetProgramPipelineInfoLog), E(FramebufferParameteri), E(GetFramebufferParameteriv), E(GetMultisamplefv), E(SampleMaski),
    E(GetTransformFeedbackVarying), E(IsTransformFeedback),
};
#undef E

}  // namespace

const char* Context::errorName(GLenum e) {
    switch (e) {
        case gl::INVALID_ENUM: return "GL_INVALID_ENUM";
        case gl::INVALID_VALUE: return "GL_INVALID_VALUE";
        case gl::INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case gl::INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case gl::OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        case gl::STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
        case gl::STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
        default: return "GL error";
    }
}

// KHR_debug: filter by the glDebugMessageControl rules (the last matching rule wins; by default HIGH and
// MEDIUM messages and the non-severity types are enabled, LOW and NOTIFICATION are off), then deliver to the
// callback or queue in the message log.
void Context::debugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, const std::string& text) {
    if (inDebugCb) return;
    bool enabled = severity != gl::DEBUG_SEVERITY_NOTIFICATION && severity != gl::DEBUG_SEVERITY_HIGH + 2;
    for (const DebugRule& r : debugRules) {
        if (r.source != gl::DONT_CARE_ENUM && r.source != source) continue;
        if (r.type != gl::DONT_CARE_ENUM && r.type != type) continue;
        if (r.severity != gl::DONT_CARE_ENUM && r.severity != severity) continue;
        if (!r.ids.empty() && std::find(r.ids.begin(), r.ids.end(), id) == r.ids.end()) continue;
        enabled = r.enabled;
    }
    if (!enabled) return;
    if (debugCb) {
        inDebugCb = true;
        debugCb(source, type, id, severity, GLsizei(text.size()), text.c_str(), debugUser);
        inDebugCb = false;
    } else if (int(debugLog.size()) < gl::kMaxDebugLoggedMessages) {
        debugLog.push_back({source, type, id, severity, text});
    }
}

// Extension-suffixed names (OES/EXT/KHR/ANGLE/NV) resolve to the core function of the base name, except
// those whose suffix is part of their only name (in the table above).
void* glProc(const char* name) {
    for (const Entry& e : kEntries)
        if (!std::strcmp(e.name, name)) return e.fn;
    static const char* const kSuffixes[] = {"OES", "EXT", "KHR", "ANGLE", "NV", "ARB", "IMG", "APPLE", "QCOM"};
    const size_t len = std::strlen(name);
    for (const char* s : kSuffixes) {
        const size_t sl = std::strlen(s);
        if (len > sl && !std::strcmp(name + len - sl, s)) {
            std::string base(name, len - sl);
            for (const Entry& e : kEntries)
                if (base == e.name) return e.fn;
        }
    }
    if (!std::strcmp(name, "glFramebufferTexture2DMultisampleIMG")) return reinterpret_cast<void*>(&FramebufferTexture2DMultisampleEXT);
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
    texStatsTick(false);
    recipeStatsTick(false);
    Context* c = currentOrLast();
    if (!c) return;
    c->flush(wait != 0);
}
