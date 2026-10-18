// GLES layer tests against the real VPVulkan driver on its null platform (dlopened, the way Klepton
// loads a Vulkan driver). The null backend executes nothing, so these check what can be checked
// without a GPU: GL state and errors, the Vulkan calls the layer makes (intercepted), and that the
// driver accepted everything (stderr scanned for vk.err / vk.unimpl / gles.err lines: every
// pipeline really went through spirv::lower and the MSL emitter inside VPVulkan).
// Usage: vpgles_tests <group> <path to libVPVulkan>
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "formats.h"
#include "gl_enums.h"
#include "vk.h"
using namespace vpgles;

// vpgles from the driver dylib (dlsym), as Klepton reaches it.
int (*vpgles_init)(void*);
void* (*vpgles_proc)(const char*);
void (*vpgles_device)(void**, void**, void**, void**);
uint32_t (*vpgles_wrap_image)(void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void (*vpgles_flush)(int);

namespace {

int failures = 0;
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);          \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

// ---- Vulkan interception ----------------------------------------------------------------------------
PFN_vkGetInstanceProcAddr realGipa = nullptr;
PFN_vkGetDeviceProcAddr realGdpa = nullptr;
struct Counts {
    int draws = 0, drawsIndexed = 0, passes = 0, pipelines = 0, copiesToImage = 0, clears = 0, dispatches = 0,
        blits = 0, copiesToBuffer = 0;
    uint32_t lastViewMask = 0, lastColorCount = 0, lastInstances = 0;
    std::vector<VkFormat> lastAttrFormats;
    VkFrontFace lastFrontFace = VK_FRONT_FACE_MAX_ENUM;
} counts;

PFN_vkCmdDraw realDraw;
PFN_vkCmdDrawIndexed realDrawIndexed;
PFN_vkCmdBeginRendering realBeginRendering;
PFN_vkCreateGraphicsPipelines realCreatePipelines;
PFN_vkCmdCopyBufferToImage realCopyToImage;
PFN_vkCmdClearAttachments realClear;
PFN_vkCmdDispatch realDispatch;
PFN_vkCmdBlitImage realBlit;
PFN_vkCmdCopyImageToBuffer realCopyToBuffer;

VKAPI_ATTR void VKAPI_CALL myDraw(VkCommandBuffer c, uint32_t v, uint32_t i, uint32_t fv, uint32_t fi) {
    ++counts.draws;
    counts.lastInstances = i;
    realDraw(c, v, i, fv, fi);
}
VKAPI_ATTR void VKAPI_CALL myDrawIndexed(VkCommandBuffer c, uint32_t n, uint32_t i, uint32_t f, int32_t b, uint32_t fi) {
    ++counts.drawsIndexed;
    counts.lastInstances = i;
    realDrawIndexed(c, n, i, f, b, fi);
}
VKAPI_ATTR void VKAPI_CALL myBeginRendering(VkCommandBuffer c, const VkRenderingInfo* ri) {
    ++counts.passes;
    counts.lastViewMask = ri->viewMask;
    counts.lastColorCount = ri->colorAttachmentCount;
    realBeginRendering(c, ri);
}
VKAPI_ATTR VkResult VKAPI_CALL myCreatePipelines(VkDevice d, VkPipelineCache pc, uint32_t n, const VkGraphicsPipelineCreateInfo* ci,
                                                 const VkAllocationCallbacks* a, VkPipeline* out) {
    ++counts.pipelines;
    counts.lastAttrFormats.clear();
    for (uint32_t i = 0; i < ci->pVertexInputState->vertexAttributeDescriptionCount; ++i)
        counts.lastAttrFormats.push_back(ci->pVertexInputState->pVertexAttributeDescriptions[i].format);
    counts.lastFrontFace = ci->pRasterizationState->frontFace;
    return realCreatePipelines(d, pc, n, ci, a, out);
}
VKAPI_ATTR void VKAPI_CALL myCopyToImage(VkCommandBuffer c, VkBuffer b, VkImage i, VkImageLayout l, uint32_t n, const VkBufferImageCopy* r) {
    ++counts.copiesToImage;
    realCopyToImage(c, b, i, l, n, r);
}
VKAPI_ATTR void VKAPI_CALL myClear(VkCommandBuffer c, uint32_t n, const VkClearAttachment* a, uint32_t rn, const VkClearRect* r) {
    ++counts.clears;
    realClear(c, n, a, rn, r);
}
VKAPI_ATTR void VKAPI_CALL myDispatch(VkCommandBuffer c, uint32_t x, uint32_t y, uint32_t z) {
    ++counts.dispatches;
    realDispatch(c, x, y, z);
}
VKAPI_ATTR void VKAPI_CALL myBlit(VkCommandBuffer c, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                  const VkImageBlit* r, VkFilter f) {
    ++counts.blits;
    realBlit(c, s, sl, d, dl, n, r, f);
}
VKAPI_ATTR void VKAPI_CALL myCopyToBuffer(VkCommandBuffer c, VkImage i, VkImageLayout l, VkBuffer b, uint32_t n, const VkBufferImageCopy* r) {
    ++counts.copiesToBuffer;
    realCopyToBuffer(c, i, l, b, n, r);
}

PFN_vkVoidFunction intercept(const char* name, PFN_vkVoidFunction real) {
#define I(n, mine, store)                                    \
    if (!std::strcmp(name, n)) {                              \
        store = reinterpret_cast<decltype(store)>(real);      \
        return real ? reinterpret_cast<PFN_vkVoidFunction>(&mine) : nullptr; \
    }
    I("vkCmdDraw", myDraw, realDraw)
    I("vkCmdDrawIndexed", myDrawIndexed, realDrawIndexed)
    I("vkCmdBeginRendering", myBeginRendering, realBeginRendering)
    I("vkCreateGraphicsPipelines", myCreatePipelines, realCreatePipelines)
    I("vkCmdCopyBufferToImage", myCopyToImage, realCopyToImage)
    I("vkCmdClearAttachments", myClear, realClear)
    I("vkCmdDispatch", myDispatch, realDispatch)
    I("vkCmdBlitImage", myBlit, realBlit)
    I("vkCmdCopyImageToBuffer", myCopyToBuffer, realCopyToBuffer)
#undef I
    return real;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL myGdpa(VkDevice d, const char* name) { return intercept(name, realGdpa(d, name)); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL myGipa(VkInstance i, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) {
        realGdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(realGipa(i, name));
        return reinterpret_cast<PFN_vkVoidFunction>(&myGdpa);
    }
    return intercept(name, realGipa(i, name));
}

// ---- stderr capture -------------------------------------------------------------------------------------
std::string stderrPath;
int savedStderr = -1;
void captureStderr() {
    char tmpl[] = "/tmp/vpgles_stderr_XXXXXX";
    int fd = mkstemp(tmpl);
    stderrPath = tmpl;
    std::fflush(stderr);
    savedStderr = dup(2);
    dup2(fd, 2);
    close(fd);
}
std::string releaseStderr() {
    std::fflush(stderr);
    dup2(savedStderr, 2);
    close(savedStderr);
    std::ifstream f(stderrPath);
    std::stringstream ss;
    ss << f.rdbuf();
    std::remove(stderrPath.c_str());
    return ss.str();
}

// ---- GL access ------------------------------------------------------------------------------------------
template <class F>
F gl(const char* name) {
    void* p = vpgles_proc(name);
    if (!p) {
        std::printf("  missing entry point %s\n", name);
        ++failures;
        std::exit(1);
    }
    return reinterpret_cast<F>(p);
}
#define G(name, sig) auto name = gl<sig>(#name)

using EGLint = int32_t;
struct Egl {
    void* dpy = nullptr;
    void* surface = nullptr;
    void* ctx = nullptr;
    void* config = nullptr;
    bool up(int w = 64, int h = 64, int depth = 24) {
        G(eglGetDisplay, void* (*)(void*));
        G(eglInitialize, uint32_t (*)(void*, EGLint*, EGLint*));
        G(eglChooseConfig, uint32_t (*)(void*, const EGLint*, void**, EGLint, EGLint*));
        G(eglCreatePbufferSurface, void* (*)(void*, void*, const EGLint*));
        G(eglCreateContext, void* (*)(void*, void*, void*, const EGLint*));
        G(eglMakeCurrent, uint32_t (*)(void*, void*, void*, void*));
        dpy = eglGetDisplay(nullptr);
        EGLint maj = 0, min = 0;
        if (!eglInitialize(dpy, &maj, &min)) return false;
        const EGLint attrs[] = {egl::RED_SIZE, 8, egl::GREEN_SIZE, 8, egl::BLUE_SIZE, 8, egl::ALPHA_SIZE, 8,
                                egl::DEPTH_SIZE, depth, egl::RENDERABLE_TYPE, egl::OPENGL_ES3_BIT, egl::NONE};
        EGLint n = 0;
        if (!eglChooseConfig(dpy, attrs, &config, 1, &n) || n != 1) return false;
        const EGLint sattrs[] = {egl::WIDTH, w, egl::HEIGHT, h, egl::NONE};
        surface = eglCreatePbufferSurface(dpy, config, sattrs);
        const EGLint cattrs[] = {egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE};
        ctx = eglCreateContext(dpy, config, nullptr, cattrs);
        return surface && ctx && eglMakeCurrent(dpy, surface, surface, ctx);
    }
    void down() {
        G(eglMakeCurrent, uint32_t (*)(void*, void*, void*, void*));
        G(eglDestroyContext, uint32_t (*)(void*, void*));
        G(eglDestroySurface, uint32_t (*)(void*, void*));
        eglMakeCurrent(dpy, nullptr, nullptr, nullptr);
        eglDestroyContext(dpy, ctx);
        eglDestroySurface(dpy, surface);
    }
};

GLuint program(const char* vs, const char* fs) {
    G(glCreateShader, GLuint (*)(GLenum));
    G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
    G(glCompileShader, void (*)(GLuint));
    G(glGetShaderiv, void (*)(GLuint, GLenum, GLint*));
    G(glCreateProgram, GLuint (*)());
    G(glAttachShader, void (*)(GLuint, GLuint));
    G(glLinkProgram, void (*)(GLuint));
    G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
    G(glGetProgramInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
    GLuint s[2];
    const char* src[2] = {vs, fs};
    for (int i = 0; i < 2; ++i) {
        s[i] = glCreateShader(i ? gl::FRAGMENT_SHADER : gl::VERTEX_SHADER);
        glShaderSource(s[i], 1, &src[i], nullptr);
        glCompileShader(s[i]);
        GLint ok = 0;
        glGetShaderiv(s[i], gl::COMPILE_STATUS, &ok);
        CHECK(ok);
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, s[0]);
    glAttachShader(p, s[1]);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, gl::LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        std::printf("  link log: %s\n", log);
    }
    CHECK(ok);
    return p;
}

GLenum err() {
    G(glGetError, GLenum (*)());
    return glGetError();
}

// ---- tests ---------------------------------------------------------------------------------------------
void testEgl() {
    G(eglGetDisplay, void* (*)(void*));
    G(eglInitialize, uint32_t (*)(void*, EGLint*, EGLint*));
    G(eglGetConfigs, uint32_t (*)(void*, void**, EGLint, EGLint*));
    G(eglGetConfigAttrib, uint32_t (*)(void*, void*, EGLint, EGLint*));
    G(eglQueryString, const char* (*)(void*, EGLint));
    G(eglGetError, EGLint (*)());
    G(eglSwapBuffers, uint32_t (*)(void*, void*));
    G(eglQuerySurface, uint32_t (*)(void*, void*, EGLint, EGLint*));
    G(eglGetCurrentContext, void* (*)());
    G(eglCreateSyncKHR, void* (*)(void*, uint32_t, const EGLint*));
    G(eglClientWaitSyncKHR, EGLint (*)(void*, void*, EGLint, uint64_t));
    void* d = eglGetDisplay(nullptr);
    EGLint maj = 0, min = 0;
    CHECK(eglInitialize(d, &maj, &min) && maj == 1 && min == 5);
    EGLint n = 0;
    CHECK(eglGetConfigs(d, nullptr, 0, &n) && n >= 4);
    std::vector<void*> cfgs(static_cast<size_t>(n), nullptr);
    eglGetConfigs(d, cfgs.data(), n, &n);
    EGLint r = 0, depth = 0;
    CHECK(eglGetConfigAttrib(d, cfgs[0], egl::RED_SIZE, &r) && r == 8);
    CHECK(eglGetConfigAttrib(d, cfgs[0], egl::DEPTH_SIZE, &depth) && depth == 24);
    CHECK(!eglGetConfigAttrib(d, cfgs[0], 0x7777, &r) && eglGetError() == egl::BAD_ATTRIBUTE);
    CHECK(std::strstr(eglQueryString(d, egl::EXTENSIONS), "EGL_KHR_fence_sync"));
    Egl e;
    CHECK(e.up(128, 96));
    CHECK(eglGetCurrentContext() == e.ctx);
    EGLint w = 0, h = 0;
    eglQuerySurface(d, e.surface, egl::WIDTH, &w);
    eglQuerySurface(d, e.surface, egl::HEIGHT, &h);
    CHECK(w == 128 && h == 96);
    G(glGetIntegerv, void (*)(GLenum, GLint*));
    GLint vp[4] = {};
    glGetIntegerv(gl::VIEWPORT, vp);
    CHECK(vp[2] == 128 && vp[3] == 96);
    G(glGetString, const GLubyte* (*)(GLenum));
    CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::VERSION)), "OpenGL ES 3.2"));
    CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::EXTENSIONS)), "GL_OVR_multiview2"));
    CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::RENDERER)), "Adreno"));
    void* sync = eglCreateSyncKHR(d, egl::SYNC_FENCE_KHR, nullptr);
    CHECK(sync && eglClientWaitSyncKHR(d, sync, 0, ~0ull) == egl::CONDITION_SATISFIED_KHR);
    CHECK(eglSwapBuffers(d, e.surface));
    // eglGetProcAddress-style lookups: core and suffixed names.
    CHECK(vpgles_proc("glGenVertexArraysOES") == vpgles_proc("glGenVertexArrays"));
    CHECK(vpgles_proc("glDrawElementsInstancedEXT") == vpgles_proc("glDrawElementsInstanced"));
    CHECK(vpgles_proc("glFramebufferTextureMultiviewOVR") != nullptr);
    CHECK(vpgles_proc("glNotARealFunction") == nullptr);
    e.down();
}

// Contexto y superficie actuales en otro hilo: BAD_ACCESS, destrucción diferida (sin UAF).
void testEglThreads() {
    G(eglMakeCurrent, uint32_t (*)(void*, void*, void*, void*));
    G(eglDestroyContext, uint32_t (*)(void*, void*));
    G(eglDestroySurface, uint32_t (*)(void*, void*));
    G(eglGetCurrentContext, void* (*)());
    G(eglGetCurrentSurface, void* (*)(EGLint));
    G(eglQuerySurface, uint32_t (*)(void*, void*, EGLint, EGLint*));
    G(eglGetError, EGLint (*)());
    G(glGetError, GLenum (*)());
    Egl e;
    CHECK(e.up(32, 32));
    void* const d = e.dpy;
    // 1. Un contexto actual en este hilo no se puede hacer actual en otro: BAD_ACCESS.
    std::atomic<int> step{0};
    std::thread other([&] {
        CHECK(!eglMakeCurrent(d, nullptr, nullptr, e.ctx) && eglGetError() == egl::BAD_ACCESS);
        step = 1;
    });
    other.join();
    CHECK(step == 1 && eglGetCurrentContext() == e.ctx);
    // 2. Destruir lo actual en este hilo difiere el borrado: sigue actual y usable; ya no se puede reusar.
    CHECK(eglDestroyContext(d, e.ctx));
    CHECK(eglGetCurrentContext() == e.ctx && eglGetCurrentSurface(egl::DRAW) == e.surface);
    CHECK(glGetError() == gl::NO_ERROR);
    EGLint w = 0;
    CHECK(eglDestroySurface(d, e.surface));
    CHECK(eglQuerySurface(d, e.surface, egl::WIDTH, &w) && w == 32);  // viva hasta soltarla
    CHECK(!eglMakeCurrent(d, e.surface, e.surface, e.ctx) && eglGetError() == egl::BAD_CONTEXT);
    CHECK(eglMakeCurrent(d, nullptr, nullptr, nullptr));  // borra contexto y superficie
    CHECK(eglGetCurrentContext() == nullptr);
    // 3. Contexto y superficie actuales en OTRO hilo: destruirlos desde este no los borra bajo sus pies.
    Egl f;
    std::atomic<bool> ready{false}, destroyed{false};
    std::atomic<int> result{0};
    std::thread render([&] {
        if (!f.up(16, 16)) return void(result = -1);
        ready = true;
        while (!destroyed) std::this_thread::yield();
        EGLint fw = 0;
        const bool alive = glGetError() == gl::NO_ERROR && eglGetCurrentContext() == f.ctx &&
                           eglQuerySurface(f.dpy, f.surface, egl::WIDTH, &fw) && fw == 16;
        eglMakeCurrent(f.dpy, nullptr, nullptr, nullptr);  // aquí se borran
        result = alive ? 1 : -2;
    });
    while (!ready) std::this_thread::yield();
    CHECK(eglDestroyContext(f.dpy, f.ctx) == 1 && eglDestroySurface(f.dpy, f.surface) == 1);
    destroyed = true;
    render.join();
    CHECK(result == 1);
}

void testFormats() {
    const FormatInfo* rgba = formatFor(gl::RGBA, gl::RGBA, gl::UNSIGNED_BYTE);
    CHECK(rgba && rgba->vk == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(formatFor(gl::RGB, gl::RGB, gl::UNSIGNED_SHORT_5_6_5)->vk == VK_FORMAT_R5G6B5_UNORM_PACK16);
    CHECK(formatFor(gl::RGBA, gl::RGBA, gl::HALF_FLOAT_OES)->vk == VK_FORMAT_R16G16B16A16_SFLOAT);
    CHECK(formatFor(gl::DEPTH_COMPONENT, gl::DEPTH_COMPONENT, gl::UNSIGNED_SHORT)->vk == VK_FORMAT_D16_UNORM);
    CHECK(formatFor(gl::COMPRESSED_RGBA_ASTC_4x4)->vk == VK_FORMAT_ASTC_4x4_UNORM_BLOCK);
    CHECK(formatFor(gl::COMPRESSED_SRGB8_ALPHA8_ASTC_4x4 + 13)->vk == VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
    CHECK(formatFor(gl::ETC1_RGB8_OES)->vk == VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
    const FormatInfo* lum = formatFor(gl::LUMINANCE_ALPHA, gl::LUMINANCE_ALPHA, gl::UNSIGNED_BYTE);
    CHECK(lum && lum->vk == VK_FORMAT_R8G8_UNORM && lum->swz[3] == VK_COMPONENT_SWIZZLE_G);
    CHECK(compressedSize(*formatFor(gl::COMPRESSED_RGBA_ASTC_4x4 + 4 /* 6x6 */), 64, 64) == 11 * 11 * 16);
    // RGB8 -> RGBA8 with alpha 255.
    std::vector<uint8_t> out;
    const uint8_t rgb[6] = {1, 2, 3, 4, 5, 6};
    CHECK(convertUpload(*formatFor(gl::RGB8), gl::RGB, gl::UNSIGNED_BYTE, rgb, 2, out));
    CHECK(out.size() == 8 && out[3] == 255 && out[4] == 4 && out[7] == 255);
    // 5551: GL R in the top bits, Vulkan A1R5G5B5 A on top.
    const uint16_t p5551 = uint16_t(31 << 11 | 1);  // red + alpha
    CHECK(convertUpload(*formatFor(gl::RGB5_A1), gl::RGBA, gl::UNSIGNED_SHORT_5_5_5_1,
                        reinterpret_cast<const uint8_t*>(&p5551), 1, out));
    uint16_t q;
    std::memcpy(&q, out.data(), 2);
    CHECK(q == uint16_t(1 << 15 | 31 << 10));
    // Readback: BGRA -> RGBA, 565 -> RGBA8.
    uint8_t bgra[4] = {10, 20, 30, 40}, rgbaOut[4];
    CHECK(convertReadback(*formatFor(gl::BGRA8_EXT), gl::RGBA, gl::UNSIGNED_BYTE, bgra, 1, rgbaOut));
    CHECK(rgbaOut[0] == 30 && rgbaOut[2] == 10 && rgbaOut[3] == 40);
    uint16_t p565 = uint16_t(31 << 11 | 63 << 5);
    CHECK(convertReadback(*formatFor(gl::RGB565), gl::RGBA, gl::UNSIGNED_BYTE, reinterpret_cast<uint8_t*>(&p565), 1, rgbaOut));
    CHECK(rgbaOut[0] == 255 && rgbaOut[1] == 255 && rgbaOut[2] == 0 && rgbaOut[3] == 255);
    CHECK(clientPixelBytes(gl::RGBA, gl::FLOAT) == 16 && clientPixelBytes(gl::RGB, gl::UNSIGNED_SHORT_5_6_5) == 2);
}

void testBuffers() {
    Egl e;
    CHECK(e.up());
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    G(glBufferSubData, void (*)(GLenum, GLintptr, GLsizeiptr, const void*));
    G(glMapBufferRange, void* (*)(GLenum, GLintptr, GLsizeiptr, GLbitfield));
    G(glUnmapBuffer, GLboolean (*)(GLenum));
    G(glGetBufferParameteriv, void (*)(GLenum, GLenum, GLint*));
    G(glCopyBufferSubData, void (*)(GLenum, GLenum, GLintptr, GLintptr, GLsizeiptr));
    G(glDeleteBuffers, void (*)(GLsizei, const GLuint*));
    G(glIsBuffer, GLboolean (*)(GLuint));
    GLuint b[2];
    glGenBuffers(2, b);
    glBindBuffer(gl::ARRAY_BUFFER, b[0]);
    float data[16];
    for (int i = 0; i < 16; ++i) data[i] = float(i);
    glBufferData(gl::ARRAY_BUFFER, sizeof data, data, gl::STATIC_DRAW);
    GLint size = 0;
    glGetBufferParameteriv(gl::ARRAY_BUFFER, gl::BUFFER_SIZE, &size);
    CHECK(size == 64 && glIsBuffer(b[0]));
    float x = 42;
    glBufferSubData(gl::ARRAY_BUFFER, 4, 4, &x);
    auto* m = static_cast<float*>(glMapBufferRange(gl::ARRAY_BUFFER, 0, 64, gl::MAP_READ_BIT));
    CHECK(m && m[1] == 42.0f && m[2] == 2.0f);
    CHECK(glUnmapBuffer(gl::ARRAY_BUFFER));
    CHECK(!glUnmapBuffer(gl::ARRAY_BUFFER) && err() == gl::INVALID_OPERATION);
    glBufferSubData(gl::ARRAY_BUFFER, 60, 8, &x);
    CHECK(err() == gl::INVALID_VALUE);
    glBindBuffer(gl::COPY_WRITE_BUFFER, b[1]);
    glBufferData(gl::COPY_WRITE_BUFFER, 64, nullptr, gl::DYNAMIC_DRAW);
    glCopyBufferSubData(gl::ARRAY_BUFFER, gl::COPY_WRITE_BUFFER, 0, 0, 64);
    CHECK(err() == gl::NO_ERROR);
    glBindBuffer(0x1234, b[0]);
    CHECK(err() == gl::INVALID_ENUM);
    glDeleteBuffers(2, b);
    CHECK(!glIsBuffer(b[0]));
    e.down();
}

void testTextures() {
    Egl e;
    CHECK(e.up());
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexImage2D, void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));
    G(glTexSubImage2D, void (*)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*));
    G(glTexStorage2D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
    G(glTexStorage3D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLsizei));
    G(glTexSubImage3D, void (*)(GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei, int64_t, int64_t, const void*));
    G(glCompressedTexImage2D, void (*)(GLenum, GLint, GLenum, GLsizei, GLsizei, GLint, GLsizei, const void*));
    G(glGenerateMipmap, void (*)(GLenum));
    G(glTexParameteri, void (*)(GLenum, GLenum, GLint));
    G(glGetTexParameteriv, void (*)(GLenum, GLenum, GLint*));
    G(glPixelStorei, void (*)(GLenum, GLint));
    GLuint t[4];
    glGenTextures(4, t);
    glBindTexture(gl::TEXTURE_2D, t[0]);
    std::vector<uint8_t> px(64 * 64 * 4, 200);
    counts = {};
    glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 64, 64, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
    CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 1);
    glGenerateMipmap(gl::TEXTURE_2D);
    CHECK(err() == gl::NO_ERROR && counts.blits == 6);
    glPixelStorei(gl::UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(gl::TEXTURE_2D, 0, 3, 5, 7, 9, gl::RGB, gl::UNSIGNED_BYTE, px.data());
    CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 2);
    glTexSubImage2D(gl::TEXTURE_2D, 0, 60, 60, 8, 8, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
    CHECK(err() == gl::INVALID_VALUE);
    glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR);
    GLint v = 0;
    glGetTexParameteriv(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, &v);
    CHECK(v == GLint(gl::LINEAR_MIPMAP_LINEAR));
    // Immutable storage, 2D array, compressed.
    glBindTexture(gl::TEXTURE_2D, t[1]);
    glTexStorage2D(gl::TEXTURE_2D, 3, gl::SRGB8_ALPHA8, 32, 32);
    glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 32, 32, 0, gl::RGBA, gl::UNSIGNED_BYTE, nullptr);
    CHECK(err() == gl::INVALID_OPERATION);
    glBindTexture(gl::TEXTURE_2D_ARRAY, t[2]);
    glTexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::RGBA8, 16, 16, 2);
    glTexSubImage3D(gl::TEXTURE_2D_ARRAY, 0, 0, 0, 1, 16, 16, 1, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
    CHECK(err() == gl::NO_ERROR);
    glBindTexture(gl::TEXTURE_2D, t[3]);
    std::vector<uint8_t> astc(4 * 4 * 16, 0);
    glCompressedTexImage2D(gl::TEXTURE_2D, 0, gl::COMPRESSED_RGBA_ASTC_4x4, 16, 16, 0, GLsizei(astc.size()), astc.data());
    CHECK(err() == gl::NO_ERROR);
    glCompressedTexImage2D(gl::TEXTURE_2D, 0, gl::COMPRESSED_RGBA_ASTC_4x4, 16, 16, 0, 10, astc.data());
    CHECK(err() == gl::INVALID_VALUE);
    glBindTexture(gl::TEXTURE_CUBE_MAP, t[0]);  // t[0] is a 2D texture
    CHECK(err() == gl::INVALID_OPERATION);
    e.down();
}

void testFbo() {
    Egl e;
    CHECK(e.up());
    G(glGenFramebuffers, void (*)(GLsizei, GLuint*));
    G(glBindFramebuffer, void (*)(GLenum, GLuint));
    G(glFramebufferTexture2D, void (*)(GLenum, GLenum, GLenum, GLuint, GLint));
    G(glFramebufferRenderbuffer, void (*)(GLenum, GLenum, GLenum, GLuint));
    G(glCheckFramebufferStatus, GLenum (*)(GLenum));
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexStorage2D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
    G(glGenRenderbuffers, void (*)(GLsizei, GLuint*));
    G(glBindRenderbuffer, void (*)(GLenum, GLuint));
    G(glRenderbufferStorageMultisample, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
    G(glGetRenderbufferParameteriv, void (*)(GLenum, GLenum, GLint*));
    G(glClearColor, void (*)(GLfloat, GLfloat, GLfloat, GLfloat));
    G(glClear, void (*)(GLbitfield));
    G(glDrawBuffers, void (*)(GLsizei, const GLenum*));
    G(glBlitFramebuffer, void (*)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, int64_t, int64_t));
    G(glGetFramebufferAttachmentParameteriv, void (*)(GLenum, GLenum, GLenum, GLint*));
    GLuint fb, tex[2], rb;
    glGenFramebuffers(1, &fb);
    glGenTextures(2, tex);
    glGenRenderbuffers(1, &rb);
    glBindFramebuffer(gl::FRAMEBUFFER, fb);
    CHECK(glCheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
    for (int i = 0; i < 2; ++i) {
        glBindTexture(gl::TEXTURE_2D, tex[i]);
        glTexStorage2D(gl::TEXTURE_2D, 1, i ? gl::RGBA16F : gl::RGBA8, 32, 32);
        glFramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0 + GLenum(i), gl::TEXTURE_2D, tex[i], 0);
    }
    glBindRenderbuffer(gl::RENDERBUFFER, rb);
    glRenderbufferStorageMultisample(gl::RENDERBUFFER, 4, gl::DEPTH24_STENCIL8, 32, 32);
    GLint samples = 0;
    glGetRenderbufferParameteriv(gl::RENDERBUFFER, gl::RENDERBUFFER_SAMPLES, &samples);
    CHECK(samples == 4);  // reported as asked (rendered 1x)
    glFramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_STENCIL_ATTACHMENT, gl::RENDERBUFFER, rb);
    CHECK(glCheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE);
    GLint type = 0;
    glGetFramebufferAttachmentParameteriv(gl::FRAMEBUFFER, (gl::COLOR_ATTACHMENT0 + 1), gl::FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &type);
    CHECK(type == GLint(gl::FLOAT));
    const GLenum bufs[2] = {gl::COLOR_ATTACHMENT0, (gl::COLOR_ATTACHMENT0 + 1)};
    glDrawBuffers(2, bufs);
    counts = {};
    glClearColor(1, 0, 0, 1);
    glClear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
    CHECK(err() == gl::NO_ERROR && counts.passes == 1 && counts.clears == 1 && counts.lastColorCount == 2);
    const GLenum bad[1] = {(gl::COLOR_ATTACHMENT0 + 1)};
    glDrawBuffers(1, bad);
    CHECK(err() == gl::INVALID_OPERATION);
    // Blit into the default framebuffer.
    glBindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 64, 64, gl::COLOR_BUFFER_BIT, gl::LINEAR);
    CHECK(err() == gl::NO_ERROR && counts.blits == 1);
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 64, 64, gl::DEPTH_BUFFER_BIT, gl::LINEAR);
    CHECK(err() == gl::INVALID_OPERATION);
    e.down();
}

const char* kVs100 =
    "attribute vec4 a_pos; attribute vec2 a_uv; attribute vec4 a_col;\n"
    "uniform mat4 u_mvp; varying vec2 v_uv; varying vec4 v_col;\n"
    "void main() { v_uv = a_uv; v_col = a_col; gl_Position = u_mvp * a_pos; }\n";
const char* kFs100 =
    "precision mediump float; varying vec2 v_uv; varying vec4 v_col; uniform sampler2D u_tex;\n"
    "void main() { gl_FragColor = texture2D(u_tex, v_uv) * v_col; }\n";

void testProgram() {
    Egl e;
    CHECK(e.up());
    GLuint p = program(kVs100, kFs100);
    G(glGetUniformLocation, GLint (*)(GLuint, const char*));
    G(glGetAttribLocation, GLint (*)(GLuint, const char*));
    G(glUseProgram, void (*)(GLuint));
    G(glUniformMatrix4fv, void (*)(GLint, GLsizei, GLboolean, const GLfloat*));
    G(glUniform1i, void (*)(GLint, GLint));
    G(glUniform1f, void (*)(GLint, GLfloat));
    G(glGetUniformfv, void (*)(GLuint, GLint, GLfloat*));
    G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
    G(glGetActiveUniform, void (*)(GLuint, GLuint, GLsizei, GLsizei*, GLint*, GLenum*, char*));
    GLint mvp = glGetUniformLocation(p, "u_mvp"), tex = glGetUniformLocation(p, "u_tex");
    CHECK(mvp >= 0 && tex >= 0 && mvp != tex);
    CHECK(glGetUniformLocation(p, "nope") == -1);
    CHECK(glGetAttribLocation(p, "a_pos") >= 0 && glGetAttribLocation(p, "a_col") >= 0);
    GLint n = 0;
    glGetProgramiv(p, gl::ACTIVE_UNIFORMS, &n);
    CHECK(n == 2);
    glGetProgramiv(p, gl::ACTIVE_ATTRIBUTES, &n);
    CHECK(n == 3);
    glUseProgram(p);
    float m[16] = {};
    for (int i = 0; i < 16; ++i) m[i] = float(i);
    glUniformMatrix4fv(mvp, 1, 0, m);
    glUniform1i(tex, 3);
    CHECK(err() == gl::NO_ERROR);
    float back[16] = {};
    glGetUniformfv(p, mvp, back);
    CHECK(back[5] == 5.0f && back[15] == 15.0f);
    glGetUniformfv(p, tex, back);
    CHECK(back[0] == 3.0f);
    glUniform1f(mvp, 1.0f);  // wrong type
    CHECK(err() == gl::INVALID_OPERATION);
    glUniform1i(tex, 99);
    CHECK(err() == gl::INVALID_VALUE);
    // A link failure is reported, not fatal.
    G(glCreateShader, GLuint (*)(GLenum));
    G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
    G(glCompileShader, void (*)(GLuint));
    G(glGetShaderiv, void (*)(GLuint, GLenum, GLint*));
    G(glGetShaderInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
    GLuint s = glCreateShader(gl::FRAGMENT_SHADER);
    const char* bad = "void main() { gl_FragColor = vec3(1.0); }";
    glShaderSource(s, 1, &bad, nullptr);
    glCompileShader(s);
    GLint ok = 1;
    glGetShaderiv(s, gl::COMPILE_STATUS, &ok);
    CHECK(!ok);
    char log[512];
    glGetShaderInfoLog(s, sizeof log, nullptr, log);
    CHECK(std::strstr(log, "ERROR") != nullptr);
    e.down();
}

void testDraw() {
    Egl e;
    CHECK(e.up());
    GLuint p = program(kVs100, kFs100);
    G(glUseProgram, void (*)(GLuint));
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    G(glGetAttribLocation, GLint (*)(GLuint, const char*));
    G(glVertexAttribPointer, void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
    G(glEnableVertexAttribArray, void (*)(GLuint));
    G(glVertexAttrib4f, void (*)(GLuint, GLfloat, GLfloat, GLfloat, GLfloat));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glDrawElements, void (*)(GLenum, GLsizei, GLenum, const void*));
    G(glDrawElementsInstanced, void (*)(GLenum, GLsizei, GLenum, const void*, GLsizei));
    G(glEnable, void (*)(GLenum));
    G(glBlendFunc, void (*)(GLenum, GLenum));
    G(glDepthFunc, void (*)(GLenum));
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexImage2D, void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));
    G(eglSwapBuffers, uint32_t (*)(void*, void*));
    G(glFinish, void (*)());
    glUseProgram(p);
    const float verts[] = {-1, -1, 0, 1, 0, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1};
    GLuint vb;
    glGenBuffers(1, &vb);
    glBindBuffer(gl::ARRAY_BUFFER, vb);
    glBufferData(gl::ARRAY_BUFFER, sizeof verts, verts, gl::STATIC_DRAW);
    const GLint pos = glGetAttribLocation(p, "a_pos"), uv = glGetAttribLocation(p, "a_uv"), col = glGetAttribLocation(p, "a_col");
    glVertexAttribPointer(GLuint(pos), 4, gl::FLOAT, 0, 24, nullptr);
    glVertexAttribPointer(GLuint(uv), 2, gl::FLOAT, 0, 24, reinterpret_cast<void*>(16));
    glEnableVertexAttribArray(GLuint(pos));
    glEnableVertexAttribArray(GLuint(uv));
    glVertexAttrib4f(GLuint(col), 1, 1, 1, 1);  // constant attribute (disabled array)
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(gl::TEXTURE_2D, t);
    const uint8_t white[4] = {255, 255, 255, 255};
    glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 1, 1, 0, gl::RGBA, gl::UNSIGNED_BYTE, white);
    glEnable(gl::BLEND);
    glBlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    glEnable(gl::DEPTH_TEST);
    glDepthFunc(gl::LEQUAL);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::NO_ERROR && counts.draws == 1 && counts.pipelines == 1);
    CHECK(counts.lastFrontFace == VK_FRONT_FACE_CLOCKWISE);  // GL CCW, rows bottom-up
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(counts.pipelines == 1 && counts.passes == 1);  // pipeline and pass reused
    // Client-side arrays + client indices, triangle fans, line loops, unsigned byte indices.
    glBindBuffer(gl::ARRAY_BUFFER, 0);
    glVertexAttribPointer(GLuint(pos), 4, gl::FLOAT, 0, 24, verts);
    glVertexAttribPointer(GLuint(uv), 2, gl::UNSIGNED_SHORT, 1, 0, verts);  // normalized ushort
    const uint8_t idx8[] = {0, 1, 2};
    glDrawElements(gl::TRIANGLE_FAN, 3, gl::UNSIGNED_BYTE, idx8);
    CHECK(err() == gl::NO_ERROR && counts.drawsIndexed == 1);
    glDrawArrays(gl::LINE_LOOP, 0, 3);
    CHECK(err() == gl::NO_ERROR && counts.drawsIndexed == 2);
    const uint16_t idx16[] = {0, 1, 2};
    glDrawElementsInstanced(gl::TRIANGLES, 3, gl::UNSIGNED_SHORT, idx16, 4);
    CHECK(err() == gl::NO_ERROR && counts.lastInstances == 4);
    glDrawArrays(0x77, 0, 3);
    CHECK(err() == gl::INVALID_ENUM);
    glUseProgram(0);
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::INVALID_OPERATION);
    glFinish();
    CHECK(eglSwapBuffers(e.dpy, e.surface));
    e.down();
}

void testMultiview() {
    Egl e;
    CHECK(e.up());
    GLuint p = program(
        "#version 300 es\n#extension GL_OVR_multiview2 : require\nlayout(num_views = 2) in;\n"
        "in vec4 a_pos; uniform mat4 u_vp[2];\n"
        "void main() { gl_Position = u_vp[gl_ViewID_OVR] * a_pos; }\n",
        "#version 300 es\nprecision mediump float; out vec4 o; void main() { o = vec4(1.0); }\n");
    G(glUseProgram, void (*)(GLuint));
    G(glGenFramebuffers, void (*)(GLsizei, GLuint*));
    G(glBindFramebuffer, void (*)(GLenum, GLuint));
    G(glFramebufferTextureMultiviewOVR, void (*)(GLenum, GLenum, GLuint, GLint, GLint, GLsizei));
    G(glCheckFramebufferStatus, GLenum (*)(GLenum));
    G(glVertexAttribPointer, void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
    G(glEnableVertexAttribArray, void (*)(GLuint));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glViewport, void (*)(GLint, GLint, GLsizei, GLsizei));
    // The swapchain path: a VkImage the host made (here: one we create through the same device),
    // wrapped as a 2D array texture, rendered with OVR_multiview.
    void *inst, *phys, *dev, *queue;
    vpgles_device(&inst, &phys, &dev, &queue);
    auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(realGipa(static_cast<VkInstance>(inst), "vkGetDeviceProcAddr"));
    auto createImage = reinterpret_cast<PFN_vkCreateImage>(gdpa(static_cast<VkDevice>(dev), "vkCreateImage"));
    auto getReq = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(gdpa(static_cast<VkDevice>(dev), "vkGetImageMemoryRequirements"));
    auto alloc = reinterpret_cast<PFN_vkAllocateMemory>(gdpa(static_cast<VkDevice>(dev), "vkAllocateMemory"));
    auto bind = reinterpret_cast<PFN_vkBindImageMemory>(gdpa(static_cast<VkDevice>(dev), "vkBindImageMemory"));
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = VK_FORMAT_R8G8B8A8_SRGB, ci.extent = {256, 256, 1};
    ci.mipLevels = 1, ci.arrayLayers = 2, ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage img;
    CHECK(createImage(static_cast<VkDevice>(dev), &ci, nullptr, &img) == VK_SUCCESS);
    VkMemoryRequirements req;
    getReq(static_cast<VkDevice>(dev), img, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    VkDeviceMemory mem;
    alloc(static_cast<VkDevice>(dev), &mai, nullptr, &mem);
    bind(static_cast<VkDevice>(dev), img, mem, 0);
    const GLuint tex = vpgles_wrap_image(img, VK_FORMAT_R8G8B8A8_SRGB, gl::TEXTURE_2D_ARRAY, 256, 256, 2, 1);
    CHECK(tex != 0);
    GLuint fb;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(gl::DRAW_FRAMEBUFFER, fb);
    glFramebufferTextureMultiviewOVR(gl::DRAW_FRAMEBUFFER, gl::COLOR_ATTACHMENT0, tex, 0, 0, 2);
    CHECK(glCheckFramebufferStatus(gl::DRAW_FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE);
    glUseProgram(p);
    glViewport(0, 0, 256, 256);
    const float tri[] = {-1, -1, 0, 1, 3, -1, 0, 1, -1, 3, 0, 1};
    glVertexAttribPointer(0, 4, gl::FLOAT, 0, 0, tri);
    glEnableVertexAttribArray(0);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::NO_ERROR && counts.draws == 1 && counts.lastViewMask == 3);
    vpgles_flush(1);
    e.down();
}

void testErrors() {
    Egl e;
    CHECK(e.up());
    G(glEnable, void (*)(GLenum));
    G(glActiveTexture, void (*)(GLenum));
    G(glViewport, void (*)(GLint, GLint, GLsizei, GLsizei));
    G(glPixelStorei, void (*)(GLenum, GLint));
    G(glBindVertexArray, void (*)(GLuint));
    G(glUseProgram, void (*)(GLuint));
    G(glIsEnabled, GLboolean (*)(GLenum));
    G(glGetIntegerv, void (*)(GLenum, GLint*));
    glEnable(0x1234);
    CHECK(err() == gl::INVALID_ENUM);
    CHECK(err() == gl::NO_ERROR);  // cleared on read
    glActiveTexture(gl::TEXTURE0 + 100);
    CHECK(err() == gl::INVALID_ENUM);
    glViewport(0, 0, -1, 5);
    CHECK(err() == gl::INVALID_VALUE);
    glPixelStorei(gl::UNPACK_ALIGNMENT, 3);
    CHECK(err() == gl::INVALID_VALUE);
    glBindVertexArray(777);
    CHECK(err() == gl::INVALID_OPERATION);
    glUseProgram(12345);
    CHECK(err() == gl::INVALID_VALUE);
    glEnable(gl::SCISSOR_TEST);
    CHECK(glIsEnabled(gl::SCISSOR_TEST));
    GLint maxTex = 0;
    glGetIntegerv(gl::MAX_TEXTURE_SIZE, &maxTex);
    CHECK(maxTex == 16384);
    e.down();
    // No current context: calls are ignored, not crashes.
    G(glClear, void (*)(GLbitfield));
    glClear(gl::COLOR_BUFFER_BIT);
}

void testReadback() {
    Egl e;
    CHECK(e.up(32, 32));
    G(glReadPixels, void (*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*));
    G(glClear, void (*)(GLbitfield));
    std::vector<uint8_t> px(32 * 32 * 4, 0xAB);
    counts = {};
    glClear(gl::COLOR_BUFFER_BIT);
    glReadPixels(0, 0, 32, 32, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
    CHECK(err() == gl::NO_ERROR && counts.copiesToBuffer == 1);
    // Partly outside: only the inside is written.
    std::vector<uint8_t> small(8 * 8 * 4, 0xCD);
    glReadPixels(28, 28, 8, 8, gl::RGBA, gl::UNSIGNED_BYTE, small.data());
    CHECK(err() == gl::NO_ERROR && small[(7 * 8 + 7) * 4] == 0xCD);
    glReadPixels(0, 0, 4, 4, gl::RGBA, 0x9999, px.data());
    CHECK(err() == gl::INVALID_ENUM);
    e.down();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    void* lib = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::printf("dlopen %s: %s\n", argv[2], dlerror());
        return 1;
    }
    realGipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
    vpgles_init = reinterpret_cast<decltype(vpgles_init)>(dlsym(lib, "vpgles_init"));
    vpgles_proc = reinterpret_cast<decltype(vpgles_proc)>(dlsym(lib, "vpgles_proc"));
    vpgles_device = reinterpret_cast<decltype(vpgles_device)>(dlsym(lib, "vpgles_device"));
    vpgles_wrap_image = reinterpret_cast<decltype(vpgles_wrap_image)>(dlsym(lib, "vpgles_wrap_image"));
    vpgles_flush = reinterpret_cast<decltype(vpgles_flush)>(dlsym(lib, "vpgles_flush"));
    if (!vpgles_init || !vpgles_proc || !vpgles_device || !vpgles_wrap_image || !vpgles_flush) {
        std::printf("libVPVulkan has no vpgles_* exports\n");
        return 1;
    }
    captureStderr();
    const int ok = vpgles_init(reinterpret_cast<void*>(&myGipa));
    const std::map<std::string, std::function<void()>> tests = {
        {"egl", testEgl}, {"eglthreads", testEglThreads}, {"formats", testFormats}, {"buffers", testBuffers}, {"textures", testTextures},
        {"fbo", testFbo}, {"program", testProgram}, {"draw", testDraw}, {"multiview", testMultiview},
        {"errors", testErrors}, {"readback", testReadback},
    };
    auto it = tests.find(argv[1]);
    if (ok && it != tests.end()) it->second();
    const std::string log = releaseStderr();
    CHECK(ok);
    // The driver accepted everything the layer gave it.
    for (const char* bad : {"vk.err", "vk.unimpl", "gles.err"}) {
        if (log.find(bad) != std::string::npos) {
            std::printf("  driver log has '%s':\n%s\n", bad, log.c_str());
            ++failures;
            break;
        }
    }
    if (std::getenv("VPGLES_LOG")) std::printf("%s\n", log.c_str());
    std::printf("%s: %s\n", argv[1], failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
