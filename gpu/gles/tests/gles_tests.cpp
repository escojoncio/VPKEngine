// GLES layer tests against the real VPVulkan driver on its null platform (dlopened, the way Klepton
// loads a Vulkan driver). The null backend executes nothing, so these check what can be checked
// without a GPU: GL state and errors, the Vulkan calls the layer makes (intercepted), and that the
// driver accepted everything (stderr scanned for vk.err / vk.unimpl / gles.err lines: every
// pipeline really went through spirv::lower and the MSL emitter inside VPVulkan).
// Usage: vpgles_tests <group> <path to libVPVulkan>
#include "xfb.h"
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "formats.h"
#include "gl_enums.h"
#include "gl_enums31.h"
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
bool expectLinkFailedVaryings = false;
bool expectMipBeforeBase = false, expectTexStats = false;  // testTextures: the diagnostics lines must be in the driver log
std::vector<std::string> expectLines;  // substrings the driver log must contain at the end of the run
std::string gSelf, gLibPath;  // argv[0] and the driver path: the `recipe` group runs `recipe_session` children of this binary
  // testProgram: the link-failure varying diagnostic must survive 70 warnings
#define CHECK(c)                                                                \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);          \
            ++failures;                                                         \
        }                                                                       \
    } while (0)

// ---- Vulkan interception ----------------------------------------------------------------------------
void* gLib = nullptr;  // the dlopened driver
PFN_vkGetInstanceProcAddr realGipa = nullptr;
PFN_vkGetDeviceProcAddr realGdpa = nullptr;
struct Counts {
    int draws = 0, drawsIndexed = 0, passes = 0, pipelines = 0, copiesToImage = 0, clears = 0, dispatches = 0,
        blits = 0, copiesToBuffer = 0, drawsIndirect = 0, drawsIndexedIndirect = 0, dispatchesIndirect = 0, copiesImage = 0,
        imagesWithStorage = 0, imagesCreated = 0, clearColorImages = 0, clearDepthImages = 0;
    uint32_t lastCopyRegions = 0;
    // Last image view created (format, type) and last image created (flags, samples).
    VkFormat lastViewFormat = VK_FORMAT_UNDEFINED;
    VkImageViewType lastViewType = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
    VkImageCreateFlags lastImageFlags = 0;
    VkSampleCountFlagBits lastImageSamples = VK_SAMPLE_COUNT_1_BIT, lastRasterSamples = VK_SAMPLE_COUNT_1_BIT;
    int imagesMultisample = 0, viewsOf3DAs2D = 0;
    // Last graphics pipeline: colour write mask of attachment 0 (0xFFFFFFFF = none yet), stencil test, depth test.
    uint32_t lastWriteMask = 0xFFFFFFFFu;
    int lastStencilTest = -1, lastDepthTest = -1;
    // The image view of the last combined image sampler descriptor written (a texture's own or the black stand-in).
    VkImageView lastSampledView = VK_NULL_HANDLE;
    uint32_t lastViewMask = 0, lastColorCount = 0, lastInstances = 0;
    std::vector<VkFormat> lastAttrFormats;
    VkFrontFace lastFrontFace = VK_FRONT_FACE_MAX_ENUM;
    // Load / store ops of the last pass (first colour attachment, depth attachment).
    VkAttachmentLoadOp colorLoad = VK_ATTACHMENT_LOAD_OP_MAX_ENUM, depthLoad = VK_ATTACHMENT_LOAD_OP_MAX_ENUM;
    VkAttachmentStoreOp colorStore = VK_ATTACHMENT_STORE_OP_MAX_ENUM, depthStore = VK_ATTACHMENT_STORE_OP_MAX_ENUM;
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
PFN_vkCmdDrawIndirect realDrawIndirect;
PFN_vkCmdDrawIndexedIndirect realDrawIndexedIndirect;
PFN_vkCmdDispatchIndirect realDispatchIndirect;
PFN_vkCmdCopyImage realCopyImage;
PFN_vkCreateImage realCreateImage;
PFN_vkCreateImageView realCreateImageView;
PFN_vkCmdClearColorImage realClearColorImage;
PFN_vkCmdClearDepthStencilImage realClearDepthImage;
PFN_vkUpdateDescriptorSets realUpdateSets;
std::map<VkImage, VkImageType> imageTypes;  // images the layer created: their type (views of a 3D image as 2D are Metal aborts)

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
    if (ri->colorAttachmentCount) counts.colorLoad = ri->pColorAttachments[0].loadOp, counts.colorStore = ri->pColorAttachments[0].storeOp;
    if (ri->pDepthAttachment) counts.depthLoad = ri->pDepthAttachment->loadOp, counts.depthStore = ri->pDepthAttachment->storeOp;
    realBeginRendering(c, ri);
}
VKAPI_ATTR VkResult VKAPI_CALL myCreatePipelines(VkDevice d, VkPipelineCache pc, uint32_t n, const VkGraphicsPipelineCreateInfo* ci,
                                                 const VkAllocationCallbacks* a, VkPipeline* out) {
    ++counts.pipelines;
    counts.lastAttrFormats.clear();
    for (uint32_t i = 0; i < ci->pVertexInputState->vertexAttributeDescriptionCount; ++i)
        counts.lastAttrFormats.push_back(ci->pVertexInputState->pVertexAttributeDescriptions[i].format);
    counts.lastFrontFace = ci->pRasterizationState->frontFace;
    counts.lastRasterSamples = ci->pMultisampleState ? ci->pMultisampleState->rasterizationSamples : VK_SAMPLE_COUNT_1_BIT;
    counts.lastWriteMask = ci->pColorBlendState && ci->pColorBlendState->attachmentCount
                               ? ci->pColorBlendState->pAttachments[0].colorWriteMask : 0xFFFFFFFFu;
    counts.lastStencilTest = ci->pDepthStencilState ? int(ci->pDepthStencilState->stencilTestEnable) : -1;
    counts.lastDepthTest = ci->pDepthStencilState ? int(ci->pDepthStencilState->depthTestEnable) : -1;
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
// VPVulkan's private hook (glInvalidateFramebuffer with an open pass): the stores of the last pass become what it says.
typedef void (*PFN_passDrop)(VkCommandBuffer, uint32_t, int, int);
PFN_passDrop realPassDrop;
VKAPI_ATTR void VKAPI_CALL myPassDrop(VkCommandBuffer c, uint32_t colorMask, int depth, int stencil) {
    counts.colorStore = (colorMask & 1) ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    counts.depthStore = depth ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    realPassDrop(c, colorMask, depth, stencil);
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

VKAPI_ATTR void VKAPI_CALL myDrawIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t st) {
    ++counts.drawsIndirect;
    realDrawIndirect(c, b, o, n, st);
}
VKAPI_ATTR void VKAPI_CALL myDrawIndexedIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t st) {
    ++counts.drawsIndexedIndirect;
    realDrawIndexedIndirect(c, b, o, n, st);
}
VKAPI_ATTR void VKAPI_CALL myDispatchIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o) {
    ++counts.dispatchesIndirect;
    realDispatchIndirect(c, b, o);
}
VKAPI_ATTR void VKAPI_CALL myCopyImage(VkCommandBuffer c, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t n, const VkImageCopy* r) {
    ++counts.copiesImage;
    counts.lastCopyRegions = n;
    realCopyImage(c, si, sl, di, dl, n, r);
}
VKAPI_ATTR VkResult VKAPI_CALL myCreateImage(VkDevice d, const VkImageCreateInfo* ci, const VkAllocationCallbacks* a, VkImage* out) {
    ++counts.imagesCreated;
    if (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT) ++counts.imagesWithStorage;
    counts.lastImageFlags = ci->flags;
    counts.lastImageSamples = ci->samples;
    if (ci->samples != VK_SAMPLE_COUNT_1_BIT) ++counts.imagesMultisample;
    VkResult r = realCreateImage(d, ci, a, out);
    if (r == VK_SUCCESS) imageTypes[*out] = ci->imageType;
    return r;
}
VKAPI_ATTR VkResult VKAPI_CALL myCreateImageView(VkDevice d, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks* a, VkImageView* out) {
    counts.lastViewFormat = ci->format;
    counts.lastViewType = ci->viewType;
    auto it = imageTypes.find(ci->image);
    if (it != imageTypes.end() && it->second == VK_IMAGE_TYPE_3D && ci->viewType != VK_IMAGE_VIEW_TYPE_3D &&
        ci->viewType != VK_IMAGE_VIEW_TYPE_2D_ARRAY)
        ++counts.viewsOf3DAs2D;
    return realCreateImageView(d, ci, a, out);
}

VKAPI_ATTR void VKAPI_CALL myClearColorImage(VkCommandBuffer c, VkImage i, VkImageLayout l, const VkClearColorValue* v, uint32_t n,
                                             const VkImageSubresourceRange* r) {
    ++counts.clearColorImages;
    realClearColorImage(c, i, l, v, n, r);
}
VKAPI_ATTR void VKAPI_CALL myClearDepthImage(VkCommandBuffer c, VkImage i, VkImageLayout l, const VkClearDepthStencilValue* v, uint32_t n,
                                             const VkImageSubresourceRange* r) {
    ++counts.clearDepthImages;
    realClearDepthImage(c, i, l, v, n, r);
}
VKAPI_ATTR void VKAPI_CALL myUpdateSets(VkDevice d, uint32_t n, const VkWriteDescriptorSet* w, uint32_t cn, const VkCopyDescriptorSet* cp) {
    for (uint32_t i = 0; i < n; ++i)
        if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && w[i].descriptorCount) counts.lastSampledView = w[i].pImageInfo[0].imageView;
    realUpdateSets(d, n, w, cn, cp);
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
    I("vpvk_cmd_pass_drop", myPassDrop, realPassDrop)
    I("vkCmdDispatch", myDispatch, realDispatch)
    I("vkCmdBlitImage", myBlit, realBlit)
    I("vkCmdCopyImageToBuffer", myCopyToBuffer, realCopyToBuffer)
    I("vkCmdDrawIndirect", myDrawIndirect, realDrawIndirect)
    I("vkCmdDrawIndexedIndirect", myDrawIndexedIndirect, realDrawIndexedIndirect)
    I("vkCmdDispatchIndirect", myDispatchIndirect, realDispatchIndirect)
    I("vkCmdCopyImage", myCopyImage, realCopyImage)
    I("vkCreateImage", myCreateImage, realCreateImage)
    I("vkCreateImageView", myCreateImageView, realCreateImageView)
    I("vkCmdClearColorImage", myClearColorImage, realClearColorImage)
    I("vkCmdClearDepthStencilImage", myClearDepthImage, realClearDepthImage)
    I("vkUpdateDescriptorSets", myUpdateSets, realUpdateSets)
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
// What the layer has logged so far (the capture goes on).
std::string peekStderr() {
    std::fflush(stderr);
    std::ifstream f(stderrPath);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
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

// Ends the open pass (glFinish submits), so the next case starts a new one.
void settle() {
    G(glFinish, void (*)());
    glFinish();
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
    for (EGLint a : {egl::TRANSPARENT_RED_VALUE, egl::TRANSPARENT_GREEN_VALUE, egl::TRANSPARENT_BLUE_VALUE}) {
        r = -1;
        CHECK(eglGetConfigAttrib(d, cfgs[0], a, &r) && r == 0);
    }
    CHECK(eglGetConfigAttrib(d, cfgs[0], egl::TRANSPARENT_TYPE, &r) && r == egl::NONE);
    CHECK(std::strstr(eglQueryString(d, egl::EXTENSIONS), "EGL_KHR_fence_sync"));
    Egl e;
    counts = {};
    CHECK(e.up(128, 96));
    // A surface's images start zeroed (once, at creation): an unwritten one that is presented as it is showed Metal's
    // uninitialised magenta. Colour and depth/stencil, pbuffer and window alike.
    CHECK(counts.clearColorImages == 1 && counts.clearDepthImages == 1);
    {
        G(eglCreateWindowSurface, void* (*)(void*, void*, void*, const EGLint*));
        G(eglDestroySurface, uint32_t (*)(void*, void*));
        counts = {};
        const EGLint none[] = {egl::NONE};
        void* ws = eglCreateWindowSurface(d, e.config, reinterpret_cast<void*>(0x1000), none);
        CHECK(ws && counts.clearColorImages == 1 && counts.clearDepthImages == 1);
        eglDestroySurface(d, ws);
    }
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
    {
        const std::string want = "OpenGL ES " + std::to_string(gl::kAdvertisedMajor) + "." + std::to_string(gl::kAdvertisedMinor);
        CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::VERSION)), want.c_str()));
    }
    CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::EXTENSIONS)), "GL_OVR_multiview2"));
    // Default: the Apple GPU's own identity; the Adreno 740 only with the Quest identity (asked on every call).
    {
        auto setQuest = reinterpret_cast<void (*)(int)>(dlsym(gLib, "vpvk_config_set_quest_identity"));
        CHECK(setQuest);
        CHECK(!std::strstr(reinterpret_cast<const char*>(glGetString(gl::RENDERER)), "Adreno"));
        CHECK(!std::strstr(reinterpret_cast<const char*>(glGetString(gl::VENDOR)), "Qualcomm"));
        if (setQuest) {
            setQuest(1);
            CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::RENDERER)), "Adreno"));
            CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::VENDOR)), "Qualcomm"));
            setQuest(0);
        }
    }
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
    // Readback of formats without a raw layout match: RG8, R16I, R16F, RGB10_A2, R11F_G11F_B10F into the usual client types.
    uint8_t rg8[2] = {255, 128}, rbOut[16];
    CHECK(convertReadback(*formatFor(gl::RG8), gl::RGBA, gl::UNSIGNED_BYTE, rg8, 1, rbOut));
    CHECK(rbOut[0] == 255 && rbOut[1] == 128 && rbOut[2] == 0 && rbOut[3] == 255);
    int16_t r16i = -32768;
    int32_t i4[4];
    CHECK(convertReadback(*formatFor(gl::R16I), gl::RGBA_INTEGER, gl::INT, reinterpret_cast<uint8_t*>(&r16i), 1, rbOut));
    std::memcpy(i4, rbOut, 16);
    CHECK(i4[0] == -32768 && i4[1] == 0 && i4[2] == 0 && i4[3] == 1);
    uint16_t r16f = 0x3800;  // 0.5
    float f4[4];
    CHECK(convertReadback(*formatFor(gl::R16F), gl::RGBA, gl::FLOAT, reinterpret_cast<uint8_t*>(&r16f), 1, rbOut));
    std::memcpy(f4, rbOut, 16);
    CHECK(f4[0] == 0.5f && f4[1] == 0 && f4[3] == 1);
    uint32_t a2 = 0xC0000000u | 1023u << 20 | 512u << 10;
    CHECK(convertReadback(*formatFor(gl::RGB10_A2), gl::RGBA, gl::UNSIGNED_BYTE, reinterpret_cast<uint8_t*>(&a2), 1, rbOut));
    CHECK(rbOut[0] == 0 && rbOut[1] == 128 && rbOut[2] == 255 && rbOut[3] == 255);
    uint32_t r11 = (0x3C00 >> 4) | (0x3800 >> 4) << 11 | (0x4000 >> 5) << 22;  // 1.0, 0.5, 2.0
    CHECK(convertReadback(*formatFor(gl::R11F_G11F_B10F), gl::RGBA, gl::FLOAT, reinterpret_cast<uint8_t*>(&r11), 1, rbOut));
    std::memcpy(f4, rbOut, 16);
    CHECK(f4[0] == 1.0f && f4[1] == 0.5f && f4[2] == 2.0f && f4[3] == 1.0f);
    CHECK(clientPixelBytes(gl::RGBA, gl::FLOAT) == 16 && clientPixelBytes(gl::RGB, gl::UNSIGNED_SHORT_5_6_5) == 2);
    // DEPTH_COMPONENT uploads: every client type into D16 / D32F / D24.
    const FormatInfo* d16 = formatFor(gl::DEPTH_COMPONENT16);
    const FormatInfo* d32f = formatFor(gl::DEPTH_COMPONENT32F);
    const FormatInfo* d24 = formatFor(gl::DEPTH_COMPONENT24);
    const uint32_t du32[3] = {0u, 0x80000000u, 0xFFFFFFFFu};
    CHECK(convertUpload(*d16, gl::DEPTH_COMPONENT, gl::UNSIGNED_INT, reinterpret_cast<const uint8_t*>(du32), 3, out));
    uint16_t o16[3];
    CHECK(out.size() == 6);
    std::memcpy(o16, out.data(), out.size() < 6 ? out.size() : 6);
    CHECK(o16[0] == 0 && o16[1] == 0x8000 && o16[2] == 0xFFFF);
    const uint16_t du16[2] = {0, 0xFFFF};
    CHECK(convertUpload(*d16, gl::DEPTH_COMPONENT, gl::UNSIGNED_SHORT, reinterpret_cast<const uint8_t*>(du16), 2, out));
    CHECK(out.size() == 4 && std::memcmp(out.data(), du16, 4) == 0);
    const float df[3] = {-1.0f, 0.5f, 2.0f};
    CHECK(convertUpload(*d16, gl::DEPTH_COMPONENT, gl::FLOAT, reinterpret_cast<const uint8_t*>(df), 3, out));
    CHECK(out.size() == 6);
    std::memcpy(o16, out.data(), out.size() < 6 ? out.size() : 6);
    CHECK(o16[0] == 0 && o16[1] == 32768 && o16[2] == 0xFFFF);
    float o32f[3];
    CHECK(convertUpload(*d32f, gl::DEPTH_COMPONENT, gl::UNSIGNED_SHORT, reinterpret_cast<const uint8_t*>(du16), 2, out));
    CHECK(out.size() == 8);
    std::memcpy(o32f, out.data(), out.size() < 8 ? out.size() : 8);
    CHECK(o32f[0] == 0.0f && o32f[1] == 1.0f);
    CHECK(convertUpload(*d32f, gl::DEPTH_COMPONENT, gl::UNSIGNED_INT, reinterpret_cast<const uint8_t*>(du32), 3, out));
    CHECK(out.size() == 12);
    std::memcpy(o32f, out.data(), out.size() < 12 ? out.size() : 12);
    CHECK(o32f[0] == 0.0f && o32f[1] > 0.499f && o32f[1] < 0.501f && o32f[2] == 1.0f);
    CHECK(convertUpload(*d32f, gl::DEPTH_COMPONENT, gl::FLOAT, reinterpret_cast<const uint8_t*>(df), 3, out));
    CHECK(out.size() == 12 && std::memcmp(out.data(), df, 12) == 0);
    CHECK(convertUpload(*d24, gl::DEPTH_COMPONENT, gl::FLOAT, reinterpret_cast<const uint8_t*>(df), 3, out));
    uint32_t o24[3];
    CHECK(out.size() == 12);
    std::memcpy(o24, out.data(), out.size() < 12 ? out.size() : 12);
    CHECK(o24[0] == 0 && o24[1] == 0x800000 && o24[2] == 0xFFFFFF);
    // UNSIGNED_SHORT into D24: scaled to 24 bits without overflow.
    const uint16_t du16b[3] = {0, 0x8000, 0xFFFF};
    CHECK(convertUpload(*d24, gl::DEPTH_COMPONENT, gl::UNSIGNED_SHORT, reinterpret_cast<const uint8_t*>(du16b), 3, out));
    CHECK(out.size() == 12);
    std::memcpy(o24, out.data(), out.size() < 12 ? out.size() : 12);
    CHECK(o24[0] == 0 && o24[1] == 0x800080 && o24[2] == 0xFFFFFF);
    // NaN depth uploads as 0.
    const float dnan[1] = {std::numeric_limits<float>::quiet_NaN()};
    CHECK(convertUpload(*d16, gl::DEPTH_COMPONENT, gl::FLOAT, reinterpret_cast<const uint8_t*>(dnan), 1, out));
    CHECK(out.size() == 2 && out[0] == 0 && out[1] == 0);
    // UNSIGNED_INT into D24 must take the top 24 bits, not a raw copy.
    CHECK(convertUpload(*d24, gl::DEPTH_COMPONENT, gl::UNSIGNED_INT, reinterpret_cast<const uint8_t*>(du32), 3, out));
    CHECK(out.size() == 12);
    std::memcpy(o24, out.data(), out.size() < 12 ? out.size() : 12);
    CHECK(o24[0] == 0 && o24[1] == 0x800000 && o24[2] == 0xFFFFFF);
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
    G(glCompressedTexSubImage2D, void (*)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLsizei, const void*));
    G(glGenerateMipmap, void (*)(GLenum));
    G(glTexParameteri, void (*)(GLenum, GLenum, GLint));
    G(glGetTexParameteriv, void (*)(GLenum, GLenum, GLint*));
    G(glPixelStorei, void (*)(GLenum, GLint));
    GLuint t[4];
    glGenTextures(4, t);
    glBindTexture(gl::TEXTURE_2D, t[0]);
    // Pure integer parameters (ES 3.2 / EXT_texture_border_clamp): the border colour keeps its exact integers.
    {
        G(glTexParameterIiv, void (*)(GLenum, GLenum, const GLint*));
        G(glGetTexParameterIiv, void (*)(GLenum, GLenum, GLint*));
        G(glTexParameterIuiv, void (*)(GLenum, GLenum, const GLuint*));
        G(glGetTexParameterIuiv, void (*)(GLenum, GLenum, GLuint*));
        CHECK(glTexParameterIiv && glGetTexParameterIiv && glTexParameterIuiv && glGetTexParameterIuiv);
        const GLint set[4] = {-5, 7, 123456789, -2000000000};
        GLint got[4] = {0, 0, 0, 0};
        glTexParameterIiv(gl::TEXTURE_2D, gl::TEXTURE_BORDER_COLOR, set);
        glGetTexParameterIiv(gl::TEXTURE_2D, gl::TEXTURE_BORDER_COLOR, got);
        CHECK(err() == gl::NO_ERROR && got[0] == -5 && got[1] == 7 && got[2] == 123456789 && got[3] == -2000000000);
        const GLint wrap = GLint(gl::CLAMP_TO_EDGE);
        GLuint ugot = 0;
        glTexParameterIiv(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, &wrap);
        glGetTexParameterIuiv(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, &ugot);
        CHECK(err() == gl::NO_ERROR && ugot == gl::CLAMP_TO_EDGE);
    }
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
    // CompressedTexSubImage with a format nobody knows is INVALID_ENUM (it used to dereference a null format); a known
    // compressed format that is not the level's is INVALID_OPERATION; the level's own format works.
    std::vector<uint8_t> blk(16, 0);
    glCompressedTexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, 4, 4, 0x1234, GLsizei(blk.size()), blk.data());
    CHECK(err() == gl::INVALID_ENUM);
    glCompressedTexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, 4, 4, gl::COMPRESSED_RGBA_ASTC_12x12, GLsizei(blk.size()), blk.data());
    CHECK(err() == gl::INVALID_OPERATION);
    glCompressedTexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, 4, 4, gl::COMPRESSED_RGBA_ASTC_4x4, GLsizei(blk.size()), blk.data());
    CHECK(err() == gl::NO_ERROR);
    glBindTexture(gl::TEXTURE_CUBE_MAP, t[0]);  // t[0] is a 2D texture
    CHECK(err() == gl::INVALID_OPERATION);
    // No texture pays an explicit clear: VPVulkan zero-fills new images lazily (only when read before written).
    CHECK(counts.clearColorImages == 0 && counts.clearDepthImages == 0);
    // Mip levels uploaded before level 0 (UE4 streams the small mips first): they are kept until the base defines the
    // image and then uploaded, not dropped.
    {
        expectMipBeforeBase = true;
        GLuint m[4];
        glGenTextures(4, m);
        std::vector<uint8_t> a2(16, 1), a1(64, 2), a0(256, 3);
        glBindTexture(gl::TEXTURE_2D, m[0]);
        counts = {};
        glCompressedTexImage2D(gl::TEXTURE_2D, 2, gl::COMPRESSED_RGBA_ASTC_4x4, 4, 4, 0, GLsizei(a2.size()), a2.data());
        glCompressedTexImage2D(gl::TEXTURE_2D, 1, gl::COMPRESSED_RGBA_ASTC_4x4, 8, 8, 0, GLsizei(a1.size()), a1.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 0);  // no image yet
        glCompressedTexImage2D(gl::TEXTURE_2D, 0, gl::COMPRESSED_RGBA_ASTC_4x4, 16, 16, 0, GLsizei(a0.size()), a0.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 3);  // level 0 plus the two kept levels
        // TexImage with NULL then TexSubImage on a level before the base: the sub-image is kept too.
        glBindTexture(gl::TEXTURE_2D, m[1]);
        glPixelStorei(gl::UNPACK_ALIGNMENT, 4);
        counts = {};
        glTexImage2D(gl::TEXTURE_2D, 2, gl::RGBA, 4, 4, 0, gl::RGBA, gl::UNSIGNED_BYTE, nullptr);
        glTexSubImage2D(gl::TEXTURE_2D, 2, 1, 1, 2, 2, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 0);
        glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 16, 16, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 2);
        // A kept level that does not fit the base (level 1 of a 32x32 base is 16x16, not 8x8) is dropped.
        glBindTexture(gl::TEXTURE_2D, m[2]);
        counts = {};
        glTexImage2D(gl::TEXTURE_2D, 1, gl::RGBA, 8, 8, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 32, 32, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 1);
        // A level redefined before the base replaces the earlier one (one copy of it, not two).
        glBindTexture(gl::TEXTURE_2D, m[3]);
        counts = {};
        glTexImage2D(gl::TEXTURE_2D, 1, gl::RGBA, 8, 8, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        glTexImage2D(gl::TEXTURE_2D, 1, gl::RGBA, 8, 8, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 16, 16, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 2);
    }
    // An empty base level (0 x 0) leaves the texture without an image and without a GL error (the other levels are accepted);
    // a level larger than the base's mip chain allows is kept as a definition only (no upload into a smaller level).
    {
        GLuint z[2];
        glGenTextures(2, z);
        glBindTexture(gl::TEXTURE_2D, z[0]);
        counts = {};
        glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 0, 0, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        glTexImage2D(gl::TEXTURE_2D, 1, gl::RGBA, 8, 8, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 0);
        glBindTexture(gl::TEXTURE_2D, z[1]);
        counts = {};
        glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 32, 32, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        glTexImage2D(gl::TEXTURE_2D, 1, gl::RGBA, 17, 17, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        CHECK(err() == gl::NO_ERROR && counts.copiesToImage == 1);
    }
    // The first texture of a format is logged when it is made: the periodic summary needs a swap / flush / terminate, which
    // a host that presents through its own swapchain and is killed never gives.
    {
        GLuint rg;
        glGenTextures(1, &rg);
        glBindTexture(gl::TEXTURE_2D, rg);
        glTexStorage2D(gl::TEXTURE_2D, 1, gl::RG8, 8, 8);
        CHECK(err() == gl::NO_ERROR);
        CHECK(peekStderr().find("gles.tex fmt=0x822b count=1") != std::string::npos);
    }
    // Per-format creation counts reach the log (summary at eglTerminate / every 10 s).
    {
        G(eglTerminate, uint32_t (*)(void*));
        expectTexStats = true;
        eglTerminate(e.dpy);
    }
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
    CHECK(err() == gl::NO_ERROR && counts.passes == 1 && counts.clears == 0 && counts.lastColorCount == 2);  // the clear is the CLEAR load op
    const GLenum bad[1] = {(gl::COLOR_ATTACHMENT0 + 1)};
    glDrawBuffers(1, bad);
    CHECK(err() == gl::INVALID_OPERATION);
    // glClear / glClearBuffer honour the colour and stencil write masks: vkCmdClearAttachments cannot mask, so a masked
    // clear is a full-target draw whose pipeline has that write mask. Unmasked (or fully masked) clears keep their paths.
    {
        G(glColorMask, void (*)(GLboolean, GLboolean, GLboolean, GLboolean));
        G(glColorMaski, void (*)(GLuint, GLboolean, GLboolean, GLboolean, GLboolean));
        G(glStencilMask, void (*)(GLuint));
        G(glClearStencil, void (*)(GLint));
        G(glClearBufferfv, void (*)(GLenum, GLint, const GLfloat*));
        G(glClearBufferuiv, void (*)(GLenum, GLint, const GLuint*));
        const VkAttachmentLoadOp LOAD = VK_ATTACHMENT_LOAD_OP_LOAD, CLEAR = VK_ATTACHMENT_LOAD_OP_CLEAR;
        const GLenum one[1] = {gl::COLOR_ATTACHMENT0};
        glDrawBuffers(1, one);
        settle();
        counts = {};
        glColorMask(1, 1, 0, 1);  // blue masked
        glClear(gl::COLOR_BUFFER_BIT);
        settle();
        CHECK(err() == gl::NO_ERROR && counts.clears == 0 && counts.draws == 1 && counts.colorLoad == LOAD && counts.lastWriteMask == 0xB);
        CHECK(counts.lastStencilTest == 0 && counts.lastDepthTest == 0);
        counts = {};
        glColorMask(1, 1, 1, 1);
        glClear(gl::COLOR_BUFFER_BIT);
        settle();
        CHECK(counts.draws == 0 && counts.clears == 0 && counts.colorLoad == CLEAR);  // unmasked: the CLEAR load op
        counts = {};
        glColorMask(0, 0, 0, 0);
        glClear(gl::COLOR_BUFFER_BIT);
        settle();
        CHECK(counts.draws == 0 && counts.clears == 0);  // everything masked: nothing to do
        // Several draw buffers, each with its own mask: one draw.
        glDrawBuffers(2, bufs);
        glColorMask(1, 1, 1, 1);
        glColorMaski(1, 0, 1, 0, 1);
        counts = {};
        glClear(gl::COLOR_BUFFER_BIT);
        settle();
        CHECK(err() == gl::NO_ERROR && counts.draws == 1 && counts.lastColorCount == 2);
        glColorMaski(1, 1, 1, 1, 1);
        glDrawBuffers(1, one);
        // Stencil write mask.
        glClearStencil(0x55);
        glStencilMask(0x0F);
        counts = {};
        glClear(gl::STENCIL_BUFFER_BIT);
        settle();
        CHECK(err() == gl::NO_ERROR && counts.clears == 0 && counts.draws == 1 && counts.lastStencilTest == 1 &&
              counts.lastWriteMask == 0 && counts.lastDepthTest == 0);
        glStencilMask(0xFF);
        counts = {};
        glClear(gl::STENCIL_BUFFER_BIT);
        settle();
        CHECK(counts.draws == 0 && counts.clears == 1);
        glStencilMask(0);
        counts = {};
        glClear(gl::STENCIL_BUFFER_BIT);
        settle();
        CHECK(counts.draws == 0 && counts.clears == 0);
        glStencilMask(0xFF);
        // Colour masked and depth cleared together: the colour is the draw, the depth stays a clear.
        glColorMask(1, 1, 0, 0);
        counts = {};
        glClear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        settle();
        CHECK(counts.draws == 1 && counts.lastWriteMask == 3 && counts.clears + (counts.depthLoad == CLEAR ? 1 : 0) == 1);
        // glClearBuffer*.
        const float red[4] = {1, 0, 0, 1};
        glColorMask(1, 0, 1, 1);
        counts = {};
        glClearBufferfv(gl::COLOR, 0, red);
        settle();
        CHECK(err() == gl::NO_ERROR && counts.clears == 0 && counts.draws == 1 && counts.lastWriteMask == 0xD);
        glColorMask(1, 1, 1, 1);
        counts = {};
        glClearBufferfv(gl::COLOR, 0, red);
        settle();
        CHECK(counts.draws == 0 && counts.colorLoad == CLEAR);
        // Integer colour attachment with a partial mask: an unsigned-integer output (the driver log must stay clean).
        GLuint ifb, itex;
        glGenFramebuffers(1, &ifb);
        glGenTextures(1, &itex);
        glBindTexture(gl::TEXTURE_2D, itex);
        glTexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8UI, 16, 16);
        glBindFramebuffer(gl::FRAMEBUFFER, ifb);
        glFramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, itex, 0);
        CHECK(glCheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE);
        const GLuint u4[4] = {1, 2, 3, 4};
        glColorMask(0, 1, 1, 1);
        counts = {};
        glClearBufferuiv(gl::COLOR, 0, u4);
        settle();
        CHECK(err() == gl::NO_ERROR && counts.clears == 0 && counts.draws == 1 && counts.lastWriteMask == 0xE);
        glColorMask(1, 1, 1, 1);
        glBindFramebuffer(gl::FRAMEBUFFER, fb);
        glDrawBuffers(2, bufs);
    }
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
    // Varying diagnostics: warnings (an unwritten fragment input links) must not use up the budget of link failures.
    {
        G(glCreateProgram, GLuint (*)());
        G(glAttachShader, void (*)(GLuint, GLuint));
        G(glLinkProgram, void (*)(GLuint));
        const char* vsU = "#version 300 es\nin vec4 a; void main() { gl_Position = a; }";
        const char* fsU = "#version 300 es\nprecision mediump float;\nin vec2 miss; out vec4 o; void main() { o = vec4(miss, 0.0, 1.0); }";
        for (int i = 0; i < 70; ++i) program(vsU, fsU);
        const char* src[2] = {"#version 300 es\nin vec4 a; out vec2 x; void main() { x = a.xy; gl_Position = a; }",
                              "#version 300 es\nprecision mediump float;\nin vec3 x; out vec4 o; void main() { o = vec4(x, 1.0); }"};
        GLuint sh[2];
        for (int i = 0; i < 2; ++i) {
            sh[i] = glCreateShader(i ? gl::FRAGMENT_SHADER : gl::VERTEX_SHADER);
            glShaderSource(sh[i], 1, &src[i], nullptr);
            glCompileShader(sh[i]);
        }
        GLuint bp = glCreateProgram();
        glAttachShader(bp, sh[0]);
        glAttachShader(bp, sh[1]);
        glLinkProgram(bp);
        GLint linked = 1;
        glGetProgramiv(bp, gl::LINK_STATUS, &linked);
        CHECK(!linked);
        expectLinkFailedVaryings = true;
    }
    // UE4-style packed uniform arrays: counts past the end clamp (no error); a call that lands on another uniform (an
    // element past the array) or has the wrong base type / count is rejected and named in a gles.uniform_err line.
    expectLines.push_back("vpk.version sha=");
    {
        G(glUniform4fv, void (*)(GLint, GLsizei, const GLfloat*));
        G(glUniform4iv, void (*)(GLint, GLsizei, const GLint*));
        G(glUniform4uiv, void (*)(GLint, GLsizei, const GLuint*));
        const char* vsP = "#version 300 es\nin vec4 a; uniform highp vec4 pu_h[4]; uniform highp ivec4 pu_i[2]; uniform highp uvec4 pu_u;\n"
                          "uniform highp mat4 pu_m[2];\n"
                          "void main() { gl_Position = a + pu_h[0] + pu_h[3] + vec4(pu_i[0]) + vec4(pu_i[1]) + vec4(pu_u) + pu_m[1][0]; }\n";
        const char* fsP = "#version 300 es\nprecision mediump float; out vec4 o; void main() { o = vec4(1.0); }\n";
        GLuint q = program(vsP, fsP);
        const GLint lh = glGetUniformLocation(q, "pu_h"), li = glGetUniformLocation(q, "pu_i"), lu = glGetUniformLocation(q, "pu_u"),
                    lm = glGetUniformLocation(q, "pu_m");
        CHECK(lh >= 0 && li == lh + 4 && lu == li + 2 && lm == lu + 1);
        glUseProgram(q);
        float h[16], back[16] = {};
        for (int i = 0; i < 16; ++i) h[i] = float(i + 1);
        glUniform4fv(lh + 1, 10, h);  // elements 1..3 only
        CHECK(err() == gl::NO_ERROR);
        glGetUniformfv(q, lh + 3, back);
        CHECK(back[0] == 9.0f && back[3] == 12.0f);
        glGetUniformfv(q, lh, back);
        CHECK(back[0] == 0.0f);
        const GLint iv[16] = {};
        glUniform4iv(li, 5, iv);  // clamps to the 2 elements
        float m[48];
        for (int i = 0; i < 48; ++i) m[i] = float(i);
        glUniformMatrix4fv(lm + 1, 3, 0, m);
        CHECK(err() == gl::NO_ERROR);
        glGetUniformfv(q, lm + 1, back);
        CHECK(back[5] == 5.0f);
        glUniform4fv(li, 1, h);  // vec4 call on an ivec4: base type mismatch
        CHECK(err() == gl::INVALID_OPERATION);
        const GLuint uv[8] = {};
        glUniform4uiv(lu, 2, uv);  // count > 1 on a non-array
        CHECK(err() == gl::INVALID_OPERATION);
        glUniform4fv(9999, 1, h);
        CHECK(err() == gl::INVALID_OPERATION);
        expectLines.push_back("gles.uniform_err name=pu_i[0] decl_type=ivec4 call_type=vec4 count=1 arraySize=2 loc=" +
                              std::to_string(li) + " reason=base_type_mismatch");
        expectLines.push_back("gles.uniform_err name=pu_u decl_type=uvec4 call_type=uvec4 count=2 arraySize=1 loc=" +
                              std::to_string(lu) + " reason=count_on_non_array");
        expectLines.push_back("gles.uniform_err name=? decl_type=? call_type=vec4 count=1 arraySize=0 loc=9999 reason=location_out_of_range");
    }
    e.down();
}

// Program binaries (ES 3.0): one vendor format; a blob re-links to the same locations; a bad blob or format
// is a failed link (app recompiles from source), never a GL error.
void testProgramBinary() {
    Egl e;
    CHECK(e.up());
    G(glCreateShader, GLuint (*)(GLenum));
    G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
    G(glCompileShader, void (*)(GLuint));
    G(glCreateProgram, GLuint (*)());
    G(glAttachShader, void (*)(GLuint, GLuint));
    G(glDeleteShader, void (*)(GLuint));
    G(glDeleteProgram, void (*)(GLuint));
    G(glBindAttribLocation, void (*)(GLuint, GLuint, const char*));
    G(glLinkProgram, void (*)(GLuint));
    G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
    G(glGetProgramInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
    G(glProgramParameteri, void (*)(GLuint, GLenum, GLint));
    G(glGetProgramBinary, void (*)(GLuint, GLsizei, GLsizei*, GLenum*, void*));
    G(glProgramBinary, void (*)(GLuint, GLenum, const void*, GLsizei));
    G(glGetIntegerv, void (*)(GLenum, GLint*));
    G(glGetUniformLocation, GLint (*)(GLuint, const char*));
    G(glGetAttribLocation, GLint (*)(GLuint, const char*));
    G(glUseProgram, void (*)(GLuint));
    G(glUniform1i, void (*)(GLint, GLint));
    G(glGetUniformiv, void (*)(GLuint, GLint, GLint*));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexImage2D, void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));

    GLint nfmt = -1;
    GLint fmts[4] = {};
    glGetIntegerv(0x87FE /*NUM_PROGRAM_BINARY_FORMATS*/, &nfmt);
    glGetIntegerv(0x87FF /*PROGRAM_BINARY_FORMATS*/, fmts);
    CHECK(nfmt == 1 && fmts[0] != 0 && err() == gl::NO_ERROR);
    const GLenum fmt = GLenum(fmts[0]);

    const char* src[2] = {kVs100, kFs100};
    GLuint sh[2];
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; ++i) {
        sh[i] = glCreateShader(i ? gl::FRAGMENT_SHADER : gl::VERTEX_SHADER);
        glShaderSource(sh[i], 1, &src[i], nullptr);
        glCompileShader(sh[i]);
        glAttachShader(p, sh[i]);
    }
    glBindAttribLocation(p, 5, "a_uv");
    glBindAttribLocation(p, 2, "a_pos");
    glProgramParameteri(p, 0x8257 /*PROGRAM_BINARY_RETRIEVABLE_HINT*/, 1);
    glLinkProgram(p);
    GLint v = -1, len = 0;
    glGetProgramiv(p, 0x8257, &v);
    CHECK(v == 1);
    glGetProgramiv(p, gl::LINK_STATUS, &v);
    CHECK(v == 1);
    glGetProgramiv(p, 0x8741 /*PROGRAM_BINARY_LENGTH*/, &len);
    CHECK(len > 0 && err() == gl::NO_ERROR);

    // A program that failed to link has no binary.
    {
        GLuint bad = glCreateProgram();
        glLinkProgram(bad);
        GLint l = -1;
        glGetProgramiv(bad, 0x8741, &l);
        CHECK(l == 0 && err() == gl::NO_ERROR);
        // GetProgramBinary of a program that is not linked: INVALID_OPERATION, nothing written.
        GLsizei ul = -7;
        GLenum uf = 0;
        char ub[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        glGetProgramBinary(bad, sizeof ub, &ul, &uf, ub);
        CHECK(err() == gl::INVALID_OPERATION && ul == -7 && uf == 0 && ub[0] == 1);
        glDeleteProgram(bad);
    }

    std::vector<char> blob(size_t(len > 0 ? len : 1));
    GLsizei got = -1;
    GLenum gotFmt = 0;
    glGetProgramBinary(p, GLsizei(blob.size()) - 1, &got, &gotFmt, blob.data());
    CHECK(err() == gl::INVALID_OPERATION);  // buffer too small
    glGetProgramBinary(p, GLsizei(blob.size()), &got, &gotFmt, blob.data());
    CHECK(err() == gl::NO_ERROR && got == len && gotFmt == fmt);

    // Later deletion of the shaders and of the source program does not matter.
    const GLint mvp = glGetUniformLocation(p, "u_mvp"), tex = glGetUniformLocation(p, "u_tex");
    glUseProgram(p);
    glUniform1i(tex, 3);
    glUseProgram(0);
    glDeleteShader(sh[0]);
    glDeleteShader(sh[1]);
    glDeleteProgram(p);

    auto linkStatus = [&](GLuint q) {
        GLint s = -1;
        glGetProgramiv(q, gl::LINK_STATUS, &s);
        return s;
    };
    GLuint q = glCreateProgram();
    glProgramBinary(q, fmt, blob.data(), got);
    CHECK(err() == gl::NO_ERROR && linkStatus(q) == 1);
    CHECK(glGetUniformLocation(q, "u_mvp") == mvp && glGetUniformLocation(q, "u_tex") == tex);
    CHECK(glGetAttribLocation(q, "a_uv") == 5 && glGetAttribLocation(q, "a_pos") == 2);
    glUseProgram(q);
    GLint unit = -1;
    glGetUniformiv(q, tex, &unit);
    CHECK(unit == 0);  // uniforms are back at their defaults
    glUniform1i(tex, 1);
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(gl::TEXTURE_2D, t);
    const uint8_t white[4] = {255, 255, 255, 255};
    glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 1, 1, 0, gl::RGBA, gl::UNSIGNED_BYTE, white);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::NO_ERROR && counts.draws == 1);
    glUseProgram(0);
    GLint len2 = 0;
    glGetProgramiv(q, 0x8741, &len2);
    CHECK(len2 == len);  // a program loaded from a binary can be saved again

    // Bad format, corrupt header / payload, truncation, empty: link fails, the log says so, no GL error.
    auto expectFailed = [&](GLenum f, const std::vector<char>& data, int line) {
        glProgramBinary(q, f, data.data(), GLsizei(data.size()));
        char log[256] = {};
        GLsizei n = 0;
        glGetProgramInfoLog(q, sizeof log, &n, log);
        GLint l = 1;
        glGetProgramiv(q, 0x8741, &l);
        if (err() != gl::NO_ERROR || linkStatus(q) != 0 || n == 0 || l != 0) {
            std::printf("  FAIL line %d: bad binary not a clean link failure\n", line);
            ++failures;
        }
    };
    expectFailed(fmt ^ 1, blob, __LINE__);
    std::vector<char> c1 = blob;
    c1[c1.size() / 2] ^= 0x5a;
    expectFailed(fmt, c1, __LINE__);
    std::vector<char> c2 = blob;
    c2[0] ^= 0x7f;
    expectFailed(fmt, c2, __LINE__);
    std::vector<char> c3(blob.begin(), blob.begin() + long(blob.size() / 2));
    expectFailed(fmt, c3, __LINE__);
    std::vector<char> c4(blob.begin(), blob.begin() + 3);
    expectFailed(fmt, c4, __LINE__);
    // The failed load replaced the linked state; the original blob still loads over it.
    glUseProgram(q);
    CHECK(err() == gl::INVALID_OPERATION);
    glProgramBinary(q, fmt, blob.data(), got);
    CHECK(err() == gl::NO_ERROR && linkStatus(q) == 1);
    GLint sepQ = -1;
    glGetProgramiv(q, 0x8258 /*PROGRAM_SEPARABLE*/, &sepQ);
    CHECK(sepQ == 0);

    // The separable flag is part of the blob; loading restores it (and clears it again for a non-separable blob).
    {
        GLuint ps = glCreateProgram();
        GLuint s2[2];
        for (int i = 0; i < 2; ++i) {
            s2[i] = glCreateShader(i ? gl::FRAGMENT_SHADER : gl::VERTEX_SHADER);
            glShaderSource(s2[i], 1, &src[i], nullptr);
            glCompileShader(s2[i]);
            glAttachShader(ps, s2[i]);
        }
        glProgramParameteri(ps, 0x8258, 1);
        glLinkProgram(ps);
        CHECK(linkStatus(ps) == 1);
        GLint blen = 0;
        glGetProgramiv(ps, 0x8741, &blen);
        std::vector<char> sblob(size_t(blen > 0 ? blen : 1));
        GLsizei sgot = 0;
        GLenum sfmt = 0;
        glGetProgramBinary(ps, GLsizei(sblob.size()), &sgot, &sfmt, sblob.data());
        CHECK(err() == gl::NO_ERROR && sgot == blen && blen != len);  // the flag changes the blob
        // Shaders deleted after the link: the program keeps the sources the blob needs.
        glDeleteShader(s2[0]);
        glDeleteShader(s2[1]);
        std::vector<char> again(sblob.size());
        glGetProgramBinary(ps, GLsizei(again.size()), &sgot, &sfmt, again.data());
        CHECK(err() == gl::NO_ERROR && again == sblob);  // built on demand: same bytes every time
        GLuint pr = glCreateProgram();
        glProgramBinary(pr, sfmt, sblob.data(), sgot);
        GLint sep = -1;
        glGetProgramiv(pr, 0x8258, &sep);
        CHECK(err() == gl::NO_ERROR && linkStatus(pr) == 1 && sep == 1);
        glProgramBinary(pr, fmt, blob.data(), got);  // non-separable blob over a separable program
        glGetProgramiv(pr, 0x8258, &sep);
        CHECK(err() == gl::NO_ERROR && linkStatus(pr) == 1 && sep == 0);
        glDeleteProgram(pr);
        glDeleteProgram(ps);
    }

    // NULL buffer: the app's bug, but it must not crash; length and format are still reported.
    {
        GLsizei nl = -1;
        GLenum nf = 0;
        glGetProgramBinary(q, GLsizei(blob.size()), &nl, &nf, nullptr);
        CHECK(err() == gl::NO_ERROR && nl == got && nf == fmt);
    }
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
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::NO_ERROR && counts.draws == 2);
    CHECK(counts.lastFrontFace == VK_FRONT_FACE_CLOCKWISE);  // GL CCW, rows bottom-up
    CHECK(counts.pipelines == 1 && counts.passes == 1);  // pipeline and pass reused
    // INT_2_10_10_10_REV normalizado: VPVulkan no lo lee en el pipeline (sin formato de vertice nativo), se convierte a float32 en la CPU.
    {
        const auto saved = counts;
        const uint32_t packed[3] = {0x3FF, 0x1FF, 0x200};
        GLuint pb;
        glGenBuffers(1, &pb);
        glBindBuffer(gl::ARRAY_BUFFER, pb);
        glBufferData(gl::ARRAY_BUFFER, sizeof packed, packed, gl::STATIC_DRAW);
        glVertexAttribPointer(GLuint(pos), 4, gl::INT_2_10_10_10_REV, 1, 4, nullptr);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::NO_ERROR);
        bool f32 = false, snorm = false;
        for (VkFormat f : counts.lastAttrFormats) f32 |= f == VK_FORMAT_R32G32B32A32_SFLOAT, snorm |= f == VK_FORMAT_A2B10G10R10_SNORM_PACK32;
        CHECK(f32 && !snorm);
        counts = saved;
        glBindBuffer(gl::ARRAY_BUFFER, vb);
        glVertexAttribPointer(GLuint(pos), 4, gl::FLOAT, 0, 24, nullptr);
    }
    // Texture completeness uses the filter the unit really samples with: a sampler object bound to the unit overrides the
    // texture's own. A 4x4 texture with only level 0 defined (mutable) is incomplete with a mipmapped min filter (it samples
    // the black stand-in) and complete with LINEAR / NEAREST.
    {
        GLuint t4;
        glGenTextures(1, &t4);
        glBindTexture(gl::TEXTURE_2D, t4);
        const std::vector<uint8_t> px4(4 * 4 * 4, 255);
        glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 4, 4, 0, gl::RGBA, gl::UNSIGNED_BYTE, px4.data());
        G(glTexParameteri, void (*)(GLenum, GLenum, GLint));
        G(glGenSamplers, void (*)(GLsizei, GLuint*));
        G(glBindSampler, void (*)(GLuint, GLuint));
        G(glSamplerParameteri, void (*)(GLuint, GLenum, GLint));
        auto sampledView = [&] {
            counts.lastSampledView = VK_NULL_HANDLE;
            glDrawArrays(gl::TRIANGLES, 0, 3);
            return counts.lastSampledView;
        };
        glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST_MIPMAP_LINEAR);
        const VkImageView black = sampledView();  // incomplete
        glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR);
        const VkImageView own = sampledView();  // complete
        CHECK(black && own && black != own);
        glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST_MIPMAP_LINEAR);
        GLuint smp;
        glGenSamplers(1, &smp);
        glBindSampler(0, smp);
        glSamplerParameteri(smp, gl::TEXTURE_MIN_FILTER, gl::NEAREST);
        CHECK(sampledView() == own);  // the sampler's NEAREST makes the incomplete-by-texture-filter texture complete
        glSamplerParameteri(smp, gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR);
        glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR);
        CHECK(sampledView() == black);  // the sampler's mipmapped filter needs levels the texture lacks
        glBindSampler(0, 0);
        CHECK(sampledView() == own && err() == gl::NO_ERROR);
        // ES 2.0: the chain runs down to 1x1 (a 128x64 texture has 8 levels): without its last level a mipmapped texture is incomplete.
        {
            GLuint tn;
            glGenTextures(1, &tn);
            glBindTexture(gl::TEXTURE_2D, tn);
            glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST_MIPMAP_NEAREST);
            const std::vector<uint8_t> big(128 * 64 * 4, 255);
            for (int l = 0; l < 7; ++l)
                glTexImage2D(gl::TEXTURE_2D, l, gl::RGBA, std::max(1, 128 >> l), std::max(1, 64 >> l), 0, gl::RGBA, gl::UNSIGNED_BYTE, big.data());
            const VkImageView missing = sampledView();
            glTexImage2D(gl::TEXTURE_2D, 7, gl::RGBA, 1, 1, 0, gl::RGBA, gl::UNSIGNED_BYTE, big.data());
            const VkImageView full = sampledView();
            CHECK(missing == black && full != black && err() == gl::NO_ERROR);
        }
    }
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
    // An instance array of uniform blocks (uniform B { .. } blk[3]): each element is a block of its own with its binding.
    {
        static const char* kVsU =
            "#version 300 es\nlayout(std140) uniform Blk { highp vec4 v; } blk[3]; in vec4 a_pos;\n"
            "void main() { gl_Position = a_pos + blk[0].v + blk[2].v; }\n";
        static const char* kFsU = "#version 300 es\nprecision mediump float; out vec4 o;\nvoid main() { o = vec4(1.0); }\n";
        G(glGetUniformBlockIndex, GLuint (*)(GLuint, const char*));
        G(glUniformBlockBinding, void (*)(GLuint, GLuint, GLuint));
        G(glBindBufferBase, void (*)(GLenum, GLuint, GLuint));
        const GLuint pu = program(kVsU, kFsU);
        glUseProgram(pu);
        GLuint ub[3];
        glGenBuffers(3, ub);
        const GLuint idx0 = glGetUniformBlockIndex(pu, "Blk[0]"), idx2 = glGetUniformBlockIndex(pu, "Blk[2]");
        CHECK(idx0 != 0xFFFFFFFFu && idx2 != 0xFFFFFFFFu && idx0 != idx2);
        for (int i = 0; i < 3; ++i) {
            const float zero[4] = {0, 0, 0, 0};
            glBindBuffer(gl::UNIFORM_BUFFER, ub[i]);
            glBufferData(gl::UNIFORM_BUFFER, sizeof zero, zero, gl::STATIC_DRAW);
            glBindBufferBase(gl::UNIFORM_BUFFER, GLuint(i), ub[i]);
            glUniformBlockBinding(pu, glGetUniformBlockIndex(pu, i == 0 ? "Blk[0]" : i == 1 ? "Blk[1]" : "Blk[2]"), GLuint(i));
        }
        // Every element of the block array lists the same member indices (the members exist once, as active uniforms).
        {
            G(glGetActiveUniformBlockiv, void (*)(GLuint, GLuint, GLenum, GLint*));
            G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
            GLint nUniforms = 0, i0 = -1, i1 = -1, n1 = -1;
            glGetProgramiv(pu, gl::ACTIVE_UNIFORMS, &nUniforms);
            glGetActiveUniformBlockiv(pu, idx0, gl::UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, &i0);
            glGetActiveUniformBlockiv(pu, idx2, gl::UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, &i1);
            glGetActiveUniformBlockiv(pu, idx2, gl::UNIFORM_BLOCK_ACTIVE_UNIFORMS, &n1);
            CHECK(nUniforms == 1 && n1 == 1 && i0 == i1 && i1 >= 0 && i1 < nUniforms && err() == gl::NO_ERROR);
        }
        glBindBuffer(gl::ARRAY_BUFFER, vb);
        glVertexAttribPointer(0, 4, gl::FLOAT, 0, 24, nullptr);
        glEnableVertexAttribArray(0);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::NO_ERROR);
        glUseProgram(p);
    }
    glUseProgram(0);
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::INVALID_OPERATION);
    glFinish();
    CHECK(eglSwapBuffers(e.dpy, e.surface));
    e.down();
}

// glInvalidateFramebuffer / glDiscardFramebufferEXT / glClear decide the load and store ops of the passes (what keeps
// depth and colour in tile memory on a tile GPU). Each case ends its pass with settle(); the load ops are read at the begin, the store ops after VPVulkan's drop hook.
void testInvalidate() {
    Egl e;
    CHECK(e.up());
    GLuint p = program(kVs100, kFs100);
    G(glUseProgram, void (*)(GLuint));
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    G(glVertexAttribPointer, void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
    G(glEnableVertexAttribArray, void (*)(GLuint));
    G(glGetAttribLocation, GLint (*)(GLuint, const char*));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glGenFramebuffers, void (*)(GLsizei, GLuint*));
    G(glBindFramebuffer, void (*)(GLenum, GLuint));
    G(glFramebufferTexture2D, void (*)(GLenum, GLenum, GLenum, GLuint, GLint));
    G(glFramebufferRenderbuffer, void (*)(GLenum, GLenum, GLenum, GLuint));
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexStorage2D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
    G(glGenRenderbuffers, void (*)(GLsizei, GLuint*));
    G(glBindRenderbuffer, void (*)(GLenum, GLuint));
    G(glRenderbufferStorage, void (*)(GLenum, GLenum, GLsizei, GLsizei));
    G(glClear, void (*)(GLbitfield));
    G(glEnable, void (*)(GLenum));
    G(glDisable, void (*)(GLenum));
    G(glScissor, void (*)(GLint, GLint, GLsizei, GLsizei));
    G(glInvalidateFramebuffer, void (*)(GLenum, GLsizei, const GLenum*));
    G(glInvalidateSubFramebuffer, void (*)(GLenum, GLsizei, const GLenum*, GLint, GLint, GLsizei, GLsizei));
    G(glDiscardFramebufferEXT, void (*)(GLenum, GLsizei, const GLenum*));
    glUseProgram(p);
    const float verts[] = {-1, -1, 0, 1, 0, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1};
    GLuint vb;
    glGenBuffers(1, &vb);
    glBindBuffer(gl::ARRAY_BUFFER, vb);
    glBufferData(gl::ARRAY_BUFFER, sizeof verts, verts, gl::STATIC_DRAW);
    const GLint pos = glGetAttribLocation(p, "a_pos");
    glVertexAttribPointer(GLuint(pos), 4, gl::FLOAT, 0, 24, nullptr);
    glEnableVertexAttribArray(GLuint(pos));
    GLuint fb, tex, rb;
    glGenFramebuffers(1, &fb);
    glGenTextures(1, &tex);
    glGenRenderbuffers(1, &rb);
    glBindTexture(gl::TEXTURE_2D, tex);
    glTexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8, 32, 32);
    glBindRenderbuffer(gl::RENDERBUFFER, rb);
    glRenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH_COMPONENT24, 32, 32);
    glBindFramebuffer(gl::FRAMEBUFFER, fb);
    glFramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, tex, 0);
    glFramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, rb);
    const GLenum depthOnly[1] = {gl::DEPTH_ATTACHMENT};
    const GLenum colorDepth[2] = {gl::COLOR_ATTACHMENT0, gl::DEPTH_ATTACHMENT};
    const VkAttachmentLoadOp LOAD = VK_ATTACHMENT_LOAD_OP_LOAD, CLEAR = VK_ATTACHMENT_LOAD_OP_CLEAR, DONT = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    const VkAttachmentStoreOp STORE = VK_ATTACHMENT_STORE_OP_STORE, DROP = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    glDrawArrays(gl::TRIANGLES, 0, 3);  // first use of the images
    settle();

    // (d) No invalidate: LOAD / STORE.
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.colorLoad == LOAD && counts.depthLoad == LOAD && counts.colorStore == STORE && counts.depthStore == STORE);

    // (a) Draw, then invalidate the depth with the pass still open: the pass ends with depth DONT_CARE, colour STORE.
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    glInvalidateFramebuffer(gl::FRAMEBUFFER, 1, depthOnly);
    settle();
    CHECK(err() == gl::NO_ERROR);
    CHECK(counts.colorLoad == LOAD && counts.depthLoad == LOAD && counts.colorStore == STORE && counts.depthStore == DROP);

    // (c) Invalidated, then drawn without a clear: depth loads DONT_CARE (colour is untouched), and the write defines it
    // again, so the pass after that loads it.
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.colorLoad == LOAD && counts.depthLoad == DONT && counts.depthStore == STORE);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.depthLoad == LOAD);

    // The same when the pass was already closed (flushed) before the invalidate.
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    glDiscardFramebufferEXT(gl::FRAMEBUFFER, 1, depthOnly);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.depthLoad == DONT && counts.depthStore == STORE && counts.colorLoad == LOAD);

    // A draw after the invalidate in the same pass defines the depth again: it is stored.
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    glInvalidateFramebuffer(gl::FRAMEBUFFER, 1, depthOnly);
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.depthStore == STORE);

    // A sub-rectangle that does not cover the framebuffer invalidates nothing; one that does, does.
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    glInvalidateSubFramebuffer(gl::FRAMEBUFFER, 1, depthOnly, 0, 0, 16, 16);
    settle();
    CHECK(counts.depthStore == STORE);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    glInvalidateSubFramebuffer(gl::FRAMEBUFFER, 1, depthOnly, 0, 0, 32, 32);
    settle();
    CHECK(counts.depthStore == DROP);

    // (b) A full glClear first in the pass is the CLEAR load op, not a clear command; a scissored one stays a command.
    counts = {};
    glClear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.colorLoad == CLEAR && counts.depthLoad == CLEAR && counts.clears == 0);
    glEnable(gl::SCISSOR_TEST);
    glScissor(0, 0, 8, 8);
    counts = {};
    glClear(gl::COLOR_BUFFER_BIT);
    settle();
    CHECK(counts.colorLoad == LOAD && counts.clears == 1);
    glScissor(0, 0, 32, 32);  // covers the whole target
    counts = {};
    glClear(gl::COLOR_BUFFER_BIT);
    settle();
    CHECK(counts.colorLoad == CLEAR && counts.clears == 0);
    glDisable(gl::SCISSOR_TEST);
    // A clear after a draw is a command (the draw's result must not be dropped).
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    glClear(gl::COLOR_BUFFER_BIT);
    settle();
    CHECK(counts.colorLoad == LOAD && counts.clears == 1);
    // A clear defines an invalidated attachment: no DONT_CARE load is left behind.
    glInvalidateFramebuffer(gl::FRAMEBUFFER, 1, depthOnly);
    glClear(gl::DEPTH_BUFFER_BIT);
    settle();
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.depthLoad == LOAD);

    // (f) A D24S8 target: a first clear of only one aspect (COLOR|DEPTH) stays a clear command (a CLEAR load op would have
    // to LOAD the stencil); clearing both is the CLEAR load op.
    {
        GLuint fb2, rb2;
        glGenFramebuffers(1, &fb2);
        glGenRenderbuffers(1, &rb2);
        glBindRenderbuffer(gl::RENDERBUFFER, rb2);
        glRenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH24_STENCIL8, 32, 32);
        glBindFramebuffer(gl::FRAMEBUFFER, fb2);
        glFramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, tex, 0);
        glFramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_STENCIL_ATTACHMENT, gl::RENDERBUFFER, rb2);
        glDrawArrays(gl::TRIANGLES, 0, 3);  // first use of the images
        settle();
        counts = {};
        glClear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        settle();
        CHECK(counts.colorLoad == CLEAR && counts.depthLoad == LOAD && counts.clears == 1);
        counts = {};
        glClear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        settle();
        CHECK(counts.colorLoad == CLEAR && counts.depthLoad == CLEAR && counts.clears == 0);
        glBindFramebuffer(gl::FRAMEBUFFER, fb);
    }

    // (g) Writing one view of an image defines the overlapping views too: colour invalidated through a layer-0 view, then
    // written through a 2-layer multiview view, is no longer DONT_CARE when the layer-0 view is used again.
    {
        GLuint pm = program(
            "#version 300 es\n#extension GL_OVR_multiview2 : require\nlayout(num_views = 2) in;\n"
            "in vec4 a_pos; uniform mat4 u_vp[2];\n"
            "void main() { gl_Position = u_vp[gl_ViewID_OVR] * a_pos; }\n",
            "#version 300 es\nprecision mediump float; out vec4 o; void main() { o = vec4(1.0); }\n");
        G(glTexStorage3D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLsizei));
        G(glFramebufferTextureLayer, void (*)(GLenum, GLenum, GLuint, GLint, GLint));
        G(glFramebufferTextureMultiviewOVR, void (*)(GLenum, GLenum, GLuint, GLint, GLint, GLsizei));
        GLuint arr, fbL, fbM;
        glGenTextures(1, &arr);
        glBindTexture(gl::TEXTURE_2D_ARRAY, arr);
        glTexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::RGBA8, 32, 32, 2);
        glGenFramebuffers(1, &fbL);
        glBindFramebuffer(gl::FRAMEBUFFER, fbL);
        glFramebufferTextureLayer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, arr, 0, 0);
        glGenFramebuffers(1, &fbM);
        glBindFramebuffer(gl::FRAMEBUFFER, fbM);
        glFramebufferTextureMultiviewOVR(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, arr, 0, 0, 2);
        glBindFramebuffer(gl::FRAMEBUFFER, fbL);
        glUseProgram(p);
        glDrawArrays(gl::TRIANGLES, 0, 3);  // first use of the layer-0 view
        settle();
        const GLenum c0[1] = {gl::COLOR_ATTACHMENT0};
        glInvalidateFramebuffer(gl::FRAMEBUFFER, 1, c0);
        glBindFramebuffer(gl::FRAMEBUFFER, fbM);
        glUseProgram(pm);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        settle();
        glBindFramebuffer(gl::FRAMEBUFFER, fbL);
        glUseProgram(p);
        counts = {};
        glDrawArrays(gl::TRIANGLES, 0, 3);
        settle();
        CHECK(err() == gl::NO_ERROR && counts.colorLoad == LOAD);
        glBindFramebuffer(gl::FRAMEBUFFER, fb);
    }

    // (e) The default framebuffer: its colour is presented, so invalidating it is ignored; its depth may be dropped.
    glBindFramebuffer(gl::FRAMEBUFFER, 0);
    const GLenum backAtts[2] = {gl::COLOR, gl::DEPTH};
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    glInvalidateFramebuffer(gl::FRAMEBUFFER, 2, backAtts);
    settle();
    CHECK(err() == gl::NO_ERROR);
    CHECK(counts.colorStore == STORE && counts.depthStore == DROP);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    settle();
    CHECK(counts.colorLoad == LOAD && counts.depthLoad == DONT);
    // Bad enums.
    const GLenum bad[1] = {gl::DEPTH_ATTACHMENT};
    glInvalidateFramebuffer(gl::FRAMEBUFFER, 1, bad);  // not a default framebuffer attachment
    CHECK(err() == gl::INVALID_ENUM);
    glInvalidateFramebuffer(0x1234, 1, bad);
    CHECK(err() == gl::INVALID_ENUM);
    glBindFramebuffer(gl::FRAMEBUFFER, fb);
    glInvalidateFramebuffer(gl::FRAMEBUFFER, 2, colorDepth);
    CHECK(err() == gl::NO_ERROR);
    e.down();
}

// GL_OES_EGL_image_external as Unity uses it for video: the texture is created with glGenTextures +
// glBindTexture(GL_TEXTURE_EXTERNAL_OES) + glTexParameteri before the SurfaceTexture exists.
void testExternal() {
    Egl e;
    CHECK(e.up());
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexParameteri, void (*)(GLenum, GLenum, GLint));
    G(glGetTexParameteriv, void (*)(GLenum, GLenum, GLint*));
    G(glTexImage2D, void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));
    G(glTexSubImage2D, void (*)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*));
    G(glGetIntegerv, void (*)(GLenum, GLint*));
    G(glGetString, const uint8_t* (*)(GLenum));
    G(glActiveTexture, void (*)(GLenum));
    G(glDeleteTextures, void (*)(GLsizei, const GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBufferBase, void (*)(GLenum, GLuint, GLuint));
    G(glUseProgram, void (*)(GLuint));
    G(glGetUniformLocation, GLint (*)(GLuint, const char*));
    G(glUniform1i, void (*)(GLint, GLint));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glGetAttribLocation, GLint (*)(GLuint, const char*));
    G(glVertexAttribPointer, void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
    G(glEnableVertexAttribArray, void (*)(GLuint));
    G(glFinish, void (*)());
    const char* ext = reinterpret_cast<const char*>(glGetString(gl::EXTENSIONS));
    CHECK(ext && std::strstr(ext, "GL_OES_EGL_image_external ") && std::strstr(ext, "GL_OES_EGL_image_external_essl3"));

    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(gl::TEXTURE_EXTERNAL_OES, t);
    CHECK(err() == gl::NO_ERROR);
    GLint v = -1;
    glGetIntegerv(gl::TEXTURE_BINDING_EXTERNAL_OES, &v);
    CHECK(err() == gl::NO_ERROR && GLuint(v) == t);
    glGetIntegerv(gl::TEXTURE_BINDING_2D, &v);
    CHECK(v == 0);
    glTexParameteri(gl::TEXTURE_EXTERNAL_OES, gl::TEXTURE_MIN_FILTER, gl::LINEAR);
    glTexParameteri(gl::TEXTURE_EXTERNAL_OES, gl::TEXTURE_MAG_FILTER, gl::LINEAR);
    glTexParameteri(gl::TEXTURE_EXTERNAL_OES, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE);
    glTexParameteri(gl::TEXTURE_EXTERNAL_OES, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE);
    CHECK(err() == gl::NO_ERROR);
    glGetTexParameteriv(gl::TEXTURE_EXTERNAL_OES, gl::TEXTURE_WRAP_S, &v);
    CHECK(err() == gl::NO_ERROR && v == GLint(gl::CLAMP_TO_EDGE));
    // Bound as another target: GL_INVALID_OPERATION; upload like a 2D texture.
    glBindTexture(gl::TEXTURE_2D, t);
    CHECK(err() == gl::INVALID_OPERATION);
    std::vector<uint8_t> px(8 * 8 * 4, 128);
    glTexImage2D(gl::TEXTURE_EXTERNAL_OES, 0, gl::RGBA, 8, 8, 0, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
    CHECK(err() == gl::NO_ERROR);
    glTexSubImage2D(gl::TEXTURE_EXTERNAL_OES, 0, 0, 0, 4, 4, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
    CHECK(err() == gl::NO_ERROR);
    // Other units and the default (name 0) external texture.
    glActiveTexture(gl::TEXTURE0 + 1);
    glBindTexture(gl::TEXTURE_EXTERNAL_OES, 0);
    glTexParameteri(gl::TEXTURE_EXTERNAL_OES, gl::TEXTURE_MIN_FILTER, gl::LINEAR);
    CHECK(err() == gl::NO_ERROR);
    glActiveTexture(gl::TEXTURE0);

    // Sampled from both GLSL versions.
    static const char* kVs =
        "attribute vec4 a_pos; varying vec2 v_uv; void main() { v_uv = a_pos.xy; gl_Position = a_pos; }";
    static const char* kFs100 =
        "#extension GL_OES_EGL_image_external : require\n"
        "precision mediump float; varying vec2 v_uv; uniform samplerExternalOES u_vid;\n"
        "void main() { gl_FragColor = texture2D(u_vid, v_uv); }";
    static const char* kFs300 =
        "#version 300 es\n#extension GL_OES_EGL_image_external_essl3 : require\n"
        "precision mediump float; in vec2 v_uv; uniform samplerExternalOES u_vid; out vec4 o;\n"
        "void main() { o = texture(u_vid, v_uv); }";
    static const char* kVs300 =
        "#version 300 es\nin vec4 a_pos; out vec2 v_uv; void main() { v_uv = a_pos.xy; gl_Position = a_pos; }";
    const float verts[] = {-1, -1, 0, 1, 1, -1, 0, 1, 0, 1, 0, 1};
    for (int es3 = 0; es3 < 2; ++es3) {
        GLuint p = program(es3 ? kVs300 : kVs, es3 ? kFs300 : kFs100);
        glUseProgram(p);
        glUniform1i(glGetUniformLocation(p, "u_vid"), 0);
        const GLint pos = glGetAttribLocation(p, "a_pos");
        glBindBuffer(gl::ARRAY_BUFFER, 0);
        glVertexAttribPointer(GLuint(pos), 4, gl::FLOAT, 0, 0, verts);
        glEnableVertexAttribArray(GLuint(pos));
        counts = {};
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::NO_ERROR && counts.draws == 1);
    }
    glFinish();

    // Buffer targets and bindings engines probe at start-up.
    const GLenum bufTargets[][2] = {{gl::SHADER_STORAGE_BUFFER, gl::SHADER_STORAGE_BUFFER_BINDING},
                                    {gl::ATOMIC_COUNTER_BUFFER, gl::ATOMIC_COUNTER_BUFFER_BINDING},
                                    {gl::DRAW_INDIRECT_BUFFER, gl::DRAW_INDIRECT_BUFFER_BINDING},
                                    {gl::DISPATCH_INDIRECT_BUFFER, gl::DISPATCH_INDIRECT_BUFFER_BINDING},
                                    {gl::TRANSFORM_FEEDBACK_BUFFER, gl::TRANSFORM_FEEDBACK_BUFFER_BINDING}};
    GLuint b;
    glGenBuffers(1, &b);
    for (const auto& bt : bufTargets) {
        glBindBuffer(bt[0], b);
        v = -1;
        glGetIntegerv(bt[1], &v);
        CHECK(err() == gl::NO_ERROR && GLuint(v) == b);
        glBindBuffer(bt[0], 0);
    }
    glBindBuffer(gl::TEXTURE_BUFFER, b);
    CHECK(err() == gl::NO_ERROR);
    glBindBufferBase(gl::SHADER_STORAGE_BUFFER, 0, b);
    CHECK(err() == gl::NO_ERROR);
    // Texture targets that only bind / take parameters.
    GLuint t2[3];
    glGenTextures(3, t2);
    const GLenum texTargets[][2] = {{gl::TEXTURE_CUBE_MAP_ARRAY, gl::TEXTURE_BINDING_CUBE_MAP_ARRAY},
                                    {gl::TEXTURE_BUFFER, gl::TEXTURE_BINDING_BUFFER},
                                    {gl::TEXTURE_2D_MULTISAMPLE_ARRAY, gl::TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY}};
    for (int i = 0; i < 3; ++i) {
        glBindTexture(texTargets[i][0], t2[i]);
        glTexParameteri(texTargets[i][0], gl::TEXTURE_MIN_FILTER, gl::NEAREST);
        v = -1;
        glGetIntegerv(texTargets[i][1], &v);
        CHECK(err() == gl::NO_ERROR && GLuint(v) == t2[i]);
        glBindTexture(texTargets[i][0], 0);
    }
    glBindTexture(gl::TEXTURE_BUFFER, t2[0]);  // already a cube array texture
    CHECK(err() == gl::INVALID_OPERATION);
    glDeleteTextures(1, &t);
    glGetIntegerv(gl::TEXTURE_BINDING_EXTERNAL_OES, &v);
    CHECK(v == 0);
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
    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = VK_FORMAT_R8G8B8A8_SRGB, ci.extent = {256, 256, 1};
    ci.mipLevels = 1, ci.arrayLayers = 2, ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage img;
    CHECK(createImage(static_cast<VkDevice>(dev), &ci, nullptr, &img) == VK_SUCCESS);
    VkMemoryRequirements req;
    getReq(static_cast<VkDevice>(dev), img, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    VkDeviceMemory mem;
    alloc(static_cast<VkDevice>(dev), &mai, nullptr, &mem);
    bind(static_cast<VkDevice>(dev), img, mem, 0);
    const GLuint tex = vpgles_wrap_image(img, VK_FORMAT_R8G8B8A8_SRGB, gl::TEXTURE_2D_ARRAY, 256, 256, 2, 1);
    CHECK(tex != 0);
    // A wrapped RGBA image keeps its alpha (passthrough eye): not the RGB8/SRGB8 row that shares its Vulkan format.
    CHECK(formatByVk(VK_FORMAT_R8G8B8A8_SRGB)->alphaBits == 8 && formatByVk(VK_FORMAT_R8G8B8A8_UNORM)->alphaBits == 8);
    // The window's D24S8 keeps its stencil (a DEPTH_COMPONENT24 row for the same VkFormat has none).
    CHECK(formatByVk(VK_FORMAT_D24_UNORM_S8_UINT)->stencil && formatByVk(VK_FORMAT_D24_UNORM_S8_UINT)->stencilBits == 8);
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


// ---- OpenGL ES 3.1 layer ---------------------------------------------------------------------------------
std::vector<std::string> g_dbg;  // debug callback captures: "source/type/id/severity/text"
void dbgCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* msg, const void* user) {
    char b[64];
    std::snprintf(b, sizeof b, "%X/%X/%u/%X/%d/%d:", source, type, id, severity, length, user == reinterpret_cast<const void*>(0x1234) ? 1 : 0);
    g_dbg.push_back(b + std::string(msg));
}

void testEs31() {
    Egl e;
    CHECK(e.up());
    G(glGetString, const GLubyte* (*)(GLenum));
    G(glGetStringi, const GLubyte* (*)(GLenum, GLuint));
    G(glGetIntegerv, void (*)(GLenum, GLint*));
    G(glGetIntegeri_v, void (*)(GLenum, GLuint, GLint*));
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glUseProgram, void (*)(GLuint));
    G(glEnable, void (*)(GLenum));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glDrawArraysInstanced, void (*)(GLenum, GLint, GLsizei, GLsizei));
    G(glGetAttribLocation, GLint (*)(GLuint, const char*));
    G(glGetUniformLocation, GLint (*)(GLuint, const char*));
    G(glEnableVertexAttribArray, void (*)(GLuint));

    // ---- version, extensions, program binary
    const std::string ver = reinterpret_cast<const char*>(glGetString(gl::VERSION));
    const std::string exts = reinterpret_cast<const char*>(glGetString(gl::EXTENSIONS));
    CHECK(ver.find("OpenGL ES " + std::to_string(gl::kAdvertisedMajor) + "." + std::to_string(gl::kAdvertisedMinor)) != std::string::npos);
    GLint major = 0, minor = 0;
    glGetIntegerv(gl::MAJOR_VERSION, &major);
    glGetIntegerv(gl::MINOR_VERSION, &minor);
    CHECK(major == gl::kAdvertisedMajor && minor == gl::kAdvertisedMinor);
    CHECK(exts.find("GL_OES_get_program_binary") != std::string::npos);
    GLint nfmt = -1;
    glGetIntegerv(0x87FE /*NUM_PROGRAM_BINARY_FORMATS*/, &nfmt);
    CHECK(nfmt == 1 && err() == gl::NO_ERROR);
    // Every extension in the list resolves through glGetStringi too (same list, same count).
    GLint nexts = 0;
    glGetIntegerv(gl::NUM_EXTENSIONS, &nexts);
    CHECK(nexts > 0 && glGetStringi(gl::EXTENSIONS, 0) != nullptr && glGetStringi(gl::EXTENSIONS, GLuint(nexts)) == nullptr);
    CHECK(err() == gl::INVALID_VALUE);
    {
        G(glGetProgramBinary, void (*)(GLuint, GLsizei, GLsizei*, GLenum*, void*));
        G(glProgramBinary, void (*)(GLuint, GLenum, const void*, GLsizei));
        GLuint p = program(kVs100, kFs100);
        GLsizei len = 99;
        GLenum fmt = 0;
        char buf[16];
        glGetProgramBinary(p, sizeof buf, &len, &fmt, buf);  // too small for a blob
        CHECK(err() == gl::INVALID_OPERATION);
        glProgramBinary(p, 0, buf, 4);  // wrong format: a failed link, not an error
        CHECK(err() == gl::NO_ERROR);
    }

    // ---- queryState: every ES 3.0 / 3.1 / 3.2 limit answers without INVALID_ENUM
    static const GLenum kPnames[] = {
        0x82DA, 0x82D9, 0x82E5, 0x90EB, 0x90DE, 0x90DC, 0x90DB, 0x90DD, 0x8C2B, 0x910E, 0x910F, 0x9110, 0x9111, 0x8E59, 0x8E5E, 0x8E5F,
        0x9315, 0x9316, 0x9317, 0x9318, 0x826C, 0x9144, 0x9143, 0x82E8, 0x8DF9, 0x92DC, 0x8F38, 0x826E, 0x8D6B, 0x84FD,
        0x919F, 0x8073, 0x88FF, 0x8DE0, 0x8E7E, 0x8E7D, 0x91BB, 0x91BC, 0x8B4A, 0x8B4D, 0x8A2B, 0x8A2D, 0x8A2E,
        0x8A2F, 0x8A30, 0x8A31, 0x8A34, 0x8CDF, 0x8824, 0x8C8A, 0x8C8B, 0x8C80, 0x8D57, 0x9122, 0x9125, 0x90D6, 0x90D7,
        0x8F39, 0x8E1F, 0x8D69, 0x92D1, 0x92D2, 0x92D3, 0x92D4, 0x92CF, 0x92D0, 0x92D5, 0x92D6, 0x92D7, 0x92D8};
    int bad = 0;
    for (GLenum pn : kPnames) {
        GLint v[16] = {};
        glGetIntegerv(pn, v);
        GLenum er = err();
        if (er == gl::INVALID_ENUM) {
            std::printf("  pname 0x%X gave INVALID_ENUM\n", pn);
            ++bad;
        }
    }
    CHECK(bad == 0);
    GLint v4[4] = {};
    glGetIntegerv(0x7777, v4);
    CHECK(err() == gl::INVALID_ENUM);
    GLint dims[3] = {};
    for (GLuint i = 0; i < 3; ++i) glGetIntegeri_v(0x91BE /*MAX_COMPUTE_WORK_GROUP_COUNT*/, i, &dims[i]);
    CHECK(dims[0] >= 65535 && err() == gl::NO_ERROR);
    for (GLuint i = 0; i < 3; ++i) glGetIntegeri_v(0x91BF /*MAX_COMPUTE_WORK_GROUP_SIZE*/, i, &dims[i]);
    CHECK(dims[0] >= 128 && dims[2] >= 64 && err() == gl::NO_ERROR);
    glGetIntegeri_v(0x91BE, 3, dims);
    CHECK(err() == gl::INVALID_VALUE);

    // ---- vertex attrib binding
    G(glBindVertexBuffer, void (*)(GLuint, GLuint, GLintptr, GLsizei));
    G(glVertexAttribFormat, void (*)(GLuint, GLint, GLenum, GLboolean, GLuint));
    G(glVertexAttribIFormat, void (*)(GLuint, GLint, GLenum, GLuint));
    G(glVertexAttribBinding, void (*)(GLuint, GLuint));
    G(glVertexBindingDivisor, void (*)(GLuint, GLuint));
    G(glGetVertexAttribiv, void (*)(GLuint, GLenum, GLint*));
    G(glVertexAttribPointer, void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
    G(glGenVertexArrays, void (*)(GLsizei, GLuint*));
    G(glBindVertexArray, void (*)(GLuint));
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint p = program(kVs100, kFs100);
    glUseProgram(p);
    const GLuint pos = GLuint(glGetAttribLocation(p, "a_pos")), uv = GLuint(glGetAttribLocation(p, "a_uv")),
                 col = GLuint(glGetAttribLocation(p, "a_col"));
    const float verts[] = {-1, -1, 0, 1, 0, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1};
    GLuint vb;
    glGenBuffers(1, &vb);
    glBindBuffer(gl::ARRAY_BUFFER, vb);
    glBufferData(gl::ARRAY_BUFFER, sizeof verts, verts, gl::STATIC_DRAW);
    glBindBuffer(gl::ARRAY_BUFFER, 0);
    glBindVertexBuffer(2, vb, 8, 24);  // UE4-style: all attributes share one binding index, unrelated to the location
    glVertexAttribFormat(pos, 4, gl::FLOAT, 0, 0);
    glVertexAttribFormat(uv, 2, gl::FLOAT, 0, 16);
    glVertexAttribIFormat(col, 4, gl::UNSIGNED_BYTE, 20);
    glVertexAttribBinding(pos, 2);
    glVertexAttribBinding(uv, 2);
    glVertexAttribBinding(col, 2);
    glEnableVertexAttribArray(pos);
    glEnableVertexAttribArray(uv);
    CHECK(err() == gl::NO_ERROR);
    GLint q = -1;
    glGetVertexAttribiv(uv, gl::VERTEX_ATTRIB_BINDING, &q);
    CHECK(q == 2);
    glGetVertexAttribiv(uv, gl::VERTEX_ATTRIB_RELATIVE_OFFSET, &q);
    CHECK(q == 16);
    glGetVertexAttribiv(col, gl::VERTEX_ATTRIB_ARRAY_INTEGER, &q);
    CHECK(q == 1);
    glGetIntegeri_v(gl::VERTEX_BINDING_STRIDE, 2, &q);
    CHECK(q == 24);
    glGetIntegeri_v(gl::VERTEX_BINDING_OFFSET, 2, &q);
    CHECK(q == 8);
    glGetIntegeri_v(gl::VERTEX_BINDING_BUFFER, 2, &q);
    CHECK(GLuint(q) == vb);
    glVertexBindingDivisor(2, 1);
    glGetIntegeri_v(gl::VERTEX_BINDING_DIVISOR, 2, &q);
    CHECK(q == 1 && err() == gl::NO_ERROR);
    glVertexBindingDivisor(2, 0);
    counts = {};
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::NO_ERROR && counts.draws == 1 && counts.pipelines == 1);
    CHECK(counts.lastAttrFormats.size() >= 2);
    bool vec4 = false, vec2 = false;
    for (VkFormat f : counts.lastAttrFormats) vec4 |= f == VK_FORMAT_R32G32B32A32_SFLOAT, vec2 |= f == VK_FORMAT_R32G32_SFLOAT;
    CHECK(vec4 && vec2);
    glVertexBindingDivisor(2, 1);
    counts = {};
    glDrawArraysInstanced(gl::TRIANGLES, 0, 3, 5);
    CHECK(err() == gl::NO_ERROR && counts.lastInstances == 5);
    // The legacy entry point keeps working on top of the same model and sets binding == index.
    glBindBuffer(gl::ARRAY_BUFFER, vb);
    glVertexAttribPointer(uv, 2, gl::FLOAT, 0, 24, reinterpret_cast<void*>(16));
    glGetVertexAttribiv(uv, gl::VERTEX_ATTRIB_BINDING, &q);
    CHECK(GLuint(q) == uv);
    glGetIntegeri_v(gl::VERTEX_BINDING_STRIDE, uv, &q);
    CHECK(q == 24);
    // Errors.
    glBindVertexBuffer(32, vb, 0, 4);
    CHECK(err() == gl::INVALID_VALUE);
    glBindVertexBuffer(0, vb, -1, 4);
    CHECK(err() == gl::INVALID_VALUE);
    glVertexAttribFormat(32, 4, gl::FLOAT, 0, 0);
    CHECK(err() == gl::INVALID_VALUE);
    glVertexAttribFormat(0, 5, gl::FLOAT, 0, 0);
    CHECK(err() == gl::INVALID_VALUE);
    glVertexAttribFormat(0, 4, 0x7777, 0, 0);
    CHECK(err() == gl::INVALID_ENUM);
    glVertexAttribFormat(0, 4, gl::FLOAT, 0, 5000);
    CHECK(err() == gl::INVALID_VALUE);
    glVertexAttribIFormat(0, 4, gl::FLOAT, 0);
    CHECK(err() == gl::INVALID_ENUM);
    glVertexAttribBinding(0, 99);
    CHECK(err() == gl::INVALID_VALUE);
    glVertexBindingDivisor(99, 1);
    CHECK(err() == gl::INVALID_VALUE);
    glBindBuffer(gl::ARRAY_BUFFER, 0);

    // ---- indirect draws
    G(glDrawArraysIndirect, void (*)(GLenum, const void*));
    G(glDrawElementsIndirect, void (*)(GLenum, GLenum, const void*));
    G(glDispatchComputeIndirect, void (*)(GLintptr));
    G(glDispatchCompute, void (*)(GLuint, GLuint, GLuint));
    glBindVertexArray(vao);
    glVertexAttribFormat(uv, 2, gl::FLOAT, 0, 16);
    glVertexAttribBinding(uv, 2);
    glVertexBindingDivisor(2, 0);
    const GLuint cmdA[4] = {3, 1, 0, 0};
    const GLuint cmdE[6] = {0, 0, 0, 0, 0, 0};
    GLuint cmdBuf[2];
    glGenBuffers(2, cmdBuf);
    glBindBuffer(gl::DRAW_INDIRECT_BUFFER, cmdBuf[0]);
    glBufferData(gl::DRAW_INDIRECT_BUFFER, sizeof cmdA, cmdA, gl::STATIC_DRAW);
    counts = {};
    glDrawArraysIndirect(gl::TRIANGLES, nullptr);
    CHECK(err() == gl::NO_ERROR && counts.drawsIndirect + counts.draws == 1);
    glDrawArraysIndirect(0x77, nullptr);
    CHECK(err() == gl::INVALID_ENUM);
    glDrawArraysIndirect(gl::TRIANGLES, reinterpret_cast<const void*>(2));
    CHECK(err() == gl::INVALID_VALUE);
    glDrawArraysIndirect(gl::TRIANGLES, reinterpret_cast<const void*>(16));
    CHECK(err() == gl::INVALID_OPERATION);  // the command does not fit in the buffer
    glBindBuffer(gl::DRAW_INDIRECT_BUFFER, 0);
    glDrawArraysIndirect(gl::TRIANGLES, nullptr);
    CHECK(err() == gl::INVALID_OPERATION);  // no DRAW_INDIRECT_BUFFER bound
    // Indexed.
    (void)cmdE;
    const GLuint cmdIdx[5] = {3, 1, 0, 0, 0};
    const uint16_t idx[] = {0, 1, 2};
    GLuint ib;
    glGenBuffers(1, &ib);
    glBindBuffer(gl::DRAW_INDIRECT_BUFFER, cmdBuf[1]);
    glBufferData(gl::DRAW_INDIRECT_BUFFER, sizeof cmdIdx, cmdIdx, gl::STATIC_DRAW);
    glBindBuffer(gl::ELEMENT_ARRAY_BUFFER, ib);
    glBufferData(gl::ELEMENT_ARRAY_BUFFER, sizeof idx, idx, gl::STATIC_DRAW);
    counts = {};
    glDrawElementsIndirect(gl::TRIANGLES, gl::UNSIGNED_SHORT, nullptr);
    CHECK(err() == gl::NO_ERROR && counts.drawsIndexedIndirect + counts.drawsIndexed == 1);
    glDrawElementsIndirect(gl::TRIANGLES, 0x7777, nullptr);
    CHECK(err() == gl::INVALID_ENUM);
    glBindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
    glDrawElementsIndirect(gl::TRIANGLES, gl::UNSIGNED_SHORT, nullptr);
    CHECK(err() == gl::INVALID_OPERATION);  // no index buffer
    glBindBuffer(gl::DRAW_INDIRECT_BUFFER, 0);
    // Indirect dispatch (needs the compute compiler; reported, not required, when it is missing).
    {
        G(glCreateShader, GLuint (*)(GLenum));
        G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
        G(glCompileShader, void (*)(GLuint));
        G(glGetShaderiv, void (*)(GLuint, GLenum, GLint*));
        G(glCreateProgram, GLuint (*)());
        G(glAttachShader, void (*)(GLuint, GLuint));
        G(glLinkProgram, void (*)(GLuint));
        G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
        const char* cs = "#version 310 es\nlayout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;\nvoid main() { }\n";
        GLuint sh = glCreateShader(gl::COMPUTE_SHADER);
        glShaderSource(sh, 1, &cs, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, gl::COMPILE_STATUS, &ok);
        GLuint cp = glCreateProgram();
        glAttachShader(cp, sh);
        glLinkProgram(cp);
        GLint linked = 0;
        glGetProgramiv(cp, gl::LINK_STATUS, &linked);
        CHECK(ok && linked);  // compute is part of the advertised level
        if (ok && linked) {
            const GLuint grid[3] = {2, 1, 1};
            GLuint db;
            glGenBuffers(1, &db);
            glUseProgram(cp);
            glBindBuffer(gl::DISPATCH_INDIRECT_BUFFER, db);
            glBufferData(gl::DISPATCH_INDIRECT_BUFFER, sizeof grid, grid, gl::STATIC_DRAW);
            counts = {};
            glDispatchComputeIndirect(0);
            CHECK(err() == gl::NO_ERROR && counts.dispatchesIndirect + counts.dispatches == 1);
            glDispatchComputeIndirect(2);
            CHECK(err() == gl::INVALID_VALUE);
            glDispatchComputeIndirect(8);
            CHECK(err() == gl::INVALID_OPERATION);  // 12 bytes do not fit
            glDispatchComputeIndirect(-4);
            CHECK(err() == gl::INVALID_VALUE);
            glBindBuffer(gl::DISPATCH_INDIRECT_BUFFER, 0);
            glDispatchComputeIndirect(0);
            CHECK(err() == gl::INVALID_OPERATION);
            counts = {};
            glDispatchCompute(4, 2, 1);
            CHECK(err() == gl::NO_ERROR && counts.dispatches == 1);
            glDispatchCompute(70000, 1, 1);
            CHECK(err() == gl::INVALID_VALUE);
            glUseProgram(p);
        }
    }

    // ---- CopyImageSubData
    G(glCopyImageSubData, void (*)(GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei));
    G(glTexStorage2D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
    G(glTexStorage3D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLsizei));
    G(glGenRenderbuffers, void (*)(GLsizei, GLuint*));
    G(glBindRenderbuffer, void (*)(GLenum, GLuint));
    G(glRenderbufferStorage, void (*)(GLenum, GLenum, GLsizei, GLsizei));
    GLuint tex[6];
    glGenTextures(6, tex);
    glBindTexture(gl::TEXTURE_2D, tex[0]);
    glTexStorage2D(gl::TEXTURE_2D, 2, gl::RGBA8, 8, 8);
    glBindTexture(gl::TEXTURE_2D, tex[1]);
    glTexStorage2D(gl::TEXTURE_2D, 2, gl::RGBA8, 8, 8);
    glBindTexture(gl::TEXTURE_2D, tex[2]);
    glTexStorage2D(gl::TEXTURE_2D, 1, gl::R8, 8, 8);
    glBindTexture(gl::TEXTURE_2D_ARRAY, tex[3]);
    glTexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::RGBA8, 8, 8, 4);
    CHECK(err() == gl::NO_ERROR);
    counts = {};
    glCopyImageSubData(tex[0], gl::TEXTURE_2D, 0, 0, 0, 0, tex[1], gl::TEXTURE_2D, 1, 0, 0, 0, 4, 4, 1);
    CHECK(err() == gl::NO_ERROR && counts.copiesImage == 1);
    glCopyImageSubData(tex[0], gl::TEXTURE_2D, 0, 0, 0, 0, tex[3], gl::TEXTURE_2D_ARRAY, 0, 0, 0, 2, 8, 8, 1);
    CHECK(err() == gl::NO_ERROR && counts.copiesImage == 2);
    glCopyImageSubData(tex[0], gl::TEXTURE_2D, 0, 0, 0, 0, tex[2], gl::TEXTURE_2D, 0, 0, 0, 0, 4, 4, 1);
    CHECK(err() == gl::INVALID_OPERATION);  // RGBA8 -> R8: incompatible
    glCopyImageSubData(tex[0], gl::TEXTURE_2D, 0, 4, 4, 0, tex[1], gl::TEXTURE_2D, 0, 0, 0, 0, 8, 8, 1);
    CHECK(err() == gl::INVALID_VALUE);  // source region out of range
    glCopyImageSubData(tex[0], gl::TEXTURE_2D, 5, 0, 0, 0, tex[1], gl::TEXTURE_2D, 0, 0, 0, 0, 1, 1, 1);
    CHECK(err() == gl::INVALID_VALUE);  // level out of range
    glCopyImageSubData(tex[0], 0x7777, 0, 0, 0, 0, tex[1], gl::TEXTURE_2D, 0, 0, 0, 0, 1, 1, 1);
    CHECK(err() == gl::INVALID_ENUM);
    glCopyImageSubData(999, gl::TEXTURE_2D, 0, 0, 0, 0, tex[1], gl::TEXTURE_2D, 0, 0, 0, 0, 1, 1, 1);
    CHECK(err() == gl::INVALID_VALUE);  // no such source object
    GLuint rb;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(gl::RENDERBUFFER, rb);
    glRenderbufferStorage(gl::RENDERBUFFER, gl::RGBA8, 8, 8);
    counts = {};
    glCopyImageSubData(rb, gl::RENDERBUFFER, 0, 0, 0, 0, tex[1], gl::TEXTURE_2D, 0, 0, 0, 0, 8, 8, 1);
    CHECK(err() == gl::NO_ERROR && counts.copiesImage == 1);

    // ---- error rules of the negative_api family
    G(glDrawRangeElements, void (*)(GLenum, GLuint, GLuint, GLsizei, GLenum, const void*));
    G(glGetInternalformativ, void (*)(GLenum, GLenum, GLenum, GLsizei, GLint*));
    glDrawRangeElements(gl::TRIANGLES, 5, 1, 3, gl::UNSIGNED_SHORT, nullptr);
    CHECK(err() == gl::INVALID_VALUE);  // end < start
    GLint nif[2] = {-1, -1};
    glGetInternalformativ(gl::TEXTURE_2D, gl::RGBA8, 0x80A9, 2, nif);
    CHECK(err() == gl::INVALID_ENUM);  // not a multisample target
    glGetInternalformativ(gl::RENDERBUFFER, gl::RGBA8, 0x1234, 2, nif);
    CHECK(err() == gl::INVALID_ENUM);  // pname
    glGetInternalformativ(gl::RENDERBUFFER, 0x1908 /* RGBA: unsized */, 0x80A9, 2, nif);
    CHECK(err() == gl::INVALID_ENUM);  // not a renderable sized format
    glVertexAttribPointer(0, 4, 0x7777, 0, 0, nullptr);
    CHECK(err() == gl::INVALID_ENUM);
    glVertexAttribPointer(0, 3, 0x8368 /* UNSIGNED_INT_2_10_10_10_REV */, 0, 0, nullptr);
    CHECK(err() == gl::INVALID_OPERATION);  // packed types need size 4

    // ---- glProgramUniform*
    G(glProgramUniform1i, void (*)(GLuint, GLint, GLint));
    G(glProgramUniform1f, void (*)(GLuint, GLint, GLfloat));
    G(glProgramUniform4f, void (*)(GLuint, GLint, GLfloat, GLfloat, GLfloat, GLfloat));
    G(glProgramUniform1iv, void (*)(GLuint, GLint, GLsizei, const GLint*));
    G(glProgramUniform2ui, void (*)(GLuint, GLint, GLuint, GLuint));
    G(glProgramUniformMatrix4fv, void (*)(GLuint, GLint, GLsizei, GLboolean, const GLfloat*));
    G(glProgramUniformMatrix2x3fv, void (*)(GLuint, GLint, GLsizei, GLboolean, const GLfloat*));
    G(glGetUniformfv, void (*)(GLuint, GLint, GLfloat*));
    G(glGetUniformiv, void (*)(GLuint, GLint, GLint*));
    G(glGetUniformuiv, void (*)(GLuint, GLint, GLuint*));
    G(glGetnUniformfv, void (*)(GLuint, GLint, GLsizei, GLfloat*));
    const char* vs300 = "#version 300 es\nin vec4 a_pos; uniform mat4 u_m; uniform vec4 u_v; uniform uvec2 u_u; uniform mat2x3 u_m23; out vec4 v;\n"
                        "void main() { v = u_v + vec4(u_u.xy, 0.0, 0.0) + vec4(u_m23[1], 0.0); gl_Position = u_m * a_pos; }\n";
    const char* fs300 = "#version 300 es\nprecision mediump float; in vec4 v; uniform sampler2D u_tex; out vec4 o;\n"
                        "void main() { o = v * texture(u_tex, v.xy); }\n";
    GLuint p3 = program(vs300, fs300);
    const GLint lm = glGetUniformLocation(p3, "u_m"), lv = glGetUniformLocation(p3, "u_v"), lu = glGetUniformLocation(p3, "u_u"),
                lt = glGetUniformLocation(p3, "u_tex"), l23 = glGetUniformLocation(p3, "u_m23");
    CHECK(lm >= 0 && lv >= 0 && lu >= 0 && lt >= 0 && l23 >= 0);
    glUseProgram(p);  // a different program stays current: ProgramUniform addresses p3 directly
    glProgramUniform1i(p3, lt, 5);
    glProgramUniform4f(p3, lv, 1, 2, 3, 4);
    glProgramUniform2ui(p3, lu, 7, 9);
    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = float(i + 1);
    glProgramUniformMatrix4fv(p3, lm, 1, 0, m);
    glProgramUniformMatrix2x3fv(p3, l23, 1, 0, m);
    CHECK(err() == gl::NO_ERROR);
    GLint iv = 0;
    glGetUniformiv(p3, lt, &iv);
    CHECK(iv == 5);
    float fv[16] = {};
    glGetUniformfv(p3, lv, fv);
    CHECK(fv[0] == 1 && fv[3] == 4);
    GLuint uv2[2] = {};
    glGetUniformuiv(p3, lu, uv2);
    CHECK(uv2[0] == 7 && uv2[1] == 9);
    glGetnUniformfv(p3, lm, sizeof fv, fv);
    CHECK(fv[0] == 1 && fv[15] == 16);
    glProgramUniform1f(p3, lt, 1.0f);
    CHECK(err() == gl::INVALID_OPERATION);  // sampler set as float
    glProgramUniform1i(9999, lt, 1);
    CHECK(err() == gl::INVALID_VALUE);
    glProgramUniform1i(p3, 9999, 1);
    CHECK(err() == gl::INVALID_OPERATION);
    const GLint two[2] = {1, 2};
    glProgramUniform1iv(p3, lt, 2, two);
    CHECK(err() == gl::INVALID_OPERATION);  // count > array size
    glProgramUniform1i(p3, -1, 3);  // location -1 is silently ignored
    CHECK(err() == gl::NO_ERROR);
    {
        G(glCreateProgram, GLuint (*)());
        GLuint unlinked = glCreateProgram();
        glProgramUniform1i(unlinked, 0, 1);
        CHECK(err() == gl::INVALID_OPERATION);
    }

    // ---- program interface / resource queries
    G(glGetProgramInterfaceiv, void (*)(GLuint, GLenum, GLenum, GLint*));
    G(glGetProgramResourceIndex, GLuint (*)(GLuint, GLenum, const char*));
    G(glGetProgramResourceName, void (*)(GLuint, GLenum, GLuint, GLsizei, GLsizei*, char*));
    G(glGetProgramResourceiv, void (*)(GLuint, GLenum, GLuint, GLsizei, const GLenum*, GLsizei, GLsizei*, GLint*));
    G(glGetProgramResourceLocation, GLint (*)(GLuint, GLenum, const char*));
    GLint n = -1, iv32 = 0;
    glGetProgramInterfaceiv(p, gl::UNIFORM_IF, gl::ACTIVE_RESOURCES, &n);
    CHECK(n == 2);
    glGetProgramInterfaceiv(p, gl::PROGRAM_INPUT, gl::ACTIVE_RESOURCES, &n);
    CHECK(n == 3);
    glGetProgramInterfaceiv(p, gl::PROGRAM_OUTPUT, gl::ACTIVE_RESOURCES, &n);
    CHECK(n == 0 || n == 1);  // gl_FragColor is not a user output
    glGetProgramInterfaceiv(p, gl::UNIFORM_IF, gl::MAX_NAME_LENGTH, &n);
    CHECK(n == 6);  // "u_mvp" + NUL
    glGetProgramInterfaceiv(p, gl::SHADER_STORAGE_BLOCK_IF, gl::ACTIVE_RESOURCES, &n);
    CHECK(n == 0 && err() == gl::NO_ERROR);
    glGetProgramInterfaceiv(p, 0x7777, gl::ACTIVE_RESOURCES, &n);
    CHECK(err() == gl::INVALID_ENUM);
    glGetProgramInterfaceiv(p, gl::UNIFORM_IF, 0x7777, &n);
    CHECK(err() == gl::INVALID_ENUM);
    const GLuint mvpIdx = glGetProgramResourceIndex(p, gl::UNIFORM_IF, "u_mvp");
    CHECK(mvpIdx != gl::INVALID_INDEX && glGetProgramResourceIndex(p, gl::UNIFORM_IF, "nope") == gl::INVALID_INDEX);
    char name[32];
    GLsizei nl = 0;
    glGetProgramResourceName(p, gl::UNIFORM_IF, mvpIdx, sizeof name, &nl, name);
    CHECK(nl == 5 && !std::strcmp(name, "u_mvp"));
    glGetProgramResourceName(p, gl::UNIFORM_IF, 99, sizeof name, &nl, name);
    CHECK(err() == gl::INVALID_VALUE);
    const GLenum props[] = {gl::TYPE, gl::ARRAY_SIZE, gl::NAME_LENGTH, gl::LOCATION, gl::BLOCK_INDEX, gl::OFFSET,
                            gl::REFERENCED_BY_VERTEX_SHADER, gl::REFERENCED_BY_FRAGMENT_SHADER};
    GLint out[8] = {};
    GLsizei got = 0;
    glGetProgramResourceiv(p, gl::UNIFORM_IF, mvpIdx, 8, props, 8, &got, out);
    CHECK(err() == gl::NO_ERROR && got == 8);
    CHECK(out[0] == 0x8B5C /*FLOAT_MAT4*/ && out[1] == 1 && out[2] == 6 && out[3] == glGetUniformLocation(p, "u_mvp") && out[4] == -1 &&
          out[5] == -1 && out[6] == 1 && out[7] == 0);
    glGetProgramResourceiv(p, gl::UNIFORM_IF, mvpIdx, 8, props, 3, &got, out);
    CHECK(got == 3);  // bufSize truncates
    const GLenum badProp[] = {0x7777};
    glGetProgramResourceiv(p, gl::UNIFORM_IF, mvpIdx, 1, badProp, 1, &got, out);
    CHECK(err() == gl::INVALID_ENUM);
    CHECK(glGetProgramResourceLocation(p, gl::UNIFORM_IF, "u_tex") == glGetUniformLocation(p, "u_tex"));
    CHECK(glGetProgramResourceLocation(p, gl::PROGRAM_INPUT, "a_col") == glGetAttribLocation(p, "a_col"));
    CHECK(glGetProgramResourceLocation(p, gl::UNIFORM_IF, "nope") == -1 && err() == gl::NO_ERROR);
    glGetProgramResourceLocation(p, gl::UNIFORM_BLOCK_IF, "x");
    CHECK(err() == gl::INVALID_ENUM);
    {  // Uniform block: members and block data.
        const char* vsb = "#version 300 es\nin vec4 a_pos; uniform Blk { vec4 a; mat4 m; } blk; uniform float loose;\n"
                          "void main() { gl_Position = blk.m * a_pos * blk.a.x * loose; }\n";
        const char* fsb = "#version 300 es\nprecision mediump float; out vec4 o; void main() { o = vec4(1.0); }\n";
        GLuint pb = program(vsb, fsb);
        glGetProgramInterfaceiv(pb, gl::UNIFORM_BLOCK_IF, gl::ACTIVE_RESOURCES, &n);
        CHECK(n == 1);
        glGetProgramInterfaceiv(pb, gl::UNIFORM_IF, gl::ACTIVE_RESOURCES, &n);
        CHECK(n == 3);  // loose + the two block members
        glGetProgramInterfaceiv(pb, gl::PROGRAM_OUTPUT, gl::ACTIVE_RESOURCES, &n);
        CHECK(n == 1);
        const GLuint bi = glGetProgramResourceIndex(pb, gl::UNIFORM_BLOCK_IF, "Blk");
        CHECK(bi == 0);
        const GLenum bp[] = {gl::BUFFER_DATA_SIZE, gl::NUM_ACTIVE_VARIABLES, gl::BUFFER_BINDING, gl::ACTIVE_VARIABLES};
        GLint bo[8] = {};
        glGetProgramResourceiv(pb, gl::UNIFORM_BLOCK_IF, bi, 4, bp, 8, &got, bo);
        CHECK(err() == gl::NO_ERROR && bo[0] >= 80 && bo[1] == 2 && bo[2] == 0 && got == 5);
        GLint inBlock = -2;
        const GLenum bidx[] = {gl::BLOCK_INDEX};
        glGetProgramResourceiv(pb, gl::UNIFORM_IF, GLuint(bo[3]), 1, bidx, 1, &got, &inBlock);
        CHECK(inBlock == 0);
    }

    // ---- compute: shader storage buffers, images, atomic counters, program resources (ES 3.1 compiler)
    {
        G(glCreateShader, GLuint (*)(GLenum));
        G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
        G(glCompileShader, void (*)(GLuint));
        G(glGetShaderiv, void (*)(GLuint, GLenum, GLint*));
        G(glGetShaderInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
        G(glCreateProgram, GLuint (*)());
        G(glAttachShader, void (*)(GLuint, GLuint));
        G(glLinkProgram, void (*)(GLuint));
        G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
        G(glGetProgramInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
        G(glBindBufferBase, void (*)(GLenum, GLuint, GLuint));
        G(glBindBufferRange, void (*)(GLenum, GLuint, GLuint, GLintptr, GLsizeiptr));
        G(glBindImageTexture, void (*)(GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum));
        G(glMemoryBarrier, void (*)(GLbitfield));
        G(glMemoryBarrierByRegion, void (*)(GLbitfield));
        G(glUniform1ui, void (*)(GLint, GLuint));
        const char* src =
            "#version 310 es\n"
            "layout(local_size_x = 8) in;\n"
            "layout(std430, binding = 3) buffer Data { uint count; vec4 items[]; } data;\n"
            "layout(rgba8, binding = 1) writeonly uniform highp image2D outImg;\n"
            "layout(binding = 2, offset = 4) uniform atomic_uint counter;\n"
            "uniform uint u_n;\n"
            "void main() {\n"
            "  uint i = gl_GlobalInvocationID.x;\n"
            "  if (i < u_n && i < uint(data.items.length())) { data.items[i] = vec4(float(i)); atomicCounterIncrement(counter); }\n"
            "  imageStore(outImg, ivec2(int(i), 0), vec4(1.0));\n"
            "}\n";
        GLuint sh = glCreateShader(gl::COMPUTE_SHADER);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, gl::COMPILE_STATUS, &ok);
        char log[1024] = {};
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        if (!ok) std::printf("  compile log: %s\n", log);
        CHECK(ok);
        GLuint cp = glCreateProgram();
        glAttachShader(cp, sh);
        glLinkProgram(cp);
        GLint linked = 0;
        glGetProgramiv(cp, gl::LINK_STATUS, &linked);
        glGetProgramInfoLog(cp, sizeof log, nullptr, log);
        if (!linked) std::printf("  link log: %s\n", log);
        CHECK(linked);
        if (linked) {
            // Reflection through the program interface queries.
            glGetProgramInterfaceiv(cp, gl::SHADER_STORAGE_BLOCK_IF, gl::ACTIVE_RESOURCES, &n);
            CHECK(n == 1);
            glGetProgramInterfaceiv(cp, gl::BUFFER_VARIABLE, gl::ACTIVE_RESOURCES, &n);
            CHECK(n == 2);
            glGetProgramInterfaceiv(cp, gl::ATOMIC_COUNTER_BUFFER_IF, gl::ACTIVE_RESOURCES, &n);
            CHECK(n == 1);
            glGetProgramiv(cp, 0x92D9, &n);
            CHECK(n == 1);
            glGetProgramInterfaceiv(cp, gl::UNIFORM_IF, gl::ACTIVE_RESOURCES, &n);
            CHECK(n == 3);  // u_n, outImg, counter
            glGetProgramInterfaceiv(cp, gl::PROGRAM_INPUT, gl::ACTIVE_RESOURCES, &n);
            CHECK(n == 0);
            const GLuint bi = glGetProgramResourceIndex(cp, gl::SHADER_STORAGE_BLOCK_IF, "Data");
            CHECK(bi == 0);
            const GLenum bprops[] = {gl::BUFFER_BINDING, gl::BUFFER_DATA_SIZE, gl::NUM_ACTIVE_VARIABLES, gl::REFERENCED_BY_COMPUTE_SHADER,
                                     gl::ACTIVE_VARIABLES};
            GLint bo[8] = {};
            glGetProgramResourceiv(cp, gl::SHADER_STORAGE_BLOCK_IF, bi, 5, bprops, 8, &got, bo);
            CHECK(err() == gl::NO_ERROR && got == 6);
            CHECK(bo[0] == 3 && bo[1] == 16 && bo[2] == 2 && bo[3] == 1);  // layout(binding = 3): the initial binding; size without the array
            const GLuint iv = glGetProgramResourceIndex(cp, gl::BUFFER_VARIABLE, "Data.items[0]");
            const GLuint iv2 = glGetProgramResourceIndex(cp, gl::BUFFER_VARIABLE, "items");
            CHECK(iv != gl::INVALID_INDEX || iv2 != gl::INVALID_INDEX);
            const GLuint vi = iv != gl::INVALID_INDEX ? iv : iv2;
            const GLenum vprops[] = {gl::TYPE, gl::ARRAY_SIZE, gl::OFFSET, gl::ARRAY_STRIDE, gl::BLOCK_INDEX, gl::TOP_LEVEL_ARRAY_STRIDE};
            GLint vo[8] = {};
            glGetProgramResourceiv(cp, gl::BUFFER_VARIABLE, vi, 6, vprops, 8, &got, vo);
            CHECK(err() == gl::NO_ERROR && got == 6 && vo[0] == 0x8B52 /*FLOAT_VEC4*/ && vo[1] == 0 && vo[2] == 16 && vo[3] == 16 &&
                  vo[4] == 0 && vo[5] == 0);  // unsized last array: size 0, offset after the counter, vec4 stride; no top-level array of aggregates
            const GLenum aprops[] = {gl::BUFFER_BINDING, gl::BUFFER_DATA_SIZE, gl::NUM_ACTIVE_VARIABLES};
            GLint ao[4] = {};
            glGetProgramResourceiv(cp, gl::ATOMIC_COUNTER_BUFFER_IF, 0, 3, aprops, 4, &got, ao);
            CHECK(ao[0] == 2 && ao[1] == 8 && ao[2] == 1);
            const GLuint ci = glGetProgramResourceIndex(cp, gl::UNIFORM_IF, "counter");
            const GLenum cprops[] = {gl::ATOMIC_COUNTER_BUFFER_INDEX, gl::OFFSET, gl::LOCATION};
            GLint co[4] = {};
            glGetProgramResourceiv(cp, gl::UNIFORM_IF, ci, 3, cprops, 4, &got, co);
            CHECK(co[0] == 0 && co[1] == 4 && co[2] == -1);
            glGetProgramInterfaceiv(cp, gl::SHADER_STORAGE_BLOCK_IF, gl::MAX_NUM_ACTIVE_VARIABLES, &n);
            CHECK(n == 2);

            // Initial units from layout(binding = N): the image uniform starts at unit 1.
            glGetUniformiv(cp, glGetUniformLocation(cp, "outImg"), &iv32);
            CHECK(iv32 == 1);
            {  // An atomic counter has no value in the default block: reads 0 instead of going out of range.
                G(glProgramUniform1i, void (*)(GLuint, GLint, GLint));
                const GLint al = glGetUniformLocation(cp, "counter");
                if (al >= 0) {
                    iv32 = 77;
                    glGetUniformiv(cp, al, &iv32);
                    CHECK(iv32 == 0 && err() == gl::NO_ERROR);
                }
                // An image unit past MAX_IMAGE_UNITS is an error, not a wrap-around.
                glProgramUniform1i(cp, glGetUniformLocation(cp, "outImg"), 8);
                CHECK(err() == gl::INVALID_VALUE);
                glGetUniformiv(cp, glGetUniformLocation(cp, "outImg"), &iv32);
                CHECK(iv32 == 1);
            }

            // Dispatch with everything bound.
            GLuint ssbo, acb, img;
            glGenBuffers(1, &ssbo);
            glGenBuffers(1, &acb);
            glGenTextures(1, &img);
            glBindBuffer(gl::SHADER_STORAGE_BUFFER, ssbo);
            glBufferData(gl::SHADER_STORAGE_BUFFER, 16 + 16 * 8, nullptr, gl::DYNAMIC_DRAW);
            glBindBuffer(gl::ATOMIC_COUNTER_BUFFER, acb);
            glBufferData(gl::ATOMIC_COUNTER_BUFFER, 8, nullptr, gl::DYNAMIC_DRAW);
            glBindTexture(gl::TEXTURE_2D, img);
            glTexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8, 8, 1);
            glBindBufferBase(gl::SHADER_STORAGE_BUFFER, 3, ssbo);
            glBindBufferBase(gl::ATOMIC_COUNTER_BUFFER, 2, acb);
            glBindImageTexture(1, img, 0, 0, 0, gl::WRITE_ONLY, gl::RGBA8);
            CHECK(err() == gl::NO_ERROR);
            glUseProgram(cp);
            glUniform1ui(glGetUniformLocation(cp, "u_n"), 8);
            counts = {};
            glDispatchCompute(1, 1, 1);
            {
                const GLenum de = err();
                if (de != gl::NO_ERROR || counts.dispatches != 1) std::printf("  dispatch: error 0x%X, %d dispatches\n", de, counts.dispatches);
                CHECK(de == gl::NO_ERROR && counts.dispatches == 1);
            }
            glMemoryBarrier(0x00000020 | 0x00002000 | 0x00001000);  // SHADER_IMAGE_ACCESS | SHADER_STORAGE | ATOMIC_COUNTER
            glMemoryBarrierByRegion(0x00000020);
            CHECK(err() == gl::NO_ERROR);
            // The same program with nothing bound reads zeros and does not fail.
            glBindBufferBase(gl::SHADER_STORAGE_BUFFER, 3, 0);
            glBindBufferBase(gl::ATOMIC_COUNTER_BUFFER, 2, 0);
            glBindImageTexture(1, 0, 0, 0, 0, gl::WRITE_ONLY, gl::RGBA8);
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.dispatches == 2);
            // Ranges: SSBO offsets must be aligned.
            glBindBufferRange(gl::SHADER_STORAGE_BUFFER, 3, ssbo, 4, 16);
            CHECK(err() == gl::INVALID_VALUE);
            glBindBufferRange(gl::SHADER_STORAGE_BUFFER, 3, ssbo, 0, 144);
            CHECK(err() == gl::NO_ERROR);
            glGetIntegeri_v(gl::SHADER_STORAGE_BUFFER_BINDING, 3, &q);
            CHECK(GLuint(q) == ssbo);
            glBindBufferBase(gl::ATOMIC_COUNTER_BUFFER, 8, acb);
            CHECK(err() == gl::INVALID_VALUE);
            glBindImageTexture(8, img, 0, 0, 0, gl::WRITE_ONLY, gl::RGBA8);
            CHECK(err() == gl::INVALID_VALUE);
            glUseProgram(p);
        }
    }

    // ---- image units: a slice of a 3D texture as image2D (Metal cannot view 3D as 2D: a 2D copy of the slice is
    // bound, copied in before the dispatch and back after it) and a format other than the texture's
    {
        G(glCreateShader, GLuint (*)(GLenum));
        G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
        G(glCompileShader, void (*)(GLuint));
        G(glGetShaderiv, void (*)(GLuint, GLenum, GLint*));
        G(glCreateProgram, GLuint (*)());
        G(glAttachShader, void (*)(GLuint, GLuint));
        G(glLinkProgram, void (*)(GLuint));
        G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
        G(glBindImageTexture, void (*)(GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum));
        G(glTexStorage3D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLsizei));
        G(glTexStorage2D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
        auto compute = [&](const char* src) {
            GLuint sh = glCreateShader(gl::COMPUTE_SHADER);
            glShaderSource(sh, 1, &src, nullptr);
            glCompileShader(sh);
            GLuint cp = glCreateProgram();
            glAttachShader(cp, sh);
            glLinkProgram(cp);
            GLint linked = 0;
            glGetProgramiv(cp, gl::LINK_STATUS, &linked);
            return linked ? cp : 0u;
        };
        const GLuint slice = compute(
            "#version 310 es\nlayout(local_size_x = 4) in;\n"
            "layout(r32ui, binding = 0) readonly uniform highp uimage2D src;\n"
            "layout(r32ui, binding = 1) writeonly uniform highp uimage2D dst;\n"
            "void main() { ivec2 p = ivec2(int(gl_GlobalInvocationID.x), 0); imageStore(dst, p, imageLoad(src, p)); }\n");
        CHECK(slice != 0);
        GLuint t3, t2;
        glGenTextures(1, &t3);
        glGenTextures(1, &t2);
        glBindTexture(gl::TEXTURE_3D, t3);
        glTexStorage3D(gl::TEXTURE_3D, 1, 0x8236 /*R32UI*/, 4, 4, 4);
        glBindTexture(gl::TEXTURE_2D, t2);
        glTexStorage2D(gl::TEXTURE_2D, 1, 0x8236, 4, 4);
        CHECK(err() == gl::NO_ERROR);
        if (slice) {
            glUseProgram(slice);
            glBindImageTexture(0, t3, 0, 0, 2, gl::READ_ONLY, 0x8236);
            glBindImageTexture(1, t2, 0, 0, 0, gl::WRITE_ONLY, 0x8236);
            CHECK(err() == gl::NO_ERROR);
            counts = {};  // (the binds made the textures storage-capable: images re-created, contents carried over)
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.dispatches == 1);
            CHECK(counts.copiesImage == 1);  // the slice in; read-only: nothing comes back
            CHECK(counts.viewsOf3DAs2D == 0);
            // Written: copied back after the dispatch.
            glBindImageTexture(0, t2, 0, 0, 0, gl::READ_ONLY, 0x8236);
            glBindImageTexture(1, t3, 0, 0, 3, gl::WRITE_ONLY, 0x8236);
            counts.copiesImage = 0;
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.copiesImage == 2 && counts.viewsOf3DAs2D == 0);
            // A layer past the depth: reads zero, no error, no 2D view of the 3D image.
            glBindImageTexture(1, t3, 0, 0, 9, gl::WRITE_ONLY, 0x8236);
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.viewsOf3DAs2D == 0);
        }
        // Format reinterpretation: an RGBA8 texture read as R32UI gets a view of that format.
        const GLuint reint = compute(
            "#version 310 es\nlayout(local_size_x = 4) in;\n"
            "layout(r32ui, binding = 0) readonly uniform highp uimage2D src;\n"
            "layout(r32ui, binding = 1) writeonly uniform highp uimage2D dst;\n"
            "void main() { ivec2 p = ivec2(int(gl_GlobalInvocationID.x), 0); imageStore(dst, p, imageLoad(src, p)); }\n");
        CHECK(reint != 0);
        GLuint ta, tb;
        glGenTextures(1, &ta);
        glGenTextures(1, &tb);
        glBindTexture(gl::TEXTURE_2D, ta);
        glTexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8, 4, 4);
        glBindTexture(gl::TEXTURE_2D, tb);
        glTexStorage2D(gl::TEXTURE_2D, 1, 0x8236, 4, 4);
        if (reint) {
            glUseProgram(reint);
            counts = {};
            glBindImageTexture(0, ta, 0, 0, 0, gl::READ_ONLY, 0x8236);
            CHECK(counts.lastImageFlags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
            glBindImageTexture(1, tb, 0, 0, 0, gl::WRITE_ONLY, 0x8236);
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.dispatches == 1);
            CHECK(counts.lastViewFormat == VK_FORMAT_R32_UINT);
        }
        glUseProgram(p);
    }

    // ---- texture buffers
    G(glTexBuffer, void (*)(GLenum, GLenum, GLuint));
    G(glTexBufferRange, void (*)(GLenum, GLenum, GLuint, GLintptr, GLsizeiptr));
    G(glGetTexLevelParameteriv, void (*)(GLenum, GLint, GLenum, GLint*));
    G(glGetTexLevelParameterfv, void (*)(GLenum, GLint, GLenum, GLfloat*));
    G(glDeleteBuffers, void (*)(GLsizei, const GLuint*));
    GLuint tb;
    glGenBuffers(1, &tb);
    glBindBuffer(gl::TEXTURE_BUFFER, tb);
    std::vector<uint8_t> blob(2048, 7);
    glBufferData(gl::TEXTURE_BUFFER, GLsizeiptr(blob.size()), blob.data(), gl::STATIC_DRAW);
    glBindTexture(gl::TEXTURE_BUFFER, tex[4]);
    glTexBuffer(gl::TEXTURE_BUFFER, gl::RGBA8, tb);
    CHECK(err() == gl::NO_ERROR);
    GLint w = -1;
    glGetTexLevelParameteriv(gl::TEXTURE_BUFFER, 0, gl::TEXTURE_WIDTH, &w);
    CHECK(w == 512);  // 2048 bytes / 4 per texel
    glGetTexLevelParameteriv(gl::TEXTURE_BUFFER, 0, gl::TEXTURE_INTERNAL_FORMAT, &w);
    CHECK(w == gl::RGBA8);
    glGetTexLevelParameteriv(gl::TEXTURE_BUFFER, 0, gl::TEXTURE_BUFFER_DATA_STORE_BINDING, &w);
    CHECK(GLuint(w) == tb);
    glTexBufferRange(gl::TEXTURE_BUFFER, gl::R32F, tb, 256, 512);
    CHECK(err() == gl::NO_ERROR);
    glGetTexLevelParameteriv(gl::TEXTURE_BUFFER, 0, gl::TEXTURE_WIDTH, &w);
    CHECK(w == 128);
    glGetTexLevelParameteriv(gl::TEXTURE_BUFFER, 0, gl::TEXTURE_BUFFER_OFFSET, &w);
    CHECK(w == 256);
    glGetTexLevelParameteriv(gl::TEXTURE_BUFFER, 0, gl::TEXTURE_BUFFER_SIZE, &w);
    CHECK(w == 512);
    glTexBufferRange(gl::TEXTURE_BUFFER, gl::R32F, tb, 4, 64);
    CHECK(err() == gl::INVALID_VALUE);  // offset not aligned to TEXTURE_BUFFER_OFFSET_ALIGNMENT
    glTexBufferRange(gl::TEXTURE_BUFFER, gl::R32F, tb, 0, 4096);
    CHECK(err() == gl::INVALID_VALUE);  // range beyond the store
    glTexBuffer(gl::TEXTURE_BUFFER, gl::RGB32F, tb);
    CHECK(err() == gl::INVALID_ENUM);  // not a texel-buffer format here
    glTexBuffer(gl::TEXTURE_2D, gl::RGBA8, tb);
    CHECK(err() == gl::INVALID_ENUM);
    glTexBuffer(gl::TEXTURE_BUFFER, gl::RGBA8, 4444);
    CHECK(err() == gl::INVALID_OPERATION);  // no such buffer
    glGetIntegerv(gl::MAX_TEXTURE_BUFFER_SIZE, &w);
    CHECK(w >= 65536);
    glDeleteBuffers(1, &tb);  // detaches it from the texture
    glGetTexLevelParameteriv(gl::TEXTURE_BUFFER, 0, gl::TEXTURE_WIDTH, &w);
    CHECK(w == 0 && err() == gl::NO_ERROR);

    {  // GL_EXT_texture_buffer end to end: samplerBuffer + imageBuffer in a compute shader, bound through glTexBuffer.
        CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::EXTENSIONS)), "GL_EXT_texture_buffer"));
        CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::EXTENSIONS)), "GL_EXT_copy_image"));
        CHECK(std::strstr(reinterpret_cast<const char*>(glGetString(gl::EXTENSIONS)), "GL_EXT_draw_buffers_indexed"));
        G(glCreateShader, GLuint (*)(GLenum));
        G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
        G(glCompileShader, void (*)(GLuint));
        G(glGetShaderiv, void (*)(GLuint, GLenum, GLint*));
        G(glGetShaderInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
        G(glCreateProgram, GLuint (*)());
        G(glAttachShader, void (*)(GLuint, GLuint));
        G(glLinkProgram, void (*)(GLuint));
        G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
        G(glGetProgramInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
        G(glGetUniformLocation, GLint (*)(GLuint, const char*));
        G(glGetActiveUniform, void (*)(GLuint, GLuint, GLsizei, GLsizei*, GLint*, GLenum*, char*));
        G(glUniform1i, void (*)(GLint, GLint));
        G(glActiveTexture, void (*)(GLenum));
        G(glBindImageTexture, void (*)(GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum));
        G(glDispatchCompute, void (*)(GLuint, GLuint, GLuint));
        G(glMemoryBarrier, void (*)(GLbitfield));
        G(glUseProgram, void (*)(GLuint));
        const char* src =
            "#version 310 es\n"
            "#extension GL_EXT_texture_buffer : require\n"
            "layout(local_size_x = 4) in;\n"
            "uniform highp samplerBuffer src;\n"
            "layout(rgba32f, binding = 0) writeonly uniform highp imageBuffer dst;\n"
            "void main() {\n"
            "  int i = int(gl_GlobalInvocationID.x);\n"
            "  imageStore(dst, i, texelFetch(src, i) * float(textureSize(src)));\n"
            "}\n";
        GLuint sh = glCreateShader(gl::COMPUTE_SHADER);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, gl::COMPILE_STATUS, &ok);
        char log[1024] = {};
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        if (!ok) std::printf("  compile log: %s\n", log);
        CHECK(ok);
        GLuint cp = glCreateProgram();
        glAttachShader(cp, sh);
        glLinkProgram(cp);
        GLint linked = 0;
        glGetProgramiv(cp, gl::LINK_STATUS, &linked);
        glGetProgramInfoLog(cp, sizeof log, nullptr, log);
        if (!linked) std::printf("  link log: %s\n", log);
        CHECK(linked);
        if (linked) {
            GLint nu = 0;
            glGetProgramiv(cp, gl::ACTIVE_UNIFORMS, &nu);
            bool sawSampler = false, sawImage = false;
            for (GLint k = 0; k < nu; ++k) {
                char name[64] = {};
                GLint size = 0;
                GLenum type = 0;
                glGetActiveUniform(cp, GLuint(k), sizeof name, nullptr, &size, &type, name);
                sawSampler |= type == gl::SAMPLER_BUFFER && !std::strcmp(name, "src");
                sawImage |= type == 0x9051 /*IMAGE_BUFFER*/ && !std::strcmp(name, "dst");
            }
            CHECK(sawSampler && sawImage);
            GLuint tbuf[2], ttex[2];
            glGenBuffers(2, tbuf);
            glGenTextures(2, ttex);
            const std::vector<float> data(16, 1.0f);
            glBindBuffer(gl::TEXTURE_BUFFER, tbuf[0]);
            glBufferData(gl::TEXTURE_BUFFER, GLsizeiptr(data.size() * 4), data.data(), gl::STATIC_DRAW);
            glBindBuffer(gl::TEXTURE_BUFFER, tbuf[1]);
            glBufferData(gl::TEXTURE_BUFFER, 16 * 4, nullptr, gl::DYNAMIC_DRAW);
            glUseProgram(cp);
            glActiveTexture(gl::TEXTURE0 + 2);
            glBindTexture(gl::TEXTURE_BUFFER, ttex[0]);
            glTexBuffer(gl::TEXTURE_BUFFER, gl::RGBA32F, tbuf[0]);
            glUniform1i(glGetUniformLocation(cp, "src"), 2);
            glBindTexture(gl::TEXTURE_BUFFER, ttex[1]);
            glTexBuffer(gl::TEXTURE_BUFFER, gl::RGBA32F, tbuf[1]);
            glBindImageTexture(0, ttex[1], 0, 0, 0, gl::WRITE_ONLY, gl::RGBA32F);
            CHECK(err() == gl::NO_ERROR);
            glActiveTexture(gl::TEXTURE0 + 2);
            glBindTexture(gl::TEXTURE_BUFFER, ttex[0]);
            const int before = counts.dispatches;
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.dispatches == before + 1);
            glMemoryBarrier(0x00000020 | 0x00000001);  // SHADER_IMAGE_ACCESS | VERTEX_ATTRIB_ARRAY
            CHECK(err() == gl::NO_ERROR);
            // An imageBuffer unit with nothing attached: the dispatch is skipped (a store has nowhere safe to go), no GL error.
            glBindImageTexture(0, 0, 0, 0, 0, gl::WRITE_ONLY, gl::RGBA32F);
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.dispatches == before + 1);
            glBindImageTexture(0, ttex[1], 0, 0, 0, gl::WRITE_ONLY, gl::RGBA32F);
            glBindTexture(gl::TEXTURE_BUFFER, 0);  // unbound samplerBuffer: reads the zero stand-in, the dispatch still runs
            glDispatchCompute(1, 1, 1);
            CHECK(err() == gl::NO_ERROR && counts.dispatches == before + 2);
            glActiveTexture(gl::TEXTURE0);
            glUseProgram(p);
        }
    }

    {  // GL_EXT_draw_buffers_indexed: per draw buffer state, its queries and errors.
        G(glEnablei, void (*)(GLenum, GLuint));
        G(glDisablei, void (*)(GLenum, GLuint));
        G(glIsEnabledi, GLboolean (*)(GLenum, GLuint));
        G(glBlendFuncSeparatei, void (*)(GLuint, GLenum, GLenum, GLenum, GLenum));
        G(glBlendEquationSeparatei, void (*)(GLuint, GLenum, GLenum));
        G(glColorMaski, void (*)(GLuint, GLboolean, GLboolean, GLboolean, GLboolean));
        G(glGetBooleani_v, void (*)(GLenum, GLuint, GLboolean*));
        auto isEnablediExt = reinterpret_cast<GLboolean (*)(GLenum, GLuint)>(gl<void*>("glIsEnablediEXT"));
        CHECK(isEnablediExt != nullptr);
        glEnablei(gl::BLEND, 2);
        CHECK(err() == gl::NO_ERROR && glIsEnabledi(gl::BLEND, 2) && !glIsEnabledi(gl::BLEND, 1) && isEnablediExt(gl::BLEND, 2));
        glDisablei(gl::BLEND, 2);
        CHECK(!glIsEnabledi(gl::BLEND, 2));
        glBlendFuncSeparatei(1, gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA, gl::ONE, gl::ZERO);
        glBlendEquationSeparatei(1, 0x800A /*FUNC_SUBTRACT*/, 0x8007 /*MIN*/);
        glColorMaski(1, 1, 0, 1, 0);
        GLint q4[4] = {};
        glGetIntegeri_v(gl::BLEND_SRC_RGB, 1, q4);
        CHECK(q4[0] == gl::SRC_ALPHA);
        glGetIntegeri_v(gl::BLEND_DST_RGB, 1, q4);
        CHECK(q4[0] == gl::ONE_MINUS_SRC_ALPHA);
        glGetIntegeri_v(gl::BLEND_SRC_ALPHA, 1, q4);
        CHECK(q4[0] == gl::ONE);
        glGetIntegeri_v(gl::BLEND_DST_ALPHA, 1, q4);
        CHECK(q4[0] == gl::ZERO);
        glGetIntegeri_v(gl::BLEND_EQUATION_RGB, 1, q4);
        CHECK(q4[0] == 0x800A);
        glGetIntegeri_v(gl::BLEND_EQUATION_ALPHA, 1, q4);
        CHECK(q4[0] == 0x8007 && err() == gl::NO_ERROR);
        glGetIntegeri_v(gl::COLOR_WRITEMASK, 1, q4);
        CHECK(q4[0] == 1 && q4[1] == 0 && q4[2] == 1 && q4[3] == 0);
        GLboolean b4[4] = {};
        glGetBooleani_v(gl::COLOR_WRITEMASK, 1, b4);
        CHECK(b4[0] && !b4[1] && b4[2] && !b4[3]);
        glGetIntegeri_v(gl::BLEND_SRC_RGB, 0, q4);
        CHECK(q4[0] == gl::ONE);  // draw buffer 0 keeps its own state
        glEnablei(gl::DEPTH_TEST, 0);
        CHECK(err() == gl::INVALID_ENUM);
        glEnablei(gl::BLEND, 64);
        CHECK(err() == gl::INVALID_VALUE);
        glDisablei(gl::BLEND, 64);
        CHECK(err() == gl::INVALID_VALUE);
        glIsEnabledi(gl::CULL_FACE, 0);
        CHECK(err() == gl::INVALID_ENUM);
        glIsEnabledi(gl::BLEND, 64);
        CHECK(err() == gl::INVALID_VALUE);
        glGetIntegeri_v(gl::BLEND_SRC_RGB, 64, q4);
        CHECK(err() == gl::INVALID_VALUE);
    }

    // ---- glGetTexLevelParameter*
    glBindTexture(gl::TEXTURE_2D, tex[0]);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 1, gl::TEXTURE_WIDTH, &w);
    CHECK(w == 4);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_HEIGHT, &w);
    CHECK(w == 8);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_DEPTH, &w);
    CHECK(w == 1);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_INTERNAL_FORMAT, &w);
    CHECK(w == gl::RGBA8);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_RED_SIZE, &w);
    CHECK(w == 8);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_RED_TYPE, &w);
    CHECK(w == 0x8C17 /*UNSIGNED_NORMALIZED*/);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_SAMPLES, &w);
    CHECK(w == 0);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, gl::TEXTURE_COMPRESSED, &w);
    CHECK(w == 0);
    GLfloat wf = -1;
    glGetTexLevelParameterfv(gl::TEXTURE_2D, 0, gl::TEXTURE_WIDTH, &wf);
    CHECK(wf == 8.0f);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 7, gl::TEXTURE_WIDTH, &w);
    CHECK(w == 0 && err() == gl::NO_ERROR);  // undefined level
    glGetTexLevelParameteriv(gl::TEXTURE_2D, -1, gl::TEXTURE_WIDTH, &w);
    CHECK(err() == gl::INVALID_VALUE);
    glGetTexLevelParameteriv(gl::TEXTURE_2D, 0, 0x7777, &w);
    CHECK(err() == gl::INVALID_ENUM);
    glGetTexLevelParameteriv(0x7777, 0, gl::TEXTURE_WIDTH, &w);
    CHECK(err() == gl::INVALID_ENUM);
    glBindTexture(gl::TEXTURE_2D_ARRAY, tex[3]);
    glGetTexLevelParameteriv(gl::TEXTURE_2D_ARRAY, 0, gl::TEXTURE_DEPTH, &w);
    CHECK(w == 4);

    // ---- multisample textures
    G(glTexStorage2DMultisample, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLboolean));
    G(glTexImage2DMultisample, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLboolean));
    glBindTexture(gl::TEXTURE_2D_MULTISAMPLE, tex[5]);
    glTexStorage2DMultisample(gl::TEXTURE_2D_MULTISAMPLE, 4, gl::RGBA8, 16, 8, 1);
    CHECK(err() == gl::NO_ERROR);
    glGetTexLevelParameteriv(gl::TEXTURE_2D_MULTISAMPLE, 0, gl::TEXTURE_SAMPLES, &w);
    CHECK(w >= 4);
    glGetTexLevelParameteriv(gl::TEXTURE_2D_MULTISAMPLE, 0, gl::TEXTURE_WIDTH, &w);
    CHECK(w == 16);
    glGetTexLevelParameteriv(gl::TEXTURE_2D_MULTISAMPLE, 0, gl::TEXTURE_FIXED_SAMPLE_LOCATIONS, &w);
    CHECK(w == 1);
    glTexStorage2DMultisample(gl::TEXTURE_2D_MULTISAMPLE, 4, gl::RGBA8, 16, 8, 1);
    CHECK(err() == gl::INVALID_OPERATION);  // immutable
    GLuint ms2;
    glGenTextures(1, &ms2);
    glBindTexture(gl::TEXTURE_2D_MULTISAMPLE, ms2);
    glTexStorage2DMultisample(gl::TEXTURE_2D_MULTISAMPLE, 0, gl::RGBA8, 16, 8, 1);
    CHECK(err() == gl::INVALID_VALUE);
    glTexStorage2DMultisample(gl::TEXTURE_2D_MULTISAMPLE, 1000, gl::RGBA8, 16, 8, 1);
    CHECK(err() == gl::INVALID_OPERATION);
    glTexStorage2DMultisample(gl::TEXTURE_2D_MULTISAMPLE, 4, gl::RGBA8, 0, 8, 1);
    CHECK(err() == gl::INVALID_VALUE);
    glTexStorage2DMultisample(gl::TEXTURE_2D, 4, gl::RGBA8, 16, 8, 1);
    CHECK(err() == gl::INVALID_ENUM);
    glTexImage2DMultisample(gl::TEXTURE_2D_MULTISAMPLE, 4, gl::RGBA8, 8, 8, 0);
    CHECK(err() == gl::NO_ERROR);
    glGetTexLevelParameteriv(gl::TEXTURE_2D_MULTISAMPLE, 0, gl::TEXTURE_FIXED_SAMPLE_LOCATIONS, &w);
    CHECK(w == 0);

    // ---- framebuffer texture 3D / layer
    G(glFramebufferTexture3D, void (*)(GLenum, GLenum, GLenum, GLuint, GLint, GLint));
    G(glFramebufferTextureLayer, void (*)(GLenum, GLenum, GLuint, GLint, GLint));
    G(glGenFramebuffers, void (*)(GLsizei, GLuint*));
    G(glBindFramebuffer, void (*)(GLenum, GLuint));
    G(glCheckFramebufferStatus, GLenum (*)(GLenum));
    G(glGetFramebufferAttachmentParameteriv, void (*)(GLenum, GLenum, GLenum, GLint*));
    GLuint t3, fb;
    glGenTextures(1, &t3);
    glBindTexture(gl::TEXTURE_3D, t3);
    glTexStorage3D(gl::TEXTURE_3D, 1, gl::RGBA8, 4, 4, 4);
    CHECK(err() == gl::NO_ERROR);
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(gl::FRAMEBUFFER, fb);
    glFramebufferTexture3D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_3D, t3, 0, 1);
    CHECK(err() == gl::NO_ERROR && glCheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE);
    glGetFramebufferAttachmentParameteriv(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER, &w);
    CHECK(w == 1);
    glFramebufferTexture3D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_3D, t3, 0, 100000);
    CHECK(err() == gl::INVALID_VALUE);  // z offset beyond MAX_3D_TEXTURE_SIZE
    glFramebufferTexture3D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, t3, 0, 0);
    CHECK(err() == gl::INVALID_ENUM || err() == gl::INVALID_OPERATION);
    glFramebufferTextureLayer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, t3, 0, 2);
    CHECK(err() == gl::NO_ERROR && glCheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE);
    glFramebufferTextureLayer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, tex[3], 0, 3);
    CHECK(err() == gl::NO_ERROR && glCheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(gl::FRAMEBUFFER, 0);

    // ---- queries
    G(glGenQueries, void (*)(GLsizei, GLuint*));
    G(glDeleteQueries, void (*)(GLsizei, const GLuint*));
    G(glIsQuery, GLboolean (*)(GLuint));
    G(glBeginQuery, void (*)(GLenum, GLuint));
    G(glEndQuery, void (*)(GLenum));
    G(glQueryCounter, void (*)(GLuint, GLenum));
    G(glGetQueryiv, void (*)(GLenum, GLenum, GLint*));
    G(glGetQueryObjectuiv, void (*)(GLuint, GLenum, GLuint*));
    G(glGetQueryObjectiv, void (*)(GLuint, GLenum, GLint*));
    G(glGetQueryObjectui64v, void (*)(GLuint, GLenum, GLuint64*));
    G(glGetQueryObjecti64v, void (*)(GLuint, GLenum, GLint64*));
    CHECK(vpgles_proc("glGetQueryObjectuivEXT") == vpgles_proc("glGetQueryObjectuiv"));
    CHECK(vpgles_proc("glGetQueryObjectui64vEXT") == vpgles_proc("glGetQueryObjectui64v"));
    CHECK(vpgles_proc("glBeginQueryEXT") == vpgles_proc("glBeginQuery"));
    CHECK(vpgles_proc("glQueryCounterEXT") == vpgles_proc("glQueryCounter"));
    glBindVertexArray(0);
    glUseProgram(p);
    GLuint qs[3];
    glGenQueries(3, qs);
    CHECK(!glIsQuery(qs[0]));  // generated but never begun
    glBeginQuery(gl::ANY_SAMPLES_PASSED, qs[0]);
    CHECK(err() == gl::NO_ERROR && glIsQuery(qs[0]));
    glGetQueryiv(gl::ANY_SAMPLES_PASSED, gl::CURRENT_QUERY, &q);
    CHECK(GLuint(q) == qs[0]);
    GLuint r = 77;
    glGetQueryObjectuiv(qs[0], gl::QUERY_RESULT, &r);
    CHECK(err() == gl::INVALID_OPERATION && r == 77);  // still active
    glBeginQuery(gl::ANY_SAMPLES_PASSED, qs[1]);
    CHECK(err() == gl::INVALID_OPERATION);  // the target already has an active query
    glBeginQuery(gl::ANY_SAMPLES_PASSED_CONSERVATIVE, qs[0]);
    CHECK(err() == gl::INVALID_OPERATION);  // the query is already active
    glDrawArrays(gl::TRIANGLES, 0, 3);
    glEndQuery(gl::ANY_SAMPLES_PASSED);
    CHECK(err() == gl::NO_ERROR);
    glGetQueryiv(gl::ANY_SAMPLES_PASSED, gl::CURRENT_QUERY, &q);
    CHECK(q == 0);
    glGetQueryObjectuiv(qs[0], gl::QUERY_RESULT_AVAILABLE, &r);
    CHECK(r == 1);
    glGetQueryObjectuiv(qs[0], gl::QUERY_RESULT, &r);
    CHECK(r <= 1);  // the GPU counts the samples (the null backend draws nothing: 0); the dEQP occlusion_query cases check the count
    glBeginQuery(gl::ANY_SAMPLES_PASSED_CONSERVATIVE, qs[1]);
    glEndQuery(gl::ANY_SAMPLES_PASSED_CONSERVATIVE);
    glGetQueryObjectuiv(qs[1], gl::QUERY_RESULT, &r);
    CHECK(err() == gl::NO_ERROR && r == 0);  // nothing was drawn
    glEndQuery(gl::ANY_SAMPLES_PASSED);
    CHECK(err() == gl::INVALID_OPERATION);  // no active query
    glBeginQuery(gl::ANY_SAMPLES_PASSED, 0);
    CHECK(err() == gl::INVALID_OPERATION);
    glBeginQuery(0x7777, qs[2]);
    CHECK(err() == gl::INVALID_ENUM);
    glBeginQuery(gl::ANY_SAMPLES_PASSED, 4242);
    CHECK(err() == gl::INVALID_OPERATION);
    glBeginQuery(gl::TIME_ELAPSED_EXT, qs[0]);
    CHECK(err() == gl::INVALID_OPERATION);  // the query already has another target
    // Time elapsed: nonzero, 64-bit.
    glBeginQuery(gl::TIME_ELAPSED_EXT, qs[2]);
    usleep(2000);
    glEndQuery(gl::TIME_ELAPSED_EXT);
    GLuint64 t64 = 0;
    glGetQueryObjectui64v(qs[2], gl::QUERY_RESULT, &t64);
    CHECK(err() == gl::NO_ERROR && t64 >= 1000000 && t64 < 10000000000ull);
    GLint64 i64 = 0;
    glGetQueryObjecti64v(qs[2], gl::QUERY_RESULT, &i64);
    CHECK(i64 == GLint64(t64));
    GLint i32 = 0;
    glGetQueryObjectiv(qs[2], gl::QUERY_RESULT_AVAILABLE, &i32);
    CHECK(i32 == 1);
    glGetQueryiv(gl::TIME_ELAPSED_EXT, gl::QUERY_COUNTER_BITS_EXT, &q);
    CHECK(q == 64);
    glGetQueryiv(gl::ANY_SAMPLES_PASSED, gl::QUERY_COUNTER_BITS_EXT, &q);
    CHECK(q == 0);
    glGetQueryiv(gl::ANY_SAMPLES_PASSED, 0x7777, &q);
    CHECK(err() == gl::INVALID_ENUM);
    // Timestamp.
    GLuint ts[2];
    glGenQueries(2, ts);
    glQueryCounter(ts[0], gl::TIMESTAMP_EXT);
    usleep(1500);
    glQueryCounter(ts[1], gl::TIMESTAMP_EXT);
    GLuint64 a = 0, b2 = 0;
    glGetQueryObjectui64v(ts[0], gl::QUERY_RESULT, &a);
    glGetQueryObjectui64v(ts[1], gl::QUERY_RESULT, &b2);
    CHECK(err() == gl::NO_ERROR && a > 0 && b2 > a);
    glQueryCounter(ts[0], gl::ANY_SAMPLES_PASSED);
    CHECK(err() == gl::INVALID_ENUM);
    GLint64 now = 0;
    G(glGetInteger64v, void (*)(GLenum, GLint64*));
    glGetInteger64v(gl::TIMESTAMP_EXT, &now);
    CHECK(err() == gl::NO_ERROR && now > 0);
    glDeleteQueries(3, qs);
    CHECK(!glIsQuery(qs[0]));
    glGetQueryObjectuiv(qs[0], gl::QUERY_RESULT, &r);
    CHECK(err() == gl::INVALID_OPERATION);

    // ---- KHR_debug
    G(glDebugMessageCallback, void (*)(void*, const void*));
    G(glDebugMessageControl, void (*)(GLenum, GLenum, GLenum, GLsizei, const GLuint*, GLboolean));
    G(glDebugMessageInsert, void (*)(GLenum, GLenum, GLuint, GLenum, GLsizei, const char*));
    G(glGetDebugMessageLog, GLuint (*)(GLuint, GLsizei, GLenum*, GLenum*, GLuint*, GLenum*, GLsizei*, char*));
    G(glPushDebugGroup, void (*)(GLenum, GLuint, GLsizei, const char*));
    G(glPopDebugGroup, void (*)());
    G(glObjectLabel, void (*)(GLenum, GLuint, GLsizei, const char*));
    G(glGetObjectLabel, void (*)(GLenum, GLuint, GLsizei, GLsizei*, char*));
    G(glGetPointerv, void (*)(GLenum, void**));
    G(glDisable, void (*)(GLenum));
    G(glIsEnabled, GLboolean (*)(GLenum));
    CHECK(vpgles_proc("glDebugMessageCallbackKHR") == vpgles_proc("glDebugMessageCallback"));
    CHECK(vpgles_proc("glLabelObjectEXT") != nullptr && vpgles_proc("glGetObjectLabelEXT") != nullptr);
    // Message log (no callback): queued while DEBUG_OUTPUT is on.
    glDebugMessageInsert(gl::DEBUG_SOURCE_APPLICATION, 0x8268 /*MARKER*/, 5, gl::DEBUG_SEVERITY_HIGH, -1, "before enable");
    char logb[256];
    GLsizei lens[4];
    GLenum srcs[4], types[4], sevs[4];
    GLuint ids[4];
    CHECK(glGetDebugMessageLog(4, sizeof logb, srcs, types, ids, sevs, lens, logb) == 0);  // output is off: nothing logged
    glEnable(gl::DEBUG_OUTPUT);
    CHECK(glIsEnabled(gl::DEBUG_OUTPUT));
    glDebugMessageInsert(gl::DEBUG_SOURCE_APPLICATION, 0x8268, 6, gl::DEBUG_SEVERITY_HIGH, -1, "queued");
    glBindBuffer(0x1234, 0);  // INVALID_ENUM is reported as an error message
    CHECK(err() == gl::INVALID_ENUM);
    GLint nlog = 0;
    glGetIntegerv(gl::DEBUG_LOGGED_MESSAGES, &nlog);
    CHECK(nlog == 2);
    CHECK(glGetDebugMessageLog(1, sizeof logb, srcs, types, ids, sevs, lens, logb) == 1);
    CHECK(srcs[0] == gl::DEBUG_SOURCE_APPLICATION && ids[0] == 6 && sevs[0] == gl::DEBUG_SEVERITY_HIGH && !std::strcmp(logb, "queued") &&
          lens[0] == 7);
    CHECK(glGetDebugMessageLog(4, sizeof logb, srcs, types, ids, sevs, lens, logb) == 1);
    CHECK(srcs[0] == gl::DEBUG_SOURCE_API && types[0] == gl::DEBUG_TYPE_ERROR && ids[0] == gl::INVALID_ENUM);
    // Callback.
    g_dbg.clear();
    glDebugMessageCallback(reinterpret_cast<void*>(&dbgCallback), reinterpret_cast<const void*>(0x1234));
    void* cbp = nullptr;
    glGetPointerv(gl::DEBUG_CALLBACK_FUNCTION, &cbp);
    CHECK(cbp == reinterpret_cast<void*>(&dbgCallback));
    glGetPointerv(gl::DEBUG_CALLBACK_USER_PARAM, &cbp);
    CHECK(cbp == reinterpret_cast<void*>(0x1234));
    glDebugMessageInsert(gl::DEBUG_SOURCE_APPLICATION, 0x8268, 9, gl::DEBUG_SEVERITY_HIGH, 3, "abcdef");
    CHECK(g_dbg.size() == 1 && g_dbg[0] == "824A/8268/9/9146/3/1:abc");
    glDebugMessageControl(gl::DEBUG_SOURCE_APPLICATION, gl::DONT_CARE_ENUM, gl::DONT_CARE_ENUM, 0, nullptr, 0);
    glDebugMessageInsert(gl::DEBUG_SOURCE_APPLICATION, 0x8268, 10, gl::DEBUG_SEVERITY_HIGH, -1, "filtered");
    CHECK(g_dbg.size() == 1);
    const GLuint only[1] = {11};
    glDebugMessageControl(gl::DEBUG_SOURCE_APPLICATION, 0x8268, gl::DONT_CARE_ENUM, 1, only, 1);
    glDebugMessageInsert(gl::DEBUG_SOURCE_APPLICATION, 0x8268, 11, gl::DEBUG_SEVERITY_HIGH, -1, "reenabled");
    glDebugMessageInsert(gl::DEBUG_SOURCE_APPLICATION, 0x8268, 12, gl::DEBUG_SEVERITY_HIGH, -1, "still filtered");
    CHECK(g_dbg.size() == 2 && g_dbg[1].find("reenabled") != std::string::npos);
    glDebugMessageControl(gl::DONT_CARE_ENUM, gl::DONT_CARE_ENUM, gl::DONT_CARE_ENUM, 0, nullptr, 1);
    glDebugMessageControl(gl::DEBUG_SOURCE_API, gl::DONT_CARE_ENUM, gl::DEBUG_SEVERITY_HIGH, 1, only, 1);
    CHECK(err() == gl::INVALID_OPERATION);  // ids with a severity filter
    glDebugMessageControl(0x7777, gl::DONT_CARE_ENUM, gl::DONT_CARE_ENUM, 0, nullptr, 1);
    CHECK(err() == gl::INVALID_ENUM);
    glDebugMessageInsert(gl::DEBUG_SOURCE_API, 0x8268, 1, gl::DEBUG_SEVERITY_HIGH, -1, "x");
    CHECK(err() == gl::INVALID_ENUM);  // only APPLICATION / THIRD_PARTY sources may be inserted
    g_dbg.clear();
    glBindBuffer(0x1234, 0);
    CHECK(err() == gl::INVALID_ENUM && g_dbg.size() == 1 && g_dbg[0].find("INVALID_ENUM") != std::string::npos);
    // Groups.
    g_dbg.clear();
    glPopDebugGroup();
    CHECK(err() == gl::STACK_UNDERFLOW);
    glPushDebugGroup(gl::DEBUG_SOURCE_APPLICATION, 1, -1, "pass A");
    glGetIntegerv(gl::DEBUG_GROUP_STACK_DEPTH, &q);
    CHECK(q == 2);
    glPushDebugGroup(gl::DEBUG_SOURCE_APPLICATION, 2, 6, "pass B");
    glGetIntegerv(gl::DEBUG_GROUP_STACK_DEPTH, &q);
    CHECK(q == 3);
    glPopDebugGroup();
    glPopDebugGroup();
    glGetIntegerv(gl::DEBUG_GROUP_STACK_DEPTH, &q);
    CHECK(q == 1 && err() == gl::NO_ERROR);
    glPushDebugGroup(gl::DEBUG_SOURCE_API, 1, -1, "x");
    CHECK(err() == gl::INVALID_ENUM);
    for (int i = 0; i < 70; ++i) glPushDebugGroup(gl::DEBUG_SOURCE_APPLICATION, 1, -1, "deep");
    CHECK(err() == gl::STACK_OVERFLOW);
    for (int i = 0; i < 70; ++i) glPopDebugGroup();
    (void)err();
    // Labels.
    GLuint lb;
    glGenBuffers(1, &lb);
    glBindBuffer(gl::ARRAY_BUFFER, lb);
    glObjectLabel(gl::BUFFER_ID, lb, -1, "vertex data");
    CHECK(err() == gl::NO_ERROR);
    GLsizei ll = 0;
    char lbl[32] = {};
    glGetObjectLabel(gl::BUFFER_ID, lb, sizeof lbl, &ll, lbl);
    CHECK(ll == 11 && !std::strcmp(lbl, "vertex data"));
    glObjectLabel(gl::TEXTURE_ID, tex[0], 4, "abcdefgh");
    glGetObjectLabel(gl::TEXTURE_ID, tex[0], sizeof lbl, &ll, lbl);
    CHECK(ll == 4 && !std::strcmp(lbl, "abcd"));
    glGetObjectLabel(gl::BUFFER_ID, lb, 4, &ll, lbl);
    CHECK(ll == 3 && !std::strcmp(lbl, "ver"));  // truncated to bufSize - 1
    glObjectLabel(gl::BUFFER_ID, 8888, -1, "x");
    CHECK(err() == gl::INVALID_VALUE);
    glObjectLabel(0x7777, lb, -1, "x");
    CHECK(err() != gl::NO_ERROR);
    glObjectLabel(gl::BUFFER_ID, lb, 4096, "x");
    CHECK(err() == gl::INVALID_VALUE);
    glObjectLabel(gl::BUFFER_ID, lb, 0, nullptr);  // null label clears
    glGetObjectLabel(gl::BUFFER_ID, lb, sizeof lbl, &ll, lbl);
    CHECK(ll == 0);
    {
        G(glLabelObjectEXT, void (*)(GLenum, GLuint, GLsizei, const char*));
        G(glGetObjectLabelEXT, void (*)(GLenum, GLuint, GLsizei, GLsizei*, char*));
        glLabelObjectEXT(0x9151 /*BUFFER_OBJECT_EXT*/, lb, -1, "ext label");
        glGetObjectLabelEXT(0x9151, lb, sizeof lbl, &ll, lbl);
        CHECK(err() == gl::NO_ERROR && !std::strcmp(lbl, "ext label"));
    }
    // ---- limits of the stages a draw binds resources for
    {
        GLint v = -1;
        glGetIntegerv(0x90DA, &v);  // MAX_FRAGMENT_SHADER_STORAGE_BLOCKS
        CHECK(v >= 4);
        glGetIntegerv(0x90D6, &v);  // MAX_VERTEX_SHADER_STORAGE_BLOCKS: 0 (VPVulkan has no vertexPipelineStoresAndAtomics)
        CHECK(v == 0);
        glGetIntegerv(0x92D0, &v);  // MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS
        CHECK(v >= 1);
    }

    // ---- program pipelines
    {
        G(glGenProgramPipelines, void (*)(GLsizei, GLuint*));
        G(glDeleteProgramPipelines, void (*)(GLsizei, const GLuint*));
        G(glIsProgramPipeline, GLboolean (*)(GLuint));
        G(glBindProgramPipeline, void (*)(GLuint));
        G(glUseProgramStages, void (*)(GLuint, GLbitfield, GLuint));
        G(glActiveShaderProgram, void (*)(GLuint, GLuint));
        G(glValidateProgramPipeline, void (*)(GLuint));
        G(glGetProgramPipelineiv, void (*)(GLuint, GLenum, GLint*));
        G(glGetProgramPipelineInfoLog, void (*)(GLuint, GLsizei, GLsizei*, char*));
        G(glProgramParameteri, void (*)(GLuint, GLenum, GLint));
        G(glLinkProgram, void (*)(GLuint));
        G(glVertexAttrib4f, void (*)(GLuint, GLfloat, GLfloat, GLfloat, GLfloat));
        GLuint pl = 0;
        glGenProgramPipelines(1, &pl);
        CHECK(pl != 0 && !glIsProgramPipeline(pl));  // exists after the first bind
        GLuint sp = program(kVs100, kFs100);
        glUseProgramStages(pl, 0x3, sp);
        CHECK(err() == gl::INVALID_OPERATION);  // not separable
        glProgramParameteri(sp, gl::PROGRAM_SEPARABLE, 1);
        glLinkProgram(sp);
        glUseProgramStages(pl, 0x4, sp);
        CHECK(err() == gl::INVALID_VALUE);  // not a stage bit
        glUseProgramStages(pl + 100, 0x3, sp);
        CHECK(err() == gl::INVALID_OPERATION);
        glUseProgramStages(pl, 0x3, sp);
        CHECK(err() == gl::NO_ERROR);
        glBindProgramPipeline(pl);
        CHECK(err() == gl::NO_ERROR && glIsProgramPipeline(pl));
        glActiveShaderProgram(pl, sp);
        GLint v = -1;
        glGetIntegerv(0x825A, &v);  // PROGRAM_PIPELINE_BINDING
        CHECK(GLuint(v) == pl);
        glGetProgramPipelineiv(pl, 0x8259, &v);
        CHECK(GLuint(v) == sp);  // ACTIVE_PROGRAM
        glGetProgramPipelineiv(pl, gl::VERTEX_SHADER, &v);
        CHECK(GLuint(v) == sp);
        glGetProgramPipelineiv(pl, gl::FRAGMENT_SHADER, &v);
        CHECK(GLuint(v) == sp);
        glGetProgramPipelineiv(pl, gl::COMPUTE_SHADER, &v);
        CHECK(v == 0);
        glGetProgramPipelineiv(pl, gl::VALIDATE_STATUS, &v);
        CHECK(v == 0);
        glValidateProgramPipeline(pl);
        glGetProgramPipelineiv(pl, gl::VALIDATE_STATUS, &v);
        CHECK(v == 1);
        glGetProgramPipelineiv(pl, 0x8B84, &v);
        CHECK(v == 0 && err() == gl::NO_ERROR);
        glGetProgramPipelineiv(pl, 0x1234, &v);
        CHECK(err() == gl::INVALID_ENUM);
        // Draw through the pipeline: no UseProgram, vertex and fragment from the same program.
        glUseProgram(0);
        glVertexAttrib4f(0, 0, 0, 0, 1);
        counts = {};
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::NO_ERROR && counts.draws == 1);
        // glUseProgram takes precedence over the pipeline.
        GLuint other = program(kVs100, kFs100);
        glUseProgram(other);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::NO_ERROR && counts.draws == 2);
        glUseProgram(0);
        // Two different programs for the two stages: no draw, an error, no crash.
        glProgramParameteri(other, gl::PROGRAM_SEPARABLE, 1);
        glLinkProgram(other);
        glUseProgramStages(pl, 0x2, other);
        glValidateProgramPipeline(pl);
        glGetProgramPipelineiv(pl, gl::VALIDATE_STATUS, &v);
        CHECK(v == 0);
        glGetProgramPipelineiv(pl, 0x8B84, &v);
        CHECK(v > 0);
        char plog[256] = {};
        GLsizei pllen = 0;
        glGetProgramPipelineInfoLog(pl, sizeof plog, &pllen, plog);
        CHECK(pllen > 0 && err() == gl::NO_ERROR);
        counts = {};
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::INVALID_OPERATION && counts.draws == 0);
        glUseProgramStages(pl, 0x2, 0);  // clearing a stage
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::INVALID_OPERATION && counts.draws == 0);
        glDeleteProgramPipelines(1, &pl);
        glGetIntegerv(0x825A, &v);
        CHECK(v == 0 && !glIsProgramPipeline(pl));
        glBindProgramPipeline(pl);
        CHECK(err() == gl::INVALID_OPERATION);
    }

    // ---- framebuffer parameters, an attachment-less framebuffer, sample positions and mask
    {
        G(glGenFramebuffers, void (*)(GLsizei, GLuint*));
        G(glBindFramebuffer, void (*)(GLenum, GLuint));
        G(glFramebufferParameteri, void (*)(GLenum, GLenum, GLint));
        G(glGetFramebufferParameteriv, void (*)(GLenum, GLenum, GLint*));
        G(glCheckFramebufferStatus, GLenum (*)(GLenum));
        G(glGenRenderbuffers, void (*)(GLsizei, GLuint*));
        G(glBindRenderbuffer, void (*)(GLenum, GLuint));
        G(glRenderbufferStorageMultisample, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
        G(glFramebufferRenderbuffer, void (*)(GLenum, GLenum, GLenum, GLuint));
        G(glGetMultisamplefv, void (*)(GLenum, GLuint, GLfloat*));
        G(glSampleMaski, void (*)(GLuint, GLbitfield));
        G(glVertexAttrib4f, void (*)(GLuint, GLfloat, GLfloat, GLfloat, GLfloat));
        GLuint fbo = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(gl::FRAMEBUFFER, fbo);
        GLint v = -1;
        for (GLenum pn : {0x9310u, 0x9311u, 0x9312u, 0x9313u, 0x9314u}) {
            v = -1;
            glGetFramebufferParameteriv(gl::FRAMEBUFFER, pn, &v);
            CHECK(v == 0 && err() == gl::NO_ERROR);
        }
        CHECK(glCheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE);
        glFramebufferParameteri(gl::FRAMEBUFFER, 0x9310, 32);
        glFramebufferParameteri(gl::FRAMEBUFFER, 0x9311, 16);
        glFramebufferParameteri(gl::FRAMEBUFFER, 0x9312, 2);
        glFramebufferParameteri(gl::FRAMEBUFFER, 0x9313, 4);
        glFramebufferParameteri(gl::FRAMEBUFFER, 0x9314, 1);
        CHECK(err() == gl::NO_ERROR);
        const GLint want[5] = {32, 16, 2, 4, 1};
        for (int i = 0; i < 5; ++i) {
            glGetFramebufferParameteriv(gl::FRAMEBUFFER, 0x9310 + GLenum(i), &v);
            CHECK(v == want[i]);
        }
        glFramebufferParameteri(gl::FRAMEBUFFER, 0x9310, -1);
        CHECK(err() == gl::INVALID_VALUE);
        glFramebufferParameteri(gl::FRAMEBUFFER, 0x9999, 1);
        CHECK(err() == gl::INVALID_ENUM);
        glGetFramebufferParameteriv(0x1234, 0x9310, &v);
        CHECK(err() == gl::INVALID_ENUM);
        CHECK(glCheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE);  // default size, no attachments
        glUseProgram(program(kVs100, kFs100));
        glVertexAttrib4f(0, 0, 0, 0, 1);
        counts = {};
        glDrawArrays(gl::TRIANGLES, 0, 3);
        CHECK(err() == gl::NO_ERROR && counts.draws == 1 && counts.lastColorCount == 0);
        // Sample positions: the framebuffer's sample count picks the standard table.
        GLfloat pos[2] = {-1, -1};
        glBindFramebuffer(gl::FRAMEBUFFER, 0);
        glGetMultisamplefv(0x8E50, 0, pos);
        CHECK(pos[0] == 0.5f && pos[1] == 0.5f && err() == gl::NO_ERROR);
        glGetMultisamplefv(0x8E50, 1, pos);
        CHECK(err() == gl::INVALID_VALUE);
        glGetMultisamplefv(0x1234, 0, pos);
        CHECK(err() == gl::INVALID_ENUM);
        GLuint rb = 0;
        glGenRenderbuffers(1, &rb);
        glBindRenderbuffer(gl::RENDERBUFFER, rb);
        glRenderbufferStorageMultisample(gl::RENDERBUFFER, 4, gl::RGBA8, 8, 8);
        glBindFramebuffer(gl::FRAMEBUFFER, fbo);
        glFramebufferRenderbuffer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::RENDERBUFFER, rb);
        CHECK(err() == gl::NO_ERROR);
        const float want4[4][2] = {{0.375f, 0.125f}, {0.875f, 0.375f}, {0.125f, 0.625f}, {0.625f, 0.875f}};
        for (GLuint i = 0; i < 4; ++i) {
            glGetMultisamplefv(0x8E50, i, pos);
            CHECK(pos[0] == want4[i][0] && pos[1] == want4[i][1]);
        }
        glGetMultisamplefv(0x8E50, 4, pos);
        CHECK(err() == gl::INVALID_VALUE);
        glBindFramebuffer(gl::FRAMEBUFFER, 0);
        // Sample mask: stored and reported.
        glSampleMaski(0, 0xF);
        CHECK(err() == gl::NO_ERROR);
        glGetIntegeri_v(0x8E52, 0, &v);
        CHECK(v == 0xF);
        glSampleMaski(1, 0);
        CHECK(err() == gl::INVALID_VALUE);
        glSampleMaski(0, 0xFFFFFFFFu);
        glGetIntegeri_v(0x8E52, 0, &v);
        CHECK(v == -1);
        // The remaining ES 3.0 entry points that used to be missing.
        G(glIsTransformFeedback, GLboolean (*)(GLuint));
        G(glGetTransformFeedbackVarying, void (*)(GLuint, GLuint, GLsizei, GLsizei*, GLsizei*, GLenum*, char*));
        CHECK(!glIsTransformFeedback(0) && !glIsTransformFeedback(1));  // names are objects only once generated and bound
        (void)glGetTransformFeedbackVarying;
    }

    // ---- STORAGE_BIT is lazy: only a texture used as an image pays for it, and keeps its contents
    {
        G(glTexStorage2D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
        G(glTexSubImage2D, void (*)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*));
        G(glBindImageTexture, void (*)(GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum));
        GLuint t[3];
        glGenTextures(3, t);
        counts = {};
        for (int i = 0; i < 3; ++i) {
            glBindTexture(gl::TEXTURE_2D, t[i]);
            glTexStorage2D(gl::TEXTURE_2D, i == 1 ? 3 : 1, gl::RGBA8, 4, 4);
        }
        std::vector<uint8_t> px(4 * 4 * 4, 0x7F);
        glTexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, 4, 4, gl::RGBA, gl::UNSIGNED_BYTE, px.data());
        CHECK(counts.imagesCreated == 3 && counts.imagesWithStorage == 0);  // RGBA8 is storage-capable but not used as an image
        counts = {};
        glBindImageTexture(0, t[0], 0, 0, 0, gl::READ_WRITE, gl::RGBA8);
        CHECK(err() == gl::NO_ERROR && counts.imagesCreated == 1 && counts.imagesWithStorage == 1);
        CHECK(counts.copiesImage == 1 && counts.lastCopyRegions == 1);  // the contents move to the new image
        glBindImageTexture(1, t[0], 0, 0, 0, gl::READ_ONLY, gl::RGBA8);
        CHECK(counts.imagesCreated == 1 && counts.copiesImage == 1);  // already has it
        glBindTexture(gl::TEXTURE_2D, t[1]);
        glBindImageTexture(2, t[1], 0, 0, 0, gl::READ_WRITE, gl::RGBA8);
        CHECK(counts.imagesWithStorage == 2 && counts.copiesImage == 2 && counts.lastCopyRegions == 3);  // every level
        // Bound before its storage exists: created with STORAGE_BIT straight away.
        GLuint late = 0;
        glGenTextures(1, &late);
        glBindTexture(gl::TEXTURE_2D, late);
        glBindImageTexture(3, late, 0, 0, 0, gl::WRITE_ONLY, gl::RGBA8);
        counts.imagesCreated = counts.imagesWithStorage = 0;
        glTexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8, 4, 4);
        CHECK(err() == gl::NO_ERROR && counts.imagesCreated == 1 && counts.imagesWithStorage == 1);
        // The sampled texture without it keeps working.
        counts = {};
        glBindImageTexture(0, 0, 0, 0, 0, gl::READ_ONLY, gl::RGBA8);
        glBindImageTexture(1, 0, 0, 0, 0, gl::READ_ONLY, gl::RGBA8);
        glBindImageTexture(2, 0, 0, 0, 0, gl::READ_ONLY, gl::RGBA8);
        glBindImageTexture(3, 0, 0, 0, 0, gl::READ_ONLY, gl::RGBA8);
        CHECK(err() == gl::NO_ERROR && counts.imagesCreated == 0 && counts.copiesImage == 0);
    }
    glDisable(gl::DEBUG_OUTPUT);
    CHECK(!glIsEnabled(gl::DEBUG_OUTPUT));
    e.down();
}

// The flow of dEQP's glsUniformBlockCase on an instance array of uniform blocks (`uniform B { T v; } b[3]`): link, the layout
// queries, the bindings, one buffer per block or one buffer for all of them, and a draw. Every stage and layout combination.
void testUbo() {
    Egl e;
    CHECK(e.up());
    G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
    G(glGetActiveUniformBlockiv, void (*)(GLuint, GLuint, GLenum, GLint*));
    G(glGetActiveUniformBlockName, void (*)(GLuint, GLuint, GLsizei, GLsizei*, char*));
    G(glGetActiveUniformsiv, void (*)(GLuint, GLsizei, const GLuint*, GLenum, GLint*));
    G(glGetActiveUniform, void (*)(GLuint, GLuint, GLsizei, GLsizei*, GLint*, GLenum*, char*));
    G(glGetUniformBlockIndex, GLuint (*)(GLuint, const char*));
    G(glUniformBlockBinding, void (*)(GLuint, GLuint, GLuint));
    G(glBindBufferBase, void (*)(GLenum, GLuint, GLuint));
    G(glBindBufferRange, void (*)(GLenum, GLuint, GLuint, GLintptr, GLsizeiptr));
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    G(glGetIntegerv, void (*)(GLenum, GLint*));
    G(glUseProgram, void (*)(GLuint));
    G(glVertexAttribPointer, void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
    G(glEnableVertexAttribArray, void (*)(GLuint));
    G(glBindAttribLocation, void (*)(GLuint, GLuint, const char*));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    GLuint vb;
    glGenBuffers(1, &vb);
    const float verts[] = {-1, -1, 0, 1, 1, -1, 0, 1, 0, 1, 0, 1};
    glBindBuffer(gl::ARRAY_BUFFER, vb);
    glBufferData(gl::ARRAY_BUFFER, sizeof verts, verts, gl::STATIC_DRAW);
    const char* types[][3] = {{"float", "1.0", "float"}, {"vec3", "vec3(1.0, 2.0, 3.0)", "vec3"}, {"mat4x3", "mat4x3(1.0)", "mat4x3"},
                              {"int", "7", "int"}, {"uvec4", "uvec4(1u, 2u, 3u, 4u)", "uvec4"}, {"bvec2", "bvec2(true, false)", "bvec2"}};
    for (const char* layout : {"std140", "shared", "packed"})
        for (const auto& t : types)
            for (int stage = 0; stage < 3; ++stage)  // 0 vertex, 1 fragment, 2 both
                for (int perBlock = 0; perBlock < 2; ++perBlock) {
                    const bool boolean = std::strcmp(t[0], "bvec2") == 0;
                    std::string decl = std::string("layout(") + layout + ") uniform Block { " + (boolean ? "" : "highp ") + t[2] + " u_var; } block[3];\n";
                    std::string cmp;
                    for (int i = 0; i < 3; ++i)
                        cmp += "    if (block[" + std::to_string(i) + "].u_var != " + t[2] + "(" + t[1] + ")) result = 0.5;\n";
                    if (std::strcmp(t[0], "mat4x3") == 0) {
                        decl = std::string("layout(") + layout + ") uniform Block { highp mat4x3 u_var; } block[3];\n";
                        cmp.clear();
                        for (int i = 0; i < 3; ++i)
                            cmp += "    if (block[" + std::to_string(i) + "].u_var[1] != vec3(0.0)) result = 0.5;\n";
                    } else if (boolean || std::strcmp(t[0], "uvec4") == 0 || std::strcmp(t[0], "vec3") == 0) {
                        cmp.clear();
                        for (int i = 0; i < 3; ++i)
                            cmp += "    if (any(notEqual(block[" + std::to_string(i) + "].u_var, " + t[2] + "(" + t[1] + ")))) result = 0.5;\n";
                    }
                    const std::string vs = "#version 300 es\nin highp vec4 a_position;\nout mediump float v_vtxResult;\n" +
                                           (stage != 1 ? decl : std::string()) + "void main() {\n    gl_Position = a_position;\n    mediump float result = 1.0;\n" +
                                           (stage != 1 ? cmp : std::string()) + "    v_vtxResult = result;\n}\n";
                    const std::string fs = "#version 300 es\nin mediump float v_vtxResult;\nlayout(location = 0) out mediump vec4 o;\n" +
                                           (stage != 0 ? decl : std::string()) + "void main() {\n    mediump float result = 1.0;\n" +
                                           (stage != 0 ? cmp : std::string()) + "    o = vec4(1.0, v_vtxResult, result, 1.0);\n}\n";
                    const GLuint pr = program(vs.c_str(), fs.c_str());
                    GLint nUniforms = 0, nBlocks = 0;
                    glGetProgramiv(pr, gl::ACTIVE_UNIFORMS, &nUniforms);
                    glGetProgramiv(pr, gl::ACTIVE_UNIFORM_BLOCKS, &nBlocks);
                    CHECK(nBlocks == 3 && nUniforms >= 1 && nUniforms <= 3);
                    std::vector<GLint> blockSize(static_cast<size_t>(nBlocks), 0);
                    for (GLint b = 0; b < nBlocks; ++b) {
                        GLint nameLen = 0, n = 0, size = 0;
                        glGetActiveUniformBlockiv(pr, GLuint(b), gl::UNIFORM_BLOCK_DATA_SIZE, &size);
                        glGetActiveUniformBlockiv(pr, GLuint(b), gl::UNIFORM_BLOCK_NAME_LENGTH, &nameLen);
                        glGetActiveUniformBlockiv(pr, GLuint(b), gl::UNIFORM_BLOCK_ACTIVE_UNIFORMS, &n);
                        std::vector<char> name(size_t(nameLen > 0 ? nameLen : 1));
                        glGetActiveUniformBlockName(pr, GLuint(b), GLsizei(name.size()), nullptr, name.data());
                        std::vector<GLint> idx(static_cast<size_t>(n), 0);
                        if (n > 0) glGetActiveUniformBlockiv(pr, GLuint(b), gl::UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, idx.data());
                        for (GLint i : idx) CHECK(i >= 0 && i < nUniforms);
                        CHECK(glGetUniformBlockIndex(pr, name.data()) == GLuint(b));
                        CHECK(size > 0 && n >= 1);
                        blockSize[size_t(b)] = size;
                    }
                    std::vector<GLuint> ids(static_cast<size_t>(nUniforms), 0);
                    for (GLint i = 0; i < nUniforms; ++i) ids[size_t(i)] = GLuint(i);
                    std::vector<GLint> bi(static_cast<size_t>(nUniforms), 0), nl(static_cast<size_t>(nUniforms), 0);
                    glGetActiveUniformsiv(pr, nUniforms, ids.data(), gl::UNIFORM_BLOCK_INDEX, bi.data());
                    glGetActiveUniformsiv(pr, nUniforms, ids.data(), gl::UNIFORM_NAME_LENGTH, nl.data());
                    for (GLint i = 0; i < nUniforms; ++i) {
                        CHECK(bi[size_t(i)] >= 0 && bi[size_t(i)] < nBlocks);
                        std::vector<char> nm(static_cast<size_t>(nl[size_t(i)]));
                        GLsizei len = 0;
                        GLint sz = 0;
                        GLenum ty = 0;
                        glGetActiveUniform(pr, GLuint(i), GLsizei(nm.size()), &len, &sz, &ty, nm.data());
                        CHECK(len + 1 == nl[size_t(i)]);
                    }
                    CHECK(err() == gl::NO_ERROR);
                    glUseProgram(pr);
                    for (GLint b = 0; b < nBlocks; ++b) glUniformBlockBinding(pr, GLuint(b), GLuint(b));
                    GLint align = 0;
                    glGetIntegerv(gl::UNIFORM_BUFFER_OFFSET_ALIGNMENT, &align);
                    if (perBlock) {
                        for (GLint b = 0; b < nBlocks; ++b) {
                            GLuint buf;
                            glGenBuffers(1, &buf);
                            std::vector<uint8_t> data(size_t(blockSize[size_t(b)]), 1);
                            glBindBuffer(gl::UNIFORM_BUFFER, buf);
                            glBufferData(gl::UNIFORM_BUFFER, GLsizeiptr(data.size()), data.data(), gl::STATIC_DRAW);
                            glBindBufferBase(gl::UNIFORM_BUFFER, GLuint(b), buf);
                        }
                    } else {
                        GLint cur = 0;
                        std::vector<GLint> offs;
                        for (GLint b = 0; b < nBlocks; ++b) {
                            if (align > 0) cur = (cur + align - 1) / align * align;
                            offs.push_back(cur);
                            cur += blockSize[size_t(b)];
                        }
                        std::vector<uint8_t> data(size_t(cur), 1);
                        GLuint buf;
                        glGenBuffers(1, &buf);
                        glBindBuffer(gl::UNIFORM_BUFFER, buf);
                        glBufferData(gl::UNIFORM_BUFFER, GLsizeiptr(data.size()), data.data(), gl::STATIC_DRAW);
                        for (GLint b = 0; b < nBlocks; ++b)
                            glBindBufferRange(gl::UNIFORM_BUFFER, GLuint(b), buf, offs[size_t(b)], blockSize[size_t(b)]);
                    }
                    CHECK(err() == gl::NO_ERROR);
                    glBindBuffer(gl::ARRAY_BUFFER, vb);
                    glVertexAttribPointer(0, 4, gl::FLOAT, 0, 16, nullptr);
                    glEnableVertexAttribArray(0);
                    glDrawArrays(gl::TRIANGLES, 0, 3);
                    CHECK(err() == gl::NO_ERROR);
                    settle();
                }
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
    // glClearBuffer*: buffer / drawbuffer are validated before the framebuffer is looked at (glClearBufferfi on COLOR crashed).
    G(glClearBufferfi, void (*)(GLenum, GLint, GLfloat, GLint));
    G(glClearBufferiv, void (*)(GLenum, GLint, const GLint*));
    glClearBufferfi(gl::COLOR, 0, 1.0f, 1);
    CHECK(err() == gl::INVALID_ENUM);
    glClearBufferfi(gl::DEPTH, 0, 1.0f, 1);
    CHECK(err() == gl::INVALID_ENUM);
    glClearBufferfi(gl::DEPTH_STENCIL, 1, 1.0f, 1);
    CHECK(err() == gl::INVALID_VALUE);
    const GLint zero4[4] = {};
    glClearBufferiv(gl::DEPTH, 0, zero4);
    CHECK(err() == gl::INVALID_ENUM);
    // Argument validation of the negative_api cases: negative counts, Is* after a first bind, transfer formats.
    G(glGenFramebuffers, void (*)(GLsizei, GLuint*));
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glIsBuffer, GLboolean (*)(GLuint));
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexImage2D, void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));
    G(glTexParameteri, void (*)(GLenum, GLenum, GLint));
    G(glMapBufferRange, void* (*)(GLenum, GLintptr, GLsizeiptr, GLbitfield));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    glGenFramebuffers(-1, nullptr);
    CHECK(err() == gl::INVALID_VALUE);
    GLuint bo = 0, tx = 0;
    glGenBuffers(1, &bo);
    CHECK(!glIsBuffer(bo));  // generated, not yet bound
    glBindBuffer(gl::ARRAY_BUFFER, bo);
    CHECK(glIsBuffer(bo));
    glBufferData(gl::ARRAY_BUFFER, 32, nullptr, gl::DYNAMIC_DRAW);
    glMapBufferRange(gl::ARRAY_BUFFER, 0, 16, gl::MAP_READ_BIT | 0x1000);
    CHECK(err() == gl::INVALID_VALUE);
    glMapBufferRange(gl::ARRAY_BUFFER, 0, 16, gl::MAP_READ_BIT | gl::MAP_UNSYNCHRONIZED_BIT);
    CHECK(err() == gl::INVALID_OPERATION);
    glGenTextures(1, &tx);
    glBindTexture(gl::TEXTURE_2D, tx);
    glTexImage2D(gl::TEXTURE_2D, 0, gl::RGB, 1, 1, 0, gl::RGBA, gl::UNSIGNED_BYTE, nullptr);
    CHECK(err() == gl::INVALID_OPERATION);
    glTexImage2D(gl::TEXTURE_2D, 0, gl::RGB, 1, 1, 0, gl::RGB, gl::UNSIGNED_SHORT_4_4_4_4, nullptr);
    CHECK(err() == gl::INVALID_OPERATION);
    glTexImage2D(0, 0, gl::RGB, 1, 1, 0, gl::RGB, gl::UNSIGNED_BYTE, nullptr);
    CHECK(err() == gl::INVALID_ENUM);
    glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, 0);
    CHECK(err() == gl::INVALID_ENUM);
    glTexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_LOD, -1000);  // not an enumerated parameter: any value (gluStateReset sets it)
    CHECK(err() == gl::NO_ERROR);
    glTexImage2D(gl::TEXTURE_2D, 0, gl::SRGB8, 1, 1, 0, 0x8C40 /* GL_SRGB_EXT */, gl::UNSIGNED_BYTE, nullptr);
    CHECK(err() == gl::NO_ERROR);
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

// ---- pipeline recipes across sessions -----------------------------------------------------------------------
// What a game's first frames ask of the layer: three programs (GLSL ES 1.00, ES 3.00, and one that a later session loads
// through glProgramBinary instead of compiling, like a UE4 program-binary cache), each drawn with three state combinations,
// a draw into an FBO, and a compute dispatch. Prints what the driver's recipe store says to stdout for the parent.
void testRecipeSession() {
    const char* dir = std::getenv("VP_PIPELINE_CACHE_DIR");
    CHECK(dir && *dir);
    const std::string binPath = std::string(dir ? dir : "") + "/prog.bin";
    Egl e;
    CHECK(e.up());
    G(glUseProgram, void (*)(GLuint));
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    G(glGetAttribLocation, GLint (*)(GLuint, const char*));
    G(glBindAttribLocation, void (*)(GLuint, GLuint, const char*));
    G(glVertexAttribPointer, void (*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
    G(glEnableVertexAttribArray, void (*)(GLuint));
    G(glVertexAttrib4f, void (*)(GLuint, GLfloat, GLfloat, GLfloat, GLfloat));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glEnable, void (*)(GLenum));
    G(glDisable, void (*)(GLenum));
    G(glBlendFunc, void (*)(GLenum, GLenum));
    G(glDepthFunc, void (*)(GLenum));
    G(glCullFace, void (*)(GLenum));
    G(glGenTextures, void (*)(GLsizei, GLuint*));
    G(glBindTexture, void (*)(GLenum, GLuint));
    G(glTexImage2D, void (*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));
    G(glTexStorage2D, void (*)(GLenum, GLsizei, GLenum, GLsizei, GLsizei));
    G(glGenFramebuffers, void (*)(GLsizei, GLuint*));
    G(glBindFramebuffer, void (*)(GLenum, GLuint));
    G(glFramebufferTexture2D, void (*)(GLenum, GLenum, GLenum, GLuint, GLint));
    G(glCreateShader, GLuint (*)(GLenum));
    G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
    G(glCompileShader, void (*)(GLuint));
    G(glCreateProgram, GLuint (*)());
    G(glAttachShader, void (*)(GLuint, GLuint));
    G(glLinkProgram, void (*)(GLuint));
    G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
    G(glProgramParameteri, void (*)(GLuint, GLenum, GLint));
    G(glGetProgramBinary, void (*)(GLuint, GLsizei, GLsizei*, GLenum*, void*));
    G(glProgramBinary, void (*)(GLuint, GLenum, const void*, GLsizei));
    G(glDispatchCompute, void (*)(GLuint, GLuint, GLuint));
    G(eglTerminate, uint32_t (*)(void*));

    static const char* kVs300 =
        "#version 300 es\nlayout(location = 0) in vec4 a_pos; layout(location = 1) in vec2 a_uv; out vec2 v_uv;\n"
        "void main() { v_uv = a_uv; gl_Position = a_pos; }\n";
    static const char* kFs300 =
        "#version 300 es\nprecision mediump float; in vec2 v_uv; uniform sampler2D u_tex; out vec4 o;\n"
        "void main() { o = texture(u_tex, v_uv); }\n";
    static const char* kCs = "#version 310 es\nlayout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;\nvoid main() { }\n";

    const GLuint pA = program(kVs100, kFs100), pB = program(kVs300, kFs300);
    // pC: compiled and saved by the first session, loaded from the saved binary by the next one.
    GLuint pC = glCreateProgram();
    bool fromBinary = false;
    {
        std::ifstream in(binPath, std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (blob.size() > 4) {
            GLenum fmt = 0;
            std::memcpy(&fmt, blob.data(), 4);
            glProgramBinary(pC, fmt, blob.data() + 4, GLsizei(blob.size() - 4));
            GLint ok = 0;
            glGetProgramiv(pC, gl::LINK_STATUS, &ok);
            CHECK(ok == 1);
            fromBinary = true;
        } else {
            const char* src[2] = {kVs100, kFs100};
            for (int i = 0; i < 2; ++i) {
                GLuint sh = glCreateShader(i ? gl::FRAGMENT_SHADER : gl::VERTEX_SHADER);
                glShaderSource(sh, 1, &src[i], nullptr);
                glCompileShader(sh);
                glAttachShader(pC, sh);
            }
            glBindAttribLocation(pC, 5, "a_uv");
            glBindAttribLocation(pC, 2, "a_pos");
            glProgramParameteri(pC, 0x8257 /*PROGRAM_BINARY_RETRIEVABLE_HINT*/, 1);
            glLinkProgram(pC);
            GLint ok = 0, len = 0;
            glGetProgramiv(pC, gl::LINK_STATUS, &ok);
            glGetProgramiv(pC, 0x8741 /*PROGRAM_BINARY_LENGTH*/, &len);
            CHECK(ok == 1 && len > 0);
            std::vector<char> out(4 + size_t(len > 0 ? len : 1));
            GLsizei got = 0;
            GLenum fmt = 0;
            glGetProgramBinary(pC, len, &got, &fmt, out.data() + 4);
            std::memcpy(out.data(), &fmt, 4);
            out.resize(4 + size_t(got));
            std::ofstream(binPath, std::ios::binary).write(out.data(), std::streamsize(out.size()));
        }
    }

    const float verts[] = {-1, -1, 0, 1, 0, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1};
    GLuint vb, t;
    glGenBuffers(1, &vb);
    glBindBuffer(gl::ARRAY_BUFFER, vb);
    glBufferData(gl::ARRAY_BUFFER, sizeof verts, verts, gl::STATIC_DRAW);
    glGenTextures(1, &t);
    glBindTexture(gl::TEXTURE_2D, t);
    const uint8_t white[4] = {255, 255, 255, 255};
    glTexImage2D(gl::TEXTURE_2D, 0, gl::RGBA, 1, 1, 0, gl::RGBA, gl::UNSIGNED_BYTE, white);

    auto attrib = [&](GLuint p, const char* name, GLint size, size_t offset) {
        const GLint loc = glGetAttribLocation(p, name);
        if (loc < 0) return;
        glVertexAttribPointer(GLuint(loc), size, gl::FLOAT, 0, 24, reinterpret_cast<const void*>(offset));
        glEnableVertexAttribArray(GLuint(loc));
    };
    auto states = [&](GLuint p) {
        glUseProgram(p);
        attrib(p, "a_pos", 4, 0);
        attrib(p, "a_uv", 2, 16);
        const GLint col = glGetAttribLocation(p, "a_col");
        if (col >= 0) glVertexAttrib4f(GLuint(col), 1, 1, 1, 1);
        glDisable(gl::BLEND), glDisable(gl::DEPTH_TEST), glDisable(gl::CULL_FACE);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        glEnable(gl::BLEND), glBlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        glEnable(gl::DEPTH_TEST), glDepthFunc(gl::LEQUAL);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        glDisable(gl::BLEND), glDisable(gl::DEPTH_TEST);
        glEnable(gl::CULL_FACE), glCullFace(gl::BACK);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        glDisable(gl::CULL_FACE);
    };
    counts = {};
    for (GLuint p : {pA, pB, pC}) states(p);
    {  // an off-screen pass: other attachment formats, so other pipelines for the same program
        GLuint fb, ft;
        glGenFramebuffers(1, &fb);
        glGenTextures(1, &ft);
        glBindTexture(gl::TEXTURE_2D, ft);
        glTexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8, 32, 32);
        glBindFramebuffer(gl::FRAMEBUFFER, fb);
        glFramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, ft, 0);
        glUseProgram(pA);
        glDrawArrays(gl::TRIANGLES, 0, 3);
        glBindFramebuffer(gl::FRAMEBUFFER, 0);
    }
    {
        GLuint sh = glCreateShader(gl::COMPUTE_SHADER);
        glShaderSource(sh, 1, &kCs, nullptr);
        glCompileShader(sh);
        GLuint cp = glCreateProgram();
        glAttachShader(cp, sh);
        glLinkProgram(cp);
        glUseProgram(cp);
        glDispatchCompute(2, 1, 1);
    }
    CHECK(err() == gl::NO_ERROR && counts.draws == 10 && counts.dispatches == 1);
    const int draws = counts.draws, pipelines = counts.pipelines;
    settle();
    vpgles_flush(1);
    eglTerminate(e.dpy);  // the closing gles.pso.recipe line
    e.down();
    // What the parent reads: the layer's own line (the last one) and what this session did.
    const std::string log = peekStderr();
    const size_t at = log.rfind("gles.pso.recipe ");
    std::printf("RECIPE %s\n", at == std::string::npos ? "none" : log.substr(at, log.find('\n', at) - at).c_str());
    std::printf("SESSION draws=%d pipelines=%d from_binary=%d\n", draws, pipelines, fromBinary ? 1 : 0);
}

// Two sessions of the same rendering with one VP_PIPELINE_CACHE_DIR (children of this binary, each a fresh process: a new
// device, new GL names, a different address space): the first records a recipe for every pipeline it creates, the second
// finds them all stored, has them compiled by the prewarm before its first draw and creates nothing the store lacked.
void testRecipe() {
    char tmpl[] = "/tmp/vpgles_recipe_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    CHECK(dir != nullptr);
    if (!dir) return;
    struct Run {
        bool ok = false;
        unsigned saved = 0, loaded = 0, prewarmed = 0, miss = 0;
        int draws = 0, pipelines = 0, fromBinary = 0;
        std::string text;
    };
    auto session = [&]() {
        Run r;
        // VPVK_GLES_SYNC_PIPELINES=1: the prewarm must have finished (prewarmed == loaded) when the session reads the counters
        const std::string cmd = std::string("VPVK_GLES_SYNC_PIPELINES=1 VP_PIPELINE_CACHE_DIR='") + dir + "' '" + gSelf + "' recipe_session '" + gLibPath + "' 2>&1";
        FILE* p = popen(cmd.c_str(), "r");
        CHECK(p != nullptr);
        if (!p) return r;
        char buf[512];
        while (std::fgets(buf, sizeof buf, p)) r.text += buf;
        const int status = pclose(p);
        r.ok = status == 0;
        const size_t a = r.text.find("RECIPE gles.pso.recipe ");
        if (a != std::string::npos)
            std::sscanf(r.text.c_str() + a, "RECIPE gles.pso.recipe saved=%u loaded=%u prewarmed=%u miss_in_game=%u", &r.saved, &r.loaded,
                        &r.prewarmed, &r.miss);
        const size_t b = r.text.find("SESSION ");
        if (b != std::string::npos) std::sscanf(r.text.c_str() + b, "SESSION draws=%d pipelines=%d from_binary=%d", &r.draws, &r.pipelines, &r.fromBinary);
        if (!r.ok || a == std::string::npos) std::printf("  session output:\n%s\n", r.text.c_str());
        return r;
    };
    const Run s1 = session(), s2 = session();
    CHECK(s1.ok && s2.ok);
    // First session: nothing stored, so every pipeline the game created was new and left a recipe.
    CHECK(s1.loaded == 0 && s1.prewarmed == 0 && s1.saved > 0 && s1.miss == s1.saved && !s1.fromBinary);
    CHECK(s1.draws == 10 && s1.pipelines >= 9);
    // Second session: all of them loaded, all prewarmed, none created that was missing, nothing new to save (pC came from the
    // program binary and still produced the very same recipes).
    CHECK(s2.loaded == s1.saved && s2.prewarmed == s2.loaded && s2.saved == 0 && s2.miss == 0 && s2.fromBinary);
    CHECK(s2.draws == s1.draws && s2.pipelines == s1.pipelines);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}


// Transform feedback: the pure assembly / copy helpers, then the GL state machine and its errors (the null backend runs no
// shader, so the captured values themselves are checked in the vpglsl "xfb" test and on Metal by dEQP).
void testXfb() {
    {
        std::vector<uint32_t> out;
        const uint32_t v[5] = {0, 1, 2, 3, 4};
        xfbAssemble(5, v, 5, false, 0, out);  // triangle strip: odd triangles swap the first two
        CHECK((out == std::vector<uint32_t>{0, 1, 2, 2, 1, 3, 2, 3, 4}));
        out.clear();
        const uint32_t r[5] = {0, 1, 2, 0xFFFFFFFFu, 3};
        xfbAssemble(4, r, 5, true, 0xFFFFFFFFu, out);  // restart splits the sequence; the lone vertex makes nothing
        CHECK((out == std::vector<uint32_t>{0, 1, 2}));
        out.clear();
        xfbAssemble(2, v, 3, false, 0, out);  // line loop closes
        CHECK((out == std::vector<uint32_t>{0, 1, 1, 2, 2, 0}));
        // Two records of 3 words; binding A takes words 0..1, binding B word 2; B only has room for one vertex.
        const uint32_t rec[6] = {10, 11, 12, 20, 21, 22};
        uint32_t a[4] = {}, b[2] = {};
        std::vector<XfbOut> outs(2);
        outs[0] = {reinterpret_cast<uint8_t*>(a), sizeof a, 0, 8, {{0, 2}}};
        outs[1] = {reinterpret_cast<uint8_t*>(b), 4, 0, 4, {{2, 1}}};
        CHECK(xfbWrite({0, 1}, 1, rec, 3, outs) == 1);  // the second point does not fit B, so it is written to neither
        CHECK(a[0] == 10 && a[1] == 11 && a[2] == 0 && b[0] == 12 && outs[0].used == 8 && outs[1].used == 4);
    }
    Egl e;
    CHECK(e.up());
    G(glGenBuffers, void (*)(GLsizei, GLuint*));
    G(glBindBuffer, void (*)(GLenum, GLuint));
    G(glBufferData, void (*)(GLenum, GLsizeiptr, const void*, GLenum));
    G(glBindBufferBase, void (*)(GLenum, GLuint, GLuint));
    G(glBindBufferRange, void (*)(GLenum, GLuint, GLuint, GLintptr, GLsizeiptr));
    G(glUseProgram, void (*)(GLuint));
    G(glDrawArrays, void (*)(GLenum, GLint, GLsizei));
    G(glDrawArraysInstanced, void (*)(GLenum, GLint, GLsizei, GLsizei));
    G(glGetIntegerv, void (*)(GLenum, GLint*));
    G(glGetIntegeri_v, void (*)(GLenum, GLuint, GLint*));
    G(glBeginTransformFeedback, void (*)(GLenum));
    G(glEndTransformFeedback, void (*)());
    G(glPauseTransformFeedback, void (*)());
    G(glResumeTransformFeedback, void (*)());
    G(glGenTransformFeedbacks, void (*)(GLsizei, GLuint*));
    G(glBindTransformFeedback, void (*)(GLenum, GLuint));
    G(glDeleteTransformFeedbacks, void (*)(GLsizei, const GLuint*));
    G(glIsTransformFeedback, GLboolean (*)(GLuint));
    G(glTransformFeedbackVaryings, void (*)(GLuint, GLsizei, const char* const*, GLenum));
    G(glGetTransformFeedbackVarying, void (*)(GLuint, GLuint, GLsizei, GLsizei*, GLsizei*, GLenum*, char*));
    G(glGetProgramiv, void (*)(GLuint, GLenum, GLint*));
    G(glLinkProgram, void (*)(GLuint));
    G(glCreateShader, GLuint (*)(GLenum));
    G(glShaderSource, void (*)(GLuint, GLsizei, const char* const*, const GLint*));
    G(glCompileShader, void (*)(GLuint));
    G(glCreateProgram, GLuint (*)());
    G(glAttachShader, void (*)(GLuint, GLuint));
    const char* vs =
        "#version 300 es\nin vec4 a; out vec3 v; out float w;\nvoid main() { v = a.xyz; w = a.w; gl_Position = a; }\n";
    const char* fs = "#version 300 es\nprecision mediump float; in vec3 v; in float w; out vec4 o;\nvoid main() { o = vec4(v, w); }\n";
    GLuint sh[2] = {glCreateShader(gl::VERTEX_SHADER), glCreateShader(gl::FRAGMENT_SHADER)};
    const char* src[2] = {vs, fs};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; ++i) {
        glShaderSource(sh[i], 1, &src[i], nullptr);
        glCompileShader(sh[i]);
        glAttachShader(p, sh[i]);
    }
    const char* names[2] = {"v", "w"};
    glTransformFeedbackVaryings(p, 2, names, 0x8C8D);  // separate
    CHECK(err() == gl::NO_ERROR);
    glTransformFeedbackVaryings(p, 2, names, gl::TRIANGLES);
    CHECK(err() == gl::INVALID_ENUM);
    glTransformFeedbackVaryings(p, -1, names, 0x8C8C);
    CHECK(err() == gl::INVALID_VALUE);
    glTransformFeedbackVaryings(p, 2, names, 0x8C8D);
    glLinkProgram(p);
    GLint v = -1, mode = 0, maxLen = 0;
    glGetProgramiv(p, gl::LINK_STATUS, &v);
    CHECK(v == 1);
    glGetProgramiv(p, gl::TRANSFORM_FEEDBACK_VARYINGS, &v);
    glGetProgramiv(p, gl::TRANSFORM_FEEDBACK_BUFFER_MODE, &mode);
    glGetProgramiv(p, gl::TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &maxLen);
    CHECK(v == 2 && mode == 0x8C8D && maxLen == 2);
    char name[8];
    GLsizei len = 0, size = 0;
    GLenum type = 0;
    glGetTransformFeedbackVarying(p, 1, sizeof name, &len, &size, &type, name);
    CHECK(err() == gl::NO_ERROR && len == 1 && size == 1 && type == 0x1406 && std::string(name) == "w");
    glGetTransformFeedbackVarying(p, 2, sizeof name, &len, &size, &type, name);
    CHECK(err() == gl::INVALID_VALUE);

    // Begin needs a program, a buffer on every binding the varyings use, and a valid primitive mode.
    glBeginTransformFeedback(gl::POINTS);
    CHECK(err() == gl::INVALID_OPERATION);  // no program in use
    glUseProgram(p);
    glBeginTransformFeedback(gl::TRIANGLE_STRIP);
    CHECK(err() == gl::INVALID_ENUM);
    glBeginTransformFeedback(gl::POINTS);
    CHECK(err() == gl::INVALID_OPERATION);  // no buffers bound
    GLuint buf[2];
    glGenBuffers(2, buf);
    for (GLuint b : buf) {
        glBindBuffer(gl::ARRAY_BUFFER, b);
        glBufferData(gl::ARRAY_BUFFER, 256, nullptr, gl::DYNAMIC_DRAW);
    }
    glBindBufferBase(gl::TRANSFORM_FEEDBACK_BUFFER, 0, buf[0]);
    glBeginTransformFeedback(gl::POINTS);
    CHECK(err() == gl::INVALID_OPERATION);  // separate mode with two varyings needs binding 1 too
    glBindBufferRange(gl::TRANSFORM_FEEDBACK_BUFFER, 1, buf[1], 3, 16);
    CHECK(err() == gl::INVALID_VALUE);  // offset not a multiple of 4
    glBindBufferRange(gl::TRANSFORM_FEEDBACK_BUFFER, 1, buf[1], 4, 14);
    CHECK(err() == gl::INVALID_VALUE);  // size not a multiple of 4
    glBindBufferRange(gl::TRANSFORM_FEEDBACK_BUFFER, 1, buf[1], 16, 64);
    CHECK(err() == gl::NO_ERROR);
    glGetIntegeri_v(0x8C84, 1, &v);  // TRANSFORM_FEEDBACK_BUFFER_START
    CHECK(v == 16);
    glGetIntegeri_v(gl::TRANSFORM_FEEDBACK_BUFFER_BINDING, 1, &v);
    CHECK(GLuint(v) == buf[1]);
    glEndTransformFeedback();
    CHECK(err() == gl::INVALID_OPERATION);  // not active
    glPauseTransformFeedback();
    CHECK(err() == gl::INVALID_OPERATION);
    glBeginTransformFeedback(gl::POINTS);
    CHECK(err() == gl::NO_ERROR);
    glGetIntegerv(0x8E24, &v);
    CHECK(v == 1);
    glBeginTransformFeedback(gl::POINTS);
    CHECK(err() == gl::INVALID_OPERATION);  // already active
    glBindBufferBase(gl::TRANSFORM_FEEDBACK_BUFFER, 0, buf[1]);
    CHECK(err() == gl::INVALID_OPERATION);  // bindings are fixed while capturing
    glUseProgram(0);
    CHECK(err() == gl::INVALID_OPERATION);
    glLinkProgram(p);
    CHECK(err() == gl::INVALID_OPERATION);
    glDrawArrays(gl::TRIANGLES, 0, 3);
    CHECK(err() == gl::INVALID_OPERATION);  // the draw mode must match Begin's primitive type
    glResumeTransformFeedback();
    CHECK(err() == gl::INVALID_OPERATION);  // not paused
    glPauseTransformFeedback();
    CHECK(err() == gl::NO_ERROR);
    glGetIntegerv(0x8E23, &v);
    CHECK(v == 1);
    glPauseTransformFeedback();
    CHECK(err() == gl::INVALID_OPERATION);
    glResumeTransformFeedback();
    CHECK(err() == gl::NO_ERROR);
    glEndTransformFeedback();
    CHECK(err() == gl::NO_ERROR);
    glGetIntegerv(0x8E24, &v);
    CHECK(v == 0);

    // Objects: the default object is always there; others exist once generated, own their bindings and cannot be deleted while active.
    GLuint t[2];
    glGenTransformFeedbacks(2, t);
    CHECK(t[0] != 0 && t[0] != t[1] && !glIsTransformFeedback(t[0]));
    glBindTransformFeedback(gl::TRANSFORM_FEEDBACK, t[0]);
    CHECK(err() == gl::NO_ERROR && glIsTransformFeedback(t[0]));
    glGetIntegerv(0x8E25, &v);
    CHECK(GLuint(v) == t[0]);
    glGetIntegeri_v(gl::TRANSFORM_FEEDBACK_BUFFER_BINDING, 0, &v);
    CHECK(v == 0);  // its own bindings, not the default object's
    glBindBufferBase(gl::TRANSFORM_FEEDBACK_BUFFER, 0, buf[0]);
    glBindBufferBase(gl::TRANSFORM_FEEDBACK_BUFFER, 1, buf[1]);
    G(glGenQueries, void (*)(GLsizei, GLuint*));
    G(glBeginQuery, void (*)(GLenum, GLuint));
    G(glEndQuery, void (*)(GLenum));
    G(glGetQueryObjectuiv, void (*)(GLuint, GLenum, GLuint*));
    GLuint q = 0, written = 0;
    glGenQueries(1, &q);
    glBeginQuery(gl::TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(gl::POINTS);
    CHECK(err() == gl::NO_ERROR);
    glDrawArrays(gl::POINTS, 0, 3);
    glDrawArraysInstanced(gl::POINTS, 0, 100, 1);  // the 12-byte varying has room for 21 vertices in all
    CHECK(err() == gl::NO_ERROR);
    glEndQuery(gl::TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectuiv(q, 0x8866 /*QUERY_RESULT*/, &written);
    CHECK(written == 21);
    glBindTransformFeedback(gl::TRANSFORM_FEEDBACK, 0);
    CHECK(err() == gl::INVALID_OPERATION);  // active and not paused
    glDeleteTransformFeedbacks(1, &t[0]);
    CHECK(err() == gl::INVALID_OPERATION);
    glEndTransformFeedback();
    glBindTransformFeedback(gl::TRANSFORM_FEEDBACK, 12345);
    CHECK(err() == gl::INVALID_OPERATION);  // never generated
    glBindTransformFeedback(gl::ARRAY_BUFFER, 0);
    CHECK(err() == gl::INVALID_ENUM);
    glDeleteTransformFeedbacks(1, &t[0]);
    CHECK(err() == gl::NO_ERROR && !glIsTransformFeedback(t[0]));
    glGetIntegerv(0x8E25, &v);
    CHECK(v == 0);  // deleting the bound object rebinds the default one
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    gSelf = argv[0];
    gLibPath = argv[2];
    void* lib = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    gLib = lib;
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
        {"fbo", testFbo}, {"program", testProgram}, {"draw", testDraw}, {"invalidate", testInvalidate}, {"multiview", testMultiview},
        {"errors", testErrors}, {"ubo", testUbo}, {"readback", testReadback}, {"external", testExternal}, {"es31", testEs31},
        {"progbin", testProgramBinary}, {"recipe", testRecipe}, {"recipe_session", testRecipeSession}, {"xfb", testXfb},
    };
    auto it = tests.find(argv[1]);
    if (ok && it != tests.end()) it->second();
    const std::string log = releaseStderr();
    CHECK(ok);
    if (expectLinkFailedVaryings) CHECK(log.find("link_failed_varyings") != std::string::npos);
    if (expectMipBeforeBase) CHECK(log.find("gles.note what=mip_before_base") != std::string::npos);
    if (expectTexStats) CHECK(log.find("gles.tex fmt=0x8058 count=") != std::string::npos);  // RGBA8
    for (const std::string& want : expectLines) {
        if (log.find(want) == std::string::npos) {
            std::printf("  FAIL driver log lacks '%s'\n", want.c_str());
            ++failures;
        }
    }
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
