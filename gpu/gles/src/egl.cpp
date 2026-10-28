// EGL 1.5 subset Android GLES games use: one display, configs, window/pbuffer surfaces, ES 2/3
// contexts with share groups, current-context binding, swap, EGLImage (texture sources), KHR fence
// sync. The C entry points are exported through vpgles_proc (no symbol clashes with a host EGL).
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "egl_internal.h"
#include "vpgles/vpgles.h"

namespace vpgles {

using namespace egl;

namespace {

struct Display {
    bool initialized = false;
};
Display gDisplay;
void* const kDisplay = &gDisplay;

thread_local EGLint tError = SUCCESS;
thread_local EGLenum tApi = OPENGL_ES_API;
thread_local Surface* tDraw = nullptr;
thread_local Surface* tRead = nullptr;

vpgles_present_fn gPresent = nullptr;
void* gPresentCtx = nullptr;
vpgles_window_size_fn gWindowSize = nullptr;
void* gWindowSizeCtx = nullptr;

// RGBA8888/D24S8 first (what Unity/Unreal ask for), then without depth, RGB565, and 4x variants
// (rendered in a 4-sample colour + depth, resolved into the presented image).
const Config kConfigs[] = {
    {1, 8, 8, 8, 8, 24, 8, 0, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D24_UNORM_S8_UINT},
    {2, 8, 8, 8, 8, 0, 0, 0, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED},
    {3, 8, 8, 8, 0, 24, 8, 0, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D24_UNORM_S8_UINT},
    {4, 8, 8, 8, 0, 0, 0, 0, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED},
    {5, 5, 6, 5, 0, 16, 0, 0, VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_D16_UNORM},
    {6, 5, 6, 5, 0, 0, 0, 0, VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_UNDEFINED},
    {7, 8, 8, 8, 8, 24, 8, 4, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D24_UNORM_S8_UINT},
    {8, 8, 8, 8, 8, 16, 0, 0, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D16_UNORM},
    {9, 10, 10, 10, 2, 24, 8, 0, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_D24_UNORM_S8_UINT},
    {10, 16, 16, 16, 16, 24, 8, 0, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT},
};

const Config* configOf(void* c) {
    for (const Config& k : kConfigs)
        if (&k == c) return &k;
    return nullptr;
}

EGLint attrOf(const Config& c, EGLint a) {
    switch (a) {
        case BUFFER_SIZE: return c.r + c.g + c.b + c.a;
        case RED_SIZE: return c.r;
        case GREEN_SIZE: return c.g;
        case BLUE_SIZE: return c.b;
        case ALPHA_SIZE: return c.a;
        case DEPTH_SIZE: return c.depth;
        case STENCIL_SIZE: return c.stencil;
        case SAMPLES: return c.samples;
        case SAMPLE_BUFFERS: return c.samples ? 1 : 0;
        case CONFIG_ID: return c.id;
        case CONFIG_CAVEAT: return NONE;
        case LEVEL: return 0;
        case MAX_PBUFFER_WIDTH: case MAX_PBUFFER_HEIGHT: return 16384;
        case MAX_PBUFFER_PIXELS: return 16384 * 16384;
        case NATIVE_RENDERABLE: return FALSE_;
        case NATIVE_VISUAL_ID: return c.r == 5 ? 4 /* RGB_565 */ : c.a ? 1 /* RGBA_8888 */ : 2 /* RGBX_8888 */;
        case NATIVE_VISUAL_TYPE: return 0;
        case SURFACE_TYPE: return WINDOW_BIT | PBUFFER_BIT;
        case TRANSPARENT_TYPE: return NONE;
        case TRANSPARENT_RED_VALUE: case TRANSPARENT_GREEN_VALUE: case TRANSPARENT_BLUE_VALUE: return 0;
        case BIND_TO_TEXTURE_RGB: case BIND_TO_TEXTURE_RGBA: return FALSE_;
        case MIN_SWAP_INTERVAL: return 0;
        case MAX_SWAP_INTERVAL: return 1;
        case LUMINANCE_SIZE: case ALPHA_MASK_SIZE: return 0;
        case COLOR_BUFFER_TYPE: return RGB_BUFFER;
        case RENDERABLE_TYPE: case CONFORMANT: return OPENGL_ES2_BIT | OPENGL_ES3_BIT;
        case RECORDABLE_ANDROID: case FRAMEBUFFER_TARGET_ANDROID: return TRUE_;
        default: return -1;
    }
}

bool err(EGLint e) {
    tError = e;
    return false;
}

std::mutex gEglLock;

}  // namespace

bool createSurfaceImages(Device* dev, Surface& s) {
    const uint32_t n = s.config->samples > 1 ? renderSamples(uint32_t(s.config->samples)) : 1;
    auto make = [&](VkFormat f, bool depth, uint32_t samples) -> ImageP {
        const VkImageUsageFlags usage =
            (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT) |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        return createImage2D(dev, formatByVk(f), s.width, s.height, 1, samples, usage);
    };
    s.color = make(s.config->color, false, 1);
    if (!s.color) return false;
    if (n > 1 && !(s.msColor = make(s.config->color, false, n))) return false;
    if (s.config->ds != VK_FORMAT_UNDEFINED) s.depthStencil = make(s.config->ds, true, n);
    return true;
}

// A surface the game presents before it wrote all of it must show black, not the uninitialised memory Metal hands out
// (magenta): its images are zeroed once, when created (no per-frame cost). VPVulkan sees a full write and drops them from
// its lazy zero-fill set. GL textures need nothing here: that lazy set already zero-fills what is read before written.
static void zeroSurfaceImage(Context& c, const Image& img) {
    const VkImageSubresourceRange r{img.aspect(), 0, 1, 0, 1};
    if (img.format->depth || img.format->stencil) {
        const VkClearDepthStencilValue z{0.0f, 0};
        c.dev->vk.vkCmdClearDepthStencilImage(c.cmd(), img.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
    } else {
        const VkClearColorValue z{};
        c.dev->vk.vkCmdClearColorImage(c.cmd(), img.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
    }
}

// ---- entry points -------------------------------------------------------------------------------------
using EGLDisplay = void*;
using EGLConfig = void*;
using EGLSurface = void*;
using EGLContext = void*;
using EGLImage = void*;
using EGLSync = void*;

// VPGLES_CRASH_TRACE=1: a fatal signal prints the stack (module + offset of every frame, for llvm-symbolizer on the dylib) to
// stderr, then goes on to the handler that was installed before (the harness's own crash report).
static struct sigaction gPrevSegv, gPrevBus, gPrevAbrt, gPrevIll, gPrevFpe;
static void crashTrace(int sig, siginfo_t* info, void* uc) {
    void* frames[48];
    const int n = backtrace(frames, 48);
    char line[512];
    int len = std::snprintf(line, sizeof line, "gles.crash sig=%d addr=%p frames=%d\n", sig, info ? info->si_addr : nullptr, n);
    if (len > 0) (void)!write(2, line, size_t(len));
    for (int i = 0; i < n; ++i) {
        Dl_info di{};
        const bool ok = dladdr(frames[i], &di) && di.dli_fbase;
        len = std::snprintf(line, sizeof line, "gles.crash frame=%d %s+0x%lx sym=%s\n", i, ok && di.dli_fname ? di.dli_fname : "?",
                            ok ? static_cast<unsigned long>(reinterpret_cast<char*>(frames[i]) - reinterpret_cast<char*>(di.dli_fbase)) : 0ul,
                            ok && di.dli_sname ? di.dli_sname : "-");
        if (len > 0) (void)!write(2, line, size_t(len));
    }
    const struct sigaction* prev = sig == SIGSEGV ? &gPrevSegv : sig == SIGBUS ? &gPrevBus : sig == SIGABRT ? &gPrevAbrt
                                   : sig == SIGILL ? &gPrevIll : &gPrevFpe;
    if (prev->sa_flags & SA_SIGINFO) {
        if (prev->sa_sigaction) return prev->sa_sigaction(sig, info, uc);
    } else if (prev->sa_handler != SIG_DFL && prev->sa_handler != SIG_IGN) {
        return prev->sa_handler(sig);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
static void installCrashTrace() {
    const char* e = std::getenv("VPGLES_CRASH_TRACE");
    if (!e || e[0] != '1') return;
    struct Slot { int sig; struct sigaction* prev; };
    for (const Slot& s : {Slot{SIGSEGV, &gPrevSegv}, Slot{SIGBUS, &gPrevBus}, Slot{SIGABRT, &gPrevAbrt}, Slot{SIGILL, &gPrevIll}, Slot{SIGFPE, &gPrevFpe}}) {
        struct sigaction cur{};
        if (sigaction(s.sig, nullptr, &cur) == 0 && cur.sa_sigaction == crashTrace && (cur.sa_flags & SA_SIGINFO)) continue;  // already ours
        struct sigaction mine{};
        mine.sa_sigaction = crashTrace;
        mine.sa_flags = SA_SIGINFO | SA_NODEFER;
        sigemptyset(&mine.sa_mask);
        sigaction(s.sig, &mine, s.prev);
    }
}

static EGLDisplay GetDisplay(void*) { return kDisplay; }
static EGLDisplay GetPlatformDisplay(EGLenum, void*, const intptr_t*) { return kDisplay; }

static EGLBoolean Initialize(EGLDisplay d, EGLint* major, EGLint* minor) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (!device()) return err(NOT_INITIALIZED);
    gDisplay.initialized = true;
    installCrashTrace();
    if (major) *major = 1;
    if (minor) *minor = 5;
    tError = SUCCESS;
    return TRUE_;
}
static EGLBoolean Terminate(EGLDisplay d) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (Device* dv = device()) {
        std::lock_guard<std::recursive_mutex> l(dv->lock);
        texStatsTick(true);
        recipeStatsTick(true);
    }
    return TRUE_;
}
static const char* QueryString(EGLDisplay d, EGLint name) {
    if (d != kDisplay && !(d == nullptr && name == EXTENSIONS)) return err(BAD_DISPLAY), nullptr;
    switch (name) {
        case VENDOR: return "VPKEngine";
        case VERSION: return "1.5 VPKEngine";
        case CLIENT_APIS: return "OpenGL_ES";
        case EXTENSIONS:
            return d ? "EGL_KHR_image_base EGL_KHR_gl_texture_2D_image EGL_KHR_fence_sync EGL_KHR_wait_sync "
                       "EGL_KHR_surfaceless_context EGL_KHR_create_context EGL_KHR_no_config_context "
                       "EGL_ANDROID_image_native_buffer EGL_ANDROID_recordable EGL_ANDROID_framebuffer_target "
                       "EGL_EXT_protected_content EGL_KHR_gl_colorspace"
                     : "EGL_EXT_client_extensions EGL_KHR_platform_android";
        default: return err(BAD_PARAMETER), nullptr;
    }
}

static bool matches(const Config& c, const EGLint* attribs) {
    if (!attribs) return true;
    for (const EGLint* a = attribs; a[0] != NONE; a += 2) {
        const EGLint k = a[0], v = a[1];
        if (v == DONT_CARE) continue;
        switch (k) {
            case RED_SIZE: case GREEN_SIZE: case BLUE_SIZE: case ALPHA_SIZE: case DEPTH_SIZE: case STENCIL_SIZE:
            case BUFFER_SIZE: case SAMPLES: case SAMPLE_BUFFERS:
                if (attrOf(c, k) < v) return false;
                break;
            case RENDERABLE_TYPE: case SURFACE_TYPE: case CONFORMANT:
                if ((attrOf(c, k) & v) != v) return false;
                break;
            case CONFIG_ID:
                if (c.id != v) return false;
                break;
            case COLOR_BUFFER_TYPE:
                if (v != RGB_BUFFER) return false;
                break;
            case NATIVE_VISUAL_ID:
                if (attrOf(c, k) != v) return false;
                break;
            default: break;  // CONFIG_CAVEAT, LEVEL, RECORDABLE_ANDROID...: every config qualifies
        }
    }
    return true;
}

static EGLBoolean ChooseConfig(EGLDisplay d, const EGLint* attribs, EGLConfig* configs, EGLint size, EGLint* num) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (!num) return err(BAD_PARAMETER);
    std::vector<const Config*> m;
    for (const Config& c : kConfigs)
        if (matches(c, attribs)) m.push_back(&c);
    // EGL sort: fewer samples, then smaller buffer size beyond the request... keep table order
    // (already the preferred order) but put exact non-MSAA first.
    std::stable_sort(m.begin(), m.end(), [](const Config* a, const Config* b) { return a->samples < b->samples; });
    EGLint n = EGLint(m.size());
    if (configs) {
        n = std::min(n, size);
        for (EGLint i = 0; i < n; ++i) configs[i] = const_cast<Config*>(m[size_t(i)]);
    }
    *num = n;
    return TRUE_;
}
static EGLBoolean GetConfigs(EGLDisplay d, EGLConfig* configs, EGLint size, EGLint* num) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (!num) return err(BAD_PARAMETER);
    EGLint n = EGLint(sizeof kConfigs / sizeof kConfigs[0]);
    if (configs) {
        n = std::min(n, size);
        for (EGLint i = 0; i < n; ++i) configs[i] = const_cast<Config*>(&kConfigs[i]);
    }
    *num = n;
    return TRUE_;
}
static EGLBoolean GetConfigAttrib(EGLDisplay d, EGLConfig c, EGLint attr, EGLint* value) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    const Config* k = configOf(c);
    if (!k) return err(BAD_CONFIG);
    EGLint v = attrOf(*k, attr);
    if (v == -1) return err(BAD_ATTRIBUTE);
    *value = v;
    return TRUE_;
}

static EGLSurface CreateSurface(EGLConfig c, Surface::Kind kind, void* window, const EGLint* attribs) {
    const Config* k = configOf(c);
    if (!k) return err(BAD_CONFIG), nullptr;
    auto* s = new Surface;
    s->kind = kind;
    s->config = k;
    s->window = window;
    if (attribs)
        for (const EGLint* a = attribs; a[0] != NONE; a += 2) {
            if ((a[0] == WIDTH || a[0] == HEIGHT) && (a[1] < 0 || a[1] > 16384)) {  // EGL_BAD_PARAMETER, not a 4G-pixel image
                delete s;
                return err(BAD_PARAMETER), nullptr;
            }
            if (a[0] == WIDTH) s->width = uint32_t(a[1]);
            if (a[0] == HEIGHT) s->height = uint32_t(a[1]);
        }
    if (kind == Surface::Window) {
        uint32_t w = 0, h = 0;
        if (gWindowSize && gWindowSize(gWindowSizeCtx, window, &w, &h)) s->width = w, s->height = h;
        if (!s->width) s->width = 1920, s->height = 1080;
    }
    if (!s->width) s->width = 1;
    if (!s->height) s->height = 1;
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    if (!createSurfaceImages(device(), *s)) {
        delete s;
        return err(BAD_ALLOC), nullptr;
    }
    // Images start in GENERAL with a first submission (no context may exist yet).
    Context tmp(device(), new ShareGroup, 3, 0);
    tmp.imageInit(*s->color);
    zeroSurfaceImage(tmp, *s->color);
    if (s->msColor) tmp.imageInit(*s->msColor);  // read before written: VPVulkan's lazy zero-fill
    if (s->depthStencil) {
        tmp.imageInit(*s->depthStencil);
        if (s->depthStencil->samples == VK_SAMPLE_COUNT_1_BIT) zeroSurfaceImage(tmp, *s->depthStencil);
    }
    tmp.flush(true);
    return s;
}
static EGLSurface CreateWindowSurface(EGLDisplay d, EGLConfig c, void* window, const EGLint* attribs) {
    if (d != kDisplay) return err(BAD_DISPLAY), nullptr;
    return CreateSurface(c, Surface::Window, window, attribs);
}
static EGLSurface CreatePlatformWindowSurface(EGLDisplay d, EGLConfig c, void* window, const intptr_t* attribs) {
    std::vector<EGLint> a;
    if (attribs)
        for (const intptr_t* p = attribs; *p != NONE; p += 2) a.push_back(EGLint(p[0])), a.push_back(EGLint(p[1]));
    a.push_back(NONE);
    return CreateWindowSurface(d, c, window, a.data());
}
static EGLSurface CreatePbufferSurface(EGLDisplay d, EGLConfig c, const EGLint* attribs) {
    if (d != kDisplay) return err(BAD_DISPLAY), nullptr;
    return CreateSurface(c, Surface::Pbuffer, nullptr, attribs);
}
// Libera la superficie tras esperar a la GPU (Device::lock tomado).
static void freeSurface(Surface* s) {
    Context* c = current();
    if (c) c->flush(true);
    else device()->waitSerial(device()->submitted);
    delete s;
}
// Un hilo deja de tener `s` como draw/read; si estaba pendiente de destruir y ya nadie la usa, se borra.
static void unbindSurface(Surface* s) {
    if (s && --s->bound == 0 && s->dead) freeSurface(s);
}
static EGLBoolean DestroySurface(EGLDisplay d, EGLSurface s) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (!s) return err(BAD_SURFACE);
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    auto* surf = static_cast<Surface*>(s);
    // EGL 1.5: una superficie actual en algún hilo se destruye cuando deja de serlo (unbindSurface).
    if (surf->bound > 0) {
        surf->dead = true;
        return TRUE_;
    }
    freeSurface(surf);
    return TRUE_;
}
static EGLBoolean QuerySurface(EGLDisplay d, EGLSurface sv, EGLint attr, EGLint* value) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    auto* s = static_cast<Surface*>(sv);
    if (!s) return err(BAD_SURFACE);
    switch (attr) {
        case WIDTH: *value = EGLint(s->width); return TRUE_;
        case HEIGHT: *value = EGLint(s->height); return TRUE_;
        case CONFIG_ID: *value = s->config->id; return TRUE_;
        case RENDER_BUFFER: *value = s->kind == Surface::Window ? BACK_BUFFER : BACK_BUFFER; return TRUE_;
        case SWAP_BEHAVIOR: *value = BUFFER_DESTROYED; return TRUE_;
        case MULTISAMPLE_RESOLVE: *value = 0x309A; return TRUE_;
        case HORIZONTAL_RESOLUTION: case VERTICAL_RESOLUTION: case PIXEL_ASPECT_RATIO: *value = -1; return TRUE_;
        case LARGEST_PBUFFER: case MIPMAP_TEXTURE: *value = 0; return TRUE_;
        case TEXTURE_FORMAT: case TEXTURE_TARGET: *value = NO_TEXTURE; return TRUE_;
        case GL_COLORSPACE: *value = GL_COLORSPACE_LINEAR; return TRUE_;
        default: return err(BAD_ATTRIBUTE);
    }
}
static EGLBoolean SurfaceAttrib(EGLDisplay d, EGLSurface s, EGLint, EGLint) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (!s) return err(BAD_SURFACE);
    return TRUE_;
}

static EGLBoolean BindAPI(EGLenum api) {
    if (api != OPENGL_ES_API) return err(BAD_PARAMETER);
    tApi = api;
    return TRUE_;
}
static EGLenum QueryAPI() { return tApi; }

static EGLContext CreateContext(EGLDisplay d, EGLConfig, EGLContext shareCtx, const EGLint* attribs) {
    if (d != kDisplay) return err(BAD_DISPLAY), nullptr;
    installCrashTrace();  // again: the harness may have installed its handler after eglInitialize
    int major = 1, minor = 0;
    if (attribs)
        for (const EGLint* a = attribs; a[0] != NONE; a += 2) {
            if (a[0] == CONTEXT_MAJOR_VERSION) major = a[1];
            if (a[0] == CONTEXT_MINOR_VERSION) minor = a[1];
        }
    if (major < 2 || major > 3) return err(BAD_MATCH), nullptr;
    // ES 2 requests get a 3.2-capable context (GLES 3 is backwards compatible), reporting 3.2.
    if (major == 3 && minor > 2) return err(BAD_MATCH), nullptr;
    // Strict (conformance) mode: only what is advertised; a 3.2 request fails so the caller falls back to 3.1.
    if (major == 3 && minor > gl::kAdvertisedMinor && vpglsl::strictMode()) return err(BAD_MATCH), nullptr;
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    if (shareCtx && static_cast<Context*>(shareCtx)->dead) return err(BAD_CONTEXT), nullptr;
    ShareGroup* sg = shareCtx ? static_cast<Context*>(shareCtx)->share : new ShareGroup;
    return new Context(device(), sg, 3, 2);
}
static EGLBoolean DestroyContext(EGLDisplay d, EGLContext c) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (!c) return err(BAD_CONTEXT);
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    auto* ctx = static_cast<Context*>(c);
    forgetContext(ctx);
    // EGL 1.5: un contexto actual en algún hilo (también en este) se borra cuando deja de serlo.
    if (ctx->owned) {
        ctx->dead = true;
        return TRUE_;
    }
    delete ctx;
    return TRUE_;
}
static EGLBoolean QueryContext(EGLDisplay d, EGLContext c, EGLint attr, EGLint* value) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (!c) return err(BAD_CONTEXT);
    switch (attr) {
        case CONTEXT_CLIENT_VERSION: *value = 3; return TRUE_;
        case CONTEXT_CLIENT_TYPE: *value = EGLint(OPENGL_ES_API); return TRUE_;
        case CONFIG_ID: *value = kConfigs[0].id; return TRUE_;
        case RENDER_BUFFER: *value = BACK_BUFFER; return TRUE_;
        default: return err(BAD_ATTRIBUTE);
    }
}

static EGLBoolean MakeCurrent(EGLDisplay d, EGLSurface draw, EGLSurface read, EGLContext c) {
    if (d != kDisplay && !(d == nullptr && !c)) return err(BAD_DISPLAY);
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    auto* ctx = static_cast<Context*>(c);
    auto* nd = static_cast<Surface*>(draw);
    auto* nr = static_cast<Surface*>(read);
    if (ctx && ctx->dead) return err(BAD_CONTEXT);
    if (ctx && ctx->owned && ctx->owner != std::this_thread::get_id()) return err(BAD_ACCESS);  // actual en otro hilo
    if ((nd && nd->dead) || (nr && nr->dead)) return err(BAD_SURFACE);
    Context* old = current();
    Surface* oldDraw = tDraw;
    Surface* oldRead = tRead;
    if (ctx) ctx->owned = true, ctx->owner = std::this_thread::get_id();
    if (nd) ++nd->bound;
    if (nr) ++nr->bound;
    if (old && old != ctx) old->flush(false), old->owned = false;
    setCurrent(ctx);
    tDraw = nd;
    tRead = nr;
    unbindSurface(oldDraw);
    unbindSurface(oldRead);
    if (old && old != ctx && old->dead) delete old;  // destruido mientras era actual (forgetContext ya hecho)
    if (ctx) {
        const bool first = !ctx->draw && !ctx->viewport[2];
        ctx->draw = tDraw;
        ctx->read = tRead;
        if (first && tDraw) {
            ctx->viewport[2] = GLint(tDraw->width), ctx->viewport[3] = GLint(tDraw->height);
            ctx->scissor[2] = GLint(tDraw->width), ctx->scissor[3] = GLint(tDraw->height);
        }
    }
    return TRUE_;
}
static EGLContext GetCurrentContext() { return current(); }
static EGLDisplay GetCurrentDisplay() { return current() ? kDisplay : nullptr; }
static EGLSurface GetCurrentSurface(EGLint which) { return which == DRAW ? tDraw : which == READ ? tRead : (err(BAD_PARAMETER), nullptr); }

static EGLBoolean SwapBuffers(EGLDisplay d, EGLSurface sv) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    auto* s = static_cast<Surface*>(sv);
    if (!s) return err(BAD_SURFACE);
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    Context* c = current();
    if (c) c->flush(true);
    ++s->presented;
    texStatsTick(false);
    recipeStatsTick(false);
    if (s->kind == Surface::Window && gPresent && s->color)
        gPresent(gPresentCtx, s->window, s->color->image, s->width, s->height, uint32_t(s->config->color));
    return TRUE_;
}
static EGLBoolean SwapInterval(EGLDisplay d, EGLint interval) {
    if (d != kDisplay) return err(BAD_DISPLAY);
    if (tDraw) tDraw->swapInterval = interval;
    return TRUE_;
}
static EGLint GetError() {
    EGLint e = tError;
    tError = SUCCESS;
    return e;
}
static EGLBoolean WaitGL() {
    if (Context* c = current()) c->flush(true);
    return TRUE_;
}
static EGLBoolean WaitNative(EGLint) { return TRUE_; }
static EGLBoolean ReleaseThread() {
    MakeCurrent(kDisplay, nullptr, nullptr, nullptr);
    return TRUE_;
}

// ---- fences --------------------------------------------------------------------------------------------
struct EglSync {
    uint64_t serial;
};
static EGLSync CreateSyncKHR(EGLDisplay d, EGLenum type, const EGLint*) {
    if (d != kDisplay) return err(BAD_DISPLAY), nullptr;
    if (type != SYNC_FENCE_KHR && type != SYNC_NATIVE_FENCE_ANDROID) return err(BAD_ATTRIBUTE), nullptr;
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    Context* c = current();
    uint64_t serial = c ? c->flush(false) : 0;
    if (!serial) serial = device()->submitted;
    return new EglSync{serial};
}
static EGLBoolean DestroySyncKHR(EGLDisplay, EGLSync s) {
    delete static_cast<EglSync*>(s);
    return TRUE_;
}
static EGLint ClientWaitSyncKHR(EGLDisplay, EGLSync s, EGLint, uint64_t timeout) {
    if (!s) return err(BAD_PARAMETER), FALSE_;
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    auto* y = static_cast<EglSync*>(s);
    device()->retire(false);
    if (device()->completed >= y->serial) return CONDITION_SATISFIED_KHR;
    if (timeout == 0) return TIMEOUT_EXPIRED_KHR;
    device()->waitSerial(y->serial);
    return CONDITION_SATISFIED_KHR;
}
static EGLint WaitSyncKHR(EGLDisplay, EGLSync, EGLint) { return TRUE_; }  // one queue: already ordered
static EGLBoolean GetSyncAttribKHR(EGLDisplay, EGLSync s, EGLint attr, EGLint* value) {
    if (!s) return err(BAD_PARAMETER);
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    device()->retire(false);
    auto* y = static_cast<EglSync*>(s);
    switch (attr) {
        case SYNC_TYPE_KHR: *value = SYNC_FENCE_KHR; return TRUE_;
        case SYNC_STATUS_KHR: *value = device()->completed >= y->serial ? SIGNALED_KHR : UNSIGNALED_KHR; return TRUE_;
        case SYNC_CONDITION_KHR: *value = SYNC_PRIOR_COMMANDS_COMPLETE_KHR; return TRUE_;
        default: return err(BAD_ATTRIBUTE);
    }
}
static EGLint DupNativeFenceFDANDROID(EGLDisplay, EGLSync) { return -1; }  // EGL_NO_NATIVE_FENCE_FD_ANDROID

// ---- EGLImage (texture sources) ------------------------------------------------------------------------
struct EglImage {
    ImageP image;
    GLenum target;
};
static EGLImage CreateImageKHR(EGLDisplay d, EGLContext c, EGLenum target, void* buffer, const EGLint*) {
    if (d != kDisplay) return err(BAD_DISPLAY), nullptr;
    std::lock_guard<std::recursive_mutex> l(device()->lock);
    if (target == 0x30B1 /* EGL_GL_TEXTURE_2D_KHR */) {
        auto* ctx = static_cast<Context*>(c);
        if (!ctx) return err(BAD_CONTEXT), nullptr;
        Texture* t = ctx->texture(GLuint(reinterpret_cast<uintptr_t>(buffer)));
        if (!t || !t->image) return err(BAD_PARAMETER), nullptr;
        return new EglImage{t->image, gl::TEXTURE_2D};
    }
    // EGL_NATIVE_BUFFER_ANDROID (AHardwareBuffer from the host's video path) is the host's (Klepton's
    // kl_egl.c) business: it keeps answering it.
    return err(BAD_PARAMETER), nullptr;
}
static EGLBoolean DestroyImageKHR(EGLDisplay, EGLImage i) {
    delete static_cast<EglImage*>(i);
    return TRUE_;
}

}  // namespace vpgles

// ---- glEGLImageTargetTexture2DOES lives with the GL entry points; it needs this ---------------------------
namespace vpgles {
bool eglImageToTexture(void* image, Texture& t) {
    auto* i = static_cast<EglImage*>(image);
    if (!i || !i->image) return false;
    t.image = i->image;
    t.levels = {{i->image->width, i->image->height, 1, i->image->format}};
    t.immutable = true, t.immutableLevels = 1;
    return true;
}

void* eglProc(const char* name) {
#define E(n) \
    if (!std::strcmp(name, "egl" #n)) return reinterpret_cast<void*>(&n);
    E(GetDisplay) E(GetPlatformDisplay) E(Initialize) E(Terminate) E(QueryString) E(ChooseConfig) E(GetConfigs)
    E(GetConfigAttrib) E(CreateWindowSurface) E(CreatePlatformWindowSurface) E(CreatePbufferSurface) E(DestroySurface)
    E(QuerySurface) E(SurfaceAttrib) E(BindAPI) E(QueryAPI) E(CreateContext) E(DestroyContext) E(QueryContext)
    E(MakeCurrent) E(GetCurrentContext) E(GetCurrentDisplay) E(GetCurrentSurface) E(SwapBuffers) E(SwapInterval)
    E(GetError) E(WaitGL) E(WaitNative) E(ReleaseThread) E(CreateSyncKHR) E(DestroySyncKHR) E(ClientWaitSyncKHR)
    E(WaitSyncKHR) E(GetSyncAttribKHR) E(DupNativeFenceFDANDROID) E(CreateImageKHR) E(DestroyImageKHR)
#undef E
    if (!std::strcmp(name, "eglCreateSync")) return reinterpret_cast<void*>(&CreateSyncKHR);
    if (!std::strcmp(name, "eglDestroySync")) return reinterpret_cast<void*>(&DestroySyncKHR);
    if (!std::strcmp(name, "eglClientWaitSync")) return reinterpret_cast<void*>(&ClientWaitSyncKHR);
    if (!std::strcmp(name, "eglCreateImage")) return reinterpret_cast<void*>(&CreateImageKHR);
    if (!std::strcmp(name, "eglDestroyImage")) return reinterpret_cast<void*>(&DestroyImageKHR);
    if (!std::strcmp(name, "eglSwapBuffersWithDamageKHR")) return reinterpret_cast<void*>(&SwapBuffers);
    return nullptr;
}
}  // namespace vpgles

// ---- host API ------------------------------------------------------------------------------------------
using namespace vpgles;

extern "C" void vpgles_set_present(vpgles_present_fn fn, void* ctx) {
    gPresent = fn;
    gPresentCtx = ctx;
}
extern "C" void vpgles_set_window_size(vpgles_window_size_fn fn, void* ctx) {
    gWindowSize = fn;
    gWindowSizeCtx = ctx;
}
