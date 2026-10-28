// Framebuffers, render passes, programs, pipelines and draws: GL state -> cached Vulkan objects.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

#include "context.h"
#include "egl_internal.h"
#include "vpgpu/core/perf.h"

namespace vpgles {

// ---- object lookup ----------------------------------------------------------------------------------
template <class M>
static auto* lookup(const M& m, GLuint n) {
    auto it = m.find(n);
    return it == m.end() ? nullptr : it->second.get();
}
Buffer* Context::buffer(GLuint n) const { return n ? lookup(share->buffers, n) : nullptr; }
Texture* Context::texture(GLuint n) const { return n ? lookup(share->textures, n) : nullptr; }
Renderbuffer* Context::rbo(GLuint n) const { return n ? lookup(share->renderbuffers, n) : nullptr; }
Program* Context::prog(GLuint n) const { return n ? lookup(share->programs, n) : nullptr; }
Shader* Context::shader(GLuint n) const { return n ? lookup(share->shaders, n) : nullptr; }
Framebuffer* Context::fbo(GLuint n) const { return lookup(fbos, n); }
VertexArray* Context::currentVao() { return lookup(vaos, vao); }
GLuint Context::genName() { return share->nextName++; }

GLuint* Context::bufferBinding(GLenum target) {
    switch (target) {
        case gl::ARRAY_BUFFER: return &arrayBuffer;
        case gl::ELEMENT_ARRAY_BUFFER: return &currentVao()->elementBuffer;
        case gl::COPY_READ_BUFFER: return &copyRead;
        case gl::COPY_WRITE_BUFFER: return &copyWrite;
        case gl::PIXEL_PACK_BUFFER: return &pixelPack;
        case gl::PIXEL_UNPACK_BUFFER: return &pixelUnpack;
        case gl::UNIFORM_BUFFER: return &uniformBuffer;
        case gl::SHADER_STORAGE_BUFFER: return &inertBuffer[0];
        case gl::ATOMIC_COUNTER_BUFFER: return &inertBuffer[1];
        case gl::DRAW_INDIRECT_BUFFER: return &inertBuffer[2];
        case gl::DISPATCH_INDIRECT_BUFFER: return &inertBuffer[3];
        case gl::TRANSFORM_FEEDBACK_BUFFER: return &inertBuffer[4];
        case gl::TEXTURE_BUFFER: return &inertBuffer[5];
        default: return nullptr;
    }
}

GLuint* Context::textureBinding(GLenum target, int unit) {
    Unit& u = units[size_t(unit)];
    switch (target) {
        case gl::TEXTURE_2D: return &u.tex2D;
        case gl::TEXTURE_CUBE_MAP: return &u.texCube;
        case gl::TEXTURE_3D: return &u.tex3D;
        case gl::TEXTURE_2D_ARRAY: return &u.tex2DArray;
        case gl::TEXTURE_EXTERNAL_OES: return &u.texExternal;
        case gl::TEXTURE_2D_MULTISAMPLE: return &u.tex2DMS;
        case gl::TEXTURE_CUBE_MAP_ARRAY: return &u.texCubeArray;
        case gl::TEXTURE_2D_MULTISAMPLE_ARRAY: return &u.tex2DMSArray;
        case gl::TEXTURE_BUFFER: return &u.texBuffer;
        default: return nullptr;
    }
}

Texture* Context::boundTexture(GLenum target) {
    GLuint* b = textureBinding(target, activeUnit);
    if (!b) return nullptr;
    // Name 0 is the default texture of the target: one per context (kept in the share map under a
    // reserved name per target).
    if (*b == 0) {
        GLuint reserved = 0xFFFF0000u | (target & 0xFFFF);
        auto& m = share->textures;
        auto it = m.find(reserved);
        if (it == m.end()) {
            auto t = std::make_unique<Texture>();
            t->name = 0;
            t->target = target;
            it = m.emplace(reserved, std::move(t)).first;
        }
        return it->second.get();
    }
    return texture(*b);
}

// ---- framebuffers -----------------------------------------------------------------------------------
bool Context::targetOf(const Attachment& a, FbTarget& out, uint32_t& w, uint32_t& h) {
    if (a.kind == Attachment::Tex) {
        Texture* t = texture(a.name);
        if (!t || !t->image) return false;
        if (uint32_t(a.level) >= t->image->levels) return false;
        out.image = t->image.get();
        out.level = uint32_t(a.level);
        out.layer = uint32_t(a.numViews ? a.baseView : a.layer);
        out.layers = a.numViews ? uint32_t(a.numViews) : 1;
        out.multiview = a.numViews > 0;
        w = std::max(1u, t->image->width >> a.level);
        h = std::max(1u, t->image->height >> a.level);
        return true;
    }
    if (a.kind == Attachment::Rbo) {
        Renderbuffer* r = rbo(a.name);
        if (!r || !r->image) return false;
        out.image = r->image.get();
        out.level = 0, out.layer = 0, out.layers = 1;
        w = uint32_t(r->width), h = uint32_t(r->height);
        return true;
    }
    return false;
}

Context::FbInfo Context::resolveFramebuffer(GLuint name, bool forDraw) {
    FbInfo fb;
    if (name == 0) {
        fb.defaultFb = true;
        Surface* s = forDraw ? draw : read;
        if (!s) {
            fb.status = gl::FRAMEBUFFER_UNDEFINED;
            return fb;
        }
        fb.width = s->width, fb.height = s->height;
        fb.color[0].image = s->color.get();
        if (s->depthStencil) {
            fb.depth.image = s->depthStencil.get();
            if (s->depthStencil->format->stencil) fb.stencil.image = s->depthStencil.get();
        }
        return fb;
    }
    Framebuffer* f = fbo(name);
    if (!f) {
        fb.status = gl::FRAMEBUFFER_UNDEFINED;
        return fb;
    }
    bool any = false, first = true;
    uint32_t views = 0;
    auto take = [&](const Attachment& a, FbTarget& t, bool color) -> bool {
        if (a.kind == Attachment::None) return true;
        uint32_t w = 0, h = 0;
        if (!targetOf(a, t, w, h)) {
            fb.status = gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            return false;
        }
        const FormatInfo* fi = t.image->format;
        if (color ? (fi->depth || fi->stencil || (!fi->renderable && fi->baseFormat != gl::RGB && !fi->integer))
                  : !(fi->depth || fi->stencil)) {
            fb.status = gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            return false;
        }
        const uint32_t v = t.multiview ? t.layers : 0;
        if (first) fb.width = w, fb.height = h, views = v, first = false;
        else {
            // GLES 3: the framebuffer is the intersection of its attachments.
            fb.width = std::min(fb.width, w), fb.height = std::min(fb.height, h);
            if (v != views) {
                fb.status = 0x9633;  // FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR
                return false;
            }
        }
        any = true;
        return true;
    };
    for (int i = 0; i < kMaxDrawBuffers; ++i)
        if (!take(f->color[size_t(i)], fb.color[size_t(i)], true)) return fb;
    if (!take(f->depth, fb.depth, false)) return fb;
    if (!take(f->stencil, fb.stencil, false)) return fb;
    if (fb.depth.image && fb.stencil.image && fb.depth.image != fb.stencil.image) {
        fb.status = gl::FRAMEBUFFER_UNSUPPORTED;  // separate depth and stencil images
        return fb;
    }
    if (!any) {
        if (f->defaultWidth > 0 && f->defaultHeight > 0) {  // ES 3.1: an attachment-less framebuffer has its default size
            fb.width = uint32_t(f->defaultWidth), fb.height = uint32_t(f->defaultHeight);
            return fb;
        }
        fb.status = gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
        return fb;
    }
    fb.viewCount = views;
    return fb;
}

bool Context::PassKey::operator==(const PassKey& o) const {
    return color == o.color && depth == o.depth && stencil == o.stencil && w == o.w && h == o.h && viewMask == o.viewMask;
}

Context::PassKey Context::passKeyFor(const FbInfo& fb) {
    PassKey key;
    key.w = fb.width, key.h = fb.height;
    key.viewMask = fb.viewCount > 1 ? (1u << fb.viewCount) - 1 : 0;
    const uint32_t layerCount = fb.viewCount > 1 ? fb.viewCount : 1;
    std::array<GLenum, kMaxDrawBuffers> bufs{};
    if (fb.defaultFb) {
        bufs[0] = fbos[0]->drawBuffers[0] == gl::NONE ? gl::NONE : gl::COLOR_ATTACHMENT0;
    } else {
        bufs = fbo(drawFbo)->drawBuffers;
    }
    for (int i = 0; i < kMaxDrawBuffers; ++i) {
        // Draw buffer i writes colour attachment drawBuffers[i] (GLES: must be COLOR_ATTACHMENTi or NONE).
        if (bufs[size_t(i)] == gl::NONE) continue;
        const FbTarget& t = fb.color[size_t(i)];
        if (t.image) key.color[size_t(i)] = t.image->attachmentView(t.level, t.layer, layerCount);
    }
    if (fb.depth.image) key.depth = fb.depth.image->attachmentView(fb.depth.level, fb.depth.layer, layerCount);
    if (fb.stencil.image) key.stencil = fb.stencil.image->attachmentView(fb.stencil.level, fb.stencil.layer, layerCount);
    return key;
}

// `firstClear`: full-target clears the caller would record right after the pass begins. When this call begins a pass they
// become its CLEAR load ops (tile GPUs then never load the old contents) and the list is emptied; when the pass goes
// on, the list is left for the caller to record as vkCmdClearAttachments.
bool Context::beginPass(const FbInfo& fb, std::vector<VkClearAttachment>* firstClear) {
    const PassKey key = passKeyFor(fb);
    // Index: colour attachment i, kMaxDrawBuffers = depth, +1 = stencil.
    std::array<VkAttachmentLoadOp, kMaxDrawBuffers + 2> load;
    load.fill(VK_ATTACHMENT_LOAD_OP_LOAD);
    std::array<VkClearValue, kMaxDrawBuffers + 2> clearValue{};
    // An attachment this call writes is defined again: no longer invalidated, no longer to be dropped. A new pass loads
    // DONT_CARE what was invalidated before it.
    uint32_t drop = passDrop_;
    auto written = [&](const FbTarget& t, VkImageView v, uint8_t bit, size_t idx, bool fresh) {
        if (!v) return;
        drop &= ~(1u << idx);
        // Other views of the image (another layer / mip range, e.g. a layer-0 view vs a multiview one) may overlap what
        // was written: they are defined again too.
        for (auto o = t.image->invalid.begin(); o != t.image->invalid.end();) {
            if (o->first == v) {
                ++o;
                continue;
            }
            o->second &= uint8_t(~bit);
            o = o->second ? std::next(o) : t.image->invalid.erase(o);
        }
        auto it = t.image->invalid.find(v);
        if (it == t.image->invalid.end()) return;
        if (fresh && (it->second & bit)) load[idx] = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        it->second &= uint8_t(~bit);
        if (!it->second) t.image->invalid.erase(it);
    };
    auto touch = [&](bool fresh) {
        for (int i = 0; i < kMaxDrawBuffers; ++i) written(fb.color[size_t(i)], key.color[size_t(i)], 1, size_t(i), fresh);
        written(fb.depth, key.depth, 2, kMaxDrawBuffers, fresh);
        written(fb.stencil, key.stencil, 4, kMaxDrawBuffers + 1, fresh);
    };
    if (passOpen_ && key == pass_) {
        touch(false);
        setPassDrop(drop);
        return true;
    }
    endPass();
    // Writes before this pass (uploads, copies, previous passes sampling these images) are visible.
    barrier();
    touch(true);
    if (firstClear) {
        // Depth and stencil share one image: a clear of only one aspect stays a vkCmdClearAttachments (a CLEAR load op
        // would have to LOAD the other aspect, and VPVulkan keeps its OpClearAttachments classification).
        const bool shared = key.depth && key.stencil;
        std::vector<VkClearAttachment> rest;
        for (const VkClearAttachment& a : *firstClear) {
            const VkImageAspectFlags ds = a.aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
            if (shared && ds && ds != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
                rest.push_back(a);
                continue;
            }
            auto set = [&](size_t idx) {
                load[idx] = VK_ATTACHMENT_LOAD_OP_CLEAR;
                clearValue[idx] = a.clearValue;
            };
            if (a.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) set(a.colorAttachment);
            if (a.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) set(kMaxDrawBuffers);
            if (a.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) set(kMaxDrawBuffers + 1);
        }
        *firstClear = std::move(rest);
    }
    auto fill = [&](VkRenderingAttachmentInfo& a, VkImageView v, size_t idx) {
        a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        a.imageView = v;
        a.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        a.loadOp = load[idx];
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a.clearValue = clearValue[idx];
    };
    VkRenderingAttachmentInfo colors[kMaxDrawBuffers]{};
    uint32_t colorCount = 0;
    for (int i = 0; i < kMaxDrawBuffers; ++i) {
        fill(colors[i], key.color[size_t(i)], size_t(i));
        if (colors[i].imageView) colorCount = uint32_t(i + 1);
    }
    VkRenderingAttachmentInfo depth{}, stencil{};
    fill(depth, key.depth, kMaxDrawBuffers);
    fill(stencil, key.stencil, kMaxDrawBuffers + 1);
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {fb.width, fb.height}};
    ri.layerCount = key.viewMask ? 1 : 1;
    ri.viewMask = key.viewMask;
    ri.colorAttachmentCount = colorCount;
    ri.pColorAttachments = colors;
    ri.pDepthAttachment = key.depth ? &depth : nullptr;
    ri.pStencilAttachment = key.stencil ? &stencil : nullptr;
    dev->vk.vkCmdBeginRendering(cmd(), &ri);
    passOpen_ = true;
    pass_ = key;
    passDrop_ = 0;
    return true;
}

// The attachments of the open pass that are undefined when it ends (a bit per slot, as in beginPass): VPVulkan turns
// their store into DONT_CARE. The set replaces the earlier one; unchanged: no call.
void Context::setPassDrop(uint32_t mask) {
    if (mask == passDrop_) return;
    passDrop_ = mask;
    if (dev->vk.vpvk_cmd_pass_drop)
        dev->vk.vpvk_cmd_pass_drop(cmd(), mask & ((1u << kMaxDrawBuffers) - 1), (mask >> kMaxDrawBuffers) & 1, (mask >> (kMaxDrawBuffers + 1)) & 1);
}

void Context::invalidateFramebuffer(GLenum target, GLsizei n, const GLenum* atts) {
    if (target != gl::FRAMEBUFFER && target != gl::DRAW_FRAMEBUFFER && target != gl::READ_FRAMEBUFFER)
        return setError(gl::INVALID_ENUM, target);
    if (n < 0) return setError(gl::INVALID_VALUE);
    const GLuint name = target == gl::READ_FRAMEBUFFER ? readFbo : drawFbo;
    FbInfo fb = resolveFramebuffer(name, target != gl::READ_FRAMEBUFFER);
    if (fb.status != gl::FRAMEBUFFER_COMPLETE) return;
    const uint32_t layerCount = fb.viewCount > 1 ? fb.viewCount : 1;
    uint32_t drop = passDrop_;
    // `bit`: 1 colour, 2 depth, 4 stencil. External images (eye / surface images that are presented or exported) and the
    // colour of the default framebuffer are never discarded.
    auto mark = [&](const FbTarget& t, uint8_t bit, size_t idx) {
        if (!t.image || t.image->external) return;
        const VkImageView v = t.image->attachmentView(t.level, t.layer, layerCount);
        t.image->invalid[v] |= bit;
        if (passOpen_ && (idx < size_t(kMaxDrawBuffers) ? pass_.color[idx] : idx == size_t(kMaxDrawBuffers) ? pass_.depth : pass_.stencil) == v)
            drop |= 1u << idx;
    };
    GLenum bad = 0;
    for (GLsizei i = 0; i < n && !bad; ++i) {
        const GLenum a = atts[i];
        if (!name) {
            if (a == gl::DEPTH) mark(fb.depth, 2, kMaxDrawBuffers);
            else if (a == gl::STENCIL) mark(fb.stencil, 4, kMaxDrawBuffers + 1);
            else if (a != gl::COLOR) bad = a;
        } else if (a >= gl::COLOR_ATTACHMENT0 && a < gl::COLOR_ATTACHMENT0 + kMaxDrawBuffers) {
            mark(fb.color[a - gl::COLOR_ATTACHMENT0], 1, a - gl::COLOR_ATTACHMENT0);
        } else if (a == gl::DEPTH_ATTACHMENT || a == gl::DEPTH_STENCIL_ATTACHMENT || a == gl::STENCIL_ATTACHMENT) {
            if (a != gl::STENCIL_ATTACHMENT) mark(fb.depth, 2, kMaxDrawBuffers);
            if (a != gl::DEPTH_ATTACHMENT) mark(fb.stencil, 4, kMaxDrawBuffers + 1);
        } else {
            bad = a;
        }
    }
    setPassDrop(drop);  // with an open pass: VPVulkan stores the marked attachments DONT_CARE
    if (bad) setError(gl::INVALID_ENUM, bad);
}

// ---- programs ---------------------------------------------------------------------------------------
Program::~Program() {
    Device* d = device();
    if (!d || !d->device) return;
    for (auto& [k, p] : pipelines) d->vk.vkDestroyPipeline(d->device, p, nullptr);
    d->vk.vkDestroyShaderModule(d->device, vsModule, nullptr);
    d->vk.vkDestroyShaderModule(d->device, fsModule, nullptr);
    d->vk.vkDestroyShaderModule(d->device, csModule, nullptr);
    d->vk.vkDestroyPipelineLayout(d->device, pipeLayout, nullptr);
    d->vk.vkDestroyDescriptorSetLayout(d->device, setLayout, nullptr);
}

// samplerBuffer / isamplerBuffer / usamplerBuffer: a texel buffer descriptor instead of a combined image sampler.
static bool isBufferSampler(uint32_t glType) {
    return glType == gl::SAMPLER_BUFFER || glType == gl::INT_SAMPLER_BUFFER || glType == gl::UNSIGNED_INT_SAMPLER_BUFFER;
}

// GL_IMAGE_* / INT_IMAGE_* / UNSIGNED_INT_IMAGE_* (0x904C..0x906A): which Vulkan view type the image uniform needs.
static bool isImageType(uint32_t t) { return t >= 0x904C && t <= 0x906A; }
// 0 = 1D, 1 = 2D, 2 = 3D, 3 = rect, 4 = cube, 5 = buffer, 6 = 1D array, 7 = 2D array, 8 = cube array, 9 / 10 = multisample.
static int imageDim(uint32_t t) {
    const uint32_t base = t >= 0x9062 ? 0x9062 : t >= 0x9057 ? 0x9057 : 0x904C;
    return int(t - base);
}

// The 13 formats of glBindImageTexture (ES 3.1 table 8.27) as the Vulkan format a view reads them with; UNDEFINED otherwise.
static VkFormat imageViewFormat(GLenum f) {
    switch (f) {
        case 0x8814: return VK_FORMAT_R32G32B32A32_SFLOAT;  // RGBA32F
        case 0x881A: return VK_FORMAT_R16G16B16A16_SFLOAT;  // RGBA16F
        case 0x822E: return VK_FORMAT_R32_SFLOAT;           // R32F
        case 0x8058: return VK_FORMAT_R8G8B8A8_UNORM;       // RGBA8
        case 0x8F97: return VK_FORMAT_R8G8B8A8_SNORM;       // RGBA8_SNORM
        case 0x8D82: return VK_FORMAT_R32G32B32A32_SINT;    // RGBA32I
        case 0x8D88: return VK_FORMAT_R16G16B16A16_SINT;    // RGBA16I
        case 0x8D8E: return VK_FORMAT_R8G8B8A8_SINT;        // RGBA8I
        case 0x8235: return VK_FORMAT_R32_SINT;             // R32I
        case 0x8D70: return VK_FORMAT_R32G32B32A32_UINT;    // RGBA32UI
        case 0x8D76: return VK_FORMAT_R16G16B16A16_UINT;    // RGBA16UI
        case 0x8D7C: return VK_FORMAT_R8G8B8A8_UINT;        // RGBA8UI
        case 0x8236: return VK_FORMAT_R32_UINT;             // R32UI
        default: return VK_FORMAT_UNDEFINED;
    }
}
// Texel size of those formats (a view may only reinterpret between equal sizes).
static uint32_t imageFormatBytes(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_UINT: return 8;
        default: return 4;
    }
}

VkImageView Context::sliceView(Texture& t, uint32_t level, GLint layer, GLenum access, VkFormat format) {
    Image& im = *t.image;
    const uint32_t w = std::max(1u, im.width >> level), h = std::max(1u, im.height >> level), depth = std::max(1u, im.depth >> level);
    if (!dispatching_ || layer < 0 || uint32_t(layer) >= depth) {
        static std::atomic<int> said{0};
        if (said.fetch_add(1) < 8)
            logf("gles.warn what=image2d_of_3d_unsupported texture=%u level=%u layer=%d dispatch=%d (the image reads as zero)", t.name, level, int(layer), int(dispatching_));
        return VK_NULL_HANDLE;
    }
    // One 2D proxy per (3D image, level): the slice is copied into it before the dispatch and back after it.
    auto& proxy = sliceProxies_[{&im, level}];
    if (proxy && (proxy->format != im.format || proxy->width != w || proxy->height != h)) {
        keep(proxy);
        proxy.reset();
    }
    if (!proxy) {
        if (sliceProxies_.size() > 16) {
            for (auto it = sliceProxies_.begin(); it != sliceProxies_.end();) {
                if (&it->second == &proxy) { ++it; continue; }
                keep(it->second);
                it = sliceProxies_.erase(it);
            }
        }
        proxy = createImage2D(dev, im.format, w, h, 1, 1,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
        if (!proxy) return VK_NULL_HANDLE;
        proxy->storage = true;
        imageInit(*proxy);
    }
    VkImageCopy r{};
    r.srcSubresource = {im.aspect(), level, 0, 1};
    r.srcOffset = {0, 0, layer};
    r.dstSubresource = {im.aspect(), 0, 0, 1};
    r.extent = {w, h, 1};
    dev->vk.vkCmdCopyImage(cmd(), im.image, VK_IMAGE_LAYOUT_GENERAL, proxy->image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier();
    SliceSync s;
    s.proxy = proxy;
    s.source = t.image;
    s.level = level, s.slice = uint32_t(layer);
    s.writeBack = access != gl::READ_ONLY;
    if (s.writeBack) sliceBack_.push_back(s);
    keep(proxy);
    return proxy->storageView(0, 0, 1, VK_IMAGE_VIEW_TYPE_2D, format);
}

// After a dispatch that used slice proxies: the written ones go back into their 3D texture.
void Context::syncSlicesBack() {
    if (sliceBack_.empty()) return;
    barrier();
    for (const SliceSync& s : sliceBack_) {
        const Image& im = *s.source;
        VkImageCopy r{};
        r.srcSubresource = {im.aspect(), 0, 0, 1};
        r.dstSubresource = {im.aspect(), s.level, 0, 1};
        r.dstOffset = {0, 0, int32_t(s.slice)};
        r.extent = {s.proxy->width, s.proxy->height, 1};
        dev->vk.vkCmdCopyImage(cmd(), s.proxy->image, VK_IMAGE_LAYOUT_GENERAL, im.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        keep(s.source);
    }
    sliceBack_.clear();
    barrier();
}

// ---- program binary ---------------------------------------------------------------------------------
// Blob (little endian): magic, version, payload size, FNV-1a 64 of the payload; payload = u32 separable flag,
// u32 stage count, per stage {u32 GL shader type, u32 length, source as compiled}, u32 binding count, per binding
// {u32 index, u32 length, attribute name}. Everything linkShaders needs, so a load is the same link as glLinkProgram.
// (No transform feedback varyings / fragment output bindings: the layer keeps none.)
static constexpr uint32_t kBinaryMagic = 0x42504B56, kBinaryVersion = 2, kBinaryHeader = 20;

static uint64_t fnv1a(const uint8_t* d, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ d[i]) * 1099511628211ull;
    return h;
}
static void put32(std::vector<uint8_t>& o, uint32_t v) {
    for (int i = 0; i < 4; ++i) o.push_back(uint8_t(v >> (8 * i)));
}
static void putStr(std::vector<uint8_t>& o, const std::string& s) {
    put32(o, uint32_t(s.size()));
    o.insert(o.end(), s.begin(), s.end());
}
std::vector<uint8_t> Context::programBinary(const Program& p) {
    std::vector<uint8_t> blob;
    if (!p.bin.valid) return blob;
    static const GLenum kTypes[3] = {gl::VERTEX_SHADER, gl::FRAGMENT_SHADER, gl::COMPUTE_SHADER};
    std::vector<uint8_t> payload;
    uint32_t n = 0;
    for (const auto& src : p.bin.sources) n += src ? 1 : 0;
    put32(payload, p.bin.separable ? 1 : 0);
    put32(payload, n);
    for (int k = 0; k < 3; ++k)
        if (p.bin.sources[k]) put32(payload, kTypes[k]), putStr(payload, *p.bin.sources[k]);
    put32(payload, uint32_t(p.bin.attribBindings.size()));
    for (const auto& [name, index] : p.bin.attribBindings) put32(payload, uint32_t(index)), putStr(payload, name);
    put32(blob, kBinaryMagic);
    put32(blob, kBinaryVersion);
    put32(blob, uint32_t(payload.size()));
    const uint64_t h = fnv1a(payload.data(), payload.size());
    put32(blob, uint32_t(h));
    put32(blob, uint32_t(h >> 32));
    blob.insert(blob.end(), payload.begin(), payload.end());
    return blob;
}

namespace {
struct BlobReader {
    const uint8_t* d;
    size_t n, at = 0;
    bool ok = true;
    uint32_t u32() {
        if (n - at < 4) return ok = false, 0;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= uint32_t(d[at + size_t(i)]) << (8 * i);
        at += 4;
        return v;
    }
    std::string str() {
        const uint32_t len = u32();
        if (!ok || n - at < len) return ok = false, std::string();
        std::string s(reinterpret_cast<const char*>(d + at), len);
        at += len;
        return s;
    }
};
}  // namespace

bool Context::loadProgramBinary(Program& p, GLenum format, const void* data, size_t bytes) {
    // Any failure: tear the old link down (the unattached link fails), then say why.
    auto fail = [&](const char* why) {
        linkShaders(p, nullptr, nullptr, nullptr);
        p.log = std::string("ERROR: program binary rejected: ") + why + "\n";
        return false;
    };
    if (format != kProgramBinaryFormat) return fail("unknown binary format");
    BlobReader h{static_cast<const uint8_t*>(data), data ? bytes : 0};
    if (h.u32() != kBinaryMagic || h.u32() != kBinaryVersion) return fail("bad header or version");
    const uint32_t size = h.u32();
    const uint64_t hashLo = h.u32(), hashHi = h.u32(), hash = hashLo | hashHi << 32;
    if (!h.ok || h.n - kBinaryHeader != size) return fail("bad size");
    if (fnv1a(h.d + kBinaryHeader, size) != hash) return fail("hash mismatch");
    BlobReader r{h.d + kBinaryHeader, size};
    Shader sh[3];
    const bool separable = r.u32() != 0;
    Shader* slot[3] = {nullptr, nullptr, nullptr};  // vertex, fragment, compute
    const uint32_t nStages = r.u32();
    for (uint32_t i = 0; r.ok && i < nStages; ++i) {
        const uint32_t type = r.u32();
        const std::string src = r.str();
        const int k = type == gl::VERTEX_SHADER ? 0 : type == gl::FRAGMENT_SHADER ? 1 : type == gl::COMPUTE_SHADER ? 2 : -1;
        if (!r.ok || k < 0 || slot[k]) return fail("bad stage");
        sh[k].type = type, sh[k].source = sh[k].compiledSource = std::make_shared<const std::string>(src);
        vpglsl::ParseResult pr = vpglsl::parse(k == 0 ? vpglsl::Stage::Vertex : k == 1 ? vpglsl::Stage::Fragment : vpglsl::Stage::Compute, *sh[k].source);
        sh[k].parsed = pr.shader, sh[k].compiled = pr.shader != nullptr, sh[k].log = pr.log;
        if (!sh[k].compiled) return fail("shader no longer compiles");
        slot[k] = &sh[k];
    }
    std::map<std::string, int> bindings;
    const uint32_t nBind = r.u32();
    for (uint32_t i = 0; r.ok && i < nBind; ++i) {
        const uint32_t index = r.u32();
        const std::string name = r.str();
        if (r.ok) bindings[name] = int(index);
    }
    if (!r.ok || r.at != size) return fail("truncated payload");
    p.attribBindings = bindings;
    p.separable = separable;
    return linkShaders(p, slot[0], slot[1], slot[2]);
}

bool Context::linkProgram(Program& p) { return linkShaders(p, shader(p.vs), shader(p.fs), shader(p.cs)); }

bool Context::linkShaders(Program& p, Shader* vs, Shader* fs, Shader* cs) {
    vpgpu::HitchSpan hitch(vpgpu::hitchPerf().glLinkUs);  // vk.hitch gles_link_ms (glLinkProgram / glProgramBinary)
    static uint64_t serials = 0;
    // Old objects: the pipelines may be referenced by recorded work; finish it first (link is rare).
    if (p.vsModule || p.csModule) {
        flush(true);
        for (auto& [k, pl] : p.pipelines) dev->vk.vkDestroyPipeline(dev->device, pl, nullptr);
        p.pipelines.clear();
        dev->vk.vkDestroyShaderModule(dev->device, p.vsModule, nullptr);
        dev->vk.vkDestroyShaderModule(dev->device, p.fsModule, nullptr);
        dev->vk.vkDestroyShaderModule(dev->device, p.csModule, nullptr);
        dev->vk.vkDestroyPipelineLayout(dev->device, p.pipeLayout, nullptr);
        dev->vk.vkDestroyDescriptorSetLayout(dev->device, p.setLayout, nullptr);
        p.vsModule = p.fsModule = p.csModule = VK_NULL_HANDLE;
        p.pipeLayout = VK_NULL_HANDLE, p.setLayout = VK_NULL_HANDLE;
    }
    p.linked = false;
    p.bin = {};
    p.log.clear();
    p.varyingDiag.clear();
    vpglsl::LinkResult lr;
    vpglsl::LinkOptions lo;
    lo.attribBindings = p.attribBindings;
    if (cs) {
        if (!cs->compiled) {
            p.log = "ERROR: compute shader not compiled\n";
            return false;
        }
        lr = vpglsl::link(cs->parsed, nullptr, lo);
    } else {
        if (!vs || !fs || !vs->compiled || !fs->compiled) {
            p.log = "ERROR: a vertex and a fragment shader, both compiled, must be attached\n";
            return false;
        }
        lr = vpglsl::link(vs->parsed, fs->parsed, lo);
    }
    p.log = lr.log;
    p.varyingDiag = lr.varyingDiag;
    logLenient(lr.lenient, "program", p.name);
    if (!lr.ok) return false;
    p.refl = lr.reflection;
    auto module = [&](const std::vector<uint32_t>& w) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = w.size() * 4, ci.pCode = w.data();
        VkShaderModule m = VK_NULL_HANDLE;
        dev->vk.vkCreateShaderModule(dev->device, &ci, nullptr, &m);
        return m;
    };
    if (cs) p.csModule = module(lr.computeSpirv);
    else p.vsModule = module(lr.vertexSpirv), p.fsModule = module(lr.fragmentSpirv);
    // Descriptor layout: binding 0 default block (dynamic UBO), samplers, uniform blocks.
    std::vector<VkDescriptorSetLayoutBinding> binds;
    const VkShaderStageFlags all = cs ? VK_SHADER_STAGE_COMPUTE_BIT : VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    binds.push_back({vpglsl::kDefaultBlockBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, all, nullptr});
    for (const auto& u : p.refl.uniforms)
        if (u.binding >= 0)
            binds.push_back({uint32_t(u.binding),
                             isImageType(u.glType)         ? (imageDim(u.glType) == 5 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                                                                                     : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                             : isBufferSampler(u.glType) ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                                                         : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                             uint32_t(u.arraySize), all, nullptr});
    {  // Shader storage blocks (one descriptor array per declaration) and atomic counter buffers.
        std::map<uint32_t, uint32_t> storageCounts;
        for (const auto& b : p.refl.storageBlocks) storageCounts[b.binding] = std::max(storageCounts[b.binding], b.element + 1);
        for (auto [b, n] : storageCounts) binds.push_back({b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n, all, nullptr});
        for (const auto& a : p.refl.atomicBuffers) binds.push_back({a.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, all, nullptr});
    }
    std::map<uint32_t, uint32_t> blockCounts;
    for (const auto& b : p.refl.blocks) blockCounts[b.binding] = std::max(blockCounts[b.binding], b.element + 1);
    for (auto [b, n] : blockCounts) binds.push_back({b, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, n, all, nullptr});
    VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dci.bindingCount = uint32_t(binds.size()), dci.pBindings = binds.data();
    dev->vk.vkCreateDescriptorSetLayout(dev->device, &dci, nullptr, &p.setLayout);
    VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pci.setLayoutCount = 1, pci.pSetLayouts = &p.setLayout;
    dev->vk.vkCreatePipelineLayout(dev->device, &pci, nullptr, &p.pipeLayout);
    // Uniform storage, locations (one per array element; struct leaves flattened), sampler units.
    p.defaultBlock.assign(std::max<uint32_t>(16, p.refl.defaultBlockSize), 0);
    p.defaultDirty = true;
    p.locations.clear();
    p.samplerUnits.assign(p.refl.uniforms.size(), {});
    for (size_t i = 0; i < p.refl.uniforms.size(); ++i) {
        const auto& u = p.refl.uniforms[i];
        if (u.uniformLocation >= 0) {
            if (p.locations.size() < size_t(u.uniformLocation + u.arraySize)) p.locations.resize(size_t(u.uniformLocation + u.arraySize), {-1, 0});
            for (int e = 0; e < u.arraySize; ++e) p.locations[size_t(u.uniformLocation + e)] = {int(i), e};
        }
        if (u.binding >= 0) {
            // layout(binding = N) gives the initial unit (consecutive for arrays).
            p.samplerUnits[i].assign(size_t(u.arraySize), 0);
            if (u.unit >= 0)
                for (int e = 0; e < u.arraySize; ++e) p.samplerUnits[i][size_t(e)] = u.unit + e;
        }
    }
    p.blockBindings.assign(p.refl.blocks.size(), 0);
    for (size_t i = 0; i < p.refl.blocks.size(); ++i)
        if (p.refl.blocks[i].initialBinding >= 0) p.blockBindings[i] = GLuint(p.refl.blocks[i].initialBinding);
    p.storageBindings.assign(p.refl.storageBlocks.size(), 0);
    for (size_t i = 0; i < p.refl.storageBlocks.size(); ++i)
        if (p.refl.storageBlocks[i].initialBinding >= 0) p.storageBindings[i] = GLuint(p.refl.storageBlocks[i].initialBinding);
    p.serial = ++serials;
    p.linked = true;
    p.bin.valid = true, p.bin.separable = p.separable, p.bin.attribBindings = p.attribBindings;
    p.bin.sources[0] = vs ? vs->compiledSource : nullptr;
    p.bin.sources[1] = fs ? fs->compiledSource : nullptr;
    p.bin.sources[2] = cs ? cs->compiledSource : nullptr;
    return true;
}

// Location of element 0 of uniform `name` ("a", "a[3]", "s.f", "s[1].f"), -1 if none.
static GLint locationOf(const Program& p, const std::string& name) {
    std::string base = name;
    int element = 0;
    if (!base.empty() && base.back() == ']') {
        size_t lb = base.rfind('[');
        if (lb == std::string::npos) return -1;
        element = std::atoi(base.c_str() + lb + 1);
        base = base.substr(0, lb);
    }
    for (const auto& u : p.refl.uniforms) {
        std::string un = u.name;
        if (u.isArray && un.size() > 3 && un.compare(un.size() - 3, 3, "[0]") == 0) un.resize(un.size() - 3);
        if (un == base && element < u.arraySize && (element == 0 || u.isArray)) return u.uniformLocation < 0 ? -1 : u.uniformLocation + element;
    }
    return -1;
}
GLint uniformLocation(const Program& p, const char* name) { return locationOf(p, name ? name : ""); }

// GLSL name of a default-block uniform's GL type enum (hex when it is not one setUniform handles).
static std::string uniformTypeName(uint32_t t) {
    switch (t) {
        case 0x1406: return "float";
        case 0x1404: return "int";
        case 0x1405: return "uint";
        case 0x8B56: return "bool";
        case 0x8B5A: return "mat2";
        case 0x8B5B: return "mat3";
        case 0x8B5C: return "mat4";
        case 0x8B65: return "mat2x3";
        case 0x8B66: return "mat2x4";
        case 0x8B67: return "mat3x2";
        case 0x8B68: return "mat3x4";
        case 0x8B69: return "mat4x2";
        case 0x8B6A: return "mat4x3";
        default: break;
    }
    if (t >= 0x8B50 && t <= 0x8B52) return "vec" + std::to_string(t - 0x8B50 + 2);
    if (t >= 0x8B53 && t <= 0x8B55) return "ivec" + std::to_string(t - 0x8B53 + 2);
    if (t >= 0x8DC6 && t <= 0x8DC8) return "uvec" + std::to_string(t - 0x8DC6 + 2);
    if (t >= 0x8B57 && t <= 0x8B59) return "bvec" + std::to_string(t - 0x8B57 + 2);
    char hex[16];
    std::snprintf(hex, sizeof hex, "0x%x", t);
    return hex;
}

// What a glUniform* call said the uniform was ("vec4", "ivec2", "mat3x4"...).
static std::string callTypeName(int comps, int cols, bool isInt, bool isUint) {
    if (cols) return cols == comps ? "mat" + std::to_string(cols) : "mat" + std::to_string(cols) + "x" + std::to_string(comps);
    if (comps == 1) return isInt ? "int" : isUint ? "uint" : "float";
    return std::string(isInt ? "i" : isUint ? "u" : "") + "vec" + std::to_string(comps);
}

// Diagnostics: a glUniform* the layer rejected, with the uniform that was addressed (gles.diag only says which source
// line rejected it). At most 64 lines per process, one per uniform name (per reason and location when it is unknown).
static void logUniformErr(const vpglsl::Variable* u, GLint location, GLsizei count, int comps, int cols, bool isInt,
                          bool isUint, const char* reason, int element = -1) {
    static std::mutex mu;
    static std::set<std::string> seen;
    const std::string name = u ? u->name : std::string("?");
    std::lock_guard<std::mutex> g(mu);
    if (seen.size() >= 64 || !seen.insert(u ? name : name + reason + std::to_string(location)).second) return;
    logf("gles.uniform_err name=%s decl_type=%s call_type=%s count=%d arraySize=%d loc=%d reason=%s element=%d base=%d", name.c_str(),
         u ? uniformTypeName(u->glType).c_str() : "?", callTypeName(comps, cols, isInt, isUint).c_str(), int(count),
         u ? u->arraySize : 0, int(location), reason, element, element >= 0 ? int(location) - element : -1);
}

void Context::setUniform(GLint location, int comps, Kind kind, GLsizei count, const void* v, int cols, bool transpose,
                         Program* target) {
    Program* p = target ? target : uniformProgram();
    const bool isInt = kind == Kind::Int, isUint = kind == Kind::Uint;
    if (!p || !p->linked) {
        logUniformErr(nullptr, location, count, comps, cols, isInt, isUint, "no_linked_program");
        return setError(gl::INVALID_OPERATION);
    }
    if (location == -1) return;
    if (location < 0 || size_t(location) >= p->locations.size()) {
        logUniformErr(nullptr, location, count, comps, cols, isInt, isUint, "location_out_of_range");
        return setError(gl::INVALID_OPERATION);
    }
    const Program::Loc loc = p->locations[size_t(location)];
    if (loc.uniform < 0) {  // a gap between explicit locations
        logUniformErr(nullptr, location, count, comps, cols, isInt, isUint, "location_unused");
        return setError(gl::INVALID_OPERATION);
    }
    const vpglsl::Variable& u = p->refl.uniforms[size_t(loc.uniform)];
    auto note = [&](const char* reason) { logUniformErr(&u, location, count, comps, cols, isInt, isUint, reason, loc.element); };
    if (count > 1 && !u.isArray) return note("count_on_non_array"), setError(gl::INVALID_OPERATION);
    const int n = std::min<int>(count, u.arraySize - loc.element);
    // Samplers: glUniform1i(v) sets the texture unit.
    if (u.binding >= 0) {
        if (kind != Kind::Int || comps != 1 || cols) return note("sampler_needs_1i"), setError(gl::INVALID_OPERATION);
        const GLint* iv = static_cast<const GLint*>(v);
        for (int i = 0; i < n; ++i) {
            if (iv[i] < 0 || iv[i] >= (isImageType(u.glType) ? kMaxImageUnits : kMaxTextureUnits))
                return note("sampler_unit_out_of_range"), setError(gl::INVALID_VALUE);
            p->samplerUnits[size_t(loc.uniform)][size_t(loc.element + i)] = iv[i];
        }
        return;
    }
    // Type check: components and base type (bool accepts any of float/int/uint).
    const uint32_t t = u.glType;
    int ucomps = 1, ucols = 0;
    Kind ukind = Kind::Float;
    switch (t) {
        case 0x1406: ucomps = 1; break;
        case 0x8B50: ucomps = 2; break;
        case 0x8B51: ucomps = 3; break;
        case 0x8B52: ucomps = 4; break;
        case 0x1404: ukind = Kind::Int, ucomps = 1; break;
        case 0x8B53: ukind = Kind::Int, ucomps = 2; break;
        case 0x8B54: ukind = Kind::Int, ucomps = 3; break;
        case 0x8B55: ukind = Kind::Int, ucomps = 4; break;
        case 0x1405: ukind = Kind::Uint, ucomps = 1; break;
        case 0x8DC6: ukind = Kind::Uint, ucomps = 2; break;
        case 0x8DC7: ukind = Kind::Uint, ucomps = 3; break;
        case 0x8DC8: ukind = Kind::Uint, ucomps = 4; break;
        case 0x8B56: ukind = Kind::Bool, ucomps = 1; break;
        case 0x8B57: ukind = Kind::Bool, ucomps = 2; break;
        case 0x8B58: ukind = Kind::Bool, ucomps = 3; break;
        case 0x8B59: ukind = Kind::Bool, ucomps = 4; break;
        case 0x8B5A: ucols = 2, ucomps = 2; break;
        case 0x8B5B: ucols = 3, ucomps = 3; break;
        case 0x8B5C: ucols = 4, ucomps = 4; break;
        case 0x8B65: ucols = 2, ucomps = 3; break;
        case 0x8B66: ucols = 2, ucomps = 4; break;
        case 0x8B67: ucols = 3, ucomps = 2; break;
        case 0x8B68: ucols = 3, ucomps = 4; break;
        case 0x8B69: ucols = 4, ucomps = 2; break;
        case 0x8B6A: ucols = 4, ucomps = 3; break;
        default: return note("uniform_type_not_settable"), setError(gl::INVALID_OPERATION);
    }
    if (ucols != cols || ucomps != comps) return note("size_mismatch"), setError(gl::INVALID_OPERATION);
    if (ukind != Kind::Bool && ukind != kind) return note("base_type_mismatch"), setError(gl::INVALID_OPERATION);
    if (ukind == Kind::Bool && cols) return note("matrix_on_bool"), setError(gl::INVALID_OPERATION);
    const uint32_t stride = u.arrayStride ? u.arrayStride : 16;
    for (int i = 0; i < n; ++i) {
        uint8_t* dst = p->defaultBlock.data() + u.offset + uint32_t(loc.element + i) * stride;
        if (cols) {
            const float* m = static_cast<const float*>(v) + size_t(i) * size_t(cols * comps);
            for (int c = 0; c < cols; ++c)
                for (int r = 0; r < comps; ++r) {
                    float x = transpose ? m[r * cols + c] : m[c * comps + r];
                    std::memcpy(dst + c * 16 + r * 4, &x, 4);
                }
            continue;
        }
        const uint8_t* src = static_cast<const uint8_t*>(v) + size_t(i) * size_t(comps) * 4;
        if (ukind == Kind::Bool) {
            for (int c = 0; c < comps; ++c) {
                uint32_t b;
                if (kind == Kind::Float) {
                    float f;
                    std::memcpy(&f, src + 4 * c, 4);
                    b = f != 0.0f;
                } else {
                    uint32_t x;
                    std::memcpy(&x, src + 4 * c, 4);
                    b = x != 0;
                }
                std::memcpy(dst + 4 * c, &b, 4);
            }
        } else {
            std::memcpy(dst, src, size_t(comps) * 4);
        }
    }
    p->defaultDirty = true;
}

void Context::getUniform(Program& p, GLint location, void* out, Kind kind) {
    if (location < 0 || size_t(location) >= p.locations.size()) return setError(gl::INVALID_OPERATION);
    const Program::Loc loc = p.locations[size_t(location)];
    if (loc.uniform < 0) return setError(gl::INVALID_OPERATION);
    const vpglsl::Variable& u = p.refl.uniforms[size_t(loc.uniform)];
    if (u.binding >= 0) {
        GLint unit = p.samplerUnits[size_t(loc.uniform)][size_t(loc.element)];
        if (kind == Kind::Float) *static_cast<float*>(out) = float(unit);
        else *static_cast<GLint*>(out) = unit;
        return;
    }
    if (u.atomicBinding >= 0) {  // atomic counters live in their buffer, not in the default block: no value here
        if (kind == Kind::Float) *static_cast<float*>(out) = 0.0f;
        else *static_cast<GLint*>(out) = 0;
        return;
    }
    const uint32_t t = u.glType;
    int comps = 4, cols = 1;
    if (t == 0x1406 || t == 0x1404 || t == 0x1405 || t == 0x8B56) comps = 1;
    else if (t == 0x8B50 || t == 0x8B53 || t == 0x8DC6 || t == 0x8B57) comps = 2;
    else if (t == 0x8B51 || t == 0x8B54 || t == 0x8DC7 || t == 0x8B58) comps = 3;
    else if (t == 0x8B5A) cols = 2, comps = 2;
    else if (t == 0x8B5B) cols = 3, comps = 3;
    else if (t == 0x8B5C) cols = 4, comps = 4;
    const bool isFloat = t == 0x1406 || (t >= 0x8B50 && t <= 0x8B52) || (t >= 0x8B5A && t <= 0x8B5C) || (t >= 0x8B65 && t <= 0x8B6A);
    const bool isUint = t == 0x1405 || (t >= 0x8DC6 && t <= 0x8DC8);
    const uint8_t* src = p.defaultBlock.data() + u.offset + uint32_t(loc.element) * (u.arrayStride ? u.arrayStride : 16);
    int k = 0;
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < comps; ++r, ++k) {
            uint32_t bits;
            std::memcpy(&bits, src + c * 16 + r * 4, 4);
            double v;
            if (isFloat) {
                float f;
                std::memcpy(&f, &bits, 4);
                v = f;
            } else if (isUint) v = double(bits);
            else v = double(int32_t(bits));
            if (kind == Kind::Float) static_cast<float*>(out)[k] = float(v);
            else if (kind == Kind::Uint) static_cast<GLuint*>(out)[k] = GLuint(v);
            else static_cast<GLint*>(out)[k] = isFloat ? GLint(std::lround(v)) : GLint(v);
        }
}

// ---- pipelines --------------------------------------------------------------------------------------
static VkCompareOp cmpOf(GLenum f) { return VkCompareOp(f - gl::NEVER); }
static VkStencilOp stencilOpOf(GLenum o) {
    switch (o) {
        case gl::ZERO: return VK_STENCIL_OP_ZERO;
        case gl::REPLACE: return VK_STENCIL_OP_REPLACE;
        case gl::INCR: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case gl::DECR: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case gl::INVERT: return VK_STENCIL_OP_INVERT;
        case gl::INCR_WRAP: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case gl::DECR_WRAP: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}
static VkBlendFactor blendOf(GLenum f) {
    switch (f) {
        case gl::ZERO: return VK_BLEND_FACTOR_ZERO;
        case gl::ONE: return VK_BLEND_FACTOR_ONE;
        case gl::SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
        case gl::ONE_MINUS_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case gl::DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
        case gl::ONE_MINUS_DST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case gl::SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
        case gl::ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case gl::DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
        case gl::ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case gl::CONSTANT_COLOR: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case gl::ONE_MINUS_CONSTANT_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case gl::CONSTANT_ALPHA: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case gl::ONE_MINUS_CONSTANT_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        case gl::SRC_ALPHA_SATURATE: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
static VkBlendOp blendEqOf(GLenum e) {
    switch (e) {
        case gl::FUNC_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
        case gl::FUNC_REVERSE_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case gl::MIN: return VK_BLEND_OP_MIN;
        case gl::MAX: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

// FNV-1a over the pipeline-relevant state.
struct Hasher {
    uint64_t h = 1469598103934665603ull;
    template <class T>
    void add(const T& v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        for (size_t i = 0; i < sizeof(T); ++i) h = (h ^ p[i]) * 1099511628211ull;
    }
};

VkPipeline Context::pipelineFor(Program& p, const FbInfo& fb, VkPrimitiveTopology topo,
                                const std::vector<VkVertexInputBindingDescription>& binds,
                                const std::vector<VkVertexInputAttributeDescription>& attrs) {
    // Formats of the attachments the pass has.
    std::array<VkFormat, kMaxDrawBuffers> colorFmt{};
    uint32_t colorCount = 0;
    for (int i = 0; i < kMaxDrawBuffers; ++i)
        if (pass_.color[size_t(i)]) {
            colorFmt[size_t(i)] = fb.color[size_t(i)].image->format->vk;
            colorCount = uint32_t(i + 1);
        }
    const VkFormat depthFmt = pass_.depth ? fb.depth.image->format->vk : VK_FORMAT_UNDEFINED;
    const VkFormat stencilFmt = pass_.stencil ? fb.stencil.image->format->vk : VK_FORMAT_UNDEFINED;
    const bool hasDepth = depthFmt != VK_FORMAT_UNDEFINED && fb.depth.image->format->depth;
    const bool hasStencil = stencilFmt != VK_FORMAT_UNDEFINED && fb.stencil.image->format->stencil;
    Hasher h;
    h.add(topo);
    h.add(colorFmt);
    h.add(depthFmt), h.add(stencilFmt), h.add(pass_.viewMask);
    for (auto& b : binds) h.add(b);
    for (auto& a : attrs) h.add(a);
    h.add(rasterizerDiscard), h.add(cullFace), h.add(cullMode), h.add(frontFace), h.add(polygonOffsetFill);
    h.add(primitiveRestart), h.add(sampleAlphaToCoverage);
    const bool dt = depthTest && hasDepth, st = stencilTest && hasStencil;
    h.add(dt), h.add(dt ? depthFunc : 0u), h.add(dt && depthMask), h.add(st);
    if (st)
        for (auto& s : stencil) h.add(s.func), h.add(s.fail), h.add(s.zfail), h.add(s.zpass);
    for (uint32_t i = 0; i < colorCount; ++i) {
        const BlendState& b = blend[i];
        h.add(b.enabled);
        if (b.enabled) h.add(b.srcRGB), h.add(b.dstRGB), h.add(b.srcA), h.add(b.dstA), h.add(b.eqRGB), h.add(b.eqA);
        h.add(b.mask);
    }
    const uint64_t key = h.h;
    auto it = p.pipelines.find(key);
    if (it != p.pipelines.end()) return it->second;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, p.vsModule, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, p.fsModule, "main", nullptr};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = uint32_t(binds.size()), vi.pVertexBindingDescriptions = binds.data();
    vi.vertexAttributeDescriptionCount = uint32_t(attrs.size()), vi.pVertexAttributeDescriptions = attrs.data();
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = topo;
    ia.primitiveRestartEnable = primitiveRestart && (topo == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
                                                     topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
                                                     topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1, vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.rasterizerDiscardEnable = rasterizerDiscard;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = !cullFace ? VK_CULL_MODE_NONE
                  : cullMode == gl::FRONT ? VK_CULL_MODE_FRONT_BIT
                  : cullMode == gl::BACK ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_FRONT_AND_BACK;
    // GL rows bottom-up = Vulkan rows top-down in memory: the winding flips (see context.h).
    rs.frontFace = frontFace == gl::CCW ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.depthBiasEnable = polygonOffsetFill;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ms.alphaToCoverageEnable = sampleAlphaToCoverage;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = dt;
    ds.depthWriteEnable = dt && depthMask;
    ds.depthCompareOp = cmpOf(depthFunc);
    ds.stencilTestEnable = st;
    for (int f = 0; f < 2; ++f) {
        VkStencilOpState& s = f ? ds.back : ds.front;
        // Faces: GL front = what the flipped winding calls front (rs.frontFace above), unchanged.
        const StencilFace& g = stencil[f];
        s.failOp = stencilOpOf(g.fail), s.depthFailOp = stencilOpOf(g.zfail), s.passOp = stencilOpOf(g.zpass);
        s.compareOp = cmpOf(g.func);
    }
    VkPipelineColorBlendAttachmentState cba[kMaxDrawBuffers]{};
    for (uint32_t i = 0; i < colorCount; ++i) {
        const BlendState& b = blend[i];
        VkPipelineColorBlendAttachmentState& a = cba[i];
        a.blendEnable = b.enabled && !fb.color[i].image->format->integer;
        a.srcColorBlendFactor = blendOf(b.srcRGB), a.dstColorBlendFactor = blendOf(b.dstRGB);
        a.srcAlphaBlendFactor = blendOf(b.srcA), a.dstAlphaBlendFactor = blendOf(b.dstA);
        a.colorBlendOp = blendEqOf(b.eqRGB), a.alphaBlendOp = blendEqOf(b.eqA);
        a.colorWriteMask = (b.mask[0] ? 1u : 0u) | (b.mask[1] ? 2u : 0u) | (b.mask[2] ? 4u : 0u) | (b.mask[3] ? 8u : 0u);
        // RGB formats stored as RGBA keep alpha = 1.
        if (fb.color[i].image->format->swz[3] == VK_COMPONENT_SWIZZLE_ONE) a.colorWriteMask &= ~8u;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = colorCount, cb.pAttachments = cba;
    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                  VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                  VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 7, dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    pr.viewMask = pass_.viewMask;
    pr.colorAttachmentCount = colorCount;
    pr.pColorAttachmentFormats = colorFmt.data();
    pr.depthAttachmentFormat = hasDepth ? depthFmt : VK_FORMAT_UNDEFINED;
    pr.stencilAttachmentFormat = hasStencil ? stencilFmt : VK_FORMAT_UNDEFINED;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &pr};
    ci.stageCount = 2, ci.pStages = stages;
    ci.pVertexInputState = &vi, ci.pInputAssemblyState = &ia, ci.pViewportState = &vp, ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms, ci.pDepthStencilState = &ds, ci.pColorBlendState = &cb, ci.pDynamicState = &dy;
    ci.layout = p.pipeLayout;
    VkPipeline pl = VK_NULL_HANDLE;
    if (dev->vk.vkCreateGraphicsPipelines(dev->device, VK_NULL_HANDLE, 1, &ci, nullptr, &pl) != VK_SUCCESS) pl = VK_NULL_HANDLE;
    p.pipelines[key] = pl;
    recipeStatsTick(false);  // a new pipeline: the log must not wait for a swap the host never calls
    return pl;
}

// ---- resources --------------------------------------------------------------------------------------
static GLuint unitTexture(const Context::Unit& u, const vpglsl::Variable& v) {
    switch (v.glType) {
        case 0x8B60: case 0x8DC5: case 0x8DCC: case 0x8DD4: return u.texCube;
        case 0x8B5F: case 0x8DCB: case 0x8DD3: return u.tex3D;
        case 0x8DC1: case 0x8DC4: case 0x8DCF: case 0x8DD7: return u.tex2DArray;
        case 0x8D66: return u.texExternal ? u.texExternal : u.tex2D;
        case 0x9108: case 0x9109: case 0x910A: return u.tex2DMS;
        case gl::SAMPLER_BUFFER: case gl::INT_SAMPLER_BUFFER: case gl::UNSIGNED_INT_SAMPLER_BUFFER: return u.texBuffer;
        default: return u.tex2D;
    }
}

// Texture completeness (GLES 3.0 §8.17): an incomplete texture samples as (0, 0, 0, 1). `sp` are the sampling parameters
// the unit really uses (a sampler object bound to the unit replaces the texture's own).
static bool complete(const Texture& t, const SamplerParams& sp) {
    if (!t.image || t.levels.empty() || !t.levels[0].format) return false;
    const bool mip = sp.minFilter != gl::NEAREST && sp.minFilter != gl::LINEAR;
    if (!mip || t.immutable) return true;
    const uint32_t base = uint32_t(std::max(0, t.baseLevel));
    if (base >= t.levels.size()) return false;
    const uint32_t last = std::min<uint32_t>(uint32_t(t.maxLevel), t.image->levels - 1);
    for (uint32_t l = base; l <= last; ++l) {
        if (l >= t.levels.size() || t.levels[l].format != t.image->format) return false;
        if (t.levels[l].w != std::max(1u, t.image->width >> l) || t.levels[l].h != std::max(1u, t.image->height >> l)) return false;
    }
    return true;
}

bool Context::bindResources(Program& p) {
    VkDescriptorSet set = allocSet(p.setLayout);
    if (!set) return false;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufInfos;
    std::vector<VkDescriptorImageInfo> imgInfos;
    bufInfos.reserve(1 + p.refl.blocks.size());
    std::vector<VkBufferView> texelViews;
    size_t imgCount = 0, texelCount = 0;
    for (const auto& u : p.refl.uniforms)
        if (u.binding >= 0)
            (isBufferSampler(u.glType) || (isImageType(u.glType) && imageDim(u.glType) == 5) ? texelCount : imgCount) += size_t(u.arraySize);
    bufInfos.reserve(1 + p.refl.blocks.size() + p.refl.storageBlocks.size() + p.refl.atomicBuffers.size());
    imgInfos.reserve(imgCount);
    texelViews.reserve(texelCount);
    // Default block: this draw's copy in the stream (offset passed as the dynamic offset).
    Alloc ub = stream(p.defaultBlock.size(), dev->props.limits.minUniformBufferOffsetAlignment);
    std::memcpy(ub.host, p.defaultBlock.data(), p.defaultBlock.size());
    bufInfos.push_back({ub.buffer, 0, p.defaultBlock.size()});
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set, w.dstBinding = vpglsl::kDefaultBlockBinding, w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    w.pBufferInfo = &bufInfos.back();
    writes.push_back(w);
    for (size_t i = 0; i < p.refl.uniforms.size(); ++i) {
        const auto& u = p.refl.uniforms[i];
        if (u.binding < 0) continue;
        if (isImageType(u.glType) && imageDim(u.glType) == 5) {  // imageBuffer: a storage texel buffer over the unit's buffer texture
            const size_t firstView = texelViews.size();
            for (int e = 0; e < u.arraySize; ++e) {
                const ImageUnit& iu = images[size_t(p.samplerUnits[i][size_t(e)]) % kMaxImageUnits];
                Texture* t = iu.texture ? texture(iu.texture) : nullptr;
                VkBufferView v = t ? texelBufferView(*t) : VK_NULL_HANDLE;
                if (!v) {
                    // The 16-byte stand-in of unbound samplerBuffers must stay zero and a store through it would run out of
                    // bounds, so a draw / dispatch with an unbound (or invalid) imageBuffer is skipped.
                    static std::atomic<int> said{0};
                    if (said.fetch_add(1) < 8) logf("gles.warn what=image_buffer_unbound program=%u unit=%d (draw / dispatch skipped)", p.name,
                                                    p.samplerUnits[i][size_t(e)]);
                    return false;
                }
                keep(t->bufferView);
                texelViews.push_back(v);
            }
            VkWriteDescriptorSet s{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            s.dstSet = set, s.dstBinding = uint32_t(u.binding), s.descriptorCount = uint32_t(u.arraySize);
            s.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            s.pTexelBufferView = &texelViews[firstView];
            writes.push_back(s);
            continue;
        }
        if (isImageType(u.glType)) {
            const size_t firstImg = imgInfos.size();
            const int dim = imageDim(u.glType);
            for (int e = 0; e < u.arraySize; ++e) {
                const ImageUnit& iu = images[size_t(p.samplerUnits[i][size_t(e)]) % kMaxImageUnits];
                Texture* t = iu.texture ? texture(iu.texture) : nullptr;
                VkDescriptorImageInfo ii{};
                ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                if (t && t->image && t->image->storage && dim != 5 && uint32_t(iu.level) < t->image->levels) {
                    Image& im = *t->image;
                    im.invalid.clear();
                    // The texel format the shader reads / writes with (glBindImageTexture's format) when it differs from the
                    // texture's: a view of the same texel size.
                    const VkFormat vf = imageViewFormat(iu.format);
                    const VkFormat fmt = vf != VK_FORMAT_UNDEFINED && imageFormatBytes(vf) == imageFormatBytes(im.format->vk) ? vf : VK_FORMAT_UNDEFINED;
                    // image3D: the whole 3D level; layered image2DArray / imageCube: all layers; image2D: one layer.
                    if (dim == 2) ii.imageView = im.storageView(uint32_t(iu.level), 0, 1, VK_IMAGE_VIEW_TYPE_3D, fmt);
                    else if (dim == 7 || dim == 4 || dim == 8) ii.imageView = im.storageView(uint32_t(iu.level), 0, im.layers, VK_IMAGE_VIEW_TYPE_2D_ARRAY, fmt);
                    else if (im.type == VK_IMAGE_TYPE_3D) ii.imageView = sliceView(*t, iu.level, iu.layered ? 0 : iu.layer, iu.access, fmt);
                    else ii.imageView = im.storageView(uint32_t(iu.level), im.layers > 1 && !iu.layered ? uint32_t(iu.layer) : 0, 1, VK_IMAGE_VIEW_TYPE_2D, fmt);
                    if (!ii.imageView) ii.imageView = blackTexture().image->storageView(0, 0, 1, VK_IMAGE_VIEW_TYPE_2D);
                    keep(t->image);
                } else {
                    ii.imageView = blackTexture().image->storageView(0, 0, 1, VK_IMAGE_VIEW_TYPE_2D);  // unbound unit: reads 0 (undefined in GL)
                }
                imgInfos.push_back(ii);
            }
            VkWriteDescriptorSet s{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            s.dstSet = set, s.dstBinding = uint32_t(u.binding), s.descriptorCount = uint32_t(u.arraySize);
            s.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            s.pImageInfo = &imgInfos[firstImg];
            writes.push_back(s);
            continue;
        }
        if (isBufferSampler(u.glType)) {
            const size_t firstView = texelViews.size();
            for (int e = 0; e < u.arraySize; ++e) {
                const Unit& un = units[size_t(p.samplerUnits[i][size_t(e)])];
                const GLuint name = unitTexture(un, u);
                Texture* t = name ? texture(name) : nullptr;
                VkBufferView v = t ? texelBufferView(*t) : VK_NULL_HANDLE;
                if (v) keep(t->bufferView);
                texelViews.push_back(v ? v : blackTexelBufferView());
            }
            VkWriteDescriptorSet s{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            s.dstSet = set, s.dstBinding = uint32_t(u.binding), s.descriptorCount = uint32_t(u.arraySize);
            s.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            s.pTexelBufferView = &texelViews[firstView];
            writes.push_back(s);
            continue;
        }
        const size_t first = imgInfos.size();
        for (int e = 0; e < u.arraySize; ++e) {
            const int unit = p.samplerUnits[i][size_t(e)];
            const Unit& un = units[size_t(unit)];
            GLuint name = unitTexture(un, u);
            Texture* t = name ? texture(name) : nullptr;
            VkDescriptorImageInfo ii{};
            ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            SamplerParams sp;
            if (t) {
                sp = t->sampler;
                if (un.sampler)
                    if (auto it = share->samplers.find(un.sampler); it != share->samplers.end()) sp = it->second->params;
            }
            if (t && complete(*t, sp)) {
                ii.sampler = samplerFor(sp, *t);
                ii.imageView = t->sampledView();
                keep(t->image);
                t->image->invalid.clear();
            } else {
                // Incomplete/missing: a 1x1 black opaque texture.
                Texture& b = blackTexture();
                ii.sampler = samplerFor(b.sampler, b);
                ii.imageView = b.sampledView();
            }
            imgInfos.push_back(ii);
        }
        VkWriteDescriptorSet s{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        s.dstSet = set, s.dstBinding = uint32_t(u.binding), s.descriptorCount = uint32_t(u.arraySize);
        s.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        s.pImageInfo = &imgInfos[first];
        writes.push_back(s);
    }
    for (size_t i = 0; i < p.refl.blocks.size(); ++i) {
        const auto& b = p.refl.blocks[i];
        const IndexedBuffer& ib = uniformBindings[p.blockBindings[i] % kMaxUniformBufferBindings];
        Buffer* buf = buffer(ib.buffer);
        VkDescriptorBufferInfo bi{};
        if (buf && buf->storage && VkDeviceSize(ib.offset) < VkDeviceSize(buf->size)) {
            useStorage(buf->storage);
            bi.buffer = buf->storage->buffer;
            bi.offset = VkDeviceSize(ib.offset);
            const VkDeviceSize avail = VkDeviceSize(buf->size) - VkDeviceSize(ib.offset);  // the buffer may have shrunk since the bind
            bi.range = ib.size ? std::min<VkDeviceSize>(VkDeviceSize(ib.size), avail) : avail;
        } else {
            Alloc z = stream(std::max<uint32_t>(16, b.size), 256);
            std::memset(z.host, 0, std::max<uint32_t>(16, b.size));
            bi.buffer = z.buffer, bi.offset = z.offset, bi.range = std::max<uint32_t>(16, b.size);
        }
        bufInfos.push_back(bi);
        VkWriteDescriptorSet s{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        s.dstSet = set, s.dstBinding = b.binding, s.dstArrayElement = b.element, s.descriptorCount = 1;
        s.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        s.pBufferInfo = &bufInfos.back();
        writes.push_back(s);
    }
    // Shader storage blocks (GL_SHADER_STORAGE_BUFFER binding points) and atomic counter buffers.
    auto storageWrite = [&](const IndexedBuffer& ib, uint32_t binding, uint32_t element, uint32_t minSize) {
        Buffer* buf = buffer(ib.buffer);
        VkDescriptorBufferInfo bi{};
        if (buf && buf->storage && VkDeviceSize(ib.offset) < VkDeviceSize(buf->size)) {
            useStorage(buf->storage);
            bi.buffer = buf->storage->buffer;
            bi.offset = VkDeviceSize(ib.offset);
            const VkDeviceSize avail = VkDeviceSize(buf->size) - VkDeviceSize(ib.offset);  // the buffer may have shrunk since the bind
            bi.range = ib.size ? std::min<VkDeviceSize>(VkDeviceSize(ib.size), avail) : avail;
        } else {  // nothing bound: a zeroed scratch range, so the shader reads 0 and its writes go nowhere
            const uint32_t n = std::max<uint32_t>(16, minSize);
            Alloc z = stream(n, 256);
            std::memset(z.host, 0, n);
            bi.buffer = z.buffer, bi.offset = z.offset, bi.range = n;
        }
        bufInfos.push_back(bi);
        VkWriteDescriptorSet s{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        s.dstSet = set, s.dstBinding = binding, s.dstArrayElement = element, s.descriptorCount = 1;
        s.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        s.pBufferInfo = &bufInfos.back();
        writes.push_back(s);
    };
    for (size_t i = 0; i < p.refl.storageBlocks.size(); ++i) {
        const auto& b = p.refl.storageBlocks[i];
        storageWrite(ssboBindings[p.storageBindings[i] % kMaxShaderStorageBindings], b.binding, b.element, b.size);
    }
    for (const auto& a : p.refl.atomicBuffers) storageWrite(atomicBindings[size_t(a.glBinding) % kMaxAtomicCounterBindings], a.binding, 0, a.size);
    dev->vk.vkUpdateDescriptorSets(dev->device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    const uint32_t dynOff = uint32_t(ub.offset);
    dev->vk.vkCmdBindDescriptorSets(cmd(), p.csModule ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    p.pipeLayout, 0, 1, &set, 1, &dynOff);
    return true;
}

Texture& Context::blackTexture() {
    if (!black_) {
        black_ = std::make_unique<Texture>();
        black_->target = gl::TEXTURE_2D;
        black_->levels = {{1, 1, 1, formatFor(gl::RGBA8)}};
        black_->sampler.minFilter = gl::NEAREST, black_->sampler.magFilter = gl::NEAREST;
        black_->wantStorage = true;  // also the stand-in for unbound image units (1x1: no compression to lose)
        realizeTexture(*black_);
        const uint8_t px[4] = {0, 0, 0, 255};
        copyBufferToImage(*black_->image, 0, 0, 0, 0, 0, 1, 1, 1, px, 4);
    }
    return *black_;
}

// ---- texel buffers ------------------------------------------------------------------------------------
VkBufferView Context::texelBufferView(Texture& t) {
    Buffer* b = t.bufferName ? buffer(t.bufferName) : nullptr;
    if (!b || !b->storage || !t.bufferFormat || t.bufferSize <= 0) return VK_NULL_HANDLE;
    if (VkDeviceSize(t.bufferOffset + t.bufferSize) > b->storage->size) return VK_NULL_HANDLE;
    TexelBufferView* cur = t.bufferView.get();
    if (!cur || cur->storage != b->storage || cur->format != t.bufferFormat->vk || cur->offset != VkDeviceSize(t.bufferOffset) ||
        cur->range != VkDeviceSize(t.bufferSize)) {
        auto nv = std::make_shared<TexelBufferView>();
        nv->dev = dev, nv->storage = b->storage, nv->format = t.bufferFormat->vk;
        nv->offset = VkDeviceSize(t.bufferOffset), nv->range = VkDeviceSize(t.bufferSize);
        VkBufferViewCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
        ci.buffer = b->storage->buffer, ci.format = nv->format, ci.offset = nv->offset, ci.range = nv->range;
        if (dev->vk.vkCreateBufferView(dev->device, &ci, nullptr, &nv->view) != VK_SUCCESS) return VK_NULL_HANDLE;
        if (t.bufferView) keep(t.bufferView);
        t.bufferView = nv;
    }
    useStorage(b->storage);
    return t.bufferView->view;
}

// A zeroed 16-byte texel buffer for samplerBuffer uniforms with no buffer attached (fetches read 0).
VkBufferView Context::blackTexelBufferView() {
    if (!blackTexelView_) {
        auto nv = std::make_shared<TexelBufferView>();
        blackTexelStorage_ = dev->newStorage(16, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT);
        if (!blackTexelStorage_) return VK_NULL_HANDLE;
        std::memset(blackTexelStorage_->host, 0, 16);
        nv->dev = dev, nv->storage = blackTexelStorage_, nv->format = VK_FORMAT_R8G8B8A8_UNORM, nv->offset = 0, nv->range = 16;
        VkBufferViewCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
        ci.buffer = blackTexelStorage_->buffer, ci.format = nv->format, ci.offset = 0, ci.range = 16;
        if (dev->vk.vkCreateBufferView(dev->device, &ci, nullptr, &nv->view) != VK_SUCCESS) return VK_NULL_HANDLE;
        blackTexelView_ = nv;
    }
    useStorage(blackTexelStorage_);
    keep(blackTexelView_);
    return blackTexelView_->view;
}

void Context::setDynamicState(const FbInfo& fb) {
    VkCommandBuffer c = cmd();
    VkViewport v{float(viewport[0]), float(viewport[1]), float(viewport[2]), float(viewport[3]), depthRange[0], depthRange[1]};
    if (v.width <= 0) v.width = 1;
    if (v.height <= 0) v.height = 1;
    dev->vk.vkCmdSetViewport(c, 0, 1, &v);
    VkRect2D s{{0, 0}, {fb.width, fb.height}};
    if (scissorTest) {
        int32_t x0 = std::max(0, scissor[0]), y0 = std::max(0, scissor[1]);
        int32_t x1 = std::min<int32_t>(int32_t(fb.width), scissor[0] + std::max(0, scissor[2]));
        int32_t y1 = std::min<int32_t>(int32_t(fb.height), scissor[1] + std::max(0, scissor[3]));
        s = {{x0, y0}, {uint32_t(std::max(0, x1 - x0)), uint32_t(std::max(0, y1 - y0))}};
    }
    dev->vk.vkCmdSetScissor(c, 0, 1, &s);
    dev->vk.vkCmdSetStencilReference(c, VK_STENCIL_FACE_FRONT_BIT, uint32_t(stencil[0].ref));
    dev->vk.vkCmdSetStencilReference(c, VK_STENCIL_FACE_BACK_BIT, uint32_t(stencil[1].ref));
    dev->vk.vkCmdSetStencilCompareMask(c, VK_STENCIL_FACE_FRONT_BIT, stencil[0].mask);
    dev->vk.vkCmdSetStencilCompareMask(c, VK_STENCIL_FACE_BACK_BIT, stencil[1].mask);
    dev->vk.vkCmdSetStencilWriteMask(c, VK_STENCIL_FACE_FRONT_BIT, stencil[0].writeMask);
    dev->vk.vkCmdSetStencilWriteMask(c, VK_STENCIL_FACE_BACK_BIT, stencil[1].writeMask);
    dev->vk.vkCmdSetBlendConstants(c, blendColor);
    dev->vk.vkCmdSetDepthBias(c, polygonOffset[1], 0.0f, polygonOffset[0]);
}

// ---- vertex input -----------------------------------------------------------------------------------
static VkFormat attribFormat(const VertexAttrib& a) {
    const int n = a.size;
    auto pick = [&](VkFormat f1, VkFormat f2, VkFormat f3, VkFormat f4) { return n == 1 ? f1 : n == 2 ? f2 : n == 3 ? f3 : f4; };
    if (a.integer) {
        switch (a.type) {
            case gl::BYTE: return pick(VK_FORMAT_R8_SINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R8G8B8A8_SINT);
            case gl::UNSIGNED_BYTE: return pick(VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8A8_UINT);
            case gl::SHORT: return pick(VK_FORMAT_R16_SINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R16G16B16A16_SINT);
            case gl::UNSIGNED_SHORT: return pick(VK_FORMAT_R16_UINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16A16_UINT);
            case gl::INT: return pick(VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT);
            case gl::UNSIGNED_INT: return pick(VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT);
            default: return VK_FORMAT_UNDEFINED;
        }
    }
    switch (a.type) {
        case gl::FLOAT: return pick(VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT);
        case gl::HALF_FLOAT:
        case gl::HALF_FLOAT_OES: return pick(VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT);
        case gl::UNSIGNED_BYTE:
            return a.normalized ? pick(VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8A8_UNORM)
                                : pick(VK_FORMAT_R8_USCALED, VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8A8_USCALED);
        case gl::BYTE:
            return a.normalized ? pick(VK_FORMAT_R8_SNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8A8_SNORM)
                                : pick(VK_FORMAT_R8_SSCALED, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R8G8B8A8_SSCALED);
        case gl::UNSIGNED_SHORT:
            return a.normalized ? pick(VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16A16_UNORM)
                                : pick(VK_FORMAT_R16_USCALED, VK_FORMAT_R16G16_USCALED, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16A16_USCALED);
        case gl::SHORT:
            return a.normalized ? pick(VK_FORMAT_R16_SNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16A16_SNORM)
                                : pick(VK_FORMAT_R16_SSCALED, VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R16G16B16A16_SSCALED);
        case gl::INT_2_10_10_10_REV: return a.normalized ? VK_FORMAT_A2B10G10R10_SNORM_PACK32 : VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
        case gl::UNSIGNED_INT_2_10_10_10_REV: return a.normalized ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_A2B10G10R10_USCALED_PACK32;
        default: return VK_FORMAT_UNDEFINED;  // INT/UINT/FIXED as float: converted on the CPU below
    }
}

uint32_t vertexAttribBytes(const VertexAttrib& a) {
    uint32_t c = 4;
    switch (a.type) {
        case gl::BYTE: case gl::UNSIGNED_BYTE: c = 1; break;
        case gl::SHORT: case gl::UNSIGNED_SHORT: case gl::HALF_FLOAT: case gl::HALF_FLOAT_OES: c = 2; break;
        case gl::INT_2_10_10_10_REV: case gl::UNSIGNED_INT_2_10_10_10_REV: return 4;
        default: c = 4;
    }
    return c * uint32_t(a.size);
}

// Scaled/fixed formats VPVulkan does not read natively are converted to float32 on the CPU.
static bool needsCpuConversion(const VertexAttrib& a, VkFormat f) {
    return f == VK_FORMAT_UNDEFINED || (f >= VK_FORMAT_R8_USCALED && f <= VK_FORMAT_R8_SSCALED) ||
           f == VK_FORMAT_R8G8_USCALED || f == VK_FORMAT_R8G8_SSCALED || f == VK_FORMAT_R8G8B8_USCALED ||
           f == VK_FORMAT_R8G8B8_SSCALED || f == VK_FORMAT_R8G8B8A8_USCALED || f == VK_FORMAT_R8G8B8A8_SSCALED ||
           f == VK_FORMAT_R16_USCALED || f == VK_FORMAT_R16_SSCALED || f == VK_FORMAT_R16G16_USCALED ||
           f == VK_FORMAT_R16G16_SSCALED || f == VK_FORMAT_R16G16B16_USCALED || f == VK_FORMAT_R16G16B16_SSCALED ||
           f == VK_FORMAT_R16G16B16A16_USCALED || f == VK_FORMAT_R16G16B16A16_SSCALED ||
           f == VK_FORMAT_A2B10G10R10_SSCALED_PACK32 || f == VK_FORMAT_A2B10G10R10_USCALED_PACK32 ||
           f == VK_FORMAT_R16G16B16_UNORM || f == VK_FORMAT_R16G16B16_SNORM || f == VK_FORMAT_R8G8B8_SINT ||
           f == VK_FORMAT_R8G8B8_UINT || f == VK_FORMAT_R16G16B16_SINT || f == VK_FORMAT_R16G16B16_UINT ||
           (a.type == gl::FIXED);
}

static float readComponent(const VertexAttrib& a, const uint8_t* p, int c) {
    switch (a.type) {
        case gl::BYTE: { int8_t v = int8_t(p[c]); return a.normalized ? std::max(-1.0f, v / 127.0f) : float(v); }
        case gl::UNSIGNED_BYTE: return a.normalized ? p[c] / 255.0f : float(p[c]);
        case gl::SHORT: { int16_t v; std::memcpy(&v, p + 2 * c, 2); return a.normalized ? std::max(-1.0f, v / 32767.0f) : float(v); }
        case gl::UNSIGNED_SHORT: { uint16_t v; std::memcpy(&v, p + 2 * c, 2); return a.normalized ? v / 65535.0f : float(v); }
        case gl::INT: { int32_t v; std::memcpy(&v, p + 4 * c, 4); return a.normalized ? std::max(-1.0f, float(v / 2147483647.0)) : float(v); }
        case gl::UNSIGNED_INT: { uint32_t v; std::memcpy(&v, p + 4 * c, 4); return a.normalized ? float(v / 4294967295.0) : float(v); }
        case gl::FIXED: { int32_t v; std::memcpy(&v, p + 4 * c, 4); return float(v) / 65536.0f; }
        case gl::FLOAT: { float v; std::memcpy(&v, p + 4 * c, 4); return v; }
        case gl::INT_2_10_10_10_REV:
        case gl::UNSIGNED_INT_2_10_10_10_REV: {
            uint32_t v;
            std::memcpy(&v, p, 4);
            const int bits = c == 3 ? 2 : 10;
            uint32_t x = (v >> (10 * c)) & ((1u << bits) - 1);
            if (a.type == gl::UNSIGNED_INT_2_10_10_10_REV) return a.normalized ? float(x) / float((1u << bits) - 1) : float(x);
            int32_t s = int32_t(x << (32 - bits)) >> (32 - bits);
            return a.normalized ? std::max(-1.0f, float(s) / float((1 << (bits - 1)) - 1)) : float(s);
        }
        default: return 0.0f;
    }
}

Program* Context::uniformProgram() {
    if (program) return prog(program);
    auto it = programPipelines.find(programPipeline);
    return it == programPipelines.end() ? nullptr : prog(it->second->active);
}

Program* Context::graphicsProgram() {
    if (forcedProgram_) return forcedProgram_;
    if (program) return prog(program);
    auto it = programPipelines.find(programPipeline);
    if (it == programPipelines.end()) return nullptr;
    const ProgramPipeline& pl = *it->second;
    if (pl.vs && pl.vs == pl.fs) return prog(pl.vs);
    if (pl.vs && pl.fs) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            logf("gles.warn program_pipeline vs=%u fs=%u are different programs: separate vertex/fragment programs are not "
                 "supported (the linker needs both stages together), draws with this pipeline are skipped", pl.vs, pl.fs);
        }
    }
    return nullptr;
}

Program* Context::computeProgram() {
    if (program) return prog(program);
    auto it = programPipelines.find(programPipeline);
    return it == programPipelines.end() ? nullptr : prog(it->second->cs);
}

bool Context::prepareDraw(GLenum mode, VkPrimitiveTopology& topo, uint32_t vertexCount, uint32_t firstVertex,
                          uint32_t instances, bool indexed, VkBuffer&) {
    const bool provokeOk = provokeOk_;
    provokeOk_ = false;
    Program* p = graphicsProgram();
    if (!p || !p->linked || !p->vsModule) {
        setError(gl::INVALID_OPERATION);
        return false;
    }
    switch (mode) {
        case gl::POINTS: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case gl::LINES: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case gl::LINE_STRIP: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case gl::LINE_LOOP: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;  // closed by the caller
        case gl::TRIANGLES: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        case gl::TRIANGLE_STRIP: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break;
        case gl::TRIANGLE_FAN: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        default: setError(gl::INVALID_ENUM); return false;
    }
    // `flat` varyings: GL takes the value from a primitive's LAST vertex, Vulkan/Metal from the FIRST. The draw becomes a
    // triangle/line list whose primitives are rotated so the last vertex comes first (a cyclic rotation keeps the winding).
    // (An indexed strip with primitive restart keeps its topology: the restart cannot be rotated here.)
    provoke_ = false;
    if (provokeOk && p->refl.flatVaryings) {
        const bool strip = mode == gl::TRIANGLE_STRIP || mode == gl::TRIANGLE_FAN || mode == gl::LINE_STRIP || mode == gl::LINE_LOOP;
        if (!(indexed && strip && primitiveRestart)) {
            if (mode == gl::TRIANGLES || mode == gl::TRIANGLE_STRIP || mode == gl::TRIANGLE_FAN)
                provoke_ = true, topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            else if (mode == gl::LINES || mode == gl::LINE_STRIP || mode == gl::LINE_LOOP)
                provoke_ = true, topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        }
    }
    drawFb_ = resolveFramebuffer(drawFbo, true);
    if (drawFb_.status != gl::FRAMEBUFFER_COMPLETE) {
        setError(gl::INVALID_FRAMEBUFFER_OPERATION);
        return false;
    }
    if (drawFb_.viewCount && drawFb_.viewCount != std::max(1u, p->refl.numViews) && p->refl.numViews > 1) {
        setError(gl::INVALID_OPERATION);  // OVR_multiview: num_views must match the framebuffer
        return false;
    }
    // Vertex input: one binding per enabled attribute that the program reads.
    VertexArray* va = currentVao();
    std::vector<VkVertexInputBindingDescription> binds;
    std::vector<VkVertexInputAttributeDescription> attrs;
    std::vector<VkBuffer> bufs;
    std::vector<VkDeviceSize> offs;
    for (const auto& av : p->refl.attributes) {
        const int cols = av.glType >= 0x8B5A && av.glType <= 0x8B5C ? int(av.glType - 0x8B5A + 2)
                         : (av.glType >= 0x8B65 && av.glType <= 0x8B6A) ? int((av.glType - 0x8B65) / 2 + 2) : 1;
        for (int c = 0; c < cols * av.arraySize; ++c) {
            const uint32_t loc = uint32_t(av.location + c);
            if (loc >= kHwVertexAttribs) continue;
            const VertexAttrib& a = va->attribs[loc];
            const uint32_t binding = uint32_t(binds.size());
            VkVertexInputAttributeDescription ad{loc, binding, VK_FORMAT_UNDEFINED, 0};
            VkVertexInputBindingDescription bd{binding, 0, VK_VERTEX_INPUT_RATE_VERTEX};
            if (!a.enabled) {
                // Current value: a constant in the stream, stride 0 (a per-instance step of 0 rows).
                Alloc s = stream(16, 16);
                std::memcpy(s.host, currentAttrib[loc].data(), 16);
                const uint8_t k = currentAttribKind[loc];
                ad.format = k == 1 ? VK_FORMAT_R32G32B32A32_SINT : k == 2 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT;
                bd.stride = 0;
                bd.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
                bufs.push_back(s.buffer), offs.push_back(s.offset);
            } else {
                VkFormat f = attribFormat(a);
                const uint32_t esize = vertexAttribBytes(a);
                const VertexBinding& vbind = va->bindings[a.binding % kMaxVertexBindings];
                const uint32_t stride = uint32_t(vbind.stride);
                const GLuint divisor = vbind.divisor;
                const uintptr_t where = vbind.offset + a.relativeOffset;  // buffer offset, or client address
                const bool perInstance = divisor > 0;
                // Rows the draw reads.
                uint32_t rows = perInstance ? (instances + divisor - 1) / divisor : firstVertex + vertexCount;
                if (indexed && !perInstance) rows = indexMax_ + 1;
                const uint32_t firstRow = perInstance ? 0 : (indexed ? 0 : firstVertex);
                Buffer* b = vbind.buffer ? buffer(vbind.buffer) : nullptr;
                const uint8_t* srcBase = nullptr;
                VkBuffer vb = VK_NULL_HANDLE;
                VkDeviceSize vo = 0;
                if (b && b->storage) {
                    srcBase = b->storage->host + where;
                    vb = b->storage->buffer, vo = where;
                    useStorage(b->storage);
                } else if (vbind.buffer == 0 && vbind.offset) {  // client array (relativeOffset alone is not a pointer)
                    srcBase = reinterpret_cast<const uint8_t*>(where);
                }
                if (!srcBase) {
                    setError(gl::INVALID_OPERATION);
                    return false;
                }
                const bool convert = needsCpuConversion(a, f) || (perInstance && divisor > 1);
                // A buffer range that does not start on a 4-byte boundary (glVertexAttribPointer accepts any byte offset) is not bound in place.
                if (convert || !vb || (where & 3)) {
                    // Client arrays, unaligned buffer offsets and formats needing conversion: copied into the stream, tightly.
                    const uint32_t outSize = convert ? 16u : esize;
                    Alloc s = stream(VkDeviceSize(rows) * outSize + 16, 16);
                    for (uint32_t r = firstRow; r < rows; ++r) {
                        const uint32_t srcRow = r;
                        const uint8_t* src = srcBase + size_t(srcRow) * stride;
                        uint8_t* dst = s.host + size_t(r) * outSize;
                        if (!convert) {
                            std::memcpy(dst, src, esize);
                            continue;
                        }
                        float v[4] = {0, 0, 0, 1};
                        for (int c2 = 0; c2 < a.size; ++c2) v[c2] = readComponent(a, src, c2);
                        std::memcpy(dst, v, 16);
                    }
                    if (convert) f = VK_FORMAT_R32G32B32A32_SFLOAT;
                    bd.stride = outSize;
                    vb = s.buffer, vo = s.offset;
                    if (perInstance && divisor > 1) {
                        // Expand divisor > 1 into one row per instance.
                        Alloc e = stream(VkDeviceSize(instances) * outSize + 16, 16);
                        for (uint32_t i = 0; i < instances; ++i)
                            std::memcpy(e.host + size_t(i) * outSize, s.host + size_t(i / divisor) * outSize, outSize);
                        vb = e.buffer, vo = e.offset;
                    }
                } else {
                    bd.stride = stride;
                }
                ad.format = f;
                bd.inputRate = perInstance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
                bufs.push_back(vb), offs.push_back(vo);
            }
            binds.push_back(bd);
            attrs.push_back(ad);
        }
    }
    if (!beginPass(drawFb_)) return false;
    VkPipeline pl = pipelineFor(*p, drawFb_, topo, binds, attrs);
    if (!pl) {
        setError(gl::INVALID_OPERATION);
        logf("gles.err what=pipeline program=%u", p->name);
        return false;
    }
    VkCommandBuffer c = cmd();
    dev->vk.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    if (!bufs.empty()) dev->vk.vkCmdBindVertexBuffers(c, 0, uint32_t(bufs.size()), bufs.data(), offs.data());
    if (!bindResources(*p)) return false;
    setDynamicState(drawFb_);
    if (!rasterizerDiscard) ++drawCount;  // occlusion queries (see Query)
    return true;
}

// The vertex list of `mode` as triangles/lines with each primitive's last vertex first (see prepareDraw).
static void provokeList(GLenum mode, const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
    const size_t n = in.size();
    switch (mode) {
        case gl::TRIANGLES:
            for (size_t i = 0; i + 2 < n; i += 3) out.insert(out.end(), {in[i + 2], in[i], in[i + 1]});
            break;
        case gl::TRIANGLE_STRIP:
            for (size_t i = 0; i + 2 < n; ++i) {
                uint32_t a = in[i], b = in[i + 1];
                if (i & 1) std::swap(a, b);  // odd triangles keep the winding
                out.insert(out.end(), {in[i + 2], a, b});
            }
            break;
        case gl::TRIANGLE_FAN:
            for (size_t i = 1; i + 1 < n; ++i) out.insert(out.end(), {in[i + 1], in[0], in[i]});
            break;
        case gl::LINES:
            for (size_t i = 0; i + 1 < n; i += 2) out.insert(out.end(), {in[i + 1], in[i]});
            break;
        case gl::LINE_STRIP: case gl::LINE_LOOP:
            for (size_t i = 0; i + 1 < n; ++i) out.insert(out.end(), {in[i + 1], in[i]});
            if (mode == gl::LINE_LOOP && n > 1) out.insert(out.end(), {in[0], in[n - 1]});
            break;
        default: break;
    }
}

void Context::drawArrays(GLenum mode, GLint first, GLsizei count, GLsizei instances) {
    if (first < 0 || count < 0 || instances < 0) return setError(gl::INVALID_VALUE, mode);
    if (count == 0 || instances == 0) return;
    VkPrimitiveTopology topo;
    VkBuffer unused;
    indexMax_ = 0;
    provokeOk_ = true;
    if (!prepareDraw(mode, topo, uint32_t(count), uint32_t(first), uint32_t(instances), false, unused)) return;
    if (provoke_) {
        std::vector<uint32_t> verts, list;
        verts.resize(size_t(count));
        for (GLsizei i = 0; i < count; ++i) verts[size_t(i)] = uint32_t(first + i);
        provokeList(mode, verts, list);
        if (list.empty()) return;
        Alloc s = stream(list.size() * 4, 4);
        std::memcpy(s.host, list.data(), list.size() * 4);
        dev->vk.vkCmdBindIndexBuffer(cmd(), s.buffer, s.offset, VK_INDEX_TYPE_UINT32);
        dev->vk.vkCmdDrawIndexed(cmd(), uint32_t(list.size()), uint32_t(instances), 0, 0, 0);
        return;
    }
    if (mode == gl::LINE_LOOP) {
        // A strip with the first vertex repeated at the end: an index list over the arrays.
        Alloc s = stream(size_t(count + 1) * 4, 4);
        auto* idx = reinterpret_cast<uint32_t*>(s.host);
        for (GLsizei i = 0; i < count; ++i) idx[i] = uint32_t(first + i);
        idx[count] = uint32_t(first);
        dev->vk.vkCmdBindIndexBuffer(cmd(), s.buffer, s.offset, VK_INDEX_TYPE_UINT32);
        dev->vk.vkCmdDrawIndexed(cmd(), uint32_t(count + 1), uint32_t(instances), 0, 0, 0);
        return;
    }
    dev->vk.vkCmdDraw(cmd(), uint32_t(count), uint32_t(instances), uint32_t(first), 0);
}

void Context::drawElements(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instances, GLint baseVertex) {
    if (count < 0 || instances < 0) return setError(gl::INVALID_VALUE, mode);
    if (type != gl::UNSIGNED_BYTE && type != gl::UNSIGNED_SHORT && type != gl::UNSIGNED_INT) return setError(gl::INVALID_ENUM, mode);
    if (count == 0 || instances == 0) return;
    const uint32_t isz = type == gl::UNSIGNED_BYTE ? 1 : type == gl::UNSIGNED_SHORT ? 2 : 4;
    // Index source: the element array buffer or client memory.
    VertexArray* va = currentVao();
    const uint8_t* src = nullptr;
    Buffer* eb = va->elementBuffer ? buffer(va->elementBuffer) : nullptr;
    if (eb) {
        if (!eb->storage) return setError(gl::INVALID_OPERATION, mode);
        const uintptr_t off = reinterpret_cast<uintptr_t>(indices);
        if (off + size_t(count) * isz > size_t(eb->size)) return setError(gl::INVALID_OPERATION, mode);
        src = eb->storage->host + off;
    } else {
        src = static_cast<const uint8_t*>(indices);
        if (!src) return setError(gl::INVALID_OPERATION, mode);
    }
    // Index range (client arrays need it; also bounds the streamed vertex copies).
    uint32_t maxIdx = 0;
    const uint32_t restart = isz == 1 ? 0xFF : isz == 2 ? 0xFFFF : 0xFFFFFFFFu;
    auto at = [&](GLsizei i) -> uint32_t {
        if (isz == 1) return src[i];
        if (isz == 2) {
            uint16_t v;
            std::memcpy(&v, src + 2 * i, 2);
            return v;
        }
        uint32_t v;
        std::memcpy(&v, src + 4 * i, 4);
        return v;
    };
    for (GLsizei i = 0; i < count; ++i) {
        uint32_t v = at(i);
        if (primitiveRestart && v == restart) continue;
        maxIdx = std::max(maxIdx, v);
    }
    indexMax_ = maxIdx + uint32_t(std::max(0, baseVertex));
    VkPrimitiveTopology topo;
    VkBuffer unused;
    provokeOk_ = true;
    if (!prepareDraw(mode, topo, 0, 0, uint32_t(instances), true, unused)) return;
    if (provoke_) {
        std::vector<uint32_t> verts, list;
        verts.resize(size_t(count));
        for (GLsizei i = 0; i < count; ++i) verts[size_t(i)] = at(i);
        provokeList(mode, verts, list);
        if (list.empty()) return;
        Alloc s = stream(list.size() * 4, 4);
        std::memcpy(s.host, list.data(), list.size() * 4);
        dev->vk.vkCmdBindIndexBuffer(cmd(), s.buffer, s.offset, VK_INDEX_TYPE_UINT32);
        dev->vk.vkCmdDrawIndexed(cmd(), uint32_t(list.size()), uint32_t(instances), 0, baseVertex, 0);
        return;
    }
    // Indices are copied into the stream (8-bit widened to 16, line loops closed).
    const bool widen = isz == 1;
    const uint32_t osz = widen ? 2 : isz;
    const GLsizei outCount = mode == gl::LINE_LOOP ? count + 1 : count;
    Alloc s = stream(size_t(outCount) * osz + 4, 4);
    for (GLsizei i = 0; i < count; ++i) {
        uint32_t v = at(i);
        if (widen && v == 0xFF && primitiveRestart) v = 0xFFFF;
        if (osz == 2) {
            uint16_t x = uint16_t(v);
            std::memcpy(s.host + 2 * i, &x, 2);
        } else {
            std::memcpy(s.host + 4 * i, &v, 4);
        }
    }
    if (mode == gl::LINE_LOOP) std::memcpy(s.host + size_t(count) * osz, s.host, osz);
    dev->vk.vkCmdBindIndexBuffer(cmd(), s.buffer, s.offset, osz == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    dev->vk.vkCmdDrawIndexed(cmd(), uint32_t(outCount), uint32_t(instances), 0, baseVertex, 0);
}

// Compute: common start of glDispatchCompute / glDispatchComputeIndirect. Ends the open pass, makes
// earlier writes visible, binds the pipeline and the program's resources.
bool Context::beginDispatch(Program*& pp) {
    Program* p = computeProgram();
    if (!p || !p->linked || !p->csModule) {
        setError(gl::INVALID_OPERATION);
        return false;
    }
    endPass();
    barrier();
    VkPipeline pl = VK_NULL_HANDLE;
    auto it = p->pipelines.find(0);
    if (it != p->pipelines.end()) pl = it->second;
    else {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, p->csModule, "main", nullptr};
        ci.layout = p->pipeLayout;
        dev->vk.vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &ci, nullptr, &pl);
        p->pipelines[0] = pl;
        recipeStatsTick(false);
    }
    if (!pl) {
        setError(gl::INVALID_OPERATION);
        return false;
    }
    dev->vk.vkCmdBindPipeline(cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, pl);
    dispatching_ = true;
    const bool bound = bindResources(*p);
    dispatching_ = false;
    if (!bound) {
        sliceBack_.clear();
        return false;
    }
    pp = p;
    return true;
}

void Context::dispatch(GLuint x, GLuint y, GLuint z) {
    // MAX_COMPUTE_WORK_GROUP_COUNT (see GetIntegeri_v): 65535 per dimension.
    if (x > 65535 || y > 65535 || z > 65535) return setError(gl::INVALID_VALUE);
    Program* p = nullptr;
    if (!beginDispatch(p)) return;
    dev->vk.vkCmdDispatch(cmd(), x, y, z);
    barrier();
    syncSlicesBack();
}

void Context::dispatchIndirect(GLintptr offset) {
    Buffer* ib = buffer(inertBuffer[3]);
    if (!ib || !ib->storage) return setError(gl::INVALID_OPERATION);
    if (offset < 0 || offset % 4) return setError(gl::INVALID_VALUE);
    if (size_t(offset) + 12 > size_t(ib->size)) return setError(gl::INVALID_OPERATION);
    Program* p = nullptr;
    if (!beginDispatch(p)) return;
    useStorage(ib->storage);
    dev->vk.vkCmdDispatchIndirect(cmd(), ib->storage->buffer, VkDeviceSize(offset));
    barrier();
    syncSlicesBack();
}

// ---- indirect draws -----------------------------------------------------------------------------------
// True when every attribute the program reads can be fetched by the GPU straight from its buffer (the
// draw's vertex count is then not needed on the CPU): enabled attributes backed by a buffer in a format
// Vulkan reads natively and without a divisor > 1 expansion; disabled ones are constants.
bool Context::attribsDirect(Program& p, VertexArray* va) {
    for (const auto& av : p.refl.attributes) {
        const int cols = av.glType >= 0x8B5A && av.glType <= 0x8B5C ? int(av.glType - 0x8B5A + 2)
                         : (av.glType >= 0x8B65 && av.glType <= 0x8B6A) ? int((av.glType - 0x8B65) / 2 + 2) : 1;
        for (int c = 0; c < cols * av.arraySize; ++c) {
            const uint32_t loc = uint32_t(av.location + c);
            if (loc >= kHwVertexAttribs) continue;
            const VertexAttrib& a = va->attribs[loc];
            if (!a.enabled) continue;
            const VertexBinding& vb = va->bindings[a.binding % kMaxVertexBindings];
            Buffer* b = vb.buffer ? buffer(vb.buffer) : nullptr;
            if (!b || !b->storage) return false;
            if (needsCpuConversion(a, attribFormat(a)) || vb.divisor > 1 || ((vb.offset + a.relativeOffset) & 3)) return false;
        }
    }
    return true;
}

// glDrawArraysIndirect / glDrawElementsIndirect. GPU path (vkCmdDrawIndirect*) when the vertex and index
// data can be read in place; otherwise the command is read on the CPU (after the GPU is done with the
// buffer) and the draw goes through the ordinary path that streams/converts the vertex data.
// ES 3.1 draw commands: {count, instanceCount, first|firstIndex, [baseVertex,] reservedMustBeZero}.
void Context::drawIndirect(GLenum mode, GLenum indexType, const void* indirect) {
    Buffer* ib = buffer(inertBuffer[2]);
    if (!ib || !ib->storage) return setError(gl::INVALID_OPERATION, mode);
    const uintptr_t off = reinterpret_cast<uintptr_t>(indirect);
    const size_t cmdSize = indexType ? 20 : 16;
    if (off % 4) return setError(gl::INVALID_VALUE, mode);
    if (off + cmdSize > size_t(ib->size)) return setError(gl::INVALID_OPERATION, mode);
    if (indexType && indexType != gl::UNSIGNED_BYTE && indexType != gl::UNSIGNED_SHORT && indexType != gl::UNSIGNED_INT)
        return setError(gl::INVALID_ENUM, indexType);
    if (mode > gl::TRIANGLE_FAN) return setError(gl::INVALID_ENUM, mode);
    Program* p = graphicsProgram();
    if (!p || !p->linked || !p->vsModule) return setError(gl::INVALID_OPERATION);
    VertexArray* va = currentVao();
    Buffer* eb = indexType && va->elementBuffer ? buffer(va->elementBuffer) : nullptr;
    if (indexType && (!eb || !eb->storage)) return setError(gl::INVALID_OPERATION, mode);
    const bool direct = mode != gl::LINE_LOOP && mode != gl::TRIANGLE_FAN && (!indexType || indexType != gl::UNSIGNED_BYTE) &&
                        attribsDirect(*p, va);
    if (direct) {
        VkPrimitiveTopology topo;
        VkBuffer unused;
        indexMax_ = 0;
        if (!prepareDraw(mode, topo, 0, 0, 1, indexType != 0, unused)) return;
        useStorage(ib->storage);
        if (indexType) {
            useStorage(eb->storage);
            dev->vk.vkCmdBindIndexBuffer(cmd(), eb->storage->buffer, 0,
                                         indexType == gl::UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
            dev->vk.vkCmdDrawIndexedIndirect(cmd(), ib->storage->buffer, VkDeviceSize(off), 1, 20);
        } else {
            dev->vk.vkCmdDrawIndirect(cmd(), ib->storage->buffer, VkDeviceSize(off), 1, 16);
        }
        return;
    }
    if (ib->storage->lastUse > dev->completed) flush(true);  // the GPU may have written the command
    uint32_t w[5] = {};
    std::memcpy(w, ib->storage->host + off, cmdSize);
    if (!indexType) return drawArrays(mode, GLint(w[2]), GLsizei(w[0]), GLsizei(w[1]));
    const uint32_t isz = indexType == gl::UNSIGNED_BYTE ? 1 : indexType == gl::UNSIGNED_SHORT ? 2 : 4;
    drawElements(mode, GLsizei(w[0]), indexType, reinterpret_cast<const void*>(uintptr_t(w[2]) * isz), GLsizei(w[1]), GLint(int32_t(w[3])));
}

// ---- clears -----------------------------------------------------------------------------------------
// The program of a masked clear: a vertex shader that makes one triangle covering the target, and a fragment shader that
// writes one uniform colour per draw buffer slot (`kinds`: 'f' / 'i' / 'u' = float / signed / unsigned integer output).
// Null (and not retried) when it does not build.
Program* Context::clearProgram(const std::string& kinds) {
    if (auto it = clearProgs_.find(kinds); it != clearProgs_.end()) return it->second.get();
    std::string fs = "#version 300 es\nprecision highp float;\nprecision highp int;\n";
    for (size_t i = 0; i < kinds.size(); ++i) {
        const char* t = kinds[i] == 'i' ? "ivec4" : kinds[i] == 'u' ? "uvec4" : "vec4";
        const std::string n = std::to_string(i);
        fs += std::string("uniform ") + t + " c" + n + ";\nlayout(location = " + n + ") out " + t + " o" + n + ";\n";
    }
    fs += "void main() {\n";
    for (size_t i = 0; i < kinds.size(); ++i) fs += "  o" + std::to_string(i) + " = c" + std::to_string(i) + ";\n";
    fs += "}\n";
    const std::string vs =
        "#version 300 es\nvoid main() {\n  gl_Position = gl_VertexID == 0 ? vec4(-1.0, -1.0, 0.0, 1.0) : "
        "gl_VertexID == 1 ? vec4(3.0, -1.0, 0.0, 1.0) : vec4(-1.0, 3.0, 0.0, 1.0);\n}\n";
    Shader sv, sf;
    sv.type = gl::VERTEX_SHADER, sf.type = gl::FRAGMENT_SHADER;
    auto build = [](Shader& s, vpglsl::Stage st, const std::string& text) {
        vpglsl::ParseResult r = vpglsl::parse(st, text);
        s.parsed = r.shader, s.compiled = r.shader != nullptr, s.log = r.log;
        return s.compiled;
    };
    auto p = std::make_unique<Program>();
    if (!build(sv, vpglsl::Stage::Vertex, vs) || !build(sf, vpglsl::Stage::Fragment, fs) || !linkShaders(*p, &sv, &sf, nullptr)) {
        logf("gles.err what=clear_program kinds=%s log=%s%s%s", kinds.c_str(), sv.log.c_str(), sf.log.c_str(), p->log.c_str());
        p.reset();
    }
    return (clearProgs_[kinds] = std::move(p)).get();
}

// Draws the clear described by `cd` over `rect` of the draw framebuffer `fb`, with the GL write masks: the colour write
// mask of each draw buffer (as the pipeline's), and for the stencil a REPLACE of the clear value under the stencil write
// mask. Everything else of the GL state is put aside for the draw and restored. False when the program does not build.
bool Context::clearByDraw(const FbInfo& fb, const VkRect2D& rect, const ClearDraw& cd) {
    const PassKey key = passKeyFor(fb);
    std::string kinds;
    for (int i = 0; i < kMaxDrawBuffers; ++i) {
        if (!key.color[size_t(i)]) continue;
        const FormatInfo* fi = fb.color[size_t(i)].image->format;
        kinds.resize(size_t(i) + 1, 'f');
        kinds[size_t(i)] = !fi->integer ? 'f' : fi->componentType == gl::UNSIGNED_INT ? 'u' : 'i';
    }
    Program* p = clearProgram(kinds);
    if (!p) return false;
    for (int i = 0; i < kMaxDrawBuffers; ++i)
        if ((cd.colorBits >> i & 1) && size_t(i) < kinds.size())
            setUniform(uniformLocation(*p, ("c" + std::to_string(i)).c_str()), 4,
                       kinds[size_t(i)] == 'i' ? Kind::Int : kinds[size_t(i)] == 'u' ? Kind::Uint : Kind::Float, 1,
                       cd.value[size_t(i)].data(), 0, false, p);
    const auto sBlend = blend;
    const StencilFace sStencil[2] = {stencil[0], stencil[1]};
    GLint sViewport[4], sScissor[4];
    std::copy(viewport, viewport + 4, sViewport);
    std::copy(scissor, scissor + 4, sScissor);
    const bool sDiscard = rasterizerDiscard, sCull = cullFace, sOffset = polygonOffsetFill, sRestart = primitiveRestart,
               sCoverage = sampleAlphaToCoverage, sDepth = depthTest, sStencilTest = stencilTest, sScissorTest = scissorTest;
    const GLenum sError = error;
    const uint64_t sDraws = drawCount;
    for (size_t i = 0; i < blend.size(); ++i) {
        blend[i].enabled = false;
        if (!(cd.colorBits >> i & 1)) blend[i].mask[0] = blend[i].mask[1] = blend[i].mask[2] = blend[i].mask[3] = false;
    }
    rasterizerDiscard = cullFace = polygonOffsetFill = primitiveRestart = sampleAlphaToCoverage = depthTest = false;
    stencilTest = cd.stencil;
    if (cd.stencil)
        for (int f = 0; f < 2; ++f)
            stencil[f] = {gl::ALWAYS, GLint(cd.stencilValue), ~0u, sStencil[0].writeMask, gl::REPLACE, gl::REPLACE, gl::REPLACE};
    viewport[0] = viewport[1] = 0, viewport[2] = GLint(fb.width), viewport[3] = GLint(fb.height);
    scissorTest = true;
    scissor[0] = rect.offset.x, scissor[1] = rect.offset.y, scissor[2] = GLint(rect.extent.width), scissor[3] = GLint(rect.extent.height);
    forcedProgram_ = p;
    occPause();  // the clear is not an app draw
    drawArrays(gl::TRIANGLES, 0, 3, 1);
    occResume();
    forcedProgram_ = nullptr;
    if (error != sError) logf("gles.err what=clear_draw error=0x%x", error);
    error = sError;  // an internal draw never raises a GL error of the app's
    drawCount = sDraws;
    blend = sBlend;
    stencil[0] = sStencil[0], stencil[1] = sStencil[1];
    std::copy(sViewport, sViewport + 4, viewport);
    std::copy(sScissor, sScissor + 4, scissor);
    rasterizerDiscard = sDiscard, cullFace = sCull, polygonOffsetFill = sOffset, primitiveRestart = sRestart;
    sampleAlphaToCoverage = sCoverage, depthTest = sDepth, stencilTest = sStencilTest, scissorTest = sScissorTest;
    return true;
}

void Context::clear(GLbitfield mask) {
    if (mask & ~(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT)) return setError(gl::INVALID_VALUE);
    FbInfo fb = resolveFramebuffer(drawFbo, true);
    if (fb.status != gl::FRAMEBUFFER_COMPLETE) return setError(gl::INVALID_FRAMEBUFFER_OPERATION);
    if (rasterizerDiscard) return;
    const PassKey key = passKeyFor(fb);  // what beginPass will attach
    VkClearRect rect{};
    rect.rect = {{0, 0}, {fb.width, fb.height}};
    if (scissorTest) {
        int32_t x0 = std::max(0, scissor[0]), y0 = std::max(0, scissor[1]);
        int32_t x1 = std::min<int32_t>(int32_t(fb.width), scissor[0] + scissor[2]);
        int32_t y1 = std::min<int32_t>(int32_t(fb.height), scissor[1] + scissor[3]);
        if (x1 <= x0 || y1 <= y0) {
            beginPass(fb);
            return;
        }
        rect.rect = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
    }
    rect.layerCount = 1;
    std::vector<VkClearAttachment> atts;
    ClearDraw cd;                                // what a write mask makes a draw
    std::vector<VkClearAttachment> viaDraw;      // the same as plain clears, if the draw cannot be made
    if (mask & gl::COLOR_BUFFER_BIT)
        for (int i = 0; i < kMaxDrawBuffers; ++i) {
            if (!key.color[size_t(i)]) continue;
            const BlendState& b = blend[size_t(i)];
            if (!b.mask[0] && !b.mask[1] && !b.mask[2] && !b.mask[3]) continue;
            VkClearAttachment a{};
            a.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, a.colorAttachment = uint32_t(i);
            const FormatInfo* fi = fb.color[size_t(i)].image->format;
            if (fi->integer) {
                for (int c = 0; c < 4; ++c) a.clearValue.color.int32[c] = int32_t(clearColor[c]);
            } else {
                for (int c = 0; c < 4; ++c) a.clearValue.color.float32[c] = clearColor[c];
                if (fi->swz[3] == VK_COMPONENT_SWIZZLE_ONE) a.clearValue.color.float32[3] = 1.0f;
            }
            if (!(b.mask[0] && b.mask[1] && b.mask[2] && b.mask[3])) {
                cd.colorBits |= 1u << i;
                std::memcpy(cd.value[size_t(i)].data(), &a.clearValue.color, 16);
                viaDraw.push_back(a);
                continue;
            }
            atts.push_back(a);
        }
    VkClearAttachment ds{};
    if ((mask & gl::DEPTH_BUFFER_BIT) && key.depth && depthMask && fb.depth.image->format->depth)
        ds.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if ((mask & gl::STENCIL_BUFFER_BIT) && key.stencil && fb.stencil.image->format->stencil) {
        const uint32_t bits = fb.stencil.image->format->stencilBits ? fb.stencil.image->format->stencilBits : 8;
        const uint32_t all = (1u << bits) - 1, wm = stencil[0].writeMask & all;
        if (wm == all) {
            ds.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        } else if (wm) {
            cd.stencil = true, cd.stencilValue = uint32_t(clearStencil) & all;
            VkClearAttachment s{};
            s.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT, s.clearValue.depthStencil = {0.0f, cd.stencilValue};
            viaDraw.push_back(s);
        }
    }
    if (ds.aspectMask) {
        ds.clearValue.depthStencil = {std::clamp(clearDepth, 0.0f, 1.0f), uint32_t(clearStencil)};
        atts.push_back(ds);
    }
    // A clear of the whole target that begins the pass is its CLEAR load op (beginPass empties `atts`).
    const bool full = rect.rect.offset.x == 0 && rect.rect.offset.y == 0 && rect.rect.extent.width == fb.width &&
                      rect.rect.extent.height == fb.height;
    if (!beginPass(fb, full ? &atts : nullptr)) return;
    if (!atts.empty()) dev->vk.vkCmdClearAttachments(cmd(), uint32_t(atts.size()), atts.data(), 1, &rect);
    if ((cd.colorBits || cd.stencil) && !clearByDraw(fb, rect.rect, cd))
        dev->vk.vkCmdClearAttachments(cmd(), uint32_t(viaDraw.size()), viaDraw.data(), 1, &rect);  // mask ignored (logged)
}

void Context::clearBuffer(GLenum buf, GLint drawbuffer, const void* value, GLenum kind, float depth, GLint sval) {
    FbInfo fb = resolveFramebuffer(drawFbo, true);
    if (fb.status != gl::FRAMEBUFFER_COMPLETE) return setError(gl::INVALID_FRAMEBUFFER_OPERATION, buf);
    const PassKey key = passKeyFor(fb);  // what beginPass will attach
    VkClearRect rect{{{0, 0}, {fb.width, fb.height}}, 0, 1};
    if (scissorTest) {
        int32_t x0 = std::max(0, scissor[0]), y0 = std::max(0, scissor[1]);
        int32_t x1 = std::min<int32_t>(int32_t(fb.width), scissor[0] + scissor[2]);
        int32_t y1 = std::min<int32_t>(int32_t(fb.height), scissor[1] + scissor[3]);
        if (x1 <= x0 || y1 <= y0) {
            beginPass(fb);
            return;
        }
        rect.rect = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
    }
    VkClearAttachment a{}, viaDraw{};
    ClearDraw cd;  // what a write mask makes a draw (viaDraw: the plain clear, if the draw cannot be made)
    if (buf == gl::COLOR) {
        if (drawbuffer < 0 || drawbuffer >= kMaxDrawBuffers) return setError(gl::INVALID_VALUE, buf);
        if (!key.color[size_t(drawbuffer)]) return;
        const BlendState& b = blend[size_t(drawbuffer)];
        if (!b.mask[0] && !b.mask[1] && !b.mask[2] && !b.mask[3]) return;  // every channel masked
        a.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, a.colorAttachment = uint32_t(drawbuffer);
        std::memcpy(&a.clearValue.color, value, 16);
        (void)kind;
        if (!(b.mask[0] && b.mask[1] && b.mask[2] && b.mask[3])) {
            cd.colorBits = 1u << drawbuffer;
            std::memcpy(cd.value[size_t(drawbuffer)].data(), value, 16);
            viaDraw = a;
            a.aspectMask = 0;
        }
    } else {
        if (buf == gl::DEPTH || buf == gl::DEPTH_STENCIL)
            if (key.depth && depthMask) a.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if ((buf == gl::STENCIL || buf == gl::DEPTH_STENCIL) && key.stencil) {
            const uint32_t bits = fb.stencil.image->format->stencilBits ? fb.stencil.image->format->stencilBits : 8;
            const uint32_t all = (1u << bits) - 1, wm = stencil[0].writeMask & all;
            if (wm == all) {
                a.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            } else if (wm) {
                cd.stencil = true, cd.stencilValue = uint32_t(sval) & all;
                viaDraw.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT, viaDraw.clearValue.depthStencil = {0.0f, cd.stencilValue};
            }
        }
        a.clearValue.depthStencil = {std::clamp(depth, 0.0f, 1.0f), uint32_t(sval)};
        if (!a.aspectMask && !cd.stencil) return;
    }
    std::vector<VkClearAttachment> one;
    if (a.aspectMask) one.push_back(a);
    const bool full = rect.rect.offset.x == 0 && rect.rect.offset.y == 0 && rect.rect.extent.width == fb.width &&
                      rect.rect.extent.height == fb.height;
    if (!beginPass(fb, full && !one.empty() ? &one : nullptr)) return;
    if (!one.empty()) dev->vk.vkCmdClearAttachments(cmd(), 1, &a, 1, &rect);
    if ((cd.colorBits || cd.stencil) && !clearByDraw(fb, rect.rect, cd))
        dev->vk.vkCmdClearAttachments(cmd(), 1, &viaDraw, 1, &rect);  // mask ignored (logged)
}

// ---- readback / blit --------------------------------------------------------------------------------
void Context::readPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void* pixels) {
    if (w < 0 || h < 0) return setError(gl::INVALID_VALUE, format);
    FbInfo fb = resolveFramebuffer(readFbo, false);
    if (fb.status != gl::FRAMEBUFFER_COMPLETE) return setError(gl::INVALID_FRAMEBUFFER_OPERATION, format);
    int idx = 0;
    if (readFbo) {
        const Framebuffer* f = fbo(readFbo);
        if (f->readBuffer == gl::NONE) return setError(gl::INVALID_OPERATION, format);
        idx = int(f->readBuffer - gl::COLOR_ATTACHMENT0);
    }
    const FbTarget src = fb.color[size_t(idx)];
    if (!src.image) return setError(gl::INVALID_OPERATION, format);
    const uint32_t outPx = clientPixelBytes(format, type);
    if (!outPx) return setError(gl::INVALID_ENUM, format);
    if (w == 0 || h == 0) return;
    // Clip to the framebuffer; outside pixels are left untouched.
    const int32_t x0 = std::max(0, x), y0 = std::max(0, y);
    const int32_t x1 = std::min<int32_t>(x + w, int32_t(fb.width)), y1 = std::min<int32_t>(y + h, int32_t(fb.height));
    if (x1 <= x0 || y1 <= y0) return;
    const uint32_t cw = uint32_t(x1 - x0), ch = uint32_t(y1 - y0);
    const uint32_t tb = src.image->format->texelBytes;
    Alloc s = stream(VkDeviceSize(cw) * ch * tb, 16);
    barrier();
    VkBufferImageCopy r{};
    r.bufferOffset = s.offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, src.level, src.layer, 1};
    r.imageOffset = {x0, y0, 0};
    r.imageExtent = {cw, ch, 1};
    src.image->invalid.clear();
    dev->vk.vkCmdCopyImageToBuffer(cmd(), src.image->image, VK_IMAGE_LAYOUT_GENERAL, s.buffer, 1, &r);
    const uint8_t* host = s.host;
    StorageP keepAlive = chunks_.back().storage;  // the chunk survives the flush below
    flush(true);
    // Into the client (or pixel pack buffer) with the pack state.
    uint8_t* dst = static_cast<uint8_t*>(pixels);
    if (pixelPack) {
        Buffer* pb = buffer(pixelPack);
        if (!pb || !pb->storage) return setError(gl::INVALID_OPERATION, format);
        prepareWrite(*pb, false);
        dst = pb->storage->host + reinterpret_cast<uintptr_t>(pixels);
    }
    if (!dst) return;
    const uint32_t rowLen = packRowLength ? uint32_t(packRowLength) : uint32_t(w);
    const size_t rowBytes = (size_t(rowLen) * outPx + packAlignment - 1) / packAlignment * packAlignment;
    std::vector<uint8_t> row(size_t(cw) * outPx);
    for (uint32_t j = 0; j < ch; ++j) {
        if (!convertReadback(*src.image->format, format, type, host + size_t(j) * cw * tb, cw, row.data()))
            return setError(gl::INVALID_OPERATION, format);
        uint8_t* out = dst + (size_t(packSkipRows) + size_t(y0 - y) + j) * rowBytes + (size_t(packSkipPixels) + size_t(x0 - x)) * outPx;
        std::memcpy(out, row.data(), row.size());
    }
}

void Context::blitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1,
                              GLbitfield mask, GLenum filter) {
    FbInfo rf = resolveFramebuffer(readFbo, false), df = resolveFramebuffer(drawFbo, true);
    if (rf.status != gl::FRAMEBUFFER_COMPLETE || df.status != gl::FRAMEBUFFER_COMPLETE)
        return setError(gl::INVALID_FRAMEBUFFER_OPERATION);
    if ((mask & (gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT)) && filter != gl::NEAREST) return setError(gl::INVALID_OPERATION);
    barrier();
    auto blit = [&](const FbTarget& s, const FbTarget& d, VkImageAspectFlags aspect) {
        if (!s.image || !d.image) return;
        s.image->invalid.clear(), d.image->invalid.clear();
        VkImageBlit b{};
        b.srcSubresource = {aspect, s.level, s.layer, 1};
        b.dstSubresource = {aspect, d.level, d.layer, 1};
        b.srcOffsets[0] = {sx0, sy0, 0}, b.srcOffsets[1] = {sx1, sy1, 1};
        b.dstOffsets[0] = {dx0, dy0, 0}, b.dstOffsets[1] = {dx1, dy1, 1};
        dev->vk.vkCmdBlitImage(cmd(), s.image->image, VK_IMAGE_LAYOUT_GENERAL, d.image->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                               filter == gl::LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    };
    if (mask & gl::COLOR_BUFFER_BIT) {
        int ri = 0;
        if (readFbo) ri = int(fbo(readFbo)->readBuffer - gl::COLOR_ATTACHMENT0);
        const auto& bufs = drawFbo ? fbo(drawFbo)->drawBuffers : fbos[0]->drawBuffers;
        for (int i = 0; i < kMaxDrawBuffers; ++i)
            if (bufs[size_t(i)] != gl::NONE) blit(rf.color[size_t(std::clamp(ri, 0, kMaxDrawBuffers - 1))], df.color[size_t(i)], VK_IMAGE_ASPECT_COLOR_BIT);
    }
    if (mask & gl::DEPTH_BUFFER_BIT) blit(rf.depth, df.depth, VK_IMAGE_ASPECT_DEPTH_BIT);
    if (mask & gl::STENCIL_BUFFER_BIT) blit(rf.stencil, df.stencil, VK_IMAGE_ASPECT_STENCIL_BIT);
    barrier();
}

}  // namespace vpgles
