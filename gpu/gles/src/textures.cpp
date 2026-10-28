// Textures: storage, uploads (pixel unpack state, conversions, PBOs), copies, mipmaps.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>

#include "context.h"

namespace vpgles {

namespace {

uint32_t mipCount(uint32_t w, uint32_t h, uint32_t d) {
    uint32_t m = std::max({w, h, d}), n = 1;
    while (m > 1) m >>= 1, ++n;
    return n;
}

bool isCubeFace(GLenum t) { return t >= gl::TEXTURE_CUBE_MAP_POSITIVE_X && t <= gl::TEXTURE_CUBE_MAP_NEGATIVE_Z; }

GLenum bindTarget(GLenum t) { return isCubeFace(t) ? gl::TEXTURE_CUBE_MAP : t; }

}  // namespace

// ---- diagnostics: textures per internal format -------------------------------------------------------------
namespace {
struct TexStats {
    std::map<GLenum, uint64_t> count, shown;  // internal format -> textures defined / count at its last line
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    int lines = 0;
} gTexStats;
}  // namespace

void texStatsCount(GLenum internalFormat) {
    // The periodic tick (eglSwapBuffers, vpgles_flush) may never run (a VR host presents through its own swapchain and the
    // process is killed without eglTerminate): the first texture of each format is logged at once.
    if (++gTexStats.count[internalFormat] == 1 && gTexStats.lines < 256) {
        ++gTexStats.lines;
        gTexStats.shown[internalFormat] = 1;
        logf("gles.tex fmt=0x%04x count=1", internalFormat);
    }
}

void texStatsTick(bool force) {
    TexStats& st = gTexStats;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - st.last < std::chrono::seconds(10)) return;
    st.last = now;
    for (const auto& [fmt, n] : st.count) {
        if (st.shown[fmt] == n) continue;
        if (st.lines >= 256) return;
        ++st.lines;
        st.shown[fmt] = n;
        logf("gles.tex fmt=0x%04x count=%llu", fmt, static_cast<unsigned long long>(n));
    }
}


// Formats of ES 3.1 images (glBindImageTexture / image2D...): the ones the spec requires to be storage-capable.
static bool storageCapable(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: return true;
        default: return false;
    }
}

bool Context::realizeTexture(Texture& t) {
    if (t.levels.empty() || !t.levels[0].format) return false;
    const Texture::Level& l0 = t.levels[0];
    const bool cube = t.target == gl::TEXTURE_CUBE_MAP;
    const bool is3D = t.target == gl::TEXTURE_3D;
    const bool array = t.target == gl::TEXTURE_2D_ARRAY;
    const uint32_t layers = cube ? 6 : array ? l0.d : 1;
    const uint32_t depth = is3D ? l0.d : 1;
    const uint32_t levels = t.immutable ? uint32_t(t.immutableLevels) : mipCount(l0.w, l0.h, depth);
    const bool storage = t.wantStorage && storageCapable(l0.format->vk);
    if (t.image && t.image->format == l0.format && t.image->width == l0.w && t.image->height == l0.h &&
        t.image->depth == depth && t.image->layers == layers && t.image->levels == levels && t.image->storage == storage)
        return true;
    auto img = std::make_shared<Image>();
    img->dev = dev;
    img->format = l0.format;
    img->width = l0.w, img->height = l0.h, img->depth = depth, img->layers = layers, img->levels = levels;
    img->cube = cube;
    img->storage = storage;
    img->type = is3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    // Slices of a 3D texture are rendered to as 2D attachments (glFramebufferTexture3D / TextureLayer).
    if (is3D && !l0.format->compressed && !l0.format->depth && !l0.format->stencil && l0.format->renderable)
        ci.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    if (l0.format->vk == VK_FORMAT_R8G8B8A8_SRGB || l0.format->vk == VK_FORMAT_R8G8B8A8_UNORM)
        ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;  // sRGB decode skip / sRGB writes
    ci.imageType = img->type;
    ci.format = l0.format->vk;
    ci.extent = {l0.w, l0.h, depth};
    ci.mipLevels = levels, ci.arrayLayers = layers, ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!l0.format->compressed && (!is3D || (ci.flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT)))
        ci.usage |= l0.format->depth || l0.format->stencil ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                                           : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (storage) {
        ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;  // lazy: ShaderWrite turns off lossless compression on Apple GPUs
        ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;  // glBindImageTexture may read the texels with another format of the same size
    }
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dev->vk.vkCreateImage(dev->device, &ci, nullptr, &img->image) != VK_SUCCESS) {
        setError(gl::OUT_OF_MEMORY);
        return false;
    }
    VkMemoryRequirements req;
    dev->vk.vkGetImageMemoryRequirements(dev->device, img->image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = uint32_t(dev->memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    dev->vk.vkAllocateMemory(dev->device, &mai, nullptr, &img->memory);
    dev->vk.vkBindImageMemory(dev->device, img->image, img->memory, 0);
    imageInit(*img);
    // Keep the old image alive until recorded work that samples it has run.
    if (t.image) keep(t.image);
    t.image = img;
    return true;
}

bool Context::makeStorage(Texture& t) {
    if (!t.image || t.image->storage) return t.image && t.image->storage;
    if (t.image->external || t.target == gl::TEXTURE_EXTERNAL_OES || !storageCapable(t.image->format->vk)) return false;
    ImageP old = t.image;
    t.wantStorage = true;
    if (!realizeTexture(t)) {
        t.wantStorage = false;  // creation with STORAGE failed: keep the texture usable without it
        return false;
    }
    if (t.image == old) return false;
    // A level 0 definition that no longer matches the old image (nothing to carry over): start empty.
    if (t.image->width != old->width || t.image->height != old->height || t.image->depth != old->depth ||
        t.image->layers != old->layers || t.image->levels != old->levels)
        return true;
    // Views, framebuffer attachments and descriptors are resolved from t.image at each draw, so the new image
    // is picked up there; the old one stays alive (realizeTexture keeps it) until the recorded work completes.
    VkImageCopy r{};
    std::vector<VkImageCopy> regions;
    for (uint32_t l = 0; l < old->levels; ++l) {
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, old->layers};
        r.extent = {std::max(1u, old->width >> l), std::max(1u, old->height >> l), std::max(1u, old->depth >> l)};
        regions.push_back(r);
    }
    barrier();
    dev->vk.vkCmdCopyImage(cmd(), old->image, VK_IMAGE_LAYOUT_GENERAL, t.image->image, VK_IMAGE_LAYOUT_GENERAL,
                           uint32_t(regions.size()), regions.data());
    barrier();
    return true;
}

void Context::copyBufferToImage(Image& img, uint32_t level, uint32_t layer, int32_t x, int32_t y, int32_t z,
                                uint32_t w, uint32_t h, uint32_t d, const uint8_t* data, size_t bytes) {
    Alloc a = stream(bytes, 16);
    std::memcpy(a.host, data, bytes);
    barrier();
    VkBufferImageCopy r{};
    r.bufferOffset = a.offset;
    r.imageSubresource = {img.aspect() & ~VkImageAspectFlags(img.format->stencil && img.format->depth ? VK_IMAGE_ASPECT_STENCIL_BIT : 0),
                          level, img.type == VK_IMAGE_TYPE_3D ? 0 : layer, img.type == VK_IMAGE_TYPE_3D ? 1 : d};
    r.imageOffset = {x, y, img.type == VK_IMAGE_TYPE_3D ? z : 0};
    r.imageExtent = {w, h, img.type == VK_IMAGE_TYPE_3D ? d : 1};
    if (img.type != VK_IMAGE_TYPE_3D) r.imageSubresource.baseArrayLayer = layer + uint32_t(z);
    img.invalid.clear();
    dev->vk.vkCmdCopyBufferToImage(cmd(), a.buffer, img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier();
}

// Uploads kept for levels above 0 are dropped when TexImage specifies that (level, face) anew. Returns the bytes freed.
static size_t dropPending(Texture& t, uint32_t level, uint32_t layer) {
    size_t freed = 0;
    for (auto it = t.pending.begin(); it != t.pending.end();) {
        if (it->level == level && it->layer == layer) {
            freed += it->data.size();
            it = t.pending.erase(it);
        } else {
            ++it;
        }
    }
    return freed;
}

void Context::queuePending(Texture& t, uint32_t level, uint32_t layer, int32_t x, int32_t y, int32_t z, uint32_t w, uint32_t h,
                           uint32_t d, const uint8_t* data, size_t bytes) {
    constexpr size_t kPendingCap = 64u << 20;  // per texture; a base that never comes must not hold memory without bound
    static std::atomic<int> said{0};
    const int n = said.fetch_add(1);
    if (t.pendingBytes + bytes > kPendingCap) {
        if (n < 16) logf("gles.note what=mip_before_base_dropped tex=%u level=%u bytes=%zu (over the %zu MiB cap)", t.name, level, bytes, kPendingCap >> 20);
        return;
    }
    if (n < 16) logf("gles.note what=mip_before_base tex=%u level=%u bytes=%zu (kept until level 0 defines the image)", t.name, level, bytes);
    t.pending.push_back({level, layer, x, y, z, w, h, d, std::vector<uint8_t>(data, data + bytes)});
    t.pendingBytes += bytes;
}

// Level 0 just created the image: the kept uploads whose level still matches the base are uploaded now, then freed.
void Context::flushPending(Texture& t) {
    if (t.pending.empty()) return;
    if (t.image) {
        const Image& im = *t.image;
        for (const Texture::PendingUpload& u : t.pending) {
            if (u.level >= im.levels || u.level >= t.levels.size() || u.layer >= im.layers) continue;
            const Texture::Level& lv = t.levels[u.level];
            const uint32_t depthAt = t.target == gl::TEXTURE_3D ? std::max(1u, im.depth >> u.level) : t.target == gl::TEXTURE_2D_ARRAY ? im.layers : lv.d;
            if (lv.format != im.format || lv.w != std::max(1u, im.width >> u.level) || lv.h != std::max(1u, im.height >> u.level) ||
                lv.d != depthAt)
                continue;  // the level no longer fits the base: it stays undefined, as GL leaves it
            copyBufferToImage(*t.image, u.level, u.layer, u.x, u.y, u.z, u.w, u.h, u.d, u.data.data(), u.data.size());
        }
    }
    t.pending.clear(), t.pending.shrink_to_fit(), t.pendingBytes = 0;
}

void Context::texImage(GLenum target, GLint level, GLenum internalFormat, GLsizei w, GLsizei h, GLsizei d, GLenum format,
                       GLenum type, const void* pixels, bool sub, GLint xo, GLint yo, GLint zo, bool compressed,
                       GLsizei imageSize) {
    if (!isCubeFace(target) && target != gl::TEXTURE_2D && target != gl::TEXTURE_3D && target != gl::TEXTURE_2D_ARRAY &&
        target != gl::TEXTURE_EXTERNAL_OES && target != gl::TEXTURE_CUBE_MAP && target != gl::TEXTURE_CUBE_MAP_ARRAY &&
        target != gl::TEXTURE_2D_MULTISAMPLE && target != gl::TEXTURE_2D_MULTISAMPLE_ARRAY && target != gl::TEXTURE_BUFFER)
        return setError(gl::INVALID_ENUM);
    Texture* t = boundTexture(bindTarget(target));
    if (!t) return setError(gl::INVALID_OPERATION);
    if (level < 0 || w < 0 || h < 0 || d < 0 || level > 16 || (compressed && imageSize < 0)) return setError(gl::INVALID_VALUE);
    {  // MAX_TEXTURE_SIZE / MAX_CUBE_MAP_TEXTURE_SIZE 16384, MAX_3D_TEXTURE_SIZE and MAX_ARRAY_TEXTURE_LAYERS 2048 (gl_api.cpp)
        const bool vol = target == gl::TEXTURE_3D;
        const GLsizei limit = vol ? 2048 : 16384;
        if (w > limit || h > limit || ((vol || target == gl::TEXTURE_2D_ARRAY) && d > 2048) || level > (vol ? 11 : 14))
            return setError(gl::INVALID_VALUE);
    }
    const bool cube = isCubeFace(target);
    if (target == gl::TEXTURE_CUBE_MAP) return setError(gl::INVALID_ENUM);
    // Targets that bind but cannot hold image data here (cube array, buffer texture, multisample array).
    if (!cube && target != gl::TEXTURE_2D && target != gl::TEXTURE_3D && target != gl::TEXTURE_2D_ARRAY &&
        target != gl::TEXTURE_EXTERNAL_OES)
        return setError(gl::INVALID_ENUM);
    if (cube && !sub && w != h) return setError(gl::INVALID_VALUE);
    const uint32_t face = cube ? target - gl::TEXTURE_CUBE_MAP_POSITIVE_X : 0;
    if (t->levels.size() <= size_t(level)) t->levels.resize(size_t(level) + 1);
    const FormatInfo* fi;
    // A level above 0 specified before level 0: there is no image yet, so its data waits in t->pending (see queuePending).
    bool early = false;
    if (!sub) {
        if (t->immutable) return setError(gl::INVALID_OPERATION);
        fi = formatFor(internalFormat, format, type);
        if (!fi) return setError(compressed ? gl::INVALID_ENUM : gl::INVALID_VALUE);
        if (compressed != fi->compressed) return setError(gl::INVALID_ENUM);
        if (!compressed)
            if (GLenum e = transferError(*fi, format, type)) return setError(e);
        if (compressed && imageSize != GLsizei(size_t(compressedSize(*fi, uint32_t(w), uint32_t(h))) * uint32_t(std::max(d, 1))) &&
            w && h && d)
            return setError(gl::INVALID_VALUE);
        Texture::Level& lv = t->levels[size_t(level)];
        lv.w = uint32_t(w), lv.h = uint32_t(h), lv.d = uint32_t(std::max(d, 1));
        lv.format = fi;
        // An empty level (0 x 0 x 0 resets a texture, as gluStateReset does) has no storage: level 0 drops the image.
        if (w == 0 || h == 0 || d == 0) {
            if (level == 0 && t->image) {
                keep(t->image);
                t->image.reset();
            }
            return;
        }
        // Level 0 defines the image; other levels must be consistent with it to land in it.
        if (level == 0 || !t->image) {
            if (level != 0 && !t->levels[0].format) {
                early = true;  // the base is defined later
                t->pendingBytes -= std::min(t->pendingBytes, dropPending(*t, uint32_t(level), face));
            } else {
                if (!realizeTexture(*t)) return;
                flushPending(*t);
                if (level == 0) texStatsCount(fi->internal);
            }
        } else if (t->image->format != fi) {
            return;  // inconsistent level: texture incomplete until redefined
        }
        if (!pixels && !pixelUnpack) return;
        xo = yo = zo = 0;
    } else {
        // A level defined before level 0 can be sub-uploaded: it is kept like its TexImage data.
        early = !t->image && level != 0 && !t->levels[0].format;
        if ((!t->image && !early) || size_t(level) >= t->levels.size() || !t->levels[size_t(level)].format)
            return setError(gl::INVALID_OPERATION);
        fi = early ? t->levels[size_t(level)].format : t->image->format;
        const Texture::Level& lv = t->levels[size_t(level)];
        if (xo < 0 || yo < 0 || zo < 0 || uint32_t(xo + w) > lv.w || uint32_t(yo + h) > lv.h ||
            (target == gl::TEXTURE_3D || target == gl::TEXTURE_2D_ARRAY ? uint32_t(zo + d) > lv.d : false))
            return setError(gl::INVALID_VALUE);
        if (compressed) {
            const FormatInfo* given = formatFor(internalFormat);
            if (!given) return setError(gl::INVALID_ENUM);  // not a format at all
            if (fi->vk != given->vk) return setError(gl::INVALID_OPERATION);  // a format, but not the level's
            if (imageSize != GLsizei(size_t(compressedSize(*fi, uint32_t(w), uint32_t(h))) * uint32_t(std::max(d, 1))))
                return setError(gl::INVALID_VALUE);
            // The region must sit on block boundaries (its far edge may be the edge of the level).
            if (xo % int(fi->blockW) || yo % int(fi->blockH) || (w % int(fi->blockW) && uint32_t(xo + w) != lv.w) ||
                (h % int(fi->blockH) && uint32_t(yo + h) != lv.h))
                return setError(gl::INVALID_OPERATION);
        } else if (GLenum e = transferError(*fi, format, type, true)) {
            return setError(e);
        }
    }
    if (!early && (!t->image || uint32_t(level) >= t->image->levels)) return;
    if (w == 0 || h == 0 || d == 0) return;
    // Source bytes: client memory or the pixel unpack buffer.
    const uint8_t* src = static_cast<const uint8_t*>(pixels);
    if (pixelUnpack) {
        Buffer* pb = buffer(pixelUnpack);
        if (!pb || !pb->storage || pb->mapped) return setError(gl::INVALID_OPERATION);
        {  // the reads must stay inside the buffer
            const size_t at = reinterpret_cast<uintptr_t>(pixels);
            size_t last;
            if (compressed) {
                last = size_t(imageSize);
            } else {
                const uint32_t px0 = clientPixelBytes(format, type);
                // The offset must be a multiple of the size of the basic machine unit of the type.
                const uint32_t unit = (type == gl::BYTE || type == gl::UNSIGNED_BYTE) ? 1u
                                      : (type == gl::SHORT || type == gl::UNSIGNED_SHORT || type == gl::HALF_FLOAT || type == 0x8033 ||
                                         type == 0x8034 || type == 0x8363) ? 2u : 4u;
                if (at % unit) return setError(gl::INVALID_OPERATION);
                const size_t rl = unpackRowLength ? size_t(unpackRowLength) : size_t(w);
                const size_t rowB = (rl * px0 + size_t(unpackAlignment) - 1) / size_t(unpackAlignment) * size_t(unpackAlignment);
                const size_t ih = unpackImageHeight ? size_t(unpackImageHeight) : size_t(h);
                last = size_t(unpackSkipImages + std::max(d, 1) - 1) * ih * rowB + size_t(unpackSkipRows + h - 1) * rowB +
                       size_t(unpackSkipPixels + w) * px0;
            }
            if (at + last > size_t(pb->size)) return setError(gl::INVALID_OPERATION);
        }
        if (pb->storage->lastUse > dev->completed) flush(true);
        src = pb->storage->host + reinterpret_cast<uintptr_t>(pixels);
    }
    if (!src) return;
    const uint32_t depthN = uint32_t(std::max(d, 1));
    const uint32_t layer = face;
    if (compressed) {
        const size_t need = size_t(compressedSize(*fi, uint32_t(w), uint32_t(h))) * depthN;
        if (size_t(imageSize) != need) return setError(gl::INVALID_VALUE);
        if (early) queuePending(*t, uint32_t(level), layer, xo, yo, zo, uint32_t(w), uint32_t(h), depthN, src, need);
        else copyBufferToImage(*t->image, uint32_t(level), layer, xo, yo, zo, uint32_t(w), uint32_t(h), depthN, src, need);
        return;
    }
    const uint32_t px = clientPixelBytes(format, type);
    if (!px) return setError(gl::INVALID_ENUM);
    // De-pad rows (unpack state).
    const uint32_t rowLen = unpackRowLength ? uint32_t(unpackRowLength) : uint32_t(w);
    const uint32_t align = uint32_t(unpackAlignment);
    const size_t rowBytes = (size_t(rowLen) * px + align - 1) / align * align;
    const uint32_t imgH = unpackImageHeight ? uint32_t(unpackImageHeight) : uint32_t(h);
    std::vector<uint8_t> tight(size_t(w) * h * depthN * px);
    for (uint32_t z = 0; z < depthN; ++z)
        for (uint32_t y = 0; y < uint32_t(h); ++y) {
            const uint8_t* row = src + (size_t(z + unpackSkipImages) * imgH + y + unpackSkipRows) * rowBytes +
                                 size_t(unpackSkipPixels) * px;
            std::memcpy(&tight[(size_t(z) * h + y) * w * px], row, size_t(w) * px);
        }
    std::vector<uint8_t> converted;
    if (!convertUpload(*fi, format, type, tight.data(), uint32_t(w) * h * depthN, converted)) {
        logf("gles.err what=upload_conversion format=0x%x type=0x%x internal=0x%x", format, type, fi->internal);
        return setError(gl::INVALID_OPERATION);
    }
    if (early)
        queuePending(*t, uint32_t(level), layer, xo, yo, zo, uint32_t(w), uint32_t(h), depthN, converted.data(), converted.size());
    else
        copyBufferToImage(*t->image, uint32_t(level), layer, xo, yo, zo, uint32_t(w), uint32_t(h), depthN, converted.data(),
                          converted.size());
}

void Context::texStorage(GLenum target, GLsizei levels, GLenum internalFormat, GLsizei w, GLsizei h, GLsizei d,
                         GLsizei samples) {
    Texture* t = boundTexture(target);
    if (!t || t->name == 0) return setError(gl::INVALID_OPERATION);
    if (t->immutable) return setError(gl::INVALID_OPERATION);
    if (levels < 1 || w < 1 || h < 1 || d < 1) return setError(gl::INVALID_VALUE);
    if (w > (target == gl::TEXTURE_3D ? 2048 : 16384) || h > (target == gl::TEXTURE_3D ? 2048 : 16384) ||
        ((target == gl::TEXTURE_3D || target == gl::TEXTURE_2D_ARRAY) && d > 2048))
        return setError(gl::INVALID_VALUE);
    const FormatInfo* fi = formatFor(internalFormat);
    if (!fi || (fi->internal != internalFormat && !fi->compressed)) return setError(gl::INVALID_ENUM);
    if (uint32_t(levels) > mipCount(uint32_t(w), uint32_t(h), target == gl::TEXTURE_3D ? uint32_t(d) : 1))
        return setError(gl::INVALID_OPERATION);
    t->immutable = true;
    t->pending.clear(), t->pending.shrink_to_fit(), t->pendingBytes = 0;  // TexStorage specifies every level anew
    texStatsCount(fi->internal);
    t->immutableLevels = levels;
    t->samples = samples > 1 ? samples : 0;
    t->levels.assign(size_t(levels), {});
    for (GLsizei i = 0; i < levels; ++i) {
        auto& lv = t->levels[size_t(i)];
        lv.w = std::max(1u, uint32_t(w) >> i);
        lv.h = std::max(1u, uint32_t(h) >> i);
        lv.d = target == gl::TEXTURE_3D ? std::max(1u, uint32_t(d) >> i) : uint32_t(d);
        lv.format = fi;
    }
    if (samples > 1) logf("gles.note what=texstorage_multisample samples=%d (created 1x: VPKTAA does AA)", samples);
    realizeTexture(*t);
    t->maxLevel = std::min(t->maxLevel, levels - 1 + t->baseLevel);
}

// glCopyImageSubData: texture or renderbuffer to texture or renderbuffer, one vkCmdCopyImage (formats
// are required to have the same texel size; compressed ones the same block size and dimensions).
void Context::copyImageSubData(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint sx, GLint sy, GLint sz, GLuint dstName,
                               GLenum dstTarget, GLint dstLevel, GLint dx, GLint dy, GLint dz, GLsizei w, GLsizei h, GLsizei d) {
    struct Side {
        Image* image = nullptr;
        uint32_t levelW = 0, levelH = 0, levelD = 1;  // size of the chosen level; layers (not depth) for arrays / cubes
        bool is3D = false;
        std::shared_ptr<Image> hold;
    } side[2];
    const GLuint names[2] = {srcName, dstName};
    const GLenum targets[2] = {srcTarget, dstTarget};
    const GLint levels[2] = {srcLevel, dstLevel};
    for (int i = 0; i < 2; ++i) {
        Side& sd = side[i];
        if (targets[i] == gl::RENDERBUFFER_TARGET) {
            Renderbuffer* r = rbo(names[i]);
            if (!r || !r->image) return setError(gl::INVALID_VALUE, targets[i]);
            if (levels[i] != 0) return setError(gl::INVALID_VALUE, targets[i]);
            sd.image = r->image.get(), sd.hold = r->image;
            sd.levelW = uint32_t(r->width), sd.levelH = uint32_t(r->height);
            continue;
        }
        const bool known = targets[i] == gl::TEXTURE_2D || targets[i] == gl::TEXTURE_3D || targets[i] == gl::TEXTURE_2D_ARRAY ||
                           targets[i] == gl::TEXTURE_CUBE_MAP || targets[i] == gl::TEXTURE_2D_MULTISAMPLE ||
                           targets[i] == gl::TEXTURE_EXTERNAL_OES;
        if (!known) return setError(gl::INVALID_ENUM, targets[i]);
        Texture* t = texture(names[i]);
        if (!t || !t->image || t->target != targets[i]) return setError(gl::INVALID_VALUE, targets[i]);
        if (levels[i] < 0 || uint32_t(levels[i]) >= t->image->levels) return setError(gl::INVALID_VALUE, targets[i]);
        sd.image = t->image.get(), sd.hold = t->image;
        sd.levelW = std::max(1u, t->image->width >> levels[i]);
        sd.levelH = std::max(1u, t->image->height >> levels[i]);
        sd.is3D = t->image->type == VK_IMAGE_TYPE_3D;
        sd.levelD = sd.is3D ? std::max(1u, t->image->depth >> levels[i]) : t->image->layers;
    }
    if (w < 0 || h < 0 || d < 0) return setError(gl::INVALID_VALUE);
    if (w == 0 || h == 0 || d == 0) return;
    const FormatInfo& sf = *side[0].image->format;
    const FormatInfo& df = *side[1].image->format;
    // Same texel size, or a compressed format and an uncompressed one whose texel is as big as the block (extent counted in source texels).
    const bool mixed = sf.compressed != df.compressed;
    const bool sameClass = sf.texelBytes == df.texelBytes && sf.depth == df.depth && sf.stencil == df.stencil &&
                           (mixed || (sf.blockW == df.blockW && sf.blockH == df.blockH));
    if (!sameClass) return setError(gl::INVALID_OPERATION);
    const uint32_t sbw = sf.compressed ? sf.blockW : 1, sbh = sf.compressed ? sf.blockH : 1;
    const GLint o[2][3] = {{sx, sy, sz}, {dx, dy, dz}};
    for (int i = 0; i < 2; ++i) {
        const Side& sd = side[i];
        const FormatInfo& f = *sd.image->format;
        uint32_t rw = i == 0 ? uint32_t(w) : (uint32_t(w) + sbw - 1) / sbw * (f.compressed ? f.blockW : 1);
        uint32_t rh = i == 0 ? uint32_t(h) : (uint32_t(h) + sbh - 1) / sbh * (f.compressed ? f.blockH : 1);
        // A compressed region that ends exactly at the level edge may cover a partial block there.
        if (f.compressed && !mixed && o[i][0] >= 0 && o[i][1] >= 0) {
            if (uint32_t(o[i][0]) + uint32_t(w) == sd.levelW) rw = uint32_t(w);
            if (uint32_t(o[i][1]) + uint32_t(h) == sd.levelH) rh = uint32_t(h);
        }
        if (o[i][0] < 0 || o[i][1] < 0 || o[i][2] < 0 || uint32_t(o[i][0]) + rw > sd.levelW || uint32_t(o[i][1]) + rh > sd.levelH ||
            uint32_t(o[i][2] + d) > sd.levelD)
            return setError(gl::INVALID_VALUE);
        if (f.compressed && (o[i][0] % f.blockW || o[i][1] % f.blockH)) return setError(gl::INVALID_VALUE);
    }
    const bool anyDepthStencil = sf.depth || sf.stencil;
    VkImageCopy r{};
    const VkImageAspectFlags aspect = anyDepthStencil ? side[0].image->aspect() : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT);
    const bool three = side[0].is3D || side[1].is3D;
    const uint32_t layers = uint32_t(d);
    r.srcSubresource = {aspect, uint32_t(srcLevel), side[0].is3D ? 0u : uint32_t(sz), side[0].is3D ? 1u : layers};
    r.dstSubresource = {aspect, uint32_t(dstLevel), side[1].is3D ? 0u : uint32_t(dz), side[1].is3D ? 1u : layers};
    r.srcOffset = {sx, sy, side[0].is3D ? sz : 0};
    r.dstOffset = {dx, dy, side[1].is3D ? dz : 0};
    r.extent = {uint32_t(w), uint32_t(h), three ? uint32_t(d) : 1u};
    // Between a 3D texture and a layered one the slice count is the layer count on the layered side.
    if (three) {
        if (!side[0].is3D) r.srcSubresource.layerCount = layers;
        if (!side[1].is3D) r.dstSubresource.layerCount = layers;
    }
    barrier();
    side[0].image->invalid.clear(), side[1].image->invalid.clear();
    dev->vk.vkCmdCopyImage(cmd(), side[0].image->image, VK_IMAGE_LAYOUT_GENERAL, side[1].image->image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    keep(side[0].hold);
    keep(side[1].hold);
    barrier();
}

void Context::generateMipmap(GLenum target) {
    if (target != gl::TEXTURE_2D && target != gl::TEXTURE_3D && target != gl::TEXTURE_2D_ARRAY && target != gl::TEXTURE_CUBE_MAP &&
        target != gl::TEXTURE_CUBE_MAP_ARRAY)
        return setError(gl::INVALID_ENUM, target);
    Texture* t = boundTexture(target);
    if (!t || !t->image) return setError(gl::INVALID_OPERATION, target);
    Image& img = *t->image;
    if (img.format->compressed || img.format->integer) return setError(gl::INVALID_OPERATION, target);
    const uint32_t base = uint32_t(std::max(0, t->baseLevel));
    img.invalid.clear();
    barrier();
    for (uint32_t l = base + 1; l < img.levels; ++l) {
        VkImageBlit b{};
        b.srcSubresource = {img.aspect(), l - 1, 0, img.layers};
        b.dstSubresource = {img.aspect(), l, 0, img.layers};
        int32_t sw = int32_t(std::max(1u, img.width >> (l - 1))), sh = int32_t(std::max(1u, img.height >> (l - 1)));
        int32_t sd = int32_t(std::max(1u, img.depth >> (l - 1)));
        b.srcOffsets[1] = {sw, sh, sd};
        b.dstOffsets[1] = {std::max(1, sw >> 1), std::max(1, sh >> 1), std::max(1, sd >> 1)};
        dev->vk.vkCmdBlitImage(cmd(), img.image, VK_IMAGE_LAYOUT_GENERAL, img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                               VK_FILTER_LINEAR);
        barrier();
        if (t->levels.size() <= l) t->levels.resize(l + 1);
        t->levels[l] = {uint32_t(b.dstOffsets[1].x), uint32_t(b.dstOffsets[1].y),
                        target == gl::TEXTURE_3D ? uint32_t(b.dstOffsets[1].z) : t->levels[0].d, img.format};
    }
}

void Context::copyTexImage(GLenum target, GLint level, GLenum internalFormat, GLint x, GLint y, GLsizei w, GLsizei h,
                           bool sub, GLint xo, GLint yo, GLint zo) {
    if (!isCubeFace(target) && target != gl::TEXTURE_2D && target != gl::TEXTURE_3D && target != gl::TEXTURE_2D_ARRAY &&
        target != gl::TEXTURE_CUBE_MAP_ARRAY)
        return setError(gl::INVALID_ENUM, target);
    if (level < 0 || level > 14 || w < 0 || h < 0 || xo < 0 || yo < 0 || zo < 0) return setError(gl::INVALID_VALUE, target);
    if (sub) {
        Texture* dst = boundTexture(bindTarget(target));
        if (!dst || size_t(level) >= dst->levels.size() || !dst->levels[size_t(level)].format) return setError(gl::INVALID_OPERATION, target);
        const Texture::Level& lv = dst->levels[size_t(level)];
        if (uint32_t(xo + w) > lv.w || uint32_t(yo + h) > lv.h) return setError(gl::INVALID_VALUE, target);
        if ((target == gl::TEXTURE_3D || target == gl::TEXTURE_2D_ARRAY) && uint32_t(zo) >= lv.d) return setError(gl::INVALID_VALUE, target);
    }
    FbInfo fb = resolveFramebuffer(readFbo, false);
    if (fb.status != gl::FRAMEBUFFER_COMPLETE) return setError(gl::INVALID_FRAMEBUFFER_OPERATION, target);
    const Framebuffer* rf = fbo(readFbo);
    int idx = 0;
    if (rf && readFbo && rf->readBuffer >= gl::COLOR_ATTACHMENT0) idx = int(rf->readBuffer - gl::COLOR_ATTACHMENT0);
    FbTarget src = fb.color[size_t(idx)];
    if (!src.image) return setError(gl::INVALID_OPERATION, target);
    if (!sub) {
        GLenum fmt = internalFormat;
        const FormatInfo* f = formatFor(internalFormat, internalFormat, gl::UNSIGNED_BYTE);
        if (!f) return setError(gl::INVALID_ENUM, target);
        // No data is sent: any type that suits the format will do.
        GLenum ty = gl::UNSIGNED_BYTE;
        for (GLenum cand : {gl::UNSIGNED_BYTE, gl::FLOAT, gl::UNSIGNED_SHORT_5_6_5, gl::UNSIGNED_SHORT_4_4_4_4, gl::UNSIGNED_SHORT_5_5_5_1,
                            gl::UNSIGNED_INT_2_10_10_10_REV, gl::UNSIGNED_INT_10F_11F_11F_REV, gl::BYTE, gl::UNSIGNED_SHORT, gl::UNSIGNED_INT})
            if (!transferError(*f, f->baseFormat, cand)) {
                ty = cand;
                break;
            }
        texImage(target, level, fmt, w, h, 1, f->baseFormat, ty, nullptr, false, 0, 0, 0, false, 0);
        xo = yo = zo = 0;
    }
    Texture* t = boundTexture(bindTarget(target));
    if (!t || !t->image) return;
    const uint32_t face = isCubeFace(target) ? target - gl::TEXTURE_CUBE_MAP_POSITIVE_X : uint32_t(zo);
    // Clip the source rectangle to the framebuffer (pixels outside are undefined in GL).
    int32_t sx = std::max(0, x), sy = std::max(0, y);
    int32_t ex = std::min<int32_t>(x + w, int32_t(fb.width)), ey = std::min<int32_t>(y + h, int32_t(fb.height));
    if (ex <= sx || ey <= sy) return;
    barrier();
    VkImageBlit b{};
    b.srcSubresource = {src.image->aspect(), src.level, src.layer, 1};
    b.dstSubresource = {t->image->aspect(), uint32_t(level), face, 1};
    b.srcOffsets[0] = {sx, sy, 0};
    b.srcOffsets[1] = {ex, ey, 1};
    b.dstOffsets[0] = {xo + (sx - x), yo + (sy - y), 0};
    b.dstOffsets[1] = {xo + (ex - x), yo + (ey - y), 1};
    src.image->invalid.clear(), t->image->invalid.clear();
    dev->vk.vkCmdBlitImage(cmd(), src.image->image, VK_IMAGE_LAYOUT_GENERAL, t->image->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                           VK_FILTER_NEAREST);
    barrier();
}

}  // namespace vpgles
