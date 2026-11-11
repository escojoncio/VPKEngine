#include "commands.h"

#include <algorithm>
#include <chrono>
#include <atomic>

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <unordered_map>

#include "texdecode.h"

namespace vpvk {

vpgpu::LoadAction toLoad(VkAttachmentLoadOp op) {
  switch (op) {
    case VK_ATTACHMENT_LOAD_OP_LOAD:
    case VK_ATTACHMENT_LOAD_OP_NONE_KHR: return vpgpu::LoadAction::Load;  // NONE (VK_EXT_load_store_op_none): contents kept, not read
    case VK_ATTACHMENT_LOAD_OP_CLEAR: return vpgpu::LoadAction::Clear;
    default: return vpgpu::LoadAction::DontCare;
  }
}

vpgpu::StoreAction toStore(VkAttachmentStoreOp op, bool resolve) {
  // NONE: contents untouched by the pass are kept (Unity uses it for read-only depth).
  const bool store = op == VK_ATTACHMENT_STORE_OP_STORE || op == VK_ATTACHMENT_STORE_OP_NONE;
  if (resolve) return store ? vpgpu::StoreAction::StoreAndResolve : vpgpu::StoreAction::Resolve;
  // DontCare / None keep the attachment on tile: the core then never writes it to memory.
  return store ? vpgpu::StoreAction::Store : vpgpu::StoreAction::DontCare;
}

// The resolve of an attachment whose MSAA the driver forced to 1x, done as a copy after the
// subpass. Rendering straight into the resolve target instead only worked for a pass that both
// draws and resolves: an SRP that draws the scene into the MSAA target over several passes and
// resolves in the last one left the eye image holding only that last pass (and uninitialised
// memory elsewhere — the full-view pink seen with VPKTAA on).
vpgpu::Copy forcedResolveCopy(const ImageView* src, const ImageView* dst, uint16_t layers) {
  vpgpu::Copy c;
  c.kind = vpgpu::CopyKind::TextureToTexture;
  const uint32_t sm = src->range.baseMipLevel, dm = dst->range.baseMipLevel;
  const uint32_t w = std::min(std::max(1u, src->image->desc.width >> sm), std::max(1u, dst->image->desc.width >> dm));
  const uint32_t h = std::min(std::max(1u, src->image->desc.height >> sm), std::max(1u, dst->image->desc.height >> dm));
  c.srcTexture = {src->image->id, uint16_t(sm), uint16_t(src->range.baseArrayLayer), layers, 0, 0, 0, w, h, 1};
  c.dstTexture = {dst->image->id, uint16_t(dm), uint16_t(dst->range.baseArrayLayer), layers, 0, 0, 0, w, h, 1};
  c.label = "vk.forced_resolve";
  return c;
}

bool resolveAsCopy(const ImageView* src) {
  return src && src->image && (src->image->forcedFrom > 1 || src->image->desc.samples <= 1);
}

// Same format, or two colour formats of the same block shape and integer-ness (sRGB / UNORM variants of one format: a
// mutable-format image's two views): the copy between them is as before.
static bool copyCompatible(vpgpu::Format a, vpgpu::Format b) {
  if (a == b) return true;
  const vpgpu::FormatInfo &x = vpgpu::formatInfo(a), &y = vpgpu::formatInfo(b);
  return !x.depth() && !x.stencil() && !y.depth() && !y.stencil() && x.bytesPerBlock == y.bytesPerBlock &&
         x.blockSize == y.blockSize && x.blockHeight == y.blockHeight && x.integer() == y.integer();
}

bool resolveCopyUsable(const ImageView* src, const ImageView* dst) {
  const char* why = nullptr;
  if (!src || !src->image || !src->image->id || !dst || !dst->image || !dst->image->id) why = "no_image";
  else if (src->image == dst->image) why = "same_image";
  else if (!copyCompatible(src->image->desc.format, dst->image->desc.format)) why = "format_differs";
  else if (src->image->desc.samples != dst->image->desc.samples) why = "samples_differ";
  static std::atomic<unsigned> copies{0}, refused{0};
  if (!why) {
    const unsigned n = copies.fetch_add(1) + 1;
    if (n <= 4 || n % 5000 == 0)
      logf("vk.resolve as_copy n=%u src_samples=%u forced_from=%u format=%d %ux%u layers=%u", n,
           unsigned(src->image->desc.samples), unsigned(src->image->forcedFrom), int(src->image->info.format),
           src->image->desc.width, src->image->desc.height, unsigned(src->image->desc.layers));
    return true;
  }
  if (refused.fetch_add(1) < 8)
    logf("vk.resolve dropped why=%s src_format=%d dst_format=%d", why, src && src->image ? int(src->image->info.format) : -1,
         dst && dst->image ? int(dst->image->info.format) : -1);
  return false;
}

// Metal resolves only float / normalized colour formats. An integer MSAA attachment's resolve
// (Vulkan: one sample, here sample 0) is a nearest blit from the stored samples into the resolve
// target after the subpass (gpu/metal2 encodeBlit: texture2d_ms, vp_blit_ms_i/_u). The integer
// test uses the format the attachment renders as (a mutable-format image's view).
bool integerResolve(const ImageView* src) {
  return src && src->image && src->image->desc.samples > 1 && vpgpu::formatInfo(attachmentFormat(*src)).integer();
}
vpgpu::Copy sampleZeroResolve(const ImageView* src, const ImageView* dst, uint16_t layers) {
  vpgpu::Copy c = forcedResolveCopy(src, dst, layers);
  c.kind = vpgpu::CopyKind::TextureBlit;
  c.blitFilter = vpgpu::Filter::Nearest;
  // Both sides seen as the attachment view's format (mutable-format images: either image may be
  // the mutable one; Vulkan requires the two views to have the same format).
  const vpgpu::Format vf = attachmentFormat(*src);
  if ((vf != src->image->desc.format || vf != dst->image->desc.format) &&
      vpgpu::formatInfo(dst->image->desc.format).bytesPerBlock == vpgpu::formatInfo(vf).bytesPerBlock)
    c.blitFormat = vf;
  c.label = "vk.resolve_int";
  return c;
}

// The format a colour attachment view renders as: the view's when it is another format of the
// same texel size (mutable-format image), else the image's (Metal aborts on other views).
vpgpu::Format attachmentFormat(const ImageView& v) {
  const vpgpu::Format img = v.image->desc.format, vf = toCoreFormat(v.format);
  const vpgpu::FormatInfo &a = vpgpu::formatInfo(img), &b = vpgpu::formatInfo(vf);
  if (vf == vpgpu::Format::Undefined || vf == img || a.depth() || a.stencil() || b.depth() || b.stencil() ||
      a.blockSize != 1 || b.blockSize != 1 || a.bytesPerBlock != b.bytesPerBlock)
    return img;
  return vf;
}

std::array<double, 4> clearColor(const VkClearColorValue& v, vpgpu::Format f) {
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(f);
  const bool sint = fi.integer() && fi.name && std::strstr(fi.name, "Sint");
  std::array<double, 4> c{};
  for (int i = 0; i < 4; ++i)
    c[i] = !fi.integer() ? double(v.float32[i]) : sint ? double(v.int32[i]) : double(v.uint32[i]);
  return c;
}

// Does `area` leave part of the image view's mip uncovered? (Load-op clears cover the whole
// attachment in Metal; Vulkan clears only the render area.)
bool partialArea(const VkRect2D& area, const ImageView* v) {
  if (!area.extent.width || !area.extent.height) return false;  // nothing rendered: nothing to clear
  const uint32_t w = std::max(1u, v->image->desc.width >> v->range.baseMipLevel);
  const uint32_t h = std::max(1u, v->image->desc.height >> v->range.baseMipLevel);
  return area.offset.x > 0 || area.offset.y > 0 || uint32_t(area.offset.x) + area.extent.width < w ||
         uint32_t(area.offset.y) + area.extent.height < h;
}

// Metal has no average resolve for depth (nor min/max for stencil): those keep sample 0.
vpgpu::ResolveMode resolveMode(VkResolveModeFlagBits m) {
  switch (m) {
    case VK_RESOLVE_MODE_NONE: return vpgpu::ResolveMode::None;
    case VK_RESOLVE_MODE_MIN_BIT: return vpgpu::ResolveMode::Min;
    case VK_RESOLVE_MODE_MAX_BIT: return vpgpu::ResolveMode::Max;
    default: return vpgpu::ResolveMode::Sample0;
  }
}

namespace {

vpgpu::ClearValue toClear(const VkClearValue& v, bool depth, vpgpu::Format f) {
  vpgpu::ClearValue c;
  if (depth) {
    c.depth = v.depthStencil.depth;
    c.stencil = static_cast<uint8_t>(v.depthStencil.stencil);
  } else {
    c.color = clearColor(v.color, f);
  }
  return c;
}


// Load/store of attachment `index` in subpass `sub`: the render pass's own ops apply at its first
// and last use; in between the contents carry over (Load / Store; the core keeps them on tile when
// the passes merge and the attachment is transient).
struct UseRange {
  int first = -1, last = -1;
};
UseRange uses(const RenderPassObj& rp, uint32_t index) {
  UseRange u;
  for (size_t s = 0; s < rp.subpasses.size(); ++s) {
    const auto& sp = rp.subpasses[s];
    bool used = sp.hasDepth && sp.depth.attachment == index;
    for (auto& c : sp.colors) used |= c.attachment == index;
    for (auto& c : sp.resolves) used |= c.attachment == index;
    for (auto& c : sp.inputs) used |= c.attachment == index;
    used |= sp.depthResolve == index;
    if (used) {
      if (u.first < 0) u.first = int(s);
      u.last = int(s);
    }
  }
  return u;
}

// A multiview mask of views 0..n-1 (0b1, 0b11, 0b1111): the pass's layers are exactly its views.
bool fullViewMask(uint32_t mask) { return (mask & (mask + 1)) == 0; }

// Multiview: the views of subpass `sub` (its mask) whose first use of attachment `index` is this
// subpass. Vulkan applies the load op per view, at the first subpass that uses that view.
uint32_t newViews(const RenderPassObj& rp, uint32_t index, uint32_t sub) {
  uint32_t earlier = 0;
  for (uint32_t s = 0; s < sub && s < rp.subpasses.size(); ++s) {
    const auto& sp = rp.subpasses[s];
    bool used = sp.hasDepth && sp.depth.attachment == index;
    for (auto& c : sp.colors) used |= c.attachment == index;
    for (auto& c : sp.resolves) used |= c.attachment == index;
    for (auto& c : sp.inputs) used |= c.attachment == index;
    used |= sp.depthResolve == index;
    if (used) earlier |= sp.viewMask;
  }
  return sub < rp.subpasses.size() ? rp.subpasses[sub].viewMask & ~earlier : 0;
}

// The store op that keeps the image's contents: per aspect for depth/stencil formats (one store
// action covers both aspects of one image, so either aspect storing keeps it).
VkAttachmentStoreOp storeOpOf(const VkAttachmentDescription2& d, const vpgpu::FormatInfo& fi) {
  auto keeps = [](VkAttachmentStoreOp op) { return op == VK_ATTACHMENT_STORE_OP_STORE || op == VK_ATTACHMENT_STORE_OP_NONE; };
  if (fi.stencil() && !fi.depth()) return d.stencilStoreOp;
  if (fi.stencil() && fi.depth() && (keeps(d.storeOp) || keeps(d.stencilStoreOp))) return VK_ATTACHMENT_STORE_OP_STORE;
  return d.storeOp;
}

bool attach(const OpBeginPass& b, uint32_t sub, uint32_t index, uint32_t resolveIndex, vpgpu::Attachment& out,
            std::vector<std::pair<const ImageView*, const ImageView*>>* forced = nullptr,
            std::vector<std::pair<const ImageView*, const ImageView*>>* copies = nullptr) {
  if (!b.framebuffer || index >= b.framebuffer->views.size()) return false;
  const ImageView* v = b.framebuffer->views[index];
  if (!v || !v->image || !v->image->id) return false;
  const VkAttachmentDescription2& d = b.pass->attachments[index];
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(v->image->desc.format);
  const bool depthFmt = fi.depth() || fi.stencil();
  const UseRange u = uses(*b.pass, index);
  out.texture = v->image->id;
  // A view of another format (mutable-format image): the pass renders as the view's format.
  const vpgpu::Format cf = attachmentFormat(*v);
  if (cf != v->image->desc.format) out.format = cf;
  out.mip = static_cast<uint16_t>(v->range.baseMipLevel);
  out.layer = static_cast<uint16_t>(v->range.baseArrayLayer);
  // Multiview with a mask that is not views 0..n-1, or views an earlier subpass already used: the
  // pass's layers hold other views' contents, so it loads (clears of the new views are quads,
  // areaClears) and stores.
  const uint32_t mask = sub < b.pass->subpasses.size() ? b.pass->subpasses[sub].viewMask : 0;
  const bool perView = mask && (!fullViewMask(mask) || newViews(*b.pass, index, sub) != mask);
  // Outside the render area the attachment keeps its contents (Vulkan); Metal's DontCare load and
  // store act on the whole attachment, so a partial render area loads and stores instead. An
  // UNDEFINED initial layout has no contents to keep.
  const bool partial = partialArea(b.area, v);
  // Transient (tile-only) attachments have no contents worth keeping outside the area, and
  // forcing a store would move them off tile memory: never promoted.
  const bool keepOutside = partial && d.initialLayout != VK_IMAGE_LAYOUT_UNDEFINED &&
                           !(v->image && v->image->desc.transient);
  // One load action for both aspects of a depth/stencil image: an aspect that loads keeps the
  // image (a cleared other aspect becomes a quad, areaClears).
  VkAttachmentLoadOp loadOp = depthFmt && !fi.depth() ? d.stencilLoadOp : d.loadOp;
  if (fi.depth() && fi.stencil()) {
    const VkAttachmentLoadOp a = d.loadOp, s = d.stencilLoadOp;
    loadOp = a == VK_ATTACHMENT_LOAD_OP_LOAD || s == VK_ATTACHMENT_LOAD_OP_LOAD     ? VK_ATTACHMENT_LOAD_OP_LOAD
             : a == VK_ATTACHMENT_LOAD_OP_CLEAR || s == VK_ATTACHMENT_LOAD_OP_CLEAR ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                                                   : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  }
  out.load = int(sub) == u.first && !perView ? toLoad(loadOp) : vpgpu::LoadAction::Load;
  if (out.load == vpgpu::LoadAction::DontCare && keepOutside) out.load = vpgpu::LoadAction::Load;
  bool resolve = false;
  if (resolveIndex != VK_ATTACHMENT_UNUSED && resolveIndex < b.framebuffer->views.size()) {
    const ImageView* r = b.framebuffer->views[resolveIndex];
    if (r && r->image && r->image->id && resolveAsCopy(v)) {
      // Game MSAA forced to 1x, or a 1x attachment the game "resolves": keep drawing into the 1x
      // source like every other pass does, store it, and copy it into the resolve target after
      // this subpass (never a Metal resolve of a 1x texture).
      out.store = vpgpu::StoreAction::Store;
      if (index < b.clears.size()) out.clear = toClear(b.clears[index], depthFmt, cf);
      if (forced && resolveCopyUsable(v, r)) forced->push_back({v, r});
      return true;
    }
    if (r && r->image && r->image->id && integerResolve(v) && copies) {
      // Integer MSAA: keep the samples (stored), resolve sample 0 by a blit after the subpass.
      copies->push_back({v, r});
      out.store = vpgpu::StoreAction::Store;
      if (index < b.clears.size()) out.clear = toClear(b.clears[index], depthFmt, cf);
      return true;
    }
    if (r && r->image && r->image->id) {
      out.resolve = r->image->id;
      out.resolveMip = static_cast<uint16_t>(r->range.baseMipLevel);
      out.resolveLayer = static_cast<uint16_t>(r->range.baseArrayLayer);
      resolve = true;
    }
  }
  const bool clientEnd = int(sub) == u.last && !(mask && !fullViewMask(mask));  // this subpass ends with the client's own store op
  out.store = clientEnd ? toStore(storeOpOf(d, fi), resolve)
                        : (resolve ? vpgpu::StoreAction::StoreAndResolve : vpgpu::StoreAction::Store);
  const vpgpu::StoreAction clientStore = out.store;
  if (partial && !(v->image && v->image->desc.transient)) {  // pixels outside the area survive
    if (out.store == vpgpu::StoreAction::DontCare) out.store = vpgpu::StoreAction::Store;
    if (out.store == vpgpu::StoreAction::Resolve) out.store = vpgpu::StoreAction::StoreAndResolve;
  }
  // Every StoreAndResolve the translator emits keeps its samples (the client's own STORE, or an upgrade for a later
  // subpass / partial area): only a client Resolve (DONT_CARE + resolve attachment) stays memoryless.
  if (out.store == vpgpu::StoreAction::StoreAndResolve) out.keepSamples = true;
  if (out.store == vpgpu::StoreAction::Store || out.store == vpgpu::StoreAction::StoreAndResolve)
    out.keepOrigin = clientEnd && out.store == clientStore ? vpgpu::KeepOrigin::Client : vpgpu::KeepOrigin::Upgrade;
  if (index < b.clears.size()) out.clear = toClear(b.clears[index], depthFmt, cf);
  return true;
}

// Rects of a clear over `area`: one per view layer of a multiview mask, else layers 0..n-1.
void addAreaRects(vpgpu::ClearRects& cr, const VkRect2D& area, uint32_t layerMask, uint16_t layers) {
  const uint32_t x = uint32_t(std::max(0, area.offset.x)), y = uint32_t(std::max(0, area.offset.y));
  if (!layerMask) {
    cr.rects.push_back({x, y, area.extent.width, area.extent.height, 0, std::max<uint16_t>(layers, 1)});
    return;
  }
  for (uint32_t m = layerMask; m; m &= m - 1)
    cr.rects.push_back({x, y, area.extent.width, area.extent.height, uint16_t(std::countr_zero(m)), 1});
}

// The clears of subpass `sub`'s first uses that load ops cannot express, moved to quads over the
// render area (the attachments then load): a render area smaller than the attachment, depth and
// stencil of one image where one is cleared and the other loaded, and multiview subpasses whose
// views are not exactly the pass's layers (an irregular mask, or a later subpass: only the views
// used here for the first time clear, see attach()). One ClearRects per set of layers.
std::vector<std::shared_ptr<vpgpu::ClearRects>> areaClears(const OpBeginPass& b, uint32_t sub,
                                                           vpgpu::RenderPassDesc& desc) {
  std::vector<std::shared_ptr<vpgpu::ClearRects>> out;
  if (!b.pass || !b.framebuffer || sub >= b.pass->subpasses.size()) return out;
  const RenderPassObj::Subpass& sp = b.pass->subpasses[sub];
  const uint32_t mask = sp.viewMask;
  auto view = [&](uint32_t index) -> const ImageView* {
    if (index == VK_ATTACHMENT_UNUSED || index >= b.framebuffer->views.size()) return nullptr;
    const ImageView* v = b.framebuffer->views[index];
    return v && v->image && v->image->id ? v : nullptr;
  };
  // Layers whose clear attachment `a` needs as a quad here: 0 = every layer of the pass, else a
  // mask of view layers; `none` when this subpass clears nothing of it (not a first use).
  constexpr uint32_t none = ~0u;
  auto quadLayers = [&](uint32_t a) -> uint32_t {
    const bool first = int(sub) == uses(*b.pass, a).first;
    if (mask && (!fullViewMask(mask) || !first)) {
      const uint32_t fresh = newViews(*b.pass, a, sub);
      return fresh ? fresh : none;
    }
    return first ? 0u : none;
  };
  // First use from an UNDEFINED layout: nothing outside the render area / the views has defined
  // contents, so the load action may clear the whole attachment (cheaper than load + quad).
  // A TRANSIENT attachment likewise: it has no contents worth keeping outside the area (attach() never promotes its load
  // or store for a partial area either). Unreal renders 2258 rows of its 2260-row 4x targets: as a Load plus clear quads
  // that first pass read both attachments back from memory every frame (Into the Radius 2).
  auto wholeClear = [&](uint32_t a) {
    if (int(sub) != uses(*b.pass, a).first) return false;
    const ImageView* v = view(a);
    return b.pass->attachments[a].initialLayout == VK_IMAGE_LAYOUT_UNDEFINED || (v && v->image->desc.transient);
  };
  std::vector<uint32_t> keys;
  auto group = [&](uint32_t layers) -> vpgpu::ClearRects& {
    for (size_t i = 0; i < keys.size(); ++i)
      if (keys[i] == layers) return *out[i];
    keys.push_back(layers);
    out.push_back(std::make_shared<vpgpu::ClearRects>());
    addAreaRects(*out.back(), b.area, layers, desc.layerCount);
    return *out.back();
  };
  int slot = 0;
  for (size_t i = 0; i < sp.colors.size() && slot < vpgpu::kMaxColorAttachments; ++i, ++slot) {
    const uint32_t a = sp.colors[i].attachment;
    const ImageView* v = view(a);
    if (!v || b.pass->attachments[a].loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR) continue;
    const uint32_t layers = quadLayers(a);
    if (layers == none) continue;
    if (wholeClear(a)) {
      desc.colors[slot].load = vpgpu::LoadAction::Clear;
      continue;
    }
    // A whole-attachment first use of every layer: the load action clears (attach() set it).
    if (!layers && desc.colors[slot].load == vpgpu::LoadAction::Clear && !partialArea(b.area, v)) continue;
    desc.colors[slot].load = vpgpu::LoadAction::Load;
    vpgpu::ClearRects& cr = group(layers);
    cr.colorMask |= uint8_t(1u << slot);
    if (a < b.clears.size())
      for (int c = 0; c < 4; ++c) cr.colorBits[slot][c] = b.clears[a].color.uint32[c];
  }
  if (sp.hasDepth) {
    const uint32_t a = sp.depth.attachment;
    const ImageView* v = view(a);
    const uint32_t layers = v ? quadLayers(a) : none;
    if (v && layers != none) {
      const vpgpu::FormatInfo& fi = vpgpu::formatInfo(v->image->desc.format);
      const VkAttachmentDescription2& d = b.pass->attachments[a];
      const bool clearD = fi.depth() && d.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
      const bool clearS = fi.stencil() && d.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
      // LOAD from an UNDEFINED layout reads undefined contents: as cheap as DONT_CARE (and keeps a
      // cleared depth on tile instead of loading it for the other aspect).
      const bool undef = d.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED;
      const bool loadD = fi.depth() && d.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && !undef;
      const bool loadS = fi.stencil() && d.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD && !undef;
      const bool partial = partialArea(b.area, v) && !wholeClear(a);
      if ((clearD || clearS) && (partial || loadD || loadS || (layers && !wholeClear(a)))) {
        desc.depth.load = vpgpu::LoadAction::Load;
        vpgpu::ClearRects& cr = group(layers);
        cr.depth = clearD;
        cr.stencil = clearS;
        if (a < b.clears.size()) {
          cr.depthValue = b.clears[a].depthStencil.depth;
          cr.stencilValue = uint8_t(b.clears[a].depthStencil.stencil);
        }
      } else if (clearD || clearS) {
        desc.depth.load = vpgpu::LoadAction::Clear;  // the other aspect is DONT_CARE
      } else if (loadD || loadS) {
        desc.depth.load = vpgpu::LoadAction::Load;
      }
    }
  }
  return out;
}

// The depth/stencil resolve attachment's own load op: CLEAR at its first use (an aspect resolved
// with mode NONE keeps that value; Metal never touches it). Metal cannot clear a resolve target in
// the pass that resolves into it, so it is a clear pass before the subpass, of the whole image —
// only when nothing outside the render area must survive (whole area, or UNDEFINED before).
bool resolveTargetClear(const OpBeginPass& b, uint32_t sub, uint16_t layers, vpgpu::RenderPassDesc& out) {
  if (!b.pass || !b.framebuffer || sub >= b.pass->subpasses.size()) return false;
  const uint32_t a = b.pass->subpasses[sub].depthResolve;
  if (a == VK_ATTACHMENT_UNUSED || a >= b.framebuffer->views.size() || a >= b.pass->attachments.size()) return false;
  const ImageView* v = b.framebuffer->views[a];
  if (!v || !v->image || !v->image->id || int(sub) != uses(*b.pass, a).first) return false;
  const VkAttachmentDescription2& d = b.pass->attachments[a];
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(v->image->desc.format);
  const bool clearD = fi.depth() && d.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
  const bool clearS = fi.stencil() && d.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
  if (!(clearD || clearS) || (partialArea(b.area, v) && d.initialLayout != VK_IMAGE_LAYOUT_UNDEFINED)) return false;
  // One aspect cleared and the other loaded would need a quad pass of its own: left as is.
  if ((fi.depth() && !clearD && d.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) ||
      (fi.stencil() && !clearS && d.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD))
    return false;
  out = {};
  out.depth.texture = v->image->id;
  out.depth.mip = uint16_t(v->range.baseMipLevel);
  out.depth.layer = uint16_t(v->range.baseArrayLayer);
  out.depth.load = vpgpu::LoadAction::Clear;
  out.depth.store = vpgpu::StoreAction::Store;
  if (a < b.clears.size()) out.depth.clear = toClear(b.clears[a], true, v->image->desc.format);
  out.layerCount = std::max<uint16_t>(layers, 1);
  out.label = "vk.resolve_clear";
  return true;
}

// "vk.rp_dump" fuse{...}: for a pass whose subpass reads a multisampled depth image as an input attachment, what keeping that
// read on tile (a colour attachment holding the depth, written by the earlier subpass) would have to touch: the input (attachment,
// format, samples) and, per subpass, the draws, the distinct pipelines and the pipelines that read an input attachment. Today the
// read is a sampled texture (depth_input_in_tile=0): the core cuts the encoder (gpu.pass.fuse says how it ended). Empty when no
// subpass reads such an image.
std::string fuseCensus(const OpBeginPass& b, const std::vector<Op>& ops, size_t at) {
  std::string in;
  for (size_t s = 0; s < b.pass->subpasses.size(); ++s)
    for (const VkAttachmentReference2& r : b.pass->subpasses[s].inputs) {
      const ImageView* v = r.attachment < b.framebuffer->views.size() ? b.framebuffer->views[r.attachment] : nullptr;
      if (!v || !v->image || v->image->desc.samples < 2) continue;
      const vpgpu::FormatInfo& fi = vpgpu::formatInfo(v->image->desc.format);
      if (fi.depth() || fi.stencil())
        in += (in.empty() ? "" : ",") + std::string("sub") + std::to_string(s) + ":att" + std::to_string(r.attachment) + "=" + fi.name + "/s" +
              std::to_string(unsigned(v->image->desc.samples));
    }
  if (in.empty()) return in;
  struct Sub {
    uint32_t draws = 0;
    std::vector<const PipelineObj*> pipes, inPipes;
  };
  std::vector<Sub> subs(1);
  auto note = [&](std::vector<const PipelineObj*>& v, const PipelineObj* p) {
    if (std::find(v.begin(), v.end(), p) == v.end()) v.push_back(p);
  };
  auto visit = [&](auto&& self, const Op& op) -> bool {  // true: the pass ended
    if (const auto* d = std::get_if<OpDraw>(&op)) {
      if (!d->ok || !d->pipe) return false;
      Sub& cur = subs.back();
      ++cur.draws;
      note(cur.pipes, d->pipe);
      for (const PipelineObj::Used& u : d->pipe->used)
        if (u.slot.type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
          note(cur.inPipes, d->pipe);
          break;
        }
    } else if (const auto* e = std::get_if<OpExecute>(&op)) {
      if (e->secondary)
        for (const Op& so : e->secondary->ops)
          if (self(self, so)) return true;
    } else if (std::holds_alternative<OpNextSubpass>(op)) {
      subs.emplace_back();
    } else if (std::holds_alternative<OpEndPass>(op) || std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op)) {
      return true;
    }
    return false;
  };
  for (size_t i = at + 1; i < ops.size(); ++i)
    if (visit(visit, ops[i])) break;
  in += " depth_input_in_tile=0";
  for (size_t s = 0; s < subs.size(); ++s)
    in += " sub" + std::to_string(s) + "{draws=" + std::to_string(subs[s].draws) + " pipes=" + std::to_string(subs[s].pipes.size()) +
          " in_att_pipes=" + std::to_string(subs[s].inPipes.size()) + "}";
  return in;
}

// "vk.rp_dump": what the game asked for in a multisampled render pass of 1 Mpx or more, as Vulkan states it (attachment
// ops, usage, subpasses) and the order of what it records inside (draw counts between subpass changes, barriers and
// clears). The first 6 such passes with a different shape, then one in 3000. One submit's core encoders (gpu.enc) only
// show what the translator made of it; this line is the request itself.
void rpDump(const OpBeginPass& b, const std::vector<Op>& ops, size_t at, uint64_t submit) {
  if (!b.pass || !b.framebuffer) return;
  bool big = false;
  for (const ImageView* v : b.framebuffer->views)
    big = big || (v && v->image && v->image->desc.samples > 1 && uint64_t(v->image->desc.width) * v->image->desc.height >= 1000000);
  if (!big) return;
  // Shape: attachment count, subpass count and ops, so a second pass kind in the same frame is also shown.
  uint64_t shape = b.pass->attachments.size() * 131 + b.pass->subpasses.size();
  for (const VkAttachmentDescription2& d : b.pass->attachments)
    shape = shape * 1000003 + uint64_t(d.format) * 64 + uint64_t(d.samples) * 16 + uint64_t(d.loadOp) * 4 + uint64_t(d.storeOp);
  // ... and the framebuffer size and whether the pass is loaded (16 draws or more): the empty pass of a loading screen has
  // the ops of the level's pass and used to take its place in the log.
  shape = shape * 1000003 + (uint64_t(b.framebuffer->width) << 20) + b.framebuffer->height;
  uint32_t ahead = 0;
  for (size_t i = at + 1; i < ops.size() && ahead < 16; ++i) {
    const Op& op = ops[i];
    if (std::holds_alternative<OpDraw>(op)) ++ahead;
    else if (const auto* e = std::get_if<OpExecute>(&op)) ahead += e->secondary ? uint32_t(std::min<size_t>(e->secondary->ops.size(), 16)) : 0u;
    else if (std::holds_alternative<OpEndPass>(op) || std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op)) break;
  }
  shape = shape * 2 + (ahead >= 16 ? 1 : 0);
  static std::mutex mu;
  static std::vector<uint64_t> seen;
  static uint64_t count = 0;
  {
    std::lock_guard<std::mutex> l(mu);
    const bool fresh = std::find(seen.begin(), seen.end(), shape) == seen.end();
    if (fresh && seen.size() < 6) seen.push_back(shape);
    else if (++count % 3000) return;
  }
  std::string att;
  for (size_t i = 0; i < b.pass->attachments.size(); ++i) {
    const VkAttachmentDescription2& d = b.pass->attachments[i];
    const ImageView* v = i < b.framebuffer->views.size() ? b.framebuffer->views[i] : nullptr;
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s%zu:f%d/s%d/l%d.%d/st%d.%d/lay%d>%d/u0x%x/%ux%ux%u%s", i ? " " : "", i, int(d.format),
                  int(d.samples), int(d.loadOp), int(d.stencilLoadOp), int(d.storeOp), int(d.stencilStoreOp), int(d.initialLayout),
                  int(d.finalLayout), v && v->image ? unsigned(v->image->info.usage) : 0u, v && v->image ? v->image->desc.width : 0u,
                  v && v->image ? v->image->desc.height : 0u, v && v->image ? unsigned(v->image->desc.layers) : 0u,
                  v && v->image && v->image->desc.transient ? "/T" : "");
    att += buf;
  }
  std::string subs;
  auto refs = [](const std::vector<VkAttachmentReference2>& r) {
    std::string o;
    for (const VkAttachmentReference2& a : r) o += (o.empty() ? "" : ",") + (a.attachment == VK_ATTACHMENT_UNUSED ? std::string("-") : std::to_string(a.attachment));
    return o;
  };
  for (size_t i = 0; i < b.pass->subpasses.size(); ++i) {
    const RenderPassObj::Subpass& sp = b.pass->subpasses[i];
    subs += (i ? " | " : "") + std::to_string(i) + ":c[" + refs(sp.colors) + "] r[" + refs(sp.resolves) + "] in[" + refs(sp.inputs) + "] d" +
            (sp.hasDepth && sp.depth.attachment != VK_ATTACHMENT_UNUSED ? std::to_string(sp.depth.attachment) + "/lay" + std::to_string(int(sp.depth.layout)) : std::string("-")) +
            " dr" + (sp.depthResolve == VK_ATTACHMENT_UNUSED ? std::string("-") : std::to_string(sp.depthResolve)) + " vm0x" + std::to_string(sp.viewMask);
  }
  // What is recorded inside: D<n> = n draws, N = next subpass, B = barrier, C(...) = vkCmdClearAttachments, X = secondary buffer.
  std::string seq;
  uint32_t draws = 0;
  auto flush = [&] {
    if (draws) seq += "D" + std::to_string(draws) + " ";
    draws = 0;
  };
  for (size_t i = at + 1; i < ops.size() && seq.size() < 400; ++i) {
    const Op& op = ops[i];
    if (const auto* d = std::get_if<OpDraw>(&op)) draws += d->ok ? 1u : 0u;
    else if (const auto* e = std::get_if<OpExecute>(&op)) {
      if (e->secondary)
        for (const Op& so : e->secondary->ops)
          if (const auto* sd = std::get_if<OpDraw>(&so)) draws += sd->ok ? 1u : 0u;
    } else if (std::holds_alternative<OpNextSubpass>(op)) flush(), seq += "N ";
    else if (std::holds_alternative<OpPassBarrier>(op)) flush(), seq += "B ";
    else if (const auto* ca = std::get_if<OpClearAttachments>(&op)) {
      flush();
      seq += "C(";
      for (const VkClearAttachment& a : ca->attachments)
        seq += (a.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? "c" + std::to_string(a.colorAttachment) : std::string("")) +
               (a.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT ? "d" : "") + (a.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT ? "s" : "");
      seq += ":" + std::to_string(ca->rects.size()) + "r";
      if (!ca->rects.empty())
        seq += " " + std::to_string(ca->rects[0].rect.extent.width) + "x" + std::to_string(ca->rects[0].rect.extent.height);
      seq += ") ";
    } else if (std::holds_alternative<OpEndPass>(op) || std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op))
      break;
  }
  flush();
  logf("vk.rp_dump submit=%llu area=%d,%d+%ux%u fb=%ux%ux%u att{%s} sub{%s} seq{%s}", static_cast<unsigned long long>(submit),
       b.area.offset.x, b.area.offset.y, b.area.extent.width, b.area.extent.height, b.framebuffer->width, b.framebuffer->height,
       unsigned(b.framebuffer->layers), att.c_str(), subs.c_str(), seq.c_str());
  if (const std::string fuse = fuseCensus(b, ops, at); !fuse.empty())
    logf("vk.pass.fuse submit=%llu fb=%ux%ux%u in=%s", static_cast<unsigned long long>(submit), b.framebuffer->width,
         b.framebuffer->height, unsigned(b.framebuffer->layers), fuse.c_str());
}

// Does subpass `sub` read one of its own colour / depth attachments as an input attachment?
bool selfReads(const RenderPassObj& rp, uint32_t sub) {
  if (sub >= rp.subpasses.size()) return false;
  const auto& sp = rp.subpasses[sub];
  for (const auto& in : sp.inputs) {
    if (in.attachment == VK_ATTACHMENT_UNUSED) continue;
    if (sp.hasDepth && sp.depth.attachment == in.attachment) return true;
    for (const auto& c : sp.colors)
      if (c.attachment == in.attachment) return true;
  }
  return false;
}

// Is there a self-dependency barrier between ops[at] (the begin of a subpass) and the end of that subpass? Secondary
// command buffers executed in it count.
bool barrierAhead(const std::vector<Op>& ops, size_t at) {
  for (size_t i = at + 1; i < ops.size(); ++i) {
    const Op& op = ops[i];
    if (std::holds_alternative<OpPassBarrier>(op)) return true;
    if (const auto* e = std::get_if<OpExecute>(&op)) {
      if (e->secondary)
        for (const Op& so : e->secondary->ops)
          if (std::holds_alternative<OpPassBarrier>(so)) return true;
    } else if (std::holds_alternative<OpNextSubpass>(op) || std::holds_alternative<OpEndPass>(op) ||
               std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op)) {
      return false;
    }
  }
  return false;
}

// Subpass `sub` of a render pass as one core pass. Input attachments are read as textures (they
// were stored by the earlier subpass); passes that the core merges keep them on tile.
// `keepForSplit`: a self-reading subpass that will be split at a barrier stores every attachment (see below). False when no
// barrier follows: the subpass is one core pass and ends with the client's own store ops.
bool beginPass(const OpBeginPass& b, uint32_t sub, vpgpu::RenderPassDesc& desc,
               std::vector<vpgpu::Copy>* resolveCopies = nullptr, bool keepForSplit = true) {
  if (!b.pass || sub >= b.pass->subpasses.size()) return false;
  const RenderPassObj::Subpass& sp = b.pass->subpasses[sub];
  std::vector<std::pair<const ImageView*, const ImageView*>> forced, intResolves;
  int slot = 0;
  for (size_t i = 0; i < sp.colors.size() && slot < vpgpu::kMaxColorAttachments; ++i, ++slot) {
    const uint32_t a = sp.colors[i].attachment;
    if (a == VK_ATTACHMENT_UNUSED) continue;
    const uint32_t r = i < sp.resolves.size() ? sp.resolves[i].attachment : VK_ATTACHMENT_UNUSED;
    if (!attach(b, sub, a, r, desc.colors[slot], &forced, &intResolves)) return false;
  }
  if (sp.hasDepth && sp.depth.attachment != VK_ATTACHMENT_UNUSED) {
    // VK_KHR_depth_stencil_resolve: modes Metal lacks (average) fall back to sample 0.
    const bool dsResolve = sp.depthResolve != VK_ATTACHMENT_UNUSED &&
                           (sp.depthResolveMode != VK_RESOLVE_MODE_NONE || sp.stencilResolveMode != VK_RESOLVE_MODE_NONE);
    if (!attach(b, sub, sp.depth.attachment, dsResolve ? sp.depthResolve : VK_ATTACHMENT_UNUSED, desc.depth, &forced))
      return false;
    desc.depth.depthResolve = resolveMode(sp.depthResolveMode);
    desc.depth.stencilResolve = resolveMode(sp.stencilResolveMode);
  }
  // A subpass reading its own attachment as an input attachment (self-dependency) is split at its
  // barriers (Translator, OpPassBarrier): every part stores, so the next part loads what it drew.
  // The last part (and a subpass with no barrier at all: Unreal's translucency subpass reads the read-only depth the
  // opaque subpass stored) ends with the client's store ops: forcing a Store there wrote 4x colour and depth to memory
  // every frame for nobody.
  if (keepForSplit && selfReads(*b.pass, sub)) {
    auto keep = [](vpgpu::Attachment& a) {
      if (!a.texture) return;
      if (a.store == vpgpu::StoreAction::DontCare) a.store = vpgpu::StoreAction::Store, a.keepOrigin = vpgpu::KeepOrigin::Upgrade;
      if (a.store == vpgpu::StoreAction::Resolve)
        a.store = vpgpu::StoreAction::StoreAndResolve, a.keepSamples = true, a.keepOrigin = vpgpu::KeepOrigin::Upgrade;
    };
    for (vpgpu::Attachment& a : desc.colors) keep(a);
    keep(desc.depth);
  }
  // Multiview: one layer per view bit (the draws are instanced once per view).
  desc.layerCount = sp.viewMask ? static_cast<uint16_t>(std::bit_width(sp.viewMask))  // view = layer = bit index
                                : static_cast<uint16_t>(b.framebuffer ? b.framebuffer->layers : 1);
  desc.label = "vk.pass";
  // Attachmentless pass: the framebuffer's size (only then: it would stop merges otherwise).
  bool attached = bool(desc.depth.texture);
  for (const vpgpu::Attachment& a : desc.colors) attached = attached || bool(a.texture);
  if (!attached && b.framebuffer) desc.width = b.framebuffer->width, desc.height = b.framebuffer->height;
  desc.renderWidth = uint32_t(std::max<int32_t>(0, b.area.offset.x)) + b.area.extent.width;
  desc.renderHeight = uint32_t(std::max<int32_t>(0, b.area.offset.y)) + b.area.extent.height;
  if (resolveCopies) {
    for (const auto& [src, dst] : forced) resolveCopies->push_back(forcedResolveCopy(src, dst, std::max<uint16_t>(desc.layerCount, 1)));
    for (const auto& [src, dst] : intResolves)
      resolveCopies->push_back(sampleZeroResolve(src, dst, std::max<uint16_t>(desc.layerCount, 1)));
  }
  return true;
}

uint32_t mipsOf(const Image* im, const VkImageSubresourceRange& r) {
  return r.levelCount == VK_REMAINING_MIP_LEVELS ? im->desc.mips - r.baseMipLevel : r.levelCount;
}
uint32_t layersOf(const Image* im, const VkImageSubresourceRange& r) {
  return r.layerCount == VK_REMAINING_ARRAY_LAYERS ? im->desc.layers - r.baseArrayLayer : r.layerCount;
}

// VPKTAA jitter: does the pass draw the eye's geometry? It must have a depth attachment and either
// be multiview (both eyes in one layered pass) or match the eye size the caller gave, by its render
// area or by its attachments. Shadow maps, UI layers and post-process targets of other sizes fail.
// Returns the size of the target (texels) through w/h when it qualifies. The eye size is the jitter's; when that is 0 it is
// the size of the eye image the host marked (markedEye, packEyeSize), so eye detection does not depend on VPKTAA.
// `ew` (diagnostics, forced MSAA rejection "eye"): when the answer is no, why, with what the pass was judged by.
struct EyeWhy {
  enum Why : uint8_t { NoDepth, NoSize, Size, Other } why = Other;  // Other: set by the caller (no jitter snapshot, subpass out of range)
  uint32_t w = 0, h = 0, fbW = 0, fbH = 0, depthW = 0, depthH = 0, viewMask = 0, eyeW = 0, eyeH = 0;
  bool jit = false;  // eyeW x eyeH is the jitter's size (else the host-marked eye image's)
};
bool taaEyePass(const TaaJitter& jit, bool hasDepth, uint32_t viewMask, const VkRect2D& area, uint32_t fbW, uint32_t fbH,
                uint32_t depthW, uint32_t depthH, uint32_t& w, uint32_t& h, uint64_t markedEye = 0, EyeWhy* ew = nullptr) {
  struct { uint32_t width, height; } j{jit.width, jit.height};
  const bool jitterSize = j.width && j.height;
  if (!jitterSize) j = {eyeSizeW(markedEye), eyeSizeH(markedEye)};
  if (ew) {
    ew->w = depthW ? depthW : fbW ? fbW : area.extent.width, ew->h = depthW ? depthH : fbW ? fbH : area.extent.height;
    ew->fbW = fbW, ew->fbH = fbH, ew->depthW = depthW, ew->depthH = depthH, ew->viewMask = viewMask;
    ew->eyeW = j.width, ew->eyeH = j.height, ew->jit = jitterSize;
  }
  auto no = [&](EyeWhy::Why why) {
    if (ew) ew->why = why;
    return false;
  };
  if (!hasDepth) return no(EyeWhy::NoDepth);
  w = depthW ? depthW : fbW ? fbW : area.extent.width;
  h = depthW ? depthH : fbW ? fbH : area.extent.height;
  const bool multiview = std::popcount(viewMask) >= 2;
  if (!j.width || !j.height) return multiview || no(EyeWhy::NoSize);
  auto is = [&](uint32_t x, uint32_t y) { return x == j.width && y == j.height; };
  if (is(area.extent.width, area.extent.height) || is(fbW, fbH) || is(depthW, depthH)) return true;
  // A multiview pass of another size is still the eye's scene when it is a render-scaled copy of
  // it (50-100 % per axis); multiview shadow/probe passes of other sizes are not, and would also
  // misreport the target size the resolve scales the jitter by.
  // (The marked size only adds detections: without a jitter size a multiview pass is an eye pass at any size, as before.)
  return (multiview && (!jitterSize || (w * 2 >= j.width && w <= j.width && h * 2 >= j.height && h <= j.height))) || no(EyeWhy::Size);
}

// Indirect arguments past these are not a real draw but garbage (an uninitialised or stale buffer): it would run for seconds
// and the GPU watchdog would kill the command buffer, and the queue with it. The draw is dropped and counted. Per Vulkan
// record, before the multiview instance multiplication; the argument patch kernel (pipelines.cpp) applies the same limits
// on the GPU.
constexpr uint32_t kMaxIndirectCount = 1u << 25;      // indices / vertices of one draw
constexpr uint32_t kMaxIndirectInstances = 1u << 20;  // instances of one draw

struct Translator {
  vpgpu::Frame& frame;
  TranslateStats& st;
  const TaaJitter* taa = nullptr;  // enabled jitter, or null
  bool taaPass = false;            // the open pass is an eye pass (taaEyePass)
  uint32_t taaW = 0, taaH = 0;     // its target size
  // Forced MSAA (msaaForce, forceMsaa below). eye: the jitter that identifies eye passes (the enabled one, else a snapshot
  // taken only while msaaForce is on: eye passes are found by eye size / multiview even with VPKTAA off); eyeScene: the
  // (sub)pass about to open passes taaEyePass with it (taaPass additionally needs the jitter enabled); forceN > 0: the open pass
  // renders into Attachment::autoMsaa twins of that many samples and its draws use pipelines of that count.
  const TaaJitter* eye = nullptr;
  uint64_t markedEye = 0;  // packEyeSize of the host-marked eye image (Device::eyeSize), read once per translate
  bool eyeScene = false;
  EyeWhy eyeWhy;  // why eyeScene is false (forceMsaa's "eye" rejection diagnostics)
  uint8_t forceN = 0;
  vpgpu::TextureId lastColor0, lastDepth;  // colour 0 and depth of the pass opened last in this translation (open)
  std::optional<vpgpu::RenderPassDesc> selfFinal;  // a self-reading subpass that splits: its pass with the client's own stores
  // VPKTAA UI mask. uiOn: classify the draws of eye passes (taa::DrawClassifier) and leave the UI
  // unjittered; `um` also lets the pass carry the R8 mask (maskPass) the UI draws mark.
  bool uiOn = false;
  UiMaskState* um = nullptr;
  taa::DrawClassifier ownClassifier;  // when there is no device state (tests)
  bool maskPass = false;              // the open pass has the mask attachment: every draw uses a mask variant
  UiMaskState::EyePass* eyeRec = nullptr;  // the open eye pass's record (depth the host did not give)
  // Foveation (foveation.h): `fovs` non-null = rate maps may apply. fovPass: the open pass carries a rate map.
  // fovSuppressed: the eye pass already went on unfoveated (UI split), a reopen must not foveate it again.
  // fovArea: the pass about to open has area-clear quads (they carry no viewport index): never foveated.
  FoveationState* fovs = nullptr;
  bool fovPass = false, fovSuppressed = false, fovArea = false;
  std::unordered_map<const OpDraw*, taa::DrawClass> classes;  // of the open eye pass, by draw
  taa::DrawClassifier& classifier() { return um ? um->classifier : ownClassifier; }
  bool passOpen = false;
  uint32_t secondaryDepth = 0;  // > 0 while replaying a secondary command buffer (OpExecute)
  vpgpu::RenderPassDesc current;  // the open core pass (reopened with Load after a split)
  const OpBeginPass* pass = nullptr;
  uint32_t subpass = 0;
  uint32_t viewMask = 0;  // multiview mask of the open (sub)pass: clears cover every view
  std::vector<vpgpu::Copy> pendingResolves;  // forced-1x resolves of the open (sub)pass
  // VK_KHR_dynamic_rendering_local_read (localReadSplit): attachments of the open dynamic rendering pass that draws after
  // an in-pass barrier read through a copy: attachment texture -> its copy. Null device (tests): no copies.
  Device* device = nullptr;
  std::vector<std::pair<vpgpu::TextureId, vpgpu::TextureId>> localCopies;
  // Depth in tile (VPVK_DEPTH_TILE, planDepthTile): a pass of N >= 2 subpasses whose later subpasses read the multisampled depth the
  // subpass 0 writes as an input attachment. Colour attachment kDepthTileSlot (a memoryless R32F texture) holds the depth in every
  // subpass: the draws of subpass 0 also write it, those of the later subpasses read it by framebuffer fetch per sample
  // (pixel-local), so the frame compiler merges the subpasses into one encoder and the 4x depth is never stored for the read. `uses`: the pipeline variant
  // of every (pipeline, subpass) the pass draws with (depthPipelineKey).
  struct DepthTile {
    struct Use {
      int kind = kDepthTileState;
      int texSlot = -1;
      vpgpu::PipelineKey key;
    };
    bool active = false;
    vpgpu::Attachment shadow;
    std::unordered_map<uint64_t, Use> uses;
  } dt;
  // The device whatever the caller passed (`device` is only set with the native extras): null in tests without one.
  const Device* anyDevice() const { return device ? device : zl && zl->dev ? zl->dev : um ? um->dev : nullptr; }

  // Zero-init touch log (vpvk.h ZeroLog): only the pending images the frame uses are logged, in
  // order. Commands emitted since the last call are inspected once (attachments, copies); draw and
  // dispatch reads come from the recorded op, so no binding list is walked here.
  ZeroLog* zl = nullptr;
  size_t zeroScanned = 0;
  bool zeroPending(const vpgpu::TextureId& t) const {
    return zl->dev->zeroPending.count(uint64_t{t.generation} << 32 | t.index) != 0;
  }
  void zeroRead(const vpgpu::TextureId& t) {
    if (!t || !zeroPending(t)) return;
    ZeroTouch z;
    z.tex = t;
    zl->touches.push_back(z);
  }
  void zeroWrite(const vpgpu::TextureId& t, uint32_t mip, uint32_t layer, uint32_t layers, const vpgpu::TextureSpan* span) {
    if (!t || !zeroPending(t)) return;
    ZeroTouch z;
    z.kind = span ? ZeroTouch::WriteSpan : ZeroTouch::WriteWhole;
    z.tex = t, z.mip = uint16_t(mip), z.layer = uint16_t(layer), z.layers = uint16_t(layers);
    if (span) z.x = span->x, z.y = span->y, z.z = span->z, z.width = span->width, z.height = span->height;
    zl->touches.push_back(z);
  }
  void zeroScan() {
    if (!zl || zl->dev->zeroPending.empty()) {
      if (zl) zeroScanned = frame.commands().size();
      return;
    }
    const auto& cmds = frame.commands();
    for (; zeroScanned < cmds.size(); ++zeroScanned) {
      const vpgpu::Command& cmd = cmds[zeroScanned];
      if (const auto* rp = std::get_if<vpgpu::RenderPass>(&cmd)) {
        auto att = [&](const vpgpu::Attachment& a) {
          if (a.texture) {
            if (a.load == vpgpu::LoadAction::Load) zeroRead(a.texture);
            else zeroWrite(a.texture, a.mip, a.layer, rp->desc.layerCount, nullptr);
          }
          if (a.resolve && (a.store == vpgpu::StoreAction::Resolve || a.store == vpgpu::StoreAction::StoreAndResolve))
            zeroWrite(a.resolve, a.resolveMip, a.resolveLayer, rp->desc.layerCount, nullptr);
        };
        for (const vpgpu::Attachment& a : rp->desc.colors) att(a);
        att(rp->desc.depth);
      } else if (const auto* cp = std::get_if<vpgpu::Copy>(&cmd)) {
        const vpgpu::TextureSpan& d = cp->dstTexture;
        switch (cp->kind) {
          case vpgpu::CopyKind::BufferToTexture: zeroWrite(d.texture, d.mip, d.layer, d.layerCount, &d); break;
          case vpgpu::CopyKind::TextureToTexture:
          case vpgpu::CopyKind::TextureBlit:
            zeroRead(cp->srcTexture.texture);
            zeroWrite(d.texture, d.mip, d.layer, d.layerCount, &d);
            break;
          case vpgpu::CopyKind::TextureToBuffer: zeroRead(cp->srcTexture.texture); break;
          default: break;
        }
      }
    }
  }
  void zeroReads(const std::vector<vpgpu::TextureId>& v) {
    if (!zl) return;
    for (const vpgpu::TextureId& t : v) zeroRead(t);
  }

  // Linear images (Image::linear): the texels live in the mapped memory as well as in the texture.
  // Before a transfer touches one, the texture is refreshed from memory (the host may have written
  // it); after a transfer writes it, memory gets the texture back (the host reads it after the
  // fence). Copies through the image's host-aliasing buffer, ordered by the frame compiler.
  void linearSync(const Image* im, bool toTexture) {
    if (!im || !im->linear || !im->id || !im->hostAlias || !im->memory) return;
    const vpgpu::TextureDesc& d = im->desc;
    const LinearLayout last = linearLayout(d, d.mips - 1u, d.layers - 1u);
    if (last.offset + last.size > im->memory->size - std::min(im->memory->size, im->offset)) return;  // memory too small
    // 3D: one layer holding the mip's depth slices, depthPitch (= rowsPerImage rows) apart.
    const bool is3d = d.type == vpgpu::TextureType::Tex3D;
    const uint16_t layers = is3d ? uint16_t(1) : uint16_t(d.layers);
    for (uint32_t m = 0; m < d.mips; ++m) {
      const LinearLayout l = linearLayout(d, m, 0);
      vpgpu::Copy c;
      c.kind = toTexture ? vpgpu::CopyKind::BufferToTexture : vpgpu::CopyKind::TextureToBuffer;
      vpgpu::BufferSpan& b = toTexture ? c.srcBuffer : c.dstBuffer;
      vpgpu::TextureSpan& t = toTexture ? c.dstTexture : c.srcTexture;
      b.buffer = im->hostAlias;
      b.offset = l.offset;
      b.size = l.arrayPitch * layers;
      b.bytesPerRow = uint32_t(l.rowPitch);
      b.rowsPerImage = l.rows;
      t.texture = im->id;
      t.mip = uint16_t(m);
      t.layer = 0;
      t.layerCount = layers;
      t.width = std::max(1u, d.width >> m);
      t.height = std::max(1u, d.height >> m);
      t.depth = is3d ? std::max(1u, d.depth >> m) : 1u;
      c.label = toTexture ? "vk.linear_in" : "vk.linear_out";
      frame.copy(std::move(c));
    }
  }

  // In-pass clear quads (vkCmdClearAttachments, area clears) into the open pass.
  void clearQuads(std::shared_ptr<const vpgpu::ClearRects> cr) {
    if (!cr || !passOpen || cr->rects.empty() || (!cr->colorMask && !cr->depth && !cr->stencil)) return;
    vpgpu::Draw dr;
    dr.clear = std::move(cr);
    frame.draw(std::move(dr));
    ++st.clears;
  }
  void clearQuads(std::vector<std::shared_ptr<vpgpu::ClearRects>> list) {
    for (auto& cr : list) clearQuads(std::shared_ptr<const vpgpu::ClearRects>(std::move(cr)));
  }

  void resolveClear(const OpBeginPass& b, uint32_t sub, uint16_t layers) {
    vpgpu::RenderPassDesc d;
    if (!resolveTargetClear(b, sub, layers, d)) return;
    frame.clear(std::move(d));
    ++st.clears;
  }

  void flushResolves() {
    for (vpgpu::Copy& c : pendingResolves) {
      frame.copy(std::move(c));
      ++st.copies;
    }
    pendingResolves.clear();
  }

  void open(vpgpu::RenderPassDesc d) {
    lastColor0 = d.colors[0].texture, lastDepth = d.depth.texture;  // what the next pass may continue (uiPass)
    // An eye pass keeps its depth: VPKTAA reads it after the frame, outside anything the frame
    // compiler can see, so a DONT_CARE store must not let the depth live only in tile memory.
    // (A forced-MSAA depth twin resolves into the game's image instead: forceMsaa.)
    if (taaPass && !taa->still && d.depth.texture && !d.depth.autoMsaa) {  // RCAS-only (still) frames never read the depth: no forced store
      if (d.depth.store == vpgpu::StoreAction::DontCare) d.depth.store = vpgpu::StoreAction::Store;
      if (d.depth.store == vpgpu::StoreAction::Resolve)
        d.depth.store = vpgpu::StoreAction::StoreAndResolve, d.depth.keepSamples = true;
      d.depth.keepOrigin = vpgpu::KeepOrigin::Required;  // VPKTAA reads these samples after the frame: msaaDeadStore must not guess
      // A TRANSIENT depth would still be dropped with the tile (the compiler keeps no Store of a transient texture): it is
      // an ordinary texture from now on, with memory the VPKTAA pass can read.
      if (const Device* dv = anyDevice())
        if (dv->gpu) dv->gpu->markTransientViolated(d.depth.texture);
    }
    // Foveation: a stereo eye pass with colour and depth renders through a rate map; the backend draws it into
    // physically smaller attachments and expands them into the logical ones when the pass ends. A reopened
    // foveated pass (self-dependency split) keeps its map.
    fovPass = false;
    if (d.rateMap) {
      fovPass = true;
    } else if (fovs && taaPass && !fovSuppressed && !fovArea && d.layerCount == fov::kViews && d.colors[0].texture &&
               d.depth.texture && !d.colors[0].autoMsaa) {  // (the backend never foveates an MSAA pass: forced MSAA wins)
      if (std::shared_ptr<const vpgpu::RateMap> m = fovs->mapFor(taaW, taaH, d.layerCount)) d.rateMap = std::move(m), fovPass = true;
    }
    if (fovPass) {
      // What a later UI split loads again must have been stored.
      for (vpgpu::Attachment& a : d.colors)
        if (a.texture && a.store == vpgpu::StoreAction::DontCare) a.store = vpgpu::StoreAction::Store;
      ++st.fovPasses;
      st.fovLogicalW = taaW, st.fovLogicalH = taaH;
    }
    ddFlush();
    eyeRec = nullptr;
    if (taaPass && um && d.depth.texture && d.colors[0].texture) {
      UiMaskState::EyePass& r = um->eyePass[um->eyePasses++ % UiMaskState::kEyePasses];
      r = {};
      // (the game's images, not the forced-MSAA twins: VPKTAA reads what the pass resolved into)
      r.color = d.colors[0].autoMsaa ? d.colors[0].resolve : d.colors[0].texture;
      r.colorLayer = d.colors[0].autoMsaa ? d.colors[0].resolveLayer : d.colors[0].layer;
      r.depth = d.depth.autoMsaa ? d.depth.resolve : d.depth.texture;
      r.depthLayer = d.depth.autoMsaa ? d.depth.resolveLayer : d.depth.layer;
      r.layers = std::max<uint16_t>(d.layerCount, 1);
      r.cleared = d.depth.load == vpgpu::LoadAction::Clear, r.clearValue = d.depth.clear.depth;
      eyeRec = &r;
    }
    current = d;
    frame.beginRenderPass(std::move(d));
    passOpen = true;
    ++st.passes;
  }
  void close() {
    ddFlush();
    if (passOpen) frame.endRenderPass();
    passOpen = false;
    eyeRec = nullptr;
  }

  void run(const Recorder& rec) {
    for (size_t opIx = 0; opIx < rec.ops.size(); ++opIx) {
      const Op& op = rec.ops[opIx];
      zeroScan();
      if (const auto* b = std::get_if<OpBeginPass>(&op)) {
        pass = b;
        subpass = 0;
        rpDump(*b, rec.ops, opIx, device ? device->submits + 1 : 0);
        vpgpu::RenderPassDesc desc;
        pendingResolves.clear();
        fovSuppressed = false;
        taaSubpass(*b, 0);
        viewMask = b->pass && !b->pass->subpasses.empty() ? b->pass->subpasses[0].viewMask : 0;
        dt = DepthTile{};
        selfFinal.reset();
        const bool split0 = barrierAhead(rec.ops, opIx);
        if (split0 && b->pass && selfReads(*b->pass, 0)) {  // the client's own stores, for the part after the last barrier
          vpgpu::RenderPassDesc fin;
          if (beginPass(*b, 0, fin, nullptr, false)) selfFinal = fin;
        }
        if (beginPass(*b, 0, desc, &pendingResolves, split0)) {
          auto cr = areaClears(*b, 0, desc);
          fovArea = !cr.empty();
          resolveClear(*b, 0, desc.layerCount);
          const bool multi = b->pass && (b->pass->subpasses.size() > 1 || !b->pass->subpasses[0].inputs.empty());
          forceMsaa(desc, rec.ops, opIx, b->area, false, multi);
          gameAuto(desc, b->area, false, multi);
          uiPass(rec.ops, opIx, desc);
          planDepthTile(*b, rec.ops, opIx, desc,
                        std::any_of(cr.begin(), cr.end(), [](const auto& q) { return q && (q->depth || q->stencil); }));
          open(std::move(desc));
          clearQuads(std::move(cr));
        } else {
          ++st.dropped;
        }
      } else if (const auto* br = std::get_if<OpBeginRendering>(&op)) {
        pass = nullptr;
        viewMask = br->viewMask;
        fovSuppressed = false;
        fovArea = br->areaClear && !br->areaClear->rects.empty();
        pendingResolves = br->resolveCopies;
        eyeWhy = EyeWhy{};
        eyeScene = eye && taaEyePass(*eye, br->depthWidth != 0, br->viewMask, br->area, 0, 0, br->depthWidth, br->depthHeight,
                                     taaW, taaH, markedEye, &eyeWhy);
        taaPass = taa && eyeScene;
        vpgpu::RenderPassDesc bd = br->desc;
        localCopies.clear();
        if (device && rec.hasLocalRead && localReadAhead(rec, opIx)) {
          // The pass is split at the barrier (localReadSplit): every attachment stores there, the next part loads it.
          auto keep = [](vpgpu::Attachment& a) {
            if (!a.texture) return;
            if (a.store == vpgpu::StoreAction::DontCare) a.store = vpgpu::StoreAction::Store, a.keepOrigin = vpgpu::KeepOrigin::Upgrade;
            if (a.store == vpgpu::StoreAction::Resolve)
              a.store = vpgpu::StoreAction::StoreAndResolve, a.keepSamples = true, a.keepOrigin = vpgpu::KeepOrigin::Upgrade;
          };
          for (vpgpu::Attachment& a : bd.colors) keep(a);
          keep(bd.depth);
        }
        forceMsaa(bd, rec.ops, opIx, br->area, rec.hasLocalRead, false);
        gameAuto(bd, br->area, rec.hasLocalRead, false);
        uiPass(rec.ops, opIx, bd);
        open(std::move(bd));
        clearQuads(br->areaClear);
      } else if (std::holds_alternative<OpNextSubpass>(op)) {
        localCopies.clear();
        close();
        flushResolves();
        ++subpass;
        vpgpu::RenderPassDesc desc;
        fovSuppressed = false;
        if (pass) taaSubpass(*pass, subpass);
        viewMask = pass && pass->pass && subpass < pass->pass->subpasses.size() ? pass->pass->subpasses[subpass].viewMask : 0;
        selfFinal.reset();
        const bool splitN = barrierAhead(rec.ops, opIx);
        if (splitN && pass && pass->pass && selfReads(*pass->pass, subpass)) {
          vpgpu::RenderPassDesc fin;
          if (beginPass(*pass, subpass, fin, nullptr, false)) selfFinal = fin;
        }
        if (pass && beginPass(*pass, subpass, desc, &pendingResolves, splitN)) {
          auto cr = areaClears(*pass, subpass, desc);
          fovArea = !cr.empty();
          resolveClear(*pass, subpass, desc.layerCount);
          forceMsaa(desc, rec.ops, opIx, pass->area, false, true);  // (a later subpass of a multi-subpass pass: never forced)
          gameAuto(desc, pass->area, false, true);
          uiPass(rec.ops, opIx, desc);
          depthTileReader(desc);
          open(std::move(desc));
          clearQuads(std::move(cr));
        } else {
          ++st.dropped;
        }
      } else if (std::holds_alternative<OpPassBarrier>(op)) {
        // Subpass self-dependency: what was drawn (or cleared) before the barrier must be visible
        // to the input-attachment reads after it. Those reads sample the attachment's memory, so
        // the pass ends here (it stores, see beginPass) and reopens loading everything; the frame
        // compiler orders the two (the reads conflict with the first part's writes).
        if (passOpen && pass && pass->pass && selfReads(*pass->pass, subpass)) {
          vpgpu::RenderPassDesc d = current;
          for (vpgpu::Attachment& a : d.colors)
            if (a.texture) a.load = vpgpu::LoadAction::Load;
          if (d.depth.texture) d.depth.load = vpgpu::LoadAction::Load;
          // The part after the last barrier ends with the client's stores (inside a secondary buffer the rest of the
          // subpass is not visible from here: it keeps storing).
          if (selfFinal && secondaryDepth == 0 && !barrierAhead(rec.ops, opIx)) {
            auto own = [](vpgpu::Attachment& a, const vpgpu::Attachment& f) {
              if (a.texture && a.texture == f.texture && !a.autoMsaa) a.store = f.store, a.keepSamples = f.keepSamples, a.keepOrigin = f.keepOrigin;
            };
            for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c) own(d.colors[c], selfFinal->colors[c]);
            own(d.depth, selfFinal->depth);
          }
          close();
          open(std::move(d));
          ++st.selfDependencySplits;
        } else if (passOpen && !pass && device && rec.hasLocalRead) {
          localReadSplit(rec, opIx);
        }
      } else if (std::holds_alternative<OpEndPass>(op)) {
        localCopies.clear();
        close();
        flushResolves();
        dt = DepthTile{};
        pass = nullptr;
        taaPass = false;
        forceN = 0;
        maskPass = false;
        fovPass = fovSuppressed = false;
      } else if (const auto* c = std::get_if<OpCopyBuffer>(&op)) {
        if (!c->src->id || !c->dst->id) { ++st.dropped; continue; }
        frame.copyBuffer({c->src->id, c->region.srcOffset, c->region.size},
                         {c->dst->id, c->region.dstOffset, c->region.size});
        ++st.copies;
      } else if (const auto* u = std::get_if<OpUpload>(&op)) {
        if (!u->dst->id) { ++st.dropped; continue; }
        frame.copyBuffer({u->src, u->srcOffset, u->size}, {u->dst->id, u->dstOffset, u->size});
        ++st.copies;
      } else if (const auto* f = std::get_if<OpFillBuffer>(&op)) {
        if (!f->dst->id) { ++st.dropped; continue; }
        const VkDeviceSize size = f->size == VK_WHOLE_SIZE ? f->dst->size - f->offset : f->size;
        frame.fillBuffer({f->dst->id, f->offset, size}, static_cast<uint8_t>(f->data));
        ++st.copies;
      } else if (const auto* cc = std::get_if<OpClearColor>(&op)) {
        if (!cc->image->id) { ++st.dropped; continue; }
        linearSync(cc->image, true);
        const bool is3d = cc->image->desc.type == vpgpu::TextureType::Tex3D;
        for (uint32_t m = 0; m < mipsOf(cc->image, cc->range); ++m) {
          // A 3D image is cleared slice by slice (the attachment layer is the depth plane there).
          const uint32_t mip = cc->range.baseMipLevel + m;
          const uint32_t slices = is3d ? std::max(1u, cc->image->desc.depth >> mip) : layersOf(cc->image, cc->range);
          for (uint32_t l = 0; l < slices; ++l) {
            vpgpu::RenderPassDesc d;
            vpgpu::Attachment& a = d.colors[0];
            a.texture = cc->image->id;
            a.mip = static_cast<uint16_t>(mip);
            a.layer = static_cast<uint16_t>(is3d ? l : cc->range.baseArrayLayer + l);
            a.load = vpgpu::LoadAction::Clear;
            a.store = vpgpu::StoreAction::Store, a.keepOrigin = vpgpu::KeepOrigin::Required;  // (an explicit write: never a dead-store guess)
            a.clear.color = clearColor(cc->color, cc->image->desc.format);
            if (isSwappedRB(cc->image->info.format)) std::swap(a.clear.color[0], a.clear.color[2]);
            d.label = "vk.clear";
            frame.clear(std::move(d));
            ++st.clears;
          }
        }
        linearSync(cc->image, false);
      } else if (const auto* cd = std::get_if<OpClearDepth>(&op)) {
        if (!cd->image->id) { ++st.dropped; continue; }
        for (uint32_t m = 0; m < mipsOf(cd->image, cd->range); ++m)
          for (uint32_t l = 0; l < layersOf(cd->image, cd->range); ++l) {
            vpgpu::RenderPassDesc d;
            d.depth.texture = cd->image->id;
            d.depth.mip = static_cast<uint16_t>(cd->range.baseMipLevel + m);
            d.depth.layer = static_cast<uint16_t>(cd->range.baseArrayLayer + l);
            d.depth.load = vpgpu::LoadAction::Clear;
            d.depth.store = vpgpu::StoreAction::Store, d.depth.keepOrigin = vpgpu::KeepOrigin::Required;
            d.depth.clear.depth = cd->value.depth;
            d.depth.clear.stencil = static_cast<uint8_t>(cd->value.stencil);
            d.label = "vk.clear";
            frame.clear(std::move(d));
            ++st.clears;
          }
      } else if (const auto* ca = std::get_if<OpClearAttachments>(&op)) {
        if (!passOpen) { ++st.dropped; continue; }
        // Quads over the rectangles and layers asked for, of the aspects asked for, inside the
        // open pass (MoltenVK's approach): nothing outside them changes and the pass stays whole.
        auto cr = std::make_shared<vpgpu::ClearRects>();
        // vkCmdClearAttachments' colorAttachment indexes the subpass's colour attachments, which
        // beginPass placed in the same slots.
        for (const VkClearAttachment& a : ca->attachments) {
          if (a.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
            if (a.colorAttachment >= vpgpu::kMaxColorAttachments || !current.colors[a.colorAttachment].texture) continue;
            cr->colorMask |= uint8_t(1u << a.colorAttachment);
            for (int c = 0; c < 4; ++c) cr->colorBits[a.colorAttachment][c] = a.clearValue.color.uint32[c];
          } else if (current.depth.texture) {
            if (a.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) cr->depth = true, cr->depthValue = a.clearValue.depthStencil.depth;
            if (a.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
              cr->stencil = true, cr->stencilValue = static_cast<uint8_t>(a.clearValue.depthStencil.stencil);
          }
        }
        const uint16_t passLayers = std::max<uint16_t>(current.layerCount, 1);
        for (const VkClearRect& r : ca->rects) {
          if (!r.rect.extent.width || !r.rect.extent.height) continue;
          vpgpu::ClearRects::Rect q;
          q.x = uint32_t(std::max(0, r.rect.offset.x));
          q.y = uint32_t(std::max(0, r.rect.offset.y));
          q.width = r.rect.extent.width;
          q.height = r.rect.extent.height;
          // Multiview: the rect's layers are 0/1 and the clear covers every view of the subpass,
          // i.e. the layers of the mask's bits (not the layers between them).
          if (viewMask) {
            for (uint32_t m = viewMask; m; m &= m - 1) {
              q.layer = uint16_t(std::countr_zero(m));
              q.layerCount = 1;
              if (q.layer < passLayers) cr->rects.push_back(q);
            }
            continue;
          }
          q.layer = uint16_t(r.baseArrayLayer);
          q.layerCount = uint16_t(std::max(1u, r.layerCount));
          cr->rects.push_back(q);
        }
        if (eyeRec && cr->depth) eyeRec->cleared = true, eyeRec->clearValue = cr->depthValue;
        if (cr->colorMask || cr->depth || cr->stencil) clearQuads(std::move(cr));
      } else if (const auto* bi = std::get_if<OpCopyBufferToImage>(&op)) {
        linearSync(bi->dst, true);
        if (bi->shrink) shrunkCopy(*bi);
        else if (bi->decodedHost) decodedVolumeCopy(*bi);
        else bufferImage(bi->src, bi->dst, bi->region, true);
        linearSync(bi->dst, false);
      } else if (const auto* ib = std::get_if<OpCopyImageToBuffer>(&op)) {
        linearSync(ib->src, true);
        bufferImage(ib->dst, ib->src, ib->region, false);
      } else if (const auto* ci = std::get_if<OpCopyImage>(&op)) {
        if (!ci->src->id || !ci->dst->id) { ++st.dropped; continue; }
        linearSync(ci->src, true);
        if (ci->dst != ci->src) linearSync(ci->dst, true);
        copyImage(*ci);
        linearSync(ci->dst, false);
      } else if (const auto* bl = std::get_if<OpBlitImage>(&op)) {
        if (!bl->src->id || !bl->dst->id) { ++st.dropped; continue; }
        linearSync(bl->src, true);
        if (bl->dst != bl->src) linearSync(bl->dst, true);
        blitImage(*bl);
        linearSync(bl->dst, false);
      } else if (const auto* rs = std::get_if<OpResolveImage>(&op)) {
        linearSync(rs->dst, true);
        resolveImage(*rs);
        linearSync(rs->dst, false);
      } else if (const auto* d = std::get_if<OpDraw>(&op)) {
        draw(*d, opIx);
      } else if (const auto* x = std::get_if<OpDispatch>(&op)) {
        if (!x->ok) { ++st.skippedDispatches; continue; }
        vpgpu::Dispatch dp = x->dispatch;
        if (x->indirect) {
          const uint8_t* p = indirectData(x->indirect, x->indirectOffset, 12);
          if (!p) { ++st.skippedDispatches; continue; }
          std::memcpy(dp.groups.data(), p, 12);
        }
        if (!dp.indirect && (!dp.groups[0] || !dp.groups[1] || !dp.groups[2])) continue;
        zeroReads(x->zeroReads);
        frame.dispatch(std::move(dp));
        ++st.dispatches;
      } else if (const auto* e = std::get_if<OpExecute>(&op)) {
        if (e->secondary) {
          ++secondaryDepth;
          run(*e->secondary);
          --secondaryDepth;
        }
      } else if (const auto* qr = std::get_if<OpQueryReset>(&op)) {
        queryReset(*qr);
      } else if (const auto* qe = std::get_if<OpQueryEnd>(&op)) {
        queryEnd(*qe);
      } else if (const auto* qc = std::get_if<OpQueryCopy>(&op)) {
        queryCopy(*qc);
      } else if (const auto* ev = std::get_if<OpEvent>(&op)) {
        if (ev->event) st.events.push_back({ev->event, ev->set});
      }
    }
  }

  // vkCmdCopyImage region (or a mip chain request) as one core copy.
  void copyImage(const OpCopyImage& opc) {
    const OpCopyImage* ci = &opc;
    const VkImageCopy& r = ci->region;
    vpgpu::Copy c;
    c.kind = vpgpu::CopyKind::TextureToTexture;
    if (ci->genMips) {
      // gpu/metal2: same texture, next mip, different size = build the mip chain.
      c.srcTexture = {ci->src->id, uint16_t(r.srcSubresource.mipLevel), 0, 1, 0, 0, 0, 2, 2, 1};
      c.dstTexture = {ci->dst->id, uint16_t(r.srcSubresource.mipLevel + 1), 0, 1, 0, 0, 0, 1, 1, 1};
      c.mipChain = ci->dst->desc.mips;
      c.label = "vk.mips";
      frame.copy(std::move(c));
      ++st.copies;
      return;
    }
    c.srcTexture = {ci->src->id, uint16_t(r.srcSubresource.mipLevel), uint16_t(r.srcSubresource.baseArrayLayer),
                    uint16_t(r.srcSubresource.layerCount), uint32_t(r.srcOffset.x), uint32_t(r.srcOffset.y),
                    uint32_t(r.srcOffset.z), r.extent.width, r.extent.height, r.extent.depth};
    // The extent is in source texels; compressed <-> uncompressed copies cover the same blocks,
    // so the destination size is in its own texels (block ratio), clipped to its mip.
    const vpgpu::FormatInfo &sf = vpgpu::formatInfo(ci->src->desc.format), &df = vpgpu::formatInfo(ci->dst->desc.format);
    auto dstSize = [&](uint32_t e, uint32_t full, int32_t at, bool rows) {
      const uint32_t sb = rows ? sf.blockHeight : sf.blockSize, db = rows ? df.blockHeight : df.blockSize;
      const uint32_t v = (e + sb - 1) / sb * db;
      const uint32_t mip = std::max(1u, full >> r.dstSubresource.mipLevel);
      return std::min(v, mip > uint32_t(std::max(0, at)) ? mip - uint32_t(std::max(0, at)) : 1u);
    };
    c.dstTexture = {ci->dst->id, uint16_t(r.dstSubresource.mipLevel), uint16_t(r.dstSubresource.baseArrayLayer),
                    uint16_t(r.dstSubresource.layerCount), uint32_t(r.dstOffset.x), uint32_t(r.dstOffset.y),
                    uint32_t(r.dstOffset.z), dstSize(r.extent.width, ci->dst->desc.width, r.dstOffset.x, false),
                    dstSize(r.extent.height, ci->dst->desc.height, r.dstOffset.y, true), r.extent.depth};
    c.label = "vk.copy_image";
    frame.copy(std::move(c));
    ++st.copies;
  }

  // vkCmdBlitImage region (sampled: scaling, mirroring, format conversion) as one core blit.
  void blitImage(const OpBlitImage& opb) {
    const OpBlitImage* bl = &opb;
    const VkImageBlit& r = bl->region;
    // Rects as origin + positive size; a reversed axis on one side only mirrors the blit.
    auto axis = [](int32_t a, int32_t b, uint32_t& origin, uint32_t& size) {
      origin = uint32_t(std::max(0, std::min(a, b)));
      size = uint32_t(a < b ? b - a : a - b);
      return b < a;
    };
    auto layers = [](const VkImageSubresourceLayers& l, const Image* im) {
      return l.layerCount == VK_REMAINING_ARRAY_LAYERS ? uint32_t(std::max<int>(1, int(im->desc.layers) - int(l.baseArrayLayer)))
                                                       : l.layerCount;
    };
    vpgpu::Copy c;
    c.kind = vpgpu::CopyKind::TextureBlit;
    uint32_t sx, sy, sw, sh, dx, dy, dw, dh;
    const bool sfx = axis(r.srcOffsets[0].x, r.srcOffsets[1].x, sx, sw), sfy = axis(r.srcOffsets[0].y, r.srcOffsets[1].y, sy, sh);
    const bool dfx = axis(r.dstOffsets[0].x, r.dstOffsets[1].x, dx, dw), dfy = axis(r.dstOffsets[0].y, r.dstOffsets[1].y, dy, dh);
    if (!sw || !sh || !dw || !dh) return;
    const uint16_t n = uint16_t(std::min(layers(r.srcSubresource, bl->src), layers(r.dstSubresource, bl->dst)));
    // 3D: the z ranges (one slice per destination depth, sampled at the matching source depth).
    uint32_t sz = 0, sd = 1, dz = 0, dd = 1;
    if (bl->src->desc.type == vpgpu::TextureType::Tex3D) axis(r.srcOffsets[0].z, r.srcOffsets[1].z, sz, sd);
    if (bl->dst->desc.type == vpgpu::TextureType::Tex3D) axis(r.dstOffsets[0].z, r.dstOffsets[1].z, dz, dd);
    sd = std::max(sd, 1u), dd = std::max(dd, 1u);
    c.srcTexture = {bl->src->id, uint16_t(r.srcSubresource.mipLevel), uint16_t(r.srcSubresource.baseArrayLayer), n,
                    sx, sy, sz, sw, sh, sd};
    c.dstTexture = {bl->dst->id, uint16_t(r.dstSubresource.mipLevel), uint16_t(r.dstSubresource.baseArrayLayer), n,
                    dx, dy, dz, dw, dh, dd};
    c.blitFilter = bl->filter == VK_FILTER_NEAREST ? vpgpu::Filter::Nearest : vpgpu::Filter::Linear;
    c.blitFlipX = sfx != dfx;
    c.blitFlipY = sfy != dfy;
    c.blitSwapRB = isSwappedRB(bl->src->info.format) != isSwappedRB(bl->dst->info.format);
    c.label = "vk.blit";
    frame.copy(std::move(c));
    ++st.copies;
  }

  // Eye-pass test for subpass `sub` of a render pass (see taaEyePass).
  void taaSubpass(const OpBeginPass& b, uint32_t sub) {
    taaPass = false;
    eyeScene = false;
    eyeWhy = EyeWhy{};
    if (!eye || !b.pass || !b.framebuffer || sub >= b.pass->subpasses.size()) return;
    const RenderPassObj::Subpass& sp = b.pass->subpasses[sub];
    uint32_t dw = 0, dh = 0;
    bool depth = false;  // a stencil-only attachment does not count
    if (sp.hasDepth && sp.depth.attachment != VK_ATTACHMENT_UNUSED && sp.depth.attachment < b.framebuffer->views.size()) {
      const ImageView* v = b.framebuffer->views[sp.depth.attachment];
      if (v && v->image && vpgpu::formatInfo(v->image->desc.format).depth()) {
        depth = true;
        dw = std::max(1u, v->image->desc.width >> v->range.baseMipLevel);
        dh = std::max(1u, v->image->desc.height >> v->range.baseMipLevel);
      }
    }
    eyeScene = taaEyePass(*eye, depth, sp.viewMask, b.area, b.framebuffer->width, b.framebuffer->height, dw, dh, taaW, taaH, markedEye,
                          &eyeWhy);
    taaPass = taa && eyeScene;
  }

  Device* devForce() const { return const_cast<Device*>(anyDevice()); }  // (the device is not const: ZeroLog only promises not to change it)

  // The body of the "vk.msaa rej_detail" / "vk.msaa rej_first" lines: reason and what taaEyePass judged the pass by (eyeWhy).
  static std::string rejDetailLine(const char* kind, const char* reason, const EyeWhy& e, const char* colorFmt, const char* depthFmt) {
    char b[400];
    std::snprintf(b, sizeof b, "vk.msaa %s reason=%s w=%u h=%u fbW=%u fbH=%u depthW=%u depthH=%u viewMask=0x%x eyeW=%u eyeH=%u jit=%d fmt=%s/%s",
                  kind, reason, e.w, e.h, e.fbW, e.fbH, e.depthW, e.depthH, e.viewMask, e.eyeW, e.eyeH, e.jit ? 1 : 0, colorFmt, depthFmt);
    return b;
  }

  // "vk.msaa rej_detail reason=.. w= h= ...": the 'eye' rejection with what taaEyePass judged the pass by (eyeWhy). At most 3 lines
  // per session: the first rejected pass, then only passes of a larger area (the eye's scene is among the biggest). Decision unchanged.
  void eyeRejDetail(Device* dev, const vpgpu::RenderPassDesc& d) const {
    static std::atomic<unsigned> logged{0};
    static std::atomic<uint64_t> best{0};
    const unsigned n = logged.load(std::memory_order_relaxed);
    const uint64_t area = uint64_t(eyeWhy.w) * eyeWhy.h;
    if (n >= 3 || (n && area <= best.load(std::memory_order_relaxed))) return;
    best.store(area, std::memory_order_relaxed);
    logged.fetch_add(1, std::memory_order_relaxed);
    vpgpu::TextureDesc gc, gz;
    const bool hc = d.colors[0].texture && dev->gpu->textureDesc(d.colors[0].texture, gc);
    const bool hz = d.depth.texture && dev->gpu->textureDesc(d.depth.texture, gz);
    static const char* const kWhy[] = {"no_depth", "no_eye_size", "size", "other"};
    log(rejDetailLine("rej_detail", kWhy[eyeWhy.why], eyeWhy, hc ? vpgpu::formatInfo(gc.format).name : "none",
                      hz ? vpgpu::formatInfo(gz.format).name : "none"));
  }

  // Does a draw of the pass about to open (ops after `at` up to its end, secondary buffers included) keep the pass at 1x?
  // 0 = none does, else the reason: 1 input attachment / pixel-local read, 2 dual-source blending, 3 extended dynamic state
  // variant, 4 a draw without its pipeline object (no sample variant can be made), 5 (only with `up`, a game MSAA pass about to be
  // upgraded) a pipeline with a baked partial sample mask or a fragment shader using SampleId / SampleMask: both depend on the
  // game's sample count. Returns true when the pass ended inside `ops`.
  // Draws from ops[from] to the end of the pass (same extent as forceScan).
  static constexpr uint32_t kSceneMinDraws = 32;  // draws of the main 3D scene pass in an intermediate target (forceMsaa sceneMain)
  static uint32_t countDraws(const std::vector<Op>& ops, size_t from) {
    uint32_t n = 0;
    for (size_t i = from; i < ops.size(); ++i) {
      const Op& op = ops[i];
      if (const auto* d = std::get_if<OpDraw>(&op)) n += d->ok ? 1u : 0u;
      else if (const auto* e = std::get_if<OpExecute>(&op)) {
        if (e->secondary) n += countDraws(e->secondary->ops, 0);
      } else if (std::holds_alternative<OpEndPass>(op) || std::holds_alternative<OpNextSubpass>(op) ||
                 std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op))
        break;
    }
    return n;
  }
  bool forceScan(const std::vector<Op>& ops, size_t from, int& why, bool up = false) {
    auto note = [&](int w) { why = why && why < w ? why : w; };  // the lowest number wins
    for (size_t i = from; i < ops.size(); ++i) {
      const Op& op = ops[i];
      if (const auto* d = std::get_if<OpDraw>(&op)) {
        if (!d->ok) continue;
        if (!d->pipe) note(4);
        else if (d->pipe->dualSource) note(2);
        else if (up && (d->pipe->sampleMask != ~0u || d->pipe->sampleBuiltin)) note(5);
        if (d->edsVariant) note(3);
        for (const vpgpu::Access& a : d->draw.bindings)
          if (a.pixelLocal) note(1);
      } else if (const auto* e = std::get_if<OpExecute>(&op)) {
        if (e->secondary && forceScan(e->secondary->ops, 0, why, up)) return true;
      } else if (std::holds_alternative<OpEndPass>(op) || std::holds_alternative<OpNextSubpass>(op) ||
                 std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op)) {
        return true;
      }
    }
    return false;
  }

  // The game's own MSAA (VPVK_GAME_MSAA_AUTO, default on; =0 off): a colour 0 / depth attachment of a multisampled image the
  // game can never read the samples of (TextureDesc::samplesReadable) and that the pass resolves is handled like a forced-MSAA
  // twin: Attachment::autoMsaa, so its samples stay on tile whatever the geometry, the store is only the Resolve, and a later
  // Load of it (another encoder or submit) is seeded from the resolve target. Left alone: any other usage, no resolve target
  // (except a depth the game discards, below), a partial render area, input attachments / later subpasses, any Load (an attachment
  // whose load is a Load continues samples an earlier pass stored: that includes a stencil aspect that loads, which makes the whole
  // attachment's load a Load) except of a transient image of vpgles (its Load is a seed), and the depth VPKTAA reads after the frame (it keeps its samples). The seed restores depth and
  // stencil, so a depth/stencil attachment is served only when its stencil survives a cut: it resolves stencil (stencilResolve
  // not None) into a target created with format views (the stencil plane is read through a stencil view).
  // A depth the game never resolves and discards (store DontCare, the usual MSAA depth) gets a 1x partner of ours
  // (MsaaTwins::acquire, samples 1) as its resolve target: nothing resolves into it unless the frame compiler cuts a heavy
  // encoder (splitHeavyAuto: every chunk resolves, the next seeds from it), so the colour of a heavy pass stays memoryless
  // without the depth samples ever being stored. Called with forceMsaa, before uiPass.
  void gameAuto(vpgpu::RenderPassDesc& d, const VkRect2D& area, bool localRead, bool laterOrMultiSubpass) {
    Device* dev = devForce();
    if (!gameMsaaAuto() || !dev || !dev->gpu || localRead || laterOrMultiSubpass || d.rateMap) return;
    bool alone = false;  // set once colour 0 is autoMsaa and no other colour is attached
    auto eligible = [&](vpgpu::Attachment& a, bool depth) {
      if (!a.texture || a.autoMsaa) return;
      // No resolve target: colour is left alone; a depth the game discards gets the 1x partner below.
      // (Only beside an autoMsaa colour 0 and no other colour: a memoryless depth next to samples that need memory would meet the
      // tiler's flush on a heavy pass, where a depth with memory used to survive it.)
      const bool ours = depth && !a.resolve && a.store == vpgpu::StoreAction::DontCare && alone;
      if (!ours && (!a.resolve || (a.store != vpgpu::StoreAction::Resolve && a.store != vpgpu::StoreAction::StoreAndResolve))) return;
      vpgpu::TextureDesc td;
      if (!dev->gpu->textureDesc(a.texture, td) || td.samples < 2 || td.samplesReadable || td.type != vpgpu::TextureType::Tex2D) return;
      // (A Load continues samples an earlier pass stored, except of a transient image: its samples never outlive a pass, so the
      // Load is a seed from the resolve target: the multisample twins of the GLES layer.)
      // (Only vpgles' own device: a transient image of a Vulkan guest keeps the rule above.)
      if (a.load == vpgpu::LoadAction::Load && !(td.transient && dev->internalClient)) return;
      const uint32_t w = std::max(1u, td.width >> a.mip), h = std::max(1u, td.height >> a.mip);
      if (area.offset.x != 0 || area.offset.y != 0 || area.extent.width < w || area.extent.height < h) return;
      if (depth) {
        if (taaPass && !taa->still) return;
      }
      const bool stencil = depth && vpgpu::formatInfo(td.format).stencil();
      if (!ours) {
        // The seed of a later Load reads the resolve target in a shader: it must be readable (entry.cpp adds no usage for this).
        vpgpu::TextureDesc rd;
        if (!dev->gpu->textureDesc(a.resolve, rd) || !(rd.usage & vpgpu::usage::sampled)) {
          ++dev->msaaTwins.skipUnreadable;
          return;
        }
        // ... and its stencil plane, when there is one, through a stencil view (format views) of a resolved stencil.
        if (stencil && (a.stencilResolve == vpgpu::ResolveMode::None || !(rd.usage & vpgpu::usage::formatView))) return;
      } else {
        const vpgpu::TextureId partner = dev->msaaTwins.acquire(dev, a.texture, td, 1, std::max<uint16_t>(d.layerCount, 1), a.mip, a.layer);
        if (!partner) return;
        a.resolve = partner, a.resolveMip = 0, a.resolveLayer = 0;
        a.depthResolve = vpgpu::ResolveMode::Sample0;
        a.stencilResolve = stencil ? vpgpu::ResolveMode::Sample0 : vpgpu::ResolveMode::None;
      }
      a.autoMsaa = true, a.needsSeed = false, a.keepSamples = false;
      a.store = ours ? vpgpu::StoreAction::DontCare : vpgpu::StoreAction::Resolve;
      ++dev->msaaTwins.gameAuto;
    };
    eligible(d.colors[0], false);
    alone = d.colors[0].autoMsaa;
    for (int i = 1; i < vpgpu::kMaxColorAttachments; ++i) alone = alone && !d.colors[i].texture;
    eligible(d.depth, true);
  }

  // Forced MSAA (msaaForce, VPVK_MSAA_FORCE): an eye / scene pass the game renders at 1x (colour 0 + depth, taaEyePass) draws
  // into N-sample twins instead (MsaaTwins) and resolves them into the game's images at its end. The twins are Attachment::autoMsaa:
  // the frame compiler keeps their samples on tile for good (memoryless, never stored; a pass continuing the picture in a new
  // encoder is seeded from the 1x), so MSAA costs shading and resolve but no memory. Passes it cannot serve stay at 1x and are
  // counted (MsaaTwins::skip*): a partial render area (the resolve would overwrite the rest), stencil that must be loaded from an
  // image without format views (the seed cannot restore it), input attachments / local reads, dual-source blending, extended dynamic state draws, integer or
  // reinterpreted formats, other attachments than colour 0 + depth, game MSAA, targets the twins cannot match. A game MSAA pass
  // of s samples below the forced N is upgraded (below): its twins are N samples resolving into the game's own 1x targets.
  // Called before uiPass
  // (the UI mask follows the pass's samples) with the description of the pass about to open.
  void forceMsaa(vpgpu::RenderPassDesc& d, const std::vector<Op>& ops, size_t at, const VkRect2D& area, bool localRead,
                 bool laterOrMultiSubpass) {
    forceN = 0;
    const int n = msaaForce();
    Device* dev = devForce();
    if (!dev || !dev->gpu) return;
    MsaaTwins& mt = dev->msaaTwins;
    if (n < 2) return void(++mt.rejN);
    mt.lastWhy = nullptr;
    struct MainNote {  // "vk.msaa main": how this call ends, whichever return it takes
      MsaaTwins& mt;
      Device* dev;
      vpgpu::TextureId game;
      const VkRect2D& area;
      const bool& eye;
      const uint8_t& forced;
      ~MainNote() { mt.noteMain(dev, game, area, eye, forced); }
    } mainNote{mt, dev, d.colors[0].texture, area, eyeScene, forceN};
    // (Every rejection from here on is counted by reason and logged by MsaaTwins::noteRej: "vk.msaa rej".)
    auto rej = [&](uint32_t& counter, const char* why) {
      ++counter;
      mt.noteRej(dev, d, why);
    };
    vpgpu::Attachment& c = d.colors[0];
    vpgpu::Attachment& z = d.depth;
    // A pass that is not at the eye's size can still be the game's main 3D scene, drawn into an intermediate target that is later
    // copied / composited into the eye: colour + depth of at least half the eye's pixels, one 2D level, at most two layers, and
    // kSceneMinDraws draws (a UI / post-process pass has a few). -1: it is big enough but not that (counted "scene"), 0: not a candidate.
    auto sceneMain = [&]() {
      vpgpu::TextureDesc tc;
      if (!c.texture || !z.texture || !eyeWhy.eyeW || !eyeWhy.eyeH || !dev->gpu->textureDesc(c.texture, tc)) return 0;
      if (tc.type != vpgpu::TextureType::Tex2D || tc.mips != 1 || tc.layers > 2) return 0;
      if (uint64_t(tc.width) * tc.height * 2 < uint64_t(eyeWhy.eyeW) * eyeWhy.eyeH) return 0;
      return countDraws(ops, at + 1) >= kSceneMinDraws ? 1 : -1;
    };
    const int scene = eyeScene ? 1 : sceneMain();
    if (scene < 1) {
      ++mt.rejEyeBy[eyeWhy.why];
      mt.eye.noteNonEye(dev->gpu->submittedFrames());
      eyeRejDetail(dev, d);
      return scene ? rej(mt.rejScene, "scene") : rej(mt.rejEye, "eye");
    }
    if (!c.texture || !z.texture) return rej(mt.rejCd, "cd");
    vpgpu::TextureDesc gc, gz;
    if (!dev->gpu->textureDesc(c.texture, gc) || !dev->gpu->textureDesc(z.texture, gz)) return rej(mt.rejDesc, "desc");
    struct EyeNote {  // "vk.msaa_eye": this eye pass with the samples it ends at (forceN), whichever return follows
      MsaaTwins& mt;
      Device* dev;
      const vpgpu::TextureDesc& gc;
      const uint8_t& forced;
      uint32_t draws;
      bool on;  // an eye pass only (not a scene pass in an intermediate target)
      ~EyeNote() {
        if (on) mt.eye.notePass(dev->gpu->submittedFrames(), gc.width, gc.height, uint16_t(gc.layers), vpgpu::formatInfo(gc.format).name,
                        uint8_t(gc.samples), forced ? forced : uint8_t(gc.samples), draws);
      }
    } eyeNote{mt, dev, gc, forceN, mt.eye.lines >= MsaaTwins::kEyeLines ? 0u : countDraws(ops, at + 1), eyeScene};  // lines used up: nothing prints, skip the count
    // "vk.msaa rej_first reason=smp|msstore|load ...": the first pass rejected for each of those reasons (once per device), with the
    // rej_detail fields and the game's samples / store / load / resolve of the pass.
    auto rejFirst = [&](int slot, const char* why) {
      if (mt.rejFirstDone[slot]) return;
      mt.rejFirstDone[slot] = true;
      char x[160];
      std::snprintf(x, sizeof x, " samples=%u/%u forced=%d store=%d/%d load=%d/%d resolve=%d/%d", unsigned(gc.samples), unsigned(gz.samples), n,
                    int(c.store), int(z.store), int(c.load), int(z.load), int(c.resolve ? 1 : 0), int(z.resolve ? 1 : 0));
      log(rejDetailLine("rej_first", why, eyeWhy, vpgpu::formatInfo(gc.format).name, vpgpu::formatInfo(gz.format).name) + x);
    };
    // The game's own MSAA of s samples (colour and depth alike) below the forced N: the pass is upgraded (renders into N-sample
    // twins that resolve into what the game resolves into), but only when the game never keeps its s samples: its colour is a
    // memoryless Resolve into a 1x target, its depth a Resolve or a DONT_CARE, nothing loads. Anything else is the game's own MSAA.
    const bool taaDepth = taaPass && !taa->still;  // the depth VPKTAA reads after the frame must reach the game's image
    const bool up = n > int(gc.samples) && gc.samples > 1 && gc.samples == gz.samples;
    if (!up && (gc.samples != 1 || gz.samples != 1)) {  // the game's own MSAA: not ours
      rejFirst(0, "smp");
      return rej(mt.rejSmp, "smp");
    }
    if (up) {
      const bool cOk = c.resolve && c.store == vpgpu::StoreAction::Resolve && c.load != vpgpu::LoadAction::Load;
      const bool zOk = z.load != vpgpu::LoadAction::Load && ((z.resolve && z.store == vpgpu::StoreAction::Resolve) ||
                                                              (!z.resolve && z.store == vpgpu::StoreAction::DontCare && !taaDepth));
      if (!cOk || !zOk) {
        rejFirst(1, "msstore");
        return rej(mt.rejStore, "msstore");
      }
    } else if (c.resolve || z.resolve) {
      return rej(mt.rejRes, "res");
    }
    bool shape = d.rateMap || laterOrMultiSubpass;
    for (int i = 1; i < vpgpu::kMaxColorAttachments; ++i) shape = shape || d.colors[i].texture;
    if (shape) return rej(mt.skipShape, "shape");
    if (localRead) return rej(mt.skipInput, "input");
    const vpgpu::FormatInfo &fc = vpgpu::formatInfo(gc.format), &fz = vpgpu::formatInfo(gz.format);
    if (fc.integer() || fc.depth() || !fz.depth() || c.format != vpgpu::Format::Undefined || gc.type != vpgpu::TextureType::Tex2D ||
        gz.type != vpgpu::TextureType::Tex2D)
      return rej(mt.skipFmt, "fmt");
    // (A stencil aspect that loads makes the attachment's load a Load: attach / the dynamic rendering merge in pipelines.cpp.)
    // The twin's Load is a seed from the game's image: depth plane as is, stencil plane through a stencil view (format views;
    // entry.cpp imageDesc gives them to a 1x depth/stencil attachment under forced MSAA). Without them the stencil is lost.
    if (fz.stencil() && z.load == vpgpu::LoadAction::Load &&
        !((gz.usage & vpgpu::usage::sampled) && (gz.usage & vpgpu::usage::formatView))) {
      rejFirst(2, "load");
      return rej(mt.skipLoad, "load");
    }
    const uint32_t w = std::max(1u, gc.width >> c.mip), h = std::max(1u, gc.height >> c.mip);
    const uint16_t layers = std::max<uint16_t>(d.layerCount, 1);
    if (std::max(1u, gz.width >> z.mip) != w || std::max(1u, gz.height >> z.mip) != h || gc.layers < c.layer + layers ||
        gz.layers < z.layer + layers)
      return rej(mt.skipFmt, "fmt_size");
    if (area.offset.x != 0 || area.offset.y != 0 || area.extent.width < w || area.extent.height < h) return rej(mt.skipArea, "area");
    if (up) {
      // The game's resolve targets must be readable, as in gameAuto: a later Load / a heavy-encoder cut seeds the twin from them
      // (a stencil plane through a stencil view: formatView).
      auto unreadable = [&](const vpgpu::Attachment& a, bool stencil) {
        vpgpu::TextureDesc rd;
        return a.resolve && (!dev->gpu->textureDesc(a.resolve, rd) || !(rd.usage & vpgpu::usage::sampled) ||
                             (stencil && !(rd.usage & vpgpu::usage::formatView)));
      };
      if (unreadable(c, false) || unreadable(z, fz.stencil())) {
        ++mt.skipUnreadable;
        return mt.noteRej(dev, d, "unreadable");
      }
    }
    int why = 0;
    forceScan(ops, at + 1, why, up);
    if (why) {
      if (why == 5) return rej(mt.rejSmask, "smask");
      return rej(why == 1 ? mt.skipInput : why == 2 ? mt.skipDsb : why == 3 ? mt.skipEds : mt.skipShape,
                 why == 1 ? "scan_input" : why == 2 ? "dsb" : why == 3 ? "eds" : "scan_shape");
    }
    const uint8_t samples = uint8_t(n);
    const vpgpu::TextureId tc = mt.acquire(dev, c.texture, gc, samples, layers, c.mip, c.layer);
    const vpgpu::TextureId tz = tc ? mt.acquire(dev, z.texture, gz, samples, layers, z.mip, z.layer) : vpgpu::TextureId{};
    // (An upgraded depth the game does not resolve gets a 1x partner of ours as resolve target, as in gameAuto.)
    const vpgpu::TextureId partner = up && !z.resolve ? mt.acquire(dev, z.texture, gz, 1, layers, z.mip, z.layer) : vpgpu::TextureId{};
    if (!tc || !tz || (up && !z.resolve && !partner)) return rej(mt.skipSamples, "samples");
    if (up) {
      // The game's resolve targets and modes (stencilResolve None included: Metal drops that aspect's resolve, and a heavy cut promotes it) stay as the game set them; only the sampled image is swapped for our twin.
      auto swap = [](vpgpu::Attachment& a, vpgpu::TextureId t) {
        a.texture = t, a.mip = 0, a.layer = 0;
        a.autoMsaa = true, a.needsSeed = false, a.keepSamples = false;
        a.store = vpgpu::StoreAction::Resolve;
      };
      swap(c, tc);
      swap(z, tz);
      if (!z.resolve) {
        z.resolve = partner, z.resolveMip = 0, z.resolveLayer = 0;
        z.depthResolve = vpgpu::ResolveMode::Sample0;
        z.store = vpgpu::StoreAction::DontCare;
      }
      forceN = samples;
      ++mt.passes, ++mt.up;
      mt.sceneUp += eyeScene ? 0u : 1u;
      if (!mt.upDetailDone) {  // "vk.msaa up_detail": the first upgraded pass (once per device)
        mt.upDetailDone = true;
        vpgpu::TextureDesc rc, rz;
        const bool hrc = c.resolve && dev->gpu->textureDesc(c.resolve, rc), hrz = z.resolve && dev->gpu->textureDesc(z.resolve, rz);
        logf("vk.msaa up_detail w=%u h=%u layers=%u color=%s depth=%s resolve=%s/%s samples=%u->%u multiview=%d viewMask=0x%x", w, h,
             unsigned(layers), vpgpu::formatInfo(gc.format).name, vpgpu::formatInfo(gz.format).name,
             hrc ? vpgpu::formatInfo(rc.format).name : "none", hrz ? vpgpu::formatInfo(rz.format).name : "none", unsigned(gc.samples),
             unsigned(samples), std::popcount(viewMask) >= 2 ? 1 : 0, viewMask);
      }
      return;
    }
    auto twin = [](vpgpu::Attachment& a, vpgpu::TextureId t, bool resolves) {
      const vpgpu::TextureId guest = a.texture;
      const uint16_t guestMip = a.mip, guestLayer = a.layer;
      a.texture = t, a.mip = 0, a.layer = 0;
      a.resolve = guest, a.resolveMip = guestMip, a.resolveLayer = guestLayer;  // (also the seed source of a Load)
      a.autoMsaa = true, a.needsSeed = false, a.keepSamples = false;
      a.store = resolves ? vpgpu::StoreAction::Resolve : vpgpu::StoreAction::DontCare;
    };
    const vpgpu::TextureId guestDepth = z.texture;
    twin(c, tc, c.store != vpgpu::StoreAction::DontCare);
    twin(z, tz, z.store != vpgpu::StoreAction::DontCare || taaDepth);
    z.depthResolve = vpgpu::ResolveMode::Sample0;
    z.stencilResolve = fz.stencil() ? vpgpu::ResolveMode::Sample0 : vpgpu::ResolveMode::None;
    if (taaDepth) dev->gpu->markTransientViolated(guestDepth);  // a transient game depth would be dropped with its tile
    forceN = samples;
    ++mt.passes;
    mt.sceneUp += eyeScene ? 0u : 1u;
  }

  // Moves the viewport of an eye-pass geometry draw by the jitter; the scissor stays (it bounds
  // pixels, not geometry). A fullscreen pass (no index buffer, at most 6 vertices, no vertex
  // buffer: a quad/triangle made from the vertex id) is left alone: its texels must not move.
  static bool fullscreenDraw(const vpgpu::Draw& dr, const OpDraw& d) {
    return (!dr.index || d.fanConverted) && dr.count <= 6 && !d.vertexBuffers;
  }
  static taa::DrawFacts factsOf(const OpDraw& d) {
    taa::DrawFacts f;
    f.depthTest = d.depthTested, f.depthWrite = d.depthWrite, f.blended = d.blended;
    f.fullscreen = fullscreenDraw(d.draw, d);
    return f;
  }
  // The class of a draw of the open eye pass: classified in order by the pre-scan (uiPass), or now.
  taa::DrawClass classOf(const OpDraw& d) {
    auto it = classes.find(&d);
    if (it != classes.end()) return it->second;
    return classes.emplace(&d, classifier().classify(factsOf(d))).first->second;
  }

  // Everything an eye-pass geometry draw needs: its class (counted), the mask variant of its
  // pipeline in a masked pass, and the jitter (viewport shift) unless it is UI.
  // "vk.depth_diag": the open eye pass's depth census (DepthDiag::Pass), folded into the device's summary when the pass ends.
  DepthDiag::Pass dd;
  void ddFlush() {
    if (!dd.any()) return;
    if (Device* dev = devForce()) dev->depthDiag.notePass(dd);
    dd = {};
  }
  void jitter(vpgpu::Draw& dr, const OpDraw& d) {
    if (eyeScene) noteDepthDraw(dd, dr, d);  // (eye passes also without VPKTAA: the census does not depend on it)
    if (!taaPass) return;
    if (eyeRec && d.depthCmp) ++(d.depthCmp > 0 ? eyeRec->greater : eyeRec->less);
    taa::DrawClass cls = taa::DrawClass::Scene;
    if (uiOn) {
      cls = classOf(d);
      ++(cls == taa::DrawClass::Ui ? st.ui.ui : cls == taa::DrawClass::Scene ? st.ui.scene : st.ui.other);
    }
    // EDS overrides live in the draw's own key and the mask variants are built from the static state.
    // EDS is only exposed behind VPVK_EXTRA_EXTENSIONS (CTS), where the native TAA is forced off, so the
    // two never meet in practice; if they do, the draw keeps its EDS key (Metal may then reject the
    // attachment mismatch with slot 7 - logged by the backend).
    if (maskPass && um && d.pipe && !d.edsVariant) {
      vpgpu::PipelineKey k;
      if (maskPipelineKey(um->dev, *d.pipe, d.texelVariant, forceN ? forceN : d.passSamples, cls == taa::DrawClass::Ui, k)) dr.pipeline = k;
      else if (passDump().on) ++passNote().fallbackMask;  // no mask variant: the draw keeps its plain pipeline
    }
    if (cls == taa::DrawClass::Ui) {  // 2D drawn into the eye buffer: no jitter, the mask marks it
      ++st.taaSkipped;
      return;
    }
    const bool fullscreen = fullscreenDraw(dr, d);
    // width 0 = "no viewport set": nothing to move (Vulkan always sets one; never seen).
    if (!d.depthTested || fullscreen || dr.viewport.width <= 0) {
      ++st.taaSkipped;
      return;
    }
    dr.viewport.x += taa->jx;
    dr.viewport.y += taa->jy;  // also right for a negative height: the whole mapping moves by jy
    ++st.taaJittered;
    st.taaTargetW = std::max(st.taaTargetW, taaW);
    st.taaTargetH = std::max(st.taaTargetH, taaH);
  }

  uint32_t scanDraws = 0;  // draws scanPass counted (uiPass resets it)
  int keepHeavyMask = -1;  // VPVK_UIMASK_HEAVY=1 (read once per translate)

  // Pre-scan of the eye pass that is about to open (ops after `at` up to its end): classifies its
  // draws in order and tells whether any is UI and which sample count the draws render at.
  // Returns true when the pass ended inside `ops` (false: it goes on, e.g. after a secondary buffer).
  bool scanPass(const std::vector<Op>& ops, size_t from, bool& hasUi, uint8_t& samples, bool& dualSource) {
    for (size_t i = from; i < ops.size(); ++i) {
      const Op& op = ops[i];
      if (const auto* d = std::get_if<OpDraw>(&op)) {
        if (!d->ok) continue;
        ++scanDraws;
        const taa::DrawClass c = classifier().classify(factsOf(*d));
        classes[d] = c;
        hasUi = hasUi || c == taa::DrawClass::Ui;
        samples = std::max(samples, d->passSamples);
        dualSource = dualSource || (d->pipe && d->pipe->dualSource);
      } else if (const auto* e = std::get_if<OpExecute>(&op)) {
        if (e->secondary && scanPass(e->secondary->ops, 0, hasUi, samples, dualSource)) return true;
      } else if (const auto* ca = std::get_if<OpClearAttachments>(&op)) {
        // A depth clear inside the pass (vpgles always loads the depth and clears with quads, like a GL
        // glClear) starts a new scene exactly as a Clear load action does.
        for (const VkClearAttachment& a : ca->attachments)
          if (a.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) {
            classifier().beginPass(false);
            break;
          }
      } else if (std::holds_alternative<OpEndPass>(op) || std::holds_alternative<OpNextSubpass>(op) ||
                 std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op)) {
        return true;
      }
    }
    return false;
  }

  // Called with the description of an eye pass before it opens: classifies its draws and, when some
  // are UI (and slot kUiMaskSlot is free), adds the R8 mask as that colour attachment, cleared to 0
  // and stored for the TAA pass (MSAA pass: a multisample twin resolved into the 1x mask).
  void uiPass(const std::vector<Op>& ops, size_t at, vpgpu::RenderPassDesc& desc) {
    maskPass = false;
    classes.clear();
    if (!taaPass || !uiOn) return;
    classifier().beginPass(desc.depth.load == vpgpu::LoadAction::Load);
    bool hasUi = false, dualSource = false;
    uint8_t samples = 1;
    scanDraws = 0;
    scanPass(ops, at + 1, hasUi, samples, dualSource);
    if (forceN) samples = std::max(samples, forceN);  // the mask twin has the pass's samples (the draws' pipelines are N-sample variants)
    ++st.ui.passes;
    if (!hasUi || !um || !um->dev) return;
    // Dual-source blending allows a single render target in Metal (the pipeline of a draw with the mask added
    // would be rejected): a pass with such a draw keeps the game's attachments and has no mask.
    if (dualSource) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 6) logf("vk.taa uimask skipped_pass why=dual_source samples=%u", unsigned(samples));
      return;
    }
    constexpr int slot = taa::kUiMaskSlot;
    if (desc.colors[slot].texture || !desc.colors[0].texture || taaW == 0 || taaH == 0) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 6)
        logf("vk.taa uimask skipped_pass why=no_free_slot_or_color samples=%u", unsigned(samples));
      return;
    }
    // A shader warm-up pass (more draws than the frame compiler's heavy limit, CompileOptions::tileMaxDraws) is HeavyGeometry whatever
    // we do, and is not shown: the mask's multisample twin would only get memory there, and as an attachment of its own it stops the
    // compiler cutting the game's forced-MSAA twins into tile-sized chunks (splitHeavyAuto). No mask for it (the UI of that pass is
    // sharpened like the rest). VPVK_UIMASK_HEAVY=1 keeps the mask there.
    // (The limit is the Device's own, VPVK_TILE_MAX_DRAWS. A pass near it can flip between masked and not from frame to frame: the
    // sharpening of its UI then flickers; the log names the limit.)
    const Device* mdev = anyDevice();
    const uint32_t maskLimit = mdev && mdev->gpu ? mdev->gpu->tileMaxDraws() : vpgpu::CompileOptions{}.tileMaxDraws;
    if (samples > 1 && scanDraws > maskLimit) {
      if (keepHeavyMask < 0) {
        const char* keep = getenv("VPVK_UIMASK_HEAVY");
        keepHeavyMask = keep && keep[0] == '1' && !keep[1] ? 1 : 0;
      }
      if (!keepHeavyMask) {
        static std::atomic<unsigned> n{0};
        if (n.fetch_add(1) < 6)
          logf("vk.taa uimask skipped_pass why=heavy_pass draws=%u limit=%u samples=%u (VPVK_UIMASK_HEAVY=1 keeps the mask)", scanDraws,
               maskLimit, unsigned(samples));
        return;
      }
    }
    // The game's own MSAA pass that continues the pass before it (same colour 0 and depth, depth Loaded: a later subpass,
    // or the second part of a split pass). The mask would be an attachment the pass it continues does not have, so the frame compiler could
    // not merge the two and every sample would go through memory between them (store + load of 4x colour and depth:
    // Into the Radius 2, 2784x2260x2, ~900 MiB per frame). Such a pass keeps the game's attachments and has no mask
    // (RCAS then also sharpens its UI/translucent pixels). Forced MSAA twins are seeded, not loaded: unaffected.
    if (!forceN && samples > 1 && desc.depth.texture && desc.depth.load == vpgpu::LoadAction::Load &&
        desc.colors[0].texture == lastColor0 && desc.depth.texture == lastDepth) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 6) logf("vk.taa uimask skipped_pass why=msaa_continuation samples=%u", unsigned(samples));
      return;
    }
    const uint32_t slotIx = um->maskedPasses++ & 1u;
    const vpgpu::TextureId tex = um->acquire(slotIx, taaW, taaH, std::max<uint16_t>(desc.layerCount, 1), samples);
    if (!tex) return;
    vpgpu::Attachment& a = desc.colors[slot];
    a.clear.color = {0, 0, 0, 0};
    a.load = vpgpu::LoadAction::Clear;
    if (samples > 1) {  // MSAA pass: draw into the multisample twin (same sample count as the draws' pipelines), average-resolve into the 1x mask
      a.texture = um->target[slotIx].ms, a.mip = 0, a.layer = 0;
      a.store = vpgpu::StoreAction::Resolve;
      a.resolve = tex, a.resolveMip = 0, a.resolveLayer = 0;
    } else {
      a.texture = tex, a.mip = 0, a.layer = 0;
      a.store = vpgpu::StoreAction::Store;
    }
    um->live[slotIx].valid = true;
    um->live[slotIx].t = um->target[slotIx];
    maskPass = true;
    ++st.ui.maskedPasses;
  }

  static const uint8_t* indirectData(const Buffer* b, VkDeviceSize offset, VkDeviceSize size) {
    if (!b || !b->memory || !b->memory->host || offset + size > b->size) return nullptr;
    return b->memory->host + b->offset + offset;
  }

  // A query ends: in a multiview pass it is viewCount consecutive queries (Vulkan), the first
  // holding the count of every view and the rest 0 (they were reset). Timestamps: the host time
  // at submission (monotonic, in timestampPeriod ticks), written now into the counter.
  void queryEnd(const OpQueryEnd& q) {
    QueryPoolObj* p = q.pool;
    if (!p || q.query >= p->count) return;
    uint32_t n = 1;
    if (passOpen && viewMask) n = std::min<uint32_t>(uint32_t(std::popcount(viewMask)), p->count - q.query);
    if (q.timestamp && p->base != UINT32_MAX && p->device && p->device->visibility.host) {
      static std::atomic<uint64_t> last{0};
      // In ticks of limits.timestampPeriod (nanoseconds / period): the Quest identity advertises Adreno's 52.08 ns tick.
      uint64_t now = uint64_t(double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch()).count()) /
                              double(deviceProperties().limits.timestampPeriod));
      // Strictly increasing and never 0, even for two writes within one clock tick.
      uint64_t prev = last.load();
      while (!last.compare_exchange_weak(prev, std::max(now, prev + 1))) {}
      now = std::max(now, prev + 1) & ((uint64_t{1} << kTimestampValidBits) - 1);
      for (uint32_t i = 0; i < n; ++i)
        std::memcpy(p->device->visibility.host + size_t(p->base + q.query + i) * 8, &now, 8);
    }
    st.queries.push_back({p, q.query, n, true});
  }
  // Occlusion counters back to 0 (GPU fill, ordered by the core like any copy); timestamps are
  // host-written, so theirs are cleared here, in recording order with the writes above.
  void queryReset(const OpQueryReset& q) {
    QueryPoolObj* p = q.pool;
    if (!p || p->base == UINT32_MAX || !p->device || !p->device->visibility.id || !q.count) return;
    if (p->type == VK_QUERY_TYPE_TIMESTAMP) {
      if (p->device->visibility.host)
        std::memset(p->device->visibility.host + size_t(p->base + q.first) * 8, 0, size_t(q.count) * 8);
      st.queries.push_back({p, q.first, q.count, false});
      return;
    }
    vpgpu::Copy c;
    c.kind = vpgpu::CopyKind::FillBuffer;
    c.dstBuffer = {p->device->visibility.id, uint64_t(p->base + q.first) * 8, uint64_t(q.count) * 8};
    c.fillByte = 0;
    c.label = "vk.query_reset";
    frame.copy(std::move(c));
    st.queries.push_back({p, q.first, q.count, false});
  }
  // Is query q available when this point of the submission executes? Ended earlier in this
  // submission (latest event wins), or by an earlier one and not reset since.
  bool queryEnded(QueryPoolObj* p, uint32_t q) {
    for (size_t i = st.queries.size(); i-- > 0;) {
      const TranslateStats::QueryEvent& e = st.queries[i];
      if (e.pool == p && q >= e.first && q - e.first < e.count) return e.end;
    }
    std::lock_guard<std::mutex> l(p->m);
    return q < p->frame.size() && p->frame[q] != 0;
  }
  // vkCmdCopyQueryPoolResults: per query, the counter (64-bit, or its low 32 bits) and, when asked,
  // availability from the command buffer's constants (uint64 0 / 1).
  void queryCopy(const OpQueryCopy& q) {
    QueryPoolObj* p = q.pool;
    if (!p || !q.dst || !q.dst->id) {
      ++st.dropped;
      return;
    }
    const bool wide = q.flags & VK_QUERY_RESULT_64_BIT;
    const uint64_t size = wide ? 8 : 4;
    const bool backed = (p->type == VK_QUERY_TYPE_OCCLUSION || p->type == VK_QUERY_TYPE_TIMESTAMP) &&
                        p->base != UINT32_MAX && p->device && p->device->visibility.id;
    auto copy = [&](vpgpu::BufferId src, uint64_t srcOff, uint64_t dstOff) {
      vpgpu::Copy c;
      c.kind = vpgpu::CopyKind::BufferToBuffer;
      c.srcBuffer = {src, srcOff, size};
      c.dstBuffer = {q.dst->id, dstOff, size};
      c.label = "vk.query_copy";
      frame.copy(std::move(c));
      ++st.copies;
    };
    // Timestamps are host values known now (queryEnd): written straight into the destination at
    // this point of the recording order, never copied later from the pool's shared slots (a
    // later submission could have rewritten them by the time the GPU copy runs).
    uint8_t* dstHost = q.dst->memory && q.dst->memory->host ? q.dst->memory->host + q.dst->offset : nullptr;
    if (backed && p->type == VK_QUERY_TYPE_TIMESTAMP && dstHost && p->device->visibility.host) {
      for (uint32_t i = 0; i < q.count; ++i) {
        const uint32_t qi = q.first + i;
        uint8_t* at = dstHost + q.offset + uint64_t(i) * q.stride;
        const bool avail = queryEnded(p, qi);
        uint64_t v = 0;
        std::memcpy(&v, p->device->visibility.host + size_t(p->base + qi) * 8, 8);
        auto put = [&](uint8_t* where, uint64_t x) {
          if (wide) std::memcpy(where, &x, 8);
          else {
            const uint32_t y = uint32_t(x);
            std::memcpy(where, &y, 4);
          }
        };
        if (avail || (q.flags & VK_QUERY_RESULT_PARTIAL_BIT)) put(at, v);
        if (q.flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) put(at + size, avail ? 1 : 0);
      }
      return;
    }
    for (uint32_t i = 0; i < q.count; ++i) {
      const uint32_t qi = q.first + i;
      const uint64_t at = q.offset + uint64_t(i) * q.stride;
      // Unbacked pools: occlusion reads as visible (1), timestamps as 0; always available.
      const bool avail = !backed || queryEnded(p, qi);
      if (backed && (avail || (q.flags & VK_QUERY_RESULT_PARTIAL_BIT)))
        copy(p->device->visibility.id, uint64_t(p->base + qi) * 8, at);
      else if (!backed)
        copy(q.constants, q.constantsOffset + (p->type == VK_QUERY_TYPE_OCCLUSION ? 8 : 0), at);
      if (q.flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) copy(q.constants, q.constantsOffset + (avail ? 8 : 0), at + size);
    }
  }

  // Foveation: 2D drawn into the eye buffer (the VPKTAA UI class) would lose resolution toward the periphery. From the
  // first UI draw of a foveated pass on, the pass goes on without the rate map: the foveated part ends (the backend
  // expands it), a pass loading everything continues. Needs the draw classification (VPKTAA UI mask on).
  void fovUiSplit(const OpDraw& d) {
    if (!fovPass || !fovs || !fovs->uiSplit || !uiOn || !passOpen) return;
    if (classOf(d) != taa::DrawClass::Ui) return;
    vpgpu::RenderPassDesc nd = current;
    nd.rateMap = nullptr;
    for (vpgpu::Attachment& a : nd.colors)
      if (a.texture) a.load = vpgpu::LoadAction::Load;
    if (nd.depth.texture) nd.depth.load = vpgpu::LoadAction::Load;
    frame.endRenderPass();
    fovPass = false, fovSuppressed = true;
    current = nd;
    frame.beginRenderPass(std::move(nd));
    ++st.passes;
    ++st.fovUiSplits;
  }

  void skipDraw(SkipReason why, BindFail bind = BindFail::None) {
    ++st.skippedDraws;
    ++st.skippedBy[size_t(why)];
    if (!passDump().on) return;
    PassNote& n = passNote();
    ++n.skipped[size_t(why)];
    if (bind != BindFail::None) ++n.bindFail[size_t(bind)];
  }

  // VPVK_PASS_DUMP: the note of the open pass (its index among the frame's commands), or of the draws outside any pass.
  PassNote& passNote() {
    const uint32_t ix = passOpen ? uint32_t(frame.commands().size() - 1) : kNoPass;
    std::vector<PassNote>& v = st.passNotes;
    if (v.empty() || v.back().pass != ix) {  // (draws outside any pass may come between the draws of one pass)
      const auto it = std::find_if(v.rbegin(), v.rend(), [ix](const PassNote& n) { return n.pass == ix; });
      if (it != v.rend()) return *it;
      v.emplace_back().pass = ix;
    }
    return v.back();
  }
  // VPVK_SKIP_PASS: is the open pass one whose draws are dropped?
  bool skippedPass() const {
    const std::vector<uint32_t>& l = skipPasses();
    return !l.empty() && passOpen && std::binary_search(l.begin(), l.end(), uint32_t(frame.commands().size() - 1));
  }

  // Dynamic rendering: does an in-pass barrier come before a draw that reads an attachment of this pass?
  static bool localReadAhead(const Recorder& rec, size_t begin) {
    bool barrier = false;
    for (size_t i = begin + 1; i < rec.ops.size(); ++i) {
      const Op& o = rec.ops[i];
      if (std::holds_alternative<OpEndPass>(o)) return false;
      if (std::holds_alternative<OpPassBarrier>(o)) barrier = true;
      if (const auto* d = std::get_if<OpDraw>(&o); d && barrier && !d->images.empty()) return true;
    }
    return false;
  }

  // A barrier inside a dynamic rendering pass (VK_KHR_dynamic_rendering_local_read): the draws after it may read what
  // the ones before it wrote to the attachments (input attachments, or the attachment image bound as a texture: Source
  // 2's soft particles read the multiview depth). Metal cannot sample a texture that is an attachment of the open
  // encoder, and framebuffer fetch reads colour only, so the pass is stored (localReadAhead made every attachment
  // store), each attachment those draws read is copied into a driver-owned texture of the same shape, the pass
  // reopens loading its contents, and the reads are rebound to the copies (rebindLocalReads). A barrier with no such
  // read costs nothing. Multisampled attachments and 3D images are not copied (their reads stay as recorded).
  void localReadSplit(const Recorder& rec, size_t at) {
    std::vector<const Image*> read;
    auto attachmentImage = [&](const vpgpu::TextureId& t) -> const Image* {
      if (!t) return nullptr;
      for (size_t i = at + 1; i < rec.ops.size(); ++i) {
        const Op& o = rec.ops[i];
        if (std::holds_alternative<OpEndPass>(o) || std::holds_alternative<OpNextSubpass>(o)) break;
        const auto* d = std::get_if<OpDraw>(&o);
        if (!d) continue;
        for (const Image* im : d->images)
          if (im->id == t) return im;
      }
      return nullptr;
    };
    for (const vpgpu::Attachment& a : current.colors)
      if (const Image* im = attachmentImage(a.texture)) read.push_back(im);
    if (const Image* im = attachmentImage(current.depth.texture)) read.push_back(im);
    if (read.empty()) return;
    ++st.localReads;
    vpgpu::RenderPassDesc d = current;
    for (vpgpu::Attachment& a : d.colors)
      if (a.texture) a.load = vpgpu::LoadAction::Load;
    if (d.depth.texture) d.depth.load = vpgpu::LoadAction::Load;
    close();
    localCopies.clear();
    for (const Image* im : read) {
      if (im->desc.samples > 1 || im->desc.type == vpgpu::TextureType::Tex3D) {
        static std::atomic<bool> msaaOnce{false}, d3Once{false};
        const bool msaa = im->desc.samples > 1;
        if (!(msaa ? msaaOnce : d3Once).exchange(true))
          logf("vk.unimpl local_read_%s: the draw reads the attachment as recorded", msaa ? "msaa" : "3d");
        continue;
      }
      if (!im->localReadCopy) {
        vpgpu::TextureDesc td = im->desc;
        td.usage = vpgpu::usage::sampled;
        td.transient = false, td.persistent = true;
        td.label = "vk.local_read";
        im->localReadCopy = device->gpu->createTexture(td);
      }
      if (!im->localReadCopy) {
        static std::atomic<bool> failOnce{false};
        if (!failOnce.exchange(true)) logf("vk.unimpl local_read_copy_fail: the draw reads the attachment as recorded");
        continue;
      }
      for (uint16_t m = 0; m < im->desc.mips; ++m) {
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::TextureToTexture;
        const uint32_t w = std::max(1u, im->desc.width >> m), h = std::max(1u, im->desc.height >> m);
        c.srcTexture = {im->id, m, 0, im->desc.layers, 0, 0, 0, w, h, 1};
        c.dstTexture = {im->localReadCopy, m, 0, im->desc.layers, 0, 0, 0, w, h, 1};
        c.label = "vk.local_read";
        frame.copy(std::move(c));
        ++st.copies;
      }
      localCopies.push_back({im->id, im->localReadCopy});
    }
    open(std::move(d));
  }

  // Reads of attachments copied by localReadSplit go to the copies.
  void rebindLocalReads(vpgpu::Draw& d) const {
    for (const auto& [attachment, copy] : localCopies)
      for (vpgpu::Access& a : d.bindings) {
        if (a.range.kind != vpgpu::ResourceRange::Kind::Texture || a.range.index != attachment.index ||
            a.range.generation != attachment.generation)
          continue;
        a.range.index = copy.index, a.range.generation = copy.generation;
        // A trivial view of the attachment has no ViewBinding (entry.cpp createViewOne): the copy, same shape, serves.
        // Any ViewBinding is a backend view of the attachment (other type, format, aspect or subresource): it cannot
        // follow the retarget, so the binding falls back to the copy's own default view (logged once).
        const auto it = std::find_if(d.views.begin(), d.views.end(),
                                     [&](const vpgpu::ViewBinding& v) { return v.kind == a.kind && v.slot == a.slot; });
        if (it == d.views.end()) continue;
        static std::atomic<bool> viewOnce{false};
        if (!viewOnce.exchange(true))
          logf("vk.unimpl local_read_view: a non-trivial view of the attachment reads the copy through its default view");
        d.views.erase(it);
      }
  }

  static uint64_t depthTileKey(const PipelineObj* p, uint32_t sub) { return uint64_t(reinterpret_cast<uintptr_t>(p)) * 2 + (sub ? 1 : 0); }

  // "gpu.pass.fuse src=vk depth_input_in_tile=N reason=..": the decision for a pass that reads a multisampled depth as an input
  // attachment. The first 64 lines, then one in 2000.
  void depthTileLog(const OpBeginPass& b, bool on, const char* why, uint32_t att, size_t draws0 = 0, size_t draws1 = 0, size_t fetchPipes = 0) {
    static std::atomic<unsigned> n{0};
    const unsigned k = n.fetch_add(1);
    if (k >= 64 && k % 2000) return;
    logf("gpu.pass.fuse src=vk depth_input_in_tile=%d reason=%s att=%u fb=%ux%u draws0=%zu draws1=%zu fetch_pipes=%zu", on ? 1 : 0, why, att,
         b.framebuffer ? b.framebuffer->width : 0, b.framebuffer ? b.framebuffer->height : 0, draws0, draws1, fetchPipes);
  }

  // Decides, when subpass 0 of `b` opens as `desc`, whether the depth the later subpass reads as an input attachment can stay on
  // tile; if so adds the shadow attachment to `desc` and fills `dt`. Any unmet condition leaves everything as it was (the
  // later subpass samples the stored depth: the encoder is cut) and logs why. A pass without such an input says nothing.
  void planDepthTile(const OpBeginPass& b, const std::vector<Op>& ops, size_t at, vpgpu::RenderPassDesc& desc, bool areaDepth) {
    dt = DepthTile{};
    if (!b.pass || !b.framebuffer) return;
    const auto& sps = b.pass->subpasses;
    uint32_t att = VK_ATTACHMENT_UNUSED;
    const ImageView* dv = nullptr;
    for (size_t s = 1; s < sps.size() && !dv; ++s)
      for (const VkAttachmentReference2& r : sps[s].inputs) {
        const ImageView* v = r.attachment < b.framebuffer->views.size() ? b.framebuffer->views[r.attachment] : nullptr;
        if (v && v->image && v->image->desc.samples >= 2 && vpgpu::formatInfo(v->image->desc.format).depth()) {
          att = r.attachment, dv = v;
          break;
        }
      }
    if (!dv) return;
    const Image* img = dv->image;
    auto no = [&](const char* why) { depthTileLog(b, false, why, att); };
    if (!depthTileOn()) return no("disabled");
    Device* dev = devForce();
    if (!dev || !dev->gpu || !img->id) return no("no_device");
    if (sps.size() < 2 || !sps[0].inputs.empty()) return no("subpasses");
    const RenderPassObj::Subpass& s0 = sps[0];
    if (!s0.hasDepth || s0.depth.attachment != att) return no("depth_attachment");
    for (size_t s = 1; s < sps.size(); ++s) {  // every later subpass: the same depth, no other input, the colours of subpass 0
      const RenderPassObj::Subpass& sn = sps[s];
      if (!sn.hasDepth || sn.depth.attachment != att) return no("depth_attachment");
      for (const VkAttachmentReference2& r : sn.inputs)
        if (r.attachment != att) return no("other_inputs");
      if (s0.viewMask != sn.viewMask) return no("view_mask");
      for (size_t c = 0; c < sn.colors.size(); ++c) {
        const uint32_t an = sn.colors[c].attachment;
        const uint32_t a0 = c < s0.colors.size() ? s0.colors[c].attachment : VK_ATTACHMENT_UNUSED;
        if (an != VK_ATTACHMENT_UNUSED && an != a0) return no("colors");
      }
    }
    // (the VPKTAA jitter only moves the viewport and the mask variant is what jitter() would replace: those stay out)
    if (maskPass || forceN || desc.rateMap || (fovs && taaPass)) return no("pass_state");
    if (desc.depth.autoMsaa || desc.depth.texture != img->id) return no("depth_attachment");
    for (const vpgpu::Attachment& a : desc.colors)
      if (a.autoMsaa) return no("auto_msaa");
    if (desc.colors[kDepthTileSlot].texture) return no("slot_taken");
    if (desc.depth.load == vpgpu::LoadAction::Load) return no("depth_load");
    if (areaDepth || partialArea(b.area, dv)) return no("area_clear");
    if (dv->range.baseMipLevel != 0 || img->desc.type != vpgpu::TextureType::Tex2D) return no("depth_view");
    if (img->desc.layers < desc.layerCount) return no("layers");
    // The draws: subpass 0 must leave the shadow equal to its depth, the later subpasses must not change the depth.
    struct Seen {
      const PipelineObj* p;
      uint32_t sub;
      int kind, texSlot;
    };
    std::vector<Seen> seen;
    std::unordered_map<uint64_t, size_t> seenAt;  // depthTileKey -> index in `seen`
    size_t draws[2] = {0, 0};
    uint32_t sub = 0;
    const char* bad = nullptr;
    auto visit = [&](auto&& self, const Op& op) -> bool {  // true: stop
      if (const auto* d = std::get_if<OpDraw>(&op)) {
        if (!d->ok) return false;
        if (!d->pipe || !d->pipe->ok || !d->pipe->depthTileSrc || d->pipe->sources.empty()) return bad = "pipeline", true;
        if (d->edsVariant || d->texelVariant) return bad = "pipeline_variant", true;
        const PipelineObj& p = *d->pipe;
        const bool writes = p.state.depthTest && p.state.depthWrite;
        if (p.dualSource) return bad = "dual_source", true;
        if (d->passSamples && d->passSamples != p.state.samples) return bad = "pass_samples", true;
        if (d->draw.depthClamp) return bad = "depth_clamp", true;
        ++draws[sub ? 1 : 0];
        int kind = kDepthTileState, texSlot = -1;
        if (sub == 0) {
          if (d->draw.depthBias != 0 || d->draw.depthBiasSlope != 0) return bad = "depth_bias", true;
          if (writes) {
            if (p.d24Emul) return bad = "d24_emul", true;
            bool frag = false;
            for (const PipelineObj::StageSrc& src : p.sources) frag = frag || (src.stage == VK_SHADER_STAGE_FRAGMENT_BIT && !src.synthetic);
            if (!frag) return bad = "depth_only", true;
            kind = kDepthTileWrite;
          }
        } else {
          if (writes) return bad = "reader_writes_depth", true;
          int inputs = 0;
          for (const PipelineObj::Used& u : p.used)
            if (u.slot.type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
              ++inputs;
              texSlot = int(u.slot.slot);
              if (u.slot.count != 1) return bad = "input_array", true;
            }
          if (inputs > 1) return bad = "other_inputs", true;
          if (inputs) {
            kind = kDepthTileFetch;
            bool found = false;
            for (const vpgpu::Access& a : d->draw.bindings)
              if (a.slot == texSlot && vpgpu::bindKind(a) == vpgpu::BindKind::Texture && !a.pixelLocal) {
                if (a.range.kind != vpgpu::ResourceRange::Kind::Texture || a.range.index != img->id.index || a.range.generation != img->id.generation)
                  return bad = "input_not_depth", true;
                found = true;
              }
            if (!found) return bad = "input_not_depth", true;
          }
        }
        const auto [it, fresh] = seenAt.emplace(depthTileKey(d->pipe, sub), seen.size());
        if (!fresh) {
          const Seen& x = seen[it->second];
          if (x.kind != kind || x.texSlot != texSlot) return bad = "pipeline_mixed", true;
          return false;
        }
        seen.push_back({d->pipe, sub, kind, texSlot});
      } else if (const auto* e = std::get_if<OpExecute>(&op)) {
        if (e->secondary)
          for (const Op& so : e->secondary->ops)
            if (self(self, so)) return true;
      } else if (std::holds_alternative<OpNextSubpass>(op)) {
        ++sub;
      } else if (std::holds_alternative<OpPassBarrier>(op)) {
        return bad = "barrier", true;
      } else if (std::holds_alternative<OpClearAttachments>(op)) {
        return bad = "clear_attachments", true;
      } else if (std::holds_alternative<OpEndPass>(op) || std::holds_alternative<OpBeginPass>(op) || std::holds_alternative<OpBeginRendering>(op)) {
        return true;
      }
      return false;
    };
    for (size_t i = at + 1; i < ops.size(); ++i)
      if (visit(visit, ops[i])) break;
    if (bad) return no(bad);
    if (sub + 1 != sps.size()) return no("subpasses");
    // The variants: all translated first (nothing requested or recorded), committed only when every one exists and the pass is taken.
    std::unordered_map<uint64_t, DepthTile::Use> uses;
    size_t fetchPipes = 0;
    for (const Seen& x : seen) {
      DepthTile::Use u;
      u.kind = x.kind, u.texSlot = x.texSlot;
      if (!depthPipelineKey(dev, *x.p, x.kind, x.texSlot, u.key, false)) return no(x.kind == kDepthTileFetch ? "lowering_fetch" : "lowering_write");
      fetchPipes += x.kind == kDepthTileFetch;
      uses[depthTileKey(x.p, x.sub)] = u;
    }
    if (!fetchPipes) return no("no_reader");
    if (!img->depthTileShadow) {
      vpgpu::TextureDesc td = img->desc;
      td.format = vpgpu::Format::R32Float;
      td.mips = 1;
      td.usage = vpgpu::usage::colorTarget;
      td.transient = true, td.persistent = false, td.samplesReadable = false;
      td.label = "vk.depth_tile";
      img->depthTileShadow = dev->gpu->createTexture(td);
    }
    if (!img->depthTileShadow) return no("texture");
    dt.shadow = {};
    dt.shadow.texture = img->depthTileShadow;
    dt.shadow.layer = uint16_t(dv->range.baseArrayLayer);
    dt.shadow.load = desc.depth.load == vpgpu::LoadAction::Clear ? vpgpu::LoadAction::Clear : vpgpu::LoadAction::DontCare;
    dt.shadow.clear.color = {double(desc.depth.clear.depth), 0, 0, 0};
    dt.shadow.store = vpgpu::StoreAction::Store;  // (the core drops it when the next subpass merges and the texture is transient)
    for (const Seen& x : seen) depthPipelineKey(dev, *x.p, x.kind, x.texSlot, uses[depthTileKey(x.p, x.sub)].key, true);
    dt.uses = std::move(uses);
    dt.active = true;
    desc.colors[kDepthTileSlot] = dt.shadow;
    depthTileLog(b, true, "ok", att, draws[0], draws[1], fetchPipes);
  }

  // A later subpass (1..N-1) of a depth-tile pass: the shadow is read, never written, and not needed after the last one.
  void depthTileReader(vpgpu::RenderPassDesc& desc) {
    if (!dt.active) return;
    if (maskPass || forceN || (fovs && taaPass)) {  // (subpass 1 is judged on its own): it samples the stored depth, as before
      dt.active = false;
      return;
    }
    desc.colors[kDepthTileSlot] = dt.shadow;
    desc.colors[kDepthTileSlot].load = vpgpu::LoadAction::Load;
    // (a subpass that another one follows keeps it, like subpass 0: the core drops the store when the next subpass merges)
    desc.colors[kDepthTileSlot].store = pass && pass->pass && subpass + 1 < pass->pass->subpasses.size() ? vpgpu::StoreAction::Store
                                                                                                          : vpgpu::StoreAction::DontCare;
  }

  // The draw with the pipeline variant of the depth-tile pass and, for a fetch, the shadow read pixel-locally instead of the depth.
  bool depthTileDraw(const OpDraw& d, const vpgpu::Draw& base, vpgpu::Draw& out) {
    if (!dt.active || !d.pipe) return false;
    const auto it = dt.uses.find(depthTileKey(d.pipe, subpass));
    if (it == dt.uses.end()) {
      if (passDump().on) ++passNote().fallbackDepthTile;  // the draw keeps its plain pipeline
      return false;
    }
    out = base;
    out.pipeline = it->second.key;
    if (it->second.kind != kDepthTileFetch) return true;
    for (vpgpu::Access& a : out.bindings)
      if (a.slot == it->second.texSlot && vpgpu::bindKind(a) == vpgpu::BindKind::Texture && !a.pixelLocal) {
        const vpgpu::StageMask stages = a.stages;
        a = vpgpu::pixelLocalRead(dt.shadow.texture, dt.shadow.mip, dt.shadow.layer);
        a.stages = stages;
        a.slot = uint16_t(it->second.texSlot);
        break;
      }
    out.views.erase(std::remove_if(out.views.begin(), out.views.end(),
                                   [&](const vpgpu::ViewBinding& v) {
                                     return v.kind == vpgpu::BindKind::Texture && v.slot == it->second.texSlot;
                                   }),
                    out.views.end());
    return true;
  }

  // The last step of a draw: into the frame, or (VPVK_SKIP_PASS) dropped here, after everything else the draw does (pass splits, zero-init
  // touches, jitter and census counters), so the pass numbering is the same with and without the skip.
  void emit(const OpDraw& d, vpgpu::Draw dr) {
    if (skippedPass()) return skipDraw(SkipReason::Experiment);
    if (passDump().on) {
      PassNote& n = passNote();
      ++n.draws;
      for (const auto& [id, flags] : d.bind.tex) n.tex[id] |= flags;
    }
    frame.draw(std::move(dr));
    ++st.draws;
  }

  void draw(const OpDraw& d, size_t opIx) {
    if (!d.ok || !passOpen) {
      skipDraw(d.ok ? SkipReason::Pass : d.reason, d.bind.fail);
      if (d.ok) logSkip("no_open_pass", d.pipe, "op=%zu", opIx);
      return;
    }
    zeroReads(d.zeroReads);
    fovUiSplit(d);
    // After a local-read split the draw's reads of the attachments go to the copies (localReadSplit).
    vpgpu::Draw rebound;
    const vpgpu::Draw* base = &d.draw;
    if (!localCopies.empty() && !d.images.empty()) {
      rebound = d.draw;
      rebindLocalReads(rebound);
      base = &rebound;
    }
    vpgpu::Draw tiled;  // depth in tile (planDepthTile)
    if (dt.active && depthTileDraw(d, *base, tiled)) base = &tiled;
    // Forced MSAA: the pipeline variant with the pass's sample count (the draws were recorded for a 1x pass).
    vpgpu::Draw forced;
    if (forceN && d.pipe && !d.edsVariant && base == &d.draw) {
      forced = d.draw;
      forced.pipeline = pipelineFor(devForce(), *d.pipe, forceN, d.texelVariant);
      base = &forced;
    }
    if (!d.indirect) {
      if (d.draw.instances > std::max<uint32_t>(1, d.viewCount)) ++st.instancedDraws;  // instances include the views
      if (taaPass) {
        vpgpu::Draw dr = *base;
        jitter(dr, d);
        emit(d, std::move(dr));
      } else {
        if (eyeScene) noteDepthDraw(dd, *base, d);
        if (base == &tiled) emit(d, std::move(tiled));  // (the variant copy is the draw: no second copy)
        else emit(d, vpgpu::Draw(*base));
      }
      return;
    }
    const bool indexed = base->index.has_value();
    const uint32_t size = indexed ? 20 : 16;
    if (d.gpuIndirect) {  // the GPU reads the records (pipelines.cpp recordIndirectDraw)
      st.indirectGpu += d.indirectCount;
      for (uint32_t i = 0; i < d.indirectCount; ++i) {
        vpgpu::Draw dr = *base;
        dr.indirect = vpgpu::Indirect{d.gpuBuffer, d.gpuOffset + uint64_t(i) * d.indirectStride};
        dr.drawIndex = i;
        jitter(dr, d);
        emit(d, std::move(dr));
      }
      return;
    }
    // Arguments read now, on the CPU: a buffer the GPU fills later in the submission is read stale (known gap for
    // multiview / count draws in secondary command buffers: pipelines.cpp recordIndirectDraw cannot patch them there).
    st.indirectCpu += d.indirectCount;
    if (secondaryDepth) st.indirectCpuSecondary += d.indirectCount;
    for (uint32_t i = 0; i < d.indirectCount; ++i) {
      const uint8_t* p = indirectData(d.indirect, d.indirectOffset + VkDeviceSize(i) * d.indirectStride, size);
      if (!p) {
        skipDraw(SkipReason::Indirect);
        logSkip("indirect_data_null", d.pipe, "op=%zu draw=%u of %u", opIx, i, d.indirectCount);
        continue;
      }
      vpgpu::Draw dr = *base;
      dr.drawIndex = i;
      uint32_t instances;  // the record's, before the multiview multiplication
      if (indexed) {
        VkDrawIndexedIndirectCommand c;
        std::memcpy(&c, p, sizeof c);
        dr.count = c.indexCount, dr.instances = c.instanceCount * d.viewCount, dr.first = c.firstIndex;
        dr.baseVertex = c.vertexOffset, dr.baseInstance = c.firstInstance * d.viewCount;
        instances = c.instanceCount;
      } else {
        VkDrawIndirectCommand c;
        std::memcpy(&c, p, sizeof c);
        dr.count = c.vertexCount, dr.instances = c.instanceCount * d.viewCount, dr.first = c.firstVertex;
        dr.baseInstance = c.firstInstance * d.viewCount;
        instances = c.instanceCount;
      }
      if (!dr.count || !dr.instances) continue;
      if (dr.count > kMaxIndirectCount || instances > kMaxIndirectInstances) {
        skipDraw(SkipReason::Indirect);
        logSkip("indirect_clamped", d.pipe, "op=%zu draw=%u of %u count=%u instances=%u views=%u", opIx, i, d.indirectCount,
                dr.count, instances, d.viewCount);
        continue;
      }
      if (dr.instances > std::max<uint32_t>(1, d.viewCount)) ++st.instancedDraws;
      jitter(dr, d);
      emit(d, std::move(dr));
    }
  }

  // vkCmdResolveImage (outside render passes): per layer, an empty pass that loads the MSAA image
  // and stores it resolved into the destination (Metal resolves at the end of the pass, the whole
  // subresource). Integer formats, partial regions and images of other sizes: a sampled blit.
  void resolveImage(const OpResolveImage& rs) {
    if (!rs.src || !rs.dst || !rs.src->id || !rs.dst->id || passOpen) {
      ++st.dropped;
      return;
    }
    const VkImageResolve& g = rs.region;
    const uint32_t sw = std::max(1u, rs.src->desc.width >> g.srcSubresource.mipLevel);
    const uint32_t sh = std::max(1u, rs.src->desc.height >> g.srcSubresource.mipLevel);
    const uint32_t dw = std::max(1u, rs.dst->desc.width >> g.dstSubresource.mipLevel);
    const uint32_t dh = std::max(1u, rs.dst->desc.height >> g.dstSubresource.mipLevel);
    // What Metal can resolve at the end of a pass: a multisampled float/normalized colour image into
    // a single-sample one of the same size (depth goes through render-pass resolves). Anything
    // else not handled below is skipped, named once in the log.
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(rs.src->desc.format);
    const bool partialRegion = g.srcOffset.x || g.srcOffset.y || g.dstOffset.x || g.dstOffset.y ||
                               g.extent.width < sw || g.extent.height < sh || sw != dw || sh != dh;
    if ((fi.integer() || partialRegion) && !fi.depth() && !fi.stencil() && rs.src->desc.samples > 1 &&
        rs.dst->desc.samples == 1) {
      // Integer formats have no average (Vulkan takes sample 0) and Metal resolves none; regions
      // and images of other sizes Metal's whole-subresource resolve cannot express. Both are a
      // blit from the multisampled source (gpu/metal2 vp_blit_ms_*: sample 0 for integers, the
      // average of the samples for float formats).
      const uint32_t w = std::min({g.extent.width, sw - std::min<uint32_t>(sw, uint32_t(std::max(0, g.srcOffset.x))),
                                   dw - std::min<uint32_t>(dw, uint32_t(std::max(0, g.dstOffset.x)))});
      const uint32_t h = std::min({g.extent.height, sh - std::min<uint32_t>(sh, uint32_t(std::max(0, g.srcOffset.y))),
                                   dh - std::min<uint32_t>(dh, uint32_t(std::max(0, g.dstOffset.y)))});
      const uint32_t n = g.srcSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS
                             ? rs.src->desc.layers - std::min<uint32_t>(rs.src->desc.layers, g.srcSubresource.baseArrayLayer)
                             : g.srcSubresource.layerCount;
      if (!w || !h || !n) return;
      vpgpu::Copy c;
      c.kind = vpgpu::CopyKind::TextureBlit;
      c.blitFilter = vpgpu::Filter::Nearest;
      c.srcTexture = {rs.src->id, uint16_t(g.srcSubresource.mipLevel), uint16_t(g.srcSubresource.baseArrayLayer), uint16_t(n),
                      uint32_t(std::max(0, g.srcOffset.x)), uint32_t(std::max(0, g.srcOffset.y)), 0, w, h, 1};
      c.dstTexture = {rs.dst->id, uint16_t(g.dstSubresource.mipLevel), uint16_t(g.dstSubresource.baseArrayLayer), uint16_t(n),
                      uint32_t(std::max(0, g.dstOffset.x)), uint32_t(std::max(0, g.dstOffset.y)), 0, w, h, 1};
      c.label = "vk.resolve_blit";
      frame.copy(std::move(c));
      ++st.copies;
      return;
    }
    const char* why = fi.integer() ? "integer_format" : (fi.depth() || fi.stencil()) ? "depth_stencil"
                      : rs.src->desc.samples <= 1 ? "source_not_multisampled"
                      : rs.dst->desc.samples != 1 ? "destination_multisampled"
                      : (sw != dw || sh != dh) ? "size_mismatch" : nullptr;
    if (why) {
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 8) logf("vk.resolve_image skipped=%s %ux%u -> %ux%u", why, sw, sh, dw, dh);
      ++st.dropped;
      return;
    }

    uint32_t layers = g.srcSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS
                          ? rs.src->desc.layers - std::min<uint32_t>(rs.src->desc.layers, g.srcSubresource.baseArrayLayer)
                          : g.srcSubresource.layerCount;
    layers = std::min<uint32_t>(std::max(1u, layers), rs.dst->desc.layers > g.dstSubresource.baseArrayLayer
                                                         ? rs.dst->desc.layers - g.dstSubresource.baseArrayLayer : 1);
    for (uint32_t l = 0; l < layers; ++l) {
      vpgpu::RenderPassDesc d;
      vpgpu::Attachment& a = d.colors[0];
      a.texture = rs.src->id;
      a.mip = uint16_t(g.srcSubresource.mipLevel);
      a.layer = uint16_t(g.srcSubresource.baseArrayLayer + l);
      a.load = vpgpu::LoadAction::Load;
      a.store = vpgpu::StoreAction::StoreAndResolve;
      a.keepSamples = true;  // vkCmdResolveImage: the source samples are the data
      a.keepOrigin = vpgpu::KeepOrigin::Required;
      a.resolve = rs.dst->id;
      a.resolveMip = uint16_t(g.dstSubresource.mipLevel);
      a.resolveLayer = uint16_t(g.dstSubresource.baseArrayLayer + l);
      d.label = "vk.resolve";
      frame.beginRenderPass(std::move(d));
      frame.endRenderPass();
      ++st.copies;
    }
  }

  // Buffer -> 3D ASTC image (created as RGBA8): every depth slice's blocks decoded into the op's arena room, then copied as RGBA8.
  // The blocks are read from the buffer's host memory now (submit), as the GPU would have at the copy.
  void decodedVolumeCopy(const OpCopyBufferToImage& op) {
    const VkBufferImageCopy& r = op.region;
    const vpgpu::Format src = toCoreFormat(op.dst->info.format);
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(src);
    const uint32_t bw = fi.blockSize, bh = fi.blockHeight, w = r.imageExtent.width, h = r.imageExtent.height, d = r.imageExtent.depth;
    if (!op.dst->id || !w || !h || !d) {
      ++st.dropped;
      return;
    }
    const uint32_t rowTexels = r.bufferRowLength ? r.bufferRowLength : w, rows = r.bufferImageHeight ? r.bufferImageHeight : h;
    const VkDeviceSize rowBytes = VkDeviceSize((rowTexels + bw - 1) / bw) * 16, sliceBytes = rowBytes * ((rows + bh - 1) / bh);
    const uint32_t bx = (w + bw - 1) / bw, by = (h + bh - 1) / bh;
    const uint8_t* base = indirectData(op.src, r.bufferOffset, sliceBytes * (d - 1) + rowBytes * (by - 1) + VkDeviceSize(bx) * 16);
    if (!base) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 8) logf("vk.err what=astc_volume_source_unreadable buffer_host=%d (the copy is dropped)", int(op.src && op.src->memory && op.src->memory->host));
      ++st.dropped;
      return;
    }
    const bool srgb = astcDecodedFormat(src) == vpgpu::Format::RGBA8Srgb;
    uint8_t texels[12 * 12 * 4];
    unsigned bad = 0;
    for (uint32_t z = 0; z < d; ++z)
      for (uint32_t j = 0; j < by; ++j)
        for (uint32_t i = 0; i < bx; ++i) {
          bad += !astcDecodeBlock(base + z * sliceBytes + j * rowBytes + i * 16, int(bw), int(bh), srgb, texels);
          for (uint32_t y = 0; y < bh && j * bh + y < h; ++y)
            for (uint32_t x = 0; x < bw && i * bw + x < w; ++x)
              std::memcpy(op.decodedHost + ((VkDeviceSize(z) * h + j * bh + y) * w + i * bw + x) * 4, texels + (y * bw + x) * 4, 4);
        }
    if (bad) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 8) logf("vk.err what=astc_volume_invalid_blocks blocks=%u of %u (error colour)", bad, bx * by * d);
    }
    vpgpu::Copy c;
    c.kind = vpgpu::CopyKind::BufferToTexture;
    c.srcBuffer.buffer = op.decoded;
    c.srcBuffer.offset = op.decodedOffset;
    c.srcBuffer.bytesPerRow = w * 4;
    c.srcBuffer.rowsPerImage = h;
    c.srcBuffer.size = VkDeviceSize(w) * h * d * 4;
    vpgpu::TextureSpan& ts = c.dstTexture;
    ts.texture = op.dst->id;
    ts.mip = static_cast<uint16_t>(r.imageSubresource.mipLevel);
    ts.x = uint32_t(r.imageOffset.x), ts.y = uint32_t(r.imageOffset.y), ts.z = uint32_t(r.imageOffset.z);
    ts.width = w, ts.height = h, ts.depth = d;
    c.label = "vk.upload_astc3d";
    st.uploadBytes += c.srcBuffer.size;
    frame.copy(std::move(c));
    ++st.copies;
  }

  // Texture budget: level 0 of an uncompressed image that lost its largest levels, box-filtered by 2^shrink into the op's arena
  // room and copied to the largest level left (sRGB texels are averaged as stored). The texels are read from the buffer's host
  // memory now (submit), as the GPU would have at the copy.
  void shrunkCopy(const OpCopyBufferToImage& op) {
    const VkBufferImageCopy& r = op.region;
    const BudgetShrink bs = budgetShrink(*op.dst, r);
    if (!op.dst->id || !bs.bytes) {
      ++st.dropped;
      return;
    }
    const uint32_t s = op.shrink, n = bs.texelBytes, w = r.imageExtent.width, h = r.imageExtent.height;
    const uint32_t rowTexels = r.bufferRowLength ? r.bufferRowLength : w, rows = r.bufferImageHeight ? r.bufferImageHeight : h;
    const VkDeviceSize rowBytes = VkDeviceSize(rowTexels) * n, layerBytes = rowBytes * rows;
    const uint8_t* base = indirectData(op.src, r.bufferOffset, layerBytes * (bs.layers - 1) + rowBytes * (h - 1) + VkDeviceSize(w) * n);
    if (!base) {
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 8) logf("vk.err what=tex_budget_source_unreadable buffer_host=%d (the copy is dropped)", int(op.src && op.src->memory && op.src->memory->host));
      ++st.dropped;
      return;
    }
    const uint32_t ox = uint32_t(r.imageOffset.x), oy = uint32_t(r.imageOffset.y);
    for (uint32_t l = 0; l < bs.layers; ++l) {
      const uint8_t* src = base + l * layerBytes;
      uint8_t* dst = op.decodedHost + VkDeviceSize(l) * bs.width * bs.height * n;
      for (uint32_t y = 0; y < bs.height; ++y) {
        // Rows and columns of the region under this texel (whole 2^s squares except at the region's edges).
        const uint32_t y0 = std::max((bs.y + y) << s, oy) - oy, y1 = std::min((bs.y + y + 1) << s, oy + h) - oy;
        for (uint32_t x = 0; x < bs.width; ++x) {
          const uint32_t x0 = std::max((bs.x + x) << s, ox) - ox, x1 = std::min((bs.x + x + 1) << s, ox + w) - ox;
          uint32_t sum[4] = {0, 0, 0, 0};
          for (uint32_t yy = y0; yy < y1; ++yy) {
            const uint8_t* p = src + yy * rowBytes + VkDeviceSize(x0) * n;
            for (uint32_t xx = x0; xx < x1; ++xx, p += n)
              for (uint32_t c = 0; c < n; ++c) sum[c] += p[c];
          }
          const uint32_t count = std::max(1u, (y1 - y0) * (x1 - x0));
          uint8_t* q = dst + (VkDeviceSize(y) * bs.width + x) * n;
          for (uint32_t c = 0; c < n; ++c) q[c] = uint8_t((sum[c] + count / 2) / count);
        }
      }
    }
    vpgpu::Copy c;
    c.kind = vpgpu::CopyKind::BufferToTexture;
    c.srcBuffer.buffer = op.decoded;
    c.srcBuffer.offset = op.decodedOffset;
    c.srcBuffer.bytesPerRow = bs.width * n;
    c.srcBuffer.rowsPerImage = bs.height;
    c.srcBuffer.size = bs.bytes;
    vpgpu::TextureSpan& ts = c.dstTexture;
    ts.texture = op.dst->id;
    ts.mip = 0;
    ts.layer = static_cast<uint16_t>(r.imageSubresource.baseArrayLayer);
    ts.layerCount = static_cast<uint16_t>(bs.layers);
    ts.x = bs.x, ts.y = bs.y, ts.z = 0;
    ts.width = bs.width, ts.height = bs.height, ts.depth = 1;
    c.label = "vk.upload_shrunk";
    st.uploadBytes += c.srcBuffer.size;
    frame.copy(std::move(c));
    ++st.copies;
  }

  void bufferImage(const Buffer* buf, const Image* img, const VkBufferImageCopy& r, bool toImage) {
    if (!buf->id || !img->id) {
      ++st.dropped;
      return;
    }
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(img->desc.format);
    const uint32_t rowTexels = r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width;
    const uint32_t rows = r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height;
    // Depth/stencil images: one aspect per region, tightly packed in the buffer (Vulkan: D16 2
    // bytes, D32/D24 4 bytes, stencil 1 byte per texel), whatever the combined format's size.
    vpgpu::Aspect aspect = vpgpu::Aspect::All;
    uint32_t texelBytes = fi.bytesPerBlock;
    if (fi.depth() || fi.stencil()) {
      if ((r.imageSubresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) && fi.stencil()) {
        aspect = vpgpu::Aspect::Stencil;
        texelBytes = 1;
      } else {
        aspect = vpgpu::Aspect::Depth;
        texelBytes = img->desc.format == vpgpu::Format::D16Unorm ? 2 : 4;
      }
    }
    vpgpu::Copy c;
    c.kind = toImage ? vpgpu::CopyKind::BufferToTexture : vpgpu::CopyKind::TextureToBuffer;
    vpgpu::BufferSpan& bs = toImage ? c.srcBuffer : c.dstBuffer;
    vpgpu::TextureSpan& ts = toImage ? c.dstTexture : c.srcTexture;
    bs.buffer = buf->id;
    bs.offset = r.bufferOffset;
    bs.bytesPerRow = (rowTexels + fi.blockSize - 1) / fi.blockSize * texelBytes;
    bs.rowsPerImage = (rows + fi.blockHeight - 1) / fi.blockHeight;
    bs.size = uint64_t(bs.bytesPerRow) * bs.rowsPerImage * r.imageExtent.depth * r.imageSubresource.layerCount;
    ts.texture = img->id;
    ts.mip = static_cast<uint16_t>(r.imageSubresource.mipLevel);
    ts.layer = static_cast<uint16_t>(r.imageSubresource.baseArrayLayer);
    ts.layerCount = static_cast<uint16_t>(r.imageSubresource.layerCount);
    ts.x = static_cast<uint32_t>(r.imageOffset.x);
    ts.y = static_cast<uint32_t>(r.imageOffset.y);
    ts.z = static_cast<uint32_t>(r.imageOffset.z);
    ts.width = r.imageExtent.width;
    ts.height = r.imageExtent.height;
    ts.depth = r.imageExtent.depth;
    ts.aspect = aspect;
    c.label = toImage ? "vk.upload" : "vk.readback";
    if (toImage) st.uploadBytes += bs.size;
    frame.copy(std::move(c));
    ++st.copies;
  }
};

}  // namespace

vpgpu::TextureId MsaaTwins::acquire(Device* dev, vpgpu::TextureId guest, const vpgpu::TextureDesc& gd, uint8_t samples,
                                    uint16_t layers, uint16_t mip, uint16_t layer) {
  if (!dev || !dev->gpu || !guest) return {};
  if ((acquires++ & 255u) == 0) {  // twins of images the game destroyed go with them
    for (size_t i = 0; i < twins.size();) {
      vpgpu::TextureDesc alive;
      if (dev->gpu->textureDesc(twins[i].guest, alive)) {
        ++i;
        continue;
      }
      dev->gpu->destroy(twins[i].twin);
      twins.erase(twins.begin() + static_cast<std::ptrdiff_t>(i));
    }
  }
  for (const Twin& t : twins)
    if (t.guest == guest && t.samples == samples && t.layers == layers && t.mip == mip && t.layer == layer) return t.twin;
  vpgpu::TextureDesc td;
  td.type = vpgpu::TextureType::Tex2D;
  td.format = gd.format;
  td.width = std::max(1u, gd.width >> mip), td.height = std::max(1u, gd.height >> mip);
  td.layers = layers, td.samples = samples;
  td.usage = vpgpu::formatInfo(gd.format).depth() ? vpgpu::usage::depthTarget : vpgpu::usage::colorTarget;
  td.transient = true;  // never read after its pass: resolved into the guest image, samples on tile only
  td.label = "vk.msaa_force";
  if (samples == 1) {
    // The 1x partner of a depth the game never resolves (gameAuto): only a cut of a heavy encoder resolves into it, and the next
    // chunk seeds its samples from it (depth plane, and the stencil plane through a stencil view: formatView). Memory only then.
    td.usage |= vpgpu::usage::sampled;
    if (vpgpu::formatInfo(gd.format).stencil()) td.usage |= vpgpu::usage::formatView;
    td.label = "vk.msaa_seed";
  }
  const vpgpu::TextureId id = dev->gpu->createTexture(td);
  if (!id) {
    logf("vk.err what=msaa_force_twin size=%ux%ux%u samples=%u", td.width, td.height, unsigned(layers), unsigned(samples));
    return {};
  }
  twins.push_back({guest, id, samples, layers, mip, layer});
  logf("vk.msaa force twin %ux%ux%u samples=%u fmt=%s (%s)", td.width, td.height, unsigned(layers), unsigned(samples),
       vpgpu::formatInfo(gd.format).name, samples == 1 ? "1x seed partner of a depth the game does not resolve" : "tile memory, resolved into the game's image");
  return id;
}

void MsaaTwins::release(Device* dev) {
  if (dev && dev->gpu)
    for (const Twin& t : twins) dev->gpu->destroy(t.twin);
  twins.clear();
}

// "vk.msaa rej <reason> WxHxL color=<fmt>/<samples>x depth=<fmt>/<samples>x": a pass forceMsaa left at 1x. Distinct (size, layers,
// reason) only: 12 below 1 Mpx and 12 at or above it, so the frame's small shadow / UI passes do not hide the eye pass.
void MsaaTwins::noteRej(Device* dev, const vpgpu::RenderPassDesc& d, const char* why) {
  lastWhy = why;
  if (rejLogged[0] >= 12 && rejLogged[1] >= 12) return;  // both log classes full: skip the texture lookups on the hot path
  vpgpu::TextureDesc gc, gz;
  const bool hc = d.colors[0].texture && dev->gpu->textureDesc(d.colors[0].texture, gc);
  const bool hz = d.depth.texture && dev->gpu->textureDesc(d.depth.texture, gz);
  const uint32_t w = hc ? std::max(1u, gc.width >> d.colors[0].mip) : hz ? std::max(1u, gz.width >> d.depth.mip) : 0;
  const uint32_t h = hc ? std::max(1u, gc.height >> d.colors[0].mip) : hz ? std::max(1u, gz.height >> d.depth.mip) : 0;
  const uint16_t layers = std::max<uint16_t>(d.layerCount, 1);
  const int cls = uint64_t(w) * h >= 1'000'000u ? 1 : 0;
  if (rejLogged[cls] >= 12) return;
  for (int i = 0; i < rejLogged[cls]; ++i) {
    const RejSeen& r = rejSeen[cls][i];
    if (r.w == w && r.h == h && r.layers == layers && !std::strcmp(r.why, why)) return;
  }
  rejSeen[cls][rejLogged[cls]++] = {w, h, layers, why};
  logf("vk.msaa rej %s %ux%ux%u color=%s/%ux depth=%s/%ux", why, w, h, unsigned(layers), hc ? vpgpu::formatInfo(gc.format).name : "none",
       hc ? unsigned(gc.samples) : 0u, hz ? vpgpu::formatInfo(gz.format).name : "none", hz ? unsigned(gz.samples) : 0u);
}

// The pass forceMsaa just judged competes for "vk.msaa main": an eye scene pass beats any other, then the larger render area. The
// game's samples are read only when it becomes the candidate (a pass or two per window), never per draw.
void MsaaTwins::noteMain(Device* dev, vpgpu::TextureId game, const VkRect2D& area, bool eye, uint8_t forced) {
  const uint64_t rank = (eye ? 1ull << 62 : 0) + uint64_t(area.extent.width) * area.extent.height;
  if (rank <= mainPass.rank) return;
  mainPass = {rank, area.extent.width, area.extent.height, 0, forced, eye, forced ? nullptr : lastWhy};
  vpgpu::TextureDesc g;
  if (game && dev->gpu->textureDesc(game, g)) mainPass.samples = uint8_t(g.samples);
}

bool MsaaTwins::takeMainLine(int64_t nowNs, MainPass& out) {
  if (!mainPass.rank) return false;
  if (!nextMainNs) nextMainNs = nowNs + kMainWindowNs;  // the window opens with the first pass
  if (nowNs < nextMainNs) return false;
  nextMainNs = nowNs + kMainWindowNs;
  const MainPass m = mainPass;
  mainPass = {};
  if (mainLines >= kMainLines) return false;
  const bool same = mainLines > 0 && (m.forced != 0) == (lastMain.forced != 0) && m.w == lastMain.w && m.h == lastMain.h &&
                    m.samples == lastMain.samples &&
                    (m.forced || std::strcmp(m.why ? m.why : "none", lastMain.why ? lastMain.why : "none") == 0);
  if (same && nowNs - lastMainLineNs < kMainBeatNs) return false;
  lastMain = m, lastMainLineNs = nowNs, ++mainLines;
  out = m;
  return true;
}

static unsigned hbucket(uint8_t s) { return s == 1 ? 0u : s == 2 ? 1u : s == 4 ? 2u : 3u; }

void MsaaTwins::EyeSummary::advance(uint64_t f) {
  if (f == frame) return;
  if (curDraws) mainDraws = curDraws, mainIn = curIn, mainOut = curOut;
  curDraws = 0, frame = f;
}

void MsaaTwins::EyeSummary::notePass(uint64_t f, uint32_t w, uint32_t h, uint16_t layers, const char* fmt, uint8_t in, uint8_t out,
                                     uint32_t draws) {
  advance(f);
  ++eyePasses, upgraded += out > in ? 1u : 0u;
  ++hin[hbucket(in)], ++hout[hbucket(out)];
  if (!curDraws || draws > curDraws) curDraws = draws, curIn = in, curOut = out;
  Size* slot = nullptr;
  for (Size& s : sizes) {
    if (s.n && s.w == w && s.h == h && s.layers == layers && s.fmt == fmt) return void(++s.n);
    if (!s.n && !slot) slot = &s;
  }
  if (slot) *slot = {w, h, 1, layers, fmt};
}

bool MsaaTwins::EyeSummary::take(uint64_t f, char* buf, size_t cap) {
  if (!nextFrame) nextFrame = f + kEyePeriod;  // the period opens with the first pass
  if (f < nextFrame || lines >= kEyeLines || !(eyePasses || nonEye)) return false;
  advance(f);
  nextFrame = f + kEyePeriod;
  ++lines;
  const Size* top[3] = {nullptr, nullptr, nullptr};
  for (const Size& s : sizes) {
    if (!s.n) continue;
    const Size* x = &s;
    for (const Size*& t : top)
      if (x && (!t || x->n > t->n)) std::swap(t, x);
  }
  int len = std::snprintf(buf, cap,
                          "vk.msaa_eye f=%llu eye_color_passes=%u upgraded=%u samples_in_hist={1:%u,2:%u,4:%u%s%.0u} samples_out_hist={1:%u,2:%u,4:%u%s%.0u} "
                          "main_pass_samples=%u->%u main_pass_draws=%u non_eye_passes=%u (top3",
                          (unsigned long long)f, eyePasses, upgraded, hin[0], hin[1], hin[2], hin[3] ? ",o:" : "", hin[3], hout[0], hout[1],
                          hout[2], hout[3] ? ",o:" : "", hout[3], unsigned(mainIn), unsigned(mainOut), mainDraws, nonEye);
  for (const Size* t : top)
    if (t && len > 0 && size_t(len) < cap)
      len += std::snprintf(buf + len, cap - size_t(len), " %ux%ux%u %s x%u", t->w, t->h, unsigned(t->layers), t->fmt ? t->fmt : "?", t->n);
  if (len > 0 && size_t(len) < cap) std::snprintf(buf + len, cap - size_t(len), ")");
  const uint64_t keepNext = nextFrame, keepFrame = frame;
  const uint32_t keepLines = lines, keepMd = mainDraws;
  const uint8_t keepIn = mainIn, keepOut = mainOut;
  *this = EyeSummary{};
  nextFrame = keepNext, frame = keepFrame, lines = keepLines, mainDraws = keepMd, mainIn = keepIn, mainOut = keepOut;
  return true;
}

void noteDepthDraw(DepthDiag::Pass& p, const vpgpu::Draw& dr, const OpDraw& d) {
  ++p.draws;
  const bool bias = dr.depthBias != 0 || dr.depthBiasSlope != 0;
  p.bias += bias;
  if (bias) p.constMax = std::max(p.constMax, std::fabs(dr.depthBias)), p.slopeMax = std::max(p.slopeMax, std::fabs(dr.depthBiasSlope));
  if (d.depthTested && !d.depthWrite) {
    p.ge += d.depthOp == VK_COMPARE_OP_GREATER_OR_EQUAL;
    p.le += d.depthOp == VK_COMPARE_OP_LESS_OR_EQUAL;
    p.eq += d.depthOp == VK_COMPARE_OP_EQUAL;
  }
  p.d24 += d.depthD24;
  p.emul += d.pipe && d.pipe->d24Emul && !d.pipe->d24Skipped;
  p.skipped += d.pipe && d.pipe->d24Emul && d.pipe->d24Skipped;
}

void DepthDiag::notePass(const Pass& p) {
  ++periodPasses, ++sessionPasses, periodD24Passes += p.d24 != 0;
  for (Pass* t : {&period, &session}) {
    t->draws += p.draws, t->bias += p.bias, t->ge += p.ge, t->le += p.le, t->eq += p.eq, t->d24 += p.d24, t->emul += p.emul, t->skipped += p.skipped;
    t->constMax = std::max(t->constMax, p.constMax), t->slopeMax = std::max(t->slopeMax, p.slopeMax);
  }
  if (p.draws > best.draws) best = p;
}

bool DepthDiag::take(uint64_t f, char* buf, size_t cap) {
  if (!nextFrame) nextFrame = f + kPeriod;  // the period opens with the first pass
  if (f < nextFrame || lines >= kLines || !periodPasses) return false;
  nextFrame = f + kPeriod;
  ++lines;
  std::snprintf(buf, cap,
                "vk.depth_diag f=%llu emul=%d eye_passes=%u d24_passes=%u draws=%u bias_on=%u ge_nowrite=%u le_nowrite=%u eq_nowrite=%u "
                "d24_draws=%u emul_draws=%u bias_const_max=%.3f bias_slope_max=%.3f main_pass{draws=%u bias_on=%u ge_nowrite=%u "
                "le_nowrite=%u eq_nowrite=%u d24_draws=%u} session{passes=%u draws=%u bias_on=%u cmp_nowrite=%u d24_draws=%u} d24_skipped=%u",
                (unsigned long long)f, int(d24Emul()), periodPasses, periodD24Passes, period.draws, period.bias, period.ge, period.le,
                period.eq, period.d24, period.emul, double(period.constMax), double(period.slopeMax), best.draws, best.bias, best.ge,
                best.le, best.eq, best.d24, sessionPasses, session.draws, session.bias, session.ge + session.le + session.eq, session.d24, period.skipped);
  period = {}, best = {}, periodPasses = periodD24Passes = 0;
  return true;
}

static void logDepthDiag(Device* dev) {
  if (!dev->gpu || dev->depthDiag.lines >= DepthDiag::kLines) return;
  char buf[640];
  if (dev->depthDiag.take(dev->gpu->submittedFrames(), buf, sizeof buf)) log(std::string(buf));
}

static void logMsaaEye(Device* dev, MsaaTwins& mt) {
  if (!dev->gpu || mt.eye.lines >= MsaaTwins::kEyeLines) return;
  char buf[384];
  if (mt.eye.take(dev->gpu->submittedFrames(), buf, sizeof buf)) log(std::string(buf));
}

static void logMsaaMain(MsaaTwins& mt) {
  if (!mt.mainPass.rank || mt.mainLines >= MsaaTwins::kMainLines) return;
  if (++mt.mainTick & 31u) return;  // the clock is read on every 32nd command buffer, not on each one
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  MsaaTwins::MainPass m;
  if (!mt.takeMainLine(now, m)) return;
  logf("vk.msaa main force=%d samples=%u result=%s%s area=%ux%u eye=%d", msaaForce(), unsigned(m.samples), m.forced ? "up" : "rej:",
       m.forced ? "" : (m.why ? m.why : "none"), m.w, m.h, m.eye ? 1 : 0);
}

// "vk.msaa force=N passes= skip{...}" about once a second while forced MSAA is on; the counts are cumulative since the device was created.
static void logMsaaForce(MsaaTwins& mt) {
  const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (now < mt.nextLogNs) return;
  mt.nextLogNs = now + 1'000'000'000LL;
  const uint32_t skipped = mt.skipLoad + mt.skipInput + mt.skipDsb + mt.skipEds + mt.skipFmt + mt.skipSamples + mt.skipArea + mt.skipShape;
  const uint32_t rejected = mt.rejEye + mt.rejScene + mt.rejCd + mt.rejSmp + mt.rejStore + mt.rejSmask + mt.rejRes + mt.rejDesc;  // (rejN counts every pass while forcing is off)
  if (!mt.passes && !skipped && !rejected && !mt.gameAuto && !mt.skipUnreadable) return;
  logf("vk.msaa force=%d passes=%u up=%u skip{load=%u,input=%u,dsb=%u,eds=%u,fmt=%u,samples=%u,area=%u,shape=%u} twins=%zu game_auto=%u skip_unreadable=%u rej{n=%u,eye=%u,cd=%u,smp=%u,msstore=%u,smask=%u,res=%u,desc=%u} eye_why{no_depth=%u,no_eye_size=%u,size=%u,other=%u} scene{up=%u,rej=%u}",
       msaaForce(), mt.passes, mt.up, mt.skipLoad, mt.skipInput, mt.skipDsb, mt.skipEds, mt.skipFmt, mt.skipSamples, mt.skipArea,
       mt.skipShape, mt.twins.size(), mt.gameAuto, mt.skipUnreadable, mt.rejN, mt.rejEye, mt.rejCd, mt.rejSmp, mt.rejStore, mt.rejSmask, mt.rejRes, mt.rejDesc,
       mt.rejEyeBy[0], mt.rejEyeBy[1], mt.rejEyeBy[2], mt.rejEyeBy[3], mt.sceneUp, mt.rejScene);
}

vpgpu::TextureId UiMaskState::acquire(uint32_t slot, uint32_t w, uint32_t h, uint16_t layers, uint8_t samples) {
  Target& t = target[slot & 1u];
  if (samples < 1) samples = 1;
  if (!(t.tex && t.w == w && t.h == h && t.layers == layers)) {
    if (t.tex && dev) dev->gpu->destroy(t.tex);  // the backend retires it after the frames that use it
    if (t.ms && dev) dev->gpu->destroy(t.ms);
    t = Target{};
    if (!dev) return {};
    vpgpu::TextureDesc td;
    td.type = vpgpu::TextureType::Tex2D;
    td.format = vpgpu::Format::R8Unorm;
    td.width = w, td.height = h, td.layers = layers;
    td.usage = vpgpu::usage::sampled | vpgpu::usage::colorTarget;
    td.persistent = true;  // the TAA pass reads it after the frame, outside the frame compiler's view
    td.label = "vk.taa_uimask" + std::to_string(slot & 1u);
    t.tex = dev->gpu->createTexture(td);
    if (!t.tex) {
      logf("vk.err what=taa_uimask_texture size=%ux%ux%u", w, h, unsigned(layers));
      return {};
    }
    t.w = w, t.h = h, t.layers = layers;
    logf("vk.taa uimask texture slot=%u %ux%u layers=%u r8unorm (%.1f MiB)", slot & 1u, w, h, unsigned(layers),
         double(w) * h * layers / 1048576.0);
  }
  if (samples > 1 && (!t.ms || t.msSamples != samples)) {
    if (t.ms) dev->gpu->destroy(t.ms);
    t.ms = {}, t.msSamples = 0;
    vpgpu::TextureDesc td;
    td.type = vpgpu::TextureType::Tex2D;
    td.format = vpgpu::Format::R8Unorm;
    td.width = w, td.height = h, td.layers = layers, td.samples = samples;
    td.usage = vpgpu::usage::colorTarget;
    td.transient = true;  // only resolved into `tex` at the pass end: tile memory, never stored
    td.label = "vk.taa_uimask" + std::to_string(slot & 1u) + ".ms";
    t.ms = dev->gpu->createTexture(td);
    if (!t.ms) {
      logf("vk.err what=taa_uimask_texture_ms size=%ux%ux%u samples=%u", w, h, unsigned(layers), unsigned(samples));
      return {};
    }
    t.msSamples = samples;
    logf("vk.taa uimask texture_ms slot=%u samples=%u", slot & 1u, unsigned(samples));
  }
  return t.tex;
}

void UiMaskState::release() {
  for (Target& t : target) {
    if (t.tex && dev) dev->gpu->destroy(t.tex);
    if (t.ms && dev) dev->gpu->destroy(t.ms);
    t = Target{};
  }
  live[0] = live[1] = Live{};
  std::lock_guard<std::mutex> l(lastMu);
  for (Published& p : last) {
    if (p.tex && dev && dev->platform) dev->platform->releaseNative(p.tex);
    p = Published{};
  }
}

void translate(const Recorder& rec, vpgpu::Frame& frame, TranslateStats& st, const TaaJitter* taa, UiMaskState* um,
               ZeroLog* zl, FoveationState* fov, Device* device) {
  Translator t{frame, st, taa && taa->enabled ? taa : nullptr};
  t.device = device;
  t.zl = zl;
  t.fovs = t.taa && !t.taa->still ? fov : nullptr;  // eye passes are only identified with the jitter on; RCAS-only (still) frames never foveate
  t.zeroScanned = frame.commands().size();  // the frame may already hold earlier command buffers: those were scanned
  t.uiOn = t.taa && taaUiMaskOn();
  t.um = t.uiOn ? um : nullptr;
  TaaJitter eyeSnap;  // forced MSAA finds the eye passes by eye size / multiview also with the VPKTAA jitter off
  t.eye = t.taa;
  if (const Device* dv = t.devForce()) t.markedEye = dv->eyeSize.load(std::memory_order_relaxed);
  if (!t.eye && msaaForce() > 0) eyeSnap = taaJitter(), t.eye = &eyeSnap;
  t.run(rec);
  t.close();
  t.zeroScan();
  if (Device* dev = t.devForce()) logDepthDiag(dev);
  if (Device* dev = t.devForce(); dev && (msaaForce() > 0 || dev->msaaTwins.gameAuto || dev->msaaTwins.skipUnreadable)) {
    logMsaaForce(dev->msaaTwins);
    logMsaaMain(dev->msaaTwins);
    logMsaaEye(dev, dev->msaaTwins);
  }
}

}  // namespace vpvk
