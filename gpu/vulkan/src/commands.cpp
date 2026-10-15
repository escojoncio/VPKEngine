#include "commands.h"

#include <algorithm>
#include <chrono>
#include <atomic>

#include <bit>
#include <cstring>

namespace vpvk {

vpgpu::LoadAction toLoad(VkAttachmentLoadOp op) {
  switch (op) {
    case VK_ATTACHMENT_LOAD_OP_LOAD: return vpgpu::LoadAction::Load;
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
    if (r && r->image && r->image->id && v->image->forcedFrom > 1) {
      // Game MSAA forced to 1x: keep drawing into the (now 1x) source like every other pass
      // does, store it, and copy it into the resolve target after this subpass.
      out.store = vpgpu::StoreAction::Store;
      if (index < b.clears.size()) out.clear = toClear(b.clears[index], depthFmt, cf);
      if (forced) forced->push_back({v, r});
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
  out.store = int(sub) == u.last && !(mask && !fullViewMask(mask))
                  ? toStore(storeOpOf(d, fi), resolve)
                  : (resolve ? vpgpu::StoreAction::StoreAndResolve : vpgpu::StoreAction::Store);
  if (partial && !(v->image && v->image->desc.transient)) {  // pixels outside the area survive
    if (out.store == vpgpu::StoreAction::DontCare) out.store = vpgpu::StoreAction::Store;
    if (out.store == vpgpu::StoreAction::Resolve) out.store = vpgpu::StoreAction::StoreAndResolve;
  }
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
  auto wholeClear = [&](uint32_t a) {
    return int(sub) == uses(*b.pass, a).first && b.pass->attachments[a].initialLayout == VK_IMAGE_LAYOUT_UNDEFINED;
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

// Subpass `sub` of a render pass as one core pass. Input attachments are read as textures (they
// were stored by the earlier subpass); passes that the core merges keep them on tile.
bool beginPass(const OpBeginPass& b, uint32_t sub, vpgpu::RenderPassDesc& desc,
               std::vector<vpgpu::Copy>* resolveCopies = nullptr) {
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
  if (selfReads(*b.pass, sub)) {
    auto keep = [](vpgpu::Attachment& a) {
      if (!a.texture) return;
      if (a.store == vpgpu::StoreAction::DontCare) a.store = vpgpu::StoreAction::Store;
      if (a.store == vpgpu::StoreAction::Resolve) a.store = vpgpu::StoreAction::StoreAndResolve;
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
// Returns the size of the target (texels) through w/h when it qualifies.
bool taaEyePass(const TaaJitter& j, bool hasDepth, uint32_t viewMask, const VkRect2D& area, uint32_t fbW, uint32_t fbH,
                uint32_t depthW, uint32_t depthH, uint32_t& w, uint32_t& h) {
  if (!hasDepth) return false;
  w = depthW ? depthW : fbW ? fbW : area.extent.width;
  h = depthW ? depthH : fbW ? fbH : area.extent.height;
  const bool multiview = std::popcount(viewMask) >= 2;
  if (!j.width || !j.height) return multiview;
  auto is = [&](uint32_t x, uint32_t y) { return x == j.width && y == j.height; };
  if (is(area.extent.width, area.extent.height) || is(fbW, fbH) || is(depthW, depthH)) return true;
  // A multiview pass of another size is still the eye's scene when it is a render-scaled copy of
  // it (50-100 % per axis); multiview shadow/probe passes of other sizes are not, and would also
  // misreport the target size the resolve scales the jitter by.
  return multiview && w * 2 >= j.width && w <= j.width && h * 2 >= j.height && h <= j.height;
}

struct Translator {
  vpgpu::Frame& frame;
  TranslateStats& st;
  const TaaJitter* taa = nullptr;  // enabled jitter, or null
  bool taaPass = false;            // the open pass is an eye pass (taaEyePass)
  uint32_t taaW = 0, taaH = 0;     // its target size
  bool passOpen = false;
  vpgpu::RenderPassDesc current;  // the open core pass (reopened with Load after a split)
  const OpBeginPass* pass = nullptr;
  uint32_t subpass = 0;
  uint32_t viewMask = 0;  // multiview mask of the open (sub)pass: clears cover every view
  std::vector<vpgpu::Copy> pendingResolves;  // forced-1x resolves of the open (sub)pass

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
    // An eye pass keeps its depth: VPKTAA reads it after the frame, outside anything the frame
    // compiler can see, so a DONT_CARE store must not let the depth live only in tile memory.
    if (taaPass && d.depth.texture) {
      if (d.depth.store == vpgpu::StoreAction::DontCare) d.depth.store = vpgpu::StoreAction::Store;
      if (d.depth.store == vpgpu::StoreAction::Resolve) d.depth.store = vpgpu::StoreAction::StoreAndResolve;
    }
    current = d;
    frame.beginRenderPass(std::move(d));
    passOpen = true;
    ++st.passes;
  }
  void close() {
    if (passOpen) frame.endRenderPass();
    passOpen = false;
  }

  void run(const Recorder& rec) {
    for (const Op& op : rec.ops) {
      if (const auto* b = std::get_if<OpBeginPass>(&op)) {
        pass = b;
        subpass = 0;
        vpgpu::RenderPassDesc desc;
        pendingResolves.clear();
        taaSubpass(*b, 0);
        viewMask = b->pass && !b->pass->subpasses.empty() ? b->pass->subpasses[0].viewMask : 0;
        if (beginPass(*b, 0, desc, &pendingResolves)) {
          auto cr = areaClears(*b, 0, desc);
          resolveClear(*b, 0, desc.layerCount);
          open(std::move(desc));
          clearQuads(std::move(cr));
        } else {
          ++st.dropped;
        }
      } else if (const auto* br = std::get_if<OpBeginRendering>(&op)) {
        pass = nullptr;
        viewMask = br->viewMask;
        pendingResolves = br->resolveCopies;
        taaPass = taa && taaEyePass(*taa, br->depthWidth != 0, br->viewMask, br->area, 0, 0, br->depthWidth,
                                    br->depthHeight, taaW, taaH);
        open(br->desc);
        clearQuads(br->areaClear);
      } else if (std::holds_alternative<OpNextSubpass>(op)) {
        close();
        flushResolves();
        ++subpass;
        vpgpu::RenderPassDesc desc;
        if (pass) taaSubpass(*pass, subpass);
        viewMask = pass && pass->pass && subpass < pass->pass->subpasses.size() ? pass->pass->subpasses[subpass].viewMask : 0;
        if (pass && beginPass(*pass, subpass, desc, &pendingResolves)) {
          auto cr = areaClears(*pass, subpass, desc);
          resolveClear(*pass, subpass, desc.layerCount);
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
          close();
          open(std::move(d));
          ++st.selfDependencySplits;
        }
      } else if (std::holds_alternative<OpEndPass>(op)) {
        close();
        flushResolves();
        pass = nullptr;
        taaPass = false;
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
            a.store = vpgpu::StoreAction::Store;
            a.clear.color = clearColor(cc->color, cc->image->desc.format);
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
            d.depth.store = vpgpu::StoreAction::Store;
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
        if (cr->colorMask || cr->depth || cr->stencil) clearQuads(std::move(cr));
      } else if (const auto* bi = std::get_if<OpCopyBufferToImage>(&op)) {
        linearSync(bi->dst, true);
        bufferImage(bi->src, bi->dst, bi->region, true);
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
        draw(*d);
      } else if (const auto* x = std::get_if<OpDispatch>(&op)) {
        if (!x->ok) { ++st.skippedDispatches; continue; }
        vpgpu::Dispatch dp = x->dispatch;
        if (x->indirect) {
          const uint8_t* p = indirectData(x->indirect, x->indirectOffset, 12);
          if (!p) { ++st.skippedDispatches; continue; }
          std::memcpy(dp.groups.data(), p, 12);
        }
        if (!dp.indirect && (!dp.groups[0] || !dp.groups[1] || !dp.groups[2])) continue;
        frame.dispatch(std::move(dp));
        ++st.dispatches;
      } else if (const auto* e = std::get_if<OpExecute>(&op)) {
        if (e->secondary) run(*e->secondary);
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
    c.label = "vk.blit";
    frame.copy(std::move(c));
    ++st.copies;
  }

  // Eye-pass test for subpass `sub` of a render pass (see taaEyePass).
  void taaSubpass(const OpBeginPass& b, uint32_t sub) {
    taaPass = false;
    if (!taa || !b.pass || !b.framebuffer || sub >= b.pass->subpasses.size()) return;
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
    taaPass = taaEyePass(*taa, depth, sp.viewMask, b.area, b.framebuffer->width, b.framebuffer->height, dw, dh,
                         taaW, taaH);
  }

  // Moves the viewport of an eye-pass geometry draw by the jitter; the scissor stays (it bounds
  // pixels, not geometry). A fullscreen pass (no index buffer, at most 6 vertices, no vertex
  // buffer: a quad/triangle made from the vertex id) is left alone: its texels must not move.
  void jitter(vpgpu::Draw& dr, const OpDraw& d) {
    if (!taaPass) return;
    const bool fullscreen = (!dr.index || d.fanConverted) && dr.count <= 6 && !d.vertexBuffers;
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

  static const uint8_t* indirectData(const Buffer* b, VkDeviceSize offset, VkDeviceSize size) {
    if (!b || !b->memory || !b->memory->host || offset + size > b->size) return nullptr;
    return b->memory->host + b->offset + offset;
  }

  // A query ends: in a multiview pass it is viewCount consecutive queries (Vulkan), the first
  // holding the count of every view and the rest 0 (they were reset). Timestamps: the host time
  // at submission (monotonic, nanoseconds = timestampPeriod 1), written now into the counter.
  void queryEnd(const OpQueryEnd& q) {
    QueryPoolObj* p = q.pool;
    if (!p || q.query >= p->count) return;
    uint32_t n = 1;
    if (passOpen && viewMask) n = std::min<uint32_t>(uint32_t(std::popcount(viewMask)), p->count - q.query);
    if (q.timestamp && p->base != UINT32_MAX && p->device && p->device->visibility.host) {
      static std::atomic<uint64_t> last{0};
      uint64_t now = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()).count());
      // Strictly increasing and never 0, even for two writes within one clock tick.
      uint64_t prev = last.load();
      while (!last.compare_exchange_weak(prev, std::max(now, prev + 1))) {}
      now = std::max(now, prev + 1);
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

  void draw(const OpDraw& d) {
    if (!d.ok || !passOpen) {
      ++st.skippedDraws;
      return;
    }
    if (!d.indirect) {
      if (taaPass) {
        vpgpu::Draw dr = d.draw;
        jitter(dr, d);
        frame.draw(std::move(dr));
      } else {
        frame.draw(d.draw);
      }
      ++st.draws;
      return;
    }
    const bool indexed = d.draw.index.has_value();
    const uint32_t size = indexed ? 20 : 16;
    if (d.gpuIndirect) {  // the GPU reads the records (pipelines.cpp recordIndirectDraw)
      for (uint32_t i = 0; i < d.indirectCount; ++i) {
        vpgpu::Draw dr = d.draw;
        dr.indirect = vpgpu::Indirect{d.gpuBuffer, d.gpuOffset + uint64_t(i) * d.indirectStride};
        dr.drawIndex = i;
        jitter(dr, d);
        frame.draw(std::move(dr));
        ++st.draws;
      }
      return;
    }
    for (uint32_t i = 0; i < d.indirectCount; ++i) {
      const uint8_t* p = indirectData(d.indirect, d.indirectOffset + VkDeviceSize(i) * d.indirectStride, size);
      if (!p) {
        ++st.skippedDraws;
        continue;
      }
      vpgpu::Draw dr = d.draw;
      dr.drawIndex = i;
      if (indexed) {
        VkDrawIndexedIndirectCommand c;
        std::memcpy(&c, p, sizeof c);
        dr.count = c.indexCount, dr.instances = c.instanceCount * d.viewCount, dr.first = c.firstIndex;
        dr.baseVertex = c.vertexOffset, dr.baseInstance = c.firstInstance * d.viewCount;
      } else {
        VkDrawIndirectCommand c;
        std::memcpy(&c, p, sizeof c);
        dr.count = c.vertexCount, dr.instances = c.instanceCount * d.viewCount, dr.first = c.firstVertex;
        dr.baseInstance = c.firstInstance * d.viewCount;
      }
      if (!dr.count || !dr.instances) continue;
      jitter(dr, d);
      frame.draw(std::move(dr));
      ++st.draws;
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
      a.resolve = rs.dst->id;
      a.resolveMip = uint16_t(g.dstSubresource.mipLevel);
      a.resolveLayer = uint16_t(g.dstSubresource.baseArrayLayer + l);
      d.label = "vk.resolve";
      frame.beginRenderPass(std::move(d));
      frame.endRenderPass();
      ++st.copies;
    }
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
    frame.copy(std::move(c));
    ++st.copies;
  }
};

}  // namespace

void translate(const Recorder& rec, vpgpu::Frame& frame, TranslateStats& st, const TaaJitter* taa) {
  Translator t{frame, st, taa && taa->enabled ? taa : nullptr};
  t.run(rec);
  t.close();
}

}  // namespace vpvk
