#include "commands.h"

#include <algorithm>

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

bool attach(const OpBeginPass& b, uint32_t sub, uint32_t index, uint32_t resolveIndex, vpgpu::Attachment& out,
            std::vector<std::pair<const ImageView*, const ImageView*>>* forced = nullptr) {
  if (!b.framebuffer || index >= b.framebuffer->views.size()) return false;
  const ImageView* v = b.framebuffer->views[index];
  if (!v || !v->image || !v->image->id) return false;
  const VkAttachmentDescription2& d = b.pass->attachments[index];
  const bool depthFmt = vpgpu::formatInfo(v->image->desc.format).depth() ||
                        vpgpu::formatInfo(v->image->desc.format).stencil();
  const UseRange u = uses(*b.pass, index);
  out.texture = v->image->id;
  out.mip = static_cast<uint16_t>(v->range.baseMipLevel);
  out.layer = static_cast<uint16_t>(v->range.baseArrayLayer);
  out.load = int(sub) == u.first ? toLoad(depthFmt && !vpgpu::formatInfo(v->image->desc.format).depth() ? d.stencilLoadOp : d.loadOp)
                                 : vpgpu::LoadAction::Load;
  bool resolve = false;
  if (resolveIndex != VK_ATTACHMENT_UNUSED && resolveIndex < b.framebuffer->views.size()) {
    const ImageView* r = b.framebuffer->views[resolveIndex];
    if (r && r->image && r->image->id && v->image->forcedFrom > 1) {
      // Game MSAA forced to 1x: keep drawing into the (now 1x) source like every other pass
      // does, store it, and copy it into the resolve target after this subpass.
      out.store = vpgpu::StoreAction::Store;
      if (index < b.clears.size()) out.clear = toClear(b.clears[index], depthFmt, v->image->desc.format);
      if (forced) forced->push_back({v, r});
      return true;
    }
    if (r && r->image && r->image->id) {
      out.resolve = r->image->id;
      out.resolveMip = static_cast<uint16_t>(r->range.baseMipLevel);
      out.resolveLayer = static_cast<uint16_t>(r->range.baseArrayLayer);
      resolve = true;
    }
  }
  out.store = int(sub) == u.last ? toStore(d.storeOp, resolve)
                                 : (resolve ? vpgpu::StoreAction::StoreAndResolve : vpgpu::StoreAction::Store);
  if (index < b.clears.size()) out.clear = toClear(b.clears[index], depthFmt, v->image->desc.format);
  return true;
}

// The clears of subpass `sub`'s first uses that load ops cannot express, moved to a quad over the
// render area (the attachments then load): a render area smaller than the attachment, and depth
// and stencil of one image where one is cleared and the other loaded. Null when there are none.
std::shared_ptr<vpgpu::ClearRects> areaClears(const OpBeginPass& b, uint32_t sub, vpgpu::RenderPassDesc& desc) {
  if (!b.pass || !b.framebuffer || sub >= b.pass->subpasses.size()) return nullptr;
  const RenderPassObj::Subpass& sp = b.pass->subpasses[sub];
  auto cr = std::make_shared<vpgpu::ClearRects>();
  auto view = [&](uint32_t index) -> const ImageView* {
    if (index == VK_ATTACHMENT_UNUSED || index >= b.framebuffer->views.size()) return nullptr;
    const ImageView* v = b.framebuffer->views[index];
    return v && v->image && v->image->id ? v : nullptr;
  };
  int slot = 0;
  for (size_t i = 0; i < sp.colors.size() && slot < vpgpu::kMaxColorAttachments; ++i, ++slot) {
    const uint32_t a = sp.colors[i].attachment;
    const ImageView* v = view(a);
    if (!v || desc.colors[slot].load != vpgpu::LoadAction::Clear || !partialArea(b.area, v)) continue;
    desc.colors[slot].load = vpgpu::LoadAction::Load;
    cr->colorMask |= uint8_t(1u << slot);
    if (a < b.clears.size())
      for (int c = 0; c < 4; ++c) cr->colorBits[slot][c] = b.clears[a].color.uint32[c];
  }
  if (sp.hasDepth) {
    const uint32_t a = sp.depth.attachment;
    const ImageView* v = view(a);
    if (v && int(sub) == uses(*b.pass, a).first) {
      const vpgpu::FormatInfo& fi = vpgpu::formatInfo(v->image->desc.format);
      const VkAttachmentDescription2& d = b.pass->attachments[a];
      const bool clearD = fi.depth() && d.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
      const bool clearS = fi.stencil() && d.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
      // LOAD from an UNDEFINED layout reads undefined contents: as cheap as DONT_CARE (and keeps a
      // cleared depth on tile instead of loading it for the other aspect).
      const bool undef = d.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED;
      const bool loadD = fi.depth() && d.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && !undef;
      const bool loadS = fi.stencil() && d.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD && !undef;
      const bool partial = partialArea(b.area, v);
      if ((clearD || clearS) && (partial || loadD || loadS)) {
        desc.depth.load = vpgpu::LoadAction::Load;
        cr->depth = clearD;
        cr->stencil = clearS;
        if (a < b.clears.size()) {
          cr->depthValue = b.clears[a].depthStencil.depth;
          cr->stencilValue = uint8_t(b.clears[a].depthStencil.stencil);
        }
      } else if (clearD || clearS) {
        desc.depth.load = vpgpu::LoadAction::Clear;  // the other aspect is DONT_CARE
      } else if (loadD || loadS) {
        desc.depth.load = vpgpu::LoadAction::Load;
      }
    }
  }
  if (!cr->colorMask && !cr->depth && !cr->stencil) return nullptr;
  cr->rects.push_back({uint32_t(std::max(0, b.area.offset.x)), uint32_t(std::max(0, b.area.offset.y)),
                       b.area.extent.width, b.area.extent.height, 0, std::max<uint16_t>(desc.layerCount, 1)});
  return cr;
}

// Subpass `sub` of a render pass as one core pass. Input attachments are read as textures (they
// were stored by the earlier subpass); passes that the core merges keep them on tile.
bool beginPass(const OpBeginPass& b, uint32_t sub, vpgpu::RenderPassDesc& desc,
               std::vector<vpgpu::Copy>* resolveCopies = nullptr) {
  if (!b.pass || sub >= b.pass->subpasses.size()) return false;
  const RenderPassObj::Subpass& sp = b.pass->subpasses[sub];
  std::vector<std::pair<const ImageView*, const ImageView*>> forced;
  int slot = 0;
  for (size_t i = 0; i < sp.colors.size() && slot < vpgpu::kMaxColorAttachments; ++i, ++slot) {
    const uint32_t a = sp.colors[i].attachment;
    if (a == VK_ATTACHMENT_UNUSED) continue;
    const uint32_t r = i < sp.resolves.size() ? sp.resolves[i].attachment : VK_ATTACHMENT_UNUSED;
    if (!attach(b, sub, a, r, desc.colors[slot], &forced)) return false;
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
  // Multiview: one layer per view bit (the draws are instanced once per view).
  desc.layerCount = sp.viewMask ? static_cast<uint16_t>(std::bit_width(sp.viewMask))  // view = layer = bit index
                                : static_cast<uint16_t>(b.framebuffer ? b.framebuffer->layers : 1);
  desc.label = "vk.pass";
  if (resolveCopies)
    for (const auto& [src, dst] : forced) resolveCopies->push_back(forcedResolveCopy(src, dst, std::max<uint16_t>(desc.layerCount, 1)));
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

  // In-pass clear quads (vkCmdClearAttachments, area clears) into the open pass.
  void clearQuads(std::shared_ptr<const vpgpu::ClearRects> cr) {
    if (!cr || !passOpen || cr->rects.empty()) return;
    vpgpu::Draw dr;
    dr.clear = std::move(cr);
    frame.draw(std::move(dr));
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
          open(std::move(desc));
          clearQuads(std::move(cr));
        } else {
          ++st.dropped;
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
        for (uint32_t m = 0; m < mipsOf(cc->image, cc->range); ++m)
          for (uint32_t l = 0; l < layersOf(cc->image, cc->range); ++l) {
            vpgpu::RenderPassDesc d;
            vpgpu::Attachment& a = d.colors[0];
            a.texture = cc->image->id;
            a.mip = static_cast<uint16_t>(cc->range.baseMipLevel + m);
            a.layer = static_cast<uint16_t>(cc->range.baseArrayLayer + l);
            a.load = vpgpu::LoadAction::Clear;
            a.store = vpgpu::StoreAction::Store;
            a.clear.color = clearColor(cc->color, cc->image->desc.format);
            d.label = "vk.clear";
            frame.clear(std::move(d));
            ++st.clears;
          }
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
          // Multiview: the rect's layers are 0/1 and the clear covers every view of the pass.
          q.layer = viewMask ? 0 : uint16_t(r.baseArrayLayer);
          q.layerCount = viewMask ? passLayers : uint16_t(std::max(1u, r.layerCount));
          cr->rects.push_back(q);
        }
        if (cr->colorMask || cr->depth || cr->stencil) clearQuads(std::move(cr));
      } else if (const auto* bi = std::get_if<OpCopyBufferToImage>(&op)) {
        bufferImage(bi->src, bi->dst, bi->region, true);
      } else if (const auto* ib = std::get_if<OpCopyImageToBuffer>(&op)) {
        bufferImage(ib->dst, ib->src, ib->region, false);
      } else if (const auto* ci = std::get_if<OpCopyImage>(&op)) {
        if (!ci->src->id || !ci->dst->id) { ++st.dropped; continue; }
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
          continue;
        }
        c.srcTexture = {ci->src->id, uint16_t(r.srcSubresource.mipLevel), uint16_t(r.srcSubresource.baseArrayLayer),
                        uint16_t(r.srcSubresource.layerCount), uint32_t(r.srcOffset.x), uint32_t(r.srcOffset.y),
                        uint32_t(r.srcOffset.z), r.extent.width, r.extent.height, r.extent.depth};
        c.dstTexture = {ci->dst->id, uint16_t(r.dstSubresource.mipLevel), uint16_t(r.dstSubresource.baseArrayLayer),
                        uint16_t(r.dstSubresource.layerCount), uint32_t(r.dstOffset.x), uint32_t(r.dstOffset.y),
                        uint32_t(r.dstOffset.z), r.extent.width, r.extent.height, r.extent.depth};
        c.label = "vk.copy_image";
        frame.copy(std::move(c));
        ++st.copies;
      } else if (const auto* bl = std::get_if<OpBlitImage>(&op)) {
        if (!bl->src->id || !bl->dst->id) { ++st.dropped; continue; }
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
        if (!sw || !sh || !dw || !dh) continue;
        const uint16_t n = uint16_t(std::min(layers(r.srcSubresource, bl->src), layers(r.dstSubresource, bl->dst)));
        c.srcTexture = {bl->src->id, uint16_t(r.srcSubresource.mipLevel), uint16_t(r.srcSubresource.baseArrayLayer), n,
                        sx, sy, 0, sw, sh, 1};
        c.dstTexture = {bl->dst->id, uint16_t(r.dstSubresource.mipLevel), uint16_t(r.dstSubresource.baseArrayLayer), n,
                        dx, dy, 0, dw, dh, 1};
        c.blitFilter = bl->filter == VK_FILTER_NEAREST ? vpgpu::Filter::Nearest : vpgpu::Filter::Linear;
        c.blitFlipX = sfx != dfx;
        c.blitFlipY = sfy != dfy;
        c.label = "vk.blit";
        frame.copy(std::move(c));
        ++st.copies;
      } else if (const auto* rs = std::get_if<OpResolveImage>(&op)) {
        resolveImage(*rs);
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
        if (!dp.groups[0] || !dp.groups[1] || !dp.groups[2]) continue;
        frame.dispatch(std::move(dp));
        ++st.dispatches;
      } else if (const auto* e = std::get_if<OpExecute>(&op)) {
        if (e->secondary) run(*e->secondary);
      } else if (const auto* qr = std::get_if<OpQueryReset>(&op)) {
        queryReset(*qr);
      } else if (const auto* qe = std::get_if<OpQueryEnd>(&op)) {
        st.queries.push_back({qe->pool, qe->query, 1, true});
      } else if (const auto* qc = std::get_if<OpQueryCopy>(&op)) {
        queryCopy(*qc);
      } else if (const auto* ev = std::get_if<OpEvent>(&op)) {
        if (ev->event) st.events.push_back({ev->event, ev->set});
      }
    }
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
    const bool fullscreen = !dr.index && dr.count <= 6 && !d.vertexBuffers;
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

  // Occlusion counters back to 0 (GPU fill, ordered by the core like any copy).
  void queryReset(const OpQueryReset& q) {
    QueryPoolObj* p = q.pool;
    if (!p || p->base == UINT32_MAX || !p->device || !p->device->visibility.id || !q.count) return;
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
    const bool backed = p->type == VK_QUERY_TYPE_OCCLUSION && p->base != UINT32_MAX && p->device &&
                        p->device->visibility.id;
    auto copy = [&](vpgpu::BufferId src, uint64_t srcOff, uint64_t dstOff) {
      vpgpu::Copy c;
      c.kind = vpgpu::CopyKind::BufferToBuffer;
      c.srcBuffer = {src, srcOff, size};
      c.dstBuffer = {q.dst->id, dstOff, size};
      c.label = "vk.query_copy";
      frame.copy(std::move(c));
      ++st.copies;
    };
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
    for (uint32_t i = 0; i < d.indirectCount; ++i) {
      const uint8_t* p = indirectData(d.indirect, d.indirectOffset + VkDeviceSize(i) * d.indirectStride, size);
      if (!p) {
        ++st.skippedDraws;
        continue;
      }
      vpgpu::Draw dr = d.draw;
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
  // and stores it resolved into the destination (Metal resolves at the end of the pass). Metal
  // resolves the whole subresource: a region smaller than it resolves more than asked (logged).
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
    // a single-sample one of the same size (integer formats have no average; depth goes through
    // render-pass resolves). Anything else is skipped, named once in the log.
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(rs.src->desc.format);
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
    if (g.srcOffset.x || g.srcOffset.y || g.dstOffset.x || g.dstOffset.y || g.extent.width < sw || g.extent.height < sh) {
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 4)
        logf("vk.resolve_image partial region %ux%u of %ux%u: the whole subresource is resolved", g.extent.width,
             g.extent.height, sw, sh);
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
    bs.rowsPerImage = (rows + fi.blockSize - 1) / fi.blockSize;
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
