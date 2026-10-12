#include "commands.h"

#include <bit>

namespace vpvk {

vpgpu::LoadAction toLoad(VkAttachmentLoadOp op) {
  switch (op) {
    case VK_ATTACHMENT_LOAD_OP_LOAD: return vpgpu::LoadAction::Load;
    case VK_ATTACHMENT_LOAD_OP_CLEAR: return vpgpu::LoadAction::Clear;
    default: return vpgpu::LoadAction::DontCare;
  }
}

vpgpu::StoreAction toStore(VkAttachmentStoreOp op, bool resolve) {
  const bool store = op == VK_ATTACHMENT_STORE_OP_STORE;
  if (resolve) return store ? vpgpu::StoreAction::StoreAndResolve : vpgpu::StoreAction::Resolve;
  // DontCare / None keep the attachment on tile: the core then never writes it to memory.
  return store ? vpgpu::StoreAction::Store : vpgpu::StoreAction::DontCare;
}

namespace {

vpgpu::ClearValue toClear(const VkClearValue& v, bool depth) {
  vpgpu::ClearValue c;
  if (depth) {
    c.depth = v.depthStencil.depth;
    c.stencil = static_cast<uint8_t>(v.depthStencil.stencil);
  } else {
    for (int i = 0; i < 4; ++i) c.color[i] = v.color.float32[i];
  }
  return c;
}

bool attach(const OpBeginPass& b, uint32_t index, uint32_t resolveIndex, vpgpu::Attachment& out) {
  if (!b.framebuffer || index >= b.framebuffer->views.size()) return false;
  const ImageView* v = b.framebuffer->views[index];
  if (!v || !v->image || !v->image->id) return false;
  const VkAttachmentDescription2& d = b.pass->attachments[index];
  const bool depthFmt = vpgpu::formatInfo(v->image->desc.format).depth() ||
                        vpgpu::formatInfo(v->image->desc.format).stencil();
  out.texture = v->image->id;
  out.mip = static_cast<uint16_t>(v->range.baseMipLevel);
  out.layer = static_cast<uint16_t>(v->range.baseArrayLayer);
  out.load = toLoad(d.loadOp);
  bool resolve = false;
  if (resolveIndex != VK_ATTACHMENT_UNUSED && resolveIndex < b.framebuffer->views.size()) {
    const ImageView* r = b.framebuffer->views[resolveIndex];
    if (r && r->image && r->image->id) {
      out.resolve = r->image->id;
      out.resolveMip = static_cast<uint16_t>(r->range.baseMipLevel);
      out.resolveLayer = static_cast<uint16_t>(r->range.baseArrayLayer);
      resolve = true;
    }
  }
  out.store = toStore(d.storeOp, resolve);
  if (index < b.clears.size()) out.clear = toClear(b.clears[index], depthFmt);
  return true;
}

// Subpass 0 of a render pass as one core pass. Multi-subpass passes (input attachments) become
// several core passes with pixel-local reads: phase 2b.
bool beginPass(const OpBeginPass& b, vpgpu::RenderPassDesc& desc) {
  if (!b.pass || b.pass->subpasses.empty()) return false;
  const RenderPassObj::Subpass& sp = b.pass->subpasses[0];
  int slot = 0;
  for (size_t i = 0; i < sp.colors.size() && slot < vpgpu::kMaxColorAttachments; ++i, ++slot) {
    const uint32_t a = sp.colors[i].attachment;
    if (a == VK_ATTACHMENT_UNUSED) continue;
    const uint32_t r = i < sp.resolves.size() ? sp.resolves[i].attachment : VK_ATTACHMENT_UNUSED;
    if (!attach(b, a, r, desc.colors[slot])) return false;
  }
  if (sp.hasDepth && sp.depth.attachment != VK_ATTACHMENT_UNUSED &&
      !attach(b, sp.depth.attachment, VK_ATTACHMENT_UNUSED, desc.depth))
    return false;
  // Multiview: one layer per view bit; the backend renders them with vertex amplification.
  desc.layerCount = sp.viewMask ? static_cast<uint16_t>(std::popcount(sp.viewMask))
                                : static_cast<uint16_t>(b.framebuffer ? b.framebuffer->layers : 1);
  desc.label = "vk.pass";
  return true;
}

}  // namespace

void translate(const Recorder& rec, vpgpu::Frame& frame, TranslateStats& st) {
  bool passOpen = false;
  for (const Op& op : rec.ops) {
    if (const auto* b = std::get_if<OpBeginPass>(&op)) {
      vpgpu::RenderPassDesc desc;
      if (beginPass(*b, desc)) {
        frame.beginRenderPass(std::move(desc));
        passOpen = true;
        ++st.passes;
      } else {
        ++st.dropped;
      }
    } else if (std::holds_alternative<OpEndPass>(op)) {
      if (passOpen) frame.endRenderPass();
      passOpen = false;
    } else if (const auto* c = std::get_if<OpCopyBuffer>(&op)) {
      if (!c->src->id || !c->dst->id) { ++st.dropped; continue; }
      frame.copyBuffer({c->src->id, c->region.srcOffset, c->region.size},
                       {c->dst->id, c->region.dstOffset, c->region.size});
      ++st.copies;
    } else if (const auto* f = std::get_if<OpFillBuffer>(&op)) {
      if (!f->dst->id) { ++st.dropped; continue; }
      const VkDeviceSize size = f->size == VK_WHOLE_SIZE ? f->dst->size - f->offset : f->size;
      const uint8_t byte = static_cast<uint8_t>(f->data);
      // The core fills bytes; a word whose four bytes differ needs a compute fill (phase 3).
      if (f->data != byte * 0x01010101u) { ++st.dropped; continue; }
      frame.fillBuffer({f->dst->id, f->offset, size}, byte);
      ++st.copies;
    } else if (const auto* cc = std::get_if<OpClearColor>(&op)) {
      if (!cc->image->id) { ++st.dropped; continue; }
      const uint32_t mips = cc->range.levelCount == VK_REMAINING_MIP_LEVELS
                                ? cc->image->desc.mips - cc->range.baseMipLevel : cc->range.levelCount;
      const uint32_t layers = cc->range.layerCount == VK_REMAINING_ARRAY_LAYERS
                                  ? cc->image->desc.layers - cc->range.baseArrayLayer : cc->range.layerCount;
      for (uint32_t m = 0; m < mips; ++m)
        for (uint32_t l = 0; l < layers; ++l) {
          vpgpu::RenderPassDesc d;
          vpgpu::Attachment& a = d.colors[0];
          a.texture = cc->image->id;
          a.mip = static_cast<uint16_t>(cc->range.baseMipLevel + m);
          a.layer = static_cast<uint16_t>(cc->range.baseArrayLayer + l);
          a.load = vpgpu::LoadAction::Clear;
          a.store = vpgpu::StoreAction::Store;
          for (int i = 0; i < 4; ++i) a.clear.color[i] = cc->color.float32[i];
          d.label = "vk.clear";
          frame.clear(std::move(d));
          ++st.clears;
        }
    } else if (const auto* bi = std::get_if<OpCopyBufferToImage>(&op)) {
      if (!bi->src->id || !bi->dst->id) { ++st.dropped; continue; }
      const vpgpu::FormatInfo& fi = vpgpu::formatInfo(bi->dst->desc.format);
      const VkBufferImageCopy& r = bi->region;
      const uint32_t rowTexels = r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width;
      const uint32_t rows = r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height;
      vpgpu::Copy c;
      c.kind = vpgpu::CopyKind::BufferToTexture;
      c.srcBuffer.buffer = bi->src->id;
      c.srcBuffer.offset = r.bufferOffset;
      c.srcBuffer.bytesPerRow = (rowTexels + fi.blockSize - 1) / fi.blockSize * fi.bytesPerBlock;
      c.srcBuffer.rowsPerImage = (rows + fi.blockSize - 1) / fi.blockSize;
      c.srcBuffer.size = uint64_t(c.srcBuffer.bytesPerRow) * c.srcBuffer.rowsPerImage *
                         r.imageExtent.depth * r.imageSubresource.layerCount;
      c.dstTexture.texture = bi->dst->id;
      c.dstTexture.mip = static_cast<uint16_t>(r.imageSubresource.mipLevel);
      c.dstTexture.layer = static_cast<uint16_t>(r.imageSubresource.baseArrayLayer);
      c.dstTexture.layerCount = static_cast<uint16_t>(r.imageSubresource.layerCount);
      c.dstTexture.x = static_cast<uint32_t>(r.imageOffset.x);
      c.dstTexture.y = static_cast<uint32_t>(r.imageOffset.y);
      c.dstTexture.z = static_cast<uint32_t>(r.imageOffset.z);
      c.dstTexture.width = r.imageExtent.width;
      c.dstTexture.height = r.imageExtent.height;
      c.dstTexture.depth = r.imageExtent.depth;
      c.label = "vk.upload";
      frame.copy(std::move(c));
      ++st.copies;
    } else if (std::holds_alternative<OpDraw>(op)) {
      ++st.skippedDraws;
    } else if (std::holds_alternative<OpDispatch>(op)) {
      ++st.skippedDispatches;
    }
  }
  if (passOpen) frame.endRenderPass();
}

}  // namespace vpvk
