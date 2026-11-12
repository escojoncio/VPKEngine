#include "vpgpu/core/commands.h"

#include <algorithm>

namespace vpgpu {

namespace {
uint64_t saturatingEnd(uint64_t begin, uint64_t size) { return size > UINT64_MAX - begin ? UINT64_MAX : begin + size; }
uint32_t saturatingEnd(uint32_t begin, uint32_t size) { return size > UINT32_MAX - begin ? UINT32_MAX : begin + size; }
uint16_t saturatingEnd(uint16_t begin, uint16_t count) {
  return static_cast<uint16_t>(std::min<uint32_t>(uint32_t{begin} + count, UINT16_MAX));
}
}  // namespace

ResourceRange ResourceRange::bytes(BufferId b, uint64_t offset, uint64_t size) {
  ResourceRange r;
  r.kind = Kind::Buffer;
  r.index = b.index;
  r.generation = b.generation;
  r.begin = offset;
  r.end = saturatingEnd(offset, size);
  return r;
}

ResourceRange ResourceRange::subresource(TextureId t, uint16_t mip, uint16_t mipCount, uint16_t layer,
                                         uint16_t layerCount) {
  ResourceRange r;
  r.kind = Kind::Texture;
  r.index = t.index;
  r.generation = t.generation;
  r.mipBegin = mip;
  r.mipEnd = saturatingEnd(mip, mipCount);
  r.layerBegin = layer;
  r.layerEnd = saturatingEnd(layer, layerCount);
  return r;
}

ResourceRange ResourceRange::sampler(SamplerId s) {
  ResourceRange r;
  r.kind = Kind::Sampler;
  r.index = s.index;
  r.generation = s.generation;
  return r;
}

BindKind bindKind(const Access& a) {
  if (a.kind != BindKind::Auto) return a.kind;
  switch (a.range.kind) {
    case ResourceRange::Kind::Buffer: return BindKind::Buffer;
    case ResourceRange::Kind::Texture: return a.write ? BindKind::StorageTexture : BindKind::Texture;
    case ResourceRange::Kind::Sampler: return BindKind::Sampler;
  }
  return BindKind::Buffer;
}

int fetchIndex(const RenderPassDesc& pass, const ResourceRange& r) {
  if (r.kind != ResourceRange::Kind::Texture) return -1;
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    const Attachment& a = pass.colors[i];
    if (a.texture && a.texture.index == r.index && a.texture.generation == r.generation && a.mip == r.mipBegin &&
        a.layer == r.layerBegin)
      return i;
  }
  return -1;
}

bool ResourceRange::overlaps(const ResourceRange& o) const {
  if (!sameResource(o)) return false;
  if (kind == Kind::Sampler) return true;
  if (kind == Kind::Buffer) return begin < o.end && o.begin < end;
  if (!(mipBegin < o.mipEnd && o.mipBegin < mipEnd && layerBegin < o.layerEnd && o.layerBegin < layerEnd)) return false;
  if (rect && o.rect) return x0 < o.x1 && o.x0 < x1 && y0 < o.y1 && o.y0 < y1;
  return true;
}

bool ResourceRange::contains(const ResourceRange& o) const {
  if (!sameResource(o)) return false;
  if (kind == Kind::Sampler) return true;
  if (kind == Kind::Buffer) return begin <= o.begin && o.end <= end;
  if (!(mipBegin <= o.mipBegin && o.mipEnd <= mipEnd && layerBegin <= o.layerBegin && o.layerEnd <= layerEnd)) return false;
  if (!rect) return true;
  return o.rect && x0 <= o.x0 && o.x1 <= x1 && y0 <= o.y0 && o.y1 <= y1;
}

bool RenderPassDesc::sameTargets(const RenderPassDesc& o) const {
  if (layerCount != o.layerCount || !depth.sameTarget(o.depth)) return false;
  if (width != o.width || height != o.height || samples != o.samples) return false;
  if ((rateMap ? rateMap->key : 0) != (o.rateMap ? o.rateMap->key : 0)) return false;
  for (int i = 0; i < kMaxColorAttachments; ++i)
    if (!colors[i].sameTarget(o.colors[i])) return false;
  return true;
}

std::vector<Access> attachmentAccesses(const RenderPassDesc& d) {
  std::vector<Access> out;
  auto add = [&](const Attachment& a) {
    if (a.texture) out.push_back(write(ResourceRange::subresource(a.texture, a.mip, 1, a.layer, d.layerCount), stage::fragment));
    if (a.resolve && (a.store == StoreAction::Resolve || a.store == StoreAction::StoreAndResolve))
      out.push_back(write(ResourceRange::subresource(a.resolve, a.resolveMip, 1, a.resolveLayer, d.layerCount), stage::fragment));
  };
  for (const Attachment& a : d.colors) add(a);
  add(d.depth);
  return out;
}

std::vector<Access> drawAccesses(const Draw& d) {
  std::vector<Access> out = d.bindings;
  if (d.index) {
    const uint64_t stride = d.index->wide ? 4 : 2;
    const uint64_t skip = uint64_t{d.first} * stride;
    out.push_back(read(ResourceRange::bytes(d.index->buffer, saturatingEnd(d.index->offset, skip), uint64_t{d.count} * stride),
                       stage::vertex));
  }
  if (d.indirect) out.push_back(read(ResourceRange::bytes(d.indirect->buffer, d.indirect->offset, d.index ? 20 : 16), stage::vertex));
  if (d.visibility) out.push_back(write(ResourceRange::bytes(d.visibility->buffer, d.visibility->offset, 8), stage::fragment));
  return out;
}

std::vector<Access> copyAccesses(const Copy& c, bool rects) {
  auto buf = [](const BufferSpan& s) { return ResourceRange::bytes(s.buffer, s.offset, s.size); };
  // A 3D texture's depth planes count as its layers (attachments name a plane by `layer`): a copy of
  // planes z..z+depth covers those layers, so clears/draws into single planes order against it.
  // 2D spans have z = 0 and depth 1 (or depth = layerCount for 2D <-> 3D copies).
  const bool sameTexture = (c.kind == CopyKind::TextureToTexture || c.kind == CopyKind::TextureBlit) &&
                           c.srcTexture.texture.index == c.dstTexture.texture.index &&
                           c.srcTexture.texture.generation == c.dstTexture.texture.generation;
  auto tex = [&](const TextureSpan& s) {
    ResourceRange r = ResourceRange::subresource(s.texture, s.mip, 1, uint16_t(s.layer + s.z),
                                                 uint16_t(std::max<uint32_t>(s.layerCount, std::max(1u, s.depth))));
    if (rects && !sameTexture && s.width && s.height) {
      r.rect = true;
      r.x0 = s.x, r.y0 = s.y;
      r.x1 = saturatingEnd(s.x, s.width), r.y1 = saturatingEnd(s.y, s.height);
    }
    return r;
  };
  switch (c.kind) {
    case CopyKind::BufferToBuffer: return {read(buf(c.srcBuffer), stage::transfer), write(buf(c.dstBuffer), stage::transfer)};
    case CopyKind::TextureToTexture:
      if (c.mipChain > 1)  // the chain: generateMipmaps reads mip 0 (up to the source mip) and writes mips 1 up, all layers
        return {read(ResourceRange::subresource(c.srcTexture.texture, 0, uint16_t(c.srcTexture.mip + 1), 0, UINT16_MAX), stage::transfer),
                write(ResourceRange::subresource(c.dstTexture.texture, 1, uint16_t(c.mipChain - 1), 0, UINT16_MAX), stage::transfer)};
      return {read(tex(c.srcTexture), stage::transfer), write(tex(c.dstTexture), stage::transfer)};
    case CopyKind::TextureBlit: return {read(tex(c.srcTexture), stage::transfer), write(tex(c.dstTexture), stage::transfer)};
    case CopyKind::BufferToTexture: return {read(buf(c.srcBuffer), stage::transfer), write(tex(c.dstTexture), stage::transfer)};
    case CopyKind::TextureToBuffer: return {read(tex(c.srcTexture), stage::transfer), write(buf(c.dstBuffer), stage::transfer)};
    case CopyKind::FillBuffer: return {write(buf(c.dstBuffer), stage::transfer)};
  }
  return {};
}

void Frame::beginRenderPass(RenderPassDesc desc) {
  commands_.emplace_back(RenderPass{std::move(desc), {}});
  inPass_ = true;
}

bool Frame::draw(Draw d) {
  if (!inPass_) return false;
  std::get<RenderPass>(commands_.back()).draws.push_back(std::move(d));
  return true;
}

void Frame::dispatch(Dispatch d) {
  inPass_ = false;
  commands_.emplace_back(std::move(d));
}

void Frame::copy(Copy c) {
  inPass_ = false;
  commands_.emplace_back(std::move(c));
}

void Frame::copyBuffer(BufferSpan src, BufferSpan dst) {
  Copy c;
  c.kind = CopyKind::BufferToBuffer;
  c.srcBuffer = src;
  c.dstBuffer = dst;
  copy(std::move(c));
}

void Frame::fillBuffer(BufferSpan dst, uint8_t value) {
  Copy c;
  c.kind = CopyKind::FillBuffer;
  c.dstBuffer = dst;
  c.fillByte = value;
  copy(std::move(c));
}

void Frame::clear(RenderPassDesc desc) {
  for (Attachment& a : desc.colors) a.load = LoadAction::Clear;
  desc.depth.load = LoadAction::Clear;
  beginRenderPass(std::move(desc));
  endRenderPass();
}

}  // namespace vpgpu
