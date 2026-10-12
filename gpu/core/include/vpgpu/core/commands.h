// Frame recording: what the client asks the GPU to do, in submission order.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "vpgpu/core/resources.h"

namespace vpgpu {

struct PipelineKey {
  uint64_t value = 0;
  friend bool operator==(PipelineKey, PipelineKey) = default;
};

// A byte range of a buffer or a mip x layer box of a texture. Hazards are tracked per range, so
// writing mip 0 does not order against sampling mip 1, nor two disjoint halves of one buffer.
struct ResourceRange {
  enum class Kind : uint8_t { Buffer, Texture, Sampler } kind = Kind::Buffer;
  uint32_t index = 0, generation = 0;
  uint64_t begin = 0, end = 0;                    // buffer bytes [begin, end)
  uint16_t mipBegin = 0, mipEnd = 0;              // texture mips [mipBegin, mipEnd)
  uint16_t layerBegin = 0, layerEnd = 0;          // texture layers [layerBegin, layerEnd)

  static ResourceRange bytes(BufferId b, uint64_t offset, uint64_t size);
  static ResourceRange wholeBuffer(BufferId b) { return bytes(b, 0, UINT64_MAX); }
  static ResourceRange subresource(TextureId t, uint16_t mip, uint16_t mipCount, uint16_t layer,
                                   uint16_t layerCount);
  static ResourceRange wholeTexture(TextureId t) { return subresource(t, 0, UINT16_MAX, 0, UINT16_MAX); }
  static ResourceRange sampler(SamplerId s);

  bool sameResource(const ResourceRange& o) const {
    return kind == o.kind && index == o.index && generation == o.generation;
  }
  bool overlaps(const ResourceRange& o) const;
  bool contains(const ResourceRange& o) const;
};

// Argument-table kind of a binding. The backend writes one 8-byte entry per non-pixel-local binding,
// sorted by (kind, slot) in this order: the MSL emitter's VpArgs contract. Auto derives the kind
// from the range (buffer -> Buffer, texture -> Texture or StorageTexture if written, sampler).
enum class BindKind : uint8_t { Auto, Buffer, ConstantBuffer, Texture, Sampler, StorageTexture };

struct Access {
  ResourceRange range;
  StageMask stages = 0;
  bool write = false;
  // Same pixel, same sample, no offset or filtering, of a colour attachment of the draw's own
  // pass. Not bound: the shader reads it as [[color(fetchIndex)]]. Lets the frame compiler keep
  // the pass in the encoder that wrote the texture and the texture on tile.
  bool pixelLocal = false;
  BindKind kind = BindKind::Auto;
  uint16_t slot = 0;  // VPIR slot index inside its kind
};

// A texture binding (kind, slot) that binds a backend view of its texture instead (other type,
// format or mip/layer subset); hazards stay on the binding's range. gpu/metal2 only.
struct ViewBinding {
  BindKind kind = BindKind::Texture;
  uint16_t slot = 0;
  TextureId view;
};
inline Access read(ResourceRange r, StageMask s) { return {r, s, false}; }
inline Access write(ResourceRange r, StageMask s) { return {r, s, true}; }
inline Access pixelLocalRead(TextureId t, uint16_t mip = 0, uint16_t layer = 0) {
  Access a{ResourceRange::subresource(t, mip, 1, layer, 1), StageMask{2} /* fragment */, false};
  a.pixelLocal = true;
  return a;
}
inline Access samplerBinding(SamplerId s, StageMask st, uint16_t slot = 0) {
  Access a{ResourceRange::sampler(s), st, false};
  a.kind = BindKind::Sampler;
  a.slot = slot;
  return a;
}
BindKind bindKind(const Access&);  // Auto resolved

enum class LoadAction : uint8_t { DontCare, Load, Clear };
// Resolve: MSAA samples are averaged into Attachment::resolve and then discarded.
enum class StoreAction : uint8_t { DontCare, Store, Resolve, StoreAndResolve };

struct ClearValue {
  std::array<float, 4> color{};
  float depth = 1.0f;
  uint8_t stencil = 0;
};

struct Attachment {
  TextureId texture;  // null = unused slot
  uint16_t mip = 0, layer = 0;
  LoadAction load = LoadAction::Load;
  StoreAction store = StoreAction::Store;
  ClearValue clear;
  TextureId resolve;  // target of a Resolve / StoreAndResolve store
  uint16_t resolveMip = 0, resolveLayer = 0;
  bool sameTarget(const Attachment& o) const {
    return texture == o.texture && mip == o.mip && layer == o.layer;
  }
};

inline constexpr int kMaxColorAttachments = 8;

struct RenderPassDesc {
  std::array<Attachment, kMaxColorAttachments> colors{};
  Attachment depth;        // depth and/or stencil, per the texture's format
  uint16_t layerCount = 1;  // layered rendering (e.g. both eyes in one pass)
  std::string label;
  bool sameTargets(const RenderPassDesc& o) const;
};

// Colour attachment index of a pixel-local read in `pass` (the [[color(n)]] the shader reads), or -1
// if the range is not one of its colour attachments. Passes merged into one encoder keep their
// attachment slots, so this is also the index inside the merged encoder.
int fetchIndex(const RenderPassDesc& pass, const ResourceRange&);

struct IndexBuffer {
  BufferId buffer;
  uint64_t offset = 0;
  bool wide = false;  // 32-bit indices, else 16-bit
};

enum class Primitive : uint8_t { Points, Lines, LineStrip, Triangles, TriangleStrip };

// Viewport transform (pixels; depth range znear..zfar for NDC z 0..1). width 0: leave the encoder's
// current one (the whole attachment at the start of an encoder).
struct Viewport {
  float x = 0, y = 0, width = 0, height = 0, znear = 0, zfar = 1;
};
// Scissor rectangle in pixels, inside the attachment (clients clamp). width 0: none set.
struct Scissor {
  uint32_t x = 0, y = 0, width = 0, height = 0;
};
enum class Winding : uint8_t { CounterClockwise, Clockwise };  // which winding is front-facing

// Arguments of an indirect draw or dispatch, read by the GPU when it gets to the command:
// draw {count, instances, first, baseInstance}, indexed draw {count, instances, first, baseVertex,
// baseInstance}, dispatch {x, y, z} (32-bit each, as Metal and the PS4 lay them out).
struct Indirect {
  BufferId buffer;
  uint64_t offset = 0;
};

struct Draw {
  PipelineKey pipeline;
  std::vector<Access> bindings;  // vertex buffers, constants, textures, storage
  Primitive primitive = Primitive::Triangles;
  uint32_t first = 0, count = 0, instances = 1;  // first/count are indices when indexed
  int32_t baseVertex = 0;
  uint32_t baseInstance = 0;  // first instance id (gpu/metal2 honours it; gpu/metal draws from 0)
  std::optional<IndexBuffer> index;
  uint8_t stencilRef = 0;
  std::array<float, 4> blendColor{};  // for BlendFactor::Constant / OneMinusConstant
  Viewport viewport;
  Scissor scissor;
  Winding frontFace = Winding::CounterClockwise;
  // Depth bias (shadow maps) and depth clamp. gpu/metal2 applies them; gpu/metal ignores them.
  float depthBias = 0, depthBiasSlope = 0, depthBiasClamp = 0;
  bool depthClamp = false;
  std::vector<ViewBinding> views;
  // Set: first/count/instances/baseVertex come from the buffer (count stays the upper bound of the
  // index range read, for hazards). gpu/metal only.
  std::optional<Indirect> indirect;
};

struct Dispatch {
  PipelineKey pipeline;
  std::vector<Access> bindings;
  std::array<uint32_t, 3> groups{1, 1, 1};
  std::string label;
  std::vector<ViewBinding> views;
  std::optional<Indirect> indirect;  // set: the group counts come from the buffer (gpu/metal only)
};

struct BufferSpan {
  BufferId buffer;
  uint64_t offset = 0, size = 0;
  uint32_t bytesPerRow = 0, rowsPerImage = 0;  // only for buffer<->texture copies
};

struct TextureSpan {
  TextureId texture;
  uint16_t mip = 0, layer = 0, layerCount = 1;
  uint32_t x = 0, y = 0, z = 0, width = 0, height = 0, depth = 1;
};

enum class CopyKind : uint8_t { BufferToBuffer, TextureToTexture, BufferToTexture, TextureToBuffer, FillBuffer };

// Copies and fills: the work Metal 4 records in a compute encoder next to dispatches.
struct Copy {
  CopyKind kind = CopyKind::BufferToBuffer;
  BufferSpan srcBuffer, dstBuffer;
  TextureSpan srcTexture, dstTexture;
  uint8_t fillByte = 0;
  std::string label;
};

struct RenderPass {
  RenderPassDesc desc;
  std::vector<Draw> draws;
};

using Command = std::variant<RenderPass, Dispatch, Copy>;

// Accesses implied by a command beyond its explicit bindings.
std::vector<Access> attachmentAccesses(const RenderPassDesc&);
std::vector<Access> drawAccesses(const Draw&);
std::vector<Access> copyAccesses(const Copy&);

// One frame of client work in submission order.
class Frame {
 public:
  void beginRenderPass(RenderPassDesc desc);
  bool draw(Draw d);  // false outside a render pass
  void endRenderPass() { inPass_ = false; }
  void dispatch(Dispatch d);
  void copy(Copy c);
  void copyBuffer(BufferSpan src, BufferSpan dst);
  void fillBuffer(BufferSpan dst, uint8_t value);
  // A texture clear is a render pass with clear loads and no draws: it can then merge with or
  // be absorbed by the passes around it.
  void clear(RenderPassDesc desc);

  const std::vector<Command>& commands() const { return commands_; }

 private:
  std::vector<Command> commands_;
  bool inPass_ = false;
};

}  // namespace vpgpu
