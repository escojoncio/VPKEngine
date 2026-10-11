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
  enum class Kind : uint8_t { Buffer, Texture } kind = Kind::Buffer;
  uint32_t index = 0, generation = 0;
  uint64_t begin = 0, end = 0;                    // buffer bytes [begin, end)
  uint16_t mipBegin = 0, mipEnd = 0;              // texture mips [mipBegin, mipEnd)
  uint16_t layerBegin = 0, layerEnd = 0;          // texture layers [layerBegin, layerEnd)

  static ResourceRange bytes(BufferId b, uint64_t offset, uint64_t size);
  static ResourceRange wholeBuffer(BufferId b) { return bytes(b, 0, UINT64_MAX); }
  static ResourceRange subresource(TextureId t, uint16_t mip, uint16_t mipCount, uint16_t layer,
                                   uint16_t layerCount);
  static ResourceRange wholeTexture(TextureId t) { return subresource(t, 0, UINT16_MAX, 0, UINT16_MAX); }

  bool sameResource(const ResourceRange& o) const {
    return kind == o.kind && index == o.index && generation == o.generation;
  }
  bool overlaps(const ResourceRange& o) const;
  bool contains(const ResourceRange& o) const;
};

struct Access {
  ResourceRange range;
  StageMask stages = 0;
  bool write = false;
};
inline Access read(ResourceRange r, StageMask s) { return {r, s, false}; }
inline Access write(ResourceRange r, StageMask s) { return {r, s, true}; }

enum class LoadAction : uint8_t { DontCare, Load, Clear };
enum class StoreAction : uint8_t { DontCare, Store };

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

struct IndexBuffer {
  BufferId buffer;
  uint64_t offset = 0;
  bool wide = false;  // 32-bit indices, else 16-bit
};

enum class Primitive : uint8_t { Points, Lines, LineStrip, Triangles, TriangleStrip };

struct Draw {
  PipelineKey pipeline;
  std::vector<Access> bindings;  // vertex buffers, constants, textures, storage
  Primitive primitive = Primitive::Triangles;
  uint32_t first = 0, count = 0, instances = 1;  // first/count are indices when indexed
  int32_t baseVertex = 0;
  std::optional<IndexBuffer> index;
};

struct Dispatch {
  PipelineKey pipeline;
  std::vector<Access> bindings;
  std::array<uint32_t, 3> groups{1, 1, 1};
  std::string label;
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
