// Resource model: handles, formats, descriptors and the table that owns them.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vpgpu {

// Handles instead of pointers: clients and command lists stay valid to copy and compare,
// and a stale handle (generation mismatch) is detected instead of dereferenced.
template <class Tag>
struct Handle {
  uint32_t index = 0;
  uint32_t generation = 0;  // 0 = null handle
  explicit operator bool() const { return generation != 0; }
  friend bool operator==(Handle, Handle) = default;
};
using BufferId = Handle<struct BufferTag>;
using TextureId = Handle<struct TextureTag>;
using SamplerId = Handle<struct SamplerTag>;

// Pipeline stages at the granularity Metal 4 barriers express.
using StageMask = uint8_t;
namespace stage {
inline constexpr StageMask vertex = 1, fragment = 2, compute = 4, transfer = 8;
}

// name, bytes per block, block edge in texels, flags
#define VPGPU_FORMATS(X)                                                                         \
  X(Undefined, 0, 1, 0)                                                                         \
  X(R8Unorm, 1, 1, 0) X(R8Snorm, 1, 1, 0) X(R8Uint, 1, 1, I) X(R8Sint, 1, 1, I)                 \
  X(RG8Unorm, 2, 1, 0) X(RG8Snorm, 2, 1, 0) X(RG8Uint, 2, 1, I) X(RG8Sint, 2, 1, I)             \
  X(RGBA8Unorm, 4, 1, 0) X(RGBA8Srgb, 4, 1, S) X(RGBA8Snorm, 4, 1, 0) X(RGBA8Uint, 4, 1, I)     \
  X(RGBA8Sint, 4, 1, I) X(BGRA8Unorm, 4, 1, 0) X(BGRA8Srgb, 4, 1, S)                            \
  X(R16Unorm, 2, 1, 0) X(R16Snorm, 2, 1, 0) X(R16Uint, 2, 1, I) X(R16Sint, 2, 1, I)             \
  X(R16Float, 2, 1, 0) X(RG16Unorm, 4, 1, 0) X(RG16Snorm, 4, 1, 0) X(RG16Uint, 4, 1, I)         \
  X(RG16Sint, 4, 1, I) X(RG16Float, 4, 1, 0) X(RGBA16Unorm, 8, 1, 0) X(RGBA16Snorm, 8, 1, 0)    \
  X(RGBA16Uint, 8, 1, I) X(RGBA16Sint, 8, 1, I) X(RGBA16Float, 8, 1, 0)                         \
  X(R32Uint, 4, 1, I) X(R32Sint, 4, 1, I) X(R32Float, 4, 1, 0)                                  \
  X(RG32Uint, 8, 1, I) X(RG32Sint, 8, 1, I) X(RG32Float, 8, 1, 0)                               \
  X(RGBA32Uint, 16, 1, I) X(RGBA32Sint, 16, 1, I) X(RGBA32Float, 16, 1, 0)                      \
  X(RGB10A2Unorm, 4, 1, 0) X(RGB10A2Uint, 4, 1, I) X(RG11B10Float, 4, 1, 0)                     \
  X(RGB9E5Float, 4, 1, 0) X(B5G6R5Unorm, 2, 1, 0) X(BGR5A1Unorm, 2, 1, 0) X(ABGR4Unorm, 2, 1, 0) \
  X(BC1, 8, 4, 0) X(BC1Srgb, 8, 4, S) X(BC2, 16, 4, 0) X(BC2Srgb, 16, 4, S)                     \
  X(BC3, 16, 4, 0) X(BC3Srgb, 16, 4, S) X(BC4Unorm, 8, 4, 0) X(BC4Snorm, 8, 4, 0)               \
  X(BC5Unorm, 16, 4, 0) X(BC5Snorm, 16, 4, 0) X(BC6HUfloat, 16, 4, 0) X(BC6HSfloat, 16, 4, 0)   \
  X(BC7, 16, 4, 0) X(BC7Srgb, 16, 4, S)                                                         \
  X(D16Unorm, 2, 1, D) X(D32Float, 4, 1, D) X(D24UnormS8, 4, 1, D | ST)                         \
  X(D32FloatS8, 8, 1, D | ST) X(S8Uint, 1, 1, ST)

enum class Format : uint8_t {
#define VPGPU_X(name, bytes, block, flags) name,
  VPGPU_FORMATS(VPGPU_X)
#undef VPGPU_X
      Count
};

struct FormatInfo {
  enum Flag : uint8_t { D = 1, ST = 2, S = 4, I = 8 };  // depth, stencil, sRGB, integer
  const char* name;
  uint8_t bytesPerBlock;
  uint8_t blockSize;  // 4 for BCn
  uint8_t flags;
  bool depth() const { return flags & D; }
  bool stencil() const { return flags & ST; }
  bool srgb() const { return flags & S; }
  bool integer() const { return flags & I; }
};
const FormatInfo& formatInfo(Format);

struct BufferDesc {
  uint64_t size = 0;
  // When set, the buffer aliases this host memory (unified memory: guest memory is used in place,
  // no staging copy). It must stay mapped for the buffer's lifetime.
  void* guestMemory = nullptr;
  std::string label;
};

enum class TextureType : uint8_t { Tex1D, Tex2D, Tex3D, Cube };  // arrays: layers > 1
namespace usage {
inline constexpr uint8_t sampled = 1, storage = 2, colorTarget = 4, depthTarget = 8;
}

struct TextureDesc {
  TextureType type = TextureType::Tex2D;
  Format format = Format::Undefined;
  uint32_t width = 1, height = 1, depth = 1;
  uint16_t mips = 1;
  uint16_t layers = 1;  // cube: 6 per cube
  uint8_t samples = 1;
  uint8_t usage = usage::sampled;
  // Contents must outlive the frame (history buffers such as TAA, previous-frame depth): the
  // frame compiler never makes it tile-local. Anything read in a later frame must set it.
  bool persistent = false;
  // Opt-in: contents are never needed after the frame, so an explicit Store on it may be dropped
  // when the frame compiler keeps it on tile. Without it only DontCare/Resolve stores are released.
  bool transient = false;
  std::string label;
};
bool validate(const TextureDesc&);

enum class Filter : uint8_t { Nearest, Linear };
enum class MipFilter : uint8_t { None, Nearest, Linear };
enum class AddressMode : uint8_t { Repeat, MirrorRepeat, ClampToEdge, ClampToBorder, MirrorClampToEdge };
enum class CompareOp : uint8_t { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always };
enum class BorderColor : uint8_t { TransparentBlack, OpaqueBlack, OpaqueWhite };

struct SamplerDesc {
  Filter minFilter = Filter::Linear, magFilter = Filter::Linear;
  MipFilter mipFilter = MipFilter::Linear;
  AddressMode u = AddressMode::Repeat, v = AddressMode::Repeat, w = AddressMode::Repeat;
  uint8_t maxAnisotropy = 1;
  std::optional<CompareOp> compare;  // set for depth-comparison samplers
  BorderColor border = BorderColor::TransparentBlack;
  float lodMin = 0.0f, lodMax = 1000.0f;
  std::string label;
};

// Generational slot pool; freed slots are reused with a bumped generation.
template <class Desc, class Id>
class Pool {
 public:
  Id create(Desc desc) {
    uint32_t index;
    if (free_.empty()) {
      index = static_cast<uint32_t>(slots_.size());
      slots_.push_back({});
    } else {
      index = free_.back();
      free_.pop_back();
    }
    Slot& s = slots_[index];
    s.desc = std::move(desc);
    s.live = true;
    return {index, s.generation};
  }
  bool destroy(Id id) {
    if (!get(id)) return false;
    Slot& s = slots_[id.index];
    s.live = false;
    s.generation = s.generation == UINT32_MAX ? 1 : s.generation + 1;
    free_.push_back(id.index);
    return true;
  }
  const Desc* get(Id id) const {
    if (id.index >= slots_.size()) return nullptr;
    const Slot& s = slots_[id.index];
    return s.live && s.generation == id.generation ? &s.desc : nullptr;
  }

 private:
  struct Slot {
    Desc desc;
    uint32_t generation = 1;
    bool live = false;
  };
  std::vector<Slot> slots_;
  std::vector<uint32_t> free_;
};

class ResourceTable {
 public:
  BufferId createBuffer(BufferDesc d) { return d.size ? buffers_.create(std::move(d)) : BufferId{}; }
  TextureId createTexture(TextureDesc d) { return validate(d) ? textures_.create(std::move(d)) : TextureId{}; }
  SamplerId createSampler(SamplerDesc d) { return samplers_.create(std::move(d)); }
  bool destroy(BufferId id) { return buffers_.destroy(id); }
  bool destroy(TextureId id) { return textures_.destroy(id); }
  bool destroy(SamplerId id) { return samplers_.destroy(id); }
  const BufferDesc* get(BufferId id) const { return buffers_.get(id); }
  const TextureDesc* get(TextureId id) const { return textures_.get(id); }
  const SamplerDesc* get(SamplerId id) const { return samplers_.get(id); }

 private:
  Pool<BufferDesc, BufferId> buffers_;
  Pool<TextureDesc, TextureId> textures_;
  Pool<SamplerDesc, SamplerId> samplers_;
};

}  // namespace vpgpu
