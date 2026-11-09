// Resource model: handles, formats, descriptors and the table that owns them.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

// name, bytes per block, block width and height in texels, flags
#define VPGPU_FORMATS(X)                                                                         \
  X(Undefined, 0, 1, 1, 0)                                                                         \
  X(R8Unorm, 1, 1, 1, 0) X(R8Snorm, 1, 1, 1, 0) X(R8Uint, 1, 1, 1, I) X(R8Sint, 1, 1, 1, I)                 \
  X(RG8Unorm, 2, 1, 1, 0) X(RG8Snorm, 2, 1, 1, 0) X(RG8Uint, 2, 1, 1, I) X(RG8Sint, 2, 1, 1, I)             \
  X(RGBA8Unorm, 4, 1, 1, 0) X(RGBA8Srgb, 4, 1, 1, S) X(RGBA8Snorm, 4, 1, 1, 0) X(RGBA8Uint, 4, 1, 1, I)     \
  X(RGBA8Sint, 4, 1, 1, I) X(BGRA8Unorm, 4, 1, 1, 0) X(BGRA8Srgb, 4, 1, 1, S)                            \
  X(R16Unorm, 2, 1, 1, 0) X(R16Snorm, 2, 1, 1, 0) X(R16Uint, 2, 1, 1, I) X(R16Sint, 2, 1, 1, I)             \
  X(R16Float, 2, 1, 1, 0) X(RG16Unorm, 4, 1, 1, 0) X(RG16Snorm, 4, 1, 1, 0) X(RG16Uint, 4, 1, 1, I)         \
  X(RG16Sint, 4, 1, 1, I) X(RG16Float, 4, 1, 1, 0) X(RGBA16Unorm, 8, 1, 1, 0) X(RGBA16Snorm, 8, 1, 1, 0)    \
  X(RGBA16Uint, 8, 1, 1, I) X(RGBA16Sint, 8, 1, 1, I) X(RGBA16Float, 8, 1, 1, 0)                         \
  X(R32Uint, 4, 1, 1, I) X(R32Sint, 4, 1, 1, I) X(R32Float, 4, 1, 1, 0)                                  \
  X(RG32Uint, 8, 1, 1, I) X(RG32Sint, 8, 1, 1, I) X(RG32Float, 8, 1, 1, 0)                               \
  X(RGBA32Uint, 16, 1, 1, I) X(RGBA32Sint, 16, 1, 1, I) X(RGBA32Float, 16, 1, 1, 0)                      \
  X(RGB10A2Unorm, 4, 1, 1, 0) X(RGB10A2Uint, 4, 1, 1, I) X(RG11B10Float, 4, 1, 1, 0)                     \
  X(RGB9E5Float, 4, 1, 1, 0) X(B5G6R5Unorm, 2, 1, 1, 0) X(BGR5A1Unorm, 2, 1, 1, 0) X(ABGR4Unorm, 2, 1, 1, 0) \
  X(BC1, 8, 4, 4, 0) X(BC1Srgb, 8, 4, 4, S) X(BC2, 16, 4, 4, 0) X(BC2Srgb, 16, 4, 4, S)                     \
  X(BC3, 16, 4, 4, 0) X(BC3Srgb, 16, 4, 4, S) X(BC4Unorm, 8, 4, 4, 0) X(BC4Snorm, 8, 4, 4, 0)               \
  X(BC5Unorm, 16, 4, 4, 0) X(BC5Snorm, 16, 4, 4, 0) X(BC6HUfloat, 16, 4, 4, 0) X(BC6HSfloat, 16, 4, 4, 0)   \
  X(BC7, 16, 4, 4, 0) X(BC7Srgb, 16, 4, 4, S)                                                         \
  X(Etc2Rgb8, 8, 4, 4, 0) X(Etc2Rgb8Srgb, 8, 4, 4, S) X(Etc2Rgb8A1, 8, 4, 4, 0) X(Etc2Rgb8A1Srgb, 8, 4, 4, S) \
  X(Etc2Rgba8, 16, 4, 4, 0) X(Etc2Rgba8Srgb, 16, 4, 4, S) X(EacR11Unorm, 8, 4, 4, 0) X(EacR11Snorm, 8, 4, 4, 0) \
  X(EacRg11Unorm, 16, 4, 4, 0) X(EacRg11Snorm, 16, 4, 4, 0)                                          \
  X(Astc4x4, 16, 4, 4, 0) X(Astc4x4Srgb, 16, 4, 4, S) X(Astc6x6, 16, 6, 6, 0) X(Astc6x6Srgb, 16, 6, 6, S)   \
  X(Astc8x8, 16, 8, 8, 0) X(Astc8x8Srgb, 16, 8, 8, S) X(Astc10x10, 16, 10, 10, 0)                         \
  X(Astc10x10Srgb, 16, 10, 10, S) X(Astc12x12, 16, 12, 12, 0) X(Astc12x12Srgb, 16, 12, 12, S)               \
  X(D16Unorm, 2, 1, 1, D) X(D32Float, 4, 1, 1, D) X(D24UnormS8, 4, 1, 1, D | ST)                         \
  X(D32FloatS8, 8, 1, 1, D | ST) X(S8Uint, 1, 1, 1, ST) \
  /* appended: existing values stay stable (pipeline recipes store formats) */ \
  X(Astc5x4, 16, 5, 4, 0) X(Astc5x4Srgb, 16, 5, 4, S) \
  X(Astc5x5, 16, 5, 5, 0) X(Astc5x5Srgb, 16, 5, 5, S) \
  X(Astc6x5, 16, 6, 5, 0) X(Astc6x5Srgb, 16, 6, 5, S) \
  X(Astc8x5, 16, 8, 5, 0) X(Astc8x5Srgb, 16, 8, 5, S) \
  X(Astc8x6, 16, 8, 6, 0) X(Astc8x6Srgb, 16, 8, 6, S) \
  X(Astc10x5, 16, 10, 5, 0) X(Astc10x5Srgb, 16, 10, 5, S) \
  X(Astc10x6, 16, 10, 6, 0) X(Astc10x6Srgb, 16, 10, 6, S) \
  X(Astc10x8, 16, 10, 8, 0) X(Astc10x8Srgb, 16, 10, 8, S) \
  X(Astc12x10, 16, 12, 10, 0) X(Astc12x10Srgb, 16, 12, 10, S) \
  X(R8Srgb, 1, 1, 1, S) X(RG8Srgb, 2, 1, 1, S)

enum class Format : uint8_t {
#define VPGPU_X(name, bytes, bw, bh, flags) name,
  VPGPU_FORMATS(VPGPU_X)
#undef VPGPU_X
      Count
};

struct FormatInfo {
  enum Flag : uint8_t { D = 1, ST = 2, S = 4, I = 8 };  // depth, stencil, sRGB, integer
  const char* name;
  uint8_t bytesPerBlock;
  uint8_t blockSize;    // block width in texels (4 for BCn)
  uint8_t blockHeight;  // block height in texels (differs from the width only for ASTC 5x4, 8x6...)
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
  // Every GPU access to the buffer is named by the bindings of the draw / dispatch / copy that makes it (an address-only
  // Access for a raw pointer into it, e.g. the gpu/ps4 FLAT GuestMap): a backend may make it resident only for the work
  // that names it instead of always (gpu/metal2: out of the residency set, useResource per encoder).
  bool residentByUse = false;
};

enum class TextureType : uint8_t { Tex1D, Tex2D, Tex3D, Cube };  // arrays: layers > 1
namespace usage {
inline constexpr uint8_t sampled = 1, storage = 2, colorTarget = 4, depthTarget = 8;
// Viewed with other formats of the same texel size (Vulkan MUTABLE_FORMAT images).
inline constexpr uint8_t formatView = 16;
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
  // Multisampled textures: false when the client's usage rules out reading the samples other than by loading them as an
  // attachment (no SAMPLED / INPUT_ATTACHMENT / TRANSFER_SRC / STORAGE in Vulkan). The Vulkan translator then treats a
  // resolved attachment of it as Attachment::autoMsaa (VPVK_GAME_MSAA_AUTO). True = unknown / readable (the default).
  bool samplesReadable = true;
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
  float lodBias = 0.0f;    // added to the computed lod by the sampler state (GLES/PS4); no effect on the device
  float shaderLodBias = 0.0f;  // added to the lod by the shader (SamplerBias argument; Vulkan mipLodBias)
  bool normalized = true;  // false: texel coordinates (Vulkan unnormalizedCoordinates)
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
  bool destroy(TextureId id) {
    memValid_.erase(uint64_t{id.generation} << 32 | id.index);
    storeHistory_.erase(uint64_t{id.generation} << 32 | id.index);
    transientViolated_.erase(uint64_t{id.generation} << 32 | id.index);
    autoSrc_.erase(uint64_t{id.generation} << 32 | id.index);
    return textures_.destroy(id);
  }
  bool destroy(SamplerId id) { return samplers_.destroy(id); }
  const BufferDesc* get(BufferId id) const { return buffers_.get(id); }
  const TextureDesc* get(TextureId id) const { return textures_.get(id); }
  const SamplerDesc* get(SamplerId id) const { return samplers_.get(id); }

  // Whether a submitted frame left defined contents in the texture's memory (stored, resolved into,
  // copied or computed into, and not discarded since). A new texture holds none. The frame compiler
  // reads it to drop loads of undefined contents; Device::submit updates it (commitMemory).
  bool memValid(TextureId id) const { return memValid_.count(uint64_t{id.generation} << 32 | id.index) != 0; }
  void setMemValid(TextureId id, bool valid) {
    if (!get(id)) return;
    const uint64_t k = uint64_t{id.generation} << 32 | id.index;
    if (valid) memValid_.insert(k);
    else memValid_.erase(k);
  }

  // Dead-store history of an MSAA attachment the client Stores (CompileOptions::msaaDeadStore): `pending` = the last frame
  // that touched it stored samples nothing in that frame read; `dead` = consecutive such stores whose next frame did not
  // need the contents; `elided` = that last store was dropped; `banned` = a frame needed contents that were elided.
  // placeTiles reads it, commitMemory updates it.
  struct StoreHistory {
    uint8_t dead = 0;
    bool pending = false, elided = false, banned = false;
  };
  StoreHistory storeHistory(TextureId id) const {
    const auto it = storeHistory_.find(uint64_t{id.generation} << 32 | id.index);
    return it == storeHistory_.end() ? StoreHistory{} : it->second;
  }
  void setStoreHistory(TextureId id, StoreHistory h) {
    if (get(id)) storeHistory_[uint64_t{id.generation} << 32 | id.index] = h;
  }

  // A TextureDesc::transient texture that a frame used the way the contract forbids (its contents loaded, sampled, copied or
  // resolved into, see CompiledFrame::transientViolations). From then on the frame compiler treats it as not transient: an
  // explicit Store is kept (the texture gets memory lazily) instead of being dropped with the tile. commitMemory sets it.
  bool transientViolated(TextureId id) const { return transientViolated_.count(uint64_t{id.generation} << 32 | id.index) != 0; }
  void setTransientViolated(TextureId id) {
    if (get(id)) transientViolated_.insert(uint64_t{id.generation} << 32 | id.index);
  }

  // An MSAA texture whose last writer (up to the last committed frame) was an Attachment::autoMsaa slot: its samples were never
  // stored, the picture lives in `src` (the 1x image the writer resolved into). A later Load of the texture is seeded from it
  // (placeTiles); any other write forgets it. commitMemory sets it (CompiledFrame::autoAfter).
  struct AutoSource {
    TextureId src;
    uint16_t mip = 0, layer = 0;
  };
  const AutoSource* autoSource(TextureId id) const {
    const auto it = autoSrc_.find(uint64_t{id.generation} << 32 | id.index);
    return it == autoSrc_.end() ? nullptr : &it->second;
  }
  void setAutoSource(TextureId id, const AutoSource* s) {
    if (!get(id)) return;
    const uint64_t k = uint64_t{id.generation} << 32 | id.index;
    if (s) autoSrc_[k] = *s;
    else autoSrc_.erase(k);
  }

 private:
  std::unordered_set<uint64_t> memValid_;
  std::unordered_map<uint64_t, StoreHistory> storeHistory_;
  std::unordered_set<uint64_t> transientViolated_;
  std::unordered_map<uint64_t, AutoSource> autoSrc_;
  Pool<BufferDesc, BufferId> buffers_;
  Pool<TextureDesc, TextureId> textures_;
  Pool<SamplerDesc, SamplerId> samplers_;
};

}  // namespace vpgpu
