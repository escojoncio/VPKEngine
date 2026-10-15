// Metal backend. Classic Metal 3 command encoding (MTLCommandQueue, render/compute/blit encoders,
// MTLFence) with Metal 3 bindless (gpuAddress / gpuResourceID argument tables) and a residency set.
// All resources are hazard-untracked: ordering comes only from the barriers the frame compiler
// placed, so Metal adds no implicit synchronization of its own.
// Render targets that are not persistent are allocated on first use that needs memory; attachments
// the frame compiler kept on tile use cached memoryless textures instead (Apple GPUs).
// Metal 4 replacement points are marked "M4:".
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#include "vpgpu/metal2/metal_backend.h"

#include <TargetConditionals.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vpgpu/core/pipeline_cache.h"

namespace vpgpu::metal2 {

void ShaderTable::add(uint64_t shaderId, MslShader shader) {
  std::lock_guard lock(mutex_);
  shaders_[shaderId] = std::move(shader);
}

bool ShaderTable::find(uint64_t shaderId, MslShader& out) const {
  std::lock_guard lock(mutex_);
  auto it = shaders_.find(shaderId);
  if (it == shaders_.end()) return false;
  out = it->second;
  return true;
}

namespace {

constexpr MTLResourceOptions kSharedUntracked = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
constexpr uint64_t kArgRingBytes = 1 << 20;   // first argument buffer of a frame slot
constexpr uint64_t kArgChunkMax = 16ull << 20;  // largest chained argument buffer
constexpr uint64_t kArgAlign = 256;           // safe offset alignment for buffer bindings on every GPU
constexpr NSUInteger kMaxSamples = 2048;      // timestamp samples per frame slot (2 per sub-encoder)

MTLPixelFormat pixelFormat(Format f) {
  switch (f) {
    case Format::Undefined: return MTLPixelFormatInvalid;
    case Format::R8Unorm: return MTLPixelFormatR8Unorm;
    case Format::R8Snorm: return MTLPixelFormatR8Snorm;
    case Format::R8Uint: return MTLPixelFormatR8Uint;
    case Format::R8Sint: return MTLPixelFormatR8Sint;
    case Format::RG8Unorm: return MTLPixelFormatRG8Unorm;
    case Format::RG8Snorm: return MTLPixelFormatRG8Snorm;
    case Format::RG8Uint: return MTLPixelFormatRG8Uint;
    case Format::RG8Sint: return MTLPixelFormatRG8Sint;
    case Format::RGBA8Unorm: return MTLPixelFormatRGBA8Unorm;
    case Format::RGBA8Srgb: return MTLPixelFormatRGBA8Unorm_sRGB;
    case Format::RGBA8Snorm: return MTLPixelFormatRGBA8Snorm;
    case Format::RGBA8Uint: return MTLPixelFormatRGBA8Uint;
    case Format::RGBA8Sint: return MTLPixelFormatRGBA8Sint;
    case Format::BGRA8Unorm: return MTLPixelFormatBGRA8Unorm;
    case Format::BGRA8Srgb: return MTLPixelFormatBGRA8Unorm_sRGB;
    case Format::R16Unorm: return MTLPixelFormatR16Unorm;
    case Format::R16Snorm: return MTLPixelFormatR16Snorm;
    case Format::R16Uint: return MTLPixelFormatR16Uint;
    case Format::R16Sint: return MTLPixelFormatR16Sint;
    case Format::R16Float: return MTLPixelFormatR16Float;
    case Format::RG16Unorm: return MTLPixelFormatRG16Unorm;
    case Format::RG16Snorm: return MTLPixelFormatRG16Snorm;
    case Format::RG16Uint: return MTLPixelFormatRG16Uint;
    case Format::RG16Sint: return MTLPixelFormatRG16Sint;
    case Format::RG16Float: return MTLPixelFormatRG16Float;
    case Format::RGBA16Unorm: return MTLPixelFormatRGBA16Unorm;
    case Format::RGBA16Snorm: return MTLPixelFormatRGBA16Snorm;
    case Format::RGBA16Uint: return MTLPixelFormatRGBA16Uint;
    case Format::RGBA16Sint: return MTLPixelFormatRGBA16Sint;
    case Format::RGBA16Float: return MTLPixelFormatRGBA16Float;
    case Format::R32Uint: return MTLPixelFormatR32Uint;
    case Format::R32Sint: return MTLPixelFormatR32Sint;
    case Format::R32Float: return MTLPixelFormatR32Float;
    case Format::RG32Uint: return MTLPixelFormatRG32Uint;
    case Format::RG32Sint: return MTLPixelFormatRG32Sint;
    case Format::RG32Float: return MTLPixelFormatRG32Float;
    case Format::RGBA32Uint: return MTLPixelFormatRGBA32Uint;
    case Format::RGBA32Sint: return MTLPixelFormatRGBA32Sint;
    case Format::RGBA32Float: return MTLPixelFormatRGBA32Float;
    case Format::RGB10A2Unorm: return MTLPixelFormatRGB10A2Unorm;
    case Format::RGB10A2Uint: return MTLPixelFormatRGB10A2Uint;
    case Format::RG11B10Float: return MTLPixelFormatRG11B10Float;
    case Format::RGB9E5Float: return MTLPixelFormatRGB9E5Float;
    case Format::B5G6R5Unorm: return MTLPixelFormatB5G6R5Unorm;
    case Format::BGR5A1Unorm: return MTLPixelFormatBGR5A1Unorm;
    case Format::ABGR4Unorm: return MTLPixelFormatABGR4Unorm;
    case Format::BC1: return MTLPixelFormatBC1_RGBA;
    case Format::BC1Srgb: return MTLPixelFormatBC1_RGBA_sRGB;
    case Format::BC2: return MTLPixelFormatBC2_RGBA;
    case Format::BC2Srgb: return MTLPixelFormatBC2_RGBA_sRGB;
    case Format::BC3: return MTLPixelFormatBC3_RGBA;
    case Format::BC3Srgb: return MTLPixelFormatBC3_RGBA_sRGB;
    case Format::BC4Unorm: return MTLPixelFormatBC4_RUnorm;
    case Format::BC4Snorm: return MTLPixelFormatBC4_RSnorm;
    case Format::BC5Unorm: return MTLPixelFormatBC5_RGUnorm;
    case Format::BC5Snorm: return MTLPixelFormatBC5_RGSnorm;
    case Format::BC6HUfloat: return MTLPixelFormatBC6H_RGBUfloat;
    case Format::BC6HSfloat: return MTLPixelFormatBC6H_RGBFloat;
    case Format::BC7: return MTLPixelFormatBC7_RGBAUnorm;
    case Format::BC7Srgb: return MTLPixelFormatBC7_RGBAUnorm_sRGB;
    // ETC2/EAC and ASTC (every LDR block size): native on Apple GPUs, used by Quest titles.
    case Format::Etc2Rgb8: return MTLPixelFormatETC2_RGB8;
    case Format::Etc2Rgb8Srgb: return MTLPixelFormatETC2_RGB8_sRGB;
    case Format::Etc2Rgb8A1: return MTLPixelFormatETC2_RGB8A1;
    case Format::Etc2Rgb8A1Srgb: return MTLPixelFormatETC2_RGB8A1_sRGB;
    case Format::Etc2Rgba8: return MTLPixelFormatEAC_RGBA8;
    case Format::Etc2Rgba8Srgb: return MTLPixelFormatEAC_RGBA8_sRGB;
    case Format::EacR11Unorm: return MTLPixelFormatEAC_R11Unorm;
    case Format::EacR11Snorm: return MTLPixelFormatEAC_R11Snorm;
    case Format::EacRg11Unorm: return MTLPixelFormatEAC_RG11Unorm;
    case Format::EacRg11Snorm: return MTLPixelFormatEAC_RG11Snorm;
    case Format::Astc4x4: return MTLPixelFormatASTC_4x4_LDR;
    case Format::Astc4x4Srgb: return MTLPixelFormatASTC_4x4_sRGB;
    case Format::Astc6x6: return MTLPixelFormatASTC_6x6_LDR;
    case Format::Astc6x6Srgb: return MTLPixelFormatASTC_6x6_sRGB;
    case Format::Astc8x8: return MTLPixelFormatASTC_8x8_LDR;
    case Format::Astc8x8Srgb: return MTLPixelFormatASTC_8x8_sRGB;
    case Format::Astc10x10: return MTLPixelFormatASTC_10x10_LDR;
    case Format::Astc10x10Srgb: return MTLPixelFormatASTC_10x10_sRGB;
    case Format::Astc12x12: return MTLPixelFormatASTC_12x12_LDR;
    case Format::Astc12x12Srgb: return MTLPixelFormatASTC_12x12_sRGB;
    case Format::Astc5x4: return MTLPixelFormatASTC_5x4_LDR;
    case Format::Astc5x4Srgb: return MTLPixelFormatASTC_5x4_sRGB;
    case Format::Astc5x5: return MTLPixelFormatASTC_5x5_LDR;
    case Format::Astc5x5Srgb: return MTLPixelFormatASTC_5x5_sRGB;
    case Format::Astc6x5: return MTLPixelFormatASTC_6x5_LDR;
    case Format::Astc6x5Srgb: return MTLPixelFormatASTC_6x5_sRGB;
    case Format::Astc8x5: return MTLPixelFormatASTC_8x5_LDR;
    case Format::Astc8x5Srgb: return MTLPixelFormatASTC_8x5_sRGB;
    case Format::Astc8x6: return MTLPixelFormatASTC_8x6_LDR;
    case Format::Astc8x6Srgb: return MTLPixelFormatASTC_8x6_sRGB;
    case Format::Astc10x5: return MTLPixelFormatASTC_10x5_LDR;
    case Format::Astc10x5Srgb: return MTLPixelFormatASTC_10x5_sRGB;
    case Format::Astc10x6: return MTLPixelFormatASTC_10x6_LDR;
    case Format::Astc10x6Srgb: return MTLPixelFormatASTC_10x6_sRGB;
    case Format::Astc10x8: return MTLPixelFormatASTC_10x8_LDR;
    case Format::Astc10x8Srgb: return MTLPixelFormatASTC_10x8_sRGB;
    case Format::Astc12x10: return MTLPixelFormatASTC_12x10_LDR;
    case Format::Astc12x10Srgb: return MTLPixelFormatASTC_12x10_sRGB;
    case Format::D16Unorm: return MTLPixelFormatDepth16Unorm;
    case Format::D32Float: return MTLPixelFormatDepth32Float;
    // Apple GPUs have no D24S8; D32S8 holds every D24 value exactly.
    case Format::D24UnormS8: return MTLPixelFormatDepth32Float_Stencil8;
    case Format::D32FloatS8: return MTLPixelFormatDepth32Float_Stencil8;
    case Format::S8Uint: return MTLPixelFormatStencil8;
    case Format::Count: break;
  }
  return MTLPixelFormatInvalid;
}

MTLBlendFactor blendFactor(BlendFactor f) {
  static constexpr MTLBlendFactor map[] = {
      MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor, MTLBlendFactorOneMinusSourceColor,
      MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorDestinationColor,
      MTLBlendFactorOneMinusDestinationColor, MTLBlendFactorDestinationAlpha,
      MTLBlendFactorOneMinusDestinationAlpha, MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor,
      MTLBlendFactorBlendAlpha, MTLBlendFactorOneMinusBlendAlpha, MTLBlendFactorSourceAlphaSaturated,
      MTLBlendFactorSource1Color, MTLBlendFactorOneMinusSource1Color, MTLBlendFactorSource1Alpha,
      MTLBlendFactorOneMinusSource1Alpha};
  static_assert(sizeof map / sizeof map[0] == size_t(BlendFactor::OneMinusSrc1Alpha) + 1);
  return map[static_cast<int>(f)];
}

MTLBlendOperation blendOp(BlendOp o) {
  static constexpr MTLBlendOperation map[] = {MTLBlendOperationAdd, MTLBlendOperationSubtract,
                                              MTLBlendOperationReverseSubtract, MTLBlendOperationMin,
                                              MTLBlendOperationMax};
  return map[static_cast<int>(o)];
}

// Core and Metal list compare functions in the same order (Never..Always).
MTLCompareFunction compareFn(CompareOp c) { return static_cast<MTLCompareFunction>(static_cast<int>(c)); }

MTLSamplerAddressMode addressMode(AddressMode m) {
  static constexpr MTLSamplerAddressMode map[] = {
      MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat, MTLSamplerAddressModeClampToEdge,
      MTLSamplerAddressModeClampToBorderColor, MTLSamplerAddressModeMirrorClampToEdge};
  return map[static_cast<int>(m)];
}

MTLPrimitiveType primitiveType(Primitive p) {
  static constexpr MTLPrimitiveType map[] = {MTLPrimitiveTypePoint, MTLPrimitiveTypeLine, MTLPrimitiveTypeLineStrip,
                                             MTLPrimitiveTypeTriangle, MTLPrimitiveTypeTriangleStrip};
  return map[static_cast<int>(p)];
}

MTLPrimitiveTopologyClass topologyClass(Primitive p) {
  if (p == Primitive::Points) return MTLPrimitiveTopologyClassPoint;
  if (p == Primitive::Lines || p == Primitive::LineStrip) return MTLPrimitiveTopologyClassLine;
  return MTLPrimitiveTopologyClassTriangle;
}

MTLColorWriteMask writeMask(uint8_t rgba) {  // core: bit0 = R .. bit3 = A
  MTLColorWriteMask m = MTLColorWriteMaskNone;
  if (rgba & 1) m |= MTLColorWriteMaskRed;
  if (rgba & 2) m |= MTLColorWriteMaskGreen;
  if (rgba & 4) m |= MTLColorWriteMaskBlue;
  if (rgba & 8) m |= MTLColorWriteMaskAlpha;
  return m;
}

MTLLoadAction loadAction(LoadAction a) {
  return a == LoadAction::Clear ? MTLLoadActionClear : a == LoadAction::Load ? MTLLoadActionLoad : MTLLoadActionDontCare;
}
MTLStoreAction storeAction(StoreAction a) {
  switch (a) {
    case StoreAction::DontCare: return MTLStoreActionDontCare;
    case StoreAction::Store: return MTLStoreActionStore;
    case StoreAction::Resolve: return MTLStoreActionMultisampleResolve;
    case StoreAction::StoreAndResolve: return MTLStoreActionStoreAndMultisampleResolve;
  }
  return MTLStoreActionStore;
}

// Core and Metal list stencil operations in the same order (Keep..DecrementWrap).
MTLStencilDescriptor* stencilFace(const StencilFace& f) {
  MTLStencilDescriptor* d = [MTLStencilDescriptor new];
  d.stencilCompareFunction = static_cast<MTLCompareFunction>(static_cast<int>(f.compare));
  d.stencilFailureOperation = static_cast<MTLStencilOperation>(static_cast<int>(f.fail));
  d.depthFailureOperation = static_cast<MTLStencilOperation>(static_cast<int>(f.depthFail));
  d.depthStencilPassOperation = static_cast<MTLStencilOperation>(static_cast<int>(f.pass));
  d.readMask = f.readMask;
  d.writeMask = f.writeMask;
  return d;
}

MTLTextureType textureType(const TextureDesc& d) {
  const bool array = d.layers > 1;
  switch (d.type) {
    case TextureType::Tex1D: return array ? MTLTextureType1DArray : MTLTextureType1D;
    case TextureType::Tex2D:
      return d.samples > 1 ? (array ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample)
                           : (array ? MTLTextureType2DArray : MTLTextureType2D);
    case TextureType::Tex3D: return MTLTextureType3D;
    case TextureType::Cube: return d.layers > 6 ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
  }
  return MTLTextureType2D;
}

MTLTextureDescriptor* textureDescriptor(const TextureDesc& d) {
  MTLTextureDescriptor* t = [MTLTextureDescriptor new];
  t.textureType = textureType(d);
  t.pixelFormat = pixelFormat(d.format);
  t.width = d.width;
  t.height = d.height;
  t.depth = d.type == TextureType::Tex3D ? d.depth : 1;
  t.mipmapLevelCount = d.mips;
  t.arrayLength = d.type == TextureType::Cube ? d.layers / 6 : d.layers;
  t.sampleCount = d.samples;
  t.storageMode = MTLStorageModePrivate;
  t.hazardTrackingMode = MTLHazardTrackingModeUntracked;
  MTLTextureUsage u = MTLTextureUsageUnknown;
  if (d.usage & usage::sampled) u |= MTLTextureUsageShaderRead;
  // Small 1-2 channel sampled formats (R8 font atlases, RG8…) are the ones views swizzle
  // (VkComponentMapping); allow format/swizzle views on them.
  {
    const FormatInfo& fi = formatInfo(d.format);
    if ((d.usage & usage::sampled) && fi.blockSize == 1 && fi.bytesPerBlock <= 2 && !fi.depth() && !fi.stencil())
      u |= MTLTextureUsagePixelFormatView;
  }
  if (d.usage & usage::formatView) u |= MTLTextureUsagePixelFormatView;  // mutable-format images
  if (d.usage & usage::storage) u |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  // imageAtomic* (Vulkan STORAGE_IMAGE_ATOMIC, mandatory for R32_UINT/R32_SINT): Metal texture
  // atomics need this usage, and only these two 32-bit formats take it.
  if ((d.usage & usage::storage) && (d.format == Format::R32Uint || d.format == Format::R32Sint))
    u |= MTLTextureUsageShaderAtomic;
  if (d.usage & (usage::colorTarget | usage::depthTarget)) u |= MTLTextureUsageRenderTarget;
  t.usage = u;
  return t;
}

// FNV-1a of the MSL text (never 0).
uint64_t sourceKey(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
  return h ? h : 1;
}

// Function cache key of a fragment shader specialised for framebuffer-fetch inputs.
uint64_t functionKey(uint64_t shaderId, const FetchInputs& f) {
  if (!f.mask) return shaderId;
  uint64_t h = shaderId ^ 0x9E3779B97F4A7C15ull;
  auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  mix(f.mask);
  for (uint16_t t : f.textureSlot) mix(t);
  return h ? h : 1;
}

// Same fetch inputs: mask and the texture slot of every fetched attachment.
bool sameFetch(const FetchInputs& a, const FetchInputs& b) {
  if (a.mask != b.mask) return false;
  for (int i = 0; i < kMaxColorAttachments; ++i)
    if ((a.mask >> i & 1) && a.textureSlot[i] != b.textureSlot[i]) return false;
  return true;
}

std::string str(NSError* e) {
  return e ? std::string(e.localizedDescription.UTF8String ?: "?") : std::string("unknown error");
}

// MTLCompilerService could not be reached (it died, or launchd is throttling its relaunch): not a
// broken shader. Log 60 (Vulkan CTS): from 2 s into the run, compiles in short windows failed with
// "Unable to reach MTLCompilerService ... Connection init failed at lookup" while most others
// worked, and a failed blit library was then cached as broken for the whole device.
bool compilerUnreachable(NSError* e) {
  if (!e) return false;
  NSString* d = e.localizedDescription;
  return [d containsString:@"MTLCompilerService"] || [d containsString:@"compiler is no longer active"] ||
         [d containsString:@"Connection interrupted"] || [d containsString:@"Connection invalid"];
}

// Runs `make` (a Metal compile returning nil + NSError on failure) again while the failure is the
// compiler being unreachable: 4 retries, ~1 s in all. `retries` (optional) says how many it took.
// After a failure that retries could not fix, 2 s in which compiles are tried once without
// sleeping: an outage costs each caller (submit thread included) one attempt, not ~1 s each.
std::atomic<double> gCompilerBackoffUntil{0.0};

double steadySeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class Make>
id compileRetrying(NSError* __autoreleasing* e, Make make, int* retries = nullptr) {
  static const int kDelayMs[] = {50, 150, 300, 450};
  id r = make(e);
  int n = 0;
  const bool backingOff = steadySeconds() < gCompilerBackoffUntil.load(std::memory_order_relaxed);
  for (; !r && !backingOff && n < 4 && compilerUnreachable(*e); ++n) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kDelayMs[n]));
    *e = nil;
    r = make(e);
  }
  if (!r && compilerUnreachable(*e)) gCompilerBackoffUntil.store(steadySeconds() + 2.0, std::memory_order_relaxed);
  if (retries) *retries = n;
  return r;
}

// One-token form for key=value log lines.
std::string token(std::string s) {
  for (char& c : s)
    if (c == ' ' || c == '\n' || c == '=') c = '_';
  return s;
}

}  // namespace

struct MetalBackend::Impl {
  struct Buffer {
    uint32_t generation = 0;
    id<MTLBuffer> buffer;
    uint64_t base = 0;  // guest pointer's offset inside a page-aligned no-copy buffer
  };
  struct Texture {
    uint32_t generation = 0;
    id<MTLTexture> texture;  // nil until first needed (render targets)
    TextureDesc desc;
    bool failed = false;     // allocation failed once (logged then): not retried every use
  };
  struct View {
    uint32_t generation = 0;
    TextureId base;
    MetalBackend::ViewDesc desc;
    id<MTLTexture> texture;     // the view, made against `of`
    __unsafe_unretained id<MTLTexture> of = nil;  // base texture the view was made from
    bool failed = false;
  };
  std::unordered_map<uint32_t, View> views;  // index has kViewBit set
  uint32_t nextView = 1;
  static constexpr uint32_t kViewBit = 0x80000000u;
  id<MTLTexture> view(TextureId);
  struct Sampler {
    uint32_t generation = 0;
    id<MTLSamplerState> sampler;
  };
  // Argument table of one stage: its declared layout and the [[buffer(n)]] it is bound at.
  struct Table {
    std::vector<ShaderArgument> args;
    uint32_t index = 30;
  };
  struct Pipeline {
    id<MTLRenderPipelineState> render;
    bool blendConstant = false;
    // The vertex function reads the draw index (SPIR-V DrawIndex): `constant uint& [[buffer(29)]]`
    // (vpmsl kDrawIndexBuffer), set per draw from Draw::drawIndex.
    bool drawIndex = false;
    id<MTLDepthStencilState> depth;
    // What `depth` was built from, for draws with dynamic stencil masks (Draw::stencil*Mask):
    // one state per mask combination, built on the submit thread.
    bool depthTest = false, depthWrite = false, stencilTest = false;
    CompareOp depthCompare = CompareOp::Always;
    StencilFace stencilFront, stencilBack;
    mutable std::unordered_map<uint32_t, id<MTLDepthStencilState>> maskedDepth;
    MTLCullMode cull = MTLCullModeNone;
    NSUInteger samples = 1;  // rasterSampleCount (an attachmentless pass takes it as its default)
    FetchInputs fetch;
    id<MTLComputePipelineState> compute;
    MTLSize threads{1, 1, 1};
    Table stage[2];  // render: vertex, fragment; compute: [0]
  };
  // Per frame in flight: things the GPU may still read until that frame completes.
  struct Slot {
    id<MTLBuffer> args;  // first argument buffer (kArgRingBytes)
    // Chained argument buffers for frames whose tables do not fit in `args` (a prewarm frame that
    // draws thousands of pipelines once): kept with the slot, reused by its later frames.
    std::vector<id<MTLBuffer>> argsMore;
    id<MTLCounterSampleBuffer> samples;
    std::vector<id<MTLFence>> fences;
    std::vector<id> retired;  // destroyed resources kept alive until the slot is reused
  };
  struct Done {
    uint64_t frame;
    GpuTimings timings;
  };

  const ShaderProvider& shaders;
  LogSink log;
  MetalOptions options;
  std::mutex logMutex;

  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLResidencySet> residency;  // M4: attached to the MTL4CommandQueue the same way
  bool residencyDirty = false;
  dispatch_semaphore_t inFlight;
  uint64_t submits = 0;
  std::vector<Slot> slots;
  std::vector<id> graveyard;
  bool stageSamples = false;
  double gpuTickNs = 1.0;  // GPU timestamp period, calibrated against the CPU clock

  std::unordered_map<uint32_t, Buffer> buffers;
  std::unordered_map<uint32_t, Texture> textures;
  std::unordered_map<uint32_t, Sampler> samplers;
  bool tileMemory = false;  // memoryless storage + framebuffer fetch
  // Set when a command buffer failed because memoryless attachments met a tiler flush (too much
  // geometry): from then on tile-local attachments get real memory (correctness over footprint).
  std::atomic<bool> memorylessFailed{false};
  MetalCaps caps;
  // Memoryless textures by shape; each encoder takes them in order, so two tile-local attachments
  // of one shape get distinct textures while encoders reuse them (contents never outlive one).
  std::unordered_map<uint64_t, std::vector<id<MTLTexture>>> memoryless;
  // Sampled blits (CopyKind::TextureBlit): one fullscreen-triangle pipeline per destination pixel
  // format, built on first use. Submit thread only, so no lock.
  id<MTLLibrary> blitLibrary;
  bool blitLibraryFailed = false;      // a real compile error: never retried
  double blitLibraryRetryAt = 0;      // compiler unreachable: retried after this (steady seconds)
  std::unordered_map<NSUInteger, id<MTLRenderPipelineState>> blitPipelines;  // nil value = failed
  id<MTLSamplerState> blitSamplers[2];  // nearest, linear
  std::unordered_set<std::string> blitWarned;  // reasons already logged
  // variant: bit 0-1 = 0 float, 1 int, 2 uint; bit 2 = 3D source.
  id<MTLRenderPipelineState> blitPipeline(MTLPixelFormat format, int variant, std::string& error);

  // In-pass clear quads (Draw::clear): pipelines by the encoder's attachment formats, sample
  // count, which attachments are cleared and layering; depth/stencil states by (depth, stencil).
  // Built on first use on the submit thread (no lock), failures cached as nil.
  std::unordered_map<std::string, id<MTLRenderPipelineState>> clearPipelines;
  id<MTLDepthStencilState> clearDepthStates[4];
  id<MTLRenderPipelineState> clearPipeline(MTLRenderPassDescriptor* rp, const ClearRects& c, bool layered,
                                           const std::array<int, kMaxColorAttachments>& kinds, std::string& error);
  id<MTLDepthStencilState> clearDepthState(bool depth, bool stencil);

  // D24 depth in buffer<->texture copies: Apple GPUs keep D24UnormS8 as Depth32Float_Stencil8, but
  // the client's buffer holds 24-bit unorm in a uint32 per texel. A compute kernel converts between
  // the client buffer and a float staging buffer that the blit moves. [0] = unorm24 → float
  // (upload), [1] = float → unorm24 (readback). Submit thread only; failure cached.
  id<MTLComputePipelineState> d24Pipelines[2];
  bool d24Failed = false;
  id<MTLComputePipelineState> d24Pipeline(bool toUnorm, std::string& error);

  // pipelineMutex guards only the maps (the submit thread takes it every frame, so it is never
  // held across a Metal compile). archiveMutex guards the session's binary archive: adding to it
  // compiles the pipeline again, so that happens after the pipeline is published (drainArchiveLocked)
  // and never on the submit thread.
  std::mutex pipelineMutex;  // guards pipelines, functions
  std::unordered_map<uint64_t, Pipeline> pipelines;
  std::unordered_map<uint64_t, id<MTLFunction>> functions;
  // Function keys whose library compile is running (guarded by pipelineMutex); a second caller
  // waits on functionReady instead of compiling the same MSL again.
  std::unordered_set<uint64_t> compiling;
  std::condition_variable functionReady;
  // Speculative library compiles (prewarmShader): two serial queues, low QoS. A pipeline worker that
  // needs a key not yet started compiles it itself at once, so a draw never waits behind them.
  dispatch_queue_t prewarmQueues[2] = {nil, nil};
  std::atomic<uint32_t> prewarmNext{0};
  std::atomic<bool> closing{false};
  std::mutex libStatsMutex;
  uint32_t libCompiled = 0, libDedup = 0, libPrewarmed = 0, libLogged = 0;
  double libMsTotal = 0, libMsMax = 0;
  // Pipeline binary archives under cacheRoot:
  //  - pipelines.metalar (legacy base, number 0) and pipelines-<n>.metalar (one per session that
  //    added pipelines) are loaded read-only into lookupArchives and never mutated or rewritten;
  //  - this session's new pipelines go into `archive`, a fresh in-memory archive whose
  //    descriptors are also kept (archiveEntries, add order) so it can be rebuilt after a failed
  //    serialize (a failure can leave the archive object empty: "Nothing to serialize");
  //  - pipelines.count lists the pipelines per file (only for logs) and the next file number.
  struct ArchiveEntry {
    id desc;  // retained copy: MTLRenderPipelineDescriptor or MTLComputePipelineDescriptor
    bool compute = false;
    std::string label;
  };
  bool archiveOn = false;  // cacheRoot set and a session archive exists: misses are archived
  NSArray<id<MTLBinaryArchive>>* lookupArchives = nil;  // set in init, immutable: read without a lock
  std::mutex archiveQueueMutex;  // guards archiveQueue only (taken briefly by compile workers)
  std::vector<ArchiveEntry> archiveQueue;  // misses not yet added to `archive`
  std::mutex archiveMutex;  // guards everything below
  id<MTLBinaryArchive> archive;  // this session's delta; nil = rebuild from archiveEntries
  std::vector<ArchiveEntry> archiveEntries;  // in `archive`, add order
  size_t archiveSavedEntries = 0;  // entries in the session file on disk after the last save
  uint32_t archiveUnsaved = 0;  // added since the last serialize
  uint64_t archiveSession = 1;  // number of this session's file (pipelines-<n>.metalar)
  // Most pipelines one file holds. Large archives fail to serialize with "Failed to generate
  // machO: cannot pack" and no pipeline index (84 entries failed where 23 and 33 saved), so files
  // are kept small; a failure of that kind lowers the cap for the rest of the session.
  size_t archivePackCap = 32;
  std::map<uint64_t, uint32_t> archiveFileCounts;  // file number -> pipelines, for files on disk
  std::unordered_set<std::string> archiveWarned;  // labels already logged as not archivable
  // Content keys of every descriptor archived (or tried) this session: pipeline states that differ
  // only in what Metal keeps outside the render pipeline (depth-stencil state, cull, dynamic
  // state) produce identical descriptors, and duplicates are never added twice.
  std::unordered_set<uint64_t> archiveSeen;
  std::chrono::steady_clock::time_point archiveSavedAt = std::chrono::steady_clock::now();
  void openArchives();
  std::filesystem::path archivePath(uint64_t n) const;
  id<MTLBinaryArchive> newArchive();
  bool addToArchive(id<MTLBinaryArchive> ar, const ArchiveEntry& e, std::string& error);
  void queueArchive(id desc, bool compute, const std::string& label);
  void drainArchiveLocked();
  id<MTLBinaryArchive> buildArchiveLocked(size_t begin, size_t end, bool dropFailures);
  bool serializeArchive(id<MTLBinaryArchive> ar, const std::filesystem::path& dst, std::string& error);
  bool probeArchiveLocked(size_t begin, size_t end);
  void dropArchiveEntryLocked(size_t i, const char* reason);
  bool dropCulpritLocked(const std::string& error);
  void logArchiveCulpritLocked(const std::string& error);
  void splitSaveLocked(const ArchiveEntry* entries, size_t n, int depth, size_t& files, size_t& saved,
                       size_t& lost);
  bool chunkSaveLocked();
  void writeArchiveCountLocked();
  bool saveArchiveLocked();
  void maybeSaveArchive();

  std::mutex doneMutex;
  std::deque<Done> done;
  // Frame completion for fences (waitFrames), under doneMutex: every frame index below
  // `finishedBelow` has completed; later ones that completed first wait in `finishedAhead`.
  std::condition_variable doneCv;
  uint64_t finishedBelow = 0;
  std::set<uint64_t> finishedAhead;
  // Any command buffer completed (frames and native work): stall detection in the waits.
  std::atomic<uint64_t> completions{0};
  // Device lost (see the header). `lostAtNs` is steady-clock ns when it happened.
  std::atomic<bool> lostFlag{false};
  std::atomic<bool> lostClaimed{false};  // markLost's single winner (lostAtNs is set before lostFlag)
  std::atomic<int64_t> lostAtNs{0};
  bool strictLost = false;  // VPVK_DEVICE_LOST_STRICT=1: every non-memoryless failure is a loss
  void frameFinished(uint64_t frameIndex);
  void markLost(const std::string& why);
  // Classifies a failed command buffer: true when it means the device is lost.
  bool lostByError(NSError* e) const;
  // The slot semaphore, with lost-device and stall handling; true = acquired.
  bool takeSlotSemaphore();
  WaitResult drain(uint64_t timeoutNs, bool& drained);
  // One drain at a time: two threads each holding part of the slots would wait on each other.
  std::mutex drainMutex;

  // Fences of encoders not yet ordered before any later encoder. A barrier waits on all of them
  // and leaves only itself, so ordering is transitive and each barrier costs few waits. It carries
  // over between frames: a frame's first encoder waits on the previous frame's tail.
  std::vector<id<MTLFence>> frontier;

  // Per-pass dump (VP_DUMP_FRAME=<n>, VP_DUMP_DIR=<dir>); -1 = off.
  int64_t dumpFrame = -1;
  std::string dumpDir;

  Impl(const ShaderProvider& s, LogSink l, MetalOptions o) : shaders(s), log(std::move(l)), options(std::move(o)) {}

  __attribute__((format(printf, 2, 3))) void err(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::lock_guard lock(logMutex);
    log(std::string("gpu.err ") + buf);
  }
  void info(const std::string& line) {
    std::lock_guard lock(logMutex);
    log(line);
  }

  void init();
  void calibrate();
  void retire(id object);
  id<MTLTexture> texture(TextureId);
  id<MTLTexture> memorylessTexture(const TextureDesc&, std::unordered_map<uint64_t, size_t>& taken);
  // `t` seen as another pixel format of the same texel size (attachments of mutable-format
  // images, Attachment::format); nil when Metal cannot (logged once per pair). Cached per texture.
  id<MTLTexture> formatViewOfPixel(id<MTLTexture> t, MTLPixelFormat pf);
  id<MTLTexture> formatViewOf(id<MTLTexture> t, Format f) {
    return t && f != Format::Undefined && pixelFormat(f) != t.pixelFormat ? formatViewOfPixel(t, pixelFormat(f)) : t;
  }
  struct FormatViewKey {
    __unsafe_unretained id<MTLTexture> base;
    MTLPixelFormat format;
    bool operator==(const FormatViewKey& o) const { return base == o.base && format == o.format; }
  };
  struct FormatViewHash {
    size_t operator()(const FormatViewKey& k) const {
      return std::hash<const void*>()((__bridge const void*)k.base) ^ (size_t(k.format) * 0x9E3779B97F4A7C15ull);
    }
  };
  std::unordered_map<FormatViewKey, id<MTLTexture>, FormatViewHash> formatViews;
  id<MTLFunction> function(uint64_t shaderId, const FetchInputs& fetch, MslShader& shader, std::string& error,
                           bool noPointSize = false, bool noFragDepth = false);
  bool buildRender(const PipelineState&, Pipeline&, std::string& error);
  bool buildCompute(const PipelineState&, Pipeline&, std::string& error);
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&);
  // Waits for a free frame slot, takes the next one and releases what it retired (submit thread).
  // Null when the device is lost (nothing more runs on the GPU).
  Slot* acquireSlot();
  // Native client command buffers (encodeNative): the event that orders them against the frames
  // (hazard-untracked resources), its last signalled value, and how many were committed.
  bool encodeNative(NativeEncodeFn fn, void* user);
  // Shared, so a lost device can release command buffers stuck waiting on it (markLost); the raw
  // pointer is what other threads read (the strong reference above keeps it alive).
  id<MTLSharedEvent> nativeEvent;
  std::atomic<void*> nativeEventRaw{nullptr};
  id<MTLBuffer> nativeScratch;  // target of the ordering encoders' one-word fills
  uint64_t nativeValue = 0, nativeSubmits = 0;
};

void MetalBackend::Impl::init() {
  device = MTLCreateSystemDefaultDevice();
  if (!device) {
    err("what=no_device");
    return;
  }
  queue = [device newCommandQueue];
  queue.label = @"vpgpu";
  caps.simulator = TARGET_OS_SIMULATOR;
  caps.metal3 = [device supportsFamily:MTLGPUFamilyMetal3];
  caps.apple4 = [device supportsFamily:MTLGPUFamilyApple4];
  caps.apple7 = [device supportsFamily:MTLGPUFamilyApple7];
  caps.argumentBuffersTier = device.argumentBuffersSupport == MTLArgumentBuffersTier2 ? 2 : 1;
  caps.bindless = caps.metal3 && caps.argumentBuffersTier == 2;
  // The simulator runs on the host GPU without memoryless storage or framebuffer fetch, whatever
  // family it reports.
  tileMemory = caps.apple4 && !caps.simulator;

  MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
  rd.label = @"vpgpu.resources";
  NSError* e = nil;
  residency = [device newResidencySetWithDescriptor:rd error:&e];
  caps.residencySet = residency != nil;
  if (residency) [queue addResidencySet:residency];
  else  // a capability, not a failure (the simulator has none): bound resources use useResource
    info("gpu.metal.info what=no_residency_set fallback=use_resource error=" + token(str(e)));

  // Timestamps at encoder boundaries, if this GPU samples there (Apple GPUs do).
  id<MTLCounterSet> timestamps = nil;
  for (id<MTLCounterSet> set in device.counterSets)
    if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
  stageSamples = timestamps && [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];

  const unsigned n = std::max(2u, options.framesInFlight);
  inFlight = dispatch_semaphore_create(n);
  slots.resize(n);
  for (Slot& s : slots) {
    s.args = [device newBufferWithLength:kArgRingBytes options:kSharedUntracked];
    s.args.label = @"vpgpu.args";
    if (!stageSamples) continue;
    MTLCounterSampleBufferDescriptor* sd = [MTLCounterSampleBufferDescriptor new];
    sd.counterSet = timestamps;
    sd.storageMode = MTLStorageModeShared;
    sd.sampleCount = kMaxSamples;
    s.samples = [device newCounterSampleBufferWithDescriptor:sd error:&e];
    if (!s.samples) {
      err("what=counter_buffer error=%s", token(str(e)).c_str());
      stageSamples = false;
    }
  }
  calibrate();

  if (const char* f = std::getenv("VP_DUMP_FRAME")) {
    char* end = nullptr;
    const long long v = std::strtoll(f, &end, 10);
    if (end != f && v >= 0) dumpFrame = v;
  }
  if (const char* d = std::getenv("VP_DUMP_DIR")) dumpDir = d;
  if (const char* e = std::getenv("VPVK_DEVICE_LOST_STRICT")) strictLost = *e == '1';
  while (dumpDir.size() > 1 && dumpDir.back() == '/') dumpDir.pop_back();
  if (dumpFrame >= 0 && dumpDir.empty()) dumpFrame = -1;
  if (dumpFrame >= 0) info("gpu.dump armed frame=" + std::to_string(dumpFrame) + " dir=" + token(dumpDir));

  if (!options.cacheRoot.empty()) openArchives();

  info("gpu.metal device=" + token(device.name.UTF8String) + " simulator=" + std::to_string(caps.simulator) +
       " metal3=" + std::to_string(caps.metal3) + " apple4=" + std::to_string(caps.apple4) +
       " apple7=" + std::to_string(caps.apple7) + " argument_buffers_tier=" + std::to_string(caps.argumentBuffersTier) +
       " bindless=" + std::to_string(caps.bindless) + " stage_timestamps=" + std::to_string(stageSamples) +
       " residency=" + std::to_string(caps.residencySet) + " archive=" + std::to_string(archiveOn) +
       " tile_memory=" + std::to_string(tileMemory));
}

void MetalBackend::Impl::calibrate() {
  MTLTimestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
  [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
  [NSThread sleepForTimeInterval:0.002];
  [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
  if (gpu1 > gpu0) gpuTickNs = double(cpu1 - cpu0) / double(gpu1 - gpu0);  // CPU side is in ns
}

void MetalBackend::Impl::retire(id object) {
  if (object) graveyard.push_back(object);
}

id<MTLTexture> MetalBackend::Impl::view(TextureId t) {
  auto it = views.find(t.index);
  if (it == views.end() || it->second.generation != t.generation) return nil;
  View& v = it->second;
  id<MTLTexture> base = texture(v.base);
  if (!base) return nil;
  if (v.texture && v.of == base) return v.texture;
  if (v.failed && v.of == base) return nil;
  v.of = base;
  const MetalBackend::ViewDesc& d = v.desc;
  MTLTextureType type = MTLTextureType2D;
  switch (d.type) {
    case TextureType::Tex1D: type = MTLTextureType1D; break;
    case TextureType::Tex3D: type = MTLTextureType3D; break;
    case TextureType::Cube: type = d.array ? MTLTextureTypeCubeArray : MTLTextureTypeCube; break;
    default: type = d.array ? MTLTextureType2DArray : MTLTextureType2D; break;
  }
  if (base.sampleCount > 1) type = d.array ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
  MTLPixelFormat pf = d.format == Format::Undefined ? base.pixelFormat : pixelFormat(d.format);
  // Stencil aspect of a depth/stencil texture: its stencil view format (no PixelFormatView usage
  // needed for these).
  if (d.format == Format::S8Uint && base.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
    // Metal exempts X32_Stencil8 views of a combined depth/stencil texture from the
    // PixelFormatView requirement, so this works for attachments too (deferred / SSAO
    // passes that sample the stencil of their own depth buffer).
    pf = MTLPixelFormatX32_Stencil8;
  }
  // Views Metal cannot make abort the process inside newTextureViewWithPixelFormat (MTLReportFailure,
  // not nil): a compressed texture seen as an uncompressed format of the same block size (Vulkan
  // BLOCK_TEXEL_VIEW_COMPATIBLE), compressed families other than an sRGB twin, depth seen as colour,
  // or a different texel size. Refused here instead, named once per pair.
  if (pf != base.pixelFormat && d.format != Format::Undefined) {
    auto vt = textures.find(v.base.index);
    const Format bf = vt != textures.end() ? vt->second.desc.format : Format::Undefined;
    const FormatInfo &bi = formatInfo(bf), &vi = formatInfo(d.format);
    auto family = [](const char* n) {
      std::string f = n ? n : "";
      if (auto at = f.find("Srgb"); at != std::string::npos) f.erase(at, 4);
      return f;
    };
    const char* why = nullptr;
    if (bf == Format::Undefined) why = nullptr;
    else if ((bi.blockSize > 1) != (vi.blockSize > 1)) why = "compressed_vs_uncompressed";
    else if (bi.blockSize > 1 && family(bi.name) != family(vi.name)) why = "compressed_family";
    else if ((bi.depth() || bi.stencil()) != (vi.depth() || vi.stencil()) && !(bi.stencil() && vi.stencil() && !vi.depth()))
      why = "depth_vs_color";
    else if (!(bi.depth() || bi.stencil()) && bi.bytesPerBlock != vi.bytesPerBlock) why = "texel_size";
    if (why) {
      v.failed = true;
      static std::mutex mu;
      static std::set<std::string> said;
      std::lock_guard<std::mutex> lk(mu);
      if (said.insert(std::string(bi.name) + ">" + vi.name).second && said.size() < 64)
        err("what=texture_view_unsupported reason=%s base=%s view=%s", why, bi.name, vi.name);
      return nil;
    }
  }
  const bool swz = d.swizzle[0] != 2 || d.swizzle[1] != 3 || d.swizzle[2] != 4 || d.swizzle[3] != 5;
  if (swz) {
    const MTLTextureSwizzleChannels ch = MTLTextureSwizzleChannelsMake(
        MTLTextureSwizzle(d.swizzle[0]), MTLTextureSwizzle(d.swizzle[1]), MTLTextureSwizzle(d.swizzle[2]),
        MTLTextureSwizzle(d.swizzle[3]));
    v.texture = [base newTextureViewWithPixelFormat:pf
                                        textureType:type
                                             levels:NSMakeRange(d.baseMip, d.mips)
                                             slices:NSMakeRange(d.baseLayer, d.layers)
                                            swizzle:ch];
    if (v.texture == nil) {  // swizzle view refused (usage): plain view rather than nothing
      static std::atomic<int> once{0};
      if (once.fetch_add(1) == 0) err("what=swizzle_view_fallback format=%d", int(pf));
      v.texture = [base newTextureViewWithPixelFormat:pf
                                          textureType:type
                                               levels:NSMakeRange(d.baseMip, d.mips)
                                               slices:NSMakeRange(d.baseLayer, d.layers)];
    }
  } else {
    v.texture = [base newTextureViewWithPixelFormat:pf
                                        textureType:type
                                             levels:NSMakeRange(d.baseMip, d.mips)
                                             slices:NSMakeRange(d.baseLayer, d.layers)];
  }
  v.failed = v.texture == nil;
  if (v.failed)
    err("what=texture_view base=%u type=%d format=%s mips=%u+%u layers=%u+%u", v.base.index, int(type),
        formatInfo(d.format).name, d.baseMip, d.mips, d.baseLayer, d.layers);
  return v.texture;
}

// Combinations Metal's texture-descriptor validation aborts the process on (MTLReportFailure, not
// a nil result), checked first so the texture fails cleanly instead. Vulkan allows more than
// Metal here; vkGetPhysicalDeviceImageFormatProperties reports the same limits.
static const char* metalInvalidTexture(const TextureDesc& d) {
  const FormatInfo& fi = formatInfo(d.format);
  const bool compressed = fi.blockSize > 1;
  const bool depthStencil = fi.depth() || fi.stencil();
  const MTLTextureType type = textureType(d);
  const bool is1d = type == MTLTextureType1D || type == MTLTextureType1DArray;
  const bool ms = type == MTLTextureType2DMultisample || type == MTLTextureType2DMultisampleArray;
  if (compressed && (is1d || ms)) return "compressed_needs_2d_3d_or_cube";
  // ETC2/EAC never come in 3D on Apple GPUs; ASTC 3D only on some (refused everywhere: no game
  // ships 3D ASTC). BC is fine in 3D.
  if (type == MTLTextureType3D && fi.name &&
      (!strncmp(fi.name, "Etc2", 4) || !strncmp(fi.name, "Eac", 3) || !strncmp(fi.name, "Astc", 4)))
    return "etc_astc_3d";
  if (is1d && d.mips > 1) return "1d_mipmaps";
  if (is1d && depthStencil) return "1d_depth";
  if (depthStencil && type == MTLTextureType3D) return "3d_depth";
  if (ms && d.mips > 1) return "multisample_mipmaps";
  if (d.samples > 1 && !ms) return "samples_on_non_2d";
  if (type == MTLTextureType3D && d.layers > 1) return "3d_array";
  if (d.width == 0 || d.height == 0 || d.mips == 0 || d.layers == 0 || (type == MTLTextureType3D && d.depth == 0))
    return "zero_extent";
  return nullptr;
}

id<MTLTexture> MetalBackend::Impl::texture(TextureId t) {
  if (t.index & kViewBit) return view(t);
  auto it = textures.find(t.index);
  if (it == textures.end() || it->second.generation != t.generation) return nil;
  Texture& tex = it->second;
  if (tex.texture || tex.failed) return tex.texture;
  // Metal aborts on a sample count the GPU lacks (8 on M2): refuse it here, logged once.
  if (tex.desc.samples > 1 && ![device supportsTextureSampleCount:tex.desc.samples]) {
    tex.failed = true;
    err("what=texture_samples index=%u samples=%u label=%s", t.index, unsigned(tex.desc.samples),
        token(tex.desc.label).c_str());
    return nil;
  }
  if (const char* why = metalInvalidTexture(tex.desc)) {
    tex.failed = true;
    err("what=texture_invalid reason=%s index=%u format=%s size=%ux%ux%u mips=%u layers=%u samples=%u label=%s", why,
        t.index, formatInfo(tex.desc.format).name, tex.desc.width, tex.desc.height, tex.desc.depth,
        unsigned(tex.desc.mips), unsigned(tex.desc.layers), unsigned(tex.desc.samples), token(tex.desc.label).c_str());
    return nil;
  }
  tex.texture = [device newTextureWithDescriptor:textureDescriptor(tex.desc)];
  if (!tex.texture) {
    tex.failed = true;
    err("what=texture index=%u format=%s size=%ux%u label=%s", t.index, formatInfo(tex.desc.format).name,
        tex.desc.width, tex.desc.height, token(tex.desc.label).c_str());
    return nil;
  }
  if (!tex.desc.label.empty()) tex.texture.label = @(tex.desc.label.c_str());
  if (residency) {
    [residency addAllocation:tex.texture];
    residencyDirty = true;
  }
  return tex.texture;
}

id<MTLTexture> MetalBackend::Impl::formatViewOfPixel(id<MTLTexture> t, MTLPixelFormat pf) {
  if (!t || t.pixelFormat == pf) return t;
  const FormatViewKey key{t, pf};
  if (auto it = formatViews.find(key); it != formatViews.end()) return it->second;
  id<MTLTexture> v = nil;
  // Only plain colour textures created for format views; Metal aborts (not nil) otherwise.
  if ((t.usage & MTLTextureUsagePixelFormatView) && t.storageMode != MTLStorageModeMemoryless) {
    const NSUInteger slices = t.textureType == MTLTextureType3D ? 1
                              : t.arrayLength * ((t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
    v = [t newTextureViewWithPixelFormat:pf
                             textureType:t.textureType
                                  levels:NSMakeRange(0, t.mipmapLevelCount)
                                  slices:NSMakeRange(0, slices)];
  }
  if (!v) {
    static std::mutex mu;
    static std::set<uint64_t> said;
    std::lock_guard<std::mutex> lk(mu);
    if (said.insert(uint64_t(t.pixelFormat) << 32 | uint64_t(pf)).second && said.size() < 64)
      err("what=attachment_format_view base=%u view=%u usage=%u", unsigned(t.pixelFormat), unsigned(pf), unsigned(t.usage));
    v = t;  // the texture's own format: rendering continues, in the wrong format
  }
  formatViews[key] = v;
  return v;
}

id<MTLTexture> MetalBackend::Impl::memorylessTexture(const TextureDesc& d, std::unordered_map<uint64_t, size_t>& taken) {
  const uint64_t key = uint64_t(d.format) | uint64_t(d.samples) << 8 | uint64_t(d.layers) << 16 |
                       uint64_t(d.width) << 32 | uint64_t(d.height) << 48;
  auto& pool = memoryless[key];
  size_t& n = taken[key];
  if (n == pool.size()) {
    if (d.samples > 1 && ![device supportsTextureSampleCount:d.samples]) {
      err("what=memoryless_samples samples=%u", unsigned(d.samples));
      return nil;
    }
    TextureDesc one = d;
    one.mips = 1;
    if (const char* why = metalInvalidTexture(one)) {
      err("what=memoryless_invalid reason=%s format=%s size=%ux%u", why, formatInfo(d.format).name, d.width, d.height);
      return nil;
    }
    MTLTextureDescriptor* td = textureDescriptor(one);
    td.storageMode = MTLStorageModeMemoryless;
    td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> t = [device newTextureWithDescriptor:td];
    if (!t) {
      err("what=memoryless format=%s size=%ux%u", formatInfo(d.format).name, d.width, d.height);
      return nil;
    }
    t.label = @"vpgpu.tile";
    pool.push_back(t);
  }
  return pool[n++];
}

id<MTLFunction> MetalBackend::Impl::function(uint64_t shaderId, const FetchInputs& fetch, MslShader& shader,
                                             std::string& error, bool noPointSize, bool noFragDepth) {
  if (!(fetch.mask ? shaders.findFetch(shaderId, fetch, shader) : shaders.find(shaderId, shader))) {
    error = "shader " + std::to_string(shaderId) + " not found";
    return nil;
  }
  // Only a source that actually declares the PointSize output needs a second
  // variant; every other shader keeps one compiled function for all topologies.
  noPointSize = noPointSize && shader.source.find("[[VP_PSZ]]") != std::string::npos;
  // Keyed by the MSL text, not the shader id: ids that differ only in argument slots (the
  // translator's positional resource names) share one library and one MTLFunction.
  uint64_t key = functionKey(sourceKey(shader.source), fetch);
  if (noPointSize) key = (key ^ 0x5053Aull) * 1099511628211ull;
  // Same for a fragment FragDepth write in a pass without a depth attachment.
  noFragDepth = noFragDepth && shader.source.find("VP_NO_FDEPTH") != std::string::npos;
  if (noFragDepth) key = (key ^ 0x4644E9ull) * 1099511628211ull;
  {
    std::unique_lock lock(pipelineMutex);
    for (;;) {
      auto it = functions.find(key);
      if (it != functions.end()) {
        std::lock_guard sl(libStatsMutex);
        ++libDedup;
        return it->second;
      }
      if (!compiling.count(key)) break;
      functionReady.wait(lock);  // the same MSL is compiling on another thread
    }
    compiling.insert(key);
  }
  // Whatever happens below, release waiters.
  struct Done {
    Impl* self;
    uint64_t key;
    ~Done() {
      { std::lock_guard l(self->pipelineMutex); self->compiling.erase(key); }
      self->functionReady.notify_all();
    }
  } done{this, key};
  const auto t0 = std::chrono::steady_clock::now();
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.mathMode = MTLMathModeFast;  // the emitter marks precise math with precise:: explicitly
  o.languageVersion = MTLLanguageVersion3_2;
  NSMutableDictionary<NSString*, NSObject*>* macros = [NSMutableDictionary dictionary];
  if (noPointSize) macros[@"VP_PSZ"] = @"user(vp_psz)";
  if (noFragDepth) macros[@"VP_NO_FDEPTH"] = @"1";
  if (macros.count) o.preprocessorMacros = macros;
  NSError* e = nil;
  int retried = 0;
  id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
    return [device newLibraryWithSource:@(shader.source.c_str()) options:o error:pe];
  }, &retried);
  if (retried) err("what=compiler_unreachable site=msl retries=%d ok=%d", retried, lib ? 1 : 0);
  if (!lib) {
    error = "msl compile: " + str(e);
    return nil;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:@(shader.entry.c_str())];
  if (!fn) {
    error = "entry " + shader.entry + " not in library";
    return nil;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  {
    std::lock_guard lock(pipelineMutex);
    functions[key] = fn;
  }
  std::string line;
  {
    std::lock_guard sl(libStatsMutex);
    ++libCompiled;
    libMsTotal += ms;
    libMsMax = std::max(libMsMax, ms);
    if (libCompiled - libLogged >= 16 || libCompiled <= 4) {
      libLogged = libCompiled;
      char b[200];
      std::snprintf(b, sizeof b, "gpu.shader.lib compiled=%u dedup=%u prewarmed=%u ms_total=%.0f ms_max=%.0f last_ms=%.0f",
                    libCompiled, libDedup, libPrewarmed, libMsTotal, libMsMax, ms);
      line = b;
    }
  }
  if (!line.empty()) info(line);
  return fn;
}

bool MetalBackend::compileShader(uint64_t shaderId, std::string& error) {
  Impl* m = impl_.get();
  if (!ok() || !shaderId) {
    error = "no metal device";
    return false;
  }
  MslShader shader;
  // Same variant as prewarmShader: no fetch inputs, no PointSize write.
  return m->function(shaderId, {}, shader, error, true) != nil;
}

void MetalBackend::prewarmShader(uint64_t shaderId) {
  Impl* m = impl_.get();
  if (!ok() || !shaderId || m->closing) return;
  {
    std::lock_guard lock(m->pipelineMutex);
    if (!m->prewarmQueues[0]) {
      dispatch_queue_attr_t a = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0);
      m->prewarmQueues[0] = dispatch_queue_create("vpgpu.shader.prewarm0", a);
      m->prewarmQueues[1] = dispatch_queue_create("vpgpu.shader.prewarm1", a);
    }
  }
  dispatch_queue_t q = m->prewarmQueues[m->prewarmNext++ & 1];
  dispatch_async(q, ^{
    if (m->closing) return;
    MslShader shader;
    std::string error;
    if (m->shaders.find(shaderId, shader)) {  // functions are keyed by source (see function())
      const uint64_t key = functionKey(sourceKey(shader.source), {});
      std::lock_guard lock(m->pipelineMutex);
      if (m->functions.count(key) || m->compiling.count(key)) return;  // a worker got there first
    }
    // Most likely variant: no fetch inputs; vertex shaders without a PointSize write (non-point topology).
    if (m->function(shaderId, {}, shader, error, true)) {
      std::lock_guard sl(m->libStatsMutex);
      ++m->libPrewarmed;
    } else {
      m->info("gpu.shader.lib prewarm_fail id=" + std::to_string(shaderId) + " error=" + token(error));
    }
  });
}

namespace {
// Sampled blit: a triangle covering the viewport (the destination rect); `rect` maps the
// viewport's unit square to source UVs (origin, size; a negative size mirrors).
constexpr const char* kBlitMsl = R"(#include <metal_stdlib>
using namespace metal;
struct VpBlitV {
    float4 pos [[position]];
    float2 uv;
};
vertex VpBlitV vp_blit_vs(uint vid [[vertex_id]], constant float4& rect [[buffer(0)]]) {
    const float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    VpBlitV o;
    o.pos = float4(p.x * 2.0f - 1.0f, 1.0f - p.y * 2.0f, 0.0f, 1.0f);
    o.uv = rect.xy + p * rect.zw;
    return o;
}
fragment float4 vp_blit_fs(VpBlitV in [[stage_in]], texture2d<float> src [[texture(0)]],
                           sampler smp [[sampler(0)]]) {
    return src.sample(smp, in.uv, level(0.0f));
}
// Nearest (and integer formats, nearest only as Vulkan requires): the source texel computed
// exactly as Vulkan defines it, u = srcOrigin + scale * (dst pixel centre - dstOrigin), from the
// fragment's window position. An interpolated UV lands a hair off on the half-texel boundaries of
// 2:1 reductions and picks the other neighbour, which compounds down a mip chain.
// lo/hi: the destination's representable range per channel (integer formats: Metal wraps what
// does not fit, Vulkan clamps), as the bits of int4 (sint) or uint4 (uint).
struct VpBlitN {
    float2 srcOrigin, scale, dstOrigin, pad;
    uint4 lo, hi;
};
template <typename T, typename V>
V vp_blit_near2(texture2d<T> src, float2 pos, constant VpBlitN& n) {
    const float2 u = n.scale * (pos - n.dstOrigin) + n.srcOrigin;
    const float2 sz = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(floor(u), float2(0.0f), sz - 1.0f)));
}
fragment int4 vp_blit_fs_i(VpBlitV in [[stage_in]], texture2d<int> src [[texture(0)]], constant VpBlitN& n [[buffer(0)]]) {
    return clamp(vp_blit_near2<int, int4>(src, in.pos.xy, n), as_type<int4>(n.lo), as_type<int4>(n.hi));
}
fragment uint4 vp_blit_fs_u(VpBlitV in [[stage_in]], texture2d<uint> src [[texture(0)]], constant VpBlitN& n [[buffer(0)]]) {
    return min(vp_blit_near2<uint, uint4>(src, in.pos.xy, n), n.hi);
}
// Float formats: sampled (nearest) at the chosen texel's centre, so compressed sources work too.
fragment float4 vp_blit_fs_n(VpBlitV in [[stage_in]], texture2d<float> src [[texture(0)]], sampler smp [[sampler(0)]],
                             constant VpBlitN& n [[buffer(0)]]) {
    const float2 u = n.scale * (in.pos.xy - n.dstOrigin) + n.srcOrigin;
    const float2 sz = float2(src.get_width(), src.get_height());
    return src.sample(smp, (clamp(floor(u), float2(0.0f), sz - 1.0f) + 0.5f) / sz, level(0.0f));
}
// Multisampled source, sample 0 (resolves Metal cannot do: integer formats).
template <typename T, typename V>
V vp_blit_ms(texture2d_ms<T> src, float2 pos, constant VpBlitN& n) {
    const float2 u = n.scale * (pos - n.dstOrigin) + n.srcOrigin;
    const float2 sz = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(floor(u), float2(0.0f), sz - 1.0f)), 0u);
}
// Float formats: the average of the samples (vkCmdResolveImage regions Metal's resolve cannot do).
fragment float4 vp_blit_ms_f(VpBlitV in [[stage_in]], texture2d_ms<float> src [[texture(0)]], constant VpBlitN& n [[buffer(0)]]) {
    const float2 u = n.scale * (in.pos.xy - n.dstOrigin) + n.srcOrigin;
    const float2 sz = float2(src.get_width(), src.get_height());
    const uint2 p = uint2(clamp(floor(u), float2(0.0f), sz - 1.0f));
    float4 acc = float4(0.0f);
    const uint count = src.get_num_samples();
    for (uint i = 0; i < count; ++i) acc += src.read(p, i);
    return acc / float(max(count, 1u));
}
fragment int4 vp_blit_ms_i(VpBlitV in [[stage_in]], texture2d_ms<int> src [[texture(0)]], constant VpBlitN& n [[buffer(0)]]) {
    return clamp(vp_blit_ms<int, int4>(src, in.pos.xy, n), as_type<int4>(n.lo), as_type<int4>(n.hi));
}
fragment uint4 vp_blit_ms_u(VpBlitV in [[stage_in]], texture2d_ms<uint> src [[texture(0)]], constant VpBlitN& n [[buffer(0)]]) {
    return min(vp_blit_ms<uint, uint4>(src, in.pos.xy, n), n.hi);
}
// 3D sources: w is the normalized depth of the destination slice being drawn.
fragment float4 vp_blit_fs3(VpBlitV in [[stage_in]], texture3d<float> src [[texture(0)]],
                            sampler smp [[sampler(0)]], constant float& w [[buffer(0)]]) {
    return src.sample(smp, float3(in.uv, w), level(0.0f));
}
template <typename T, typename V>
V vp_blit_read3(texture3d<T> src, float3 uvw) {
    const float3 sz = float3(src.get_width(), src.get_height(), src.get_depth());
    return src.read(uint3(clamp(floor(uvw * sz), float3(0.0f), sz - 1.0f)));
}
fragment int4 vp_blit_fs3_i(VpBlitV in [[stage_in]], texture3d<int> src [[texture(0)]], constant float& w [[buffer(0)]]) {
    return vp_blit_read3<int, int4>(src, float3(in.uv, w));
}
fragment uint4 vp_blit_fs3_u(VpBlitV in [[stage_in]], texture3d<uint> src [[texture(0)]], constant float& w [[buffer(0)]]) {
    return vp_blit_read3<uint, uint4>(src, float3(in.uv, w));
}
)";
}  // namespace

// kinds[i]: 0 float, 1 int, 2 uint (the clear colour's interpretation and the output type).
id<MTLRenderPipelineState> MetalBackend::Impl::clearPipeline(MTLRenderPassDescriptor* rp, const ClearRects& c,
                                                             bool layered,
                                                             const std::array<int, kMaxColorAttachments>& kinds,
                                                             std::string& error) {
  NSUInteger samples = 1;
  std::string key = "L" + std::to_string(layered) + "M" + std::to_string(c.colorMask);
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    id<MTLTexture> t = rp.colorAttachments[i].texture;
    key += "," + std::to_string(t ? NSUInteger(t.pixelFormat) : 0) + ":" + std::to_string(kinds[i]);
    if (t) samples = t.sampleCount;
  }
  id<MTLTexture> dt = rp.depthAttachment.texture, stt = rp.stencilAttachment.texture;
  if (dt) samples = dt.sampleCount;
  else if (stt) samples = stt.sampleCount;
  key += "D" + std::to_string(dt ? NSUInteger(dt.pixelFormat) : 0) + "S" +
         std::to_string(stt ? NSUInteger(stt.pixelFormat) : 0) + "x" + std::to_string(samples);
  if (auto it = clearPipelines.find(key); it != clearPipelines.end()) {
    if (!it->second) error = "pipeline_failed";
    return it->second;
  }
  // Instances: one rect (NDC x0 y0 x1 y1), its depth and layer; triangle strip of 4 vertices.
  std::string src = "#include <metal_stdlib>\nusing namespace metal;\n"
                    "struct VpClearRect { float4 ndc; float depth; uint layer; uint pad0; uint pad1; };\n"
                    "struct VpClearV { float4 pos [[position]];";
  if (layered) src += " uint layer [[render_target_array_index]];";
  src += " };\n"
         "vertex VpClearV vp_clear_vs(uint vid [[vertex_id]], uint iid [[instance_id]],\n"
         "                            constant VpClearRect* rects [[buffer(0)]]) {\n"
         "    const VpClearRect r = rects[iid];\n"
         "    VpClearV o;\n"
         "    o.pos = float4((vid & 1u) ? r.ndc.z : r.ndc.x, (vid & 2u) ? r.ndc.w : r.ndc.y, r.depth, 1.0f);\n";
  if (layered) src += "    o.layer = r.layer;\n";
  src += "    return o;\n}\n";
  if (c.colorMask) {
    static const char* type[] = {"float4", "int4", "uint4"};
    src += "struct VpClearF {";
    for (int i = 0; i < kMaxColorAttachments; ++i)
      if ((c.colorMask >> i & 1) && rp.colorAttachments[i].texture)
        src += std::string(" ") + type[kinds[i]] + " c" + std::to_string(i) + " [[color(" + std::to_string(i) + ")]];";
    src += " };\nfragment VpClearF vp_clear_fs(constant uint4* colors [[buffer(0)]]) {\n    VpClearF o;\n";
    for (int i = 0; i < kMaxColorAttachments; ++i)
      if ((c.colorMask >> i & 1) && rp.colorAttachments[i].texture)
        src += std::string("    o.c") + std::to_string(i) + " = as_type<" + type[kinds[i]] + ">(colors[" +
               std::to_string(i) + "]);\n";
    src += "    return o;\n}\n";
  }
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.languageVersion = MTLLanguageVersion3_2;
  NSError* e = nil;
  id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
    return [device newLibraryWithSource:@(src.c_str()) options:o error:pe];
  });
  id<MTLFunction> vs = lib ? [lib newFunctionWithName:@"vp_clear_vs"] : nil;
  id<MTLFunction> fs = lib && c.colorMask ? [lib newFunctionWithName:@"vp_clear_fs"] : nil;
  id<MTLRenderPipelineState> p = nil;
  if (vs && (fs || !c.colorMask)) {
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.label = @"vpgpu.clear";
    d.vertexFunction = vs;
    if (fs) d.fragmentFunction = fs;
    for (int i = 0; i < kMaxColorAttachments; ++i) {
      id<MTLTexture> t = rp.colorAttachments[i].texture;
      if (!t) continue;
      d.colorAttachments[i].pixelFormat = t.pixelFormat;
      d.colorAttachments[i].writeMask = (c.colorMask >> i & 1) && fs ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    }
    if (dt) d.depthAttachmentPixelFormat = dt.pixelFormat;
    if (stt) d.stencilAttachmentPixelFormat = stt.pixelFormat;
    d.rasterSampleCount = samples;
    if (layered) d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    p = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
      return [device newRenderPipelineStateWithDescriptor:d error:pe];
    });
  }
  if (!p) error = "clear pipeline: " + str(e);
  // A compiler that could not be reached is retried on the next clear, not cached as broken.
  if (p || !compilerUnreachable(e)) clearPipelines[key] = p;
  return p;
}

id<MTLDepthStencilState> MetalBackend::Impl::clearDepthState(bool depth, bool stencil) {
  id<MTLDepthStencilState> __strong& s = clearDepthStates[(depth ? 1 : 0) | (stencil ? 2 : 0)];
  if (s) return s;
  MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
  d.depthCompareFunction = MTLCompareFunctionAlways;
  d.depthWriteEnabled = depth;
  if (stencil) {
    MTLStencilDescriptor* sd = [MTLStencilDescriptor new];
    sd.stencilCompareFunction = MTLCompareFunctionAlways;
    sd.stencilFailureOperation = MTLStencilOperationReplace;
    sd.depthFailureOperation = MTLStencilOperationReplace;
    sd.depthStencilPassOperation = MTLStencilOperationReplace;
    sd.readMask = 0xFF;
    sd.writeMask = 0xFF;
    d.frontFaceStencil = sd;
    d.backFaceStencil = sd;
  }
  d.label = @"vpgpu.clear";
  s = [device newDepthStencilStateWithDescriptor:d];
  return s;
}

id<MTLRenderPipelineState> MetalBackend::Impl::blitPipeline(MTLPixelFormat format, int variant, std::string& error) {
  const NSUInteger key = NSUInteger(format) * 16 + NSUInteger(variant & 15);
  if (auto it = blitPipelines.find(key); it != blitPipelines.end()) {
    if (!it->second) error = "pipeline_failed";
    return it->second;
  }
  if (!blitLibrary) {
    const double now = steadySeconds();
    if (blitLibraryFailed || now < blitLibraryRetryAt) {
      error = "library_failed";
      return nil;
    }
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion3_2;
    NSError* e = nil;
    blitLibrary = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
      return [device newLibraryWithSource:@(kBlitMsl) options:o error:pe];
    });
    for (int i = 0; i < 2 && blitLibrary; ++i) {
      MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
      sd.minFilter = sd.magFilter = i ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
      sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
      sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
      blitSamplers[i] = [device newSamplerStateWithDescriptor:sd];
      if (!blitSamplers[i]) blitLibrary = nil;
    }
    if (!blitLibrary) {
      if (compilerUnreachable(e))
        blitLibraryRetryAt = now + 2.0;  // the service comes back: try again in 2 s
      else
        blitLibraryFailed = true;
      error = "library: " + str(e);
      return nil;
    }
  }
  MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
  d.label = @"vpgpu.blit";
  d.vertexFunction = [blitLibrary newFunctionWithName:@"vp_blit_vs"];
  // variant: bits 0-1 float (linear) / int / uint / float nearest, bit 2 a 3D source, bit 3 a
  // multisampled source (sample 0; never 3D).
  static NSString* const fs[16] = {@"vp_blit_fs",   @"vp_blit_fs_i",  @"vp_blit_fs_u",  @"vp_blit_fs_n",
                                   @"vp_blit_fs3",  @"vp_blit_fs3_i", @"vp_blit_fs3_u", @"vp_blit_fs3",
                                   @"vp_blit_ms_f", @"vp_blit_ms_i",  @"vp_blit_ms_u",  @"vp_blit_ms_f",
                                   @"vp_blit_ms_f", @"vp_blit_ms_i",  @"vp_blit_ms_u",  @"vp_blit_ms_f"};
  d.fragmentFunction = [blitLibrary newFunctionWithName:fs[variant & 15]];
  d.colorAttachments[0].pixelFormat = format;
  d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
  NSError* e = nil;
  id<MTLRenderPipelineState> p =
      d.vertexFunction && d.fragmentFunction
          ? compileRetrying(&e, [&](NSError* __autoreleasing* pe) { return [device newRenderPipelineStateWithDescriptor:d error:pe]; })
          : nil;
  if (!p) error = "pipeline: " + str(e);
  // A failure is cached too (logged once, not retried every frame) — unless the compiler was
  // only unreachable.
  if (p || !compilerUnreachable(e)) blitPipelines[key] = p;
  return p;
}

constexpr const char* kD24Msl = R"(#include <metal_stdlib>
using namespace metal;
struct VpD24 { uint width, height, images, srcRow, srcImage, dstRow, dstImage, pad; };
kernel void vp_d24_to_float(device const uint* src [[buffer(0)]], device float* dst [[buffer(1)]],
                            constant VpD24& p [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.width || g.y >= p.height || g.z >= p.images) return;
  const uint v = src[g.z * p.srcImage + g.y * p.srcRow + g.x] & 0xFFFFFFu;
  dst[g.z * p.dstImage + g.y * p.dstRow + g.x] = float(v) / 16777215.0f;
}
kernel void vp_float_to_d24(device const float* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                            constant VpD24& p [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.width || g.y >= p.height || g.z >= p.images) return;
  const float d = clamp(src[g.z * p.srcImage + g.y * p.srcRow + g.x], 0.0f, 1.0f);
  dst[g.z * p.dstImage + g.y * p.dstRow + g.x] = uint(rint(d * 16777215.0f));
}
)";

id<MTLComputePipelineState> MetalBackend::Impl::d24Pipeline(bool toUnorm, std::string& error) {
  if (d24Pipelines[toUnorm ? 1 : 0]) return d24Pipelines[toUnorm ? 1 : 0];
  if (d24Failed) {
    error = "d24_failed";
    return nil;
  }
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.languageVersion = MTLLanguageVersion3_2;
  NSError* e = nil;
  id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
    return [device newLibraryWithSource:@(kD24Msl) options:o error:pe];
  });
  for (int i = 0; i < 2 && lib; ++i) {
    id<MTLFunction> f = [lib newFunctionWithName:i ? @"vp_float_to_d24" : @"vp_d24_to_float"];
    d24Pipelines[i] = f ? [device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (!d24Pipelines[i]) lib = nil;
  }
  if (!lib) {
    d24Failed = !compilerUnreachable(e);
    d24Pipelines[0] = d24Pipelines[1] = nil;
    error = "d24: " + str(e);
    return nil;
  }
  return d24Pipelines[toUnorm ? 1 : 0];
}

bool MetalBackend::Impl::buildRender(const PipelineState& s, Pipeline& p, std::string& error) {
  MslShader vs, fs;
  MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
  d.label = @(s.label.c_str());
  if (s.fetch.mask && !tileMemory) {
    error = "framebuffer fetch needs an Apple GPU";
    return false;
  }
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    const FormatInfo& f = formatInfo(s.colorFormats[i]);
    if ((s.fetch.mask >> i & 1) && (s.colorFormats[i] == Format::Undefined || f.depth() || f.stencil())) {
      error = "fetch input " + std::to_string(i) + " has no colour format";
      return false;
    }
  }
  // Functions are checked before they reach the descriptor: Metal's setters assert on nil (a
  // shader that is not registered, e.g. a recipe whose SPIR-V this process never saw).
  id<MTLFunction> vfn = function(s.vertexShader, {}, vs, error, s.primitive != Primitive::Points);
  if (!vfn) return false;
  d.vertexFunction = vfn;
  if (s.fragmentShader) {  // none = depth-only
    // Vulkan ignores FragDepth writes without a depth attachment; Metal rejects the pipeline.
    id<MTLFunction> ffn = function(s.fragmentShader, s.fetch, fs, error, false, !formatInfo(s.depthFormat).depth());
    if (!ffn) return false;
    d.fragmentFunction = ffn;
  }
  // Attachments the fragment shader does not write (fetch-only G-buffer, depth-only pipelines)
  // must keep their contents: nothing written, nothing blended.
  const uint8_t written = s.fragmentShader ? fs.colorOutputs : 0;
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    MTLRenderPipelineColorAttachmentDescriptor* c = d.colorAttachments[i];
    const BlendState& b = s.blend[i];
    const bool writes = written >> i & 1;
    c.pixelFormat = pixelFormat(s.colorFormats[i]);
    c.writeMask = writes ? writeMask(b.writeMask) : MTLColorWriteMaskNone;
    c.blendingEnabled = writes && b.enabled;
    c.sourceRGBBlendFactor = blendFactor(b.srcColor);
    c.destinationRGBBlendFactor = blendFactor(b.dstColor);
    c.sourceAlphaBlendFactor = blendFactor(b.srcAlpha);
    c.destinationAlphaBlendFactor = blendFactor(b.dstAlpha);
    c.rgbBlendOperation = blendOp(b.colorOp);
    c.alphaBlendOperation = blendOp(b.alphaOp);
    for (BlendFactor f : {b.srcColor, b.dstColor, b.srcAlpha, b.dstAlpha})
      p.blendConstant = p.blendConstant || (c.blendingEnabled && (f == BlendFactor::Constant || f == BlendFactor::OneMinusConstant ||
                                                                  f == BlendFactor::ConstantAlpha || f == BlendFactor::OneMinusConstantAlpha));
  }
  const FormatInfo& df = formatInfo(s.depthFormat);
  const MTLPixelFormat dsf = pixelFormat(s.depthFormat);
  if (df.depth()) d.depthAttachmentPixelFormat = dsf;
  if (df.stencil()) d.stencilAttachmentPixelFormat = dsf;
  if (s.samples > 1 && ![device supportsTextureSampleCount:s.samples]) {
    error = "sample count " + std::to_string(s.samples) + " not supported by this GPU";
    return false;
  }
  d.rasterSampleCount = s.samples;
  d.alphaToCoverageEnabled = s.alphaToCoverage;
  d.inputPrimitiveTopology = topologyClass(s.primitive);

  NSError* e = nil;
  // The loaded archives are never mutated, so concurrent lookups need no lock.
  NSArray<id<MTLBinaryArchive>>* ars = lookupArchives;
  bool hit = false;
  if (ars.count) {  // archive lookup only: a miss fails fast instead of compiling
    d.binaryArchives = ars;
    p.render = [device newRenderPipelineStateWithDescriptor:d options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                 reflection:nil error:&e];
    hit = p.render != nil;
    e = nil;
    d.binaryArchives = nil;  // the full compile below and the archived copy never reference them
  }
  if (!p.render) {
    int retried = 0;
    p.render = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
      return [device newRenderPipelineStateWithDescriptor:d error:pe];
    }, &retried);
    if (retried) err("what=compiler_unreachable site=render retries=%d ok=%d", retried, p.render ? 1 : 0);
  }
  if (!p.render) {
    error = "render pipeline: " + str(e);
    return false;
  }
  if (archiveOn && !hit) queueArchive([d copy], false, s.label);
  MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
  ds.depthCompareFunction = s.depthTest ? compareFn(s.depthCompare) : MTLCompareFunctionAlways;
  ds.depthWriteEnabled = s.depthTest && s.depthWrite;
  if (s.stencilTest) {
    ds.frontFaceStencil = stencilFace(s.stencilFront);
    ds.backFaceStencil = stencilFace(s.stencilBack);
  }
  p.depth = [device newDepthStencilStateWithDescriptor:ds];
  p.samples = std::max<NSUInteger>(1, s.samples);
  p.depthTest = s.depthTest, p.depthWrite = s.depthWrite, p.stencilTest = s.stencilTest;
  p.depthCompare = s.depthCompare;
  p.stencilFront = s.stencilFront, p.stencilBack = s.stencilBack;
  p.cull = s.cull == CullMode::Front ? MTLCullModeFront : s.cull == CullMode::Back ? MTLCullModeBack : MTLCullModeNone;
  p.fetch = s.fetch;
  p.stage[0] = {vs.arguments, vs.argumentBufferIndex};
  if (s.fragmentShader) p.stage[1] = {fs.arguments, fs.argumentBufferIndex};
  p.drawIndex = vs.source.find("[[buffer(29)]]") != std::string::npos;
  return true;
}

bool MetalBackend::Impl::buildCompute(const PipelineState& s, Pipeline& p, std::string& error) {
  MslShader cs;
  MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
  d.label = @(s.label.c_str());
  // -[MTLComputePipelineDescriptor setComputeFunction:] asserts on nil (CTS log 60: recipes whose
  // shader this process never registered aborted the process from the compile workers).
  id<MTLFunction> cfn = function(s.computeShader, {}, cs, error);
  if (!cfn) return false;
  d.computeFunction = cfn;
  NSError* e = nil;
  NSArray<id<MTLBinaryArchive>>* ars = lookupArchives;  // immutable: no lock
  bool hit = false;
  if (ars.count) {
    d.binaryArchives = ars;
    p.compute = [device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                   reflection:nil error:&e];
    hit = p.compute != nil;
    e = nil;
    d.binaryArchives = nil;
  }
  if (!p.compute) {
    int retried = 0;
    p.compute = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
      return [device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:pe];
    }, &retried);
    if (retried) err("what=compiler_unreachable site=compute retries=%d ok=%d", retried, p.compute ? 1 : 0);
  }
  if (!p.compute) {
    error = "compute pipeline: " + str(e);
    return false;
  }
  if (archiveOn && !hit) queueArchive([d copy], true, s.label);
  p.threads = MTLSizeMake(cs.threadsPerGroup[0], cs.threadsPerGroup[1], cs.threadsPerGroup[2]);
  p.stage[0] = {cs.arguments, cs.argumentBufferIndex};
  return true;
}

// ---------------------------------------------------------------------------------------------
// Submission

namespace {

// The Metal encoder currently open. A core compute encoder becomes compute and blit encoders as
// its items alternate between dispatches and copies (M4: one MTL4ComputeCommandEncoder does both).
struct Open {
  enum Kind { None, Render, Compute, Blit } kind = None;
  id<MTLRenderCommandEncoder> render;
  id<MTLComputeCommandEncoder> compute;
  id<MTLBlitCommandEncoder> blit;
  id<MTLCommandEncoder> any() const {
    if (render) return render;
    if (compute) return compute;
    return blit;
  }
};

// ---- Per-pass diagnostic dump (VP_DUMP_FRAME / VP_DUMP_DIR) ----------------------------------
// At one frame, every render pass's colour attachment 0 (or its depth when it has no colour) and
// every storage texture a compute encoder wrote is copied, once the encoder ends, into a shared
// buffer; after the frame completes each one becomes an 8-bit PNG preview plus a stats log line.
// Off by default; the frame is fully serialised while dumping (diagnostics only).
constexpr uint64_t kDumpMaxItemBytes = 128ull << 20;
constexpr uint64_t kDumpMaxTotalBytes = 768ull << 20;
constexpr uint32_t kDumpMaxItems = 256;
constexpr uint32_t kDumpMaxWidth = 1024;

struct DumpItem {
  uint32_t index = 0;
  std::string what;   // render | compute
  std::string label;
  MTLPixelFormat format = MTLPixelFormatInvalid;
  uint32_t width = 0, height = 0;
  uint32_t draws = 0;
  int load = -1, store = -1;
  NSUInteger row = 0;
  id<MTLBuffer> buffer;  // nil: nothing copied (see note)
  std::string note;
};
struct DumpCopy {
  id<MTLTexture> texture;
  NSUInteger slice = 0, level = 0, z = 0;
  MTLBlitOption options = MTLBlitOptionNone;
  size_t item = 0;
};

// Bytes per pixel of the formats the preview understands (0 = log only).
uint32_t dumpBytesPerPixel(MTLPixelFormat f, MTLBlitOption& opt) {
  opt = MTLBlitOptionNone;
  switch (f) {
    case MTLPixelFormatR8Unorm: return 1;
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatDepth16Unorm: return 2;
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatDepth32Float: return 4;
    case MTLPixelFormatDepth32Float_Stencil8: opt = MTLBlitOptionDepthFromDepthStencil; return 4;
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatRG32Float: return 8;
    case MTLPixelFormatRGBA32Float: return 16;
    default: return 0;
  }
}

const char* dumpFormatName(MTLPixelFormat f) {
  switch (f) {
    case MTLPixelFormatR8Unorm: return "R8";
    case MTLPixelFormatRG8Unorm: return "RG8";
    case MTLPixelFormatR16Float: return "R16F";
    case MTLPixelFormatR16Unorm: return "R16";
    case MTLPixelFormatDepth16Unorm: return "D16";
    case MTLPixelFormatRGBA8Unorm: return "RGBA8";
    case MTLPixelFormatRGBA8Unorm_sRGB: return "RGBA8sRGB";
    case MTLPixelFormatBGRA8Unorm: return "BGRA8";
    case MTLPixelFormatBGRA8Unorm_sRGB: return "BGRA8sRGB";
    case MTLPixelFormatRGB10A2Unorm: return "RGB10A2";
    case MTLPixelFormatRG11B10Float: return "RG11B10F";
    case MTLPixelFormatRG16Float: return "RG16F";
    case MTLPixelFormatR32Float: return "R32F";
    case MTLPixelFormatDepth32Float: return "D32F";
    case MTLPixelFormatDepth32Float_Stencil8: return "D32FS8";
    case MTLPixelFormatRGBA16Float: return "RGBA16F";
    case MTLPixelFormatRG32Float: return "RG32F";
    case MTLPixelFormatRGBA32Float: return "RGBA32F";
    default: return nullptr;
  }
}

std::string dumpFormatToken(MTLPixelFormat f) {
  const char* n = dumpFormatName(f);
  return n ? std::string(n) : "mtl" + std::to_string(unsigned(f));
}

float halfToFloat(uint16_t h) {
  const uint32_t s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0) v = std::ldexp(float(m), -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp(float(m | 1024), int(e) - 25);
  return s ? -v : v;
}
// Unsigned small floats (5-bit exponent, `mb` mantissa bits) of RG11B10Float.
float smallFloat(uint32_t bits, int mb) {
  const uint32_t e = bits >> mb, m = bits & ((1u << mb) - 1);
  if (e == 0) return std::ldexp(float(m), -14 - mb);
  if (e == 31) return m ? NAN : INFINITY;
  return std::ldexp(float(m | (1u << mb)), int(e) - 15 - mb);
}

// Decodes one pixel to raw floats; returns how the preview should map it.
enum class DumpKind { Unorm, Float, Depth };
DumpKind dumpDecode(MTLPixelFormat f, const uint8_t* p, float out[4]) {
  auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, p + o, 4); return v; };
  auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, p + o, 2); return v; };
  auto f32 = [&](size_t o) { float v; std::memcpy(&v, p + o, 4); return v; };
  out[0] = out[1] = out[2] = 0, out[3] = 1;
  switch (f) {
    case MTLPixelFormatR8Unorm: out[0] = out[1] = out[2] = p[0] / 255.f; return DumpKind::Unorm;
    case MTLPixelFormatRG8Unorm: out[0] = p[0] / 255.f, out[1] = p[1] / 255.f; return DumpKind::Unorm;
    case MTLPixelFormatR16Unorm: out[0] = out[1] = out[2] = u16(0) / 65535.f; return DumpKind::Unorm;
    case MTLPixelFormatDepth16Unorm: out[0] = out[1] = out[2] = u16(0) / 65535.f; return DumpKind::Depth;
    case MTLPixelFormatR16Float: out[0] = out[1] = out[2] = halfToFloat(u16(0)); return DumpKind::Float;
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
      for (int i = 0; i < 4; ++i) out[i] = p[i] / 255.f;
      return DumpKind::Unorm;
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
      out[0] = p[2] / 255.f, out[1] = p[1] / 255.f, out[2] = p[0] / 255.f, out[3] = p[3] / 255.f;
      return DumpKind::Unorm;
    case MTLPixelFormatRGB10A2Unorm: {
      const uint32_t v = u32(0);
      out[0] = (v & 1023) / 1023.f, out[1] = ((v >> 10) & 1023) / 1023.f, out[2] = ((v >> 20) & 1023) / 1023.f;
      out[3] = (v >> 30) / 3.f;
      return DumpKind::Unorm;
    }
    case MTLPixelFormatRG11B10Float: {
      const uint32_t v = u32(0);
      out[0] = smallFloat(v & 2047, 6), out[1] = smallFloat((v >> 11) & 2047, 6), out[2] = smallFloat(v >> 22, 5);
      return DumpKind::Float;
    }
    case MTLPixelFormatRG16Float: out[0] = halfToFloat(u16(0)), out[1] = halfToFloat(u16(2)); return DumpKind::Float;
    case MTLPixelFormatR32Float: out[0] = out[1] = out[2] = f32(0); return DumpKind::Float;
    case MTLPixelFormatDepth32Float:
    case MTLPixelFormatDepth32Float_Stencil8: out[0] = out[1] = out[2] = f32(0); return DumpKind::Depth;
    case MTLPixelFormatRGBA16Float:
      for (int i = 0; i < 4; ++i) out[i] = halfToFloat(u16(2 * i));
      return DumpKind::Float;
    case MTLPixelFormatRG32Float: out[0] = f32(0), out[1] = f32(4); return DumpKind::Float;
    case MTLPixelFormatRGBA32Float:
      for (int i = 0; i < 4; ++i) out[i] = f32(4 * i);
      return DumpKind::Float;
    default: return DumpKind::Unorm;
  }
}

bool writePng(const std::string& path, const std::vector<uint8_t>& rgbx, uint32_t w, uint32_t h) {
  bool ok = false;
  CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
  CFDataRef data = CFDataCreate(nullptr, rgbx.data(), CFIndex(rgbx.size()));
  CGDataProviderRef prov = data ? CGDataProviderCreateWithCFData(data) : nullptr;
  CGImageRef img = (cs && prov) ? CGImageCreate(w, h, 8, 32, size_t(w) * 4, cs, (CGBitmapInfo)kCGImageAlphaNoneSkipLast,
                                                prov, nullptr, false, kCGRenderingIntentDefault)
                                : nullptr;
  CFURLRef url = CFURLCreateFromFileSystemRepresentation(nullptr, reinterpret_cast<const UInt8*>(path.c_str()),
                                                         CFIndex(path.size()), false);
  CGImageDestinationRef dst = (img && url) ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr) : nullptr;
  if (dst) {
    CGImageDestinationAddImage(dst, img, nullptr);
    ok = CGImageDestinationFinalize(dst);
    CFRelease(dst);
  }
  if (url) CFRelease(url);
  if (img) CGImageRelease(img);
  if (prov) CGDataProviderRelease(prov);
  if (data) CFRelease(data);
  if (cs) CGColorSpaceRelease(cs);
  return ok;
}

// Turns one copied item into a PNG (≤ kDumpMaxWidth wide) and a stats line.
std::string dumpWrite(const DumpItem& it, uint64_t frame, const std::string& dir) {
  char head[160];
  std::snprintf(head, sizeof head, "gpu.dump.stats pass=%u fmt=%s %ux%u", it.index, dumpFormatToken(it.format).c_str(),
                it.width, it.height);
  if (!it.buffer) return std::string(head) + " skipped=" + (it.note.empty() ? "?" : it.note);
  MTLBlitOption opt;
  const uint32_t bpp = dumpBytesPerPixel(it.format, opt);
  const uint8_t* base = static_cast<const uint8_t*>(it.buffer.contents);
  if (!bpp || !base || !it.width || !it.height) return std::string(head) + " skipped=no_data";
  const uint32_t step = (it.width + kDumpMaxWidth - 1) / kDumpMaxWidth;
  const uint32_t ow = (it.width + step - 1) / step, oh = (it.height + step - 1) / step;
  std::vector<float> px(size_t(ow) * oh * 4);
  DumpKind kind = DumpKind::Unorm;
  double sum[4] = {}, mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, mn[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  uint64_t nan = 0, black = 0, n = 0;
  for (uint32_t y = 0; y < oh; ++y)
    for (uint32_t x = 0; x < ow; ++x) {
      float* o = &px[(size_t(y) * ow + x) * 4];
      kind = dumpDecode(it.format, base + size_t(y) * step * it.row + size_t(x) * step * bpp, o);
      bool isNan = false;
      for (int c = 0; c < 4; ++c) {
        if (std::isnan(o[c])) {
          isNan = true;
          continue;
        }
        sum[c] += o[c], mx[c] = std::max<double>(mx[c], o[c]), mn[c] = std::min<double>(mn[c], o[c]);
      }
      nan += isNan;
      black += !isNan && std::fabs(o[0]) < 1e-3f && std::fabs(o[1]) < 1e-3f && std::fabs(o[2]) < 1e-3f;
      ++n;
    }
  // Depth: stretched between its own min and max so the shape shows.
  const float dlo = float(mn[0]), dhi = float(mx[0]);
  std::vector<uint8_t> out(size_t(ow) * oh * 4);
  for (size_t i = 0; i < size_t(ow) * oh; ++i)
    for (int c = 0; c < 4; ++c) {
      float v = px[i * 4 + c];
      if (std::isnan(v)) v = c == 1 ? 1.f : 0.f;  // NaN shows as green
      else if (kind == DumpKind::Depth) v = dhi > dlo ? (v - dlo) / (dhi - dlo) : 0.f;
      else if (kind == DumpKind::Float) v = std::pow(std::min(std::max(v, 0.f), 1.f), 1.f / 2.2f);
      out[i * 4 + c] = uint8_t(std::min(std::max(v, 0.f), 1.f) * 255.f + 0.5f);
    }
  const std::string name = "pass-" + std::to_string(frame) + "-" + std::to_string(it.index) + "-" +
                           dumpFormatToken(it.format) + "-" + std::to_string(it.width) + "x" + std::to_string(it.height) + ".png";
  const bool ok = writePng(dir + "/" + name, out, ow, oh);
  char tail[400];
  const double d = n ? double(n) : 1.0;
  std::snprintf(tail, sizeof tail,
                " mean=%.4g,%.4g,%.4g,%.4g min=%.4g,%.4g,%.4g,%.4g max=%.4g,%.4g,%.4g,%.4g nan=%llu black_pct=%.1f png=%s",
                sum[0] / d, sum[1] / d, sum[2] / d, sum[3] / d, mn[0], mn[1], mn[2], mn[3], mx[0], mx[1], mx[2], mx[3],
                (unsigned long long)nan, 100.0 * double(black) / d, ok ? name.c_str() : "write_failed");
  return std::string(head) + tail;
}

}  // namespace

// ---- completion, lost device, bounded waits ------------------------------------------------
namespace {
int64_t steadyNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
// Waits on the GPU give up on the device after this long with work in flight and no command
// buffer completing (Metal's own hang detection reports a stuck command buffer well before).
constexpr int64_t kStallNs = 20'000'000'000;
// After a loss, command buffers already queued get this long to finish before waits stop for them.
constexpr int64_t kLostGraceNs = 2'000'000'000;
constexpr int64_t kSliceNs = 100'000'000;
// A wait loop that comes back this long after its last pass was not running (app suspended or in
// the background): the stall clock starts over instead of counting the pause as a stuck GPU.
constexpr int64_t kPauseNs = 2'000'000'000;
}  // namespace

void MetalBackend::Impl::frameFinished(uint64_t frameIndex) {
  {
    std::lock_guard lock(doneMutex);
    if (frameIndex == finishedBelow) {
      ++finishedBelow;
      while (!finishedAhead.empty() && *finishedAhead.begin() == finishedBelow) {
        finishedAhead.erase(finishedAhead.begin());
        ++finishedBelow;
      }
    } else if (frameIndex > finishedBelow) {
      finishedAhead.insert(frameIndex);
    }
  }
  completions.fetch_add(1);
  doneCv.notify_all();
}

void MetalBackend::Impl::markLost(const std::string& why) {
  if (lostClaimed.exchange(true)) return;
  lostAtNs.store(steadyNs());  // before the flag: a drain that sees the flag also sees the time
  lostFlag.store(true);
  err("what=device_lost reason=%s (no more GPU work; waits return VK_ERROR_DEVICE_LOST)", token(why).c_str());
  // Native command buffers waiting on the ordering event would never run: release them.
  if (void* ev = nativeEventRaw.load()) ((__bridge id<MTLSharedEvent>)ev).signaledValue = ~uint64_t(0) >> 1;
  { std::lock_guard lock(doneMutex); }  // a waiter between its check and its wait sees the flag
  doneCv.notify_all();
}

bool MetalBackend::Impl::lostByError(NSError* e) const {
  if (!e) return false;
  const std::string why = str(e);
  if (e.code == MTLCommandBufferErrorMemoryless || why.find("memoryless") != std::string::npos ||
      why.find("ParameterBuffer") != std::string::npos)
    return false;  // tiler flush: recoverable (memoryless off, see memorylessFailed)
  // Always a loss: the process may no longer use the GPU — 4 AccessRevoked (Blacklisted),
  // 11 DeviceRemoved, or IOGPU's "Ignored (for causing prior/excessive GPU errors)": every later
  // command buffer would fail too.
  const long c = static_cast<long>(e.code);
  if (c == 4 || c == 11) return true;
  for (const char* k : {"Ignored", "Blacklisted", "Revoked", "Removed"})
    if (why.find(k) != std::string::npos) return true;
  // A single hung or faulting frame (IOGPU "Caused GPU Hang Error", page faults) and work refused
  // while the app is in the background (7 NotPermitted) leave the queue usable: a game goes on
  // (a stuck queue is still caught by the 20 s stall rule). Conformance runs set
  // VPVK_DEVICE_LOST_STRICT=1, where every such failure is a loss, as Vulkan defines it.
  return strictLost;
}

bool MetalBackend::Impl::takeSlotSemaphore() {
  uint64_t seen = completions.load();
  int64_t quietSince = steadyNs(), lastPass = quietSince;
  for (;;) {
    if (lostFlag.load()) return false;
    if (dispatch_semaphore_wait(inFlight, dispatch_time(DISPATCH_TIME_NOW, kSliceNs)) == 0) {
      if (!lostFlag.load()) return true;
      dispatch_semaphore_signal(inFlight);
      return false;
    }
    const uint64_t c = completions.load();
    const int64_t now = steadyNs();
    if (c != seen || now - lastPass > kPauseNs) {
      seen = c;
      quietSince = now;
    } else if (now - quietSince > kStallNs) {
      markLost("gpu_stuck (every frame slot busy, no command buffer completed for 20 s)");
    }
    lastPass = now;
  }
}

MetalBackend::WaitResult MetalBackend::Impl::drain(uint64_t timeoutNs, bool& drained) {
  // Holding every slot means every completion handler has run (not only that the GPU finished).
  drained = false;
  std::lock_guard drainLock(drainMutex);
  const int64_t start = steadyNs();
  const int64_t deadline = timeoutNs >= uint64_t(INT64_MAX) - uint64_t(start) ? INT64_MAX : start + int64_t(timeoutNs);
  uint64_t seen = completions.load();
  int64_t quietSince = start, lastPass = start;
  size_t got = 0;
  WaitResult r = WaitResult::Done;
  while (got < slots.size()) {
    const int64_t now = steadyNs();
    const int64_t slice = std::max<int64_t>(0, std::min(kSliceNs, deadline - now));
    if (dispatch_semaphore_wait(inFlight, dispatch_time(DISPATCH_TIME_NOW, slice)) == 0) {
      ++got;
      continue;
    }
    const int64_t t = steadyNs();
    if (lostFlag.load() && t - lostAtNs.load() > kLostGraceNs) {
      r = WaitResult::Lost;
      break;
    }
    if (t >= deadline) {
      r = WaitResult::Timeout;
      break;
    }
    const uint64_t c = completions.load();
    if (c != seen || t - lastPass > kPauseNs) {
      seen = c;
      quietSince = t;
    } else if (t - quietSince > kStallNs) {
      markLost("gpu_stuck (waiting for idle, no command buffer completed for 20 s)");
    }
    lastPass = t;
  }
  for (size_t i = 0; i < got; ++i) dispatch_semaphore_signal(inFlight);
  drained = got == slots.size();
  if (drained && lostFlag.load()) r = WaitResult::Lost;
  return r;
}

MetalBackend::Impl::Slot* MetalBackend::Impl::acquireSlot() {
  if (!takeSlotSemaphore()) return nullptr;
  // Consecutive command buffers use different slots, so the frontier carried over from the
  // previous one never holds a fence this one updates.
  Slot& slot = slots[submits++ % slots.size()];
  // The slot's previous frame finished, and so did every frame that could use its leftovers:
  // only now do they leave the residency set and get released.
  for (id obj : slot.retired)
    if (residency && [obj conformsToProtocol:@protocol(MTLAllocation)]) {
      [residency removeAllocation:obj];
      residencyDirty = true;
    }
  slot.retired = std::move(graveyard);
  graveyard.clear();
  if (residencyDirty) {
    [residency commit];
    residencyDirty = false;
  }
  return &slot;
}

// Native client work (encodeNative). Resources are hazard-untracked, so queue order alone orders
// nothing: the client's encoders (which know nothing of the frontier fences) are fenced in with an
// MTLEvent signalled in one command buffer and waited on at the start of the next — the documented
// cross-command-buffer use of an event, nothing that depends on how a signal and a wait behave
// inside a single command buffer. Three command buffers, committed in this order:
//   head:   a blit encoder waits on the frontier (the tail of every earlier frame; transitive, see
//           `frontier`), then signals A: A fires once the frames' work is complete;
//   client: waits for A, the client's encoders, signals B;
//   tail:   waits for B, a blit encoder updates a fence that becomes the whole frontier (the next
//           frame's first encoder waits on it) and the frame slot is released when it completes.
// A watchdog logs `what=native_stuck` with the three statuses if the tail has not completed after
// 2 s, so a hang here is named in the log instead of showing up as a frozen picture.
bool MetalBackend::Impl::encodeNative(NativeEncodeFn fn, void* user) {
  if (!device || !queue || lostFlag.load()) return false;
  if (!nativeEvent) {
    nativeEvent = [device newSharedEvent];
    nativeEventRaw.store((__bridge void*)nativeEvent);
  }
  if (!nativeScratch) nativeScratch = [device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
  if (!nativeEvent || !nativeScratch) {
    err("what=native_event");
    return false;
  }
  Slot* slotp = acquireSlot();
  if (!slotp) return false;  // device lost
  Slot& slot = *slotp;
  if (slot.fences.empty()) slot.fences.push_back([device newFence]);
  id<MTLFence> tailFence = slot.fences[0];
  const uint64_t n = ++nativeSubmits;
  Impl* impl = this;  // the destructor waits for every slot, so no handler outlives *this

  auto make = [&](const char* part) {
    MTLCommandBufferDescriptor* cbd = [MTLCommandBufferDescriptor new];
    cbd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> c = [queue commandBufferWithDescriptor:cbd];
    c.label = [NSString stringWithFormat:@"vpgpu.native%llu.%s", (unsigned long long)n, part];
    [c addCompletedHandler:^(id<MTLCommandBuffer> finished) {
      if (finished.status != MTLCommandBufferStatusError) return;
      std::string encs;
      NSArray* infos = finished.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
      for (id<MTLCommandBufferEncoderInfo> i in infos)
        if (i.errorState == MTLCommandEncoderErrorStateFaulted || i.errorState == MTLCommandEncoderErrorStateAffected)
          encs += (encs.empty() ? "" : ",") + token(i.label.UTF8String ?: "?");
      impl->err("what=native_command_buffer n=%llu part=%s error=%s encoders=%s", (unsigned long long)n, part,
                token(str(finished.error)).c_str(), encs.c_str());
      if (impl->lostByError(finished.error))
        impl->markLost(std::string("native_command_buffer ") + part + " " + str(finished.error));
    }];
    return c;
  };
  id<MTLCommandBuffer> headCb = make("head");
  id<MTLCommandBuffer> clientCb = make("client");
  id<MTLCommandBuffer> tailCb = make("tail");
  const uint64_t afterFrames = ++nativeValue, afterClient = ++nativeValue;

  id<MTLBlitCommandEncoder> head = [headCb blitCommandEncoder];
  head.label = @"vpgpu.native.after_frames";
  for (id<MTLFence> f : frontier) [head waitForFence:f];
  // A real (tiny) command: an encoder with none could be dropped together with its fence wait.
  [head fillBuffer:nativeScratch range:NSMakeRange(0, 4) value:0];
  [head endEncoding];
  frontier.clear();
  [headCb encodeSignalEvent:nativeEvent value:afterFrames];

  [clientCb encodeWaitForEvent:nativeEvent value:afterFrames];
  fn((__bridge void*)clientCb, user);
  [clientCb encodeSignalEvent:nativeEvent value:afterClient];

  [tailCb encodeWaitForEvent:nativeEvent value:afterClient];
  id<MTLBlitCommandEncoder> end = [tailCb blitCommandEncoder];
  end.label = @"vpgpu.native.before_frames";
  [end fillBuffer:nativeScratch range:NSMakeRange(4, 4) value:0];
  [end updateFence:tailFence];
  [end endEncoding];
  frontier.push_back(tailFence);
  if (residencyDirty) {
    [residency commit];
    residencyDirty = false;
  }

  dispatch_semaphore_t sem = inFlight;
  // Shared with the watchdog under one mutex: while the watchdog holds it the tail's handler
  // cannot release the slot, so the backend (whose destructor waits for every slot) stays alive.
  struct Watch {
    std::mutex m;
    bool done = false;
  };
  auto watch = std::make_shared<Watch>();
  [tailCb addCompletedHandler:^(id<MTLCommandBuffer>) {
    {
      std::lock_guard g(watch->m);
      watch->done = true;
    }
    impl->completions.fetch_add(1);
    impl->doneCv.notify_all();
    dispatch_semaphore_signal(sem);  // last: once every slot is back the backend may be destroyed
  }];
  [headCb commit];
  [clientCb commit];
  [tailCb commit];

  // Watchdog (see above). Checked and logged under the mutex (see Watch).
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC),
                 dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
    std::lock_guard g(watch->m);
    if (watch->done) return;
    auto st = [](id<MTLCommandBuffer> c) { return static_cast<int>(c.status); };
    impl->err("what=native_stuck n=%llu head=%d client=%d tail=%d (status 4=completed 5=error; "
              "2/3=waiting on the event or the GPU)", (unsigned long long)n, st(headCb), st(clientCb), st(tailCb));
  });
  return true;
}

void MetalBackend::Impl::submit(uint64_t frameIndex, const Frame& frame, const CompiledFrame& compiled) {
  if (compiled.encoders.empty()) {  // nothing for the GPU; keeps the frontier's fences out of reuse
    {
      std::lock_guard lock(doneMutex);
      done.push_back({frameIndex, {}});
    }
    frameFinished(frameIndex);
    return;
  }
  Slot* slotp = acquireSlot();
  if (!slotp) {  // device lost: the frame does not run (waits report the loss)
    frameFinished(frameIndex);
    return;
  }
  Slot& slot = *slotp;

  MTLCommandBufferDescriptor* cbd = [MTLCommandBufferDescriptor new];
  cbd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;  // per-encoder fault in the log
  id<MTLCommandBuffer> cb = [queue commandBufferWithDescriptor:cbd];
  cb.label = [NSString stringWithFormat:@"vpgpu.f%llu", (unsigned long long)frameIndex];

  const auto& cmds = frame.commands();
  // Argument tables: `slot.args`, then the chained buffers (argChunk = index in the chain, 0 =
  // slot.args). A table never straddles two buffers; a full buffer moves on to the next one.
  id<MTLBuffer> argBuf = slot.args;
  uint8_t* argBase = static_cast<uint8_t*>(argBuf.contents);
  uint64_t argUsed = 0, argCap = kArgRingBytes;
  size_t argChunk = 0;
  uint32_t missingPipelines = 0, missingResources = 0, argOverflow = 0;
  size_t fenceUsed = 0;
  NSUInteger sampleUsed = 0;
  // Per core encoder: first start sample and last end sample (NSNotFound = unsampled).
  std::vector<std::pair<NSUInteger, NSUInteger>> encSamples(compiled.encoders.size(), {NSNotFound, NSNotFound});

  // Draws not executed, with the first one named in the summary line (one line per frame, not per
  // draw, so a broken draw repeated every frame cannot flood the log).
  uint32_t skippedDraws = 0;
  std::string firstSkip;
  auto skip = [&](uint32_t command, uint32_t item, const std::string& why) {
    if (!skippedDraws++) firstSkip = "c" + std::to_string(command) + ".d" + std::to_string(item) + ":" + why;
  };
  // Writes one stage's argument table: each argument the shader declares, looked up among the
  // bindings by (kind, slot), at its declared offset; bindings it does not declare are ignored.
  // Returns the table's offset in `argBuf` (the buffer it lands in, passed back through `into`),
  // kNoTable for a stage without arguments, or kSkip (no buffer could be allocated, or a declared
  // argument has no binding: `why` says which).
  constexpr int64_t kNoTable = -1, kSkip = -2;
  // No residency set: resources reached through argument tables must be declared to each encoder
  // (useResource). `pendingUse` collects them for the draw being encoded, `used` dedupes per encoder.
  std::vector<std::pair<id<MTLResource>, MTLResourceUsage>> pendingUse;
  std::unordered_set<uintptr_t> used;
  // A stage's table takes the bindings for that stage (Access::stages; 0 = any stage): vertex and
  // fragment shaders number their slots independently.
  auto table = [&](const Table& stage, const std::vector<Access>& bindings, StageMask forStage, std::string& why,
                   id<MTLBuffer> __strong& into, const std::vector<ViewBinding>& views = {}) -> int64_t {
    if (stage.args.empty()) return kNoTable;
    uint64_t bytes = 8;
    for (const ShaderArgument& a : stage.args) bytes = std::max<uint64_t>(bytes, uint64_t{a.offset} + 8);
    if (argUsed + bytes > argCap) {
      // Next buffer of the chain: the slot's own if a previous frame made it, else a new one
      // twice the size of the last (up to kArgChunkMax). Draws are never skipped for room.
      ++argChunk;
      if (argChunk - 1 >= slot.argsMore.size()) {
        uint64_t size = std::min<uint64_t>(kArgChunkMax, std::max<uint64_t>(argCap * 2, kArgRingBytes));
        size = std::max<uint64_t>(size, (bytes + kArgAlign - 1) / kArgAlign * kArgAlign);
        id<MTLBuffer> b = [device newBufferWithLength:size options:kSharedUntracked];
        if (!b) {
          --argChunk;
          ++argOverflow;
          why = "arg_overflow";
          return kSkip;
        }
        b.label = [NSString stringWithFormat:@"vpgpu.args.%zu", argChunk];
        slot.argsMore.push_back(b);
        info("gpu.metal.args grow f=" + std::to_string(frameIndex) + " chunk=" + std::to_string(argChunk) +
             " bytes=" + std::to_string(size) + " slot_chain=" + std::to_string(slot.argsMore.size()));
      }
      argBuf = slot.argsMore[argChunk - 1];
      argBase = static_cast<uint8_t*>(argBuf.contents);
      argCap = argBuf.length;
      argUsed = 0;
    }
    uint8_t* t = argBase + argUsed;
    std::memset(t, 0, bytes);
    for (const ShaderArgument& a : stage.args) {
      // A BufferSize entry reads the Buffer binding of its slot (SPIR-V OpArrayLength).
      const BindKind want = a.kind == BindKind::BufferSize ? BindKind::Buffer : a.kind;
      auto it = std::find_if(bindings.begin(), bindings.end(), [&](const Access& x) {
        return !x.pixelLocal && bindKind(x) == want && x.slot == a.slot && (x.stages == 0 || (x.stages & forStage));
      });
      if (it == bindings.end()) {
        why = "unbound_kind" + std::to_string(int(a.kind)) + "_slot" + std::to_string(a.slot);
        {
          // Once per (stage, kind, slot): what the draw DID bind, so a layout/shader slot
          // disagreement can be read off the log instead of guessed.
          static std::mutex m;
          static std::unordered_set<uint32_t> seen;
          const uint32_t k = uint32_t(forStage) << 24 | uint32_t(a.kind) << 16 | a.slot;
          std::lock_guard lk(m);
          if (seen.insert(k).second) {
            std::string have;
            for (const Access& x : bindings)
              have += " k" + std::to_string(int(bindKind(x))) + "s" + std::to_string(x.slot) + "m" +
                      std::to_string(int(x.stages)) + (x.write ? "w" : "");
            std::string want;
            for (const ShaderArgument& w : stage.args) want += " k" + std::to_string(int(w.kind)) + "s" + std::to_string(w.slot);
            err("what=unbound stage=%u kind=%d slot=%u shader_wants=[%s] draw_binds=[%s]", unsigned(forStage),
                int(a.kind), unsigned(a.slot), want.c_str(), have.c_str());
          }
        }
        return kSkip;
      }
      const ResourceRange& r = it->range;
      if (a.kind == BindKind::BufferSize) {
        // Bytes of the bound range; an open-ended range runs to the end of the buffer.
        uint64_t rangeBytes = 0;  // 0 when the buffer is gone (as its Buffer entry is null)
        if (r.kind == ResourceRange::Kind::Buffer) {
          auto b = buffers.find(r.index);
          if (b != buffers.end() && b->second.generation == r.generation && b->second.buffer) {
            const uint64_t avail = b->second.buffer.length > b->second.base + r.begin
                                       ? b->second.buffer.length - b->second.base - r.begin : 0;
            rangeBytes = std::min(r.end > r.begin ? r.end - r.begin : 0, avail);
          }
        }
        std::memcpy(t + a.offset, &rangeBytes, sizeof rangeBytes);
        continue;
      }
      MTLResourceUsage u = MTLResourceUsageRead;
      if (it->write) u |= MTLResourceUsageWrite;
      uint64_t v = 0;
      MTLResourceID rid{};
      switch (r.kind) {
        case ResourceRange::Kind::Buffer: {
          auto b = buffers.find(r.index);
          if (b != buffers.end() && b->second.generation == r.generation) {
            v = b->second.buffer.gpuAddress + b->second.base + r.begin;
            if (!residency) pendingUse.push_back({b->second.buffer, u});
          }
          break;
        }
        case ResourceRange::Kind::Texture: {
          TextureId use{r.index, r.generation};
          for (const ViewBinding& vb : views)
            if (vb.kind == bindKind(*it) && vb.slot == it->slot) use = vb.view;
          if (id<MTLTexture> tex = texture(use)) {
            rid = tex.gpuResourceID;
            if (!residency) pendingUse.push_back({tex, u});
          }
          std::memcpy(&v, &rid, sizeof rid);
          break;
        }
        case ResourceRange::Kind::Sampler: {
          auto sm = samplers.find(r.index);
          if (sm != samplers.end() && sm->second.generation == r.generation) rid = sm->second.sampler.gpuResourceID;
          std::memcpy(&v, &rid, sizeof rid);
          break;
        }
      }
      missingResources += v == 0;
      std::memcpy(t + a.offset, &v, sizeof v);
    }
    const int64_t at = static_cast<int64_t>(argUsed);
    argUsed += (bytes + kArgAlign - 1) / kArgAlign * kArgAlign;
    into = argBuf;
    return at;
  };
  auto pipeline = [&](PipelineKey k) -> const Pipeline* {
    std::lock_guard lock(pipelineMutex);
    auto it = pipelines.find(k.value);
    return it == pipelines.end() ? nullptr : &it->second;  // node-stable: safe after unlock
  };
  auto buffer = [&](const BufferSpan& s, uint64_t& offset) -> id<MTLBuffer> {
    auto it = buffers.find(s.buffer.index);
    if (it == buffers.end() || it->second.generation != s.buffer.generation) return nil;
    offset = it->second.base + s.offset;
    return it->second.buffer;
  };
  auto texture = [&](TextureId t) { return this->texture(t); };
  auto formatOf = [&](TextureId t) { return formatInfo(textures.at(t.index).desc.format); };
  auto nextFence = [&]() -> id<MTLFence> {
    if (fenceUsed == slot.fences.size()) slot.fences.push_back([device newFence]);
    return slot.fences[fenceUsed++];
  };
  // Two sample indices for the encoder about to open, recorded against core encoder `e`.
  auto samplePair = [&](uint32_t e, NSUInteger& start, NSUInteger& end) {
    start = end = MTLCounterDontSample;
    if (!stageSamples || sampleUsed + 2 > kMaxSamples) return;
    start = sampleUsed++, end = sampleUsed++;
    if (encSamples[e].first == NSNotFound) encSamples[e].first = start;
    encSamples[e].second = end;
  };

  Open open;
  id<MTLFence> openFence = nil;
  // Per-pass dump of this frame (diagnostics): items to write after completion, copies waiting for
  // the open encoder to end.
  const bool dumping = dumpFrame >= 0 && frameIndex == uint64_t(dumpFrame);
  auto dumpItems = std::make_shared<std::vector<DumpItem>>();
  std::vector<DumpCopy> dumpCopies;
  std::unordered_set<uintptr_t> dumpSeen;  // storage textures already queued for the open encoder
  uint64_t dumpBytes = 0;
  // Queues texture `t` (level, slice) for copying when the open encoder ends.
  auto dumpQueue = [&](DumpItem it, id<MTLTexture> t, NSUInteger level, NSUInteger slice) {
    if (dumpItems->size() >= kDumpMaxItems) return;
    it.index = uint32_t(dumpItems->size());
    if (!t) it.note = "no_texture";
    else {
      it.format = t.pixelFormat;
      level = std::min<NSUInteger>(level, t.mipmapLevelCount ? t.mipmapLevelCount - 1 : 0);
      it.width = uint32_t(std::max<NSUInteger>(1, t.width >> level));
      it.height = uint32_t(std::max<NSUInteger>(1, t.height >> level));
      MTLBlitOption opt;
      const uint32_t bpp = dumpBytesPerPixel(t.pixelFormat, opt);
      const NSUInteger row = (NSUInteger(it.width) * bpp + 255) / 256 * 256;
      const uint64_t bytes = uint64_t(row) * it.height;
      if (t.storageMode == MTLStorageModeMemoryless) it.note = "memoryless";
      else if (t.sampleCount > 1) it.note = "msaa";
      else if (!bpp) it.note = "unsupported_format";
      else if (bytes > kDumpMaxItemBytes || dumpBytes + bytes > kDumpMaxTotalBytes) it.note = "too_big";
      else if (!(it.buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared])) it.note = "no_memory";
      else {
        dumpBytes += bytes;
        it.row = row;
        DumpCopy c;
        c.texture = t;
        c.level = level;
        c.options = opt;
        c.item = dumpItems->size();
        if (t.textureType == MTLTextureType3D) c.z = std::min<NSUInteger>(slice, std::max<NSUInteger>(1, t.depth >> level) - 1);
        else {
          const NSUInteger slices = t.arrayLength * ((t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
          c.slice = slice < slices ? slice : 0;
        }
        dumpCopies.push_back(c);
      }
    }
    char line[512];
    std::snprintf(line, sizeof line, "gpu.dump pass=%u what=%s label=%s fmt=%s %ux%u draws=%u load=%d store=%d%s%s",
                  it.index, it.what.c_str(), it.label.empty() ? "-" : token(it.label).c_str(),
                  dumpFormatToken(it.format).c_str(), it.width, it.height, it.draws, it.load, it.store,
                  it.note.empty() ? "" : " note=", it.note.c_str());
    info(line);
    dumpItems->push_back(std::move(it));
  };
  auto close = [&]() {
    if (open.kind == Open::None) return;
    if (open.render) [open.render updateFence:openFence afterStages:MTLRenderStageFragment];
    else if (open.compute) [open.compute updateFence:openFence];
    else [open.blit updateFence:openFence];
    [open.any() endEncoding];
    frontier.push_back(openFence);
    open = {};
    used.clear();
    dumpSeen.clear();
    if (dumpCopies.empty()) return;
    // Copies of what the encoder just wrote, ordered after everything so far; the frame keeps
    // waiting on them (every encoder is a boundary while dumping).
    if (id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]) {
      bl.label = @"vpgpu.dump";
      for (id<MTLFence> f : frontier) [bl waitForFence:f];
      for (const DumpCopy& c : dumpCopies) {
        const DumpItem& it = (*dumpItems)[c.item];
        [bl copyFromTexture:c.texture
                   sourceSlice:c.slice
                   sourceLevel:c.level
                  sourceOrigin:MTLOriginMake(0, 0, c.z)
                    sourceSize:MTLSizeMake(it.width, it.height, 1)
                      toBuffer:it.buffer
             destinationOffset:0
        destinationBytesPerRow:it.row
      destinationBytesPerImage:it.row * it.height
                       options:c.options];
      }
      id<MTLFence> f = nextFence();
      [bl updateFence:f];
      [bl endEncoding];
      frontier.clear();
      frontier.push_back(f);
    } else {
      for (const DumpCopy& c : dumpCopies) {
        (*dumpItems)[c.item].buffer = nil;
        (*dumpItems)[c.item].note = "no_blit_encoder";
      }
    }
    dumpCopies.clear();
  };
  // Starts waiting on everything unordered so far; `before` picks the first render stage held.
  auto waitFrontier = [&](StageMask before) {
    for (id<MTLFence> f : frontier) {
      if (open.render)
        [open.render waitForFence:f beforeStages:(before & stage::vertex) ? MTLRenderStageVertex : MTLRenderStageFragment];
      else if (open.compute) [open.compute waitForFence:f];
      else [open.blit waitForFence:f];
    }
    frontier.clear();
  };
  auto opened = [&](NSString* label, bool wait, StageMask before) {
    open.any().label = label;
    openFence = nextFence();
    if (wait) waitFrontier(before);
  };

  // Declares the current draw's/dispatch's table resources to the open encoder (no residency set).
  auto useBound = [&]() {
    for (const auto& [res, u] : pendingUse) {
      const uintptr_t key = reinterpret_cast<uintptr_t>((__bridge const void*)res) | ((u & MTLResourceUsageWrite) ? 1 : 0);
      if (!used.insert(key).second) continue;
      if (open.render) [open.render useResource:res usage:u stages:MTLRenderStageVertex | MTLRenderStageFragment];
      else if (open.compute) [open.compute useResource:res usage:u];
    }
    pendingUse.clear();
  };

  // A sampled blit (CopyKind::TextureBlit) inside a core compute encoder: one render encoder per
  // destination layer drawing a triangle over the destination rect that samples the source mip/
  // layer through a 2D view. The first one waits on the frontier when the item has a barrier;
  // close() fences it like any render encoder, so later items order after it.
  auto blitUnsupported = [&](const Copy& c, const std::string& reason) {
    if (blitWarned.insert(reason).second)
      err("what=blit_unsupported reason=%s label=%s src=%ux%u dst=%ux%u", token(reason).c_str(), token(c.label).c_str(),
          c.srcTexture.width, c.srcTexture.height, c.dstTexture.width, c.dstTexture.height);
  };
  // In-pass clear (Draw::clear) into the open render encoder: instanced quads over the rects and
  // layers, writing only the cleared attachments; viewport and scissor cover the attachment (the
  // draws after it set their own).
  uint32_t clearFailures = 0;
  auto encodeClear = [&](const ClearRects& c, MTLRenderPassDescriptor* rp, const RenderPassDesc& pass) {
    if (open.kind != Open::Render || !open.render) return;
    std::array<int, kMaxColorAttachments> kinds{};
    for (int i = 0; i < kMaxColorAttachments; ++i) {
      if (!pass.colors[i].texture || !rp.colorAttachments[i].texture) continue;
      const FormatInfo& fi = pass.colors[i].format != Format::Undefined ? formatInfo(pass.colors[i].format)
                                                                         : formatOf(pass.colors[i].texture);
      kinds[i] = !fi.integer() ? 0 : (fi.name && std::strstr(fi.name, "Sint")) ? 1 : 2;
    }
    // Attachment size at its mip, from the first bound attachment.
    id<MTLTexture> ref = nil;
    NSUInteger level = 0;
    for (int i = 0; i < kMaxColorAttachments && !ref; ++i)
      if (rp.colorAttachments[i].texture) ref = rp.colorAttachments[i].texture, level = rp.colorAttachments[i].level;
    if (!ref && rp.depthAttachment.texture) ref = rp.depthAttachment.texture, level = rp.depthAttachment.level;
    if (!ref && rp.stencilAttachment.texture) ref = rp.stencilAttachment.texture, level = rp.stencilAttachment.level;
    if (!ref) return;
    const double w = double(std::max<NSUInteger>(1, ref.width >> level));
    const double h = double(std::max<NSUInteger>(1, ref.height >> level));
    const bool layered = rp.renderTargetArrayLength > 1;
    ClearRects cc = c;  // only what this encoder has
    for (int i = 0; i < kMaxColorAttachments; ++i)
      if (!rp.colorAttachments[i].texture) cc.colorMask &= uint8_t(~(1u << i));
    cc.depth = c.depth && rp.depthAttachment.texture;
    cc.stencil = c.stencil && rp.stencilAttachment.texture;
    if (!cc.colorMask && !cc.depth && !cc.stencil) return;
    std::string error;
    id<MTLRenderPipelineState> ps = clearPipeline(rp, cc, layered, kinds, error);
    if (!ps) {
      if (!clearFailures++) err("what=clear_quads f=%llu error=%s", (unsigned long long)frameIndex, token(error).c_str());
      return;
    }
    struct alignas(16) Inst {
      float ndc[4];
      float depth;
      uint32_t layer, pad0, pad1;
    };
    std::vector<Inst> inst;
    // Without a depth clear the quad's depth is irrelevant (depth state: always, no write): keep
    // it off the clip planes anyway.
    const float depthValue = cc.depth ? std::clamp(cc.depthValue, 0.0f, 1.0f) : 0.5f;
    for (const ClearRects::Rect& r : c.rects) {
      const double x0 = std::min<double>(r.x, w), y0 = std::min<double>(r.y, h);
      const double x1 = std::min<double>(double(r.x) + r.width, w), y1 = std::min<double>(double(r.y) + r.height, h);
      if (x1 <= x0 || y1 <= y0) continue;
      const uint16_t layers = layered ? r.layerCount : 1;
      for (uint16_t l = 0; l < layers; ++l) {
        if (layered && NSUInteger(r.layer + l) >= rp.renderTargetArrayLength) break;
        inst.push_back({{float(x0 / w * 2.0 - 1.0), float(1.0 - y0 / h * 2.0), float(x1 / w * 2.0 - 1.0),
                         float(1.0 - y1 / h * 2.0)},
                        depthValue, uint32_t(layered ? r.layer + l : 0), 0, 0});
      }
    }
    if (inst.empty()) return;
    uint32_t colors[kMaxColorAttachments][4];
    std::memcpy(colors, cc.colorBits.data(), sizeof colors);
    {
      // Diagnostics (CTS: colour clear quads not landing): the first clears of the process.
      static std::atomic<int> logged{0};
      if (logged.fetch_add(1) < 12) {
        char line[384];
        id<MTLTexture> c0 = rp.colorAttachments[0].texture;
        std::snprintf(line, sizeof line,
                      "gpu.clear f=%llu mask=0x%x d=%d s=%d n=%zu fmt0=%lu samples0=%lu kind0=%d bits0=%08x,%08x,%08x,%08x "
                      "ndc0=%.3f,%.3f,%.3f,%.3f z=%.3f layered=%d size=%gx%g load0=%lu",
                      (unsigned long long)frameIndex, unsigned(cc.colorMask), int(cc.depth), int(cc.stencil), inst.size(),
                      (unsigned long)(c0 ? c0.pixelFormat : 0), (unsigned long)(c0 ? c0.sampleCount : 0), kinds[0],
                      colors[0][0], colors[0][1], colors[0][2], colors[0][3], inst[0].ndc[0], inst[0].ndc[1], inst[0].ndc[2],
                      inst[0].ndc[3], inst[0].depth, int(layered), w, h,
                      (unsigned long)rp.colorAttachments[0].loadAction);
        info(line);
      }
    }
    [open.render setRenderPipelineState:ps];
    [open.render setDepthStencilState:clearDepthState(cc.depth, cc.stencil)];
    [open.render setCullMode:MTLCullModeNone];
    [open.render setTriangleFillMode:MTLTriangleFillModeFill];
    [open.render setDepthBias:0 slopeScale:0 clamp:0];
    // Clamp, not clip: a colour-only clear sits at z = 1 (ClearRects::depthValue's default) and a
    // depth clear at 1.0 is the common case — on the far plane, where clipping can drop the quad.
    [open.render setDepthClipMode:MTLDepthClipModeClamp];
    [open.render setStencilReferenceValue:cc.stencilValue];
    [open.render setViewport:(MTLViewport){0, 0, w, h, 0, 1}];
    [open.render setScissorRect:(MTLScissorRect){0, 0, NSUInteger(w), NSUInteger(h)}];
    if (cc.colorMask) [open.render setFragmentBytes:colors length:sizeof colors atIndex:0];
    constexpr size_t kBatch = 4096 / sizeof(Inst) - 1;  // setVertexBytes: under 4 KB per call
    for (size_t at = 0; at < inst.size(); at += kBatch) {
      const size_t n = std::min(kBatch, inst.size() - at);
      [open.render setVertexBytes:inst.data() + at length:n * sizeof(Inst) atIndex:0];
      [open.render drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:n];
    }
  };
  // Returns false when nothing was encoded (the caller then carries the item's wait forward).
  auto encodeBlit = [&](const Copy& c, uint32_t e, uint32_t k, bool wait, StageMask waitBefore) -> bool {
    const TextureSpan &s = c.srcTexture, &d = c.dstTexture;
    id<MTLTexture> st = texture(s.texture), dt = texture(d.texture);
    if (!st || !dt) {
      ++missingResources;
      return false;
    }
    // Copy::blitFormat: both sides seen as that format (mutable-format images). Only through
    // textures made for format views; otherwise their own formats.
    MTLPixelFormat spf = st.pixelFormat;
    bool asFormat = c.blitFormat != Format::Undefined;
    if (asFormat) {
      const MTLPixelFormat pf = pixelFormat(c.blitFormat);
      id<MTLTexture> dv = formatViewOfPixel(dt, pf);
      if ((pf == st.pixelFormat || (st.usage & MTLTextureUsagePixelFormatView)) && dv && dv.pixelFormat == pf) {
        spf = pf;
        dt = dv;
      } else {
        asFormat = false;  // (logged by formatViewOfPixel when the destination view failed)
      }
    }
    const FormatInfo &sf = asFormat ? formatInfo(c.blitFormat) : formatOf(s.texture),
                     &df = asFormat ? formatInfo(c.blitFormat) : formatOf(d.texture);
    std::string why;
    if (!(dt.usage & MTLTextureUsageRenderTarget)) why = "dst_not_renderable";
    else if (dt.storageMode == MTLStorageModeMemoryless) why = "dst_memoryless";
    else if (dt.sampleCount > 1 || (st.sampleCount > 1 && st.textureType == MTLTextureType3D)) why = "msaa";
    else if (sf.integer() != df.integer()) why = "integer_vs_float";
    else if (sf.depth() || sf.stencil() || df.depth() || df.stencil()) why = "depth_stencil";
    else if (!(st.usage & MTLTextureUsageShaderRead)) why = "src_not_sampled";
    if (!why.empty()) {
      blitUnsupported(c, why + "_" + std::to_string(unsigned(dt.pixelFormat)));
      return false;
    }
    const bool s3 = st.textureType == MTLTextureType3D, d3 = dt.textureType == MTLTextureType3D;
    // Integer formats: signed or unsigned by the format name (Sint / Uint), read nearest; float
    // nearest from a 2D source: the exact texel (vp_blit_fs_n).
    const bool ms = st.sampleCount > 1;  // sample 0 of each pixel (integer MSAA resolves)
    const int ik = !sf.integer() ? ((c.blitFilter == Filter::Nearest || ms) && !s3 ? 3 : 0)
                                 : (std::strstr(sf.name, "Sint") ? 1 : 2);
    std::string error;
    id<MTLRenderPipelineState> ps = blitPipeline(dt.pixelFormat, ik | (s3 ? 4 : 0) | (ms ? 8 : 0), error);
    if (!ps) {
      blitUnsupported(c, "pipeline_" + std::to_string(unsigned(dt.pixelFormat)) + "_" + error);
      return false;
    }
    const uint32_t sw = std::max(1u, uint32_t(st.width) >> s.mip), sh = std::max(1u, uint32_t(st.height) >> s.mip);
    const uint32_t dw = std::max(1u, uint32_t(dt.width) >> d.mip), dh = std::max(1u, uint32_t(dt.height) >> d.mip);
    if (s.mip >= st.mipmapLevelCount || d.mip >= dt.mipmapLevelCount || d.x >= dw || d.y >= dh) {
      blitUnsupported(c, "out_of_range");
      return false;
    }
    // Source UV rect (mirrored axes start at the far edge and run backwards).
    float rect[4] = {float(s.x) / float(sw), float(s.y) / float(sh), float(s.width) / float(sw), float(s.height) / float(sh)};
    const uint32_t vw = std::min(d.width, dw - d.x), vh = std::min(d.height, dh - d.y);
    // A destination clipped at the texture edge keeps the same scale: shrink the source span too.
    if (d.width > 0 && vw < d.width) rect[2] *= float(vw) / float(d.width);
    if (d.height > 0 && vh < d.height) rect[3] *= float(vh) / float(d.height);
    if (c.blitFlipX) rect[0] += rect[2], rect[2] = -rect[2];
    if (c.blitFlipY) rect[1] += rect[3], rect[3] = -rect[3];
    // Exact nearest (2D sources): source texel origin and texels per destination pixel, a mirrored
    // axis starting at the far edge with a negative scale (VpBlitN).
    struct {
      float f[8];
      uint32_t lo[4], hi[4];
    } nearest = {{float(c.blitFlipX ? s.x + s.width : s.x), float(c.blitFlipY ? s.y + s.height : s.y),
                  float(s.width) / float(std::max(1u, d.width)) * (c.blitFlipX ? -1.0f : 1.0f),
                  float(s.height) / float(std::max(1u, d.height)) * (c.blitFlipY ? -1.0f : 1.0f), float(d.x), float(d.y),
                  0.0f, 0.0f},
                 {0, 0, 0, 0},
                 {~0u, ~0u, ~0u, ~0u}};
    if (df.integer()) {
      // Destination channel widths (integer colour formats), for the clamp to its range.
      const Format dfmt = asFormat ? c.blitFormat : textures.at(d.texture.index).desc.format;
      uint32_t bits[4] = {32, 32, 32, 32};
      switch (dfmt) {
        case Format::R8Uint: case Format::R8Sint: case Format::RG8Uint: case Format::RG8Sint:
        case Format::RGBA8Uint: case Format::RGBA8Sint: bits[0] = bits[1] = bits[2] = bits[3] = 8; break;
        case Format::R16Uint: case Format::R16Sint: case Format::RG16Uint: case Format::RG16Sint:
        case Format::RGBA16Uint: case Format::RGBA16Sint: bits[0] = bits[1] = bits[2] = bits[3] = 16; break;
        case Format::RGB10A2Uint: bits[0] = bits[1] = bits[2] = 10, bits[3] = 2; break;
        default: break;
      }
      const bool sint = df.name && std::strstr(df.name, "Sint");
      for (int i = 0; i < 4; ++i) {
        if (bits[i] >= 32) {
          nearest.lo[i] = sint ? 0x80000000u : 0u;
          nearest.hi[i] = sint ? 0x7FFFFFFFu : ~0u;
        } else if (sint) {
          nearest.lo[i] = uint32_t(-(int64_t(1) << (bits[i] - 1)));
          nearest.hi[i] = uint32_t((1u << (bits[i] - 1)) - 1);
        } else {
          nearest.hi[i] = (1u << bits[i]) - 1;
        }
      }
    }
    const bool whole = d.x == 0 && d.y == 0 && vw == dw && vh == dh;
    const NSUInteger srcSlices = st.arrayLength * ((st.textureType == MTLTextureTypeCube || st.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
    const NSUInteger dstSlices = dt.arrayLength * ((dt.textureType == MTLTextureTypeCube || dt.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
    bool encoded = false;
    // 3D: one draw per destination depth slice (dst 3D) or per layer; a 3D source is sampled at the
    // matching normalized depth.
    const uint32_t sDepth = s3 ? std::max<uint32_t>(1, uint32_t(st.depth) >> s.mip) : 1;
    const uint32_t dDepth = d3 ? std::max<uint32_t>(1, uint32_t(dt.depth) >> d.mip) : 1;
    const uint32_t iterations = d3 ? std::max<uint32_t>(1, d.depth) : s.layerCount;
    for (uint32_t l = 0; l < iterations; ++l) {
      // A 2D source into a 3D destination: every slice samples the source layer (clamped).
      const uint32_t sl = !s3 && d3 ? std::min<uint32_t>(l, std::max<uint32_t>(1, s.layerCount) - 1) : l;
      if ((!s3 && NSUInteger(s.layer) + sl >= srcSlices) || (!d3 && NSUInteger(d.layer) + l >= dstSlices) ||
          (d3 && d.z + l >= dDepth)) {
        blitUnsupported(c, "layer_out_of_range");
        break;
      }
      // Same pixel format, one mip and slice: allowed without MTLTextureUsagePixelFormatView. The
      // command buffer retains it until the GPU is done.
      id<MTLTexture> sv = s3 ? [st newTextureViewWithPixelFormat:st.pixelFormat
                                                    textureType:MTLTextureType3D
                                                         levels:NSMakeRange(s.mip, 1)
                                                         slices:NSMakeRange(0, 1)]
                             : [st newTextureViewWithPixelFormat:spf
                                                    textureType:ms ? MTLTextureType2DMultisample : MTLTextureType2D
                                                         levels:NSMakeRange(s.mip, 1)
                                                         slices:NSMakeRange(s.layer + sl, 1)];
      if (!sv) {
        blitUnsupported(c, "src_view");
        break;
      }
      const float w = s3 ? (float(s.z) + (float(l) + 0.5f) * float(std::max<uint32_t>(1, s.depth)) / float(iterations)) /
                               float(sDepth)
                         : 0.0f;
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = dt;
      rp.colorAttachments[0].level = d.mip;
      if (d3) rp.colorAttachments[0].depthPlane = d.z + l;
      else rp.colorAttachments[0].slice = d.layer + l;
      rp.colorAttachments[0].loadAction = whole ? MTLLoadActionDontCare : MTLLoadActionLoad;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      close();
      open.kind = Open::Render;
      open.render = [cb renderCommandEncoderWithDescriptor:rp];
      if (!open.render) {
        err("what=render_encoder f=%llu enc=%u item=%u blit=1", (unsigned long long)frameIndex, e, k);
        open = {};
        return encoded;
      }
      opened([NSString stringWithFormat:@"blit_draw%u.%u.%u", e, k, unsigned(l)], wait && !encoded, waitBefore);
      encoded = true;
      [open.render setRenderPipelineState:ps];
      [open.render setViewport:(MTLViewport){double(d.x), double(d.y), double(vw), double(vh), 0.0, 1.0}];
      [open.render setScissorRect:(MTLScissorRect){d.x, d.y, vw, vh}];
      [open.render setVertexBytes:rect length:sizeof rect atIndex:0];
      [open.render setFragmentTexture:sv atIndex:0];
      [open.render setFragmentSamplerState:blitSamplers[c.blitFilter == Filter::Nearest ? 0 : 1] atIndex:0];
      if (s3) [open.render setFragmentBytes:&w length:sizeof w atIndex:0];
      else if (ik) [open.render setFragmentBytes:&nearest length:sizeof nearest atIndex:0];
      [open.render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    return encoded;
  };

  size_t b = 0;  // cursor into compiled.barriers (sorted by encoder, item)
  for (uint32_t e = 0; e < compiled.encoders.size(); ++e) {
    const Encoder& enc = compiled.encoders[e];
    // The first encoder of a frame always waits: frames are hazard-checked independently.
    bool boundary = e == 0;
    StageMask before = stage::vertex;
    if (b < compiled.barriers.size() && compiled.barriers[b].encoder == e && compiled.barriers[b].item == 0) {
      boundary = true;
      before = compiled.barriers[b++].before;
    }
    if (dumping) boundary = true, before = stage::vertex;  // serialised: dump copies sit between encoders
    close();

    if (enc.kind == Encoder::Kind::Render) {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      std::unordered_map<uint64_t, size_t> taken;
      // Tile-local attachments render into a memoryless stand-in: same shape, no memory.
      auto target = [&](const Attachment& src, int slot) -> id<MTLTexture> {
        if (tileMemory && !memorylessFailed.load(std::memory_order_relaxed) && enc.tileLocal(slot)) {
          auto it = textures.find(src.texture.index);
          if (it != textures.end() && it->second.generation == src.texture.generation && !it->second.texture) {
            TextureDesc td = it->second.desc;
            if (src.format != Format::Undefined && slot != kDepthSlot) td.format = src.format;
            return memorylessTexture(td, taken);
          }
        }
        id<MTLTexture> t = texture(src.texture);
        return slot != kDepthSlot ? formatViewOf(t, src.format) : t;
      };
      auto attach = [&](MTLRenderPassAttachmentDescriptor* a, const Attachment& src, id<MTLTexture> t) {
        a.texture = t;
        a.level = src.mip;
        if (t.textureType == MTLTextureType3D) a.depthPlane = src.layer;
        else a.slice = src.layer;
        a.loadAction = loadAction(src.load);
        a.storeAction = storeAction(src.store);
        if (src.store == StoreAction::Resolve || src.store == StoreAction::StoreAndResolve) {
          // A resolve target of another format than the samples (mutable-format images): seen as
          // the attachment's pixel format (Metal requires them to match).
          a.resolveTexture = src.resolve ? texture(src.resolve) : nil;
          if (a.resolveTexture && t && a.resolveTexture.pixelFormat != t.pixelFormat)
            a.resolveTexture = formatViewOfPixel(a.resolveTexture, t.pixelFormat);
          a.resolveLevel = src.resolveMip;
          a.resolveSlice = src.resolveLayer;
          // No resolve target: a resolve store would fail validation. Keep the samples instead
          // (memoryless ones cannot be stored, so they are dropped).
          if (!a.resolveTexture) {
            ++missingResources;
            a.storeAction = t.storageMode == MTLStorageModeMemoryless ? MTLStoreActionDontCare : MTLStoreActionStore;
          }
        }
      };
      for (int i = 0; i < kMaxColorAttachments; ++i) {
        const Attachment& c = enc.pass.colors[i];
        if (!c.texture) continue;
        attach(rp.colorAttachments[i], c, target(c, i));
        rp.colorAttachments[i].clearColor = MTLClearColorMake(c.clear.color[0], c.clear.color[1], c.clear.color[2], c.clear.color[3]);
      }
      if (const Attachment& d = enc.pass.depth; d.texture) {
        id<MTLTexture> t = target(d, kDepthSlot);
        const MTLPixelFormat pf = t.pixelFormat;
        const bool hasStencil = pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatStencil8;
        // An aspect left out of a depth/stencil resolve keeps its plain store.
        auto unresolved = [](Attachment a, ResolveMode m) {
          if (m == ResolveMode::None) {
            if (a.store == StoreAction::StoreAndResolve) a.store = StoreAction::Store;
            else if (a.store == StoreAction::Resolve) a.store = StoreAction::DontCare;
          }
          return a;
        };
        if (pf != MTLPixelFormatStencil8) {
          attach(rp.depthAttachment, unresolved(d, d.depthResolve), t);
          rp.depthAttachment.clearDepth = d.clear.depth;
          rp.depthAttachment.depthResolveFilter = d.depthResolve == ResolveMode::Min   ? MTLMultisampleDepthResolveFilterMin
                                                  : d.depthResolve == ResolveMode::Max ? MTLMultisampleDepthResolveFilterMax
                                                                                       : MTLMultisampleDepthResolveFilterSample0;
        }
        if (hasStencil) {
          attach(rp.stencilAttachment, unresolved(d, d.stencilResolve), t);
          rp.stencilAttachment.clearStencil = d.clear.stencil;
          rp.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
        }
      }
      if (enc.pass.layerCount > 1) rp.renderTargetArrayLength = enc.pass.layerCount;
      NSUInteger s0, s1;
      samplePair(e, s0, s1);
      if (s0 != MTLCounterDontSample) {
        rp.sampleBufferAttachments[0].sampleBuffer = slot.samples;
        rp.sampleBufferAttachments[0].startOfVertexSampleIndex = s0;
        rp.sampleBufferAttachments[0].endOfVertexSampleIndex = MTLCounterDontSample;
        rp.sampleBufferAttachments[0].startOfFragmentSampleIndex = MTLCounterDontSample;
        rp.sampleBufferAttachments[0].endOfFragmentSampleIndex = s1;
      }
      // Every attachment failed (an invalid texture): Metal asserts on a pass with no attachments
      // and no render-target size, so the pass is dropped like a failed encoder.
      {
        bool any = rp.depthAttachment.texture || rp.stencilAttachment.texture;
        for (uint32_t i = 0; !any && i < 8; ++i) any = rp.colorAttachments[i].texture != nil;
        // Attachmentless pass: Metal renders it given a target size and sample count.
        if (!any && enc.pass.width && enc.pass.height) {
          // The pipelines drawn here fix the sample count (Vulkan: rasterizationSamples).
          NSUInteger samples = std::max<NSUInteger>(1, enc.pass.samples);
          for (uint32_t ci : enc.commands) {
            bool found = false;
            for (const Draw& d : std::get<RenderPass>(cmds[ci]).draws)
              if (const Pipeline* p = d.clear ? nullptr : pipeline(d.pipeline); p && p->render) {
                samples = p->samples, found = true;
                break;
              }
            if (found) break;
          }
          rp.renderTargetWidth = enc.pass.width;
          rp.renderTargetHeight = enc.pass.height;
          rp.defaultRasterSampleCount = samples;
        }
        if (!any && rp.renderTargetWidth == 0) {
          err("what=render_no_attachments f=%llu enc=%u", (unsigned long long)frameIndex, e);
          open = {};
          continue;
        }
      }
      // Occlusion queries: the first counting draw's buffer is the encoder's visibility result
      // buffer; counters accumulate across passes (Vulkan resets them with vkCmdResetQueryPool).
      id<MTLBuffer> visBuf = nil;
      uint64_t visBase = 0;
      BufferId visId;
      for (uint32_t ci : enc.commands) {
        for (const Draw& d : std::get<RenderPass>(cmds[ci]).draws)
          if (d.visibility) {
            BufferSpan whole{d.visibility->buffer, 0, 0};
            visBuf = buffer(whole, visBase);
            visId = d.visibility->buffer;
            break;
          }
        if (visBuf || visId) break;
      }
      if (visBuf) {
        rp.visibilityResultBuffer = visBuf;
        rp.visibilityResultType = MTLVisibilityResultTypeAccumulate;
      }
      bool visCounting = false;
      open.kind = Open::Render;
      open.render = [cb renderCommandEncoderWithDescriptor:rp];
      if (!open.render) {
        err("what=render_encoder f=%llu enc=%u", (unsigned long long)frameIndex, e);
        open = {};
        continue;
      }
      opened([NSString stringWithFormat:@"render%u", e], boundary, before);
      if (dumping) {
        DumpItem it;
        it.what = "render";
        for (uint32_t ci : enc.commands) {
          const RenderPass& rpc = std::get<RenderPass>(cmds[ci]);
          it.draws += uint32_t(rpc.draws.size());
          if (it.label.empty()) it.label = rpc.desc.label;
        }
        const Attachment& c0 = enc.pass.colors[0];
        const Attachment& src = c0.texture ? c0 : enc.pass.depth;
        if (!c0.texture) it.what = "render_depth";
        it.load = int(src.load), it.store = int(src.store);
        id<MTLTexture> t = c0.texture ? rp.colorAttachments[0].texture : (src.texture ? rp.depthAttachment.texture : nil);
        NSUInteger level = src.mip, slice = src.layer;
        // MSAA: the resolved image, when the pass resolves.
        if (t && t.sampleCount > 1 && c0.texture && rp.colorAttachments[0].resolveTexture) {
          t = rp.colorAttachments[0].resolveTexture;
          level = c0.resolveMip, slice = c0.resolveLayer;
        }
        dumpQueue(std::move(it), t, level, slice);
      }
      for (uint32_t ci : enc.commands) {
        const auto& draws = std::get<RenderPass>(cmds[ci]).draws;
        for (uint32_t j = 0; j < draws.size(); ++j) {
          const Draw& d = draws[j];
          // Clears never count; a draw counts when it carries this encoder's visibility buffer.
          const bool counts = visBuf && !d.clear && d.visibility && d.visibility->buffer == visId;
          if (visBuf && (counts || visCounting)) {
            const uint64_t off = counts ? visBase + d.visibility->offset : 0;
            if (counts && (off & 7)) {
              if (blitWarned.insert("visibility_align").second) err("what=visibility_unaligned off=%llu", (unsigned long long)off);
            } else {
              [open.render setVisibilityResultMode:counts ? MTLVisibilityResultModeCounting : MTLVisibilityResultModeDisabled
                                            offset:counts ? NSUInteger(off) : 0];
              visCounting = counts;
            }
          }
          if (visBuf && !d.clear && d.visibility && !(d.visibility->buffer == visId) &&
              blitWarned.insert("visibility_buffers").second)
            err("what=visibility_second_buffer: draws of one encoder count into two buffers (only the first counts)");
          if (d.clear) {
            encodeClear(*d.clear, rp, enc.pass);
            continue;
          }
          // Pixel-local reads no attachment serves (frame compiler), or fetches that do not match
          // what the pipeline's fragment shader was specialised for: [[color(n)]] would be garbage.
          FetchInputs fi;
          if (compiled.skipped(ci, j) || !fetchInputs(d, enc.pass, fi)) {
            skip(ci, j, "bad_fetch");
            continue;
          }
          const Pipeline* p = pipeline(d.pipeline);
          if (!p || !p->render) {
            ++missingPipelines;
            continue;
          }
          if (!sameFetch(fi, p->fetch)) {
            skip(ci, j, "fetch_mismatch");
            continue;
          }
          std::string why;
          pendingUse.clear();
          id<MTLBuffer> vtBuf = nil, ftBuf = nil;
          const int64_t vt = table(p->stage[0], d.bindings, stage::vertex, why, vtBuf, d.views);
          const int64_t ft = vt == kSkip ? kSkip : table(p->stage[1], d.bindings, stage::fragment, why, ftBuf, d.views);
          if (ft == kSkip) {
            skip(ci, j, why);
            continue;
          }
          useBound();
          [open.render setRenderPipelineState:p->render];
          id<MTLDepthStencilState> dss = p->depth;
          if (p->stencilTest && (d.stencilReadMask[0] >= 0 || d.stencilReadMask[1] >= 0 || d.stencilWriteMask[0] >= 0 ||
                                 d.stencilWriteMask[1] >= 0)) {
            StencilFace f = p->stencilFront, b = p->stencilBack;
            if (d.stencilReadMask[0] >= 0) f.readMask = uint8_t(d.stencilReadMask[0]);
            if (d.stencilWriteMask[0] >= 0) f.writeMask = uint8_t(d.stencilWriteMask[0]);
            if (d.stencilReadMask[1] >= 0) b.readMask = uint8_t(d.stencilReadMask[1]);
            if (d.stencilWriteMask[1] >= 0) b.writeMask = uint8_t(d.stencilWriteMask[1]);
            const uint32_t key = uint32_t(f.readMask) | uint32_t(f.writeMask) << 8 | uint32_t(b.readMask) << 16 |
                                 uint32_t(b.writeMask) << 24;
            id<MTLDepthStencilState> __strong& m = p->maskedDepth[key];
            if (!m) {
              MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
              ds.depthCompareFunction = p->depthTest ? compareFn(p->depthCompare) : MTLCompareFunctionAlways;
              ds.depthWriteEnabled = p->depthTest && p->depthWrite;
              ds.frontFaceStencil = stencilFace(f);
              ds.backFaceStencil = stencilFace(b);
              m = [device newDepthStencilStateWithDescriptor:ds];
            }
            if (m) dss = m;
          }
          [open.render setDepthStencilState:dss];
          [open.render setCullMode:p->cull];
          [open.render setStencilFrontReferenceValue:d.stencilRef
                                  backReferenceValue:d.stencilRefBack >= 0 ? uint8_t(d.stencilRefBack) : d.stencilRef];
          [open.render setFrontFacingWinding:d.frontFace == Winding::Clockwise ? MTLWindingClockwise : MTLWindingCounterClockwise];
          [open.render setDepthBias:d.depthBias slopeScale:d.depthBiasSlope clamp:d.depthBiasClamp];
          [open.render setDepthClipMode:d.depthClamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
          [open.render setTriangleFillMode:d.wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
          if (d.viewport.width > 0)
            [open.render setViewport:(MTLViewport){d.viewport.x, d.viewport.y, d.viewport.width, d.viewport.height,
                                                   d.viewport.znear, d.viewport.zfar}];
          if (d.scissor.width > 0 && d.scissor.height > 0)
            [open.render setScissorRect:(MTLScissorRect){d.scissor.x, d.scissor.y, d.scissor.width, d.scissor.height}];
          if (p->blendConstant)
            [open.render setBlendColorRed:d.blendColor[0] green:d.blendColor[1] blue:d.blendColor[2] alpha:d.blendColor[3]];
          if (vt >= 0) [open.render setVertexBuffer:vtBuf offset:vt atIndex:p->stage[0].index];
          if (p->drawIndex) [open.render setVertexBytes:&d.drawIndex length:sizeof d.drawIndex atIndex:29];
          if (ft >= 0) [open.render setFragmentBuffer:ftBuf offset:ft atIndex:p->stage[1].index];
          // Indirect: the GPU reads {count, instances, first, (baseVertex,) baseInstance} — Vulkan's
          // and Metal's layouts are the same.
          id<MTLBuffer> argBuf = nil;
          uint64_t argOff = 0;
          if (d.indirect) {
            argBuf = buffer(BufferSpan{d.indirect->buffer, d.indirect->offset, 0, 0, 0}, argOff);
            if (!argBuf || (argOff & 3)) {
              ++missingResources;
              continue;
            }
          }
          if (d.index) {
            uint64_t off = 0;
            BufferSpan span{d.index->buffer, d.index->offset, 0, 0, 0};
            id<MTLBuffer> ib = buffer(span, off);
            if (!ib) {
              ++missingResources;
              continue;
            }
            const uint64_t stride = d.index->wide ? 4 : 2;
            if (argBuf) {
              if (off % stride) {  // Metal: the index buffer offset is a multiple of the index size
                ++missingResources;
                continue;
              }
              [open.render drawIndexedPrimitives:primitiveType(d.primitive)
                                       indexType:d.index->wide ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                                     indexBuffer:ib
                               indexBufferOffset:off
                                  indirectBuffer:argBuf
                            indirectBufferOffset:argOff];
              continue;
            }
            [open.render drawIndexedPrimitives:primitiveType(d.primitive)
                                    indexCount:d.count
                                     indexType:d.index->wide ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                                   indexBuffer:ib
                             indexBufferOffset:off + uint64_t{d.first} * stride
                                 instanceCount:d.instances
                                    baseVertex:d.baseVertex
                                  baseInstance:d.baseInstance];
          } else if (argBuf) {
            [open.render drawPrimitives:primitiveType(d.primitive) indirectBuffer:argBuf indirectBufferOffset:argOff];
          } else {
            [open.render drawPrimitives:primitiveType(d.primitive)
                            vertexStart:d.first
                            vertexCount:d.count
                          instanceCount:d.instances
                           baseInstance:d.baseInstance];
          }
        }
      }
      continue;
    }

    // Compute encoder: dispatches and copies in order, barriers between items where placed.
    // A wait of a sampled blit that encoded nothing passes to the next item (its barrier may
    // cover that item's hazards too).
    bool carryWait = false;
    StageMask carryBefore = 0;
    for (uint32_t k = 0; k < enc.commands.size(); ++k) {
      const Command& cmd = cmds[enc.commands[k]];
      const bool isDispatch = std::holds_alternative<Dispatch>(cmd);
      bool wait = (k == 0 && boundary) || carryWait;
      StageMask waitBefore = (k == 0 ? before : 0) | carryBefore;
      carryWait = false;
      carryBefore = 0;
      if (k > 0 && b < compiled.barriers.size() && compiled.barriers[b].encoder == e && compiled.barriers[b].item == k) {
        waitBefore |= compiled.barriers[b++].before;
        // A memory barrier only orders work inside this encoder; earlier unordered encoders
        // (frontier) need a fence wait, which means a new encoder.
        if (open.kind == Open::Compute && isDispatch && frontier.empty())
          [open.compute memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
        else wait = true;
      }
      // Sampled blits draw: each one runs in render encoders of its own.
      if (!isDispatch && std::get<Copy>(cmd).kind == CopyKind::TextureBlit) {
        if (!encodeBlit(std::get<Copy>(cmd), e, k, wait, waitBefore) && wait) {
          carryWait = true;
          carryBefore = waitBefore;
        }
        continue;
      }
      const Open::Kind want = isDispatch ? Open::Compute : Open::Blit;
      if (wait || open.kind != want) {
        close();
        NSUInteger s0, s1;
        samplePair(e, s0, s1);
        if (isDispatch) {
          MTLComputePassDescriptor* cp = [MTLComputePassDescriptor computePassDescriptor];
          cp.dispatchType = MTLDispatchTypeConcurrent;  // ordering only where barriers say so
          if (s0 != MTLCounterDontSample) {
            cp.sampleBufferAttachments[0].sampleBuffer = slot.samples;
            cp.sampleBufferAttachments[0].startOfEncoderSampleIndex = s0;
            cp.sampleBufferAttachments[0].endOfEncoderSampleIndex = s1;
          }
          open.kind = Open::Compute;
          open.compute = [cb computeCommandEncoderWithDescriptor:cp];
        } else {
          MTLBlitPassDescriptor* bp = [MTLBlitPassDescriptor blitPassDescriptor];
          if (s0 != MTLCounterDontSample) {
            bp.sampleBufferAttachments[0].sampleBuffer = slot.samples;
            bp.sampleBufferAttachments[0].startOfEncoderSampleIndex = s0;
            bp.sampleBufferAttachments[0].endOfEncoderSampleIndex = s1;
          }
          open.kind = Open::Blit;
          open.blit = [cb blitCommandEncoderWithDescriptor:bp];
        }
        opened([NSString stringWithFormat:@"%@%u.%u", isDispatch ? @"compute" : @"blit", e, k], wait, waitBefore);
      }

      if (isDispatch) {
        const Dispatch& d = std::get<Dispatch>(cmd);
        const Pipeline* p = pipeline(d.pipeline);
        if (!p || !p->compute) {
          ++missingPipelines;
          continue;
        }
        std::string why;
        pendingUse.clear();
        id<MTLBuffer> atBuf = nil;
        const int64_t at = table(p->stage[0], d.bindings, stage::compute, why, atBuf, d.views);
        if (at == kSkip) {
          skip(enc.commands[k], 0, why);
          continue;
        }
        useBound();
        if (dumping)
          for (const Access& a : d.bindings) {
            if (!a.write || a.range.kind != ResourceRange::Kind::Texture) continue;
            TextureId use{a.range.index, a.range.generation};
            for (const ViewBinding& vb : d.views)
              if (vb.kind == bindKind(a) && vb.slot == a.slot) use = vb.view;
            id<MTLTexture> t = texture(use);
            if (!t || !dumpSeen.insert(reinterpret_cast<uintptr_t>((__bridge const void*)t)).second) continue;
            DumpItem it;
            it.what = "compute";
            it.label = "c" + std::to_string(enc.commands[k]) + ".slot" + std::to_string(a.slot);
            it.draws = 1;
            dumpQueue(std::move(it), t, 0, 0);
          }
        [open.compute setComputePipelineState:p->compute];
        if (at >= 0) [open.compute setBuffer:atBuf offset:at atIndex:p->stage[0].index];
        if (d.indirect) {
          uint64_t ao = 0;
          id<MTLBuffer> ab = buffer(BufferSpan{d.indirect->buffer, d.indirect->offset, 0, 0, 0}, ao);
          if (!ab || (ao & 3)) {
            skip(enc.commands[k], 0, "indirect_buffer");
            continue;
          }
          [open.compute dispatchThreadgroupsWithIndirectBuffer:ab indirectBufferOffset:ao threadsPerThreadgroup:p->threads];
          continue;
        }
        [open.compute dispatchThreadgroups:MTLSizeMake(d.groups[0], d.groups[1], d.groups[2]) threadsPerThreadgroup:p->threads];
        continue;
      }

      const Copy& c = std::get<Copy>(cmd);
      uint64_t so = 0, dO = 0;
      id<MTLBuffer> sb = buffer(c.srcBuffer, so), db = buffer(c.dstBuffer, dO);
      id<MTLTexture> st = texture(c.srcTexture.texture), dt = texture(c.dstTexture.texture);
      // Row pitch defaults to tightly packed (in blocks for compressed formats).
      auto pitch = [&](const BufferSpan& s, const TextureSpan& t, NSUInteger& row, NSUInteger& image) {
        const FormatInfo& fi = formatOf(t.texture);
        const uint32_t bw = (t.width + fi.blockSize - 1) / fi.blockSize, bh = (t.height + fi.blockHeight - 1) / fi.blockHeight;
        const uint32_t texel = t.aspect == Aspect::Stencil ? 1
                               : t.aspect == Aspect::Depth ? (fi.bytesPerBlock == 2 ? 2 : 4)
                                                           : fi.bytesPerBlock;
        row = s.bytesPerRow ? s.bytesPerRow : bw * texel;
        image = row * (s.rowsPerImage ? s.rowsPerImage : bh);
      };
      // Metal moves one aspect of a combined depth/stencil texture only when told which (without
      // the option a Depth32Float_Stencil8 copy is invalid); single-aspect textures take none.
      auto aspectOption = [](id<MTLTexture> t, Aspect a) -> MTLBlitOption {
        if (t.pixelFormat != MTLPixelFormatDepth32Float_Stencil8) return MTLBlitOptionNone;
        return a == Aspect::Stencil ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;
      };
      // D24 depth: the blit moves float32 between the texture and a staging buffer; a compute
      // encoder (fenced after everything so far, and everything after it waits on it) converts
      // between staging and the client's unorm24. Staging is retained by the command buffer.
      auto isD24 = [&](const TextureSpan& t) {
        return t.aspect == Aspect::Depth && textures.at(t.texture.index).desc.format == Format::D24UnormS8;
      };
      struct D24Params { uint32_t width, height, images, srcRow, srcImage, dstRow, dstImage, pad; };
      auto d24Staging = [&](const TextureSpan& t, bool toUnorm, id<MTLComputePipelineState> __strong& p) -> id<MTLBuffer> {
        std::string why;
        p = d24Pipeline(toUnorm, why);
        const NSUInteger bytes = NSUInteger(t.width) * t.height * 4 * t.depth * t.layerCount;
        id<MTLBuffer> b = p && bytes ? [device newBufferWithLength:bytes
                                                           options:MTLResourceStorageModePrivate |
                                                                   MTLResourceHazardTrackingModeUntracked]
                                     : nil;
        if (!b && blitWarned.insert("d24").second)
          err("what=d24_copy_unconverted reason=%s %ux%u", p ? "no_memory" : token(why).c_str(), t.width, t.height);
        return b;
      };
      auto d24Convert = [&](id<MTLComputePipelineState> p, id<MTLBuffer> src, NSUInteger srcOff, id<MTLBuffer> dst,
                            NSUInteger dstOff, const D24Params& prm) {
        close();
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        ce.label = @"vpgpu.d24";
        for (id<MTLFence> f : frontier) [ce waitForFence:f];
        [ce setComputePipelineState:p];
        [ce setBuffer:src offset:srcOff atIndex:0];
        [ce setBuffer:dst offset:dstOff atIndex:1];
        [ce setBytes:&prm length:sizeof prm atIndex:2];
        [ce dispatchThreads:MTLSizeMake(prm.width, prm.height, prm.images) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        id<MTLFence> f = nextFence();
        [ce updateFence:f];
        [ce endEncoding];
        frontier.clear();
        frontier.push_back(f);
      };
      auto reopenBlit = [&]() {
        close();
        open.kind = Open::Blit;
        open.blit = [cb blitCommandEncoder];
        opened(@"blit.d24", true, 0);
      };
      switch (c.kind) {
        case CopyKind::BufferToBuffer:
          if (!sb || !db) break;
          [open.blit copyFromBuffer:sb sourceOffset:so toBuffer:db destinationOffset:dO size:c.dstBuffer.size];
          continue;
        case CopyKind::FillBuffer:
          if (!db) break;
          [open.blit fillBuffer:db range:NSMakeRange(dO, c.dstBuffer.size) value:c.fillByte];
          continue;
        case CopyKind::TextureToTexture: {
          if (!st || !dt) break;
          const TextureSpan &s = c.srcTexture, &d = c.dstTexture;
          // A copy from a mip into the next one of the same texture with a different size is a mip
          // chain request (VPVulkan's scaled vkCmdBlitImage chains): the blit encoder builds them all.
          if (st == dt && d.mip == s.mip + 1 && (s.width != d.width || s.height != d.height)) {
            if (st.mipmapLevelCount > 1) [open.blit generateMipmapsForTexture:st];
            continue;
          }
          // 2D (array) <-> 3D: each array layer is one depth slice of the 3D side (Vulkan
          // maintenance1); 2D sides copy one slice per layer, two 3D sides one block of depth.
          const bool s3 = st.textureType == MTLTextureType3D, d3 = dt.textureType == MTLTextureType3D;
          const uint32_t n = s3 == d3 ? s.layerCount : std::max<uint32_t>(s.layerCount, d.layerCount);
          const uint32_t depth = s3 && d3 ? s.depth : 1;
          // Different pixel formats (Vulkan "size-compatible" copies, compressed <-> uncompressed):
          // through a staging buffer. A texture-to-texture blit across formats is not a raw copy on
          // Apple GPUs (lossless compression keeps per-format layouts).
          const FormatInfo &sfi = formatOf(s.texture), &dfi = formatOf(d.texture);
          id<MTLBuffer> stage = nil;
          NSUInteger sRow = 0, sImage = 0;
          if (st.pixelFormat != dt.pixelFormat) {
            const NSUInteger bw = (s.width + sfi.blockSize - 1) / sfi.blockSize;
            const NSUInteger bh = (s.height + sfi.blockHeight - 1) / sfi.blockHeight;
            sRow = bw * sfi.bytesPerBlock;
            sImage = sRow * bh;
            const NSUInteger bytes = sImage * depth * n;
            if (sfi.bytesPerBlock != dfi.bytesPerBlock || !bytes ||
                // (tracked: Metal orders the texture->buffer and buffer->texture blits within the encoder)
                !(stage = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate])) {
              if (blitWarned.insert("copy_formats").second)
                err("what=copy_image_formats_unsupported src=%s dst=%s", sfi.name, dfi.name);
              continue;
            }
          }
          for (uint32_t l = 0; l < n; ++l) {
            const NSUInteger sSlice = s3 ? 0 : s.layer + l, sz = s3 ? s.z + (d3 ? 0 : l) : 0;
            const NSUInteger dSlice = d3 ? 0 : d.layer + l, dz = d3 ? d.z + (s3 ? 0 : l) : 0;
            if (!stage) {
              [open.blit copyFromTexture:st
                             sourceSlice:sSlice
                             sourceLevel:s.mip
                            sourceOrigin:MTLOriginMake(s.x, s.y, sz)
                              sourceSize:MTLSizeMake(s.width, s.height, depth)
                               toTexture:dt
                        destinationSlice:dSlice
                        destinationLevel:d.mip
                       destinationOrigin:MTLOriginMake(d.x, d.y, dz)];
              continue;
            }
            const NSUInteger at = sImage * depth * l;
            [open.blit copyFromTexture:st
                           sourceSlice:sSlice
                           sourceLevel:s.mip
                          sourceOrigin:MTLOriginMake(s.x, s.y, sz)
                            sourceSize:MTLSizeMake(s.width, s.height, depth)
                              toBuffer:stage
                     destinationOffset:at
                destinationBytesPerRow:sRow
              destinationBytesPerImage:sImage];
            [open.blit copyFromBuffer:stage
                         sourceOffset:at
                    sourceBytesPerRow:sRow
                  sourceBytesPerImage:sImage
                           sourceSize:MTLSizeMake(d.width, d.height, depth)
                            toTexture:dt
                     destinationSlice:dSlice
                     destinationLevel:d.mip
                    destinationOrigin:MTLOriginMake(d.x, d.y, dz)];
          }
          continue;
        }
        case CopyKind::BufferToTexture: {
          if (!sb || !dt) break;
          const TextureSpan& d = c.dstTexture;
          NSUInteger row, image;
          pitch(c.srcBuffer, d, row, image);
          const MTLBlitOption opt = aspectOption(dt, d.aspect);
          id<MTLComputePipelineState> d24p = nil;
          if (id<MTLBuffer> stage = isD24(d) ? d24Staging(d, false, d24p) : nil) {
            const NSUInteger sRow = NSUInteger(d.width) * 4, sImage = sRow * d.height;
            const D24Params prm{d.width, d.height, uint32_t(d.depth) * d.layerCount, uint32_t(row / 4),
                                uint32_t(image / 4), d.width, uint32_t(sImage / 4), 0};
            d24Convert(d24p, sb, so, stage, 0, prm);
            reopenBlit();
            for (uint16_t l = 0; l < d.layerCount; ++l)
              [open.blit copyFromBuffer:stage
                           sourceOffset:l * sImage * d.depth
                      sourceBytesPerRow:sRow
                    sourceBytesPerImage:sImage
                             sourceSize:MTLSizeMake(d.width, d.height, d.depth)
                              toTexture:dt
                       destinationSlice:d.layer + l
                       destinationLevel:d.mip
                      destinationOrigin:MTLOriginMake(d.x, d.y, d.z)
                                options:opt];
            continue;
          }
          for (uint16_t l = 0; l < d.layerCount; ++l)
            [open.blit copyFromBuffer:sb
                         sourceOffset:so + l * image * d.depth
                    sourceBytesPerRow:row
                  sourceBytesPerImage:image
                           sourceSize:MTLSizeMake(d.width, d.height, d.depth)
                            toTexture:dt
                     destinationSlice:d.layer + l
                     destinationLevel:d.mip
                    destinationOrigin:MTLOriginMake(d.x, d.y, d.z)
                              options:opt];
          continue;
        }
        case CopyKind::TextureBlit: continue;  // drawn by encodeBlit above
        case CopyKind::TextureToBuffer: {
          if (!st || !db) break;
          const TextureSpan& s = c.srcTexture;
          NSUInteger row, image;
          pitch(c.dstBuffer, s, row, image);
          const MTLBlitOption opt = aspectOption(st, s.aspect);
          id<MTLComputePipelineState> d24p = nil;
          if (id<MTLBuffer> stage = isD24(s) ? d24Staging(s, true, d24p) : nil) {
            const NSUInteger sRow = NSUInteger(s.width) * 4, sImage = sRow * s.height;
            for (uint16_t l = 0; l < s.layerCount; ++l)
              [open.blit copyFromTexture:st
                             sourceSlice:s.layer + l
                             sourceLevel:s.mip
                            sourceOrigin:MTLOriginMake(s.x, s.y, s.z)
                              sourceSize:MTLSizeMake(s.width, s.height, s.depth)
                                toBuffer:stage
                       destinationOffset:l * sImage * s.depth
                  destinationBytesPerRow:sRow
                destinationBytesPerImage:sImage
                                 options:opt];
            const D24Params prm{s.width, s.height, uint32_t(s.depth) * s.layerCount, s.width, uint32_t(sImage / 4),
                                uint32_t(row / 4), uint32_t(image / 4), 0};
            d24Convert(d24p, stage, 0, db, dO, prm);
            continue;
          }
          for (uint16_t l = 0; l < s.layerCount; ++l)
            [open.blit copyFromTexture:st
                           sourceSlice:s.layer + l
                           sourceLevel:s.mip
                          sourceOrigin:MTLOriginMake(s.x, s.y, s.z)
                            sourceSize:MTLSizeMake(s.width, s.height, s.depth)
                              toBuffer:db
                     destinationOffset:dO + l * image * s.depth
                destinationBytesPerRow:row
              destinationBytesPerImage:image
                               options:opt];
          continue;
        }
      }
      ++missingResources;
    }
  }
  close();
  if (residencyDirty) {  // render targets allocated lazily while encoding
    [residency commit];
    residencyDirty = false;
  }

  if (missingPipelines || missingResources || argOverflow || skippedDraws)
    err("what=submit f=%llu missing_pipelines=%u missing_resources=%u arg_overflow=%u skipped_draws=%u first_skip=%s",
        (unsigned long long)frameIndex, missingPipelines, missingResources, argOverflow, skippedDraws,
        skippedDraws ? token(firstSkip).c_str() : "-");

  id<MTLCounterSampleBuffer> samples = slot.samples;
  const NSUInteger sampleCount = sampleUsed;
  dispatch_semaphore_t sem = inFlight;
  Impl* impl = this;  // the destructor waits for every frame, so the handler never outlives *this
  if (dumping) info("gpu.dump frame=" + std::to_string(frameIndex) + " items=" + std::to_string(dumpItems->size()) +
                    " copy_mb=" + std::to_string(dumpBytes >> 20));
  std::shared_ptr<std::vector<DumpItem>> dumpDone = dumping ? dumpItems : nullptr;
  const std::string dumpTo = dumpDir;
  [cb addCompletedHandler:^(id<MTLCommandBuffer> finished) {
    if (dumpDone) {  // before the frame slot is released; diagnostics frame only
      @autoreleasepool {
        std::error_code ec;
        std::filesystem::create_directories(dumpTo, ec);
        if (finished.status == MTLCommandBufferStatusError) impl->info("gpu.dump note=command_buffer_error (images may be partial)");
        for (const DumpItem& it : *dumpDone) impl->info(dumpWrite(it, frameIndex, dumpTo));
        impl->info("gpu.dump done frame=" + std::to_string(frameIndex));
      }
    }
    Done d{frameIndex, {}};
    d.timings.frameMs = (finished.GPUEndTime - finished.GPUStartTime) * 1000.0;
    if (finished.status == MTLCommandBufferStatusError) {
      std::string encs;
      NSArray* infos = finished.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
      for (id<MTLCommandBufferEncoderInfo> i in infos)
        if (i.errorState == MTLCommandEncoderErrorStateFaulted || i.errorState == MTLCommandEncoderErrorStateAffected)
          encs += (encs.empty() ? "" : ",") + token(i.label.UTF8String ?: "?");
      impl->err("what=command_buffer f=%llu error=%s encoders=%s", (unsigned long long)frameIndex,
                token(str(finished.error)).c_str(), encs.c_str());
      const std::string why = str(finished.error);
      if ((finished.error.code == MTLCommandBufferErrorMemoryless || why.find("memoryless") != std::string::npos ||
           why.find("ParameterBuffer") != std::string::npos) &&
          !impl->memorylessFailed.exchange(true))
        impl->err("what=memoryless_disabled f=%llu reason=tiler_flush (attachments now in memory)",
                  (unsigned long long)frameIndex);
      if (impl->lostByError(finished.error))
        impl->markLost("command_buffer f=" + std::to_string(frameIndex) + " " + why +
                       (encs.empty() ? "" : " encoders=" + encs));
    }
    if (sampleCount) {
      NSData* data = [samples resolveCounterRange:NSMakeRange(0, sampleCount)];
      const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
      for (const auto& [s0, s1] : encSamples) {
        double ms = 0;
        if (ts && s0 != NSNotFound && ts[s0].timestamp != MTLCounterErrorValue &&
            ts[s1].timestamp != MTLCounterErrorValue && ts[s1].timestamp >= ts[s0].timestamp)
          ms = double(ts[s1].timestamp - ts[s0].timestamp) * impl->gpuTickNs / 1e6;
        d.timings.encoderMs.push_back(ms);
      }
    }
    {
      std::lock_guard lock(impl->doneMutex);
      impl->done.push_back(std::move(d));
    }
    impl->frameFinished(frameIndex);
    dispatch_semaphore_signal(sem);  // last: once every slot is back the backend may be destroyed
  }];
  [cb commit];
}

// ---------------------------------------------------------------------------------------------

MetalBackend::MetalBackend(const ShaderProvider& shaders, LogSink log, MetalOptions options)
    : impl_(std::make_unique<Impl>(shaders, std::move(log), std::move(options))) {
  @autoreleasepool {
    impl_->init();
  }
}

MetalBackend::~MetalBackend() {
  impl_->closing = true;
  for (dispatch_queue_t q : impl_->prewarmQueues)
    if (q) dispatch_sync(q, ^{});  // queued prewarms see `closing`; the running one finishes
  if (ok()) {
    bool drained = false;
    impl_->drain(UINT64_MAX, drained);
    if (!drained) {
      // A lost device's stuck command buffers still reference the backend in their completion
      // handlers: it is leaked (never freed under them) rather than destroyed.
      impl_->err("what=device_lost_leak (command buffers never completed; backend state kept alive)");
      (void)impl_.release();
    }
  }
}

bool MetalBackend::ok() const { return impl_->device != nil && impl_->queue != nil; }
bool MetalBackend::tileMemory() const { return impl_->tileMemory; }
const MetalCaps& MetalBackend::caps() const { return impl_->caps; }

void MetalBackend::createBuffer(BufferId handle, const BufferDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  Impl::Buffer b{handle.generation, nil, 0};
  if (d.guestMemory) {
    // No-copy buffers must start on a page and span whole pages: widen the window around the
    // guest range and remember where the guest pointer sits in it.
    const uint64_t page = getpagesize();
    const auto addr = reinterpret_cast<uintptr_t>(d.guestMemory);
    const uintptr_t start = addr & ~(page - 1);
    const uint64_t length = (addr - start + d.size + page - 1) & ~(page - 1);
    b.base = addr - start;
    b.buffer = [m.device newBufferWithBytesNoCopy:reinterpret_cast<void*>(start)
                                           length:length
                                          options:kSharedUntracked
                                      deallocator:nil];
  } else {
    b.buffer = [m.device newBufferWithLength:d.size options:kSharedUntracked];
  }
  if (!b.buffer) {
    m.err("what=buffer index=%u size=%llu guest=%d label=%s", handle.index, (unsigned long long)d.size,
          d.guestMemory != nullptr, token(d.label).c_str());
    return;
  }
  if (!d.label.empty()) b.buffer.label = @(d.label.c_str());
  if (m.residency) {
    [m.residency addAllocation:b.buffer];
    m.residencyDirty = true;
  }
  m.buffers[handle.index] = b;
}

void MetalBackend::createTexture(TextureId handle, const TextureDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  m.textures[handle.index] = {handle.generation, nil, d, false};
  // Render targets wait for their first use: one that only ever lives on tile never gets memory.
  const bool deferred = !d.persistent && m.tileMemory && (d.usage & (usage::colorTarget | usage::depthTarget));
  if (!deferred) m.texture(handle);
}

void MetalBackend::createSampler(SamplerId handle, const SamplerDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  MTLSamplerDescriptor* s = [MTLSamplerDescriptor new];
  s.minFilter = d.minFilter == Filter::Linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
  s.magFilter = d.magFilter == Filter::Linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
  s.mipFilter = d.mipFilter == MipFilter::Linear    ? MTLSamplerMipFilterLinear
                : d.mipFilter == MipFilter::Nearest ? MTLSamplerMipFilterNearest
                                                    : MTLSamplerMipFilterNotMipmapped;
  s.sAddressMode = addressMode(d.u);
  s.tAddressMode = addressMode(d.v);
  s.rAddressMode = addressMode(d.w);
  s.maxAnisotropy = std::max<uint8_t>(1, d.maxAnisotropy);
  if (d.compare) s.compareFunction = compareFn(*d.compare);
  s.borderColor = d.border == BorderColor::OpaqueWhite   ? MTLSamplerBorderColorOpaqueWhite
                  : d.border == BorderColor::OpaqueBlack ? MTLSamplerBorderColorOpaqueBlack
                                                         : MTLSamplerBorderColorTransparentBlack;
  s.lodMinClamp = d.lodMin;
  s.lodMaxClamp = d.lodMax;
  // lodBias is recent in MTLSamplerDescriptor: set through KVC only where the
  // runtime has it, so the build never depends on the SDK revision.
  const bool hasLodBias = [s respondsToSelector:@selector(setLodBias:)];
  if (hasLodBias && d.lodBias != 0.0f) [s setValue:@(d.lodBias) forKey:@"lodBias"];
  if (!d.normalized) {
    // Metal's rules for pixel coordinates: not mipmapped, no anisotropy, lod clamps at 0 (Vulkan
    // demands lod 0 and nearest mips of such samplers anyway).
    s.normalizedCoordinates = NO;
    s.mipFilter = MTLSamplerMipFilterNotMipmapped;
    s.maxAnisotropy = 1;
    s.lodMinClamp = 0.0f;
    s.lodMaxClamp = 0.0f;
    if (hasLodBias) [s setValue:@(0.0f) forKey:@"lodBias"];
    // Metal allows only ClampToEdge / ClampToZero here (Vulkan: edge or border on U/V; W unused).
    // A border becomes zero: exact for transparent black, alpha 0 instead of 1 for opaque ones.
    auto clampOnly = [](AddressMode a) -> MTLSamplerAddressMode {
      return a == AddressMode::ClampToBorder ? MTLSamplerAddressModeClampToZero : MTLSamplerAddressModeClampToEdge;
    };
    s.sAddressMode = clampOnly(d.u);
    s.tAddressMode = clampOnly(d.v);
    s.rAddressMode = MTLSamplerAddressModeClampToEdge;
  }
  s.supportArgumentBuffers = YES;  // reachable through gpuResourceID in argument tables
  if (!d.label.empty()) s.label = @(d.label.c_str());
  id<MTLSamplerState> st = [m.device newSamplerStateWithDescriptor:s];
  if (!st && !d.normalized) {  // refused: normalized coordinates rather than no sampler at all
    m.err("what=sampler_unnormalized index=%u fallback=normalized", handle.index);
    s.normalizedCoordinates = YES;
    st = [m.device newSamplerStateWithDescriptor:s];
  }
  if (!st) {
    m.err("what=sampler index=%u label=%s", handle.index, token(d.label).c_str());
    return;
  }
  m.samplers[handle.index] = {handle.generation, st};
}

void MetalBackend::destroyBuffer(BufferId handle) {
  auto it = impl_->buffers.find(handle.index);
  if (it == impl_->buffers.end() || it->second.generation != handle.generation) return;
  impl_->retire(it->second.buffer);
  impl_->buffers.erase(it);
}

void MetalBackend::destroyTexture(TextureId handle) {
  auto it = impl_->textures.find(handle.index);
  if (it == impl_->textures.end() || it->second.generation != handle.generation) return;
  if (id<MTLTexture> t = it->second.texture)
    for (auto f = impl_->formatViews.begin(); f != impl_->formatViews.end();) {
      if (f->first.base == t) {
        if (f->second != t) impl_->retire(f->second);
        f = impl_->formatViews.erase(f);
      } else {
        ++f;
      }
    }
  impl_->retire(it->second.texture);
  impl_->textures.erase(it);
}

void MetalBackend::destroySampler(SamplerId handle) {
  auto it = impl_->samplers.find(handle.index);
  if (it == impl_->samplers.end() || it->second.generation != handle.generation) return;
  impl_->retire(it->second.sampler);
  impl_->samplers.erase(it);
}

bool MetalBackend::compilePipeline(const PipelineState& s, std::string& error) {
  if (!ok()) {
    error = "no Metal device";
    return false;
  }
  @autoreleasepool {
    Impl::Pipeline p;
    const bool built = s.computeShader ? impl_->buildCompute(s, p, error) : impl_->buildRender(s, p, error);
    if (!built) {
      impl_->err("what=pipeline key=%016llx label=%s error=%s", (unsigned long long)s.key().value,
                 token(s.label).c_str(), token(error).c_str());
      return false;
    }
    {
      std::lock_guard lock(impl_->pipelineMutex);
      impl_->pipelines.emplace(s.key().value, std::move(p));  // first one wins: entries never move
    }
    impl_->maybeSaveArchive();  // runs on the compile worker, never on the submit thread
    return true;
  }
}

void MetalBackend::submit(uint64_t frameIndex, const Frame& frame, const CompiledFrame& compiled) {
  if (!ok()) {
    impl_->frameFinished(frameIndex);  // nothing runs without a device; fences must not wait for it
    return;
  }
  @autoreleasepool {
    impl_->submit(frameIndex, frame, compiled);
  }
}

bool MetalBackend::takeTimings(uint64_t& frameIndex, GpuTimings& out) {
  std::lock_guard lock(impl_->doneMutex);
  if (impl_->done.empty()) return false;
  frameIndex = impl_->done.front().frame;
  out = std::move(impl_->done.front().timings);
  impl_->done.pop_front();
  return true;
}

bool MetalBackend::encodeNative(NativeEncodeFn fn, void* user) {
  if (!ok() || !fn) return false;
  @autoreleasepool {
    return impl_->encodeNative(fn, user);
  }
}

MetalBackend::WaitResult MetalBackend::waitIdle(uint64_t timeoutNs) {
  if (!ok()) return WaitResult::Done;
  bool drained = false;
  return impl_->drain(timeoutNs, drained);
}

MetalBackend::WaitResult MetalBackend::waitFrames(uint64_t frames, uint64_t timeoutNs) {
  Impl& m = *impl_;
  const int64_t start = steadyNs();
  const int64_t deadline = timeoutNs >= uint64_t(INT64_MAX) - uint64_t(start) ? INT64_MAX : start + int64_t(timeoutNs);
  uint64_t seen = m.completions.load();
  int64_t quietSince = start, lastPass = start;
  std::unique_lock lock(m.doneMutex);
  for (;;) {
    if (m.lostFlag.load()) return WaitResult::Lost;
    if (m.finishedBelow >= frames) return WaitResult::Done;
    const int64_t now = steadyNs();
    if (now >= deadline) return WaitResult::Timeout;
    m.doneCv.wait_for(lock, std::chrono::nanoseconds(std::min(kSliceNs, deadline - now)));
    const uint64_t c = m.completions.load();
    const int64_t t = steadyNs();
    if (c != seen || t - lastPass > kPauseNs) {
      seen = c;
      quietSince = t;
    } else if (t - quietSince > kStallNs && m.finishedBelow < frames) {
      lock.unlock();
      m.markLost("gpu_stuck (fence wait, no command buffer completed for 20 s)");
      lock.lock();
    }
    lastPass = t;
  }
}

bool MetalBackend::lost() const { return impl_->lostFlag.load(); }

void* MetalBackend::contents(BufferId handle) const {
  auto it = impl_->buffers.find(handle.index);
  if (it == impl_->buffers.end() || it->second.generation != handle.generation) return nullptr;
  return static_cast<uint8_t*>(it->second.buffer.contents) + it->second.base;
}

TextureId MetalBackend::createView(TextureId base, const ViewDesc& d) {
  Impl& m = *impl_;
  const uint32_t index = Impl::kViewBit | m.nextView++;
  Impl::View v;
  v.generation = 1, v.base = base, v.desc = d;
  m.views[index] = v;
  return TextureId{index, 1};
}

void MetalBackend::destroyView(TextureId t) {
  auto it = impl_->views.find(t.index);
  if (it == impl_->views.end()) return;
  if (it->second.texture) impl_->retire(it->second.texture);  // may still be in flight
  impl_->views.erase(it);
}

void* MetalBackend::nativeDevice() const { return (__bridge void*)impl_->device; }

void* MetalBackend::nativeTexture(TextureId handle) { return (__bridge void*)impl_->texture(handle); }

// ---------------------------------------------------------------------------------------------
// Pipeline binary archives

namespace {

constexpr size_t kMaxArchiveFiles = 64;     // files loaded per session; the oldest are evicted
constexpr size_t kMinPackCap = 8;           // archivePackCap never goes below this
constexpr int kArchiveDrops = 4;            // culprits dropped per save before splitting
constexpr int kArchiveSplitDepth = 4;       // halvings of the unsaved entries (<= 16 files)

// FNV-1a over the descriptor fields that make a pipeline distinct to the archive.
struct DescHash {
  uint64_t h = 1469598103934665603ull;
  void add(uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      h ^= (v >> (8 * i)) & 0xff;
      h *= 1099511628211ull;
    }
  }
  void add(id object) { add(uint64_t(uintptr_t((__bridge void*)object))); }
};

// Functions are cached for the whole session (Impl::functions, never erased), so the object
// pointer identifies the compiled variant (shader id, fetch inputs, macros).
uint64_t descriptorKey(id desc, bool compute) {
  DescHash k;
  k.add(uint64_t(compute));
  if (compute) {
    MTLComputePipelineDescriptor* c = (MTLComputePipelineDescriptor*)desc;
    k.add(c.computeFunction);
    k.add(uint64_t(c.maxTotalThreadsPerThreadgroup));
    k.add(uint64_t(c.threadGroupSizeIsMultipleOfThreadExecutionWidth));
    return k.h;
  }
  MTLRenderPipelineDescriptor* d = (MTLRenderPipelineDescriptor*)desc;
  k.add(d.vertexFunction);
  k.add(d.fragmentFunction);
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    MTLRenderPipelineColorAttachmentDescriptor* c = d.colorAttachments[i];
    k.add(uint64_t(c.pixelFormat));
    k.add(uint64_t(c.writeMask));
    k.add(uint64_t(c.blendingEnabled));
    k.add(uint64_t(c.sourceRGBBlendFactor) | uint64_t(c.destinationRGBBlendFactor) << 8 |
          uint64_t(c.sourceAlphaBlendFactor) << 16 | uint64_t(c.destinationAlphaBlendFactor) << 24 |
          uint64_t(c.rgbBlendOperation) << 32 | uint64_t(c.alphaBlendOperation) << 40);
  }
  k.add(uint64_t(d.depthAttachmentPixelFormat));
  k.add(uint64_t(d.stencilAttachmentPixelFormat));
  k.add(uint64_t(d.rasterSampleCount));
  k.add(uint64_t(d.alphaToCoverageEnabled));
  k.add(uint64_t(d.alphaToOneEnabled));
  k.add(uint64_t(d.rasterizationEnabled));
  k.add(uint64_t(d.inputPrimitiveTopology));
  MTLVertexDescriptor* vd = d.vertexDescriptor;  // unused today (vertex pulling); hashed if set
  if (vd) {
    for (NSUInteger i = 0; i < 31; ++i) {
      MTLVertexAttributeDescriptor* a = vd.attributes[i];
      k.add(uint64_t(a.format) | uint64_t(a.offset) << 16 | uint64_t(a.bufferIndex) << 48);
      MTLVertexBufferLayoutDescriptor* l = vd.layouts[i];
      k.add(uint64_t(l.stride) | uint64_t(l.stepFunction) << 32 | uint64_t(l.stepRate) << 40);
    }
  }
  return k.h;
}

// One-line summary of an archive entry, for the log.
std::string describeEntry(id desc, bool compute) {
  char b[512];
  if (compute) {
    MTLComputePipelineDescriptor* c = (MTLComputePipelineDescriptor*)desc;
    std::snprintf(b, sizeof b, "kind=compute cs=%p", (__bridge void*)c.computeFunction);
    return b;
  }
  MTLRenderPipelineDescriptor* d = (MTLRenderPipelineDescriptor*)desc;
  std::string colors;
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    MTLRenderPipelineColorAttachmentDescriptor* c = d.colorAttachments[i];
    if (c.pixelFormat == MTLPixelFormatInvalid) continue;
    colors += (colors.empty() ? "" : ",") + std::to_string(i) + ":" + std::to_string(unsigned(c.pixelFormat)) +
              (c.blendingEnabled ? "b" : "") + "m" + std::to_string(unsigned(c.writeMask));
  }
  std::snprintf(b, sizeof b, "kind=render vs=%p fs=%p colors=%s depth=%u stencil=%u samples=%u a2c=%d topo=%u vdesc=%d",
                (__bridge void*)d.vertexFunction, (__bridge void*)d.fragmentFunction,
                colors.empty() ? "none" : colors.c_str(), unsigned(d.depthAttachmentPixelFormat),
                unsigned(d.stencilAttachmentPixelFormat), unsigned(d.rasterSampleCount),
                int(d.alphaToCoverageEnabled), unsigned(d.inputPrimitiveTopology), d.vertexDescriptor != nil);
  return b;
}

// "pipelines-<n>.metalar" -> n (n >= 1).
bool archiveNumber(const std::string& name, uint64_t& n) {
  static const std::string pre = "pipelines-", suf = ".metalar";
  if (name.size() <= pre.size() + suf.size() || name.compare(0, pre.size(), pre) != 0 ||
      name.compare(name.size() - suf.size(), suf.size(), suf) != 0)
    return false;
  const size_t digits = name.size() - pre.size() - suf.size();
  if (digits > 18) return false;
  n = 0;
  for (size_t i = pre.size(); i < pre.size() + digits; ++i) {
    const char c = name[i];
    if (c < '0' || c > '9') return false;
    n = n * 10 + uint64_t(c - '0');
  }
  return n > 0;
}

// The pipeline number in a packer error ("... in pipeline no. '30'"); the quotes may be
// typographic (multi-byte), so up to a few non-digit bytes are skipped.
bool archiveErrorIndex(const std::string& s, size_t& n) {
  static const std::string key = "pipeline no.";
  size_t i = s.find(key);
  if (i == std::string::npos) return false;
  i += key.size();
  for (size_t skip = 0; i < s.size() && skip < 8 && (s[i] < '0' || s[i] > '9'); ++i, ++skip) {}
  if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
  n = 0;
  for (size_t d = 0; i < s.size() && s[i] >= '0' && s[i] <= '9' && d < 9; ++i, ++d) n = n * 10 + size_t(s[i] - '0');
  return true;
}

}  // namespace

std::filesystem::path MetalBackend::Impl::archivePath(uint64_t n) const {
  if (n == 0) return options.cacheRoot / "pipelines.metalar";  // legacy single-file cache
  return options.cacheRoot / ("pipelines-" + std::to_string(n) + ".metalar");
}

id<MTLBinaryArchive> MetalBackend::Impl::newArchive() {
  NSError* e = nil;
  id<MTLBinaryArchive> ar = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
  if (!ar) err("what=archive_create error=%s", token(str(e)).c_str());
  return ar;
}

void MetalBackend::Impl::openArchives() {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(options.cacheRoot, ec);
  // pipelines.count: "v2 <next file number>" then "<n> <pipelines>" per file. The legacy form is
  // a single number: the pipelines in pipelines.metalar.
  uint64_t next = 1;
  std::map<uint64_t, uint32_t> counts;
  {
    std::ifstream cf(options.cacheRoot / "pipelines.count");
    std::string head;
    if (cf >> head) {
      if (head == "v2") {
        uint64_t n = 0;
        uint32_t c = 0;
        if (cf >> n) next = std::max<uint64_t>(next, n);
        while (cf >> n >> c) counts[n] = c;
      } else {
        counts[0] = uint32_t(std::strtoul(head.c_str(), nullptr, 10));
      }
    }
  }
  std::vector<uint64_t> found;
  std::vector<fs::path> stale;  // left by an interrupted save or probe
  for (fs::directory_iterator it(options.cacheRoot, ec), end; !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    uint64_t n = 0;
    if (name == "pipelines.metalar") found.push_back(0);
    else if (archiveNumber(name, n)) found.push_back(n);
    else if (name.compare(0, 9, "pipelines") == 0 &&
             ((name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) || name == "pipelines-probe.metalar"))
      stale.push_back(it->path());
  }
  for (const fs::path& p : stale) fs::remove(p, ec);
  std::sort(found.begin(), found.end());
  if (!found.empty()) next = std::max(next, found.back() + 1);
  // Too many files: evict the oldest. Their pipelines miss once, recompile and land in this
  // session's file, which compacts the cache over time.
  size_t evicted = 0;
  while (found.size() > kMaxArchiveFiles) {
    const fs::path p = archivePath(found.front());
    fs::remove(p, ec);
    info("gpu.pso.cache evicted file=" + token(p.filename().string()));
    found.erase(found.begin());
    ++evicted;
  }
  NSMutableArray<id<MTLBinaryArchive>>* loaded = [NSMutableArray array];
  uint32_t pipelinesLoaded = 0;
  for (uint64_t n : found) {
    const fs::path p = archivePath(n);
    MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
    ad.url = [NSURL fileURLWithPath:@(p.c_str())];
    NSError* ae = nil;
    id<MTLBinaryArchive> ar = [device newBinaryArchiveWithDescriptor:ad error:&ae];
    if (!ar) {  // stale or from another OS version: remove it, its pipelines are rebuilt
      err("what=archive_load file=%s error=%s", token(p.filename().string()).c_str(), token(str(ae)).c_str());
      fs::remove(p, ec);
      continue;
    }
    [loaded addObject:ar];
    const auto c = counts.find(n);
    const uint32_t pc = c == counts.end() ? 0 : c->second;
    archiveFileCounts[n] = pc;
    pipelinesLoaded += pc;
  }
  lookupArchives = [loaded copy];
  archiveSession = next;
  archive = newArchive();
  archiveOn = archive != nil;  // without it, lookups in the loaded files still work
  info("gpu.pso.cache loaded=" + std::to_string(pipelinesLoaded) + " path=" + token(options.cacheRoot.string()) +
       " archive=" + std::to_string(archive != nil));
  info("gpu.pso.cache archives=" + std::to_string(lookupArchives.count) + " session=" + std::to_string(archiveSession) +
       " evicted=" + std::to_string(evicted));
}

bool MetalBackend::Impl::addToArchive(id<MTLBinaryArchive> ar, const ArchiveEntry& e, std::string& error) {
  NSError* ae = nil;
  const bool added = e.compute
                         ? [ar addComputePipelineFunctionsWithDescriptor:(MTLComputePipelineDescriptor*)e.desc error:&ae]
                         : [ar addRenderPipelineFunctionsWithDescriptor:(MTLRenderPipelineDescriptor*)e.desc error:&ae];
  if (!added) error = str(ae);
  return added;
}

void MetalBackend::Impl::queueArchive(id desc, bool compute, const std::string& label) {
  std::lock_guard lock(archiveQueueMutex);
  archiveQueue.push_back(ArchiveEntry{desc, compute, label});
}

void MetalBackend::Impl::drainArchiveLocked() {
  for (;;) {
    std::vector<ArchiveEntry> batch;
    {
      std::lock_guard lock(archiveQueueMutex);
      batch.swap(archiveQueue);
    }
    if (batch.empty()) return;
    for (ArchiveEntry& q : batch) {
      // A pipeline without its main function cannot be packed ("missing 'vertex' stage").
      const bool complete = q.compute ? [(MTLComputePipelineDescriptor*)q.desc computeFunction] != nil
                                      : [(MTLRenderPipelineDescriptor*)q.desc vertexFunction] != nil;
      if (!complete) {
        if (archiveWarned.insert(q.label).second)
          info("gpu.pso.cache dropped label=" + token(q.label) +
               " reason=" + (q.compute ? "no_compute_function" : "no_vertex_function"));
        continue;
      }
      // Depth-only pipelines (no fragment function) break packing of the whole archive
      // ("expecting 'fragment' stage in pipeline no. N"); they are cheap to compile, so they
      // are simply not archived.
      if (!q.compute && [(MTLRenderPipelineDescriptor*)q.desc fragmentFunction] == nil) {
        if (archiveWarned.insert(q.label).second)
          info("gpu.pso.cache dropped label=" + token(q.label) + " reason=no_fragment_function");
        continue;
      }
      if (!archiveSeen.insert(descriptorKey(q.desc, q.compute)).second) continue;  // same content already in
      if (!archive) archive = buildArchiveLocked(0, archiveEntries.size(), true);
      if (!archive) continue;  // newArchive logged why; this pipeline is simply not cached
      std::string error;
      if (!addToArchive(archive, q, error)) {
        err("what=archive_add label=%s error=%s", token(q.label).c_str(), token(error).c_str());
        if (archiveWarned.insert(q.label).second)
          info("gpu.pso.cache dropped label=" + token(q.label) + " reason=add_failed");
        continue;
      }
      archiveEntries.push_back(std::move(q));
      ++archiveUnsaved;
    }
  }
}

// A fresh archive with archiveEntries[begin, end). With dropFailures, entries that fail to add are
// dropped (only for the full range); otherwise the first failure returns nil.
id<MTLBinaryArchive> MetalBackend::Impl::buildArchiveLocked(size_t begin, size_t end, bool dropFailures) {
  id<MTLBinaryArchive> ar = newArchive();
  if (!ar) return nil;
  for (size_t i = begin; i < end && i < archiveEntries.size();) {
    std::string error;
    if (addToArchive(ar, archiveEntries[i], error)) {
      ++i;
      continue;
    }
    if (!dropFailures) return nil;
    err("what=archive_add label=%s error=%s", token(archiveEntries[i].label).c_str(), token(error).c_str());
    dropArchiveEntryLocked(i, "add_failed");
    --end;
  }
  return ar;
}

// Serializes to <dst>.tmp and renames it over dst, so a reader never sees a partial file.
bool MetalBackend::Impl::serializeArchive(id<MTLBinaryArchive> ar, const std::filesystem::path& dst,
                                          std::string& error) {
  std::filesystem::path tmp = dst;
  tmp += ".tmp";
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  NSError* e = nil;
  if (![ar serializeToURL:[NSURL fileURLWithPath:@(tmp.c_str())] error:&e]) {
    error = str(e);
    std::filesystem::remove(tmp, ec);
    return false;
  }
  std::filesystem::rename(tmp, dst, ec);
  if (ec) {
    error = "rename: " + ec.message();
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

// True when archiveEntries[begin, end) serializes on its own (written to a scratch file).
bool MetalBackend::Impl::probeArchiveLocked(size_t begin, size_t end) {
  id<MTLBinaryArchive> ar = buildArchiveLocked(begin, end, false);
  if (!ar) return false;
  const auto probe = options.cacheRoot / "pipelines-probe.metalar";
  std::string error;
  const bool good = serializeArchive(ar, probe, error);
  std::error_code ec;
  std::filesystem::remove(probe, ec);
  return good;
}

void MetalBackend::Impl::dropArchiveEntryLocked(size_t i, const char* reason) {
  info("gpu.pso.cache dropped label=" + token(archiveEntries[i].label) + " reason=" + reason);
  archiveEntries.erase(archiveEntries.begin() + std::ptrdiff_t(i));
  if (i < archiveSavedEntries) --archiveSavedEntries;
  archive = nil;  // rebuilt from the remaining entries
}

// Finds and drops the entry that makes serialization fail; false when none can be isolated.
bool MetalBackend::Impl::dropCulpritLocked(const std::string& error) {
  const size_t count = archiveEntries.size();
  size_t n = 0;
  if (archiveErrorIndex(error, n)) {
    // The packer's numbering is undocumented: try it 0- and 1-based, over all entries and over
    // render entries only. The culprit is the candidate that also fails on its own.
    std::vector<size_t> cands;
    auto add = [&](size_t i) {
      if (i < count && std::find(cands.begin(), cands.end(), i) == cands.end()) cands.push_back(i);
    };
    add(n);
    if (n) add(n - 1);
    for (size_t i = 0, r = 0; i < count; ++i)
      if (!archiveEntries[i].compute) {
        if (r == n || r + 1 == n) add(i);
        ++r;
      }
    for (size_t i : cands)
      if (!probeArchiveLocked(i, i + 1)) {
        dropArchiveEntryLocked(i, "serialize_failed");
        return true;
      }
  }
  // Bisect a failing range down to one entry. Start with the entries added since the last good
  // save: the earlier ones already serialized together.
  size_t lo = 0, hi = count;
  if (archiveSavedEntries > 0 && archiveSavedEntries < count && !probeArchiveLocked(archiveSavedEntries, count))
    lo = archiveSavedEntries;
  while (hi - lo > 1) {
    const size_t mid = lo + (hi - lo) / 2;
    if (!probeArchiveLocked(lo, mid)) hi = mid;
    else if (!probeArchiveLocked(mid, hi)) lo = mid;
    else return false;  // fails only in combination
  }
  if (lo >= hi) return false;
  dropArchiveEntryLocked(lo, "serialize_failed");
  return true;
}

void MetalBackend::Impl::writeArchiveCountLocked() {
  const auto path = options.cacheRoot / "pipelines.count";
  auto tmp = path;
  tmp += ".tmp";
  {
    std::ofstream f(tmp, std::ios::trunc);
    f << "v2 " << archiveSession + 1 << '\n';
    for (const auto& [n, c] : archiveFileCounts) f << n << ' ' << c << '\n';
    f.flush();
    if (!f) {
      err("what=archive_count error=write_failed");
      return;
    }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) err("what=archive_count error=%s", token(ec.message()).c_str());
}

bool MetalBackend::Impl::saveArchiveLocked() {
  if (!archiveOn) return false;
  drainArchiveLocked();
  if (!archiveUnsaved) return true;
  if (archiveEntries.size() > archivePackCap) return chunkSaveLocked();
  const auto file = archivePath(archiveSession);
  std::string error;
  for (int drops = 0;; ++drops) {
    // After any failure the archive object is rebuilt from the kept descriptors: a failed
    // serialize can leave it empty ("Nothing to serialize").
    if (!archive && !archiveEntries.empty()) archive = buildArchiveLocked(0, archiveEntries.size(), true);
    if (!archive || archiveEntries.empty()) break;
    if (serializeArchive(archive, file, error)) {
      archiveSavedEntries = archiveEntries.size();
      archiveFileCounts[archiveSession] = uint32_t(archiveSavedEntries);
      archiveUnsaved = 0;
      archiveSavedAt = std::chrono::steady_clock::now();
      uint64_t total = 0;
      for (const auto& fc : archiveFileCounts) total += fc.second;
      info("gpu.pso.cache saved=" + std::to_string(total) + " file=" + token(file.filename().string()) +
           " entries=" + std::to_string(archiveSavedEntries) + " drops=" + std::to_string(drops));
      if (archiveSavedEntries >= archivePackCap) {  // seal: keep each file small enough to pack
        ++archiveSession;
        archiveEntries.clear();
        archiveSavedEntries = 0;
        archive = newArchive();
      }
      writeArchiveCountLocked();
      return true;
    }
    err("what=archive_save file=%s entries=%zu error=%s", token(file.filename().string()).c_str(),
        archiveEntries.size(), token(error).c_str());
    logArchiveCulpritLocked(error);
    archive = nil;
    // "cannot pack" without a pipeline number: the file is too big for the packer, not one bad
    // entry. Bisecting would only re-serialize large sets; split right away with a lower cap.
    size_t cited = 0;
    if (error.find("cannot pack") != std::string::npos && !archiveErrorIndex(error, cited)) {
      const size_t cap = std::max(kMinPackCap, archiveEntries.size() / 2);
      if (cap < archivePackCap) {
        archivePackCap = cap;
        info("gpu.pso.cache pack_cap=" + std::to_string(cap) + " reason=cannot_pack entries=" +
             std::to_string(archiveEntries.size()));
      }
      break;
    }
    if (drops >= kArchiveDrops || !dropCulpritLocked(error)) break;
  }
  // No single culprit (the entries only fail in combination) or too many drops. Seal the session
  // file (it holds the last good save) and write the entries not on disk yet into new files of
  // their own, halving the set until each part serializes, so good pipelines are kept.
  const size_t kept = std::min(archiveSavedEntries, archiveEntries.size());
  std::vector<ArchiveEntry> pending(archiveEntries.begin() + std::ptrdiff_t(kept), archiveEntries.end());
  if (archiveFileCounts.count(archiveSession)) ++archiveSession;
  archiveEntries.clear();
  archiveSavedEntries = 0;
  archiveUnsaved = 0;
  archive = nil;
  size_t files = 0, saved = 0, lost = 0;
  for (size_t i = 0; i < pending.size(); i += archivePackCap)
    splitSaveLocked(pending.data() + i, std::min(archivePackCap, pending.size() - i), 0, files, saved, lost);
  if (!pending.empty())
    info("gpu.pso.cache split entries=" + std::to_string(pending.size()) + " files=" + std::to_string(files) +
         " saved=" + std::to_string(saved) + " dropped=" + std::to_string(lost));
  if (lost) err("what=archive_save gave_up=%zu file=%s", lost, token(file.filename().string()).c_str());
  // splitSaveLocked advanced archiveSession past every file it wrote: the next one is fresh.
  archive = newArchive();
  archiveSavedAt = std::chrono::steady_clock::now();
  writeArchiveCountLocked();
  return lost == 0 && !pending.empty();
}

// More entries than one file may hold: the session file keeps what it already has on disk (sealed),
// full chunks of archivePackCap entries become files of their own, and the remainder becomes the
// new session archive (saved now, extended by later saves).
bool MetalBackend::Impl::chunkSaveLocked() {
  const size_t cap = archivePackCap;
  const size_t kept = std::min(archiveSavedEntries, archiveEntries.size());
  std::vector<ArchiveEntry> pending(archiveEntries.begin() + std::ptrdiff_t(kept), archiveEntries.end());
  if (archiveFileCounts.count(archiveSession)) ++archiveSession;  // the file with the `kept` entries
  archiveEntries.clear();
  archiveSavedEntries = 0;
  archiveUnsaved = 0;
  archive = nil;
  const size_t full = pending.size() / cap * cap;
  size_t files = 0, saved = 0, lost = 0;
  for (size_t i = 0; i < full; i += cap) splitSaveLocked(pending.data() + i, cap, 0, files, saved, lost);
  info("gpu.pso.cache chunked entries=" + std::to_string(pending.size()) + " cap=" + std::to_string(cap) +
       " files=" + std::to_string(files) + " saved=" + std::to_string(saved) + " dropped=" + std::to_string(lost) +
       " rest=" + std::to_string(pending.size() - full));
  if (lost) err("what=archive_save gave_up=%zu chunked=1", lost);
  // splitSaveLocked advanced archiveSession past every file it wrote: the remainder starts a fresh one.
  archiveEntries.assign(pending.begin() + std::ptrdiff_t(full), pending.end());
  archiveUnsaved = uint32_t(archiveEntries.size());
  archive = archiveEntries.empty() ? newArchive() : buildArchiveLocked(0, archiveEntries.size(), true);
  archiveSavedAt = std::chrono::steady_clock::now();
  writeArchiveCountLocked();
  if (archiveEntries.empty()) return lost == 0;
  // Fewer than `cap` entries now: the ordinary path (with its own failure handling).
  return saveArchiveLocked() && lost == 0;
}

// Serializes entries[0, n) into the next session file number; on failure halves the set and
// recurses (up to kArchiveSplitDepth). A single entry that fails is dropped.
void MetalBackend::Impl::splitSaveLocked(const ArchiveEntry* entries, size_t n, int depth, size_t& files,
                                         size_t& saved, size_t& lost) {
  if (!n) return;
  std::string error;
  id<MTLBinaryArchive> ar = newArchive();
  bool built = ar != nil;
  for (size_t i = 0; built && i < n; ++i) built = addToArchive(ar, entries[i], error);
  const auto file = archivePath(archiveSession);
  if (built && serializeArchive(ar, file, error)) {
    archiveFileCounts[archiveSession] = uint32_t(n);
    ++archiveSession;
    ++files;
    saved += n;
    info("gpu.pso.cache saved_split file=" + token(file.filename().string()) + " entries=" + std::to_string(n) +
         " depth=" + std::to_string(depth));
    return;
  }
  if (n == 1 || depth >= kArchiveSplitDepth || !ar) {
    for (size_t i = 0; i < n; ++i)
      info("gpu.pso.cache dropped label=" + token(entries[i].label) +
           " reason=" + (n == 1 ? "serialize_failed" : "gave_up") + " error=" + token(error));
    lost += n;
    return;
  }
  const size_t half = n / 2;
  splitSaveLocked(entries, half, depth + 1, files, saved, lost);
  splitSaveLocked(entries + half, n - half, depth + 1, files, saved, lost);
}

// Logs the entries the packer's error cites ("pipeline no. N"), under every numbering
// dropCulpritLocked tries (0- or 1-based, over all entries or render entries only).
void MetalBackend::Impl::logArchiveCulpritLocked(const std::string& error) {
  size_t n = 0;
  if (!archiveErrorIndex(error, n)) return;
  std::vector<size_t> cands;
  auto add = [&](size_t i) {
    if (i < archiveEntries.size() && std::find(cands.begin(), cands.end(), i) == cands.end()) cands.push_back(i);
  };
  add(n);
  if (n) add(n - 1);
  for (size_t i = 0, r = 0; i < archiveEntries.size(); ++i)
    if (!archiveEntries[i].compute) {
      if (r == n || r + 1 == n) add(i);
      ++r;
    }
  for (size_t i : cands)
    info("gpu.pso.cache culprit no=" + std::to_string(n) + " entry=" + std::to_string(i) +
         " label=" + token(archiveEntries[i].label) + " " + describeEntry(archiveEntries[i].desc, archiveEntries[i].compute));
}

// On a compile worker, after the pipeline is published: adds this session's misses to the
// archive and saves every 32 additions or 30 s. If another worker holds the archive it drains
// the queue as well, so this one returns at once instead of waiting.
void MetalBackend::Impl::maybeSaveArchive() {
  if (!archiveOn) return;
  std::unique_lock lock(archiveMutex, std::try_to_lock);
  if (!lock.owns_lock()) return;
  drainArchiveLocked();
  if (!archiveUnsaved) return;
  if (archiveUnsaved < 32 && std::chrono::steady_clock::now() - archiveSavedAt < std::chrono::seconds(30)) return;
  saveArchiveLocked();
}

bool MetalBackend::saveArchive() {
  @autoreleasepool {
    std::lock_guard lock(impl_->archiveMutex);
    return impl_->saveArchiveLocked();
  }
}

}  // namespace vpgpu::metal2
