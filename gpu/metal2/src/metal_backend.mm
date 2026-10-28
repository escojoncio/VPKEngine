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
#include "vpgpu/metal2/archive_index.h"
#include "vpgpu/metal2/mtl_attr.h"

#include <TargetConditionals.h>
#include <mach/mach.h>
#if __has_include(<os/proc.h>) && (TARGET_OS_VISION || TARGET_OS_IOS)
#include <os/proc.h>
#endif
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
#include <functional>
#include <fstream>
#include <map>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vpgpu/core/activity_gauge.h"
#include "vpgpu/core/archive_harvest.h"
#include "vpgpu/core/background.h"
#include "vpgpu/core/perf.h"
#include "vpgpu/core/pipeline_cache.h"
#include "vpgpu/core/scoped_pool.h"

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
std::atomic<bool> gDumpRequest{false};  // vpgpu_metal2_request_dump: the next real frame, pass by pass
// VP_KNOCK=auto, host side: gKnockEnv = the env is set (set once at init); vpgpu_metal2_knock_arm raises gKnockArmReq, which the
// submit thread takes at the start of a frame; vpgpu_metal2_knock_progress reads the three progress values the knock publishes.
std::atomic<bool> gKnockEnv{false}, gKnockArmReq{false};
std::atomic<int> gKnockPhase{0}, gKnockDone{0}, gKnockTotal{0};  // phase: 0 idle, 1 warm-up, 2 measure, 3 drain, 4 done, 5 timeout
// Command buffers of this process that the GPU watchdog killed (not refused submissions, not victims of another client's hang):
// vpgpu_metal2_gpu_hangs, read by the host (the A/B test counts the hangs a setting causes).
std::atomic<uint32_t> gGpuHangs{0};
// One mutex for every binary-archive operation of the process (open, add, serialize), whatever MetalBackend instance does
// it: two devices (the game's and an internal client's) each have a backend, and Metal's archive code crashed inside
// strlen(NULL) when one serialized a pipelines-N.metalar.tmp while the other opened archives. Leaked on purpose (no
// static-destruction order against threads that are still running).
std::mutex& processArchiveMutex() {
  static auto* m = new std::mutex;
  return *m;
}
// Exclusive GPU time of each completed command buffer (vpgpu_metal2_cb_busy_ms, for the host's pace governor): its
// encoders' stage-boundary timestamps placed on one timeline, each counted only for how far it moves the latest end
// seen so far. A command buffer's [GPUStartTime, GPUEndTime] span holds the waits on earlier command buffers and the
// time other queues (the host's compositor) had the GPU, so it reads up to twice the work. A frame is several command
// buffers (chunks): each one's own sample groups are resolved in its own completed handler. Off until the host first
// calls vpgpu_metal2_cb_busy_ms / _stats (gBusyOn): sampling the four stage boundaries of every render encoder
// (VP_EXCL_TIMING does it too) is not done for a host that never asks. Entries are consumed by the lookup; a ring of
// 64 covers the frames in flight (12 slots, up to 8 command buffers each is more than a frame's worth).
std::atomic<bool> gBusyOn{false};
constexpr size_t kBusyRing = 64;
constexpr size_t kBusyMaxPairs = 64;  // encoders per command buffer whose (start, end) ticks are kept (vpgpu_metal2_cb_busy_intervals)
struct BusyEntry {
  const void* cb = nullptr;
  double ms = -1;  // -1: none of its encoders was sampled
  // The GPU tick pair of each sampled encoder, in encoder order (stage-adjusted on render encoders: fragment start / vertex
  // end stand in for a missing start / end; not clipped against gBusyLastEnd), for the host's overlap accounting. The ms
  // and the intervals are consumed apart (msTaken / ivTaken): the entry is freed once both were asked for.
  uint64_t pairs[kBusyMaxPairs][2] = {};
  uint32_t nPairs = 0;
  double nsPerTick = 0;
  bool msTaken = false, ivTaken = false;
};
struct BusyRec {  // the sample indices of one Metal encoder; kind 'r' render (vertex end / fragment start set), 'c' other
  char kind = 'c';
  NSUInteger start = MTLCounterDontSample, end = MTLCounterDontSample;
  NSUInteger vertexEnd = MTLCounterDontSample, fragmentStart = MTLCounterDontSample;
};
std::mutex gBusyMutex;
BusyEntry gBusyRing[kBusyRing];
size_t gBusyNext = 0;
uint64_t gBusyLastEnd = 0;  // GPU ticks
std::atomic<uint64_t> gBusyCbs{0}, gBusyUnsampled{0};
void busyAccount(const void* cb, const std::vector<std::pair<uint64_t, uint64_t>>& span,
                 const std::vector<std::pair<uint64_t, uint64_t>>& stages, const std::vector<char>& kinds,
                 double msPerTick) {
  double ms = 0;
  bool sampled = false;
  BusyEntry rec;
  std::lock_guard lock(gBusyMutex);
  for (size_t i = 0; i < span.size(); ++i) {
    uint64_t start = span[i].first, end = span[i].second;
    if (i < kinds.size() && kinds[i] == 'r' && i < stages.size()) {
      const uint64_t vtxEnd = stages[i].first, fragStart = stages[i].second;
      if (fragStart && (!start || fragStart < start)) start = fragStart;  // no vertex work: starts with fragments
      if (!end) end = vtxEnd;
    }
    if (!start || !end || end < start) continue;
    sampled = true;
    if (rec.nPairs < kBusyMaxPairs) rec.pairs[rec.nPairs][0] = start, rec.pairs[rec.nPairs][1] = end, ++rec.nPairs;
    const uint64_t from = std::max(start, gBusyLastEnd);
    if (end > from) ms += double(end - from) * msPerTick;
    gBusyLastEnd = std::max(gBusyLastEnd, end);
  }
  for (BusyEntry& e : gBusyRing)  // an entry never consumed (a queue the host does not watch) whose address came back
    if (e.cb == cb) e.cb = nullptr;
  rec.cb = cb;
  rec.ms = sampled ? ms : -1.0;
  rec.nsPerTick = msPerTick * 1e6;
  gBusyRing[gBusyNext++ % kBusyRing] = rec;
  gBusyCbs.fetch_add(1);
  if (!sampled) gBusyUnsampled.fetch_add(1);
}

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
    case Format::R8Srgb: return MTLPixelFormatR8Unorm_sRGB;
    case Format::RG8Srgb: return MTLPixelFormatRG8Unorm_sRGB;
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

// gpu.tex: live MTLTextures by the Metal pixel format they were really created with (not the Vulkan format asked for).
enum TexFmt : uint8_t {
  kFmtBC1, kFmtBC2, kFmtBC3, kFmtBC4, kFmtBC5, kFmtBC6H, kFmtBC7, kFmtAstc, kFmtEtc2, kFmtRgba8, kFmtRgba16F, kFmtRg11B10F,
  kFmtR8, kFmtR16, kFmtD32, kFmtD32S8, kFmtD16, kFmtOther, kFmtCount
};
constexpr const char* kTexFmtName[kFmtCount] = {"BC1", "BC2", "BC3", "BC4", "BC5", "BC6H", "BC7", "ASTC", "ETC2",
                                                "RGBA8", "RGBA16F", "RG11B10F", "R8", "R16", "D32", "D32S8", "D16", "other"};

TexFmt texFmtOf(MTLPixelFormat f) {
#define VP_ASTC(s) case MTLPixelFormatASTC_##s##_LDR: case MTLPixelFormatASTC_##s##_sRGB:
  switch (f) {
    case MTLPixelFormatBC1_RGBA: case MTLPixelFormatBC1_RGBA_sRGB: return kFmtBC1;
    case MTLPixelFormatBC2_RGBA: case MTLPixelFormatBC2_RGBA_sRGB: return kFmtBC2;
    case MTLPixelFormatBC3_RGBA: case MTLPixelFormatBC3_RGBA_sRGB: return kFmtBC3;
    case MTLPixelFormatBC4_RUnorm: case MTLPixelFormatBC4_RSnorm: return kFmtBC4;
    case MTLPixelFormatBC5_RGUnorm: case MTLPixelFormatBC5_RGSnorm: return kFmtBC5;
    case MTLPixelFormatBC6H_RGBUfloat: case MTLPixelFormatBC6H_RGBFloat: return kFmtBC6H;
    case MTLPixelFormatBC7_RGBAUnorm: case MTLPixelFormatBC7_RGBAUnorm_sRGB: return kFmtBC7;
    VP_ASTC(4x4) VP_ASTC(5x4) VP_ASTC(5x5) VP_ASTC(6x5) VP_ASTC(6x6) VP_ASTC(8x5) VP_ASTC(8x6) VP_ASTC(8x8)
    VP_ASTC(10x5) VP_ASTC(10x6) VP_ASTC(10x8) VP_ASTC(10x10) VP_ASTC(12x10) VP_ASTC(12x12) return kFmtAstc;
    case MTLPixelFormatETC2_RGB8: case MTLPixelFormatETC2_RGB8_sRGB: case MTLPixelFormatETC2_RGB8A1:
    case MTLPixelFormatETC2_RGB8A1_sRGB: case MTLPixelFormatEAC_RGBA8: case MTLPixelFormatEAC_RGBA8_sRGB:
    case MTLPixelFormatEAC_R11Unorm: case MTLPixelFormatEAC_R11Snorm: case MTLPixelFormatEAC_RG11Unorm:
    case MTLPixelFormatEAC_RG11Snorm: return kFmtEtc2;
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA8Sint: case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB: return kFmtRgba8;
    case MTLPixelFormatRGBA16Float: return kFmtRgba16F;
    case MTLPixelFormatRG11B10Float: return kFmtRg11B10F;
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm: case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint:
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint:
      return kFmtR8;
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Uint: case MTLPixelFormatR16Sint:
    case MTLPixelFormatR16Float: case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Uint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRG16Float:
    case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint: case MTLPixelFormatR32Float: return kFmtR16;
    case MTLPixelFormatDepth32Float: return kFmtD32;
    case MTLPixelFormatDepth32Float_Stencil8: return kFmtD32S8;
    case MTLPixelFormatDepth16Unorm: return kFmtD16;
    default: return kFmtOther;
  }
#undef VP_ASTC
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
  if (d.format == Format::RGB9E5Float) u |= MTLTextureUsagePixelFormatView;  // raw R32Uint view for copies
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

// gpu.excl: one core encoder of a submitted frame as the exclusive-time accounting sees it (VP_EXCL_TIMING).
struct EncDiag {
  char kind = 'c';  // 'r' render, 'c' compute / copy
  uint32_t color0 = 0, depth = 0;  // MTLPixelFormat of colour 0 / depth
  uint32_t width = 0, height = 0, layers = 1, samples = 1;
  uint32_t draws = 0;
  int8_t view = -1;  // Encoder::view: the one view this encoder draws of a split pass, -1 = all of them
  uint64_t sig = 0;  // VP_KNOCK: hash of the render target signature (formats, size, layers, samples); 0 = not computed
  bool knockCounted = false;  // VP_KNOCK warm-up: the encoder passed the warm-up gate (only these can win the vote)
  // The attachments' memory traffic from their load / store actions (MB at the render area's logical size, before any
  // rasterization rate map), "c0:CR d:CD" per attachment (load L/C/D, store S/R/SR/D, m = memoryless, then the frame
  // compiler's tile verdict when not on tile), and whether the encoder waited on the previous frame's join fence.
  double loadMb = 0, storeMb = 0;
  std::string attach, label;
  bool joined = false;
};

}  // namespace

struct MetalBackend::Impl {
  struct Buffer {
    uint32_t generation = 0;
    id<MTLBuffer> buffer;
    uint64_t base = 0;  // guest pointer's offset inside a page-aligned no-copy buffer
    uint64_t bytes = 0;  // MTLBuffer length, counted in bufOwnBytes (own) or bufGuestBytes (guest memory)
    bool guest = false;
    // Host address of byte 0 of the client's range and its usable length (VP_CB_ORDER=2 hazards by address).
    uintptr_t host = 0;
    uint64_t span = 0;
  };
  struct Texture {
    uint32_t generation = 0;
    id<MTLTexture> texture;  // nil until first needed (render targets)
    TextureDesc desc;
    bool failed = false;     // allocation failed once (logged then): not retried every use
    uint8_t cat = 0;         // TexCat, once allocated
    uint64_t bytes = 0;      // allocatedSize counted in texBytes[cat]
    uint64_t msaaLogical = 0;  // deferred render target still without memory: its size (msaaDeferredBytes)
    // Deferred render target (colour or depth, any sample count): memory is only a TRANSIENT backing. A frame that cannot stay on tile
    // (HeavyGeometry, loads, samples...) materialises it; after msaaReleaseFrames consecutive tile-local frames it
    // is given back (msaa_released) and the memoryless stand-in is used again.
    bool deferredRT = false;
    bool pinned = false;        // handed out by nativeTexture(): the client may keep it, never released
    bool tileBanned = false;    // a command buffer that used its memoryless stand-in failed (tiler flush): real memory from now on
    uint32_t streak = 0;        // consecutive frames served by the stand-in since the last memory use
    uint64_t usedFrame = ~0ull; // last frame that needed the real texture
    uint64_t tileFrame = ~0ull; // last frame counted in `streak`
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
    uint8_t blendMask = 0;  // colour attachments with blending on (gpu.hang draw lines)
    // The vertex function reads VpDrawParams at [[buffer(29)]] (SPIR-V DrawIndex / BaseVertex; vpmsl
    // kDrawParamsDecl), set per draw from Draw::drawIndex / first / index.
    bool drawParams = false;
    // Foveation (RenderPassDesc::rateMap): the vertex function writes [[viewport_array_index]] (multiview with a
    // rate map layer per view: the encoder sets one viewport and scissor per view), the fragment function reads
    // [[position]] (fsPosition) and, if so built, maps it back to screen space through the rate map data at
    // [[buffer(28)]] (rrmFs; vpmsl kRateMapDecl). A foveated pass only draws pipelines that can.
    bool vpIndexed = false, fsPosition = false, rrmFs = false;
    // 2: the vertex shader multiviews through vertex amplification ([[amplification_id]]): its draws run with the encoder's
    // amplification count 2 and view mappings (view v -> layer v, and viewport v when vpIndexed).
    NSUInteger amp = 1;
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
    // Impl::functions keys of the vertex / fragment (compute: [0]) functions: counted in functionUsers
    // while this pipeline is published, so the function is dropped with its last pipeline.
    uint64_t fnKeys[2] = {0, 0};
    id<MTLComputePipelineState> compute;
    MTLSize threads{1, 1, 1};
    Table stage[2];  // render: vertex, fragment; compute: [0]
    // Hang diagnostics (gpu.hang): recipe label, shader ids and what the shaders do that can
    // stall a GPU (storage buffers in the vertex stage, draw params, loops, robustness guards).
    std::string label;
    uint64_t vsId = 0, fsId = 0;
    std::string traits;
    // VP_KNOCK_FSR (only built with knockFsrOn): this pipeline's descriptor without its fragment function, and the type of each
    // colour output a trivial fragment function must write ('f' float, 'u' uint, 'i' int, '-' not written). Empty kinds = no
    // substitute is possible for it (fsr=na).
    id fsrDesc;
    std::string fsrKinds;
  };
  // Hang diagnostics: per committed, not yet completed command buffer, the pipelines it runs (by
  // encoder) and how much work. Printed when the device is lost (gpu_stuck or a fault).
  // One draw of a command buffer in flight (gpu.hang draw lines): what it asked the GPU to do.
  struct DrawRec {
    uint32_t encoder = 0, chunk = 0;
    uint64_t vsId = 0, fsId = 0;
    uint32_t count = 0, instances = 0, first = 0;
    int32_t baseVertex = 0;
    uint32_t rtWidth = 0, rtHeight = 0;
    uint16_t layers = 1;
    uint8_t samples = 1;
    uint8_t kind = 0;  // bit 0 indexed, bit 1 indirect (count/first then live in the GPU buffer)
    // State at the draw (viewport/scissor as last set, in the pass's logical space; blend = colour attachments mask).
    float vpX = 0, vpY = 0, vpW = 0, vpH = 0;
    uint32_t scX = 0, scY = 0, scW = 0, scH = 0;
    uint8_t blend = 0;
    bool depthTest = false, depthWrite = false;
  };
  // Per frame command buffer: commit-call time, commit -> scheduled and scheduled -> completed (gpu.cbtime,
  // gpu.hang cb_live / cb_slow). Tells "never started" (scheduled=no) from "runs slowly".
  struct CbTimes {
    std::string label;
    uint64_t frame = 0, seq = 0;
    std::atomic<int64_t> commitStartNs{0}, commitEndNs{0}, scheduledNs{0};
  };
  std::shared_ptr<CbTimes> trackCb(id<MTLCommandBuffer> c, uint64_t frame);
  void timedCommit(id<MTLCommandBuffer> c, const std::shared_ptr<CbTimes>& t);
  static constexpr double kSlowCbTimeMs = 200.0;
  static constexpr uint32_t kCbTimeMaxLines = 64;
  std::atomic<uint32_t> cbTimeLines{0};
  std::atomic<uint64_t> cbSeq{0};
  std::map<uint64_t, std::shared_ptr<CbTimes>> liveCbs;  // under inFlightMutex: committed, not completed
  std::deque<std::string> slowCbRecent;                  // under inFlightMutex: last gpu.cbtime lines (cap 16)
  std::string memStats();                                // device / process memory for hang and stall reports
  // Stall warnings before kStallNs (3 s and 8 s without completions): once per (completions, level).
  std::atomic<uint64_t> stallWarned{0};
  void stallWarn(uint64_t completionsNow, int64_t quietNs, const char* where);
  std::atomic<uint32_t> splitLogLines{0};
  struct InFlight {
    std::string label;
    int64_t committedNs = 0;
    uint32_t draws = 0, dispatches = 0;
    std::vector<std::string> pipelines;  // "e<encoder>:<label>[traits]" once each, capped
    size_t dropped = 0;
    // Hang split (VPVK_HANG_SPLIT): render encoders of more than kHangSplitMin draws are cut into
    // chunks of `hangSplit` draws, each committed in a command buffer of its own; `chunksDone`
    // counts the completed ones, so a hang names its chunk (and its draws) exactly.
    uint32_t chunks = 0;
    std::shared_ptr<std::atomic<uint32_t>> chunksDone;
    std::vector<DrawRec> recs;  // capped at kMaxDrawRecs
  };
  static constexpr size_t kMaxDrawRecs = 16384;
  static constexpr uint32_t kHangSplitMin = 2048;  // Quest frames stay far below; a warmup goes above
  uint32_t hangSplit = 256;  // VPVK_HANG_SPLIT=<draws per chunk>; 0 = never split
  // Draw budget per command buffer (VPVK_MAX_DRAWS_PER_CB; 0 = off). A frame with more draws is sent
  // as several consecutive command buffers on the same queue, so the compositor (which shares the
  // GPU) can slot its work between them instead of waiting behind one huge buffer. Cuts fall only
  // between encoders (a cut inside one would force Store/Load of its MSAA and depth); only the hang
  // split cuts inside an encoder.
  uint32_t maxDrawsPerCb = 256;  // command buffer budget in draw units (VPVK_MAX_DRAWS_PER_CB), see encoderCost
  // Encoder budget per command buffer (VPVK_MAX_ENCODERS_PER_CB; 0 = off), counting EVERY Metal encoder (blit, compute,
  // render), unlike the draw budget above. Each open encoder pins an IOAccelerator region of 64-128 KiB that Metal does
  // not count and frees only when the command buffer completes (AC Nexus: ~39000 blit encoders in 3 s, +3 GB, jetsam).
  // 512 encoders = 32-64 MiB per command buffer: far above a normal frame (tens of encoders, one command buffer, so
  // nothing changes for them) and small enough that a few in flight stay in the tens of MiB. A cut falls only where no
  // encoder is open (between encoders), like the draw budget.
  uint32_t maxEncodersPerCb = 512;
  // Chunks (command buffers committed before the last one) of one frame that may be uncompleted at once
  // (VPVK_MAX_PENDING_CHUNKS; 0 = unlimited). Above it submit waits for the oldest: that wait is what bounds the memory,
  // the budget alone only batches it. Two keep the GPU fed (one running, one queued) while encoding goes on.
  uint32_t maxPendingChunks = 2;
  // Fences a slot keeps between frames (one per encoder is created within a frame; the excess is released on reuse).
  static constexpr size_t kFenceKeep = 256;
  double encoderCost(const Encoder&, const std::vector<Command>&);
  // gpu.slowcb: one line per command buffer that kept the GPU longer than kSlowCbMs, capped.
  static constexpr double kSlowCbMs = 8.0;
  static constexpr uint32_t kSlowCbMaxLines = 64;
  std::atomic<uint32_t> slowCbLines{0};
  std::string hangDir;       // VPVK_HANG_DIR (else KL_CAPTURE_DIR, else Documents/Registros): MSL dumps
  std::atomic<bool> hangDumped{false};
  std::mutex inFlightMutex;
  std::map<uint64_t, InFlight> inFlightCbs;  // by frame index
  std::atomic<uint32_t> lastFrameSplits{0};  // hang-split cuts (store+load of the attachments) in the newest submitted frame
  std::atomic<uint32_t> resolveRefused{0};   // 1x resolves turned into plain stores (VPVulkan makes them copies: should stay 0)
  std::atomic<uint32_t> hangEncReports{0};   // gpu.hang.enc reports written (capped)
  // gpu.hang.enc: every core encoder of a frame whose command buffer failed, with Metal's verdict on it and, when the GPU
  // sampled it, when it started and ended. Completion handler only; nothing is recorded while encoding.
  void hangEncoders(uint64_t frame, NSError* error, const std::vector<std::pair<NSUInteger, NSUInteger>>& encSamples,
                    id<MTLCounterSampleBuffer> samples, NSUInteger sampleCount);
  // A command buffer refused because the app is in the background (isBackgroundError): not a hang. One gpu.cb.background
  // line per second at most (a pause refuses every frame), never a draw / pipeline / MSL dump. Any thread.
  std::atomic<uint64_t> bgErrors{0};
  std::atomic<int64_t> bgLoggedNs{0};
  void noteBackgroundError();
  void dumpInFlight(const char* why, bool early = false);  // early: stall warning, leaves the once-per-run MSL dump alone
  void dumpHungDraws(const InFlight&, bool early = false);
  // Per frame in flight: things the GPU may still read until that frame completes.
  struct Slot {
    id<MTLBuffer> args;  // first argument buffer (kArgRingBytes)
    // Chained argument buffers for frames whose tables do not fit in `args` (a prewarm frame that
    // draws thousands of pipelines once): kept with the slot, reused by its later frames.
    std::vector<id<MTLBuffer>> argsMore;
    id<MTLCounterSampleBuffer> samples;
    std::vector<id<MTLFence>> fences;
    std::vector<id> retired;  // destroyed resources kept alive until the slot is reused
    uint64_t retiredBytes = 0;  // allocatedSize of `retired` (diagnostics: how much the graveyard pins)
    uint64_t frame = UINT64_MAX;  // the frame that last used the slot (UINT64_MAX: none, or native work)
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
  // No residency set: buffers whose GPU address was handed out (bufferAddress) are reached by shaders through raw pointers,
  // so every encoder declares them with useResource (useBound). Guarded by addressMutex; removed at destroyBuffer.
  std::mutex addressMutex;
  std::vector<id<MTLBuffer>> addressTaken;
  dispatch_semaphore_t inFlight;
  uint64_t submits = 0;
  std::vector<Slot> slots;
  // Destroyed textures the host got a raw pointer to (nativeTexture): kept a while before they enter the graveyard, in case
  // the host still uses the pointer. Released (promotePinned) after kPinnedHoldNs AND 4 completed frames, or once the GPU is idle.
  struct PinnedHold {
    id object;
    int64_t ns;
    uint64_t frames;  // finishedBelow when destroyed
  };
  std::vector<PinnedHold> pinnedHold;
  void promotePinned();
  std::vector<id> graveyard;
  uint64_t graveyardBytes = 0;  // allocatedSize of `graveyard` (retire() callers are serialized like the rest)
  // Per slot: every command buffer of its last frame completed (set by that frame's publish, cleared by
  // acquireSlot). A completed slot's retired objects have no GPU reader left (frames complete in queue order).
  // The graveyard is only read by the last acquired frame and earlier ones: free once that slot is complete.
  std::unique_ptr<std::atomic<bool>[]> slotDone;
  int lastSlot = -1;  // slot of the last acquireSlot (submit thread, serialized like the graveyard)
  std::atomic<int64_t> lastDrainNs{0};
  // CPU-visible contents of the buffers VPKEngine itself allocates (own shared buffers, argument rings), for the
  // memory-ownership dump: which IOAccelerator regions of the process are ours. Any thread; a plain map.
  std::mutex rangesMutex;
  std::map<uintptr_t, uint64_t> ownRanges;
  void rangeAdd(const void* p, uint64_t n) {
    if (!p || !n) return;
    std::lock_guard lock(rangesMutex);
    ownRanges[reinterpret_cast<uintptr_t>(p)] = n;
  }
  void rangeDel(const void* p) {
    if (!p) return;
    std::lock_guard lock(rangesMutex);
    ownRanges.erase(reinterpret_cast<uintptr_t>(p));
  }
  size_t drainIdle(std::string& report);
  // Diagnostics (gpu.metal counters per second, read by vpvk_diag_line): relaxed atomics, no locks.
  struct DiagCounters {
    std::atomic<uint64_t> bufNew{0}, bufNewBytes{0}, bufNoCopy{0}, bufNoCopyBytes{0}, bufInternal{0}, bufInternalBytes{0};
    std::atomic<uint64_t> texNew{0}, texNewBytes{0}, texViews{0};
    std::atomic<uint64_t> cbs{0}, encRender{0}, encCompute{0}, encBlit{0};
    std::atomic<uint64_t> cutEnc{0}, chunkWaits{0}, chunkWaitUs{0};  // encoder-budget cuts; backpressure waits and their time
    std::atomic<uint64_t> resAdd{0}, resRemove{0}, resCommit{0};
    std::atomic<uint64_t> heldObjs{0}, heldBytes{0};  // gauges: retired objects (graveyard + slots) and their bytes
    std::atomic<uint64_t> drainedObjs{0}, drainedBytes{0}, drainRuns{0};
    // Texture lifetime: textures-map entries (created / destroyed; live = difference), explicit views destroyed, format views
    // cached / dropped. A live MTLTexture count that grows while g_tex_live and g_views stay flat is held outside the maps.
    std::atomic<uint64_t> texCreated{0}, texDestroyed{0}, viewsDestroyed{0}, fmtViewNew{0}, fmtViewDrop{0};
  } cnt;
  static void bump(std::atomic<uint64_t>& a, uint64_t n = 1) { a.fetch_add(n, std::memory_order_relaxed); }
  bool stageSamples = false;
  double gpuTickNs = 1.0;  // GPU timestamp period, calibrated against the CPU clock

  std::unordered_map<uint32_t, Buffer> buffers;
  std::unordered_map<uint32_t, Texture> textures;
  std::unordered_map<uint32_t, Sampler> samplers;
  // Guards the resource tables (buffers, textures, samplers, views, formatViews, backedMsaa), the graveyard, the pinned
  // holds and the residency set. A guest thread creating or destroying an image or buffer rehashed a map while submit
  // looked into it (SIGSEGV in Impl::texture()). submit holds it while it encodes, EXCEPT while it waits for something
  // outside the tables (a parked commit, the GPU): those waits need other threads (the host's main thread, completion
  // handlers) that may be calling into the tables. Nothing under it takes another backend mutex that a holder of
  // that one could be waiting on tableMutex for (doneMutex/inFlightMutex/pipelineMutex holders never touch the tables).
  // Recursive: texture() is reached from locked callers.
  std::recursive_mutex tableMutex;
  bool tileMemory = false;  // memoryless storage + framebuffer fetch
  // Global kill switch, last resort: a memoryless failure in a command buffer with no stand-ins recorded, or more than
  // kMaxMemorylessBans failed command buffers. From then on every tile-local attachment gets real memory (correctness over footprint).
  // Normally a failed command buffer bans only its own stand-in textures (Texture::tileBanned).
  std::atomic<bool> memorylessFailed{false};
  // A memoryless stand-in used by a command buffer: the texture (index, generation). Recorded per command buffer at
  // encode time by the submit thread; completion handlers (any thread) read their own copy of the shared list.
  struct StandIn {
    uint32_t index = 0, generation = 0;
    uint32_t encoder = 0;  // core encoder that rendered into it (the label digits Metal's encoder info gives back)
    bool heavy = false;    // that encoder was past half of the HeavyGeometry limits (Encoder::heavy)
  };
  using StandIns = std::vector<StandIn>;
  static constexpr uint32_t kMaxMemorylessBans = 8;
  std::mutex banMutex;
  std::vector<StandIns> pendingBans;       // guarded by banMutex: one group per failed command buffer, drained by submit
  uint32_t memorylessBans = 0;             // submit thread only: failed command buffers counted so far
  // Completion handler (any thread). `faulted`: core encoders Metal reported as faulted (may be empty).
  void memorylessFailure(uint64_t frame, const StandIns& stand, const std::vector<uint32_t>& faulted);
  void tileRaisedFailure(uint64_t frame, NSError* e);
  void applyMemorylessBans();                                       // submit thread, start of a frame
  MetalCaps caps;
  // Memoryless textures by shape; each encoder takes them in order, so two tile-local attachments
  // of one shape get distinct textures while encoders reuse them (contents never outlive one).
  struct MemorylessPool {
    std::vector<id<MTLTexture>> textures;
    uint64_t lastFrame = 0;
  };
  std::unordered_map<uint64_t, MemorylessPool> memoryless;
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

  // ---- Foveation (RenderPassDesc::rateMap, gpu/vulkan/src/foveation.h) ----
  // A foveated encoder renders into physically smaller shadow attachments through a MTLRasterizationRateMap (one
  // layer per view, selected by viewport index) and expands them into the game's (logical) attachments when it
  // ends, so nothing outside the encoder sees the physical layout: a render encoder before it "folds" the logical
  // contents into the shadows when the pass loads them, one after it "expands" what it stored. Submit thread only.
  struct FovMap {
    id<MTLRasterizationRateMap> map;
    id<MTLBuffer> data;  // parameter data the shaders read ([[buffer(0)]] of fold/expand, [[buffer(28)]] of draws)
    NSUInteger physW = 0, physH = 0;  // largest physical size over the layers
  };
  std::unordered_map<uint64_t, FovMap> fovMaps;  // by RateMap::key; failure cached (map nil)
  FovMap* fovMapFor(const RateMap&);
  FovMap fovIdentity;  // identity map: bound to draws of pipelines that map FragCoord outside a foveated pass
  bool fovIdentityTried = false;
  id<MTLBuffer> fovIdentityData();
  struct FovShadowPool {
    std::vector<id<MTLTexture>> textures;
    uint64_t lastFrame = 0;
    uint64_t bytes = 0;
    uint8_t cat = 0;
  };
  std::unordered_map<uint64_t, FovShadowPool> fovShadows;  // by (format, physical size, layers)
  id<MTLTexture> fovShadow(MTLPixelFormat pf, NSUInteger w, NSUInteger h, NSUInteger layers, bool depth,
                           std::unordered_map<uint64_t, size_t>& taken, uint64_t frame);
  id<MTLLibrary> fovLibrary;
  bool fovLibraryFailed = false;
  double fovLibraryRetryAt = 0;
  std::unordered_map<uint64_t, id<MTLRenderPipelineState>> fovPipelines;  // by kind << 32 | pixel format; nil = failed
  id<MTLDepthStencilState> fovDepthState;
  // kind: 0 expand colour, 1 fold colour, 2 expand depth, 3 fold depth.
  id<MTLRenderPipelineState> fovPipeline(int kind, MTLPixelFormat pf, std::string& error);
  // Forced-MSAA seed (Attachment::needsSeed): the start of an encoder whose autoMsaa slots continue a picture an earlier encoder
  // resolved draws one fullscreen triangle per layer that fills every sample from the 1x partner (colour: the texel read as is;
  // depth: depth(any)). One pipeline (and library) per signature of the encoder's attachments; nil = failed (logged once).
  std::unordered_map<uint64_t, id<MTLRenderPipelineState>> seedPipelines;
  id<MTLDepthStencilState> seedDepthOff, seedDepthOn;  // compare Always, depth write off / on
  id<MTLRenderPipelineState> seedPipeline(MTLRenderPassDescriptor* rp, bool layered, bool color, bool depth, std::string& error);
  std::atomic<uint64_t> seedDrawsDone{0};
  uint64_t fovEncoders = 0;  // foveated encoders so far (log cadence)
  std::unordered_set<std::string> fovWarned;

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
  // Published pipelines using each function key (guarded by pipelineMutex). A function whose last
  // pipeline is destroyed leaves `functions` (the PSO does not need it; its library is freed with it).
  std::unordered_map<uint64_t, uint32_t> functionUsers;
  // Pipeline keys destroyPipeline asked to free; taken by the submit thread (the only reader of the
  // `pipelines` nodes) at the start of a submit. A compilePipeline of the key meanwhile cancels it.
  std::unordered_set<uint64_t> pipelineDrops;
  void dropPipelines();
  // Live memory, for memStats (any thread): texture bytes by TexCat, buffers Metal allocated vs
  // guest memory wrapped no-copy, and the tile-local attachments that had to stay in memory.
  enum TexCat { kTexAstc, kTexColor, kTexDepth, kTexMsaa, kTexCats };
  std::atomic<uint64_t> texBytes[kTexCats] = {};
  // gpu.tex: live MTLTextures by real Metal pixel format (views excluded), memoryless apart from those with memory.
  // Bumped where textures are created and retired, logged every 60 s by the submit thread.
  std::atomic<uint64_t> texFmtN[kFmtCount] = {}, texFmtBytes[kFmtCount] = {}, texFmtMlN[kFmtCount] = {};
  int64_t texLineNs = 0;  // steady ns of the last gpu.tex line (0 = none yet); submit thread
  void texLive(id<MTLTexture> t, bool add);
  void texLine();
  // Render targets (colour or depth, any sample count) created deferred that have no memory right now (kept on tile,
  // memoryless): their size had they been resident. Moves to texBytes[cat] when something forces the allocation.
  // (The msaa_* names are kept in the logs for compatibility.)
  std::atomic<uint64_t> msaaDeferredBytes{0};
  // Bytes a texture of `d` takes: what the memoryless counter adds for one that has no memory. Metal's real texel
  // size: Apple GPUs have no D24S8, it is Depth32Float_Stencil8 (8 bytes).
  static uint64_t logicalBytes(const TextureDesc& d) {
    const uint64_t texel = d.format == Format::D24UnormS8 || d.format == Format::D32FloatS8
                               ? 8
                               : formatInfo(d.format).bytesPerBlock;
    return uint64_t{d.width} * d.height * std::max<uint32_t>(1, d.layers) * std::max<uint32_t>(1, d.samples) * texel;
  }
  // Memory ledger (the heartbeat's "where did the RAM go"): every MTLTexture allocation, by why it was made.
  //   rt_resident   render targets / sampled textures created with memory at once (persistent)
  //   msaa_mat      deferred MSAA targets that a frame forced into memory (msaa_materialized), still held
  // and the largest live textures, so a jump names its owner (largestTextures, memLedger).
  std::atomic<uint64_t> msaaMaterializations{0}, msaaReleases{0};
  std::string memLedger();  // submit thread (walks `textures`)
  // The last ledger, refreshed by the submit thread every kLedgerFrames frames and on every materialise/release, read
  // by memStats() from any thread.
  static constexpr uint64_t kLedgerFrames = 120;
  std::mutex ledgerMutex;
  std::string ledgerLast;
  uint64_t ledgerFrame = 0, ledgerEvents = 0;
  // Why the next texture() allocation happens (the render-target lambda sets it: the frame compiler's verdict);
  // only read for the log line of an MSAA colour target materialising. Submit thread only.
  const char* matReason = nullptr;
  unsigned matLoad = ~0u, matStore = ~0u;  // client load/store of that attachment (LoadAction / StoreAction values; ~0u = none), log only
  // Transient MSAA backing: deferred MSAA textures that currently own memory (candidates for release).
  std::unordered_set<uint32_t> backedMsaa;
  uint32_t msaaReleaseFrames = 120;  // VPVK_MSAA_RELEASE_FRAMES; 0 = sticky (old behaviour)
  void releaseMsaaBackings(uint64_t frameIndex);
  // trimMemory (any thread) asks the submit thread, the only owner of textures and slots, to give back what
  // is rebuildable: 0 = nothing pending, else the level; trimSlotsLeft = slots whose chained argument buffers
  // are still to be dropped (each is visited when the submit acquires it, i.e. when it is idle).
  std::atomic<uint64_t> trimmedArgsBytes{0};  // chained argument buffers given back by trim requests (cumulative)
  std::atomic<uint64_t> trimmedMsaaBytes{0};  // MSAA backings given back by trim requests (cumulative)
  std::atomic<int> trimLevel{0};
  std::atomic<uint32_t> trimSlotsLeft{0};
  std::atomic<uint64_t> argsMoreBytes{0};  // bytes of every slot's chained argument buffers
  std::atomic<uint64_t> argsRingBytes{0};  // bytes of every slot's first argument buffer
  std::atomic<uint64_t> bufIntLiveBytes{0};  // internal staging buffers whose command buffer has not completed
  std::atomic<uint64_t> fovShadowBytes{0};   // bytes of the pooled foveation shadow textures
  std::atomic<uint64_t> liveViews{0};        // views created and not destroyed
  std::atomic<uint64_t> bufOwnBytes{0}, bufGuestBytes{0}, tileRejectedTotal{0};
  // Function keys whose library compile is running (guarded by pipelineMutex); a second caller
  // waits on functionReady instead of compiling the same MSL again.
  std::unordered_set<uint64_t> compiling;
  std::condition_variable functionReady;
  // Speculative library compiles (prewarmShader): two serial queues, low QoS. A pipeline worker that
  // needs a key not yet started compiles it itself at once, so a draw never waits behind them: a key is
  // in `compiling` only while a thread is really compiling it, and a prewarm job that finds it there (or
  // ready) just returns. A prewarm job that is about to compile waits (kPrewarmYieldMs at most) while
  // compileGauge is busy, so speculative compiles do not slow the ones a draw waits for.
  dispatch_queue_t prewarmQueues[2] = {nil, nil};
  static constexpr int kPrewarmYieldMs = 2000;
  // Compiles a draw can be waiting for: compilePipeline (the whole pipeline build) and the library compiles of
  // function() that are not prewarm.
  ActivityGauge compileGauge;
  std::atomic<uint32_t> prewarmNext{0};
  std::atomic<bool> closing{false};
  std::mutex libStatsMutex;
  // AOT accounting: pipelines served from the loaded binary archives vs compiled in this process
  // (hot), and the time spent on the hot ones (the stutter an archive miss costs).
  std::atomic<uint32_t> psoArchived{0}, psoHot{0};
  std::mutex psoHotMutex;
  double psoHotMs = 0, psoHotMaxMs = 0;
  void notePso(bool archived, double ms, const PipelineState& s, const char* kind);
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
    uint64_t key = 0;  // descriptorKey: what archive_index persists
    std::chrono::steady_clock::time_point queuedAt{};  // when queueArchive got it (the harvest's anti-starvation age)
  };
  // Lookup index, built in openArchives and immutable afterwards. Files with a .keys sidecar are
  // asked only for their own pipelines (keyFile: descriptorKey -> index in keyedArchives); files
  // without one (older caches) are searched for the keys the index does not know.
  NSArray<NSArray<id<MTLBinaryArchive>>*>* keyedArchives = nil;  // one single-element array per keyed file
  NSArray<id<MTLBinaryArchive>>* legacyArchives = nil;
  std::unordered_map<uint64_t, uint32_t> keyFile;
  std::atomic<uint32_t> archiveEvicted{0};
  // vk.pso.census: the fate of every pipeline queued for the archive (MemStats::archiveAdded...). archiveAdded counts the
  // entries added and not lost, so a drop at save time moves one from it to archiveSkipOther. archiveSerialized counts
  // those of them already written to disk; archiveSpilled the ones queueArchive refused at its fuse; archivePendingN the
  // queue length (written under archiveQueueMutex, read lock free).
  std::atomic<uint32_t> archiveAdded{0}, archiveSkipDepth{0}, archiveSkipOther{0}, archiveDup{0};
  std::atomic<uint32_t> archiveSerialized{0}, archiveSpilled{0}, archivePendingN{0};
  std::atomic<uint32_t> psoFailLogged{0};  // vk.pso.fail lines written (capped at 64)
  NSArray<id<MTLBinaryArchive>>* archivesFor(uint64_t key) const {
    const auto it = keyFile.find(key);
    if (it != keyFile.end()) return keyedArchives[it->second];
    return legacyArchives.count ? legacyArchives : nil;
  }
  void writeKeysLocked(uint64_t n, const ArchiveEntry* entries, size_t count);
  bool archiveOn = false;  // cacheRoot set and a session archive exists: misses are archived
  NSArray<id<MTLBinaryArchive>>* lookupArchives = nil;  // set in init, immutable: read without a lock
  std::mutex archiveQueueMutex;  // guards archiveQueue only (taken briefly by compile workers)
  // Misses not yet added to `archive`, oldest first. The harvest (adding recompiles each one in Metal's compiler service)
  // runs continuously on the save queue in batches of vpgpu::kHarvestBatch and steps aside only while a game thread waits
  // for a pipeline (vpgpu::demandWaiters, archiveHarvestYields; maybeSaveArchive). Nothing is dropped short of the fuse
  // (the descriptors keep their functions alive): beyond kArchiveFuse entries queueArchive counts archiveSpilled.
  std::deque<ArchiveEntry> archiveQueue;
  static constexpr size_t kArchiveFuse = 32768;
  std::atomic<uint32_t> archiveYields{0};  // times the harvest stepped aside for a demand wait
  std::atomic<uint32_t> archiveFlushWaiters{0};  // saveArchive() calls waiting for archiveMutex: the harvest job lets them in
  uint64_t pauseHook = 0;  // BackgroundGate pause hook id (0 = none): the app went to the background, flush
  double archiveAddMs = 0;  // addToArchive time since the last gpu.archive.save line (guarded by archiveMutex)
  std::atomic<uint32_t> archiveEntriesSeen{0};  // archiveEntries.size() at the last memStats that got the lock
  // Every archive operation of the process: open, add, serialize (see processArchiveMutex). One mutex shared by all
  // backends, so a second device's openArchives never overlaps another's serialize.
  std::mutex& archiveMutex = processArchiveMutex();  // guards everything below
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
  // Content keys of every descriptor archived this session (a failed add is not recorded): pipeline states that differ
  // only in what Metal keeps outside the render pipeline (depth-stencil state, cull, dynamic
  // state) produce identical descriptors, and duplicates are never added twice.
  std::unordered_set<uint64_t> archiveSeen;
  std::chrono::steady_clock::time_point archiveSavedAt = std::chrono::steady_clock::now();
  // Depth-only render pipelines (no fragment function) make an archive that holds anything else fail to serialize
  // ("expecting 'fragment' stage"), so they have archives of their own: pipelines-depth-<n>.metalar (+ .keys), opened at
  // start like the main ones. If one fails to serialize only it is dropped (depthOff: no more of them this session, they
  // count in archiveSkipDepth); the main archive is never involved. Guarded by archiveMutex.
  id<MTLBinaryArchive> depthArchive;
  std::vector<ArchiveEntry> depthEntries;  // in depthArchive, add order
  uint32_t depthUnsaved = 0;               // added since the last serialize
  uint64_t depthSession = 1;               // number of this session's depth file
  bool depthOff = false;
  std::chrono::steady_clock::time_point depthSavedAt = std::chrono::steady_clock::now();
  void openArchives();
  std::filesystem::path archivePath(uint64_t n) const;
  std::filesystem::path depthPath(uint64_t n) const;
  void saveDepthLocked(bool force);
  id<MTLBinaryArchive> newArchive();
  bool addToArchive(id<MTLBinaryArchive> ar, const ArchiveEntry& e, std::string& error);
  void queueArchive(id desc, bool compute, const std::string& label);
  // Adds up to `max` queued entries to the archive, vpgpu::kHarvestBatch at a time; returns how many it processed. With
  // `yield` it stops early when the harvest must step aside (harvestShouldYield): the rest goes back to the front of the
  // queue and *yielded is set. Without it (final flush, saves) it never stops short.
  size_t drainArchiveLocked(bool yield, size_t max, bool* yielded = nullptr);
  size_t drainStepLocked(bool yield, bool* yielded);  // drainArchiveLocked for one harvest step (fits the session file)
  bool archiveSaveDue() const;
  bool flushArchiveLocked(std::chrono::milliseconds budget);  // final / background flush, bounded by `budget`
  void harvestEntryLocked(ArchiveEntry& q);  // one queued entry into the archive (or counted as skipped)
  bool harvestShouldYield(std::chrono::steady_clock::time_point oldest, size_t inHand);
  id<MTLBinaryArchive> buildArchiveLocked(size_t begin, size_t end, bool dropFailures);
  bool serializeArchive(id<MTLBinaryArchive> ar, const std::filesystem::path& dst, std::string& error,
                        size_t entries);
  bool probeArchiveLocked(size_t begin, size_t end);
  void dropArchiveEntryLocked(size_t i, const char* reason);
  bool dropCulpritLocked(const std::string& error);
  void logArchiveCulpritLocked(const std::string& error);
  void splitSaveLocked(const ArchiveEntry* entries, size_t n, int depth, size_t& files, size_t& saved,
                       size_t& lost);
  bool chunkSaveLocked();
  void writeArchiveCountLocked();
  bool saveArchiveLocked(bool drain = true);  // drain: add the queued entries first
  bool maybeSaveArchive();  // true: postponed (a demand wait is in progress), the save job looks again shortly
  // <cacheRoot>/archive.inflight exists while archive code runs (open, add, save, lookups), so a crash inside it -- also in
  // Apple's <redacted> frames -- leaves the file for the host to find (AVP turns the archive off, VP_PIPELINE_ARCHIVE=0).
  // Nested / concurrent sections share one file.
  std::mutex inflightMutex;
  int inflightDepth = 0;
  void inflightEnter(const char* what);
  void inflightLeave();
  struct Inflight {
    Impl* m;
    Inflight(Impl* impl, const char* what) : m(impl) { m->inflightEnter(what); }
    ~Inflight() { m->inflightLeave(); }
    Inflight(const Inflight&) = delete;
    Inflight& operator=(const Inflight&) = delete;
  };
  // Background saving: compile workers only call requestArchiveSave() (never blocks); one job at a
  // time runs maybeSaveArchive() on a utility-QoS serial queue. A request that arrives while the job
  // runs sets saveWant and makes the job loop again (coalesced, nothing is lost). The destructor waits
  // for the job (it uses `this`); saveArchive() (final / background flush) takes archiveMutex directly.
  void requestArchiveSave();
  void runArchiveSaveJob();
  void waitArchiveSaveJob();
  std::mutex saveMutex;  // guards the four below
  std::condition_variable saveCv;
  bool saveWant = false, saveRunning = false;  // saveRunning covers "queued" and "running"
  dispatch_queue_t saveQueue = nil;
  std::atomic<uint32_t> saveLogged{0};  // gpu.archive.save lines written (capped)

  std::mutex doneMutex;
  std::deque<Done> done;
  // Frame completion for fences (waitFrames), under doneMutex: every frame index below
  // `finishedBelow` has completed; later ones that completed first wait in `finishedAhead`.
  std::condition_variable doneCv;
  uint64_t finishedBelow = 0;
  std::set<uint64_t> finishedAhead;
  // Any command buffer completed (frames and native work): stall detection in the waits.
  std::atomic<uint64_t> completions{0};
  std::atomic<uint64_t> chunkCompletions{0};  // command buffers of multi-chunk frames completed (progress for the backlog wait)
  // Device lost (see the header). `lostAtNs` is steady-clock ns when it happened.
  std::atomic<bool> lostFlag{false};
  std::mutex lostWhyMutex;
  std::string lostWhy;  // first loss reason (lostReason())
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

  // Frame ordering (VP_CB_ORDER; 0 = off, the default: only a frame's FIRST encoder waits on the previous frame's tail).
  // Resources are hazard-untracked and the frame compiler checks hazards per frame, so a later encoder without an in-frame
  // barrier may run while the previous frame is still drawing (a copy or compute writing vertex/constant data that a
  // render pass of the previous submission still reads). cbOrder (VP_CB_ORDER=1): every encoder of a frame waits (at its
  // vertex stage) on a join fence that follows the previous frame's whole tail. Measured (gpu.order): frames whose first
  // encoder started before the previous one's last ended, completions out of submission order, slots found busy.
  // A "command buffer" below is a submitted frame (all its chunks).
  bool cbOrder = false;
  std::atomic<uint64_t> carriedWaits{0};
  uint64_t lastCbFrame = UINT64_MAX;  // submit thread
  struct CbSpan { uint64_t start = 0, end = 0, prev = UINT64_MAX; };
  std::map<uint64_t, CbSpan> cbSpans;  // guarded by doneMutex; last ~64 frames
  uint64_t cbCompleted = 0, cbOverlaps = 0, cbOutOfOrder = 0;  // doneMutex
  std::atomic<uint64_t> slotBusyWaits{0};
  double cbMaxOverlapMs = 0;
  // VP_CB_ORDER=2 (tracked): the previous frame's tail still goes through the join fence, but only the encoders that touch
  // what a frame still in flight writes (or write what it reads) wait on it; the rest may overlap the previous frames
  // (vertex work of the next pass under the fragment tail of the last one). Hazards are checked by memory: buffer ranges by
  // host address (aliased VkBuffers of one VkDeviceMemory overlap there), textures by identity. The join fence stays in
  // the frontier, so the next frame's join still follows everything. Frames may then complete out of order: a slot is
  // reused (and its retired objects released) only once every frame up to its own completed.
  bool cbTracked = false;
  struct CbAccess {
    uint64_t frame = 0;
    std::vector<std::pair<uintptr_t, uintptr_t>> bufW, bufR;  // host address ranges [begin, end)
    std::vector<uint64_t> texW, texR;                         // (generation << 32) | index
  };
  std::deque<CbAccess> cbAccess;  // submit thread: frames submitted and possibly still running
  std::atomic<uint64_t> trackedJoined{0}, trackedFree{0};
  // encodeNative ran since the last frame: its client work is in nobody's access list, the next frame joins every encoder.
  // Written by encodeNative and read by submit: both run under the device's submit mutex (entry.cpp), so no atomic.
  bool nativeSinceFrame = false;
  // Every frame up to `frame` completed (frameFinished); with cbTracked a slot's leftovers may be used by all of them.
  bool framesDoneThrough(uint64_t frame) {
    std::lock_guard lock(doneMutex);
    return finishedBelow > frame;
  }
  // The slot's frame completed and, when completion may be out of order, so did every frame before it.
  bool slotSettled(size_t ix) {
    return slotDone[ix].load(std::memory_order_acquire) &&
           (!cbTracked || slots[ix].frame == UINT64_MAX || framesDoneThrough(slots[ix].frame));
  }

  // Exclusive GPU time per encoder (gpu.excl; VP_EXCL_TIMING=1 turns it on, VP_EXCL_SECONDS the window). Every encoder's
  // start/end timestamps (render: vertex start/end and fragment start/end) are placed on one GPU timeline across frames:
  // an encoder's exclusive time is how far it moved the latest end seen so far (submission order), so the exclusive times
  // plus the idle gaps add up to the GPU's busy span. Summed per frame type (same encoders, formats, sizes) over the
  // window, then logged.
  struct ExclEnc {
    std::string desc;  // kind, size, format, samples, attachments: from the first frame seen
    int8_t view = -1;  // Encoder::view, same
    double excl = 0, span = 0, vtx = 0, frag = 0, draws = 0, loadMb = 0, storeMb = 0;
    uint32_t vtxN = 0, fragN = 0, joined = 0;
  };
  struct ExclType {
    uint32_t n = 0;  // frames of this type in the window
    double gpu = 0, excl = 0, idle = 0;
    std::vector<ExclEnc> enc;
  };
  bool exclOn = false;
  // On-device per-pipeline cost ranking (VP_KNOCK=auto; needs exclOn, which it turns on). Apple GPUs give no per-draw
  // timestamps, so: ~300 warm-up frames find the dominant render encoder (largest exclusive time, grouped by its render
  // target signature) and count the draws of each (vsId, fsId) pair in it; then, per candidate pair, blocks of 8 frames
  // alternate with the pair's draws left out of the dominant encoders (the draw call only: every state is still bound) and
  // untouched, two passes; the medians of the dominant encoders' exclusive / vertex / fragment time of each block give the
  // pair's cost. The submit thread owns the phase; the completion handlers only add samples (knockMutex). VP_KNOCK=auto
  // starts Idle: the host's vpgpu_metal2_knock_arm (in game) starts the warm-up; only frames with a render encoder of
  // samples > 1 and >= kKnockWarmDraws draws count towards the 300.
  // VP_KNOCK_FSR (default on, =0 off): one extra pass after the knock passes, per candidate a baseline block and a block with
  // the pair's pipelines swapped for ones whose fragment function is a trivial constant-colour one (same vertex function,
  // depth-stencil, blend, formats and coverage): base - that = the pair's fragment cost. The substitutes are built on
  // knockFsrQueue from the end of the warm-up; a candidate whose are not all ready when its blocks come is not measured
  // (fsr=pending), one that cannot have them is fsr=na.
  using KnockKey = std::pair<uint64_t, uint64_t>;  // (vsId, fsId)
  struct KnockPair {
    uint32_t draws = 0;
    uint64_t verts = 0;  // view-vertices of those draws (drawViewVertices; an indirect draw counts 3072)
    std::string label, traits;
    std::set<uint64_t> pipes;  // PipelineKey values drawn with this pair (VP_KNOCK_FSR builds a substitute of each)
  };
  struct KnockSigInfo {
    uint64_t encoders = 0, draws = 0;
    uint64_t frames = 0, lastFrame = ~0ull;  // submits that had an encoder of this signature (frames, by submit index)
    double score = 0;  // exclusive ms of the frame's heaviest encoder when it had this signature, summed
    std::string desc;
    std::map<KnockKey, KnockPair> pairs;
  };
  struct KnockCand {
    KnockKey key;
    double draws = 0;  // per encoder of the dominant signature
    std::string label, traits;
    std::vector<uint64_t> pipes;
    double verts = 0;  // view-vertices per frame (submit) of the dominant signature
  };
  struct KnockBlock {
    int32_t cand = 0;
    uint8_t pass = 0;
    bool knock = false, dead = false;
    bool fsr = false;  // VP_KNOCK_FSR pass: `knock` marks the block whose fragment function is the trivial one (else: the baseline)
    uint32_t issued = 0, skipped = 0;  // frames encoded for it, draws left out in them
    uint32_t partial = 0;  // completed frames dropped: a dominant encoder had no usable timestamps (a partial sum is not a sample)
    std::vector<double> excl, vtx, frag;  // one sample per completed frame (dominant encoders summed)
  };
  // What one frame is encoded as: mode 0 nothing, 1 warm-up (count the pairs), 2 measuring (`knock`: leave `vs`/`fs` out
  // of the encoders whose signature is `sig`; `block` collects the samples).
  struct KnockFrame {
    uint8_t mode = 0;
    bool knock = false;
    bool fsr = false;  // measuring: bind the trivial-fragment-function pipelines for `vs`/`fs` instead (draws kept)
    uint64_t sig = 0, vs = 0, fs = 0;
    int64_t block = -1;
  };
  // What a completed frame reports to the knock (block -1 and !warm: nothing).
  struct KnockTag {
    bool warm = false;
    int64_t block = -1;
    uint64_t sig = 0;
    uint32_t expect = 0;  // dominant encoders encoded in the frame: a sample needs all of them timed
    uint32_t gen = 0;  // the run (knockGen) it was encoded in; samples and votes of an earlier run are ignored
  };
  enum class KnockPhase : uint8_t { Idle, Warm, Measure, Drain, Done };
  static constexpr uint32_t kKnockWarmFrames = 300, kKnockTopN = 48, kKnockMaxCands = 96, kKnockBlockFrames = 8;
  // A submit is not a presented frame (Alyx: ~7 submits per frame, the dominant encoder in ~1 of them): submits without the
  // dominant encoder are neutral. kKnockMissAbort: submits since the last dominant one before the block in progress is dropped.
  static constexpr uint32_t kKnockPasses = 2, kKnockMissAbort = 300, kKnockDrainFrames = 700, kKnockTop = 20;
  static constexpr uint32_t kKnockTimeoutHits = 8000;      // Measure phase cap: submits with the dominant encoder since it started
  static constexpr uint32_t kKnockTimeoutFrames = 60000;   // Measure phase hard cap: submits since it started
  static constexpr uint32_t kKnockWarmDraws = 200;       // draws a multisampled render encoder needs to count in warm-up
  static constexpr uint32_t kKnockRelaxFrames = 600;     // armed frames with no counted one before the samples > 1 gate is dropped
  bool knockOn = false;  // submit thread: a knock run is active (armed and not done yet)
  bool knockRelaxed = false;  // submit thread: the warm-up gate is draws >= kKnockWarmDraws only
  bool knockFsrOn = false;  // VP_KNOCK=auto and VP_KNOCK_FSR != 0 (set once at init): pipelines keep a descriptor for the substitute
  dispatch_queue_t knockFsrQueue = nil;  // serial, utility: builds the substitutes (created by the submit thread, drained in ~MetalBackend)
  std::map<std::string, id<MTLFunction>> knockFsrFns;  // trivial fragment functions by Pipeline::fsrKinds (knockFsrQueue only)
  std::mutex knockMutex;  // guards everything below
  KnockPhase knockPhase = KnockPhase::Idle;
  bool knockTimedOut = false, knockNoData = false;  // how the run ended (phase 5 / phase 6 for the host)
  uint32_t knockGen = 0;  // incremented on each arm
  uint32_t knockFrames = 0, knockMissed = 0, knockDrain = 0, knockMeasureFrames = 0, knockArmedFrames = 0;
  uint32_t knockMeasureHits = 0;  // Measure: submits that had the dominant encoder (knockMissed: submits since the last one)
  std::map<uint64_t, KnockSigInfo> knockSigs;
  uint64_t knockSig = 0;  // the dominant signature
  std::vector<KnockCand> knockCands;
  std::vector<KnockBlock> knockBlocks;
  size_t knockStep = 0;     // block being issued: pass = step / (2 * cands), cand = step % (2 * cands) / 2, knock = step & 1
  int64_t knockCur = -1;    // its index in knockBlocks (-1 = not made yet)
  // VP_KNOCK_FSR: the substitute pipeline of each PipelineKey of the candidates (state 0 building, 1 ready, 2 na / failed).
  struct KnockFsr {
    id<MTLRenderPipelineState> pso;
    uint8_t state = 0;
  };
  std::unordered_map<uint64_t, KnockFsr> knockFsr;
  bool knockFsrRun = false;  // this run has the fsr pass (pass index kKnockPasses)
  size_t knockSteps() const { return size_t(kKnockPasses + (knockFsrRun ? 1 : 0)) * knockCands.size() * 2; }
  int knockFsrStatusLocked(const KnockCand& k) const;  // 0 every pipeline ready, 1 some still building, 2 some cannot have one
  id<MTLRenderPipelineState> knockFsrPso(uint64_t pipelineKey);
  void knockFsrBuild(uint64_t pipelineKey, uint32_t gen);
  void knockArm();
  void knockPublishLocked();
  void knockBegin(KnockFrame& f);
  void knockEnd(const KnockFrame& f, bool seen, uint32_t skipped, bool warmSeen, KnockTag& tag);
  void knockMerge(uint64_t sig, uint64_t frame, const std::map<KnockKey, KnockPair>& local);
  void knockVote(const EncDiag& g, double ms, uint32_t gen);
  void knockSample(int64_t block, double excl, double vtx, double frag, bool complete, uint32_t gen);
  void knockPlanLocked();
  void knockReportLocked();
  bool stateElide = true;  // VP_STATE_ELIDE=0: every draw re-emits all encoder state (A/B)
  // VP_BUF_IMMUTABLE=0 (A/B): pipelines do not declare their read-only buffers (argument table, rate map data, draw
  // params) immutable. Part of the pipeline descriptor, hence of descriptorKey and of the binary archives' entries.
  bool bufImmutable = true;
  bool fullBarriers = false;  // VPVK_FULL_BARRIERS=1: every encoder and every compute item waits for all earlier work (hazard vs math bisect)
  double exclWindowS = 5.0;
  std::mutex exclMutex;  // guards everything below
  std::map<uint64_t, ExclType> exclTypes;
  uint64_t exclLastEnd = 0;     // GPU ticks: the latest encoder end seen
  double exclWindowStart = 0;   // host seconds (command buffer GPUEndTime) of the window's first one
  uint64_t exclCbs = 0, exclReports = 0;
  void exclReportLocked(double now);
  // One completed frame: per core encoder its first start / last end ticks (`span`) and, for render encoders, the vertex
  // end / fragment start ticks (`stages`); 0 = not sampled.
  void exclAccount(uint64_t type, const std::vector<EncDiag>& diag, const std::vector<std::pair<uint64_t, uint64_t>>& span,
                   const std::vector<std::pair<uint64_t, uint64_t>>& stages, double hostEnd, double cbMs,
                   const KnockTag& knockTag);

  // Per-pass dump (VP_DUMP_FRAME=<n>, VP_DUMP_DIR=<dir>); -1 = off.
  int64_t dumpFrame = -1;
  bool dumpTaken = false;  // submit thread: one dump per session
  // A dump covers this many consecutive frames (Source 2 splits a frame into several command buffers: the shadow maps,
  // the scene and the post chain each land in a different one), with one byte budget.
  uint32_t dumpSeqLeft = 0;
  uint64_t dumpSeqBytes = 0;
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
  // gpu.missing: which resource a `missing_resources` count of `gpu.err what=submit` was. The backend knows the VPIR
  // (kind, slot) of an argument, not the Vulkan set/binding it came from: `stage` is the StageMask of the table
  // (1 vertex, 2 fragment, 4 compute; 0 = not a table), `res`/`gen` the resource the binding named. One line per
  // distinct (kind, stage, slot, reason), 32 lines per session at most.
  std::mutex missingMutex;
  std::unordered_set<std::string> missingSeen;
  std::atomic<uint32_t> missingLines{0};  // = missingSeen.size(), readable without the lock
  void missing(const char* kind, unsigned stage, unsigned slot, const char* reason, uint32_t res, uint32_t gen) {
    if (missingLines.load(std::memory_order_relaxed) >= 32) return;  // cap reached: no formatting, no lock
    char line[256];
    std::snprintf(line, sizeof line, "gpu.missing kind=%s stage=%u slot=%u reason=%s res=%u gen=%u", kind, stage, slot,
                  reason, res, gen);
    char key[160];
    std::snprintf(key, sizeof key, "%s/%u/%u/%s", kind, stage, slot, reason);
    {
      std::lock_guard lock(missingMutex);
      if (missingSeen.size() >= 32 || !missingSeen.insert(key).second) return;
      missingLines.store(uint32_t(missingSeen.size()), std::memory_order_relaxed);
    }
    info(line);
  }
  static const char* bindKindName(BindKind k) {
    switch (k) {
      case BindKind::Auto: return "auto";
      case BindKind::Buffer: return "buffer";
      case BindKind::ConstantBuffer: return "constant_buffer";
      case BindKind::Texture: return "texture";
      case BindKind::Sampler: return "sampler";
      case BindKind::StorageTexture: return "storage_texture";
      case BindKind::BufferSize: return "buffer_size";
    }
    return "?";
  }

  void init();
  void calibrate();
  void retire(id object);
  id<MTLTexture> texture(TextureId);
  id<MTLTexture> memorylessTexture(const TextureDesc&, std::unordered_map<uint64_t, size_t>& taken, uint64_t frame);
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
  // prewarm (the speculative compile of prewarmShader, queuedAt = when it was queued): returns nil with an EMPTY error when the
  // library is ready or being compiled by another thread (nothing to do), and waits for compileGauge to be quiet before it
  // compiles. Otherwise a nil result always sets `error`.
  id<MTLFunction> function(uint64_t shaderId, const FetchInputs& fetch, MslShader& shader, std::string& error,
                           bool noPointSize = false, bool noFragDepth = false, uint64_t* keyOut = nullptr,
                           bool prewarm = false, std::chrono::steady_clock::time_point queuedAt = {});
  bool buildRender(const PipelineState&, Pipeline&, std::string& error);
  bool buildCompute(const PipelineState&, Pipeline&, std::string& error);
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&);
  // Waits for a free frame slot, takes the next one and releases what it retired (submit thread).
  // Null when the device is lost (nothing more runs on the GPU).
  Slot* acquireSlot(uint64_t frame = UINT64_MAX);
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

// EXPERIMENT (temporary): VPVK_SGSELFTEST=1 runs hand-written MSL kernels with the CTS subgroup patterns on the device and
// logs what the SIMD ops return in divergent code (SGTEST lines).
static void sgSelfTest(id<MTLDevice> dev, id<MTLCommandQueue> q, const std::function<void(const std::string&)>& say) {
  @autoreleasepool {
    static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
inline uint actv() { return as_type<uint2>(simd_vote::vote_t(simd_active_threads_mask())).x; }
inline uint ball() { return as_type<uint2>(simd_vote::vote_t(simd_ballot(true))).x; }
kernel void k_cts(device uint* out [[buffer(0)]], const device uint* d [[buffer(1)]],
                  uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint base = tid * 8u;
  uint mask = ball();
  uint ref = 0u;
  for (uint i = 0u; i < 32u; ++i) if (((mask >> i) & 1u) != 0u) ref += d[i];
  uint s1 = simd_sum(d[lane]);
  out[base + 0] = ref; out[base + 1] = s1; out[base + 2] = mask;
  if ((lane & 1u) == 1u) {
    uint mask2 = ball();
    uint ref2 = 0u;
    for (uint i = 0u; i < 32u; ++i) if (((mask2 >> i) & 1u) != 0u) ref2 += d[i];
    uint s2 = simd_sum(d[lane]);
    out[base + 3] = ref2; out[base + 4] = s2; out[base + 5] = mask2; out[base + 6] = actv();
    uint am = actv(); uint acc = 0u; uint x = d[lane];
    for (uint k = 0u; k < 32u; ++k) { uint v = simd_shuffle(x, ushort(k)); if (((am >> k) & 1u) != 0u) acc += v; }
    out[base + 7] = acc;
  } else {
    out[base + 3] = 0xEEEEu;
  }
}
kernel void k_bf(device uint* out [[buffer(0)]], const device uint* d [[buffer(1)]],
                 uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint base = tid * 8u;
  uint mask = ball();
  uint first = 0u;
  for (uint i = 0u; i < 32u; ++i) { if (((mask >> i) & 1u) != 0u) { first = i; break; } }
  uint a = simd_broadcast_first(d[lane]);
  uint am = actv();
  uint b = simd_shuffle(d[lane], ushort(ctz(am)));
  out[base + 0] = first; out[base + 1] = a; out[base + 2] = b; out[base + 3] = d[first];
  if (first != lane) {
    uint mask2 = ball();
    uint first2 = 0u;
    for (uint i = 0u; i < 32u; ++i) { if (((mask2 >> i) & 1u) != 0u) { first2 = i; break; } }
    uint a2 = simd_broadcast_first(d[lane]);
    uint am2 = actv();
    uint b2 = simd_shuffle(d[lane], ushort(ctz(am2)));
    out[base + 4] = first2; out[base + 5] = a2; out[base + 6] = b2; out[base + 7] = mask2;
  } else {
    out[base + 4] = 0xEEEEu;
  }
}
#define CTSK(NAME, M1, M2) \
kernel void NAME(device uint* out [[buffer(0)]], const device uint* d [[buffer(1)]], \
                 uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  uint base = tid * 8u; \
  uint mask = M1; \
  uint ref = 0u; \
  for (uint i = 0u; i < 32u; ++i) if (((mask >> i) & 1u) != 0u) ref += d[i]; \
  uint s1 = simd_sum(d[lane]); \
  out[base + 0] = ref; out[base + 1] = s1; out[base + 2] = mask; \
  if ((lane & 1u) == 1u) { \
    uint mask2 = M2; \
    uint ref2 = 0u; \
    for (uint i = 0u; i < 32u; ++i) if (((mask2 >> i) & 1u) != 0u) ref2 += d[i]; \
    uint s2 = simd_sum(d[lane]); \
    out[base + 3] = ref2; out[base + 4] = s2; out[base + 5] = mask2; \
  } else { out[base + 3] = 0xEEEEu; } \
}
CTSK(v_a, ball(), ball())
CTSK(v_b, actv(), actv())
CTSK(v_c, (ball() & actv()), (ball() & actv()))
CTSK(v_d, ball(), as_type<uint2>(simd_vote::vote_t(simd_ballot(lane < 32u))).x)
CTSK(v_e, ball(), as_type<uint2>(simd_vote::vote_t(simd_ballot(d[100] != 0u))).x)
CTSK(v_f, ball(), as_type<uint2>(simd_vote::vote_t(simd_ballot((lane & 1u) == 1u))).x)
CTSK(v_g, actv(), as_type<uint2>(simd_vote::vote_t(simd_ballot(d[100] != 0u))).x)
CTSK(v_h, as_type<uint2>(simd_vote::vote_t(simd_ballot(d[100] != 0u))).x, as_type<uint2>(simd_vote::vote_t(simd_ballot(d[100] != 0u))).x)
inline uint orm(uint lane) { return simd_or(1u << lane); }
inline uint orb(uint lane, bool p) { return simd_or(p ? (1u << lane) : 0u); }
CTSK(w_a, orm(lane), orm(lane))
CTSK(w_b, orb(lane, true), orb(lane, true))
CTSK(w_c, orb(lane, d[100] != 0u), orb(lane, d[100] != 0u))
kernel void y_vote(device uint* out [[buffer(0)]], const device uint* d [[buffer(1)]],
                   uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint base = tid * 8u;
  bool P = (d[lane] & 1u) != 0u;
  out[base + 0] = simd_all(P) ? 1u : 0u;
  out[base + 1] = simd_any(!P) ? 1u : 0u;
  out[base + 2] = simd_prefix_exclusive_sum(d[lane]);
  out[base + 3] = simd_min(d[lane]);
  if ((lane & 1u) == 1u) {
    out[base + 4] = simd_all(P) ? 1u : 0u;
    out[base + 5] = simd_any(!P) ? 1u : 0u;
    out[base + 6] = simd_prefix_exclusive_sum(d[lane]);
    out[base + 7] = simd_min(d[lane]);
  } else { out[base + 4] = 0xEEEEu; }
}
)MSL";
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&e];
    if (!lib) { say(std::string("SGTEST compile_failed ") + (e ? e.localizedDescription.UTF8String : "?")); return; }
    uint32_t data[128];
    for (uint32_t i = 0; i < 128; ++i) data[i] = (i + 1) * (i + 1) + 3;
    struct Shape { uint32_t x, y, z; };
    const Shape shapes[] = {{32, 1, 1}, {8, 1, 1}};
    for (const char* kn : {"w_a", "w_b", "w_c", "y_vote"}) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(kn)];
      id<MTLComputePipelineState> ps = fn ? [dev newComputePipelineStateWithFunction:fn error:&e] : nil;
      if (!ps) { say(std::string("SGTEST pipeline_failed ") + kn); continue; }
      for (const Shape& sh : shapes) {
        const uint32_t n = sh.x * sh.y * sh.z;
        id<MTLBuffer> out = [dev newBufferWithLength:n * 8 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> dd = [dev newBufferWithBytes:data length:sizeof data options:MTLResourceStorageModeShared];
        memset(out.contents, 0xAB, n * 8 * 4);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBuffer:dd offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(sh.x, sh.y, sh.z)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t* o = static_cast<const uint32_t*>(out.contents);
        for (uint32_t t : {0u, 1u, 2u, 3u, 31u}) {
          if (t >= n) continue;
          char buf[256];
          std::snprintf(buf, sizeof buf, "SGTEST %s shape=%ux%ux%u tid=%u: %x %x %x | %x %x %x %x %x", kn, sh.x, sh.y, sh.z, t, o[t * 8 + 0],
                        o[t * 8 + 1], o[t * 8 + 2], o[t * 8 + 3], o[t * 8 + 4], o[t * 8 + 5], o[t * 8 + 6], o[t * 8 + 7]);
          say(buf);
        }
      }
    }
  }
}

void MetalBackend::Impl::init() {
  device = MTLCreateSystemDefaultDevice();
  if (!device) {
    err("what=no_device");
    return;
  }
  queue = [device newCommandQueueWithMaxCommandBufferCount:256];  // room for the draw-budget cuts
  queue.label = @"vpgpu";
  if (const char* st = std::getenv("VPVK_SGSELFTEST"); st && *st == '1') sgSelfTest(device, queue, [this](const std::string& l) { info(l); });  // EXPERIMENT
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
  slotDone.reset(new std::atomic<bool>[n]);
  for (unsigned i = 0; i < n; ++i) slotDone[i].store(true);
  for (Slot& s : slots) {
    s.args = [device newBufferWithLength:kArgRingBytes options:kSharedUntracked];
    s.args.label = @"vpgpu.args";
    attr::track("m2.args", s.args);
    if (s.args) argsRingBytes += kArgRingBytes;
    rangeAdd(s.args.contents, s.args.length);
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
  if (const char* e = std::getenv("VPVK_HANG_SPLIT")) hangSplit = uint32_t(std::strtoul(e, nullptr, 10));
  if (const char* e = std::getenv("VPVK_MAX_DRAWS_PER_CB")) maxDrawsPerCb = uint32_t(std::strtoul(e, nullptr, 10));
  if (const char* e = std::getenv("VPVK_MAX_ENCODERS_PER_CB")) maxEncodersPerCb = uint32_t(std::strtoul(e, nullptr, 10));
  if (const char* e = std::getenv("VPVK_MAX_PENDING_CHUNKS")) maxPendingChunks = uint32_t(std::strtoul(e, nullptr, 10));
  if (const char* e = std::getenv("VPVK_MSAA_RELEASE_FRAMES")) msaaReleaseFrames = uint32_t(std::strtoul(e, nullptr, 10));
  if (const char* e = std::getenv("VPVK_HANG_DIR"); e && *e) hangDir = e;
  else if (const char* k = std::getenv("KL_CAPTURE_DIR"); k && *k) hangDir = k;
  else if (const char* h = std::getenv("HOME"); h && *h) hangDir = std::string(h) + "/Documents/Registros";
  while (dumpDir.size() > 1 && dumpDir.back() == '/') dumpDir.pop_back();
  if (dumpFrame >= 0 && dumpDir.empty()) dumpFrame = -1;
  if (dumpFrame >= 0) info("gpu.dump armed frame=" + std::to_string(dumpFrame) + " dir=" + token(dumpDir));
  // Off unless asked (avp-port's own ordering): 1 = join fence, 2 = tracked.
  if (const char* o = std::getenv("VP_CB_ORDER"); o && (*o == '1' || *o == '2')) cbOrder = true, cbTracked = *o == '2';
  if (cbOrder) info("gpu.order armed cb_order=" + std::to_string(cbTracked ? 2 : 1) + " slots=" + std::to_string(n));
  if (const char* x = std::getenv("VP_EXCL_TIMING"); x && *x && *x != '0') exclOn = true;
  if (const char* x = std::getenv("VP_KNOCK"); x && std::strcmp(x, "auto") == 0) {
    gKnockEnv.store(true);  // idle until the host calls vpgpu_metal2_knock_arm
    exclOn = true;  // the knock reads the per-encoder stage timestamps
    knockFsrOn = true;  // VP_KNOCK_FSR=0 turns the trivial-fragment-function pass off
    if (const char* f = std::getenv("VP_KNOCK_FSR"); f && *f == '0') knockFsrOn = false;
    info("gpu.knock armed (idle until the host arms it) warm_frames=" + std::to_string(kKnockWarmFrames) + " top_n=" + std::to_string(kKnockTopN) +
         " block_frames=" + std::to_string(kKnockBlockFrames) + " passes=" + std::to_string(kKnockPasses) +
         " fsr_passes=" + std::to_string(knockFsrOn ? 1 : 0));
  }
  if (const char* x = std::getenv("VP_EXCL_SECONDS"); x && *x) exclWindowS = std::max(1.0, std::strtod(x, nullptr));
  if (const char* x = std::getenv("VP_STATE_ELIDE"); x && *x == '0') stateElide = false;
  info(std::string("gpu.state elide on=") + (stateElide ? "1" : "0"));
  if (const char* x = std::getenv("VP_BUF_IMMUTABLE"); x && *x == '0') bufImmutable = false;
  info(std::string("gpu.state buf_immutable on=") + (bufImmutable ? "1" : "0"));
  if (const char* x = std::getenv("VPVK_FULL_BARRIERS"); x && *x && *x != '0') fullBarriers = true;
  info(std::string("gpu.state full_barriers on=") + (fullBarriers ? "1" : "0"));
  // Vertex amplification (multiview vertex shaders translated with VPVK_AMPLIFY=1 declare [[amplification_id]]).
  caps.vertexAmplification = [device supportsVertexAmplificationCount:2];
  if (exclOn || cbOrder)
    info("gpu.excl armed on=" + std::to_string(exclOn ? 1 : 0) + " window_s=" + std::to_string(int(exclWindowS)) +
         " stage_timestamps=" + std::to_string(stageSamples ? 1 : 0) + " vertex_amplification2=" +
         std::to_string(caps.vertexAmplification ? 1 : 0));

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

namespace {
int64_t steadyNs();  // defined with the stall detection below
}

void MetalBackend::Impl::retire(id object) {
  if (!object) return;
  uint64_t bytes = 0;
  if ([object conformsToProtocol:@protocol(MTLResource)]) bytes = ((id<MTLResource>)object).allocatedSize;
  graveyard.push_back(object);
  graveyardBytes += bytes;
  bump(cnt.heldObjs);
  bump(cnt.heldBytes, bytes);
}

// Hold of a destroyed pinned texture before retire(): at least this long and 4 completed frames (or an idle GPU).
constexpr int64_t kPinnedHoldNs = 1'000'000'000;

// Moves the pinned textures whose hold is over into the graveyard. Serialized like retire().
void MetalBackend::Impl::promotePinned() {
  if (pinnedHold.empty()) return;
  const int64_t now = steadyNs();
  uint64_t done;
  {
    std::lock_guard lock(doneMutex);
    done = finishedBelow;
  }
  const bool idle = lastSlot < 0 || slotSettled(size_t(lastSlot));
  size_t keep = 0;
  for (PinnedHold& h : pinnedHold) {
    if (now - h.ns >= kPinnedHoldNs && (idle || done >= h.frames + 4)) retire(h.object);
    else pinnedHold[keep++] = h;
  }
  pinnedHold.resize(keep);
}

// Gives back the retired objects that no command buffer can still read. Never waits, and does not need an idle
// device: a slot whose frame completed (slotDone: its publish ran, after every command buffer of the frame) keeps
// nothing a running frame uses, so its retired list goes while the other frames run. The graveyard was retired after
// the last acquired frame was submitted, so it goes once that frame completed. Same caller rules as submit/destroy
// (serialized: it touches the graveyard, the slots and the residency set); try-lock only, a busy backend means "not now".
// The objects are released after the residency commit, so a big texture's teardown does not hold up the submit thread.
size_t MetalBackend::Impl::drainIdle(std::string& report) {
  if (!device || lostFlag.load() || slots.empty()) return 0;
  promotePinned();
  bool any = !graveyard.empty();
  for (const Slot& sl : slots) any = any || !sl.retired.empty();
  if (!any) return 0;
  if (!drainMutex.try_lock()) return 0;  // waitIdle/destructor is holding slots
  const int64_t now = steadyNs();
  std::vector<id> doomed;
  uint64_t bytes = 0;
  auto takeList = [&](std::vector<id>& list, uint64_t& listBytes) {
    for (id obj : list)
      if (residency && [obj conformsToProtocol:@protocol(MTLAllocation)]) {
        [residency removeAllocation:obj];
        bump(cnt.resRemove);
        residencyDirty = true;
      }
    doomed.insert(doomed.end(), list.begin(), list.end());
    bytes += listBytes;
    list.clear();
    listBytes = 0;
  };
  for (size_t i = 0; i < slots.size(); ++i)
    if (!slots[i].retired.empty() && slotSettled(i))
      takeList(slots[i].retired, slots[i].retiredBytes);
  if (!graveyard.empty() && (lastSlot < 0 || slotSettled(size_t(lastSlot))))
    takeList(graveyard, graveyardBytes);
  if (residencyDirty && residency) {
    [residency commit];
    bump(cnt.resCommit);
    residencyDirty = false;
  }
  drainMutex.unlock();
  if (doomed.empty()) return 0;
  const size_t n = doomed.size();
  lastDrainNs.store(now, std::memory_order_relaxed);
  bump(cnt.drainRuns);
  bump(cnt.drainedObjs, n);
  bump(cnt.drainedBytes, bytes);
  cnt.heldObjs.fetch_sub(std::min<uint64_t>(cnt.heldObjs.load(), n), std::memory_order_relaxed);
  cnt.heldBytes.fetch_sub(std::min<uint64_t>(cnt.heldBytes.load(), bytes), std::memory_order_relaxed);
  char b[160];
  std::snprintf(b, sizeof b, "retired_released=%zu retired_mib=%.1f", n, double(bytes) / 1048576.0);
  if (!report.empty()) report += ' ';
  report += b;
  doomed.clear();  // ARC releases here, after the residency commit
  return size_t(bytes);
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
  // A 3D texture cannot be seen as 2D (Vulkan EXT_image_2d_view_of_3d): Metal aborts inside newTextureView
  // ("source texture MTLTextureType3D not compatible with texture view MTLTextureType2D").
  if (base.textureType == MTLTextureType3D && type != MTLTextureType3D) {
    v.failed = true;
    static std::atomic<int> said{0};
    if (said.fetch_add(1) < 8) err("what=texture_view_unsupported reason=3d_as_2d base=%u", v.base.index);
    return nil;
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
  bump(cnt.texViews);
  attr::trackCount("m2.view", v.texture);
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
  bump(cnt.texNew);
  if (tex.texture) bump(cnt.texNewBytes, tex.texture.allocatedSize);
  if (!tex.texture) {
    tex.failed = true;
    err("what=texture index=%u format=%s size=%ux%u label=%s", t.index, formatInfo(tex.desc.format).name,
        tex.desc.width, tex.desc.height, token(tex.desc.label).c_str());
    return nil;
  }
  if (!tex.desc.label.empty()) tex.texture.label = @(tex.desc.label.c_str());
  texLive(tex.texture, true);
  {
    const FormatInfo& fi = formatInfo(tex.desc.format);
    tex.cat = tex.desc.samples > 1                  ? kTexMsaa
              : fi.depth() || fi.stencil()          ? kTexDepth
              : std::strncmp(fi.name, "Astc", 4) == 0 ? kTexAstc
                                                    : kTexColor;
    tex.bytes = tex.texture.allocatedSize;
    texBytes[tex.cat] += tex.bytes;
    static const char* const kTexTag[kTexCats] = {"m2.tex.astc", "m2.tex.color", "m2.tex.depth", "m2.tex.msaa"};
    attr::track(tex.deferredRT ? (tex.cat == kTexMsaa ? "m2.tex.msaa.mat" : "m2.tex.rt.mat") : kTexTag[tex.cat],
                tex.texture);  // deferred targets that got memory after all (released by releaseMsaaBackings)
    if (tex.deferredRT) {
      // A deferred MSAA target needs memory after all (a frame that cannot stay on tile). Transient: see
      // releaseMsaaBackings.
      msaaDeferredBytes -= std::min<uint64_t>(msaaDeferredBytes.load(), tex.msaaLogical);
      tex.streak = 0;
      backedMsaa.insert(t.index);
      ++msaaMaterializations;
      char line[512];
      std::snprintf(line, sizeof line,
                    "gpu.metal.info what=msaa_materialized index=%u %ux%ux%u samples=%u fmt=%s mb=%.1f why=%s load=%u store=%u label=%s",
                    t.index, tex.desc.width, tex.desc.height, unsigned(tex.desc.layers), unsigned(tex.desc.samples),
                    formatInfo(tex.desc.format).name, double(tex.bytes) / 1048576.0, matReason ? matReason : "other_use",
                    matLoad, matStore, token(tex.desc.label).c_str());
      info(line);
      tex.msaaLogical = 0;
    }
  }
  if (residency) {
    [residency addAllocation:tex.texture];
    bump(cnt.resAdd);
    residencyDirty = true;
  }
  return tex.texture;
}

// Where the texture memory is, by why it exists, plus the largest live textures by name and shape: the line a memory
// jump or a jetsam is read with. `resident` = persistent textures (memory from creation), `msaa_mat` = deferred MSAA
// targets a frame forced into memory and still holding it (released after msaaReleaseFrames tile-local frames),
// `rt_mat` = other deferred render targets that got memory, `memoryless` = deferred targets without memory now (bytes
// they would take). Submit thread only (walks `textures`).
std::string MetalBackend::Impl::memLedger() {
  uint64_t resident = 0, msaaMat = 0, rtMat = 0, memorylessB = 0;
  uint32_t residentN = 0, msaaMatN = 0, rtMatN = 0, memorylessN = 0;
  struct Big { uint64_t bytes; uint32_t index; const Texture* t; };
  Big top[8] = {};
  for (const auto& [index, t] : textures) {
    if (!t.texture) {
      if (t.msaaLogical) memorylessB += t.msaaLogical, ++memorylessN;
      continue;
    }
    if (t.deferredRT) msaaMat += t.bytes, ++msaaMatN;
    else if (!t.desc.persistent) rtMat += t.bytes, ++rtMatN;
    else resident += t.bytes, ++residentN;
    if (t.bytes > top[7].bytes) {
      int k = 7;
      while (k > 0 && top[k - 1].bytes < t.bytes) top[k] = top[k - 1], --k;
      top[k] = {t.bytes, index, &t};
    }
  }
  char b[256];
  std::snprintf(b, sizeof b,
                "resident=%.1f/%u msaa_mat=%.1f/%u rt_mat=%.1f/%u memoryless=%.1f/%u msaa_materialized=%llu "
                "msaa_released=%llu",
                double(resident) / 1048576.0, residentN, double(msaaMat) / 1048576.0, msaaMatN, double(rtMat) / 1048576.0,
                rtMatN, double(memorylessB) / 1048576.0, memorylessN, (unsigned long long)msaaMaterializations.load(),
                (unsigned long long)msaaReleases.load());
  std::string s = b;
  size_t poolN = 0;
  uint64_t poolB = 0;
  for (const auto& entry : memoryless)
    for (id<MTLTexture> t : entry.second.textures) ++poolN, poolB += t.allocatedSize;
  std::snprintf(b, sizeof b, " ml_pool=%zu/%.1f", poolN, double(poolB) / 1048576.0);
  s += b;
  std::snprintf(b, sizeof b, " held_b=%llu views=%llu args_b=%llu buf_int_b=%llu fov_b=%llu",
                (unsigned long long)cnt.heldBytes.load(std::memory_order_relaxed),
                (unsigned long long)liveViews.load(std::memory_order_relaxed),
                (unsigned long long)(argsRingBytes.load(std::memory_order_relaxed) + argsMoreBytes.load(std::memory_order_relaxed)),
                (unsigned long long)bufIntLiveBytes.load(std::memory_order_relaxed),
                (unsigned long long)fovShadowBytes.load(std::memory_order_relaxed));
  s += b;
  s += " top=";
  for (int k = 0; k < 8 && top[k].t; ++k) {
    const TextureDesc& d = top[k].t->desc;
    std::snprintf(b, sizeof b, "%s%u:%ux%ux%u/s%u/%s/%s/%.1f", k ? "," : "", top[k].index, d.width, d.height,
                  unsigned(d.layers), unsigned(d.samples), formatInfo(d.format).name,
                  top[k].t->deferredRT ? "msaa_mat" : !d.persistent ? "rt_mat" : top[k].t->pinned ? "pinned" : "resident",
                  double(top[k].bytes) / 1048576.0);
    s += b;
  }
  return s;
}

// gpu.tex counters: `t` was created (add) or is given back to retire() (!add). Same texture, same bucket both times.
void MetalBackend::Impl::texLive(id<MTLTexture> t, bool add) {
  const TexFmt f = texFmtOf(t.pixelFormat);
  const bool ml = t.storageMode == MTLStorageModeMemoryless;
  std::atomic<uint64_t>& n = ml ? texFmtMlN[f] : texFmtN[f];
  const uint64_t bytes = ml ? 0 : uint64_t(t.allocatedSize);
  if (add) {
    n.fetch_add(1, std::memory_order_relaxed);
    texFmtBytes[f].fetch_add(bytes, std::memory_order_relaxed);
  } else {
    n.fetch_sub(1, std::memory_order_relaxed);
    texFmtBytes[f].fetch_sub(bytes, std::memory_order_relaxed);
  }
}

// gpu.tex live=<textures with memory> mb=<their allocatedSize> ml=<memoryless textures> by: <format>=<n>/<mb>,... [mlby: <format>=<n>,...]
void MetalBackend::Impl::texLine() {
  uint64_t n = 0, bytes = 0, ml = 0;
  std::string by, mlBy;
  char b[96];
  for (int f = 0; f < kFmtCount; ++f) {
    const uint64_t fn = texFmtN[f].load(std::memory_order_relaxed), fb = texFmtBytes[f].load(std::memory_order_relaxed),
                   fm = texFmtMlN[f].load(std::memory_order_relaxed);
    n += fn, bytes += fb, ml += fm;
    if (fn) {
      std::snprintf(b, sizeof b, "%s%s=%llu/%.1f", by.empty() ? "" : ",", kTexFmtName[f], (unsigned long long)fn,
                    double(fb) / 1048576.0);
      by += b;
    }
    if (fm) {
      std::snprintf(b, sizeof b, "%s%s=%llu", mlBy.empty() ? "" : ",", kTexFmtName[f], (unsigned long long)fm);
      mlBy += b;
    }
  }
  std::snprintf(b, sizeof b, "gpu.tex live=%llu mb=%.1f ml=%llu by: ", (unsigned long long)n, double(bytes) / 1048576.0,
                (unsigned long long)ml);
  info(std::string(b) + by + (mlBy.empty() ? "" : " mlby: " + mlBy));
}

// End of a frame: deferred MSAA textures that were served by the memoryless stand-in for msaaReleaseFrames
// consecutive frames (and not used this frame) give their memory back; the next frame that cannot stay on tile
// allocates again. Nothing is lost: a tile-local verdict means contents are neither loaded, stored nor sampled.
void MetalBackend::Impl::releaseMsaaBackings(uint64_t frameIndex) {
  {  // gpu.tex: one line every 60 s (the first 60 s after the first frame), whatever follows
    const int64_t now = steadyNs();
    if (!texLineNs) texLineNs = now;
    else if (now - texLineNs >= 60'000'000'000) texLineNs = now, texLine();
  }
  // The memory ledger for the heartbeat (any thread reads ledgerLast): every kLedgerFrames frames, and at once after a
  // frame that materialised or released MSAA memory, so a jump is attributed by the next heartbeat.
  {
    const uint64_t events = msaaMaterializations.load() + msaaReleases.load();
    if (frameIndex >= ledgerFrame + kLedgerFrames || events != ledgerEvents) {
      std::string l = memLedger();
      std::lock_guard lock(ledgerMutex);
      ledgerLast = std::move(l);
      ledgerFrame = frameIndex;
      ledgerEvents = events;
    }
  }
  const bool trim = trimLevel.load(std::memory_order_relaxed) != 0;  // memory warning: any unused backing goes
  if (!msaaReleaseFrames && !trim) return;
  for (auto bi = backedMsaa.begin(); bi != backedMsaa.end();) {
    auto it = textures.find(*bi);
    if (it == textures.end() || !it->second.texture) {
      bi = backedMsaa.erase(bi);
      continue;
    }
    Texture& tex = it->second;
    if (tex.pinned || tex.streak < (trim ? 1u : msaaReleaseFrames) || tex.usedFrame == frameIndex) {
      ++bi;
      continue;
    }
    id<MTLTexture> t = tex.texture;
    for (auto f = formatViews.begin(); f != formatViews.end();) {
      if (f->first.base == t) {
        if (f->second != t) retire(f->second);
        bump(cnt.fmtViewDrop);
        f = formatViews.erase(f);
      } else {
        ++f;
      }
    }
    for (auto& [vi, v] : views)  // explicit views keep the base alive
      if (v.base.index == *bi && v.texture) {
        retire(v.texture);
        v.texture = nil;
        v.of = nil;
      }
    const double mb = double(tex.bytes) / 1048576.0;
    tex.msaaLogical = logicalBytes(tex.desc);
    msaaDeferredBytes += tex.msaaLogical;
    texBytes[tex.cat] -= tex.bytes;
    tex.bytes = 0;
    texLive(tex.texture, false);
    retire(tex.texture);  // in-flight frames may still hold it: graveyard frees it when their slot is reused
    tex.texture = nil;
    tex.streak = 0;
    char line[320];
    std::snprintf(line, sizeof line, "gpu.metal.info what=msaa_released index=%u %ux%ux%u samples=%u mb=%.1f after_tile_frames=%u label=%s",
                  *bi, tex.desc.width, tex.desc.height, unsigned(tex.desc.layers), unsigned(tex.desc.samples), mb,
                  msaaReleaseFrames, token(tex.desc.label).c_str());
    info(line);
    trimmedMsaaBytes += uint64_t(mb * 1048576.0);
    ++msaaReleases;
    bi = backedMsaa.erase(bi);
  }
  if (trim && !trimSlotsLeft.load()) trimLevel.store(0);  // slots done too: the request is complete
}

// Estimated cost of a render encoder in "draw units" (a draw on a plain 1x two-layer 3840x2485 target = 1):
//   draws x clamp(pixels x samples x layers / ref, 1, 8)  +  2 per MB of attachment traffic (loads, stores that
//   leave the tile). Upper-bound model, uncalibrated: gpu.slowcb prints est_cost next to the measured ms so the
//   weights can be fitted (goal: no command buffer past ~100 ms).
double MetalBackend::Impl::encoderCost(const Encoder& enc, const std::vector<Command>& cmds) {
  uint64_t draws = 0;
  for (uint32_t ci : enc.commands)
    if (const auto* rp = std::get_if<RenderPass>(&cmds[ci])) draws += rp->draws.size();
  double px = 0, mb = 0;
  auto visit = [&](const Attachment& a, int slot) {
    if (!a.texture) return;
    auto it = textures.find(a.texture.index);
    if (it == textures.end() || it->second.generation != a.texture.generation) return;
    const TextureDesc& d = it->second.desc;
    const double layers = enc.view >= 0 ? 1 : std::max<uint32_t>(1, d.layers);  // a view-split encoder draws one layer
    px = std::max(px, double(d.width) * d.height * std::max<uint32_t>(1, d.samples) * layers);
    const double bytes = double(d.width) * d.height * std::max<uint32_t>(1, d.samples) * layers * formatInfo(d.format).bytesPerBlock;
    const bool stores = a.store == StoreAction::Store || a.store == StoreAction::StoreAndResolve;
    if (a.load == LoadAction::Load || (stores && !enc.tileLocal(slot))) mb += bytes / 1048576.0;
  };
  for (int i = 0; i < kMaxColorAttachments; ++i) visit(enc.pass.colors[i], i);
  visit(enc.pass.depth, kDepthSlot);
  constexpr double kRefPx = 3840.0 * 2485.0 * 2.0;
  return double(draws) * std::clamp(px / kRefPx, 1.0, 8.0) + 2.0 * mb;
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
  if (v != t) attr::trackCount("m2.view.fmt", v);
  formatViews[key] = v;
  bump(cnt.fmtViewNew);
  return v;
}

id<MTLTexture> MetalBackend::Impl::memorylessTexture(const TextureDesc& d, std::unordered_map<uint64_t, size_t>& taken,
                                                      uint64_t frame) {
  const uint64_t key = uint64_t(d.format) | uint64_t(d.samples) << 8 | uint64_t(d.layers) << 16 |
                       uint64_t(d.width) << 32 | uint64_t(d.height) << 48;
  MemorylessPool& mp = memoryless[key];
  mp.lastFrame = frame;
  auto& pool = mp.textures;
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
    texLive(t, true);
    attr::track("m2.ml_pool", t);
    pool.push_back(t);
    // Shapes of earlier configurations (dynamic resolution, re-created targets) are given back once unused for a while.
    if (memoryless.size() > 16)
      for (auto it = memoryless.begin(); it != memoryless.end();) {
        if (it->first != key && frame > it->second.lastFrame + 600) {
          for (id<MTLTexture> old : it->second.textures) texLive(old, false), retire(old);
          it = memoryless.erase(it);
        } else {
          ++it;
        }
      }
  }
  return pool[n++];
}

id<MTLFunction> MetalBackend::Impl::function(uint64_t shaderId, const FetchInputs& fetch, MslShader& shader,
                                             std::string& error, bool noPointSize, bool noFragDepth,
                                             uint64_t* keyOut, bool prewarm,
                                             std::chrono::steady_clock::time_point queuedAt) {
  const auto requestedAt = prewarm ? queuedAt : std::chrono::steady_clock::now();  // wait_ms of gpu.shader.lib counts from here
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
  if (keyOut) *keyOut = key;
  if (prewarm) {
    // Speculative: nothing to do when a thread already has or is making it, and it must not compete with the compiles a
    // draw waits for. The key is not claimed while it waits, so a draw that needs it meanwhile compiles it at once.
    {
      std::lock_guard lock(pipelineMutex);
      if (functions.count(key) || compiling.count(key)) {
        error.clear();
        return nil;
      }
    }
    const auto yieldUntil = std::chrono::steady_clock::now() + std::chrono::milliseconds(kPrewarmYieldMs);
    while (compileGauge.active() && !closing && std::chrono::steady_clock::now() < yieldUntil)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (closing) {  // teardown: do not start a speculative compile
      error.clear();
      return nil;
    }
  }
  {
    std::unique_lock lock(pipelineMutex);
    for (;;) {
      auto it = functions.find(key);
      if (it != functions.end()) {
        if (prewarm) {  // a worker got there while this job waited
          error.clear();
          return nil;
        }
        std::lock_guard sl(libStatsMutex);
        ++libDedup;
        return it->second;
      }
      if (!compiling.count(key)) break;
      if (prewarm) {
        error.clear();
        return nil;
      }
      functionReady.wait(lock);  // the same MSL is compiling on another thread
    }
    compiling.insert(key);
  }
  // A compile a draw may be waiting for keeps the prewarm and the archive harvest away until it ends.
  const ActivityGauge::Scope demandCompile(compileGauge, !prewarm);
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
  const double waitMs = std::chrono::duration<double, std::milli>(t0 - requestedAt).count();
  // Cache key note: `key` (function label, hence the binary-archive key of every pipeline using it) is the hash of
  // the MSL text plus the variant macros below; it does NOT include the translator version, so a translator change
  // that leaves a shader's MSL identical keeps its archived pipelines. If mathMode / languageVersion / a new macro
  // ever changes the compiled code for the same text, mix a constant into `key` above at that moment.
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.mathMode = MTLMathModeFast;  // the emitter marks precise math with precise:: explicitly
  o.languageVersion = MTLLanguageVersion3_2;
  // [[invariant]] positions (SPIR-V Invariant) only stay invariant with this set; no effect on the rest,
  // so the key above needs no constant (only MSL text that declares [[position, invariant]] is affected).
  if (shader.source.find("invariant]]") != std::string::npos) o.preserveInvariance = YES;
  NSMutableDictionary<NSString*, NSObject*>* macros = [NSMutableDictionary dictionary];
  if (noPointSize) macros[@"VP_PSZ"] = @"user(vp_psz)";
  if (noFragDepth) macros[@"VP_NO_FDEPTH"] = @"1";
  if (macros.count) o.preprocessorMacros = macros;
  NSError* e = nil;
  int retried = 0;
  const attr::Probe probe(device);
  id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
    vpgpu::HitchSpan hitch(vpgpu::hitchPerf().libCompileUs, false);  // vk.hitch lib_compile_ms
    return [device newLibraryWithSource:@(shader.source.c_str()) options:o error:pe];
  }, &retried);
  if (retried) err("what=compiler_unreachable site=msl retries=%d ok=%d", retried, lib ? 1 : 0);
  if (!lib) {
    error = "msl compile: " + str(e);
    return nil;
  }
  attr::trackObject("m2.lib.shader", lib, probe);
  id<MTLFunction> fn = [lib newFunctionWithName:@(shader.entry.c_str())];
  if (!fn) {
    error = "entry " + shader.entry + " not in library";
    return nil;
  }
  attr::trackCount("m2.fn.shader", fn);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  // The archive key of a descriptor identifies its functions by this label, not by object address:
  // a function dropped with its last pipeline and compiled again is the same variant.
  fn.label = [NSString stringWithFormat:@"%016llx", (unsigned long long)key];
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
      char b[320];
      // bytes / wait_ms / src describe the library this line was written for: its MSL size, the time from the request
      // (the prewarm job's queueing, or the call for a draw) to the start of its compile, and who asked.
      std::snprintf(b, sizeof b,
                    "gpu.shader.lib compiled=%u dedup=%u prewarmed=%u ms_total=%.0f ms_max=%.0f last_ms=%.0f bytes=%zu wait_ms=%.0f src=%s",
                    libCompiled, libDedup, libPrewarmed, libMsTotal, libMsMax, ms, shader.source.size(), waitMs,
                    prewarm ? "prewarm" : "demand");
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
  ScopedPool pool;
  return m->function(shaderId, {}, shader, error, true) != nil;
}

void MetalBackend::drainShaderPrewarm() {
  Impl* m = impl_.get();
  dispatch_queue_t qs[2] = {nil, nil};
  {
    std::lock_guard lock(m->pipelineMutex);
    qs[0] = m->prewarmQueues[0];
    qs[1] = m->prewarmQueues[1];
  }
  for (dispatch_queue_t q : qs)
    if (q) dispatch_sync(q, ^{});  // serial: returns once everything queued before it ran
}

void MetalBackend::prewarmShader(uint64_t shaderId) {
  Impl* m = impl_.get();
  if (!ok() || !shaderId || m->closing) return;
  ScopedPool pool;
  {
    std::lock_guard lock(m->pipelineMutex);
    if (!m->prewarmQueues[0]) {
      dispatch_queue_attr_t a = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0);
      m->prewarmQueues[0] = dispatch_queue_create("vpgpu.shader.prewarm0", a);
      m->prewarmQueues[1] = dispatch_queue_create("vpgpu.shader.prewarm1", a);
    }
  }
  dispatch_queue_t q = m->prewarmQueues[m->prewarmNext++ & 1];
  const auto queuedAt = std::chrono::steady_clock::now();
  dispatch_async(q, ^{
    if (m->closing) return;
    MslShader shader;
    std::string error;
    // Most likely variant: no fetch inputs; vertex shaders without a PointSize write (non-point topology).
    // function() returns nil with no error when a worker got there first (ready or compiling): nothing to do.
    if (m->function(shaderId, {}, shader, error, true, false, nullptr, true, queuedAt)) {
      std::lock_guard sl(m->libStatsMutex);
      ++m->libPrewarmed;
    } else if (!error.empty()) {
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
  const attr::Probe probe(device);  // covers the library and the pipeline (the delta goes to the pipeline's tag)
  id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
    vpgpu::HitchSpan hitch(vpgpu::hitchPerf().libCompileUs, false);  // vk.hitch lib_compile_ms
    return [device newLibraryWithSource:@(src.c_str()) options:o error:pe];
  });
  attr::trackCount("m2.lib.clear", lib);
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
    attr::trackObject("m2.pso.clear", p, probe);
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
    const attr::Probe probe(device);
    blitLibrary = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
      return [device newLibraryWithSource:@(kBlitMsl) options:o error:pe];
    });
    attr::trackObject("m2.lib.blit", blitLibrary, probe);
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
  const attr::Probe probe(device);
  id<MTLRenderPipelineState> p =
      d.vertexFunction && d.fragmentFunction
          ? compileRetrying(&e, [&](NSError* __autoreleasing* pe) { return [device newRenderPipelineStateWithDescriptor:d error:pe]; })
          : nil;
  attr::trackObject("m2.pso.blit", p, probe);
  if (!p) error = "pipeline: " + str(e);
  // A failure is cached too (logged once, not retried every frame) — unless the compiler was
  // only unreachable.
  if (p || !compilerUnreachable(e)) blitPipelines[key] = p;
  return p;
}

// ---- Foveation: fold / expand between the logical attachments and the physical shadows ----
// One fullscreen triangle per view (instance = layer). Destination pixel -> source coordinates through the rate map
// data at [[buffer(0)]]: expand writes the logical attachment reading the physical shadow
// (map_screen_to_physical_coordinates), fold writes the physical shadow reading the logical attachment
// (map_physical_to_screen_coordinates). Colour is filtered bilinearly (sRGB formats blend in linear space as the
// hardware does); depth is read at the texel the coordinate falls in.
constexpr const char* kFovMsl = R"(#include <metal_stdlib>
using namespace metal;
struct VpFovV {
    float4 pos [[position]];
    uint layer [[render_target_array_index]];
};
vertex VpFovV vp_fov_vs(uint vid [[vertex_id]], uint inst [[instance_id]]) {
    VpFovV o;
    o.pos = float4(vid == 1u ? 3.0f : -1.0f, vid == 2u ? 3.0f : -1.0f, 0.0f, 1.0f);
    o.layer = inst;
    return o;
}
struct VpFovF {
    float4 pos [[position]];
};
fragment float4 vp_fov_expand_c(VpFovF in [[stage_in]], uint layer [[render_target_array_index]],
                                texture2d_array<float> src [[texture(0)]],
                                constant rasterization_rate_map_data& rr [[buffer(0)]]) {
    rasterization_rate_map_decoder map(rr);
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    const float2 phys = map.map_screen_to_physical_coordinates(in.pos.xy, layer);
    return src.sample(s, phys / float2(float(src.get_width()), float(src.get_height())), layer, level(0.0f));
}
fragment float4 vp_fov_fold_c(VpFovF in [[stage_in]], uint layer [[render_target_array_index]],
                              texture2d_array<float> src [[texture(0)]],
                              constant rasterization_rate_map_data& rr [[buffer(0)]]) {
    rasterization_rate_map_decoder map(rr);
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    const float2 scr = map.map_physical_to_screen_coordinates(in.pos.xy, layer);
    return src.sample(s, scr / float2(float(src.get_width()), float(src.get_height())), layer, level(0.0f));
}
struct VpFovD {
    float depth [[depth(any)]];
};
fragment VpFovD vp_fov_expand_d(VpFovF in [[stage_in]], uint layer [[render_target_array_index]],
                                depth2d_array<float> src [[texture(0)]],
                                constant rasterization_rate_map_data& rr [[buffer(0)]]) {
    rasterization_rate_map_decoder map(rr);
    const float2 phys = map.map_screen_to_physical_coordinates(in.pos.xy, layer);
    const float2 hi = float2(float(src.get_width()) - 1.0f, float(src.get_height()) - 1.0f);
    VpFovD o;
    o.depth = src.read(uint2(clamp(floor(phys), float2(0.0f), hi)), layer);
    return o;
}
fragment VpFovD vp_fov_fold_d(VpFovF in [[stage_in]], uint layer [[render_target_array_index]],
                              depth2d_array<float> src [[texture(0)]],
                              constant rasterization_rate_map_data& rr [[buffer(0)]]) {
    rasterization_rate_map_decoder map(rr);
    const float2 scr = map.map_physical_to_screen_coordinates(in.pos.xy, layer);
    const float2 hi = float2(float(src.get_width()) - 1.0f, float(src.get_height()) - 1.0f);
    VpFovD o;
    o.depth = src.read(uint2(clamp(floor(scr), float2(0.0f), hi)), layer);
    return o;
}
)";

id<MTLRenderPipelineState> MetalBackend::Impl::fovPipeline(int kind, MTLPixelFormat pf, std::string& error) {
  const uint64_t key = uint64_t(kind) << 32 | uint64_t(pf);
  if (auto it = fovPipelines.find(key); it != fovPipelines.end()) {
    if (!it->second) error = "pipeline_failed";
    return it->second;
  }
  if (!fovLibrary) {
    const double now = steadySeconds();
    if (fovLibraryFailed || now < fovLibraryRetryAt) {
      error = "library_failed";
      return nil;
    }
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion3_2;
    NSError* e = nil;
    const attr::Probe probe(device);
    fovLibrary = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
      return [device newLibraryWithSource:@(kFovMsl) options:o error:pe];
    });
    attr::trackObject("m2.lib.fov", fovLibrary, probe);
    if (!fovLibrary) {
      if (compilerUnreachable(e)) fovLibraryRetryAt = now + 2.0;
      else fovLibraryFailed = true;
      error = "library: " + str(e);
      return nil;
    }
  }
  static NSString* const fs[4] = {@"vp_fov_expand_c", @"vp_fov_fold_c", @"vp_fov_expand_d", @"vp_fov_fold_d"};
  MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
  d.label = @"vpgpu.fov";
  d.vertexFunction = [fovLibrary newFunctionWithName:@"vp_fov_vs"];
  d.fragmentFunction = [fovLibrary newFunctionWithName:fs[kind & 3]];
  if (kind >= 2) d.depthAttachmentPixelFormat = pf;
  else d.colorAttachments[0].pixelFormat = pf;
  d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
  NSError* e = nil;
  const attr::Probe probe(device);
  id<MTLRenderPipelineState> p =
      d.vertexFunction && d.fragmentFunction
          ? compileRetrying(&e, [&](NSError* __autoreleasing* pe) { return [device newRenderPipelineStateWithDescriptor:d error:pe]; })
          : nil;
  attr::trackObject("m2.pso.fov", p, probe);
  if (!p) error = "pipeline: " + str(e);
  if (p || !compilerUnreachable(e)) fovPipelines[key] = p;
  return p;
}

// ---- Forced MSAA: seed the samples of an autoMsaa attachment from its 1x partner ----
// The source is bound as a 2D-array view (one mip, the pass's layers), so one shader serves 2D and array images; the layer is the
// render target array index. The pipeline declares every attachment of the encoder (Metal checks them all); only colour 0 and
// depth are written.
id<MTLRenderPipelineState> MetalBackend::Impl::seedPipeline(MTLRenderPassDescriptor* rp, bool layered, bool color, bool depth,
                                                           std::string& error) {
  uint64_t key = 1469598103934665603ull;
  auto mix = [&](uint64_t v) { key = (key ^ v) * 1099511628211ull; };
  mix(uint64_t(layered) | uint64_t(color) << 1 | uint64_t(depth) << 2);
  NSUInteger samples = 1;
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    id<MTLTexture> t = rp.colorAttachments[i].texture;
    mix(t ? uint64_t(t.pixelFormat) | uint64_t(t.sampleCount) << 32 : 0);
    if (t) samples = t.sampleCount;
  }
  id<MTLTexture> dt = rp.depthAttachment.texture ? rp.depthAttachment.texture : rp.stencilAttachment.texture;
  mix(dt ? uint64_t(dt.pixelFormat) | uint64_t(dt.sampleCount) << 32 : 0);
  if (dt) samples = dt.sampleCount;
  if (auto it = seedPipelines.find(key); it != seedPipelines.end()) {
    if (!it->second) error = "pipeline_failed";
    return it->second;
  }
  std::string src =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "struct VpSeedV { float4 pos [[position]]; };\n"
      "struct VpSeedVL { float4 pos [[position]]; uint layer [[render_target_array_index]]; };\n";
  src += layered ? "vertex VpSeedVL vp_seed_vs(uint vid [[vertex_id]], uint inst [[instance_id]]) {\n  VpSeedVL o;\n  o.layer = inst;\n"
                 : "vertex VpSeedV vp_seed_vs(uint vid [[vertex_id]]) {\n  VpSeedV o;\n";
  src += "  o.pos = float4(vid == 1u ? 3.0f : -1.0f, vid == 2u ? 3.0f : -1.0f, 0.0f, 1.0f);\n  return o;\n}\n";
  src += "struct VpSeedF {";
  if (color) src += " float4 c [[color(0)]];";
  if (depth) src += " float d [[depth(any)]];";
  src += " };\nfragment VpSeedF vp_seed_fs(float4 pos [[position]]";
  if (layered) src += ", uint layer [[render_target_array_index]]";
  if (color) src += ", texture2d_array<float> sc [[texture(0)]]";
  if (depth) src += ", depth2d_array<float> sd [[texture(1)]]";
  src += ") {\n  const uint2 p = uint2(pos.xy);\n";
  src += layered ? "  const uint l = layer;\n" : "  const uint l = 0u;\n";
  src += "  VpSeedF o;\n";
  if (color) src += "  o.c = sc.read(p, l);\n";
  if (depth) src += "  o.d = sd.read(p, l);\n";
  src += "  return o;\n}\n";
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.languageVersion = MTLLanguageVersion3_2;
  NSError* e = nil;
  const attr::Probe probe(device);
  id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
    return [device newLibraryWithSource:@(src.c_str()) options:o error:pe];
  });
  attr::trackObject("m2.lib.seed", lib, probe);
  if (!lib) {
    error = "library: " + str(e);
    if (!compilerUnreachable(e)) seedPipelines[key] = nil;
    return nil;
  }
  MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
  d.label = @"vpgpu.seed";
  d.vertexFunction = [lib newFunctionWithName:@"vp_seed_vs"];
  d.fragmentFunction = [lib newFunctionWithName:@"vp_seed_fs"];
  for (int i = 0; i < kMaxColorAttachments; ++i)
    if (id<MTLTexture> t = rp.colorAttachments[i].texture) {
      d.colorAttachments[i].pixelFormat = t.pixelFormat;
      if (i != 0 || !color) d.colorAttachments[i].writeMask = MTLColorWriteMaskNone;
    }
  if (rp.depthAttachment.texture) d.depthAttachmentPixelFormat = rp.depthAttachment.texture.pixelFormat;
  if (rp.stencilAttachment.texture) d.stencilAttachmentPixelFormat = rp.stencilAttachment.texture.pixelFormat;
  d.rasterSampleCount = samples;
  d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
  const attr::Probe probe2(device);
  id<MTLRenderPipelineState> p =
      d.vertexFunction && d.fragmentFunction
          ? compileRetrying(&e, [&](NSError* __autoreleasing* pe) { return [device newRenderPipelineStateWithDescriptor:d error:pe]; })
          : nil;
  attr::trackObject("m2.pso.seed", p, probe2);
  if (!p) error = "pipeline: " + str(e);
  if (p || !compilerUnreachable(e)) seedPipelines[key] = p;
  if (p) {
    char line[160];
    std::snprintf(line, sizeof line, "gpu.seed pipeline samples=%lu layered=%d colour=%d depth=%d", (unsigned long)samples, int(layered),
                  int(color), int(depth));
    info(line);
  }
  return p;
}

MetalBackend::Impl::FovMap* MetalBackend::Impl::fovMapFor(const RateMap& rm) {
  if (auto it = fovMaps.find(rm.key); it != fovMaps.end()) return it->second.map ? &it->second : nullptr;
  FovMap& fm = fovMaps[rm.key];  // inserted now: a failure is cached (map stays nil)
  if (![device supportsRasterizationRateMapWithLayerCount:rm.layers]) {
    err("what=fov_rate_map layers=%u error=unsupported", unsigned(rm.layers));
    return nullptr;
  }
  NSMutableArray<MTLRasterizationRateLayerDescriptor*>* layerDescs = [NSMutableArray array];
  for (uint16_t l = 0; l < rm.layers; ++l)
    [layerDescs addObject:[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(rm.zonesX, rm.zonesY, 1)
                                                                                horizontal:&rm.qx[size_t(l) * rm.zonesX]
                                                                                  vertical:&rm.qy[size_t(l) * rm.zonesY]]];
  // Unretained view of layerDescs (which keeps the descriptors alive for the whole call).
  std::vector<__unsafe_unretained MTLRasterizationRateLayerDescriptor*> layerPtrs;
  for (MTLRasterizationRateLayerDescriptor* ld in layerDescs) layerPtrs.push_back(ld);
  MTLRasterizationRateMapDescriptor* d =
      [MTLRasterizationRateMapDescriptor rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(rm.width, rm.height, 0)
                                                                           layerCount:layerDescs.count
                                                                               layers:layerPtrs.data()];
  d.label = @(rm.label.c_str());
  id<MTLRasterizationRateMap> map = [device newRasterizationRateMapWithDescriptor:d];
  if (!map) {
    err("what=fov_rate_map size=%ux%u layers=%u error=create_failed", rm.width, rm.height, unsigned(rm.layers));
    return nullptr;
  }
  NSUInteger pw = 0, ph = 0;
  for (uint16_t l = 0; l < rm.layers; ++l) {
    const MTLSize ps = [map physicalSizeForLayer:l];
    pw = std::max<NSUInteger>(pw, ps.width), ph = std::max<NSUInteger>(ph, ps.height);
  }
  const MTLSizeAndAlign sa = map.parameterBufferSizeAndAlign;
  id<MTLBuffer> data = [device newBufferWithLength:std::max<NSUInteger>(sa.size, 16) options:kSharedUntracked];
  if (!data || !pw || !ph) {
    err("what=fov_rate_map size=%ux%u error=no_parameter_buffer", rm.width, rm.height);
    return nullptr;
  }
  data.label = @"vpgpu.fov.map";
  attr::track("m2.fov.map", data);
  if (residency) {
    [residency addAllocation:data];
    bump(cnt.resAdd);
    residencyDirty = true;
  }
  [map copyParameterDataToBuffer:data offset:0];
  fm.map = map, fm.data = data, fm.physW = pw, fm.physH = ph;
  {
    char line[320];
    std::snprintf(line, sizeof line,
                  "gpu.fov map key=%016llx logical=%ux%u physical=%lux%lu fragments=%.1f%% layers=%u zones=%ux%u est=%.1f%% param_bytes=%lu",
                  (unsigned long long)rm.key, rm.width, rm.height, (unsigned long)pw, (unsigned long)ph,
                  100.0 * double(pw) * double(ph) / (double(rm.width) * double(rm.height)), unsigned(rm.layers),
                  unsigned(rm.zonesX), unsigned(rm.zonesY), double(rm.keptFraction) * 100.0, (unsigned long)sa.size);
    info(line);
  }
  return &fm;
}

// A rate map that maps nothing (quality 1, one zone): the data of the draws of pipelines that read FragCoord through
// vp_rrm outside a foveated pass. Its screen size only has to cover the largest render target.
id<MTLBuffer> MetalBackend::Impl::fovIdentityData() {
  if (fovIdentity.data || fovIdentityTried) return fovIdentity.data;
  fovIdentityTried = true;
  if (![device supportsRasterizationRateMapWithLayerCount:2]) return nil;
  const float q = 1.0f;
  NSMutableArray<MTLRasterizationRateLayerDescriptor*>* layerDescs = [NSMutableArray array];
  for (NSUInteger l = 0; l < 2; ++l)
    [layerDescs addObject:[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(1, 1, 1)
                                                                                horizontal:&q
                                                                                  vertical:&q]];
  std::vector<__unsafe_unretained MTLRasterizationRateLayerDescriptor*> layerPtrs;
  for (MTLRasterizationRateLayerDescriptor* ld in layerDescs) layerPtrs.push_back(ld);
  MTLRasterizationRateMapDescriptor* d =
      [MTLRasterizationRateMapDescriptor rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(16384, 16384, 0)
                                                                           layerCount:layerDescs.count
                                                                               layers:layerPtrs.data()];
  d.label = @"vpgpu.fov.identity";
  id<MTLRasterizationRateMap> map = [device newRasterizationRateMapWithDescriptor:d];
  if (!map) {
    err("what=fov_identity error=create_failed");
    return nil;
  }
  const MTLSizeAndAlign sa = map.parameterBufferSizeAndAlign;
  id<MTLBuffer> data = [device newBufferWithLength:std::max<NSUInteger>(sa.size, 16) options:kSharedUntracked];
  if (!data) return nil;
  data.label = @"vpgpu.fov.identity";
  attr::track("m2.fov.identity", data);
  if (residency) {
    [residency addAllocation:data];
    bump(cnt.resAdd);
    residencyDirty = true;
  }
  [map copyParameterDataToBuffer:data offset:0];
  fovIdentity.map = map, fovIdentity.data = data;
  return data;
}

// A physical shadow attachment of `layers` layers: pooled by shape, handed out in order per encoder (two
// attachments of one shape get two textures; contents never outlive an encoder, so encoders and frames share them,
// ordered by the fences every foveated encoder waits on).
id<MTLTexture> MetalBackend::Impl::fovShadow(MTLPixelFormat pf, NSUInteger w, NSUInteger h, NSUInteger layers, bool depth,
                                             std::unordered_map<uint64_t, size_t>& taken, uint64_t frame) {
  const uint64_t key = uint64_t(pf) | uint64_t(w) << 16 | uint64_t(h) << 36 | uint64_t(layers) << 56 | (depth ? 1ull << 63 : 0);
  FovShadowPool& pool = fovShadows[key];
  pool.lastFrame = frame;
  size_t& n = taken[key];
  if (n == pool.textures.size()) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:w height:h mipmapped:NO];
    td.textureType = MTLTextureType2DArray;
    td.arrayLength = layers;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    td.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    id<MTLTexture> t = [device newTextureWithDescriptor:td];
    if (!t) {
      err("what=fov_shadow format=%lu size=%lux%lux%lu", (unsigned long)pf, (unsigned long)w, (unsigned long)h, (unsigned long)layers);
      return nil;
    }
    t.label = depth ? @"vpgpu.fov.depth" : @"vpgpu.fov.color";
    texLive(t, true);
    attr::track(depth ? "m2.fov.shadow.depth" : "m2.fov.shadow.color", t);
    pool.cat = depth ? kTexDepth : kTexColor;
    pool.bytes += t.allocatedSize;
    fovShadowBytes += t.allocatedSize;
    texBytes[pool.cat] += t.allocatedSize;
    if (residency) {
      [residency addAllocation:t];
      bump(cnt.resAdd);
      residencyDirty = true;
    }
    pool.textures.push_back(t);
    // Sizes of earlier configurations (a resolution change) are given back once unused for a while.
    if (fovShadows.size() > 8)
      for (auto it = fovShadows.begin(); it != fovShadows.end();) {
        if (it->first != key && frame > it->second.lastFrame + 600) {
          for (id<MTLTexture> old : it->second.textures) texLive(old, false), retire(old);
          texBytes[it->second.cat] -= std::min<uint64_t>(texBytes[it->second.cat].load(), it->second.bytes);
          fovShadowBytes -= std::min<uint64_t>(fovShadowBytes.load(), it->second.bytes);
          it = fovShadows.erase(it);
        } else {
          ++it;
        }
      }
  }
  return pool.textures[n++];
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
  const attr::Probe probe(device);
  id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
    return [device newLibraryWithSource:@(kD24Msl) options:o error:pe];
  });
  attr::trackObject("m2.lib.d24", lib, probe);
  for (int i = 0; i < 2 && lib; ++i) {
    id<MTLFunction> f = [lib newFunctionWithName:i ? @"vp_float_to_d24" : @"vp_d24_to_float"];
    d24Pipelines[i] = f ? [device newComputePipelineStateWithFunction:f error:&e] : nil;
    attr::trackObject("m2.pso.d24", d24Pipelines[i], probe);
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

namespace {
int64_t steadyNs();
// A command buffer refused while the app is in the background: MTLCommandBufferErrorNotPermitted (7), whose IOGPU text is
// "(00000006:kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted)". Nothing ran and nothing hung.
bool isBackgroundError(NSError* e) {
  if (!e) return false;
  return e.code == 7 || str(e).find("BackgroundExecution") != std::string::npos;
}
// What a shader does that matters when the GPU hangs (gpu.hang lines), read off the MSL text:
// ssbo_r / ssbo_rw = it declares a read-only / writable storage buffer (declared, not proven
// written on this path), robust = bounds-checked accesses, loop = the emitter's hang guard,
// int16 = the translator lowered 16-bit integers (its `// vp:int16` marker), atomic = atomics.
std::string shaderTraits(const MslShader& sh, const char* stg) {
  std::string t;
  const std::string& src = sh.source;
  const bool ssbo = src.find(" device uchar*") != std::string::npos;
  const bool ssboWrite = src.find("    device uchar*") != std::string::npos;
  if (ssbo) t += std::string(" ") + stg + (ssboWrite ? ":ssbo_rw" : ":ssbo_r");
  if (src.find("ul <= vp_args.bs") != std::string::npos) t += std::string(" ") + stg + ":robust";
  if (src.find("vp_guard") != std::string::npos) t += std::string(" ") + stg + ":loop";
  if (src.find("\n// vp:int16\n") != std::string::npos) t += std::string(" ") + stg + ":int16";
  if (src.find("atomic_") != std::string::npos) t += std::string(" ") + stg + ":atomic";
  return t;
}
}  // namespace

void MetalBackend::Impl::dumpInFlight(const char* why, bool early) {
  // Copied under the lock, logged outside it (the log sink may be slow; completion handlers
  // take the same lock).
  std::map<uint64_t, InFlight> copy;
  std::map<uint64_t, std::shared_ptr<CbTimes>> live;
  std::deque<std::string> slow;
  {
    std::lock_guard g(inFlightMutex);
    copy = inFlightCbs;
    live = liveCbs;
    slow = slowCbRecent;
  }
  const int64_t now = steadyNs();
  err("what=hang_report reason=%s in_flight=%zu", why, copy.size());
  info("gpu.hang mem " + memStats());
  // Per command buffer committed and not completed: scheduled=no means the GPU never started it (queue
  // behind another, or the driver holds it); scheduled=yes with a big run_ms means it runs slowly.
  for (const auto& [seq, t] : live) {
    const int64_t cs = t->commitStartNs.load(), ce = t->commitEndNs.load(), sc = t->scheduledNs.load();
    char b[256];
    std::snprintf(b, sizeof b, "gpu.hang cb_live f=%llu label=%s age_ms=%lld commit_ms=%.1f scheduled=%s sched_ms=%.1f",
                  (unsigned long long)t->frame, t->label.c_str(), (long long)((now - cs) / 1000000),
                  ce >= cs && ce ? double(ce - cs) / 1e6 : -1.0, sc ? "yes" : "no", sc ? double(sc - cs) / 1e6 : -1.0);
    info(b);
  }
  for (const std::string& l : slow) info("gpu.hang cb_slow " + l);
  for (const auto& [f, c] : copy) {
    info("gpu.hang cb=" + c.label + " age_ms=" + std::to_string((now - c.committedNs) / 1000000) +
         " draws=" + std::to_string(c.draws) + " dispatches=" + std::to_string(c.dispatches) +
         " pipelines=" + std::to_string(c.pipelines.size() + c.dropped));
    for (const std::string& pl : c.pipelines) info("gpu.hang   " + pl);
    if (c.dropped) info("gpu.hang   (+" + std::to_string(c.dropped) + " more)");
    dumpHungDraws(c, early);
  }
}

std::string MetalBackend::Impl::memStats() {
  double footprint = -1, compressed = -1;
  task_vm_info_data_t vm{};
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &cnt) == KERN_SUCCESS) {
    footprint = double(vm.phys_footprint) / 1048576.0;
    compressed = double(vm.compressed) / 1048576.0;
  }
  double osAvail = -1;
#if __has_include(<os/proc.h>) && (TARGET_OS_VISION || TARGET_OS_IOS)
  if (__builtin_available(iOS 13.0, visionOS 1.0, *)) osAvail = double(os_proc_available_memory()) / 1048576.0;
#endif
  char b[256];
  std::snprintf(b, sizeof b,
                "device_alloc_mb=%.0f recommended_max_mb=%.0f os_available_mb=%.0f footprint_mb=%.0f compressed_mb=%.0f",
                double(device.currentAllocatedSize) / 1048576.0, double(device.recommendedMaxWorkingSetSize) / 1048576.0,
                osAvail, footprint, compressed);
  return b;
}

// 3 s and 8 s without completions (the loss comes at kStallNs = 20 s): memory and the in-flight dump
// early, once per (completion count, level) across the three wait loops. No behaviour change.
void MetalBackend::Impl::stallWarn(uint64_t completionsNow, int64_t quietNs, const char* where) {
  const uint64_t level = quietNs > 8'000'000'000 ? 2 : quietNs > 3'000'000'000 ? 1 : 0;
  if (!level) return;
  const uint64_t key = (completionsNow << 2) | level;
  uint64_t prev = stallWarned.load();
  while (prev < key)
    if (stallWarned.compare_exchange_weak(prev, key)) {
      char b[160];
      std::snprintf(b, sizeof b, "gpu.stall_warn level=%llus where=%s quiet_ms=%lld completions=%llu ",
                    (unsigned long long)(level == 1 ? 3 : 8), where, (long long)(quietNs / 1000000),
                    (unsigned long long)completionsNow);
      info(std::string(b) + memStats());
      dumpInFlight(level == 1 ? "stall_warn_3s" : "stall_warn_8s", true);
      return;
    }
}

std::shared_ptr<MetalBackend::Impl::CbTimes> MetalBackend::Impl::trackCb(id<MTLCommandBuffer> c, uint64_t frame) {
  auto t = std::make_shared<CbTimes>();
  t->label = token(c.label ? std::string(c.label.UTF8String) : std::string("?"));
  t->frame = frame;
  t->seq = ++cbSeq;
  {
    std::lock_guard g(inFlightMutex);
    liveCbs[t->seq] = t;
  }
  // By-value copies: handlers run after submit returned. They must be added before the publishing
  // handler (it frees the backend), see slowLog.
  Impl* const hSelf = this;
  const std::shared_ptr<CbTimes> hT = t;
  [c addScheduledHandler:^(id<MTLCommandBuffer>) {
    hT->scheduledNs.store(steadyNs());
  }];
  [c addCompletedHandler:^(id<MTLCommandBuffer> finished) {
    const int64_t done = steadyNs();
    const int64_t cs = hT->commitStartNs.load(), ce = hT->commitEndNs.load(), sc = hT->scheduledNs.load();
    const double commitMs = ce >= cs && ce ? double(ce - cs) / 1e6 : 0.0;
    const double schedMs = sc ? double(sc - cs) / 1e6 : -1.0;  // commit -> scheduled
    const double runMs = double(done - (sc ? sc : cs)) / 1e6;  // scheduled -> completed (wall)
    const double gpuMs = (finished.GPUEndTime - finished.GPUStartTime) * 1000.0;
    if (finished.GPUStartTime > 0 && gpuMs >= 0) vpgpu::gpuPerf().addCb(uint64_t(gpuMs * 1000.0));  // vk.perf: GPU time per command buffer
    {
      std::lock_guard g(hSelf->inFlightMutex);
      hSelf->liveCbs.erase(hT->seq);
    }
    if (std::max({commitMs, schedMs, runMs}) <= kSlowCbTimeMs || hSelf->cbTimeLines.fetch_add(1) >= kCbTimeMaxLines) return;
    char line[256];
    std::snprintf(line, sizeof line, "f=%llu label=%s commit_ms=%.1f sched_ms=%.1f run_ms=%.1f gpu_ms=%.1f",
                  (unsigned long long)hT->frame, hT->label.c_str(), commitMs, schedMs, runMs, gpuMs);
    hSelf->info(std::string("gpu.cbtime ") + line);
    std::lock_guard g(hSelf->inFlightMutex);
    hSelf->slowCbRecent.push_back(line);
    if (hSelf->slowCbRecent.size() > 16) hSelf->slowCbRecent.pop_front();
  }];
  return t;
}

void MetalBackend::Impl::timedCommit(id<MTLCommandBuffer> c, const std::shared_ptr<CbTimes>& t) {
  vpgpu::backgroundGate().waitForeground();  // parks while the app is in the background (visionOS fails the commit)
  t->commitStartNs.store(steadyNs());
  [c commit];
  t->commitEndNs.store(steadyNs());
}

// The draws of the chunk the GPU is stuck in (the first not completed; without a split, the whole
// command buffer, up to 256 lines), and the MSL of their shaders written to hangDir once per run.
void MetalBackend::Impl::dumpHungDraws(const InFlight& c, bool early) {
  const uint32_t done = c.chunksDone ? c.chunksDone->load() : 0;
  info("gpu.hang   chunks=" + std::to_string(c.chunks) + " chunks_done=" + std::to_string(done) +
       " recorded_draws=" + std::to_string(c.recs.size()) + (c.chunks ? "" : " (not split)"));
  static const char* kKinds[] = {"draw", "indexed", "indirect", "indexed_indirect"};
  std::vector<uint64_t> ids;
  size_t lines = 0;
  for (const DrawRec& r : c.recs) {
    if (c.chunks && r.chunk != done) continue;
    if (lines++ >= 256) break;
    char b[512];
    std::snprintf(b, sizeof b,
                  "gpu.hang   draw e%u.c%u %s count=%u inst=%u first=%u base_vertex=%d rt=%ux%u layers=%u samples=%u "
                  "vp=%.0f,%.0f,%.0fx%.0f sc=%u,%u,%ux%u blend=%02x depth=%c%c vs=%016llx fs=%016llx",
                  r.encoder, r.chunk, kKinds[r.kind & 3], r.count, r.instances, r.first, r.baseVertex, r.rtWidth,
                  r.rtHeight, unsigned(r.layers), unsigned(r.samples), double(r.vpX), double(r.vpY), double(r.vpW),
                  double(r.vpH), r.scX, r.scY, r.scW, r.scH, unsigned(r.blend), r.depthTest ? 't' : '-',
                  r.depthWrite ? 'w' : '-', (unsigned long long)r.vsId, (unsigned long long)r.fsId);
    info(b);
    for (uint64_t id : {r.vsId, r.fsId})
      if (id && std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
  }
  if (early || hangDir.empty() || ids.empty() || hangDumped.exchange(true)) return;  // stall warnings leave the MSL dump to the real report
  std::error_code ec;
  std::filesystem::create_directories(hangDir, ec);
  size_t written = 0;
  for (uint64_t id : ids) {
    if (written >= 32) break;
    MslShader sh;
    if (!shaders.find(id, sh)) continue;
    char name[64];
    std::snprintf(name, sizeof name, "/gpu-hang-%016llx.metal", (unsigned long long)id);
    std::ofstream f(hangDir + name, std::ios::binary | std::ios::trunc);
    f << sh.source;
    if (f) ++written;
  }
  info("gpu.hang   msl_dumped=" + std::to_string(written) + " dir=" + token(hangDir));
}

void MetalBackend::Impl::noteBackgroundError() {
  const uint64_t n = bgErrors.fetch_add(1) + 1;
  const int64_t now = steadyNs();
  int64_t last = bgLoggedNs.load();
  if (now - last >= 1'000'000'000 && bgLoggedNs.compare_exchange_strong(last, now))
    info("gpu.cb.background count=" + std::to_string(n));
}

void MetalBackend::Impl::hangEncoders(uint64_t frame, NSError* error,
                                      const std::vector<std::pair<NSUInteger, NSUInteger>>& encSamples,
                                      id<MTLCounterSampleBuffer> samples, NSUInteger sampleCount) {
  const std::string what = str(error);
  if (what.find("Ignored") != std::string::npos || what.find("InnocentVictim") != std::string::npos) return;  // ran nothing
  if (isBackgroundError(error)) return;  // refused in the background (noteBackgroundError): ran nothing either
  if (hangEncReports.fetch_add(1) >= 8) return;
  const size_t n = encSamples.size();
  // Metal's verdict per core encoder (labels start their digits with the core index, see faultedEncoders), strongest wins.
  std::vector<int> rank(n, 0);  // 0 completed / not named, 1 pending, 2 affected, 3 faulted
  NSArray* infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
  for (id<MTLCommandBufferEncoderInfo> i in infos) {
    const char* dig = i.label.UTF8String ?: "";
    while (*dig && (*dig < '0' || *dig > '9')) ++dig;
    if (!*dig) continue;
    const unsigned long e = std::strtoul(dig, nullptr, 10);
    const int r = i.errorState == MTLCommandEncoderErrorStateFaulted    ? 3
                  : i.errorState == MTLCommandEncoderErrorStateAffected ? 2
                  : i.errorState == MTLCommandEncoderErrorStatePending  ? 1
                                                                        : 0;
    if (e < n) rank[e] = std::max(rank[e], r);
  }
  // GPU ticks of each encoder's first start / last end sample (0 = not written: a killed encoder never writes its end).
  std::vector<std::pair<uint64_t, uint64_t>> raw(n, {0, 0});
  if (samples && sampleCount) {
    NSData* data = [samples resolveCounterRange:NSMakeRange(0, sampleCount)];
    const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
    auto tick = [&](NSUInteger k) -> uint64_t {
      if (!ts || k >= sampleCount) return 0;  // NSNotFound / MTLCounterDontSample: not sampled
      const uint64_t t = ts[k].timestamp;
      return t == MTLCounterErrorValue ? 0 : t;
    };
    for (size_t i = 0; i < n; ++i) raw[i] = {tick(encSamples[i].first), tick(encSamples[i].second)};
  }
  uint64_t base = 0;
  for (const auto& r : raw)
    if (r.first && (!base || r.first < base)) base = r.first;
  auto at = [&](uint64_t t) -> std::string {
    if (!t) return "never";
    if (!base || t < base) return "?";
    char b[32];
    std::snprintf(b, sizeof b, "%.3f", double(t - base) * gpuTickNs / 1e6);
    return b;
  };
  static const char* kMetal[] = {"completed", "pending", "affected", "faulted"};
  std::string stuck;
  size_t lines = 0;
  for (size_t e = 0; e < n; ++e) {
    const bool sampled = encSamples[e].first != NSNotFound;
    const auto [t0, t1] = raw[e];
    // Metal's faulted verdict is the proof; the timestamps are a hint (a pass without draws writes no end sample).
    const char* why = rank[e] == 3 ? "metal_faulted"
                      : !sampled   ? nullptr
                      : (t0 && !t1) ? "time_started_no_end"
                      : (t0 && t1 >= t0 && double(t1 - t0) * gpuTickNs / 1e6 > 500.0) ? "time_over_500ms"
                                                                                       : nullptr;
    if (why) stuck += (stuck.empty() ? "" : ",") + std::to_string(e) + ":" + why;
    if (lines >= 64) continue;
    ++lines;
    char dur[32] = "-";
    if (t0 && t1 >= t0) std::snprintf(dur, sizeof dur, "%.3f", double(t1 - t0) * gpuTickNs / 1e6);
    char b[256];
    std::snprintf(b, sizeof b, "gpu.hang.enc f=%llu i=%zu metal=%s start_ms=%s end_ms=%s dur_ms=%s stuck=%s",
                  (unsigned long long)frame, e, kMetal[rank[e]], sampled ? at(t0).c_str() : "unsampled",
                  sampled ? at(t1).c_str() : "unsampled", dur, why ? why : "0");
    info(b);
  }
  info("gpu.hang.enc f=" + std::to_string(frame) + " error=" + token(what) + " encoders=" + std::to_string(n) +
       (n > lines ? " (truncated)" : "") + " stuck=" + (stuck.empty() ? std::string("-") : stuck) +
       " (the pipelines of encoder i are the 'e<i>:' lines of gpu.hang cb=)");
}

namespace {
uint64_t descriptorKey(id desc, bool compute);  // defined with the archive code below
}

void MetalBackend::Impl::notePso(bool archived, double ms, const PipelineState& s, const char* kind) {
  const std::string& label = s.label;
  const uint32_t a = archived ? ++psoArchived : psoArchived.load();
  const uint32_t h = archived ? psoHot.load() : ++psoHot;
  double total, worst;
  {
    std::lock_guard lock(psoHotMutex);
    if (!archived) psoHotMs += ms, psoHotMaxMs = std::max(psoHotMaxMs, ms);
    total = psoHotMs, worst = psoHotMaxMs;
  }
  if (!archived && h <= 32)
    info("gpu.pso.hot label=" + token(label) + " ms=" + std::to_string(int(ms + 0.5)));
  // The first 20 pipelines of the session the binary archives did not have (a full compile): which ones (the label
  // changes between runs, the key does not), for the next prewarm.
  if (!archived && h <= 20)
    info("gpu.pso.miss n=" + std::to_string(h) + " kind=" + kind + " label=" + token(label) +
         " key=" + std::to_string(s.key().value) + " ms=" + std::to_string(int(ms)));
  if ((a + h) <= 8 || (a + h) % 64 == 0 || (!archived && h <= 32)) {
    char b[160];
    std::snprintf(b, sizeof b, "gpu.pso.aot archived=%u hot=%u hot_ms_total=%.0f hot_ms_max=%.0f", a, h, total, worst);
    info(b);
  }
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
  id<MTLFunction> vfn = function(s.vertexShader, {}, vs, error, s.primitive != Primitive::Points, false, &p.fnKeys[0]);
  if (!vfn) return false;
  d.vertexFunction = vfn;
  if (vs.source.find("[[amplification_id]]") != std::string::npos) {
    if (!caps.vertexAmplification) {
      error = "vertex amplification x2 not supported by this GPU (VPVK_AMPLIFY=1)";
      return false;
    }
    d.maxVertexAmplificationCount = 2;
    p.amp = 2;
  }
  if (s.fragmentShader) {  // none = depth-only
    // Vulkan ignores FragDepth writes without a depth attachment; Metal rejects the pipeline.
    id<MTLFunction> ffn = function(s.fragmentShader, s.fetch, fs, error, false, !formatInfo(s.depthFormat).depth(), &p.fnKeys[1]);
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
    if (c.blendingEnabled) p.blendMask |= uint8_t(1u << i);
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
  {  // Metal: dual-source blending with more than one render target is a validation abort, not an error.
    bool dual = false;
    int targets = 0;
    for (int i = 0; i < kMaxColorAttachments; ++i) {
      if (s.colorFormats[i] != Format::Undefined) ++targets;
      if (!d.colorAttachments[i].blendingEnabled) continue;
      for (BlendFactor f : {s.blend[i].srcColor, s.blend[i].dstColor, s.blend[i].srcAlpha, s.blend[i].dstAlpha})
        dual = dual || f == BlendFactor::Src1Color || f == BlendFactor::OneMinusSrc1Color ||
               f == BlendFactor::Src1Alpha || f == BlendFactor::OneMinusSrc1Alpha;
    }
    if (dual && targets > 1) {
      error = "dual-source blending with " + std::to_string(targets) + " colour attachments (Metal allows one)";
      return false;
    }
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
  if (bufImmutable) {
    // Buffers a shader only reads (`constant` arguments): each stage's argument table (its pointers lead to the
    // buffers the shader writes; the table itself is rewritten only ahead of the draws that use it, in the frame's
    // own ring), the rate map data of FragCoord (filled once, buffer 28) and the draw params (setVertexBytes, 29).
    if (vs.argumentBufferIndex < 31) d.vertexBuffers[vs.argumentBufferIndex].mutability = MTLMutabilityImmutable;
    if (vs.source.find("constant VpDrawParams& vp_draw") != std::string::npos)  // vpmsl kDrawParamsDecl
      d.vertexBuffers[29].mutability = MTLMutabilityImmutable;
    if (s.fragmentShader) {
      if (fs.argumentBufferIndex < 31) d.fragmentBuffers[fs.argumentBufferIndex].mutability = MTLMutabilityImmutable;
      if (fs.source.find("constant rasterization_rate_map_data& vp_rrm") != std::string::npos)  // vpmsl kRateMapDecl
        d.fragmentBuffers[28].mutability = MTLMutabilityImmutable;
    }
  }

  NSError* e = nil;
  const attr::Probe probe(device);
  // The loaded archives are never mutated, so concurrent lookups need no lock.
  NSArray<id<MTLBinaryArchive>>* ars = archivesFor(descriptorKey(d, false));
  bool hit = false;
  if (ars.count) {  // archive lookup only: a miss fails fast instead of compiling
    Inflight guard(this, "lookup");
    d.binaryArchives = ars;
    p.render = [device newRenderPipelineStateWithDescriptor:d options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                 reflection:nil error:&e];
    hit = p.render != nil;
    e = nil;
    d.binaryArchives = nil;  // the full compile below and the archived copy never reference them
  }
  const auto hot0 = std::chrono::steady_clock::now();
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
  attr::trackObject("m2.pso.render", p.render, probe);
  notePso(hit, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hot0).count(), s, "render");
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
  p.drawParams = vs.source.find("constant VpDrawParams& vp_draw") != std::string::npos;  // vpmsl kDrawParamsDecl
  p.vpIndexed = vs.source.find("[[viewport_array_index]]") != std::string::npos;
  if (s.fragmentShader) {
    p.fsPosition = fs.source.find("[[position]]") != std::string::npos;
    p.rrmFs = fs.source.find("constant rasterization_rate_map_data& vp_rrm") != std::string::npos;  // vpmsl kRateMapDecl
  }
  p.label = s.label;
  p.vsId = s.vertexShader, p.fsId = s.fragmentShader;
  p.traits = shaderTraits(vs, "vs") + (s.fragmentShader ? shaderTraits(fs, "fs") : std::string());
  if (p.drawParams) p.traits += " vs:drawparams";
  // VP_KNOCK_FSR: no substitute when the fragment shader writes depth (into a depth attachment) or the sample mask, reads the
  // framebuffer or is dual-source: a trivial one would change what the pipeline does beyond its cost.
  if (knockFsrOn && s.fragmentShader && !s.fetch.mask && fs.source.find("[[sample_mask]]") == std::string::npos &&
      fs.source.find("index(1)") == std::string::npos && !(df.depth() && fs.source.find("[[depth(") != std::string::npos)) {
    p.fsrKinds.assign(kMaxColorAttachments, '-');
    for (int i = 0; i < kMaxColorAttachments; ++i) {
      if (!(fs.colorOutputs >> i & 1)) continue;
      const FormatInfo& f = formatInfo(s.colorFormats[i]);
      p.fsrKinds[size_t(i)] = !f.integer() ? 'f' : std::strstr(f.name, "Sint") ? 'i' : 'u';
    }
    MTLRenderPipelineDescriptor* sub = [d copy];
    sub.fragmentFunction = nil;  // the substitute is set when it is built; the original must not outlive its pipeline here
    p.fsrDesc = sub;
  }
  return true;
}

bool MetalBackend::Impl::buildCompute(const PipelineState& s, Pipeline& p, std::string& error) {
  MslShader cs;
  MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
  d.label = @(s.label.c_str());
  // -[MTLComputePipelineDescriptor setComputeFunction:] asserts on nil (CTS log 60: recipes whose
  // shader this process never registered aborted the process from the compile workers).
  id<MTLFunction> cfn = function(s.computeShader, {}, cs, error, false, false, &p.fnKeys[0]);
  if (!cfn) return false;
  d.computeFunction = cfn;
  if (bufImmutable && cs.argumentBufferIndex < 31) d.buffers[cs.argumentBufferIndex].mutability = MTLMutabilityImmutable;  // the argument table (see buildRender)
  NSError* e = nil;
  const attr::Probe probe(device);
  NSArray<id<MTLBinaryArchive>>* ars = archivesFor(descriptorKey(d, true));  // immutable: no lock
  bool hit = false;
  if (ars.count) {
    Inflight guard(this, "lookup");
    d.binaryArchives = ars;
    p.compute = [device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                   reflection:nil error:&e];
    hit = p.compute != nil;
    e = nil;
    d.binaryArchives = nil;
  }
  const auto hot0 = std::chrono::steady_clock::now();
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
  attr::trackObject("m2.pso.compute", p.compute, probe);
  notePso(hit, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hot0).count(), s, "compute");
  if (archiveOn && !hit) queueArchive([d copy], true, s.label);
  p.threads = MTLSizeMake(cs.threadsPerGroup[0], cs.threadsPerGroup[1], cs.threadsPerGroup[2]);
  p.stage[0] = {cs.arguments, cs.argumentBufferIndex};
  p.label = s.label;
  p.vsId = s.computeShader;
  p.traits = shaderTraits(cs, "cs");
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
constexpr uint32_t kDumpSequence = 12;  // frames per dump (about two Source 2 frames)
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
  { std::lock_guard lk(lostWhyMutex); lostWhy = why; }
  lostAtNs.store(steadyNs());  // before the flag: a drain that sees the flag also sees the time
  lostFlag.store(true);
  err("what=device_lost reason=%s (no more GPU work; waits return VK_ERROR_DEVICE_LOST)", token(why).c_str());
  dumpInFlight("device_lost");
  // Native command buffers waiting on the ordering event would never run: release them.
  if (void* ev = nativeEventRaw.load()) ((__bridge id<MTLSharedEvent>)ev).signaledValue = ~uint64_t(0) >> 1;
  { std::lock_guard lock(doneMutex); }  // a waiter between its check and its wait sees the flag
  doneCv.notify_all();
}

// A command buffer failed because memoryless attachments met a tiler flush (too much geometry).
static bool isMemorylessError(NSError* e) {
  if (!e) return false;
  const std::string why = str(e);
  return e.code == MTLCommandBufferErrorMemoryless || why.find("memoryless") != std::string::npos ||
         why.find("ParameterBuffer") != std::string::npos;
}

// A command buffer the GPU watchdog killed (IOGPU "Caused GPU Hang Error", code Timeout): not a refused submission ("Ignored")
// and not a victim of another client's hang.
static bool isGpuHang(NSError* e) {
  if (!e) return false;
  const std::string why = str(e);
  if (why.find("Ignored") != std::string::npos || why.find("InnocentVictim") != std::string::npos) return false;
  return e.code == MTLCommandBufferErrorTimeout || why.find("Hang") != std::string::npos;
}

// Core encoder indices of the encoders Metal reports as faulted (MTLCommandBufferEncoderInfoErrorKey): the encoder labels
// ("render<e>", "render<e>.c<chunk>", "blit_draw<e>.<k>.<l>") start their digits with the core index. Others are skipped.
static std::vector<uint32_t> faultedEncoders(NSError* e) {
  std::vector<uint32_t> out;
  NSArray* infos = e.userInfo[MTLCommandBufferEncoderInfoErrorKey];
  for (id<MTLCommandBufferEncoderInfo> i in infos) {
    if (i.errorState != MTLCommandEncoderErrorStateFaulted) continue;
    const char* dig = i.label.UTF8String ?: "";
    while (*dig && (*dig < '0' || *dig > '9')) ++dig;
    if (*dig) out.push_back(uint32_t(std::strtoul(dig, nullptr, 10)));
  }
  return out;
}

// A tiler flush kills a memoryless command buffer: memoryless / ParameterBuffer, or "Too much geometry to support memoryless...".
static bool isTileFlushError(NSError* e) {
  return isMemorylessError(e) || (e && str(e).find("Too much geometry") != std::string::npos);
}

// Completion handler (any thread) of a command buffer of a frame compiled with a raised tileMaxVertices (CompileStats::tileRaised)
// that failed on a tiler flush: the raised value was the cause, so no stand-in is blamed (no memorylessFailure / bans). The canary
// frame logs "gpu.tile probe fail", a frame under the accepted value "gpu.tile revert"; either way Device is back on the start
// value from the next frame (see vpgpu_metal2_tile_reverted).
void MetalBackend::Impl::tileRaisedFailure(uint64_t frame, NSError* e) {
  const int r = vpgpu::tileRaisedFailed();
  if (r == vpgpu::kRaisedNone) return;
  info(std::string(r == vpgpu::kRaisedProbeFail ? "gpu.tile probe fail" : "gpu.tile revert") + " reason=" + std::to_string(long(e.code)) +
       ":" + token(str(e)) + " f=" + std::to_string(frame));
}

// Completion handler (any thread) of a command buffer that hit the tiler flush: the stand-ins to blame are banned
// (applyMemorylessBans, submit thread) and the failure counts once. A client may encode a whole frame in ONE command
// buffer, so banning all of its stand-ins would put every memoryless target of the frame back in memory for one bad
// encoder. Narrowest attribution first: (1) the stand-ins of the core encoders Metal reported faulted; (2) else those
// of the encoders past half of the HeavyGeometry limits (Encoder::heavy); (3) else all of them. With no stand-in
// recorded the cause is not attributable: global switch.
void MetalBackend::Impl::memorylessFailure(uint64_t frame, const StandIns& stand, const std::vector<uint32_t>& faulted) {
  // Attachment::autoMsaa slots have no other way out than their stand-ins like the rest: banned ones get real memory (still
  // never stored: their load is DontCare / a seed, their store a Resolve), so the tiler can flush them.
  {
    static std::atomic<unsigned> fallbackLogs{0};
    if (fallbackLogs.fetch_add(1) < 8)
      info("gpu.msaa fallback=materialize why=memoryless_failed f=" + std::to_string(frame) + " standins=" + std::to_string(stand.size()) +
           " (autoMsaa attachments of the banned encoders now have memory; their samples are still not stored)");
  }
  if (stand.empty()) {
    if (!memorylessFailed.exchange(true))
      err("what=memoryless_disabled f=%llu reason=tiler_flush_no_standins (attachments now in memory)",
          (unsigned long long)frame);
    return;
  }
  StandIns pick;
  const char* by = "faulted_encoder";
  for (const StandIn& s : stand)
    if (std::find(faulted.begin(), faulted.end(), s.encoder) != faulted.end()) pick.push_back(s);
  if (pick.empty()) {
    by = "heavy_encoders";
    for (const StandIn& s : stand)
      if (s.heavy) pick.push_back(s);
  }
  if (pick.empty()) {
    by = "all_standins";
    pick = stand;
  }
  err("what=memoryless_attrib f=%llu by=%s banned_candidates=%zu of=%zu faulted_encoders=%zu", (unsigned long long)frame,
      by, pick.size(), stand.size(), faulted.size());
  std::lock_guard lock(banMutex);
  pendingBans.push_back(std::move(pick));  // one group per failed command buffer, however many stand-ins it used
}

// Submit thread, start of a frame: the bans handlers queued take effect. Each texture is logged once. A failed command
// buffer counts only when it newly bans a stand-in (several command buffers failing on the same stand-ins are one
// event); more than kMaxMemorylessBans of those (not textures: one command buffer may use many stand-ins): the global
// switch (every attachment in memory).
void MetalBackend::Impl::applyMemorylessBans() {
  std::vector<StandIns> groups;
  {
    std::lock_guard lock(banMutex);
    groups.swap(pendingBans);
  }
  bool any = false;
  for (const StandIns& group : groups) {
    bool fresh = false;
    for (const StandIn& s : group) {
      auto it = textures.find(s.index);
      if (it == textures.end() || it->second.generation != s.generation || it->second.tileBanned) continue;
      Texture& tx = it->second;
      tx.tileBanned = true;
      fresh = true;
      err("what=memoryless_banned tex=%u %ux%u s%u %s", s.index, tx.desc.width, tx.desc.height, unsigned(tx.desc.samples),
          formatInfo(tx.desc.format).name);
    }
    if (fresh) ++memorylessBans, any = true;
  }
  if (any && memorylessBans > kMaxMemorylessBans && !memorylessFailed.exchange(true))
    err("what=memoryless_disabled reason=failed_cbs=%u (attachments now in memory)", memorylessBans);
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
    if (c != seen || vpgpu::backgroundGate().active() || now - lastPass > kPauseNs) {
      seen = c;
      quietSince = now;
    } else if (now - quietSince > kStallNs) {
      markLost("gpu_stuck (every frame slot busy, no command buffer completed for 20 s)");
    } else {
      stallWarn(c, now - quietSince, "frame_slot");
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
    if (c != seen || vpgpu::backgroundGate().active() || t - lastPass > kPauseNs) {
      seen = c;
      quietSince = t;
    } else if (t - quietSince > kStallNs) {
      markLost("gpu_stuck (waiting for idle, no command buffer completed for 20 s)");
    } else {
      stallWarn(c, t - quietSince, "drain");
    }
    lastPass = t;
  }
  for (size_t i = 0; i < got; ++i) dispatch_semaphore_signal(inFlight);
  drained = got == slots.size();
  if (drained && lostFlag.load()) r = WaitResult::Lost;
  return r;
}

MetalBackend::Impl::Slot* MetalBackend::Impl::acquireSlot(uint64_t frame) {
  if (!takeSlotSemaphore()) return nullptr;
  {
    // The semaphore counts frames in flight, not which ones: a frame that completes before an earlier one (a short
    // compute-only frame next to a long render one) frees a count while the slot about to be taken may still be read by its
    // previous frame (argument ring, fences). Wait for that frame itself (with tracked ordering, for every frame up to it).
    // Outside the tables: nothing here needs them, and the guest threads must keep making progress.
    const size_t ix = size_t(submits % slots.size());
    const uint64_t prev = slots[ix].frame;
    bool busy = false;
    uint64_t seen = completions.load();
    int64_t quietSince = steadyNs(), lastPass = quietSince;
    while (!slotSettled(ix)) {
      if (lostFlag.load()) {
        dispatch_semaphore_signal(inFlight);
        return nullptr;
      }
      busy = true;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      const uint64_t c = completions.load();
      const int64_t now = steadyNs();
      if (c != seen || vpgpu::backgroundGate().active() || now - lastPass > kPauseNs) seen = c, quietSince = now;
      else if (now - quietSince > kStallNs) markLost("gpu_stuck (frame slot's own frame still running, no command buffer completed for 20 s)");
      lastPass = now;
    }
    if (busy) {
      const uint64_t total = slotBusyWaits.fetch_add(1) + 1;
      if (total <= 8 || total % 256 == 0)
        info("gpu.order slot_busy f=" + (frame == UINT64_MAX ? std::string("-") : std::to_string(frame)) +
             " waited_for=" + (prev == UINT64_MAX ? std::string("-") : std::to_string(prev)) + " total=" + std::to_string(total));
    }
  }
  std::lock_guard<std::recursive_mutex> tables(tableMutex);  // after the wait: graveyard / residency vs retire() of guest threads
  // Consecutive command buffers use different slots, so the frontier carried over from the
  // previous one never holds a fence this one updates.
  promotePinned();  // before the graveyard moves into this slot
  Slot& slot = slots[submits++ % slots.size()];
  const size_t ix = size_t(&slot - slots.data());
  slotDone[ix].store(false, std::memory_order_release);  // its frame is in flight until its publish says otherwise
  slot.frame = frame;
  lastSlot = int(ix);
  // The slot's previous frame finished, and so did every frame that could use its leftovers:
  // only now do they leave the residency set and get released.
  for (id obj : slot.retired)
    if (residency && [obj conformsToProtocol:@protocol(MTLAllocation)]) {
      [residency removeAllocation:obj];
      bump(cnt.resRemove);
      residencyDirty = true;
    }
  if (!slot.retired.empty()) {
    cnt.heldObjs.fetch_sub(std::min<uint64_t>(cnt.heldObjs.load(), slot.retired.size()), std::memory_order_relaxed);
    cnt.heldBytes.fetch_sub(std::min<uint64_t>(cnt.heldBytes.load(), slot.retiredBytes), std::memory_order_relaxed);
  }
  slot.retired = std::move(graveyard);
  slot.retiredBytes = graveyardBytes;
  graveyard.clear();
  graveyardBytes = 0;
  if (residencyDirty) {
    [residency commit];
    bump(cnt.resCommit);
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
  if (!nativeScratch) {
    nativeScratch = [device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
    attr::track("m2.native.scratch", nativeScratch);
  }
  if (!nativeEvent || !nativeScratch) {
    err("what=native_event");
    return false;
  }
  Slot* slotp = acquireSlot();
  if (!slotp) return false;  // device lost
  nativeSinceFrame = true;  // VP_CB_ORDER=2: the next frame joins every encoder (the client's accesses are unknown)
  Slot& slot = *slotp;
  const size_t slotIx = size_t(slotp - slots.data());
  if (slot.fences.empty()) slot.fences.push_back([device newFence]);
  id<MTLFence> tailFence = slot.fences[0];
  const uint64_t n = ++nativeSubmits;
  Impl* impl = this;  // the destructor waits for every slot, so no handler outlives *this

  auto make = [&](const char* part) {
    MTLCommandBufferDescriptor* cbd = [MTLCommandBufferDescriptor new];
    cbd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> c = [queue commandBufferWithDescriptor:cbd];
    bump(cnt.cbs);
    c.label = [NSString stringWithFormat:@"vpgpu.native%llu.%s", (unsigned long long)n, part];
    // Handler runs long after this lambda and submit's frame are gone: the block must hold copies, not
    // references to the enclosing function's locals (the lambda captures them by reference).
    Impl* const hImpl = impl;
    const uint64_t hN = n;
    [c addCompletedHandler:^(id<MTLCommandBuffer> finished) {
      if (finished.status != MTLCommandBufferStatusError) return;
      std::string encs;
      NSArray* infos = finished.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
      for (id<MTLCommandBufferEncoderInfo> i in infos)
        if (i.errorState == MTLCommandEncoderErrorStateFaulted || i.errorState == MTLCommandEncoderErrorStateAffected)
          encs += (encs.empty() ? "" : ",") + token(i.label.UTF8String ?: "?");
      // Refused in the background: no error line (one gpu.cb.background line per second instead).
      if (isBackgroundError(finished.error)) hImpl->noteBackgroundError();
      else hImpl->err("what=native_command_buffer n=%llu part=%s error=%s encoders=%s", (unsigned long long)hN, part,
                      token(str(finished.error)).c_str(), encs.c_str());
      if (hImpl->lostByError(finished.error))
        hImpl->markLost(std::string("native_command_buffer ") + part + " " + str(finished.error));
    }];
    return c;
  };
  id<MTLCommandBuffer> headCb = make("head");
  id<MTLCommandBuffer> clientCb = make("client");
  id<MTLCommandBuffer> tailCb = make("tail");
  const uint64_t afterFrames = ++nativeValue, afterClient = ++nativeValue;

  id<MTLBlitCommandEncoder> head = [headCb blitCommandEncoder];
  bump(cnt.encBlit);
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
  bump(cnt.encBlit);
  end.label = @"vpgpu.native.before_frames";
  [end fillBuffer:nativeScratch range:NSMakeRange(4, 4) value:0];
  [end updateFence:tailFence];
  [end endEncoding];
  frontier.push_back(tailFence);
  if (residencyDirty) {
    [residency commit];
    bump(cnt.resCommit);
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
    impl->slotDone[slotIx].store(true, std::memory_order_release);  // the tail completed: nothing of this slot runs
    impl->completions.fetch_add(1);
    impl->doneCv.notify_all();
    dispatch_semaphore_signal(sem);  // last: once every slot is back the backend may be destroyed
  }];
  vpgpu::backgroundGate().waitForeground();  // once for the three: they commit back to back
  [headCb commit];
  [clientCb commit];
  [tailCb commit];

  // Watchdog (see above). Checked and logged under the mutex (see Watch). The block holds the command buffers weakly:
  // a strong capture would pin them (and what they retain) for the 2 s even when they completed long before.
  __weak id<MTLCommandBuffer> weakHead = headCb;
  __weak id<MTLCommandBuffer> weakClient = clientCb;
  __weak id<MTLCommandBuffer> weakTail = tailCb;
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC),
                 dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
    std::lock_guard g(watch->m);
    if (watch->done) return;
    id<MTLCommandBuffer> h = weakHead, c = weakClient, t = weakTail;  // already released: status -1
    auto st = [](id<MTLCommandBuffer> cb) { return cb ? static_cast<int>(cb.status) : -1; };
    impl->err("what=native_stuck n=%llu head=%d client=%d tail=%d (status 4=completed 5=error; "
              "2/3=waiting on the event or the GPU; -1=released)", (unsigned long long)n, st(h), st(c), st(t));
  });
  return true;
}

void MetalBackend::Impl::submit(uint64_t frameIndex, const Frame& frame, const CompiledFrame& compiled) {
  tileRejectedTotal += compiled.stats.tileRejected;
  {
    std::lock_guard<std::recursive_mutex> t(tableMutex);  // retire() fills the graveyard that destroy* of guest threads fill too
    dropPipelines();  // before any lookup of this frame: the only thread that holds Pipeline pointers
  }
  if (compiled.encoders.empty()) {  // nothing for the GPU; keeps the frontier's fences out of reuse
    {
      std::lock_guard lock(doneMutex);
      done.push_back({frameIndex, {}});
    }
    frameFinished(frameIndex);
    return;
  }
  Slot* slotp = acquireSlot(frameIndex);
  if (!slotp) {  // device lost: the frame does not run (waits report the loss)
    frameFinished(frameIndex);
    return;
  }
  Slot& slot = *slotp;
  const size_t slotIx = size_t(slotp - slots.data());
  // Encoding reads the resource tables throughout: held until the last commit, released only around the waits in
  // commitChunk (see tableMutex).
  std::unique_lock<std::recursive_mutex> tables(tableMutex);
  applyMemorylessBans();  // failures of earlier frames' command buffers: their stand-ins get real memory from here on
  if (trimLevel.load(std::memory_order_relaxed) && trimSlotsLeft.load()) {  // idle slot: drop its chained argument buffers
    uint64_t freed = 0;
    for (id<MTLBuffer> b : slot.argsMore) {
      freed += b.length;
      rangeDel(b.contents);
      retire(b);
    }
    slot.argsMore.clear();
    argsMoreBytes -= std::min<uint64_t>(argsMoreBytes.load(), freed);
    trimmedArgsBytes += freed;
    trimSlotsLeft.fetch_sub(1);
  }
  // One fence per encoder was created in this slot's earlier frames (tens of thousands in a copy-heavy one); the slot's
  // previous frame is complete, so release the excess. Index 0 stays (encodeNative's tail fence); a fence still in the
  // frontier is retained by it.
  if (slot.fences.size() > kFenceKeep) slot.fences.erase(slot.fences.begin() + kFenceKeep, slot.fences.end());

  MTLCommandBufferDescriptor* cbd = [MTLCommandBufferDescriptor new];
  cbd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;  // per-encoder fault in the log
  id<MTLCommandBuffer> cb = [queue commandBufferWithDescriptor:cbd];
  bump(cnt.cbs);
  cb.label = [NSString stringWithFormat:@"vpgpu.f%llu", (unsigned long long)frameIndex];
  // Hang diagnostics: the pipelines this command buffer runs, per encoder (see InFlight).
  InFlight track;
  track.label = "vpgpu.f" + std::to_string(frameIndex);
  track.chunksDone = std::make_shared<std::atomic<uint32_t>>(0);
  uint32_t chunk = 0;  // command buffers of this frame committed so far (hang split, draw budget)
  uint32_t splitCount = 0;  // of those, the cuts inside a render encoder (hang split)
  uint32_t cbDraws = 0, cbPasses = 0;  // draws / core encoders in the command buffer being filled
  uint32_t cbEncoders = 0;  // Metal encoders (all kinds) opened in the command buffer being filled
  uint64_t cbIntBytes = 0;  // internal staging buffers created for the command buffer being filled (bufIntLiveBytes)
  // Memoryless stand-ins used by the command buffer being filled (a new list per command buffer; the completion
  // handlers of a committed one hold their own copy of the pointer and only read it).
  auto cbStand = std::make_shared<StandIns>();
  // Sample groups of the command buffer being filled, for vpgpu_metal2_cb_busy_ms (see gBusyOn); only filled when on.
  const bool busyOn = gBusyOn.load(std::memory_order_relaxed);
  auto cbBusy = std::make_shared<std::vector<BusyRec>>();
  std::deque<dispatch_semaphore_t> chunkDone;  // one per committed chunk, signalled by its completion handler, oldest first
  std::function<void()> cutIfFull;  // set after commitChunk; encodeBlit (defined before it) calls it
  double cbCost = 0;  // estimated cost (draw units, see encoderCost) of the command buffer being filled
  Impl* const self = this;  // the destructor waits for every frame, so handlers never outlive *this
  uint32_t encDrawsCut = 0;  // draws in the chunk of the render encoder being cut (hang split only)
  // Frame completion: `us` sums the GPU time of every command buffer; `remaining` counts the ones not
  // finished (the final one is the initial 1). Whichever handler ends last, in any order, publishes.
  struct FrameTail {
    std::atomic<uint64_t> us{0};
    std::atomic<uint32_t> remaining{1};
    id<MTLCommandBuffer> last = nil;
    std::function<void(id<MTLCommandBuffer>)> publish;
  };
  auto tail = std::make_shared<FrameTail>();
  // Per draw: two counters and an integer set lookup; the text is built once per (encoder, pipeline).
  std::unordered_set<uint64_t> trackSeen;
  auto trackPipeline = [&](uint32_t e, const Pipeline* p, bool dispatch) {
    if (dispatch) ++track.dispatches;
    else ++track.draws;
    const uint64_t key = uint64_t(reinterpret_cast<uintptr_t>(p)) ^ (uint64_t(e) * 0x9E3779B97F4A7C15ull);
    if (!trackSeen.insert(key).second) return;
    if (track.pipelines.size() >= 48) {
      ++track.dropped;
      return;
    }
    char ids[64];
    std::snprintf(ids, sizeof ids, " vs=%016llx fs=%016llx", (unsigned long long)p->vsId, (unsigned long long)p->fsId);
    track.pipelines.push_back("e" + std::to_string(e) + ":" + (p->label.empty() ? "?" : p->label) + ids + p->traits);
  };

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
  // Render encoders (gpu.excl): vertex-end / fragment-start sample indices, and what each core encoder is.
  std::vector<std::pair<NSUInteger, NSUInteger>> encStages(compiled.encoders.size(), {NSNotFound, NSNotFound});
  const auto diag = std::make_shared<std::vector<EncDiag>>(exclOn ? compiled.encoders.size() : 0);  // gpu.excl only
  // VP_KNOCK: what this frame is encoded as (mode 0 = nothing: the only cost off is the branches on `knockFrame.mode`).
  KnockFrame knockFrame;
  uint32_t knockSeen = 0;       // encoders with the dominant signature that were encoded
  uint32_t knockSkipped = 0;    // draws left out
  bool knockWarmSeen = false;   // warm-up: an encoder that counts (samples > 1, >= kKnockWarmDraws draws) was encoded
  if (gKnockArmReq.load(std::memory_order_relaxed)) knockArm();
  if (knockOn) knockBegin(knockFrame);
  // Frame ordering (see cbOrder): the previous frame's tail, which every encoder of this one waits on through one fence (an
  // encoder at the start of the frame waits on carriedIn and updates it), so each encoder adds one wait, at its vertex
  // stage, whatever stage its own barrier asked for.
  const std::vector<id<MTLFence>> carriedIn = cbOrder ? frontier : std::vector<id<MTLFence>>{};
  id<MTLFence> joinFence = nil;
  uint32_t curEnc = 0;  // core encoder being encoded
  // Per core encoder, whether it waits on the join fence and from which render stage: 2 = before its vertex stage (always,
  // unless VP_CB_ORDER=2), 1 = before its fragment stage (only fragment-stage accesses touch what a frame in flight uses),
  // 0 = no wait (nothing in common).
  std::vector<uint8_t> needJoin(compiled.encoders.size(), 2);

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
        bump(cnt.bufInternal);
        bump(cnt.bufInternalBytes, size);
        if (!b) {
          --argChunk;
          ++argOverflow;
          why = "arg_overflow";
          return kSkip;
        }
        b.label = [NSString stringWithFormat:@"vpgpu.args.%zu", argChunk];
        attr::track("m2.args.more", b);
        slot.argsMore.push_back(b);
        rangeAdd(b.contents, b.length);
        argsMoreBytes += size;
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
      const char* missWhy = "null_resource";  // why `v` stays 0 (gpu.missing)
      uint32_t missIndex = r.index, missGen = r.generation;
      switch (r.kind) {
        case ResourceRange::Kind::Buffer: {
          auto b = buffers.find(r.index);
          if (b != buffers.end() && b->second.generation == r.generation) {
            v = b->second.buffer.gpuAddress + b->second.base + r.begin;
            if (!residency) pendingUse.push_back({b->second.buffer, u});
          } else {
            missWhy = b == buffers.end() ? "buffer_unknown" : "buffer_stale_generation";
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
          } else {
            missIndex = use.index, missGen = use.generation;
            auto tx = textures.find(use.index);
            missWhy = use.index & kViewBit ? "view_unavailable"
                      : tx == textures.end() ? "texture_unknown"
                      : tx->second.generation != use.generation ? "texture_stale_generation"
                                                                : "texture_not_created";
          }
          std::memcpy(&v, &rid, sizeof rid);
          break;
        }
        case ResourceRange::Kind::Sampler: {
          auto sm = samplers.find(r.index);
          if (sm != samplers.end() && sm->second.generation == r.generation) rid = sm->second.sampler.gpuResourceID;
          else missWhy = sm == samplers.end() ? "sampler_unknown" : "sampler_stale_generation";
          std::memcpy(&v, &rid, sizeof rid);
          break;
        }
      }
      if (v == 0) {
        ++missingResources;
        missing(bindKindName(a.kind), unsigned(forStage), a.slot, missWhy, missIndex, missGen);
      }
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
    if (busyOn) cbBusy->push_back({'c', start, end, MTLCounterDontSample, MTLCounterDontSample});
  };
  // Render encoders, gpu.excl: vertex start, vertex end, fragment start, fragment end.
  auto sampleQuad = [&](uint32_t e, NSUInteger (&q)[4]) {
    q[0] = q[1] = q[2] = q[3] = MTLCounterDontSample;
    if (!stageSamples || sampleUsed + 4 > kMaxSamples) return;
    for (NSUInteger& x : q) x = sampleUsed++;
    if (encSamples[e].first == NSNotFound) encSamples[e].first = q[0];
    encSamples[e].second = q[3];
    encStages[e] = {q[1], q[2]};
    if (busyOn) cbBusy->push_back({'r', q[0], q[3], q[1], q[2]});
  };

  Open open;
  id<MTLFence> openFence = nil;
  // Per-pass dump of this frame (diagnostics): items to write after completion, copies waiting for
  // the open encoder to end.
  // The first command buffer at or after VP_DUMP_FRAME that renders a real frame (8+ encoders, a layered pass):
  // Source 2 also submits tiny command buffers, which would take the dump with nothing in it (items=0).
  bool dumpLayered = false;
  for (const Encoder& en : compiled.encoders) dumpLayered |= en.kind == Encoder::Kind::Render && en.pass.layerCount > 1;
  const bool realFrame = compiled.encoders.size() >= 8 && dumpLayered;
  // A sequence already started takes the following frames too, whatever they hold.
  if (dumpSeqLeft == 0 && dumpFrame >= 0 && !dumpTaken && frameIndex >= uint64_t(dumpFrame) && realFrame) {
    dumpTaken = true;
    dumpSeqLeft = kDumpSequence;
    dumpSeqBytes = 0;
    info("gpu.dump sequence f=" + std::to_string(frameIndex) + " command_buffers=" + std::to_string(kDumpSequence));
  }
  // Asked for from the host (vpgpu_metal2_request_dump: the user saw a glitch): the next real frame, as one more
  // sequence (needs VP_DUMP_DIR; without it the request stays pending, nothing is written).
  if (dumpSeqLeft == 0 && realFrame && !dumpDir.empty() && gDumpRequest.exchange(false)) {
    dumpSeqLeft = kDumpSequence;
    dumpSeqBytes = 0;
    info("gpu.dump requested f=" + std::to_string(frameIndex) + " command_buffers=" + std::to_string(kDumpSequence));
  }
  const bool dumping = dumpSeqLeft > 0;
  if (dumping) --dumpSeqLeft;
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
      else if (bytes > kDumpMaxItemBytes || dumpSeqBytes + dumpBytes + bytes > kDumpMaxTotalBytes) it.note = "too_big";
      else if (!(it.buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared])) it.note = "no_memory";
      else {
        attr::track("m2.dump", it.buffer);
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
      bump(cnt.encBlit), ++cbEncoders;
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
    // Ordered after the whole previous frame, not only through this one's first encoder
    // (VP_CB_ORDER=2: only the encoders that touch what a frame in flight uses).
    const uint8_t join = curEnc < needJoin.size() ? needJoin[curEnc] : 2;
    if (joinFence && join) {
      if (open.render) [open.render waitForFence:joinFence beforeStages:join > 1 ? MTLRenderStageVertex : MTLRenderStageFragment];
      else if (open.compute) [open.compute waitForFence:joinFence];
      else [open.blit waitForFence:joinFence];
      carriedWaits.fetch_add(1, std::memory_order_relaxed);
      if (curEnc < diag->size()) (*diag)[curEnc].joined = true;
    }
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
    if (!residency) {
      std::lock_guard<std::mutex> lock(addressMutex);
      for (id<MTLBuffer> b : addressTaken) {
        const uintptr_t key = reinterpret_cast<uintptr_t>((__bridge const void*)b) | 1;
        if (!used.insert(key).second) continue;
        if (open.render) [open.render useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageVertex | MTLRenderStageFragment];
        else if (open.compute) [open.compute useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      }
    }
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
  // `view` >= 0 (Encoder::view): the encoder draws one layer of the pass as its layer 0; only the rects covering that view are drawn.
  auto encodeClear = [&](const ClearRects& c, MTLRenderPassDescriptor* rp, const RenderPassDesc& pass, int view) {
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
    // A render target reduced to the render area (renderTargetWidth / Height, 0 = the attachments'): the quads'
    // viewport and scissor stay inside it.
    double w = double(std::max<NSUInteger>(1, ref.width >> level));
    double h = double(std::max<NSUInteger>(1, ref.height >> level));
    if (rp.renderTargetWidth && rp.renderTargetHeight)
      w = std::min<double>(w, double(rp.renderTargetWidth)), h = std::min<double>(h, double(rp.renderTargetHeight));
    const bool layered = rp.renderTargetArrayLength > 1;
    ClearRects cc = c;  // only what this encoder has
    for (int i = 0; i < kMaxColorAttachments; ++i)
      if (!rp.colorAttachments[i].texture) cc.colorMask &= uint8_t(~(1u << i));
    cc.depth = c.depth && rp.depthAttachment.texture;
    cc.stencil = c.stencil && rp.stencilAttachment.texture;
    if ((c.stencil && !cc.stencil) || (c.depth && !cc.depth)) {  // diagnostics: a depth/stencil clear the encoder cannot take
      static std::atomic<int> dropped{0};
      if (dropped.fetch_add(1) < 8)
        err("what=clear_ds_dropped depth=%d stencil=%d depthTex=%d stencilTex=%d depthPf=%lu", int(c.depth), int(c.stencil),
            int(rp.depthAttachment.texture != nil), int(rp.stencilAttachment.texture != nil),
            (unsigned long)(rp.depthAttachment.texture ? rp.depthAttachment.texture.pixelFormat : 0));
    }
    if (!cc.colorMask && !cc.depth && !cc.stencil) return;
    std::string error;
    id<MTLRenderPipelineState> ps = clearPipeline(rp, cc, layered, kinds, error);
    if (!ps) {
      ++missingPipelines;
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
      if (view >= 0 && !(r.layer <= view && view < r.layer + r.layerCount)) continue;
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
      const TextureId bad = !st ? s.texture : d.texture;
      missing("texture", 0, 0, !st ? "blit_src_texture_nil" : "blit_dst_texture_nil", bad.index, bad.generation);
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
      // Copy::blitSwapRB: the source read with red and blue swapped (a swizzled view of the view).
      if (sv && c.blitSwapRB)
        sv = [sv newTextureViewWithPixelFormat:sv.pixelFormat
                                   textureType:sv.textureType
                                        levels:NSMakeRange(0, 1)
                                        slices:NSMakeRange(0, 1)
                                       swizzle:MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue, MTLTextureSwizzleGreen,
                                                                             MTLTextureSwizzleRed, MTLTextureSwizzleAlpha)];
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
      if (cutIfFull) cutIfFull();
      open.kind = Open::Render;
      open.render = [cb renderCommandEncoderWithDescriptor:rp];
      bump(cnt.encRender), ++cbEncoders;
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

  // gpu.slowcb for the command buffer being filled (call before it is committed).
  auto slowLog = [&](id<MTLCommandBuffer> c, uint32_t draws, uint32_t passes, double cost) {
    const std::string label = token(c.label ? std::string(c.label.UTF8String) : std::string("?"));
    // By-value copies: `self` and `frameIndex` are submit's locals, reached here through this lambda's
    // reference capture; a block would keep the dangling reference and the handler runs after submit returns.
    Impl* const hSelf = self;
    const uint64_t hFrame = frameIndex;
    [c addCompletedHandler:^(id<MTLCommandBuffer> finished) {
      const double ms = (finished.GPUEndTime - finished.GPUStartTime) * 1000.0;
      if (ms <= kSlowCbMs || hSelf->slowCbLines.fetch_add(1) >= kSlowCbMaxLines) return;
      char line[256];
      std::snprintf(line, sizeof line, "gpu.slowcb f=%llu ms=%.1f draws=%u passes=%u est_cost=%.0f label=%s",
                    (unsigned long long)hFrame, ms, draws, passes, cost, label.c_str());
      hSelf->info(line);
    }];
  };
  // Backpressure: waits for the oldest committed chunk of this frame. Chunks of one frame are ordered only by queue order
  // and fences, so this wait cannot be on something the CPU still has to do: a committed command buffer waits only on
  // earlier ones (fences updated before, the frame's earlier chunks, earlier frames' command buffers and
  // encodeNative's events, all already committed). Stuck detection mirrors takeSlotSemaphore (20 s, counting
  // completed chunks as progress: a long frame completes chunks, not frames).
  auto awaitOldestChunk = [&]() {
    dispatch_semaphore_t s = chunkDone.front();
    chunkDone.pop_front();
    if (lostFlag.load()) return;  // no more GPU work will run: stop throttling
    if (dispatch_semaphore_wait(s, DISPATCH_TIME_NOW) == 0) return;  // already complete: not a wait (keeps chunk_wait honest)
    const int64_t start = steadyNs();
    uint64_t seen = completions.load() + chunkCompletions.load();
    int64_t quietSince = start, lastPass = start;
    while (dispatch_semaphore_wait(s, dispatch_time(DISPATCH_TIME_NOW, kSliceNs)) != 0) {
      if (lostFlag.load()) break;
      const uint64_t c = completions.load() + chunkCompletions.load();
      const int64_t now = steadyNs();
      if (c != seen || vpgpu::backgroundGate().active() || now - lastPass > kPauseNs) {
        seen = c;
        quietSince = now;
      } else if (now - quietSince > kStallNs) {
        markLost("gpu_stuck (frame command buffer backlog, no command buffer completed for 20 s)");
      } else {
        stallWarn(c, now - quietSince, "chunk_backlog");
      }
      lastPass = now;
    }
    bump(cnt.chunkWaits);
    bump(cnt.chunkWaitUs, uint64_t((steadyNs() - start) / 1000));
  };
  // vpgpu_metal2_cb_busy_ms: when `c` completes, its sample groups are resolved and its exclusive GPU time put in the
  // ring. Added before the handler that counts the chunk done (it reads `slot.samples`, which that handler's frame
  // publish frees) and before commit, so the host's own completed handler, added inside -commit, finds the entry. Touches
  // only the globals: nothing of the backend.
  auto addBusyHandler = [&](id<MTLCommandBuffer> c) {
    if (!busyOn || !stageSamples) return;
    const std::shared_ptr<std::vector<BusyRec>> recs = cbBusy;
    id<MTLCounterSampleBuffer> sb = slot.samples;
    const double msPerTick = gpuTickNs / 1e6;
    [c addCompletedHandler:^(id<MTLCommandBuffer> finished) {
      if (finished.status != MTLCommandBufferStatusCompleted) return;
      NSUInteger lo = NSUIntegerMax, hi = 0;
      for (const BusyRec& r : *recs)
        for (NSUInteger k : {r.start, r.end, r.vertexEnd, r.fragmentStart})
          if (k != MTLCounterDontSample) lo = std::min(lo, k), hi = std::max(hi, k + 1);
      std::vector<std::pair<uint64_t, uint64_t>> span, stages;
      std::vector<char> kinds;
      if (lo < hi) {
        NSData* data = [sb resolveCounterRange:NSMakeRange(lo, hi - lo)];
        const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
        auto tick = [&](NSUInteger k) -> uint64_t {
          if (!ts || k == MTLCounterDontSample || k < lo || k >= hi) return 0;  // not sampled
          const uint64_t t = ts[k - lo].timestamp;
          return t == MTLCounterErrorValue ? 0 : t;
        };
        for (const BusyRec& r : *recs) {
          span.push_back({tick(r.start), tick(r.end)});
          stages.push_back({tick(r.vertexEnd), tick(r.fragmentStart)});
          kinds.push_back(r.kind);
        }
      }
      busyAccount((__bridge const void*)finished, span, stages, kinds, msPerTick);
    }];
  };
  // Commits the command buffer being filled (no encoder open) and starts the next one on the same
  // queue; queue order keeps the work in order and the fences keep ordering the encoders across it.
  auto commitChunk = [&]() {
    if (residencyDirty) {
      [residency commit];
      bump(cnt.resCommit);
      residencyDirty = false;
    }
    // Handlers that touch the backend go before the one that counts this chunk done: the frame's publishing
    // handler (which lets the backend be destroyed) can only run after every chunk has been counted.
    slowLog(cb, cbDraws, cbPasses, cbCost);
    const std::shared_ptr<CbTimes> times = trackCb(cb, frameIndex);
    addBusyHandler(cb);
    std::shared_ptr<std::atomic<uint32_t>> doneChunks = track.chunksDone;
    const uint32_t thisChunk = chunk;
    // By-value copies (see slowLog): the block must not reference submit's locals through this lambda.
    Impl* const hSelf = self;
    const uint64_t hFrame = frameIndex;
    const bool hRaised = compiled.stats.tileRaised > 0;  // compiled under a raised tileMaxVertices: a tiler flush is its doing
    const std::shared_ptr<FrameTail> hTail = tail;
    const std::shared_ptr<StandIns> hStand = cbStand;  // this chunk's stand-ins (complete: it is being committed)
    const uint64_t hIntBytes = cbIntBytes;  // staging buffers this chunk retains until it completes
    cbIntBytes = 0;
    // Chunk completion for the backpressure wait. Touches the backend, so it goes before the handler that counts this
    // chunk (which may publish the frame and let the backend be destroyed).
    dispatch_semaphore_t chunkSem = dispatch_semaphore_create(0);
    chunkDone.push_back(chunkSem);
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
      hSelf->bufIntLiveBytes.fetch_sub(hIntBytes, std::memory_order_relaxed);
      hSelf->chunkCompletions.fetch_add(1, std::memory_order_relaxed);
      dispatch_semaphore_signal(chunkSem);
    }];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> finished) {
      if (finished.status == MTLCommandBufferStatusError) {
        // Refused in the background: no error line, no hang report (one gpu.cb.background line per second instead).
        const bool background = isBackgroundError(finished.error);
        if (background) hSelf->noteBackgroundError();
        else hSelf->err("what=command_buffer f=%llu chunk=%u error=%s", (unsigned long long)hFrame, thisChunk,
                        token(str(finished.error)).c_str());
        if (hRaised && isTileFlushError(finished.error)) hSelf->tileRaisedFailure(hFrame, finished.error);
        else if (isMemorylessError(finished.error)) hSelf->memorylessFailure(hFrame, *hStand, faultedEncoders(finished.error));
        if (isGpuHang(finished.error)) gGpuHangs.fetch_add(1);
        if (hSelf->lostByError(finished.error))
          hSelf->markLost("command_buffer f=" + std::to_string(hFrame) + " chunk=" + std::to_string(thisChunk) + " " +
                          str(finished.error));
        if (!background) hSelf->dumpInFlight("command_buffer_error");  // names the frame's pipelines and chunk, lost or not
      }
      hTail->us.fetch_add(uint64_t((finished.GPUEndTime - finished.GPUStartTime) * 1e6));
      doneChunks->fetch_add(1);
      if (hTail->remaining.fetch_sub(1) == 1) {
        auto publish = std::move(hTail->publish);
        publish(hTail->last);
      }
    }];
    tail->remaining.fetch_add(1);
    if (chunk == 0) {  // listed before the first commit, so a partial-frame error finds its frame
      InFlight first = track;
      first.chunks = 1;
      first.committedNs = steadyNs();
      std::lock_guard g(inFlightMutex);
      inFlightCbs[frameIndex] = std::move(first);
    }
    tables.unlock();  // the commit may park (app in the background) and the throttle below waits for the GPU
    timedCommit(cb, times);
    ++chunk;
    cbDraws = cbPasses = 0;
    cbCost = 0;
    cb = [queue commandBufferWithDescriptor:cbd];
    cbStand = std::make_shared<StandIns>();
    cbBusy = std::make_shared<std::vector<BusyRec>>();
    bump(cnt.cbs);
    cb.label = [NSString stringWithFormat:@"vpgpu.f%llu.c%u", (unsigned long long)frameIndex, chunk];
    cbEncoders = 0;
    // Throttle: at most `maxPendingChunks` earlier chunks of this frame uncompleted (their encoder memory stays
    // pinned until they complete). Nothing is open and the next command buffer is empty, so waiting here is safe.
    while (maxPendingChunks && chunkDone.size() > maxPendingChunks) awaitOldestChunk();
    tables.lock();
  };
  // Encoder budget (see maxEncodersPerCb): called only where no encoder is open.
  cutIfFull = [&]() {
    if (!maxEncodersPerCb || dumping || open.kind != Open::None || cbEncoders < maxEncodersPerCb) return;
    bump(cnt.cutEnc);
    commitChunk();
  };

  // VP_CB_ORDER=2: this frame's memory accesses, and which encoders touch what an in-flight one does.
  CbAccess mine;
  if (cbTracked) {
    using Range = std::pair<uintptr_t, uintptr_t>;
    auto mergeRanges = [](std::vector<Range>& v) {
      std::sort(v.begin(), v.end());
      size_t o = 0;
      for (size_t i = 0; i < v.size(); ++i) {
        if (o && v[i].first <= v[o - 1].second) v[o - 1].second = std::max(v[o - 1].second, v[i].second);
        else v[o++] = v[i];
      }
      v.resize(o);
    };
    auto uniqueKeys = [](std::vector<uint64_t>& v) {
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    {  // frames that completed can no longer conflict
      std::lock_guard lock(doneMutex);
      for (auto it = cbAccess.begin(); it != cbAccess.end();) {
        const bool finished = it->frame < finishedBelow || finishedAhead.count(it->frame);
        it = finished ? cbAccess.erase(it) : std::next(it);
      }
    }
    std::vector<Range> inW, inR;
    std::vector<uint64_t> inTW, inTR;
    for (const CbAccess& a : cbAccess) {
      inW.insert(inW.end(), a.bufW.begin(), a.bufW.end());
      inR.insert(inR.end(), a.bufR.begin(), a.bufR.end());
      inTW.insert(inTW.end(), a.texW.begin(), a.texW.end());
      inTR.insert(inTR.end(), a.texR.begin(), a.texR.end());
    }
    mergeRanges(inW), mergeRanges(inR), uniqueKeys(inTW), uniqueKeys(inTR);
    // Disjoint sorted ranges: the first one ending after r.first decides.
    auto hits = [](const std::vector<Range>& v, const Range& r) {
      auto it = std::upper_bound(v.begin(), v.end(), r.first, [](uintptr_t x, const Range& y) { return x < y.second; });
      return it != v.end() && it->first < r.second;
    };
    std::vector<Range> eW, eR;
    std::vector<uint64_t> eTW, eTR;
    // Accesses of the encoder's vertex stage (or of any stage) mark `early`: the join then goes before the
    // vertex stage; only fragment-stage accesses (attachments, fragment shader bindings) wait before fragments.
    std::vector<uint8_t> eWEarly, eREarly, eTWEarly, eTREarly;
    auto visit = [&](const Access& a) {
      const ResourceRange& r = a.range;
      const uint8_t early = (a.stages == 0 || (a.stages & ~stage::fragment)) ? 1 : 0;
      // A pixel-local read makes the texture an attachment of the encoder (loaded and stored back): a write.
      const bool wr = a.write || a.pixelLocal;
      if (r.kind == ResourceRange::Kind::Texture) {
        (wr ? eTW : eTR).push_back(uint64_t(r.generation) << 32 | r.index);
        (wr ? eTWEarly : eTREarly).push_back(early);
        return;
      }
      if (r.kind != ResourceRange::Kind::Buffer) return;
      auto it = buffers.find(r.index);
      if (it == buffers.end() || it->second.generation != r.generation || !it->second.host) return;
      const uint64_t b0 = std::min<uint64_t>(r.begin, it->second.span), b1 = std::min<uint64_t>(r.end, it->second.span);
      if (b1 <= b0) return;
      (wr ? eW : eR).push_back({it->second.host + uintptr_t(b0), it->second.host + uintptr_t(b1)});
      (wr ? eWEarly : eREarly).push_back(early);
    };
    const bool afterNative = nativeSinceFrame;
    nativeSinceFrame = false;
    for (uint32_t e = 0; e < compiled.encoders.size(); ++e) {
      eW.clear(), eR.clear(), eTW.clear(), eTR.clear();
      eWEarly.clear(), eREarly.clear(), eTWEarly.clear(), eTREarly.clear();
      // The merged encoder's own attachments (slots a fetch brought in included), then each pass's.
      if (compiled.encoders[e].kind == Encoder::Kind::Render)
        for (const Access& a : attachmentAccesses(compiled.encoders[e].pass)) visit(a);
      for (uint32_t ci : compiled.encoders[e].commands) {
        const Command& c = cmds[ci];
        if (const auto* rpc = std::get_if<RenderPass>(&c)) {
          for (const Access& a : attachmentAccesses(rpc->desc)) visit(a);
          // Bindings, index / indirect ranges and the visibility buffer of every draw.
          for (const Draw& d : rpc->draws)
            if (!d.clear)
              for (const Access& a : drawAccesses(d)) visit(a);
        } else if (const auto* dp = std::get_if<Dispatch>(&c)) {
          for (const Access& a : dp->bindings) visit(a);
          if (dp->indirect) visit(vpgpu::read(ResourceRange::wholeBuffer(dp->indirect->buffer), stage::compute));
        } else {
          for (const Access& a : copyAccesses(std::get<Copy>(c))) visit(a);
        }
      }
      uint8_t join = 0;  // 2: some conflicting access is early (vertex or any stage), 1: fragment only
      auto note = [&](bool conflict, uint8_t early) {
        if (conflict) join = std::max<uint8_t>(join, early ? 2 : 1);
      };
      for (size_t k = 0; k < eW.size() && join < 2; ++k) note(hits(inW, eW[k]) || hits(inR, eW[k]), eWEarly[k]);
      for (size_t k = 0; k < eR.size() && join < 2; ++k) note(hits(inW, eR[k]), eREarly[k]);
      for (size_t k = 0; k < eTW.size() && join < 2; ++k)
        note(std::binary_search(inTW.begin(), inTW.end(), eTW[k]) || std::binary_search(inTR.begin(), inTR.end(), eTW[k]),
             eTWEarly[k]);
      for (size_t k = 0; k < eTR.size() && join < 2; ++k) note(std::binary_search(inTW.begin(), inTW.end(), eTR[k]), eTREarly[k]);
      // Compute and copy encoders have no stages to split; a dump serialises everything anyway.
      if (join && compiled.encoders[e].kind != Encoder::Kind::Render) join = 2;
      needJoin[e] = dumping || afterNative ? 2 : join;
      (needJoin[e] ? trackedJoined : trackedFree).fetch_add(1, std::memory_order_relaxed);
      mine.bufW.insert(mine.bufW.end(), eW.begin(), eW.end());
      mine.bufR.insert(mine.bufR.end(), eR.begin(), eR.end());
      mine.texW.insert(mine.texW.end(), eTW.begin(), eTW.end());
      mine.texR.insert(mine.texR.end(), eTR.begin(), eTR.end());
    }
    mergeRanges(mine.bufW), mergeRanges(mine.bufR), uniqueKeys(mine.texW), uniqueKeys(mine.texR);
    mine.frame = frameIndex;
  }
  if (!carriedIn.empty()) {
    if (id<MTLBlitCommandEncoder> jb = [cb blitCommandEncoder]) {
      bump(cnt.encBlit), ++cbEncoders;
      jb.label = @"vpgpu.join";
      for (id<MTLFence> f : carriedIn) [jb waitForFence:f];  // updated by the previous frame (or encodeNative's tail)
      joinFence = nextFence();
      [jb updateFence:joinFence];
      [jb endEncoding];
      // Tracked: this frame's own barriers order only its encoders; the previous tail is the join's.
      if (cbTracked) frontier.clear();
    }
  }
  size_t b = 0;  // cursor into compiled.barriers (sorted by encoder, item)
  for (uint32_t e = 0; e < compiled.encoders.size(); ++e) {
    const Encoder& enc = compiled.encoders[e];
    curEnc = e;
    if (exclOn) (*diag)[e].kind = enc.kind == Encoder::Kind::Render ? 'r' : 'c';
    // The first encoder of a frame always waits: frames are hazard-checked independently.
    bool boundary = e == 0;
    StageMask before = stage::vertex;
    if (b < compiled.barriers.size() && compiled.barriers[b].encoder == e && compiled.barriers[b].item == 0) {
      boundary = true;
      before = compiled.barriers[b++].before;
    }
    if (dumping || fullBarriers) boundary = true, before = stage::vertex;  // serialised: dump copies sit between encoders
    close();
    cutIfFull();  // before this encoder opens (the fences of the closed ones carry the order across the cut)
    // Draw budget: this render encoder would take the command buffer past it, so the buffer goes now
    // (between encoders is always a safe cut; the encoder itself may be cut further below).
    // Cost, not just draw count: a draw on a 4x MSAA two-layer 3840x2485 target costs several plain ones, and
    // attachments the encoder loads or stores add memory traffic. Estimated per encoder (encoderCost), budget
    // `maxDrawsPerCb` draw units; one encoder is never cut here (MSAA stays whole), so an over-budget encoder
    // simply gets a command buffer of its own.
    if (maxDrawsPerCb && !dumping && enc.kind == Encoder::Kind::Render) {
      const double c = encoderCost(enc, cmds);
      if (cbDraws && cbCost + c > maxDrawsPerCb) commitChunk();
      cbCost += c;
    }
    ++cbPasses;

    if (enc.kind == Encoder::Kind::Render) {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      // Encoder::view >= 0 (CompileOptions::splitViews): this encoder draws one layer of the pass: slice `viewOff` of every
      // attachment (and resolve target) as layer 0 of a one-layer render target, with an amplification count of 1.
      const NSUInteger viewOff = enc.view > 0 ? NSUInteger(enc.view) : 0;
      const uint32_t passLayers = enc.view >= 0 ? 1 : std::max<uint32_t>(1, enc.pass.layerCount);
      std::unordered_map<uint64_t, size_t> taken;
      // Tile-local attachments render into a memoryless stand-in: same shape, no memory.
      auto target = [&](const Attachment& src, int slot) -> id<MTLTexture> {
        // memBacked: HeavyGeometry encoder whose samples are not stored (MSAA resolve): real memory, no stand-in.
        if (tileMemory && !memorylessFailed.load(std::memory_order_relaxed) && enc.tileLocal(slot) && !enc.memBacked[slot]) {
          auto it = textures.find(src.texture.index);
          // A deferred target whose memory is only a transient backing goes back to the stand-in. A banned one (its
          // stand-in met a tiler flush) gets real memory.
          if (it != textures.end() && it->second.generation == src.texture.generation && !it->second.tileBanned &&
              (!it->second.texture || (it->second.deferredRT && !it->second.pinned && msaaReleaseFrames))) {
            Texture& tx = it->second;
            if (tx.texture && tx.tileFrame != frameIndex) {
              tx.tileFrame = frameIndex;
              if (tx.usedFrame != frameIndex) ++tx.streak;
            }
            TextureDesc td = tx.desc;
            if (src.format != Format::Undefined && slot != kDepthSlot) td.format = src.format;
            if (id<MTLTexture> ml = memorylessTexture(td, taken, frameIndex)) {
              cbStand->push_back({src.texture.index, src.texture.generation, e, enc.heavy > 0.5f});
              return ml;
            }
          }
        }
        matReason = toString(enc.tile[slot]);  // why an MSAA colour target is allocated here (log only)
        matLoad = unsigned(src.load), matStore = unsigned(src.store);
        if (auto bt = textures.find(src.texture.index); bt != textures.end() && bt->second.deferredRT) {
          bt->second.usedFrame = frameIndex;
          bt->second.streak = 0;
        }
        id<MTLTexture> t = texture(src.texture);
        matReason = nullptr, matLoad = matStore = ~0u;
        return slot != kDepthSlot ? formatViewOf(t, src.format) : t;
      };
      auto attach = [&](MTLRenderPassAttachmentDescriptor* a, const Attachment& src, id<MTLTexture> t) {
        a.texture = t;
        a.level = src.mip;
        if (t.textureType == MTLTextureType3D) a.depthPlane = src.layer;
        else a.slice = src.layer + viewOff;
        a.loadAction = loadAction(src.load);
        a.storeAction = storeAction(src.store);
        if (src.store == StoreAction::Resolve || src.store == StoreAction::StoreAndResolve) {
          // A resolve target of another format than the samples (mutable-format images): seen as
          // the attachment's pixel format (Metal requires them to match).
          a.resolveTexture = src.resolve ? texture(src.resolve) : nil;
          if (a.resolveTexture && t && a.resolveTexture.pixelFormat != t.pixelFormat)
            a.resolveTexture = formatViewOfPixel(a.resolveTexture, t.pixelFormat);
          a.resolveLevel = src.resolveMip;
          a.resolveSlice = src.resolveLayer + viewOff;
          // No resolve target: a resolve store would fail validation. Keep the samples instead
          // (memoryless ones cannot be stored, so they are dropped).
          if (!a.resolveTexture) {
            ++missingResources;
            missing("texture", 0, 0, "resolve_target_nil", src.resolve.index, src.resolve.generation);
            a.storeAction = t.storageMode == MTLStorageModeMemoryless ? MTLStoreActionDontCare : MTLStoreActionStore;
          } else if (t.sampleCount <= 1 || a.resolveTexture.sampleCount > 1) {
            // A resolve from a 1x texture (or into an MSAA one) is undefined behaviour in Metal and hung the GPU on the eye
            // passes: VPVulkan turns those into copies; anything that still gets here keeps its samples, no resolve.
            const NSUInteger dstSamples = a.resolveTexture.sampleCount;
            a.resolveTexture = nil;
            a.storeAction = t.storageMode == MTLStorageModeMemoryless ? MTLStoreActionDontCare : MTLStoreActionStore;
            if ((id)a != (id)rp.stencilAttachment && resolveRefused.fetch_add(1) < 8)
              err("what=resolve_refused f=%llu enc=%u src_samples=%lu dst_samples=%lu format=%lu (a 1x resolve hangs the GPU)",
                  (unsigned long long)frameIndex, e, (unsigned long)t.sampleCount, (unsigned long)dstSamples,
                  (unsigned long)t.pixelFormat);
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
      // ---- Foveation (RenderPassDesc::rateMap): the pass draws into physically smaller shadow attachments
      // through the rate map; a render encoder before it folds what it loads and one after it expands what it
      // stores, so the game's (logical) attachments keep their size. Anything the scheme cannot serve declines
      // (logged once per reason) and the pass renders at full rate as without foveation.
      struct FovAtt {
        id<MTLTexture> logical, shadow;
        bool depth = false, fold = false, expand = false;
      };
      std::vector<FovAtt> fovAtts;
      FovMap* fovM = nullptr;
      // (A view-split encoder (Encoder::view) has none: the frame compiler only splits rate-map passes with multisampled
      // attachments, which are declined below with reason "msaa" anyway.)
      if (enc.pass.rateMap && enc.view < 0) {
        const RateMap& rm = *enc.pass.rateMap;
        std::string why;
        do {
          if (dumping) {
            why = "dump";
            break;
          }
          if (enc.pass.layerCount != 2 || rm.layers != 2) {
            why = "layers";
            break;
          }
          for (uint32_t ci : enc.commands) {
            for (const Draw& d : std::get<RenderPass>(cmds[ci]).draws) {
              const Pipeline* p = d.clear ? nullptr : pipeline(d.pipeline);
              if (d.clear) why = "area_clear";
              else if (!p || !p->render) continue;
              else if (!p->vpIndexed) why = "pipeline_not_view_indexed";
              else if (p->fsPosition && !p->rrmFs) why = "pipeline_fragcoord";
              else if (p->samples > 1) why = "msaa";
              if (!why.empty()) break;
            }
            if (!why.empty()) break;
          }
          if (!why.empty()) break;
          FovMap* m = fovMapFor(rm);
          if (!m) {
            why = "rate_map";
            break;
          }
          std::vector<FovAtt> atts;
          std::unordered_map<uint64_t, size_t> fovTaken;
          std::string perr;
          auto add = [&](MTLRenderPassAttachmentDescriptor* a, const Attachment& src, bool depth) {
            id<MTLTexture> t = a.texture;
            if (!t) {
              why = "attachment_missing";
              return;
            }
            const bool resolves = src.store == StoreAction::Resolve || src.store == StoreAction::StoreAndResolve;
            if (src.mip || src.layer) why = "attachment_slice";
            else if (t.textureType != MTLTextureType2DArray || t.arrayLength < 2) why = "attachment_type";
            else if (t.sampleCount != 1) why = "msaa";
            else if (t.width != rm.width || t.height != rm.height) why = "attachment_size";
            else if (t.storageMode == MTLStorageModeMemoryless) why = "memoryless";
            else if (resolves) why = "resolve";
            else if (src.load == LoadAction::Load && !(t.usage & MTLTextureUsageShaderRead)) why = "load_not_readable";
            else if (!depth && (src.format != Format::Undefined || formatOf(src.texture).integer())) why = "colour_format";
            else if (depth && t.pixelFormat != MTLPixelFormatDepth32Float && t.pixelFormat != MTLPixelFormatDepth16Unorm)
              why = "depth_format";
            if (!why.empty()) return;
            FovAtt f;
            f.logical = t, f.depth = depth;
            f.fold = src.load == LoadAction::Load, f.expand = src.store != StoreAction::DontCare;
            if (f.fold && !fovPipeline((depth ? 2 : 0) + 1, t.pixelFormat, perr)) why = "fold_pipeline";
            else if (f.expand && !fovPipeline(depth ? 2 : 0, t.pixelFormat, perr)) why = "expand_pipeline";
            else if (!(f.shadow = fovShadow(t.pixelFormat, m->physW, m->physH, 2, depth, fovTaken, frameIndex)))
              why = "shadow";
            if (why.empty()) atts.push_back(f);
          };
          for (int i = 0; i < kMaxColorAttachments && why.empty(); ++i)
            if (enc.pass.colors[i].texture) add(rp.colorAttachments[i], enc.pass.colors[i], false);
          if (why.empty() && enc.pass.depth.texture) add(rp.depthAttachment, enc.pass.depth, true);
          if (why.empty() && rp.stencilAttachment.texture) why = "stencil";
          if (!why.empty()) break;
          if (atts.empty()) {
            why = "no_attachments";
            break;
          }
          // Accepted: the pass draws into the shadows (same load/store, slice 0).
          size_t at = 0;
          for (int i = 0; i < kMaxColorAttachments; ++i)
            if (enc.pass.colors[i].texture) rp.colorAttachments[i].texture = atts[at++].shadow;
          if (enc.pass.depth.texture) rp.depthAttachment.texture = atts[at++].shadow;
          rp.rasterizationRateMap = m->map;
          fovM = m;
          fovAtts = std::move(atts);
        } while (false);
        if (!why.empty() && fovWarned.insert(why).second)
          info("gpu.fov decline reason=" + why + " label=" + token(enc.pass.label) + " logical=" + std::to_string(rm.width) +
               "x" + std::to_string(rm.height));
      }
      // (A view-split encoder draws one layer: its vertex function still writes render_target_array_index = 0.)
      if (enc.pass.layerCount > 1) rp.renderTargetArrayLength = enc.view >= 0 ? 1 : enc.pass.layerCount;
      if (!fovM) {
        // Render area: the largest of the passes merged into this encoder (0 = one of them covers the whole
        // target), never past the smallest attachment. A foveated pass renders into the physically smaller
        // shadows through the rate map, whose size is the logical one: left alone.
        uint32_t w = 0, h = 0;
        bool whole = false;
        for (uint32_t ci : enc.commands) {
          const RenderPassDesc& pd = std::get<RenderPass>(cmds[ci]).desc;
          if (!pd.renderWidth || !pd.renderHeight) whole = true;
          w = std::max(w, pd.renderWidth), h = std::max(h, pd.renderHeight);
        }
        NSUInteger minW = NSUIntegerMax, minH = NSUIntegerMax;
        auto clampTo = [&](id<MTLTexture> t, NSUInteger level) {
          if (!t) return;
          minW = std::min<NSUInteger>(minW, std::max<NSUInteger>(1, t.width >> level));
          minH = std::min<NSUInteger>(minH, std::max<NSUInteger>(1, t.height >> level));
        };
        for (int i = 0; i < kMaxColorAttachments; ++i) clampTo(rp.colorAttachments[i].texture, rp.colorAttachments[i].level);
        clampTo(rp.depthAttachment.texture, rp.depthAttachment.level);
        clampTo(rp.stencilAttachment.texture, rp.stencilAttachment.level);
        if (!whole && w && h && minW != NSUIntegerMax && (w < minW || h < minH)) {
          rp.renderTargetWidth = std::min<NSUInteger>(w, minW);
          rp.renderTargetHeight = std::min<NSUInteger>(h, minH);
        }
      }
      NSUInteger s0, s1, sq[4] = {MTLCounterDontSample, MTLCounterDontSample, MTLCounterDontSample, MTLCounterDontSample};
      if (exclOn || busyOn) sampleQuad(e, sq), s0 = sq[0], s1 = sq[3];  // vertex start / end, fragment start / end
      else samplePair(e, s0, s1);
      if (s0 != MTLCounterDontSample) {
        rp.sampleBufferAttachments[0].sampleBuffer = slot.samples;
        rp.sampleBufferAttachments[0].startOfVertexSampleIndex = s0;
        rp.sampleBufferAttachments[0].endOfVertexSampleIndex = sq[1];
        rp.sampleBufferAttachments[0].startOfFragmentSampleIndex = sq[2];
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
      uint64_t visLastOff = 0;  // offset of the last counting draw: disabling stays inside a range this encoder owns
      // Hang split (see InFlight): an encoder of more than kHangSplitMin draws (a shader warmup that
      // draws every pipeline once) runs as chunks of hangSplit draws, each in its own command buffer,
      // the attachments stored and loaded between chunks. Chunk cost with the GPU free: 26-136 ms
      // (AC Nexus warmup, 3840x2485x2 4x MSAA, 256-draw chunks); with the GPU saturated by other work
      // 94-2492 ms. Without the split the whole encoder is one 20 s+ command buffer and the device is
      // lost, so MSAA and big targets split too.
      // Not split: while dumping, and with memoryless attachments. Memoryless (tile-local: store
      // DontCare, kept on tile) attachments cannot be stored, and `target` already picked the stand-in
      // before this point, so a cut would lose their tile contents and a Load of one is invalid: the
      // encoder runs whole (nothing materialises them here: that would need `target` to skip the
      // stand-in for an encoder that is going to be cut, then Store/Load like any other attachment).
      // A memoryless encoder that is also huge can still end in the 20 s loss; gpu.hang split=off
      // reason=memoryless names it.
      uint32_t encDraws = 0;
      for (uint32_t ci : enc.commands) encDraws += uint32_t(std::get<RenderPass>(cmds[ci]).draws.size());
      const bool hangBig = hangSplit && encDraws > kHangSplitMin;
      bool memoryless = false;
      for (int i = 0; i < 8; ++i)
        if (rp.colorAttachments[i].texture.storageMode == MTLStorageModeMemoryless) memoryless = true;
      if (rp.depthAttachment.texture.storageMode == MTLStorageModeMemoryless ||
          rp.stencilAttachment.texture.storageMode == MTLStorageModeMemoryless)
        memoryless = true;
      // Forced MSAA (Attachment::autoMsaa): the samples are never stored, so a split (store + Load between chunks) is never made,
      // even if a banned stand-in gave them real memory.
      bool autoMsaaEnc = enc.pass.depth.autoMsaa;
      for (const Attachment& ca : enc.pass.colors) autoMsaaEnc = autoMsaaEnc || ca.autoMsaa;
      bool canSplit = hangBig && !dumping && !memoryless && !autoMsaaEnc;
      if (enc.view >= 0 && hangBig && memoryless && blitWarned.insert("split_hang_guard").second)
        info("gpu.split hang_guard=off enc_draws=" + std::to_string(encDraws) + " view=" + std::to_string(int(enc.view)) +
             " (a view-split encoder that stays memoryless is not hang-split)");
      encDrawsCut = 0;
      if (hangBig && !canSplit &&
          blitWarned.insert(std::string(dumping ? "hang_split_dump" : autoMsaaEnc ? "hang_split_auto_msaa" : "hang_split_memoryless")).second)
        info("gpu.hang split=off enc_draws=" + std::to_string(encDraws) + " reason=" + (dumping ? "dump" : autoMsaaEnc ? "auto_msaa" : "memoryless") +
             " (the split needs stored attachments)");
      track.recs.reserve(std::min<size_t>(kMaxDrawRecs, track.recs.size() + encDraws));
      MTLStoreAction origColor[8] = {}, origDepth = MTLStoreActionDontCare, origStencil = MTLStoreActionDontCare;
      MTLRenderPassDescriptor* cont = nil;  // descriptor of the continuation chunks
      if (canSplit) {
        for (int i = 0; i < 8; ++i)
          if (rp.colorAttachments[i].texture) {
            origColor[i] = rp.colorAttachments[i].storeAction;
            rp.colorAttachments[i].storeAction = MTLStoreActionUnknown;
          }
        if (rp.depthAttachment.texture) {
          origDepth = rp.depthAttachment.storeAction;
          rp.depthAttachment.storeAction = MTLStoreActionUnknown;
        }
        if (rp.stencilAttachment.texture) {
          origStencil = rp.stencilAttachment.storeAction;
          rp.stencilAttachment.storeAction = MTLStoreActionUnknown;
        }
      }
      // Render target of the pass, for the per-draw hang records.
      uint32_t rtW = enc.pass.width, rtH = enc.pass.height;
      uint8_t rtSamples = uint8_t(std::max<NSUInteger>(1, rp.defaultRasterSampleCount));
      {
        MTLRenderPassAttachmentDescriptor* a0 = rp.colorAttachments[0].texture ? (MTLRenderPassAttachmentDescriptor*)rp.colorAttachments[0]
                                                                               : (MTLRenderPassAttachmentDescriptor*)rp.depthAttachment;
        if (id<MTLTexture> t = a0.texture) {
          rtW = uint32_t(std::max<NSUInteger>(1, t.width >> a0.level));
          rtH = uint32_t(std::max<NSUInteger>(1, t.height >> a0.level));
          rtSamples = uint8_t(t.sampleCount);
        }
        if (fovM) rtW = enc.pass.rateMap->width, rtH = enc.pass.rateMap->height;  // logical, as the draws see it
        // A render target reduced to the render area: the default viewport / scissor must stay inside it.
        else if (rp.renderTargetWidth && rp.renderTargetHeight)
          rtW = std::min<uint32_t>(rtW, uint32_t(rp.renderTargetWidth)), rtH = std::min<uint32_t>(rtH, uint32_t(rp.renderTargetHeight));
      }
      // Draws per chunk: hangSplit as is (256). Measured on AC Nexus: 13 chunks of 256 draws, 0.6 s in
      // total. Smaller chunks for MSAA/big targets would mean many more Store/Load cuts of ~1 GB:
      // unmeasured, so not done.
      const uint32_t splitLimit = hangSplit;
      if (canSplit && splitLogLines.fetch_add(1) < 32)
        info("gpu.hang split=on enc_draws=" + std::to_string(encDraws) + " chunk=" + std::to_string(splitLimit) +
             " chunks=" + std::to_string((encDraws + splitLimit - 1) / splitLimit) + " rt=" + std::to_string(rtW) + "x" +
             std::to_string(rtH) + " layers=" + std::to_string(enc.pass.layerCount) + " samples=" + std::to_string(rtSamples));
      // Fold (logical -> shadow) and expand (shadow -> logical) of a foveated pass: one fullscreen triangle per view.
      auto fovPass = [&](bool expandPass, const FovAtt& a) {
        close();
        std::string perr;
        id<MTLTexture> dst = expandPass ? a.logical : a.shadow;
        id<MTLRenderPipelineState> ps = fovPipeline((a.depth ? 2 : 0) + (expandPass ? 0 : 1), dst.pixelFormat, perr);
        MTLRenderPassDescriptor* d = [MTLRenderPassDescriptor renderPassDescriptor];
        MTLRenderPassAttachmentDescriptor* ad = a.depth ? (MTLRenderPassAttachmentDescriptor*)d.depthAttachment
                                                        : (MTLRenderPassAttachmentDescriptor*)d.colorAttachments[0];
        ad.texture = dst;
        ad.loadAction = MTLLoadActionDontCare;  // the triangle covers every pixel of every layer
        ad.storeAction = MTLStoreActionStore;
        d.renderTargetArrayLength = 2;
        if (a.depth && !fovDepthState) {
          MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
          ds.depthCompareFunction = MTLCompareFunctionAlways;
          ds.depthWriteEnabled = YES;
          fovDepthState = [device newDepthStencilStateWithDescriptor:ds];
        }
        open.kind = Open::Render;
        open.render = ps ? [cb renderCommandEncoderWithDescriptor:d] : nil;
        bump(cnt.encRender), ++cbEncoders;
        if (!open.render) {
          err("what=fov_pass f=%llu enc=%u expand=%d", (unsigned long long)frameIndex, e, int(expandPass));
          open = {};
          return;
        }
        opened(expandPass ? @"fov.expand" : @"fov.fold", true, stage::vertex);
        [open.render setRenderPipelineState:ps];
        if (a.depth) [open.render setDepthStencilState:fovDepthState];
        [open.render setViewport:(MTLViewport){0, 0, double(dst.width), double(dst.height), 0.0, 1.0}];
        [open.render setFragmentTexture:expandPass ? a.shadow : a.logical atIndex:0];
        [open.render setFragmentBuffer:fovM->data offset:0 atIndex:0];
        [open.render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:2];
        ++cbPasses;
        close();
      };
      if (fovM) {
        for (const FovAtt& a : fovAtts)
          if (a.fold) fovPass(false, a);
        if (fovEncoders++ < 8 || fovEncoders % 600 == 0) {
          const RateMap& rm = *enc.pass.rateMap;
          char line[256];
          std::snprintf(line, sizeof line,
                        "gpu.fov enc f=%llu logical=%ux%u physical=%lux%lu fragments=%.1f%% attachments=%zu fold=%d expand=%d",
                        (unsigned long long)frameIndex, rm.width, rm.height, (unsigned long)fovM->physW,
                        (unsigned long)fovM->physH,
                        100.0 * double(fovM->physW) * double(fovM->physH) / (double(rm.width) * double(rm.height)),
                        fovAtts.size(), int(std::count_if(fovAtts.begin(), fovAtts.end(), [](const FovAtt& a) { return a.fold; })),
                        int(std::count_if(fovAtts.begin(), fovAtts.end(), [](const FovAtt& a) { return a.expand; })));
          info(line);
        }
      }
      // Viewport and scissor as the draws last set them (logical space when foveated), kept to write one per view.
      MTLViewport curVp = {0, 0, double(rtW), double(rtH), 0.0, 1.0};
      MTLScissorRect curSc = {0, 0, rtW, rtH};
      bool vpDirty = false, scDirty = false;
      // Last per-draw state emitted on the open encoder (VP_STATE_ELIDE): a draw re-emits only what differs. A fresh
      // EncState is "unknown": reset it wherever the encoder is opened anew (splitChunk) or something else sets state on it
      // (clear quads). Strong refs: an address reused by a new object cannot alias a stale entry.
      struct EncState {
        bool known = false;  // everything but the blend colour holds what the encoder has
        id<MTLRenderPipelineState> ps = nil;
        id<MTLDepthStencilState> dss = nil;
        MTLCullMode cull = MTLCullModeNone;
        uint8_t stencilFront = 0, stencilBack = 0;
        MTLWinding winding = MTLWindingClockwise;
        float bias = 0, slope = 0, clamp = 0;
        MTLDepthClipMode clip = MTLDepthClipModeClip;
        MTLTriangleFillMode fill = MTLTriangleFillModeFill;
        bool blendKnown = false;
        float blend[4] = {0, 0, 0, 0};
      } st;
      constexpr uint32_t kMaxPassViews = 16;
      const uint32_t passViews = std::clamp<uint32_t>(enc.pass.layerCount, 2, kMaxPassViews);
      // Ends the chunk's encoder: intermediate chunks store everything, the last one does what
      // the pass asked for.
      auto setStores = [&](bool last) {
        for (int i = 0; i < 8; ++i)
          if (rp.colorAttachments[i].texture)
            [open.render setColorStoreAction:last ? origColor[i] : MTLStoreActionStore atIndex:i];
        if (rp.depthAttachment.texture) [open.render setDepthStoreAction:last ? origDepth : MTLStoreActionStore];
        if (rp.stencilAttachment.texture) [open.render setStencilStoreAction:last ? origStencil : MTLStoreActionStore];
      };
      bool encoderGone = false;
      NSUInteger curAmp = 1;  // the encoder's vertex amplification count (a new encoder starts at 1)
      bool viewBaseBound = false;  // the encoder has the view base of its amplified vertex functions (buffer 27) bound
      auto splitChunk = [&]() {
        ++splitCount;
        setStores(false);
        close();
        st = EncState{};  // the continuation encoder starts from Metal's defaults
        commitChunk();
        cbPasses = 1;  // the continuation encoder
        encDrawsCut = 0;
        if (!cont) {  // made after the first encoder: `rp` (its clears) is untouched by the copy's edits
          cont = [rp copy];
          for (int i = 0; i < 8; ++i)
            if (cont.colorAttachments[i].texture) cont.colorAttachments[i].loadAction = MTLLoadActionLoad;
          if (cont.depthAttachment.texture) cont.depthAttachment.loadAction = MTLLoadActionLoad;
          if (cont.stencilAttachment.texture) cont.stencilAttachment.loadAction = MTLLoadActionLoad;
          cont.sampleBufferAttachments[0].sampleBuffer = nil;
        }
        open.kind = Open::Render;
        open.render = [cb renderCommandEncoderWithDescriptor:cont];
        bump(cnt.encRender), ++cbEncoders;
        if (!open.render) {
          err("what=render_encoder f=%llu enc=%u chunk=%u", (unsigned long long)frameIndex, e, chunk);
          open = {};
          encoderGone = true;
          return;
        }
        opened([NSString stringWithFormat:@"render%u.c%u", e, chunk], true, stage::vertex);
        visCounting = false;
        curAmp = 1;
        viewBaseBound = false;
      };
      open.kind = Open::Render;
      open.render = [cb renderCommandEncoderWithDescriptor:rp];
      bump(cnt.encRender), ++cbEncoders;
      if (!open.render) {
        err("what=render_encoder f=%llu enc=%u", (unsigned long long)frameIndex, e);
        open = {};
        continue;
      }
      opened([NSString stringWithFormat:@"render%u", e], boundary || fovM, fovM ? stage::vertex : before);
      if (exclOn) {
        EncDiag& g = (*diag)[e];
        g.width = rtW, g.height = rtH, g.layers = passLayers, g.samples = rtSamples, g.view = enc.view;
        if (!enc.commands.empty()) g.label = std::get<RenderPass>(cmds[enc.commands[0]]).desc.label;
        if (rp.colorAttachments[0].texture) g.color0 = uint32_t(rp.colorAttachments[0].texture.pixelFormat);
        if (rp.depthAttachment.texture) g.depth = uint32_t(rp.depthAttachment.texture.pixelFormat);
        else if (rp.stencilAttachment.texture) g.depth = uint32_t(rp.stencilAttachment.texture.pixelFormat);
        // What the load/store actions move (logical size: the render area x views x samples).
        auto traffic = [&](MTLRenderPassAttachmentDescriptor* a, const Attachment& src, const std::string& name, int tileSlot) {
          id<MTLTexture> t = a.texture;
          if (!t) return;
          auto it = textures.find(src.texture.index);
          const Format f = src.format != Format::Undefined ? src.format
                           : it != textures.end()          ? it->second.desc.format
                                                           : Format::Undefined;
          const double bpp = f != Format::Undefined ? double(formatInfo(f).bytesPerBlock) : 4.0;
          const double mb = double(rtW) * double(rtH) * double(g.layers) * double(t.sampleCount) * bpp / 1e6;
          const bool memoryless = t.storageMode == MTLStorageModeMemoryless;
          const bool stores = a.storeAction == MTLStoreActionStore || a.storeAction == MTLStoreActionStoreAndMultisampleResolve ||
                              a.storeAction == MTLStoreActionUnknown;  // unknown: a hang-split encoder stores (setStores)
          const bool resolves = a.storeAction == MTLStoreActionMultisampleResolve ||
                                a.storeAction == MTLStoreActionStoreAndMultisampleResolve;
          if (a.loadAction == MTLLoadActionLoad && !memoryless) g.loadMb += mb;
          if (stores && !memoryless) g.storeMb += mb;
          if (resolves && a.resolveTexture) g.storeMb += mb / double(std::max<NSUInteger>(1, t.sampleCount));
          g.attach += (g.attach.empty() ? "" : ",") + name + ":" +
                      (a.loadAction == MTLLoadActionLoad ? "L" : a.loadAction == MTLLoadActionClear ? "C" : "D") +
                      (stores && resolves ? "SR" : stores ? "S" : resolves ? "R" : "D") + (memoryless ? "m" : "");
          const TileVerdict v = enc.tile[tileSlot];
          if (v != TileVerdict::None && v != TileVerdict::TileLocal) g.attach += std::string(":") + toString(v);
        };
        for (int i = 0; i < kMaxColorAttachments; ++i)
          if (enc.pass.colors[i].texture) traffic(rp.colorAttachments[i], enc.pass.colors[i], "c" + std::to_string(i), i);
        if (enc.pass.depth.texture) {
          if (rp.depthAttachment.texture) traffic(rp.depthAttachment, enc.pass.depth, "d", kDepthSlot);
          else traffic(rp.stencilAttachment, enc.pass.depth, "s", kDepthSlot);
        }
      }
      // VP_KNOCK: this encoder's signature; while measuring, whether it is one of the dominant encoders (whose draws of the
      // knocked pair are left out). Warm-up: the draws per pair of this encoder, merged after its draws.
      uint64_t knockEncSig = 0;
      bool knockHere = false;
      std::map<KnockKey, KnockPair> knockLocal;
      if (knockFrame.mode) {
        EncDiag& g = (*diag)[e];
        g.sig = 1469598103934665603ull;
        for (uint64_t v : {uint64_t(g.color0), uint64_t(g.depth), uint64_t(g.width) << 20 | g.height,
                           uint64_t(g.layers) << 8 | g.samples})
          g.sig = (g.sig ^ v) * 1099511628211ull;
        knockEncSig = g.sig;
        knockHere = knockFrame.mode == 2 && g.sig == knockFrame.sig;
        if (knockHere) ++knockSeen;
      }
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
        NSUInteger level = src.mip, slice = src.layer + viewOff;
        // MSAA: the resolved image, when the pass resolves.
        if (t && t.sampleCount > 1 && c0.texture && rp.colorAttachments[0].resolveTexture) {
          t = rp.colorAttachments[0].resolveTexture;
          level = c0.resolveMip, slice = c0.resolveLayer + viewOff;
        }
        dumpQueue(std::move(it), t, level, slice);
      }
      // Forced MSAA: autoMsaa slots that continue a picture (Attachment::needsSeed) start from their 1x partner.
      if (open.render && !fovM) {
        const Attachment& sc = enc.pass.colors[0];
        const Attachment& sd = enc.pass.depth;
        const bool wantC = sc.texture && sc.needsSeed, wantD = sd.texture && sd.needsSeed;
        bool otherSeed = false;
        for (int i = 1; i < kMaxColorAttachments; ++i) otherSeed = otherSeed || (enc.pass.colors[i].texture && enc.pass.colors[i].needsSeed);
        if (otherSeed && blitWarned.insert("seed_other_slot").second)
          err("what=seed_unsupported f=%llu enc=%u (only colour 0 and depth are seeded)", (unsigned long long)frameIndex, e);
        if (wantC || wantD) {
          // A 2D-array view of the 1x source: its mip, the pass's layers starting at the resolve layer (+ the view of a split pass).
          auto sourceView = [&](const Attachment& a, id<MTLTexture> target) -> id<MTLTexture> {
            id<MTLTexture> t = a.seedSrc ? texture(a.seedSrc) : nil;
            if (!t || !target || t.sampleCount != 1 || !(t.usage & MTLTextureUsageShaderRead) || a.seedMip >= t.mipmapLevelCount) return nil;
            if (t.textureType != MTLTextureType2D && t.textureType != MTLTextureType2DArray) return nil;
            const NSUInteger first = NSUInteger(a.seedLayer) + viewOff;
            const NSUInteger have = t.textureType == MTLTextureType2DArray ? t.arrayLength : 1;
            if (first + passLayers > have) return nil;
            return [t newTextureViewWithPixelFormat:target.pixelFormat
                                        textureType:MTLTextureType2DArray
                                             levels:NSMakeRange(a.seedMip, 1)
                                             slices:NSMakeRange(first, passLayers)];
          };
          id<MTLTexture> vc = wantC ? sourceView(sc, rp.colorAttachments[0].texture) : nil;
          id<MTLTexture> vd = wantD ? sourceView(sd, rp.depthAttachment.texture) : nil;
          const bool useC = wantC && vc, useD = wantD && vd;
          if ((wantC && !vc) || (wantD && !vd)) {
            if (blitWarned.insert("seed_source").second)
              err("what=seed_source f=%llu enc=%u colour=%d depth=%d (1x partner missing, not readable or too small: samples start undefined)",
                  (unsigned long long)frameIndex, e, int(wantC && !vc), int(wantD && !vd));
          }
          std::string perr;
          id<MTLRenderPipelineState> ps = useC || useD ? seedPipeline(rp, passLayers > 1, useC, useD, perr) : nil;
          if ((useC || useD) && !ps) ++missingPipelines;
          if ((useC || useD) && !ps && blitWarned.insert("seed_pipeline").second)
            err("what=seed_pipeline f=%llu enc=%u error=%s", (unsigned long long)frameIndex, e, perr.c_str());
          if (ps) {
            __strong id<MTLDepthStencilState>& dss = useD ? seedDepthOn : seedDepthOff;
            if (!dss) {
              MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
              ds.depthCompareFunction = MTLCompareFunctionAlways;
              ds.depthWriteEnabled = useD ? YES : NO;
              dss = [device newDepthStencilStateWithDescriptor:ds];
            }
            [open.render setRenderPipelineState:ps];
            if (rp.depthAttachment.texture || rp.stencilAttachment.texture) [open.render setDepthStencilState:dss];
            [open.render setCullMode:MTLCullModeNone];
            [open.render setViewport:(MTLViewport){0, 0, double(rtW), double(rtH), 0.0, 1.0}];
            if (useC) [open.render setFragmentTexture:vc atIndex:0];
            if (useD) [open.render setFragmentTexture:vd atIndex:1];
            [open.render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:passLayers];
            st = EncState{};  // the draws that follow set everything they need again
            vpDirty = scDirty = true;
            seedDrawsDone.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
      uint32_t drawIx = 0;  // draws of the encoder's commands in order: a chunk of a split encoder runs [drawBegin, drawEnd)
      for (uint32_t ci : enc.commands) {
        if (encoderGone) break;
        const auto& draws = std::get<RenderPass>(cmds[ci]).draws;
        for (uint32_t j = 0; j < draws.size(); ++j) {
          const uint32_t ix = drawIx++;
          if (ix < enc.drawBegin || ix >= enc.drawEnd) continue;
          const Draw& d = draws[j];
          if (canSplit && encDrawsCut >= splitLimit) {
            splitChunk();
            if (encoderGone) break;
          }
          // Clears never count; a draw counts when it carries this encoder's visibility buffer.
          const bool counts = visBuf && !d.clear && d.visibility && d.visibility->buffer == visId;
          if (visBuf && (counts || visCounting)) {
            const uint64_t off = counts ? visBase + d.visibility->offset : visLastOff;
            if (counts && (off & 7)) {
              if (blitWarned.insert("visibility_align").second) err("what=visibility_unaligned off=%llu", (unsigned long long)off);
            } else {
              [open.render setVisibilityResultMode:counts ? MTLVisibilityResultModeCounting : MTLVisibilityResultModeDisabled
                                            offset:NSUInteger(off)];
              visCounting = counts;
              visLastOff = off;
            }
          }
          if (visBuf && !d.clear && d.visibility && !(d.visibility->buffer == visId) &&
              blitWarned.insert("visibility_buffers").second)
            err("what=visibility_second_buffer: draws of one encoder count into two buffers (only the first counts)");
          if (d.clear) {
            if (curAmp > 1) {  // the clear's pipelines are not amplified
              [open.render setVertexAmplificationCount:1 viewMappings:nil];
              curAmp = 1;
            }
            encodeClear(*d.clear, rp, enc.pass, enc.view);
            st = EncState{};  // the clear quads set their own pipeline, depth-stencil, cull, bias, clip, fill and stencil ref
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
          // An amplified pipeline maps view v to layer v: only in a pass with that many layers.
          if (p->amp > enc.pass.layerCount) {
            skip(ci, j, "amplified_in_unlayered_pass");
            continue;
          }
          if (enc.view >= 0 && p->amp < 2) {  // the frame compiler only splits passes of amplified draws
            skip(ci, j, "view_split_not_amplified");
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
          trackPipeline(e, p, false);
          ++cbDraws;
          ++encDrawsCut;
          if (exclOn) ++(*diag)[e].draws;
          size_t recIdx = SIZE_MAX;
          if (track.recs.size() < kMaxDrawRecs) {
            DrawRec r;
            r.encoder = e, r.chunk = chunk, r.vsId = p->vsId, r.fsId = p->fsId;
            r.count = d.count, r.instances = d.instances, r.first = d.first, r.baseVertex = d.baseVertex;
            r.rtWidth = rtW, r.rtHeight = rtH, r.layers = uint16_t(passLayers), r.samples = rtSamples;
            r.kind = uint8_t((d.index ? 1 : 0) | (d.indirect ? 2 : 0));
            r.blend = p->blendMask, r.depthTest = p->depthTest, r.depthWrite = p->depthTest && p->depthWrite;
            track.recs.push_back(r);
            recIdx = track.recs.size() - 1;
          }
          // Vertex amplification: the vertex shader writes layer / viewport index 0; the view mappings add the view (amplification
          // id) to the layer, and to the viewport index when the shader has that output (foveation: one viewport per view and
          // the rate map's layer follows it). The encoder's count never exceeds the bound pipeline's
          // maxVertexAmplificationCount: raising it binds the amplified pipeline first, lowering it drops the count before the
          // plain pipeline is bound.
          // A view-split encoder (Encoder::view) draws its one view with the count left at 1: the pipeline is the same one.
          const NSUInteger drawAmp = enc.view >= 0 ? 1 : p->amp;
          const bool elide = stateElide && st.known;  // else: emit everything
          const bool ampChange = drawAmp != curAmp;   // rebind the pipeline around a count change, as before
          if (drawAmp < curAmp) {
            [open.render setVertexAmplificationCount:1 viewMappings:nil];
            curAmp = 1;
          }
          id<MTLRenderPipelineState> ps = p->render;
          if (knockHere && knockFrame.fsr && p->vsId == knockFrame.vs && p->fsId == knockFrame.fs) {
            // VP_KNOCK_FSR: the same pipeline with a trivial fragment function (the draw and every other state are untouched);
            // one whose substitute is not ready draws as it is.
            if (id<MTLRenderPipelineState> sub = knockFsrPso(d.pipeline.value)) ps = sub, ++knockSkipped;
          }
          if (!elide || ampChange || st.ps != ps) {
            [open.render setRenderPipelineState:ps];
            st.ps = ps;
          }
          // The amplified vertex function adds the encoder's view base to [[amplification_id]] (buffer 27, vpmsl kViewBaseDecl): 0,
          // or the view a split encoder draws (the one amplification id is 0). Bound once per encoder.
          if (p->amp > 1 && !viewBaseBound) {
            const uint32_t viewBase = uint32_t(viewOff);
            [open.render setVertexBytes:&viewBase length:sizeof viewBase atIndex:27];
            if (enc.view >= 0) [open.render setVertexAmplificationCount:1 viewMappings:nil];  // explicit: one view, layer 0
            viewBaseBound = true;
          }
          if (drawAmp > curAmp) {
            MTLVertexAmplificationViewMapping maps[2];
            for (uint32_t v = 0; v < 2; ++v) maps[v].viewportArrayIndexOffset = p->vpIndexed ? v : 0, maps[v].renderTargetArrayIndexOffset = v;
            [open.render setVertexAmplificationCount:drawAmp viewMappings:maps];
            curAmp = drawAmp;
          }
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
          if (!elide || st.dss != dss) {
            [open.render setDepthStencilState:dss];
            st.dss = dss;
          }
          if (!elide || st.cull != p->cull) {
            [open.render setCullMode:p->cull];
            st.cull = p->cull;
          }
          const uint8_t stencilBack = d.stencilRefBack >= 0 ? uint8_t(d.stencilRefBack) : d.stencilRef;
          if (!elide || st.stencilFront != d.stencilRef || st.stencilBack != stencilBack) {
            [open.render setStencilFrontReferenceValue:d.stencilRef backReferenceValue:stencilBack];
            st.stencilFront = d.stencilRef, st.stencilBack = stencilBack;
          }
          const MTLWinding winding = d.frontFace == Winding::Clockwise ? MTLWindingClockwise : MTLWindingCounterClockwise;
          if (!elide || st.winding != winding) {
            [open.render setFrontFacingWinding:winding];
            st.winding = winding;
          }
          if (!elide || st.bias != d.depthBias || st.slope != d.depthBiasSlope || st.clamp != d.depthBiasClamp) {
            [open.render setDepthBias:d.depthBias slopeScale:d.depthBiasSlope clamp:d.depthBiasClamp];
            st.bias = d.depthBias, st.slope = d.depthBiasSlope, st.clamp = d.depthBiasClamp;
          }
          const MTLDepthClipMode clip = d.depthClamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
          if (!elide || st.clip != clip) {
            [open.render setDepthClipMode:clip];
            st.clip = clip;
          }
          const MTLTriangleFillMode fill = d.wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
          if (!elide || st.fill != fill) {
            [open.render setTriangleFillMode:fill];
            st.fill = fill;
          }
          st.known = true;
          if (d.viewport.width > 0) {
            curVp = (MTLViewport){d.viewport.x, d.viewport.y, d.viewport.width, d.viewport.height, d.viewport.znear, d.viewport.zfar};
            vpDirty = true;
          }
          if (d.scissor.width > 0 && d.scissor.height > 0) {
            curSc = (MTLScissorRect){d.scissor.x, d.scissor.y, d.scissor.width, d.scissor.height};
            scDirty = true;
          }
          if (p->vpIndexed) {
            // The vertex shader writes viewport_array_index = view: one viewport and one scissor per view.
            MTLViewport vps[kMaxPassViews];
            MTLScissorRect scs[kMaxPassViews];
            for (uint32_t k = 0; k < passViews; ++k) vps[k] = curVp, scs[k] = curSc;
            [open.render setViewports:vps count:passViews];
            [open.render setScissorRects:scs count:passViews];
          } else {
            if (vpDirty) [open.render setViewport:curVp];
            if (scDirty) [open.render setScissorRect:curSc];
          }
          vpDirty = scDirty = false;
          if (recIdx != SIZE_MAX) {  // viewport/scissor in force for this draw
            DrawRec& r = track.recs[recIdx];
            r.vpX = float(curVp.originX), r.vpY = float(curVp.originY), r.vpW = float(curVp.width), r.vpH = float(curVp.height);
            r.scX = uint32_t(curSc.x), r.scY = uint32_t(curSc.y), r.scW = uint32_t(curSc.width), r.scH = uint32_t(curSc.height);
          }
          if (p->blendConstant &&
              (!stateElide || !st.blendKnown || std::memcmp(st.blend, d.blendColor.data(), sizeof st.blend) != 0)) {
            [open.render setBlendColorRed:d.blendColor[0] green:d.blendColor[1] blue:d.blendColor[2] alpha:d.blendColor[3]];
            std::memcpy(st.blend, d.blendColor.data(), sizeof st.blend);
            st.blendKnown = true;
          }
          if (knockFrame.mode) {  // VP_KNOCK
            if (knockFrame.mode == 1) {
              KnockPair& kp = knockLocal[{p->vsId, p->fsId}];
              if (!kp.draws++) kp.label = p->label, kp.traits = p->traits;
              kp.verts += drawViewVertices(d, passLayers);
              if (knockFsrOn) kp.pipes.insert(d.pipeline.value);
            } else if (knockHere && knockFrame.knock && p->vsId == knockFrame.vs && p->fsId == knockFrame.fs) {
              // Only the draw call goes: pipeline, depth-stencil, viewport and every other state above is bound (and in `st` /
              // curVp), so the draws after it find the encoder as an unskipped run would.
              ++knockSkipped;
              continue;
            }
          }
          if (vt >= 0) [open.render setVertexBuffer:vtBuf offset:vt atIndex:p->stage[0].index];
          if (p->drawParams) {
            // Indexed draws (direct or indirect): Metal's [[base_vertex]] is Vulkan's vertexOffset.
            // Non-indexed direct draws: firstVertex. Non-indexed GPU-indirect draws: Metal's
            // [[base_vertex]] (firstVertex lives in the GPU buffer; not read back here).
            const uint32_t prm[3] = {d.drawIndex, d.first, (d.index || d.indirect) ? 1u : 0u};
            [open.render setVertexBytes:prm length:sizeof prm atIndex:29];
          }
          if (ft >= 0) [open.render setFragmentBuffer:ftBuf offset:ft atIndex:p->stage[1].index];
          if (p->rrmFs) {  // FragCoord back to screen space: the pass's rate map data, or the identity outside one
            if (id<MTLBuffer> rb = fovM ? fovM->data : fovIdentityData()) [open.render setFragmentBuffer:rb offset:0 atIndex:28];
          }
          // Indirect: the GPU reads {count, instances, first, (baseVertex,) baseInstance} — Vulkan's
          // and Metal's layouts are the same.
          id<MTLBuffer> argBuf = nil;
          uint64_t argOff = 0;
          if (d.indirect) {
            argBuf = buffer(BufferSpan{d.indirect->buffer, d.indirect->offset, 0, 0, 0}, argOff);
            if (!argBuf || (argOff & 3)) {
              ++missingResources;
              missing("buffer", 0, 0, !argBuf ? "indirect_buffer_nil" : "indirect_offset_unaligned", d.indirect->buffer.index,
                      d.indirect->buffer.generation);
              continue;
            }
          }
          if (d.index) {
            uint64_t off = 0;
            BufferSpan span{d.index->buffer, d.index->offset, 0, 0, 0};
            id<MTLBuffer> ib = buffer(span, off);
            if (!ib) {
              ++missingResources;
              missing("buffer", 0, 0, "index_buffer_nil", d.index->buffer.index, d.index->buffer.generation);
              continue;
            }
            const uint64_t stride = d.index->wide ? 4 : 2;
            if (argBuf) {
              if (off % stride) {  // Metal: the index buffer offset is a multiple of the index size
                ++missingResources;
                missing("buffer", 0, 0, "index_offset_unaligned", d.index->buffer.index, d.index->buffer.generation);
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
      if (knockFrame.mode == 1) {  // only the encoders that make the frame count (a loading screen or menu has none)
        uint32_t knockEncDraws = 0;
        for (const auto& kv : knockLocal) knockEncDraws += kv.second.draws;
        if (knockEncDraws >= kKnockWarmDraws && ((*diag)[e].samples > 1 || knockRelaxed)) {
          (*diag)[e].knockCounted = true;
          knockMerge(knockEncSig, frameIndex, knockLocal);
          knockWarmSeen = true;
        }
      }
      if (canSplit && open.render) setStores(true);
      if (fovM && !encoderGone) {
        for (const FovAtt& a : fovAtts)
          if (a.expand) fovPass(true, a);
      }
      continue;
    }

    // Ordering assumption (CompileOptions::copyRects): every texture and buffer here is MTLHazardTrackingModeUntracked,
    // so Metal orders nothing between commands, and a blit encoder has no in-encoder barrier: the only way to order two
    // dependent copies is a fence between encoders (what a placed barrier becomes below). Two copies the frame compiler
    // leaves unordered may therefore overlap on the GPU. That is harmless for disjoint memory but NOT proven safe for
    // disjoint rects of one texture (lossless-compression metadata / partial-tile read-modify-write), hence copyRects is
    // off unless VPVK_COPY_RECT_HAZARD=1.
    // Compute encoder: dispatches and copies in order, barriers between items where placed.
    // A wait of a sampled blit that encoded nothing passes to the next item (its barrier may
    // cover that item's hazards too).
    bool carryWait = false;
    StageMask carryBefore = 0;
    for (uint32_t k = 0; k < enc.commands.size(); ++k) {
      const Command& cmd = cmds[enc.commands[k]];
      const bool isDispatch = std::holds_alternative<Dispatch>(cmd);
      bool wait = (k == 0 && boundary) || carryWait || (fullBarriers && k > 0);
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
        cutIfFull();  // a barrier inside a copy chain: the chain's encoders are cut between, fences keep their order
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
          bump(cnt.encCompute), ++cbEncoders;
        } else {
          MTLBlitPassDescriptor* bp = [MTLBlitPassDescriptor blitPassDescriptor];
          if (s0 != MTLCounterDontSample) {
            bp.sampleBufferAttachments[0].sampleBuffer = slot.samples;
            bp.sampleBufferAttachments[0].startOfEncoderSampleIndex = s0;
            bp.sampleBufferAttachments[0].endOfEncoderSampleIndex = s1;
          }
          open.kind = Open::Blit;
          open.blit = [cb blitCommandEncoderWithDescriptor:bp];
          bump(cnt.encBlit), ++cbEncoders;
        }
        opened([NSString stringWithFormat:@"%@%u.%u", isDispatch ? @"compute" : @"blit", e, k], wait, waitBefore);
      }

      if (isDispatch) {
        const Dispatch& d = std::get<Dispatch>(cmd);
        const Pipeline* p = pipeline(d.pipeline);
        if (!p || !p->compute) {
          ++missingPipelines;
          // An engine-internal kernel (the indirect-argument patch): Device::submit waited for it, so this is a failed compile.
          if (d.required && blitWarned.insert("required_pipeline").second)
            err("what=required_pipeline_missing f=%llu label=%s (draws that read its output draw nothing)",
                (unsigned long long)frameIndex, token(d.label).c_str());
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
        trackPipeline(e, p, true);
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
      // RGB9E5: Metal decodes and re-encodes (canonicalising) the texels a blit moves, but Vulkan copies are raw bits
      // (the CTS compares them bitwise): they go through an R32Uint view of the same memory.
      {
        auto rawBits = [&](id<MTLTexture> t) { return t && t.pixelFormat == MTLPixelFormatRGB9E5Float ? formatViewOfPixel(t, MTLPixelFormatR32Uint) : t; };
        st = rawBits(st);
        dt = rawBits(dt);
      }
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
        if (b) {
          attr::track("m2.d24staging", b);
          bump(cnt.bufInternal);
          bump(cnt.bufInternalBytes, bytes);
          bump(bufIntLiveBytes, bytes);
          cbIntBytes += bytes;
        }
        if (!b && blitWarned.insert("d24").second)
          err("what=d24_copy_unconverted reason=%s %ux%u", p ? "no_memory" : token(why).c_str(), t.width, t.height);
        return b;
      };
      auto d24Convert = [&](id<MTLComputePipelineState> p, id<MTLBuffer> src, NSUInteger srcOff, id<MTLBuffer> dst,
                            NSUInteger dstOff, const D24Params& prm) {
        close();
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        bump(cnt.encCompute), ++cbEncoders;
        ce.label = @"vpgpu.d24";
        for (id<MTLFence> f : frontier) [ce waitForFence:f];
        if (joinFence && (curEnc >= needJoin.size() || needJoin[curEnc])) [ce waitForFence:joinFence];
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
        bump(cnt.encBlit), ++cbEncoders;
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
            attr::track("m2.copystage", stage);
            bump(cnt.bufInternal);
            bump(cnt.bufInternalBytes, bytes);
            bump(bufIntLiveBytes, bytes);
            cbIntBytes += bytes;
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
      {  // the `break` of a case above: which side of the copy had no Metal resource (gpu.missing)
        const CopyKind ck = c.kind;
        const bool needSrcBuf = ck == CopyKind::BufferToBuffer || ck == CopyKind::BufferToTexture;
        const bool needDstBuf = ck == CopyKind::BufferToBuffer || ck == CopyKind::FillBuffer || ck == CopyKind::TextureToBuffer;
        const bool needSrcTex = ck == CopyKind::TextureToTexture || ck == CopyKind::TextureToBuffer;
        const bool needDstTex = ck == CopyKind::TextureToTexture || ck == CopyKind::BufferToTexture;
        if (needSrcBuf && !sb) missing("buffer", 0, 0, "copy_src_buffer_nil", c.srcBuffer.buffer.index, c.srcBuffer.buffer.generation);
        else if (needDstBuf && !db) missing("buffer", 0, 0, "copy_dst_buffer_nil", c.dstBuffer.buffer.index, c.dstBuffer.buffer.generation);
        else if (needSrcTex && !st) missing("texture", 0, 0, "copy_src_texture_nil", c.srcTexture.texture.index, c.srcTexture.texture.generation);
        else if (needDstTex && !dt) missing("texture", 0, 0, "copy_dst_texture_nil", c.dstTexture.texture.index, c.dstTexture.texture.generation);
        else missing("copy", 0, 0, "copy_unhandled", 0, 0);
      }
    }
  }
  close();
  KnockTag knockTag;
  if (knockFrame.mode) knockEnd(knockFrame, knockSeen != 0, knockSkipped, knockWarmSeen, knockTag);
  knockTag.expect = knockSeen;
  if (cbTracked) {
    // Encoders that skipped the join may still run when the next frame starts: its join follows this one's tail AND this
    // join (transitively, every earlier frame).
    if (joinFence) frontier.push_back(joinFence);
    cbAccess.push_back(std::move(mine));
  }
  // gpu.excl: frames with the same encoders (kind, formats, size, layers, samples) are one type.
  uint64_t exclType = 1469598103934665603ull ^ uint64_t(compiled.encoders.size());
  if (exclOn)
    for (const EncDiag& g : *diag)
      for (uint64_t v : {uint64_t(uint8_t(g.kind)), uint64_t(g.color0), uint64_t(g.depth), uint64_t(g.width) << 20 | g.height,
                         uint64_t(g.layers) << 8 | g.samples})
        exclType = (exclType ^ v) * 1099511628211ull;
  releaseMsaaBackings(frameIndex);
  if (residencyDirty) {  // render targets allocated lazily while encoding
    [residency commit];
    bump(cnt.resCommit);
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
  dumpSeqBytes += dumpBytes;
  std::shared_ptr<std::vector<DumpItem>> dumpDone = dumping ? dumpItems : nullptr;
  const std::string dumpTo = dumpDir;
  const bool tileRaisedFrame = compiled.stats.tileRaised > 0;
  const std::shared_ptr<StandIns> finalStand = cbStand;  // stand-ins of the last command buffer (the one `finished` is)
  const uint64_t prevCb = lastCbFrame;  // the frame submitted just before (gpu.order measurement)
  lastCbFrame = frameIndex;
  tail->publish = [=](id<MTLCommandBuffer> finished) {
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
    // Whole frame: every command buffer of it, whatever order their handlers ran in.
    d.timings.frameMs = double(tail->us.load()) / 1000.0;
    if (finished.status == MTLCommandBufferStatusError) {
      std::string encs;
      NSArray* infos = finished.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
      for (id<MTLCommandBufferEncoderInfo> i in infos)
        if (i.errorState == MTLCommandEncoderErrorStateFaulted || i.errorState == MTLCommandEncoderErrorStateAffected)
          encs += (encs.empty() ? "" : ",") + token(i.label.UTF8String ?: "?");
      // Refused in the background: no error line, no hang report (one gpu.cb.background line per second instead).
      const bool background = isBackgroundError(finished.error);
      if (background) impl->noteBackgroundError();
      else impl->err("what=command_buffer f=%llu error=%s encoders=%s", (unsigned long long)frameIndex,
                     token(str(finished.error)).c_str(), encs.c_str());
      const std::string why = str(finished.error);
      if (tileRaisedFrame && isTileFlushError(finished.error)) impl->tileRaisedFailure(frameIndex, finished.error);
      else if (isMemorylessError(finished.error)) impl->memorylessFailure(frameIndex, *finalStand, faultedEncoders(finished.error));
      // The failed command buffer is still listed: its pipelines name the suspect.
      if (isGpuHang(finished.error)) gGpuHangs.fetch_add(1);
      if (impl->lostByError(finished.error))
        impl->markLost("command_buffer f=" + std::to_string(frameIndex) + " " + why +
                       (encs.empty() ? "" : " encoders=" + encs));
      if (!background) {
        impl->dumpInFlight("command_buffer_error");
        impl->hangEncoders(frameIndex, finished.error, encSamples, samples, sampleCount);
      }
    } else if (vpgpu::tileProbeCompleted(frameIndex)) {  // the canary of a raised tileMaxVertices ran clean: Device keeps the value
      impl->info("gpu.tile probe ok f=" + std::to_string(frameIndex));
    }
    {
      std::lock_guard g(impl->inFlightMutex);
      impl->inFlightCbs.erase(frameIndex);
    }
    // Per core encoder: raw GPU ticks of its first start / last end sample (0 = not written: a render encoder without
    // vertex work never writes its start, an encoder the watchdog killed never its end).
    const bool wantRaw = impl->exclOn || impl->cbOrder;  // gpu.excl / gpu.order only
    std::vector<std::pair<uint64_t, uint64_t>> raw(wantRaw ? encSamples.size() : 0, {0, 0});
    std::vector<std::pair<uint64_t, uint64_t>> stg(impl->exclOn ? encStages.size() : 0, {0, 0});  // render: vertex end, fragment start
    if (sampleCount) {
      NSData* data = [samples resolveCounterRange:NSMakeRange(0, sampleCount)];
      const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
      auto tick = [&](NSUInteger k) -> uint64_t {
        if (!ts || k >= sampleCount) return 0;  // NSNotFound / MTLCounterDontSample: not sampled
        const uint64_t t = ts[k].timestamp;
        return t == MTLCounterErrorValue ? 0 : t;
      };
      for (size_t i = 0; i < stg.size(); ++i) stg[i] = {tick(encStages[i].first), tick(encStages[i].second)};
      for (size_t i = 0; i < encSamples.size(); ++i) {
        const auto [s0, s1] = encSamples[i];
        if (wantRaw) raw[i] = {tick(s0), tick(s1)};
        double ms = 0;
        if (ts && s0 != NSNotFound && ts[s0].timestamp != MTLCounterErrorValue &&
            ts[s1].timestamp != MTLCounterErrorValue && ts[s1].timestamp >= ts[s0].timestamp)
          ms = double(ts[s1].timestamp - ts[s0].timestamp) * impl->gpuTickNs / 1e6;
        d.timings.encoderMs.push_back(ms);
      }
    }
    if (impl->exclOn && sampleCount && finished.status == MTLCommandBufferStatusCompleted)
      impl->exclAccount(exclType, *diag, raw, stg, finished.GPUEndTime, d.timings.frameMs, knockTag);
    // Ordering measurement: this frame's first encoder start / last encoder end (GPU ticks, one clock for every command
    // buffer), against the frame submitted just before it.
    uint64_t spanStart = 0, spanEnd = 0;
    for (const auto& [t0, t1] : raw) {
      if (t0 && (!spanStart || t0 < spanStart)) spanStart = t0;
      if (t1 > spanEnd) spanEnd = t1;
    }
    std::string orderLine;
    {
      std::lock_guard lock(impl->doneMutex);
      impl->done.push_back(std::move(d));
      if (impl->cbOrder) {
        if (impl->finishedBelow < frameIndex) ++impl->cbOutOfOrder;  // an earlier frame is still running
        ++impl->cbCompleted;
        if (spanStart && spanEnd) {
          auto overlap = [&](const CbSpan& before, const CbSpan& after) {
            if (!before.end || !after.start || after.start >= before.end) return;
            ++impl->cbOverlaps;
            impl->cbMaxOverlapMs = std::max(impl->cbMaxOverlapMs, double(before.end - after.start) * impl->gpuTickNs / 1e6);
          };
          const CbSpan own{spanStart, spanEnd, prevCb};
          if (auto it = impl->cbSpans.find(prevCb); it != impl->cbSpans.end()) overlap(it->second, own);
          for (const auto& [f, sp] : impl->cbSpans)
            if (sp.prev == frameIndex) overlap(own, sp);
          impl->cbSpans[frameIndex] = own;
          while (impl->cbSpans.size() > 64) impl->cbSpans.erase(impl->cbSpans.begin());
        }
        if (impl->cbCompleted % 600 == 0) {
          char buf[256];
          std::snprintf(buf, sizeof buf,
                        "gpu.order f=%llu cbs=%llu overlapped=%llu max_overlap_ms=%.2f out_of_order=%llu slot_busy=%llu "
                        "carried_waits=%llu cb_order=%d tracked_joined=%llu tracked_free=%llu",
                        (unsigned long long)frameIndex, (unsigned long long)impl->cbCompleted,
                        (unsigned long long)impl->cbOverlaps, impl->cbMaxOverlapMs, (unsigned long long)impl->cbOutOfOrder,
                        (unsigned long long)impl->slotBusyWaits.load(), (unsigned long long)impl->carriedWaits.load(),
                        impl->cbTracked ? 2 : 1, (unsigned long long)impl->trackedJoined.load(),
                        (unsigned long long)impl->trackedFree.load());
          orderLine = buf;
        }
      }
    }
    if (!orderLine.empty()) impl->info(orderLine);
    impl->slotDone[slotIx].store(true, std::memory_order_release);  // every command buffer of the frame completed
    impl->frameFinished(frameIndex);
    dispatch_semaphore_signal(sem);  // last: once every slot is back the backend may be destroyed
  };
  // Before the publishing handler: handlers run in the order added, and publish frees the slot (and so the
  // backend); the slow-cb handler touches the backend and must run first.
  slowLog(cb, cbDraws, cbPasses, cbCost);
  const std::shared_ptr<CbTimes> times = trackCb(cb, frameIndex);
  addBusyHandler(cb);
  Impl* const hSelf = self;           // by-value copies (see slowLog): the block runs after submit returns
  const uint64_t hIntBytes = cbIntBytes;  // staging buffers the last command buffer retains until it completes
  [cb addCompletedHandler:^(id<MTLCommandBuffer> finished) {
    hSelf->bufIntLiveBytes.fetch_sub(hIntBytes, std::memory_order_relaxed);  // before the count that may publish the frame
    tail->us.fetch_add(uint64_t((finished.GPUEndTime - finished.GPUStartTime) * 1e6));
    tail->last = finished;
    if (tail->remaining.fetch_sub(1) == 1) {
      auto publish = std::move(tail->publish);
      publish(finished);
    }
  }];
  track.chunks = chunk ? chunk + 1 : 0;
  track.committedNs = steadyNs();
  lastFrameSplits.store(splitCount);
  {
    std::lock_guard g(inFlightMutex);
    inFlightCbs[frameIndex] = std::move(track);
  }
  tables.unlock();  // the last commit may park (app in the background); nothing below reads the tables
  timedCommit(cb, times);
}

namespace {
// gpu.excl description of one encoder (from the first frame of its type in the window).
std::string exclDescribe(const EncDiag& g) {
  char b[512];
  std::string label = g.label.empty() ? "-" : token(g.label).substr(0, 24);
  if (g.kind == 'r')
    std::snprintf(b, sizeof b, "kind=r label=%s fmt=%s depth=%s %ux%ux%u s%u att=%s", label.c_str(),
                  g.color0 ? dumpFormatToken(MTLPixelFormat(g.color0)).c_str() : "-",
                  g.depth ? dumpFormatToken(MTLPixelFormat(g.depth)).c_str() : "-", g.width, g.height, g.layers, g.samples,
                  g.attach.empty() ? "-" : g.attach.c_str());
  else
    std::snprintf(b, sizeof b, "kind=c label=%s", label.c_str());
  return b;
}
}  // namespace

void MetalBackend::Impl::exclAccount(uint64_t type, const std::vector<EncDiag>& diag,
                                     const std::vector<std::pair<uint64_t, uint64_t>>& span,
                                     const std::vector<std::pair<uint64_t, uint64_t>>& stages, double hostEnd, double cbMs,
                                     const KnockTag& knockTag) {
  const double msPerTick = gpuTickNs / 1e6;
  std::lock_guard lock(exclMutex);
  // VP_KNOCK: warm-up votes for the frame's heaviest render encoder; a measuring frame sums its dominant encoders.
  const bool knockWatch = knockTag.warm || knockTag.block >= 0;
  double kExcl = 0, kVtx = 0, kFrag = 0, bestMs = 0;
  size_t bestEnc = SIZE_MAX;
  uint32_t kTimed = 0;  // dominant encoders with a usable start / end pair
  if (exclWindowStart <= 0) exclWindowStart = hostEnd;
  ExclType& t = exclTypes[type];
  if (t.enc.size() < diag.size()) t.enc.resize(diag.size());
  ++t.n, ++exclCbs;
  t.gpu += cbMs;
  for (size_t i = 0; i < diag.size() && i < span.size(); ++i) {
    const EncDiag& g = diag[i];
    ExclEnc& x = t.enc[i];
    if (x.desc.empty()) x.desc = exclDescribe(g), x.view = g.view;
    x.draws += g.draws;
    x.loadMb += g.loadMb, x.storeMb += g.storeMb;
    x.joined += g.joined ? 1 : 0;
    uint64_t start = span[i].first, end = span[i].second;
    double vtxMs = 0, fragMs = 0;
    if (g.kind == 'r' && i < stages.size()) {
      const uint64_t vtxEnd = stages[i].first, fragStart = stages[i].second;
      // A render encoder without vertex work never writes its vertex samples: it starts with its fragments.
      if (fragStart && (!start || fragStart < start)) start = fragStart;
      if (!end) end = vtxEnd;
      if (span[i].first && vtxEnd >= span[i].first) vtxMs = double(vtxEnd - span[i].first) * msPerTick, x.vtx += vtxMs, ++x.vtxN;
      if (fragStart && span[i].second >= fragStart) fragMs = double(span[i].second - fragStart) * msPerTick, x.frag += fragMs, ++x.fragN;
    }
    if (!start || !end || end < start) continue;  // not sampled (or a broken pair)
    x.span += double(end - start) * msPerTick;
    // Idle: the GPU had nothing of ours between the latest end and this start (another process's work,
    // the compositor's, or waiting for the next submission).
    if (exclLastEnd && start > exclLastEnd) t.idle += double(start - exclLastEnd) * msPerTick;
    const uint64_t from = std::max(start, exclLastEnd);
    double exclMs = 0;
    if (end > from) {
      exclMs = double(end - from) * msPerTick;
      x.excl += exclMs, t.excl += exclMs;
    }
    exclLastEnd = std::max(exclLastEnd, end);
    if (knockWatch && g.kind == 'r' && g.sig) {
      if (knockTag.warm && g.knockCounted && exclMs > bestMs) bestMs = exclMs, bestEnc = i;
      if (knockTag.block >= 0 && g.sig == knockTag.sig) kExcl += exclMs, kVtx += vtxMs, kFrag += fragMs, ++kTimed;
    }
  }
  if (knockTag.warm && bestEnc != SIZE_MAX) knockVote(diag[bestEnc], bestMs, knockTag.gen);
  // Every dominant encoder of the frame must be timed (sample budget spent, broken pair): a partial sum reads as a fraction of
  // the frame's cost.
  if (knockTag.block >= 0) knockSample(knockTag.block, kExcl, kVtx, kFrag, kTimed && kTimed == knockTag.expect, knockTag.gen);
  if (hostEnd - exclWindowStart >= exclWindowS) exclReportLocked(hostEnd);
}

// Per window: one summary line, then the heaviest frame types (share of the exclusive time) with their heaviest
// encoders, all per frame of that type (averages).
void MetalBackend::Impl::exclReportLocked(double now) {
  const double window = std::max(1e-3, now - exclWindowStart);
  double total = 0, idle = 0;
  std::vector<std::pair<double, uint64_t>> order;
  for (const auto& [k, t] : exclTypes) {
    total += t.excl, idle += t.idle;
    order.push_back({t.excl, k});
  }
  std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  char buf[768];
  std::snprintf(buf, sizeof buf, "gpu.excl window_s=%.1f cbs=%llu types=%zu busy_ms_per_s=%.1f idle_ms_per_s=%.1f report=%llu",
                window, (unsigned long long)exclCbs, exclTypes.size(), total / window, idle / window,
                (unsigned long long)++exclReports);
  info(buf);
  size_t shown = 0;
  for (const auto& [ex, k] : order) {
    if (shown >= 5 || (shown >= 2 && total > 0 && ex / total < 0.03)) break;
    ++shown;
    const ExclType& t = exclTypes[k];
    const double n = std::max(1u, t.n);
    std::snprintf(buf, sizeof buf,
                  "gpu.excl type=%016llx cbs=%u per_s=%.1f gpu_ms=%.2f excl_ms=%.2f idle_before_ms=%.2f share=%.0f%% encoders=%zu",
                  (unsigned long long)k, t.n, double(t.n) / window, t.gpu / n, t.excl / n, t.idle / n,
                  total > 0 ? 100.0 * ex / total : 0.0, t.enc.size());
    info(buf);
    // Its encoders that matter (exclusive >= 0.05 ms or span >= 0.5 ms per frame), the 12 heaviest, in submission order.
    std::vector<size_t> pick;
    for (size_t i = 0; i < t.enc.size(); ++i)
      if (t.enc[i].excl / n >= 0.05 || t.enc[i].span / n >= 0.5) pick.push_back(i);
    std::sort(pick.begin(), pick.end(), [&](size_t a, size_t b) { return t.enc[a].excl > t.enc[b].excl; });
    if (pick.size() > 12) pick.resize(12);
    std::sort(pick.begin(), pick.end());
    for (size_t i : pick) {
      const ExclEnc& x = t.enc[i];
      std::snprintf(buf, sizeof buf,
                    "gpu.excl type=%016llx i=%zu %s draws=%.0f excl_ms=%.3f span_ms=%.3f vtx_ms=%.3f frag_ms=%.3f load_mb=%.1f "
                    "store_mb=%.1f joined=%.0f%% view=%s",
                    (unsigned long long)k, i, x.desc.c_str(), x.draws / n, x.excl / n, x.span / n,
                    x.vtxN ? x.vtx / x.vtxN : -1.0, x.fragN ? x.frag / x.fragN : -1.0, x.loadMb / n, x.storeMb / n,
                    100.0 * x.joined / n, x.view < 0 ? "all" : x.view ? "1" : "0");
      info(buf);
    }
  }
  exclTypes.clear();
  exclCbs = 0;
  exclWindowStart = now;
}

// ---------------------------------------------------------------------------------------------
// VP_KNOCK=auto (see Impl::knockOn).

namespace {
// Occurrences of `word` as a whole identifier followed by '(' (blanks allowed): a call `.sample(` or a loop `for (`.
uint32_t knockCountCalls(const std::string& s, const char* word) {
  auto ident = [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
  const size_t n = std::strlen(word);
  uint32_t count = 0;
  for (size_t p = s.find(word); p != std::string::npos; p = s.find(word, p + n)) {
    if (p && ident(s[p - 1])) continue;
    size_t q = p + n;
    while (q < s.size() && (s[q] == ' ' || s[q] == '\t')) ++q;
    if (q < s.size() && s[q] == '(') ++count;
  }
  return count;
}

// Non-overlapping occurrences of `text`.
uint32_t knockCountText(const std::string& s, const char* text) {
  const size_t n = std::strlen(text);
  uint32_t count = 0;
  for (size_t p = s.find(text); p != std::string::npos; p = s.find(text, p + n)) ++count;
  return count;
}

double knockMedian(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() & 1 ? v[m] : (v[m - 1] + v[m]) / 2;
}
}  // namespace

// Publishes the phase and progress for vpgpu_metal2_knock_progress (knockMutex held).
void MetalBackend::Impl::knockPublishLocked() {
  int phase = 0, done = 0, total = 0;
  const int blocks = int(knockSteps());
  switch (knockPhase) {
    case KnockPhase::Idle: break;
    case KnockPhase::Warm: phase = 1, done = int(knockFrames), total = int(kKnockWarmFrames); break;
    case KnockPhase::Measure: phase = 2, done = int(std::min<size_t>(knockStep, size_t(blocks))), total = blocks; break;
    case KnockPhase::Drain: phase = 3, done = blocks, total = blocks; break;
    case KnockPhase::Done:
      phase = knockNoData ? 6 : knockTimedOut ? 5 : 4;
      total = blocks;
      done = knockTimedOut ? int(std::min<size_t>(knockStep, size_t(blocks))) : blocks;
      break;
  }
  gKnockPhase.store(phase), gKnockDone.store(done), gKnockTotal.store(total);
}

// Submit thread, start of a frame: the host asked (vpgpu_metal2_knock_arm). Idle or Done: a fresh run starts its warm-up;
// a run in progress ignores the request.
void MetalBackend::Impl::knockArm() {
  if (!gKnockArmReq.exchange(false)) return;
  std::lock_guard lock(knockMutex);
  if (knockPhase != KnockPhase::Idle && knockPhase != KnockPhase::Done) return;
  knockFrames = knockMissed = knockDrain = knockMeasureFrames = knockMeasureHits = knockArmedFrames = 0;
  knockSigs.clear(), knockCands.clear(), knockBlocks.clear(), knockFsr.clear();
  knockSig = 0, knockStep = 0, knockCur = -1, knockTimedOut = knockNoData = false, knockFsrRun = false;
  knockRelaxed = false;
  ++knockGen;
  knockPhase = KnockPhase::Warm;
  knockOn = true;
  knockPublishLocked();
  info("gpu.knock arm");
}

// Submit thread, start of a frame: moves the phase on and says what this frame is encoded as.
void MetalBackend::Impl::knockBegin(KnockFrame& f) {
  std::lock_guard lock(knockMutex);
  struct Publish {
    Impl& s;
    ~Publish() { s.knockPublishLocked(); }
  } publish{*this};  // runs before `lock` is released
  if (knockPhase == KnockPhase::Warm) {
    if (knockFrames < kKnockWarmFrames) {  // knockEnd counts the frame, once it is known to have had a qualifying encoder
      if (!knockRelaxed && knockFrames == 0 && ++knockArmedFrames >= kKnockRelaxFrames) {
        knockRelaxed = true;  // nothing multisampled with enough draws in all that time: count any encoder with enough draws
        info("gpu.knock gate relaxed armed_frames=" + std::to_string(knockArmedFrames));
      }
      f.mode = 1;
      return;
    }
    knockPlanLocked();
  }
  if (knockPhase == KnockPhase::Measure) {
    // The fsr pass (last): a candidate whose substitutes are not all ready (or cannot be) is not measured; its two blocks are left out.
    while (knockFsrRun && knockCur < 0 && !(knockStep & 1) && knockStep < knockSteps() &&
           knockStep >= size_t(kKnockPasses) * knockCands.size() * 2 &&
           knockFsrStatusLocked(knockCands[knockStep % (knockCands.size() * 2) / 2]) != 0)
      knockStep += 2;
    if (++knockMeasureFrames > kKnockTimeoutFrames || knockMeasureHits >= kKnockTimeoutHits) {  // never got through every block (the encoder keeps vanishing): rank what has data
      info("gpu.knock timeout hits=" + std::to_string(knockMeasureHits) + " submits=" + std::to_string(knockMeasureFrames) +
           " step=" + std::to_string(knockStep) + " of=" + std::to_string(knockSteps()));
      knockReportLocked();
      knockPhase = KnockPhase::Done;
      knockTimedOut = true;
    } else if (knockStep >= knockSteps()) {
      knockPhase = KnockPhase::Drain;
      knockDrain = 0;
    } else {
      if (knockCur < 0) {
        const size_t per = knockCands.size() * 2;
        KnockBlock b;
        b.pass = uint8_t(knockStep / per), b.cand = int32_t(knockStep % per / 2), b.knock = (knockStep & 1) != 0;
        b.fsr = b.pass >= kKnockPasses;
        knockBlocks.push_back(std::move(b));
        knockCur = int64_t(knockBlocks.size() - 1);
      }
      const KnockBlock& b = knockBlocks[size_t(knockCur)];
      f.mode = 2, f.knock = b.knock && !b.fsr, f.fsr = b.knock && b.fsr, f.sig = knockSig, f.block = knockCur;
      f.vs = knockCands[size_t(b.cand)].key.first, f.fs = knockCands[size_t(b.cand)].key.second;
      return;
    }
  }
  if (knockPhase == KnockPhase::Drain) {  // every issued frame's samples in (or given up on): rank, then render normally
    bool ready = true;
    for (const KnockBlock& b : knockBlocks)
      if (!b.dead && b.excl.size() + b.partial < b.issued) ready = false;
    if (ready || ++knockDrain >= kKnockDrainFrames) {
      knockReportLocked();
      knockPhase = KnockPhase::Done;
    }
  }
  if (knockPhase == KnockPhase::Done) knockOn = false;
}

// Submit thread, end of a frame's encoding. `seen`: it had an encoder with the dominant signature. `warmSeen`: a warm-up frame
// that had an encoder worth counting (otherwise it does not advance the warm-up and its completion casts no vote).
void MetalBackend::Impl::knockEnd(const KnockFrame& f, bool seen, uint32_t skipped, bool warmSeen, KnockTag& tag) {
  std::lock_guard lock(knockMutex);
  struct Publish {
    Impl& s;
    ~Publish() { s.knockPublishLocked(); }
  } publish{*this};
  tag.gen = knockGen;
  if (f.mode == 1) {
    if (warmSeen) ++knockFrames, tag.warm = true;
    return;
  }
  KnockBlock& b = knockBlocks[size_t(f.block)];
  if (!seen) {
    // A submit without the dominant encoder is neutral (nothing issued, nothing sampled). Only when none had it for
    // kKnockMissAbort submits (loading, a menu): drop the partial block, and with it the baseline it was to be compared
    // with, and start that pair again once the encoder is back.
    if (++knockMissed >= kKnockMissAbort && b.issued) {
      b.dead = true;
      if (knockStep & 1) knockBlocks[size_t(f.block) - 1].dead = true, --knockStep;
      knockCur = -1;
    }
    return;
  }
  knockMissed = 0;
  ++knockMeasureHits;
  tag.block = f.block, tag.sig = f.sig;
  b.skipped += skipped;
  if (++b.issued >= kKnockBlockFrames) {
    knockCur = -1;
    ++knockStep;
  }
}

// Submit thread: one encoder of a warm-up frame, its draws per (vs, fs) pair.
void MetalBackend::Impl::knockMerge(uint64_t sig, uint64_t frame, const std::map<KnockKey, KnockPair>& local) {
  std::lock_guard lock(knockMutex);
  KnockSigInfo& s = knockSigs[sig];
  ++s.encoders;
  if (s.lastFrame != frame) s.lastFrame = frame, ++s.frames;
  for (const auto& [k, v] : local) {
    KnockPair& p = s.pairs[k];
    if (p.label.empty()) p.label = v.label, p.traits = v.traits;
    p.pipes.insert(v.pipes.begin(), v.pipes.end());
    p.draws += v.draws;
    p.verts += v.verts;
    s.draws += v.draws;
  }
}

// Completion: a warm-up frame's heaviest render encoder.
void MetalBackend::Impl::knockVote(const EncDiag& g, double ms, uint32_t gen) {
  std::lock_guard lock(knockMutex);
  if (gen != knockGen) return;
  KnockSigInfo& s = knockSigs[g.sig];
  s.score += ms;
  if (s.desc.empty()) s.desc = exclDescribe(g);
}

// Completion: a measuring frame's dominant encoders (summed).
void MetalBackend::Impl::knockSample(int64_t block, double excl, double vtx, double frag, bool complete, uint32_t gen) {
  std::lock_guard lock(knockMutex);
  if (gen != knockGen || size_t(block) >= knockBlocks.size() || knockBlocks[size_t(block)].dead) return;
  KnockBlock& b = knockBlocks[size_t(block)];
  if (!complete) {
    ++b.partial;
    return;
  }
  b.excl.push_back(excl), b.vtx.push_back(vtx), b.frag.push_back(frag);
}

// VP_KNOCK_FSR, knockMutex held: whether every pipeline of the candidate has its substitute (0), some are still building (1),
// or some cannot have one (2: the shader writes depth / sample mask, fetches the framebuffer, is dual-source, or the build failed).
int MetalBackend::Impl::knockFsrStatusLocked(const KnockCand& k) const {
  if (k.pipes.empty()) return 2;
  int status = 0;
  for (uint64_t key : k.pipes) {
    const auto it = knockFsr.find(key);
    if (it != knockFsr.end() && it->second.state == 2) return 2;
    if (it == knockFsr.end() || !it->second.state) status = 1;
  }
  return status;
}

// Submit thread: the substitute of a pipeline when it is ready (else nil).
id<MTLRenderPipelineState> MetalBackend::Impl::knockFsrPso(uint64_t pipelineKey) {
  std::lock_guard lock(knockMutex);
  const auto it = knockFsr.find(pipelineKey);
  return it != knockFsr.end() && it->second.state == 1 ? it->second.pso : nil;
}

// knockFsrQueue (never the render thread): the pipeline's descriptor with a trivial fragment function that writes a constant to
// every colour output the original wrote, no discard, no resource reads. The fragment function is one per output layout.
void MetalBackend::Impl::knockFsrBuild(uint64_t pipelineKey, uint32_t gen) {
  if (closing) return;
  @autoreleasepool {
    MTLRenderPipelineDescriptor* d = nil;
    std::string kinds;
    {
      std::lock_guard lock(pipelineMutex);
      const auto it = pipelines.find(pipelineKey);
      if (it != pipelines.end() && it->second.fsrDesc && !it->second.fsrKinds.empty()) {
        d = [it->second.fsrDesc copy];
        kinds = it->second.fsrKinds;
      }
    }
    id<MTLRenderPipelineState> pso = nil;
    std::string error;
    if (d) {
      id<MTLFunction> fn = nil;
      if (const auto f = knockFsrFns.find(kinds); f != knockFsrFns.end()) {
        fn = f->second;
      } else {
        std::string members, init;
        for (size_t i = 0; i < kinds.size(); ++i) {
          if (kinds[i] == '-') continue;
          const std::string n = std::to_string(i), type = kinds[i] == 'u' ? "uint4" : kinds[i] == 'i' ? "int4" : "float4";
          members += "    " + type + " c" + n + " [[color(" + n + ")]];\n";
          init += "    o.c" + n + " = " + type + (kinds[i] == 'f' ? "(0.5f);\n" : "(1);\n");
        }
        const std::string src = "#include <metal_stdlib>\nusing namespace metal;\n" +
                                (members.empty() ? std::string("fragment void vp_knock_fs() {}\n")
                                                 : "struct VpKnockOut {\n" + members + "};\nfragment VpKnockOut vp_knock_fs() {\n    VpKnockOut o;\n" +
                                                       init + "    return o;\n}\n");
        MTLCompileOptions* o = [MTLCompileOptions new];
        o.languageVersion = MTLLanguageVersion3_2;
        NSError* e = nil;
        id<MTLLibrary> lib = compileRetrying(&e, [&](NSError* __autoreleasing* pe) {
          return [device newLibraryWithSource:@(src.c_str()) options:o error:pe];
        });
        if (lib) fn = [lib newFunctionWithName:@"vp_knock_fs"];
        if (fn) knockFsrFns[kinds] = fn;
        else error = "msl compile: " + str(e);
      }
      if (fn) {
        d.fragmentFunction = fn;
        NSError* e = nil;
        pso = compileRetrying(&e, [&](NSError* __autoreleasing* pe) { return [device newRenderPipelineStateWithDescriptor:d error:pe]; });
        if (!pso) error = "render pipeline: " + str(e);
      }
    }
    static std::atomic<int> logged{0};
    if (d && !pso && logged.fetch_add(1) < 8)
      info("gpu.knock fsr_build_fail pipeline=" + std::to_string(pipelineKey) + " kinds=" + kinds + " error=" + token(error));
    std::lock_guard lock(knockMutex);
    const auto it = knockFsr.find(pipelineKey);  // gone: the run ended (or a new one started)
    if (gen != knockGen || it == knockFsr.end()) return;
    it->second.pso = pso;
    it->second.state = pso ? 1 : 2;
  }
}

// End of warm-up: the dominant signature and the pairs worth measuring (the 48 with most draws, and any with >= 2 % of the
// encoder's draws; at most 96, none under half a draw per encoder: too rare to move a block median).
void MetalBackend::Impl::knockPlanLocked() {
  uint64_t best = 0;
  double bestScore = 0;
  for (const auto& [s, v] : knockSigs)
    if (v.draws && v.score > bestScore) best = s, bestScore = v.score;  // draws: a qualifying encoder (see kKnockWarmDraws)
  if (!bestScore) {
    info("gpu.knock done reason=no_timed_render_encoder");
    knockPhase = KnockPhase::Done;
    knockNoData = true;
    return;
  }
  knockSig = best;
  const KnockSigInfo& si = knockSigs[best];
  std::vector<const std::pair<const KnockKey, KnockPair>*> order;
  for (const auto& kv : si.pairs) order.push_back(&kv);
  std::sort(order.begin(), order.end(), [](const auto* a, const auto* b) { return a->second.draws > b->second.draws; });
  const double enc = double(std::max<uint64_t>(1, si.encoders)), frames = double(std::max<uint64_t>(1, si.frames));
  for (size_t i = 0; i < order.size() && knockCands.size() < kKnockMaxCands; ++i) {
    const KnockPair& p = order[i]->second;
    if (double(p.draws) / enc < 0.5) break;
    if (i >= kKnockTopN && double(p.draws) < 0.02 * double(si.draws)) break;
    knockCands.push_back({order[i]->first, double(p.draws) / enc, p.label, p.traits,
                          std::vector<uint64_t>(p.pipes.begin(), p.pipes.end()), double(p.verts) / frames});
  }
  if (knockCands.empty()) {
    info("gpu.knock done reason=no_candidates");
    knockPhase = KnockPhase::Done;
    knockNoData = true;
    return;
  }
  if (knockFsrOn) {  // the substitutes, in candidate order, built off the render thread while the knock passes run
    knockFsrRun = true;
    if (!knockFsrQueue)
      knockFsrQueue = dispatch_queue_create("vpgpu.knock.fsr", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
    std::unordered_set<uint64_t> queued;
    for (const KnockCand& k : knockCands)
      for (uint64_t key : k.pipes) {
        if (!queued.insert(key).second) continue;
        knockFsr[key];  // state 0: building
        const uint32_t gen = knockGen;
        dispatch_async(knockFsrQueue, ^{ knockFsrBuild(key, gen); });
      }
  }
  char b[256];
  std::snprintf(b, sizeof b,
                "gpu.knock plan sig=%016llx encoders=%llu draws_per_encoder=%.1f candidates=%zu passes=%u fsr_passes=%u block_frames=%u frames=%zu ",
                (unsigned long long)best, (unsigned long long)si.encoders, double(si.draws) / enc, knockCands.size(),
                kKnockPasses, knockFsrRun ? 1u : 0u, kKnockBlockFrames, knockSteps() * kKnockBlockFrames);
  info(std::string(b) + si.desc);
  knockPhase = KnockPhase::Measure;
}

// Per candidate the base / knock medians of the dominant encoders (averaged over the passes that got both blocks), then the 20
// that cost most, with what their shaders do (MSL text) and the recipe label to find them again.
void MetalBackend::Impl::knockReportLocked() {
  struct Res {
    size_t cand;
    double base, knock, dVtx, dFrag, skipped;
    bool unstable;
    std::string fsr;  // VP_KNOCK_FSR: " fsr_ms=.. fsr_frag=.." or " fsr=na|pending|unstable" ("" when the pass is off)
  };
  std::vector<Res> res;
  const size_t cands = knockCands.size();
  const size_t passes = kKnockPasses + (knockFsrRun ? 1 : 0);  // the last one is the fsr pass
  std::vector<int64_t> at(cands * passes * 2, -1);  // [(cand * passes + pass) * 2 + knock] -> block (last good one)
  for (size_t i = 0; i < knockBlocks.size(); ++i) {
    const KnockBlock& b = knockBlocks[i];
    if (!b.dead && b.excl.size() >= 3) at[(size_t(b.cand) * passes + b.pass) * 2 + (b.knock ? 1 : 0)] = int64_t(i);
  }
  // Baselines of one scene cost the same whatever the candidate: a pass whose base is >25 % off the median of all of them was
  // measured over a different load (a rhythm change, a stall); it is left out and the candidate is flagged unstable=1.
  std::vector<double> bases;
  for (size_t i = 0; i < at.size(); i += 2)
    if (at[i] >= 0) bases.push_back(knockMedian(knockBlocks[size_t(at[i])].excl));
  const double gBase = bases.size() >= 3 ? knockMedian(bases) : 0;
  uint32_t partialFrames = 0;
  for (const KnockBlock& b : knockBlocks) partialFrames += b.partial;
  auto verts = [](const KnockCand& k) {  // " verts=": the candidate's view-vertices per frame
    char v[48];
    std::snprintf(v, sizeof v, " verts=%.0f", k.verts);
    return std::string(v);
  };
  for (size_t c = 0; c < cands; ++c) {
    Res r{c, 0, 0, 0, 0, 0, false};
    uint32_t n = 0;
    for (uint32_t p = 0; p < kKnockPasses; ++p) {
      const int64_t ib = at[(c * passes + p) * 2], ik = at[(c * passes + p) * 2 + 1];
      if (ib < 0 || ik < 0) continue;
      const KnockBlock& base = knockBlocks[size_t(ib)];
      const KnockBlock& knock = knockBlocks[size_t(ik)];
      const double bm = knockMedian(base.excl);
      if (gBase > 0 && (bm > gBase * 1.25 || bm < gBase * 0.75)) {
        r.unstable = true;
        continue;
      }
      r.base += knockMedian(base.excl), r.knock += knockMedian(knock.excl);
      r.dVtx += knockMedian(base.vtx) - knockMedian(knock.vtx);
      r.dFrag += knockMedian(base.frag) - knockMedian(knock.frag);
      r.skipped += double(knock.skipped) / double(std::max<uint32_t>(1, knock.issued));
      ++n;
    }
    char b[320];
    const KnockCand& k = knockCands[c];
    if (knockFsrRun) {  // the fsr pass: same rules (complete frames only, a baseline off the global one is left out)
      const int64_t ib = at[(c * passes + kKnockPasses) * 2], ik = at[(c * passes + kKnockPasses) * 2 + 1];
      if (ib >= 0 && ik >= 0) {
        const KnockBlock& base = knockBlocks[size_t(ib)];
        const KnockBlock& sub = knockBlocks[size_t(ik)];
        const double bm = knockMedian(base.excl);
        if (gBase > 0 && (bm > gBase * 1.25 || bm < gBase * 0.75)) {
          r.fsr = " fsr=unstable";
        } else {
          std::snprintf(b, sizeof b, " fsr_ms=%.3f fsr_frag=%.3f", bm - knockMedian(sub.excl), knockMedian(base.frag) - knockMedian(sub.frag));
          r.fsr = b;
        }
      } else {
        r.fsr = knockFsrStatusLocked(k) == 2 ? " fsr=na" : " fsr=pending";
      }
    }
    if (!n) {
      std::snprintf(b, sizeof b, "gpu.knock vs=%016llx fs=%016llx draws=%.1f unmeasured n=0 unstable=%d", (unsigned long long)k.key.first,
                    (unsigned long long)k.key.second, k.draws, r.unstable ? 1 : 0);
      info(std::string(b) + r.fsr + verts(k));
      continue;
    }
    r.base /= n, r.knock /= n, r.dVtx /= n, r.dFrag /= n, r.skipped /= n;
    std::snprintf(b, sizeof b,
                  "gpu.knock vs=%016llx fs=%016llx draws=%.1f base=%.3f knock=%.3f d_ms=%.3f d_vtx=%.3f d_frag=%.3f skipped=%.1f n=%u unstable=%d",
                  (unsigned long long)k.key.first, (unsigned long long)k.key.second, k.draws, r.base, r.knock, r.base - r.knock,
                  r.dVtx, r.dFrag, r.skipped, n, r.unstable ? 1 : 0);
    info(std::string(b) + r.fsr + verts(k));
    res.push_back(r);
  }
  std::sort(res.begin(), res.end(), [](const Res& a, const Res& b) { return a.base - a.knock > b.base - b.knock; });
  // Static shader stats from the MSL text: size, texture reads, loops, storage-buffer accesses, discards, SIMD ops.
  auto stats = [&](uint64_t id, const char* stg) -> std::string {
    MslShader sh;
    if (!id || !shaders.find(id, sh)) return std::string(stg) + ":none";
    const std::string& s = sh.source;
    char b[256];
    std::snprintf(b, sizeof b, "%s:chars=%zu sample=%u read=%u loops=%u device=%u discard=%u simd=%u", stg, s.size(),
                  knockCountCalls(s, "sample"), knockCountCalls(s, "read"),
                  knockCountCalls(s, "for") + knockCountCalls(s, "while"), knockCountText(s, "device "),
                  knockCountText(s, "discard_fragment"), knockCountText(s, "simd_"));
    return b;
  };
  const size_t top = std::min<size_t>(kKnockTop, res.size());
  char b[320];
  std::snprintf(b, sizeof b, "gpu.knock.top n=%zu measured=%zu candidates=%zu sig=%016llx ", top, res.size(), cands,
                (unsigned long long)knockSig);
  info(std::string(b) + knockSigs[knockSig].desc);
  for (size_t i = 0; i < top; ++i) {
    const Res& r = res[i];
    const KnockCand& k = knockCands[r.cand];
    std::snprintf(b, sizeof b, "gpu.knock.top rank=%zu vs=%016llx fs=%016llx draws=%.1f base=%.3f d_ms=%.3f d_vtx=%.3f d_frag=%.3f ",
                  i + 1, (unsigned long long)k.key.first, (unsigned long long)k.key.second, k.draws, r.base, r.base - r.knock,
                  r.dVtx, r.dFrag);
    info(std::string(b) + "label=" + (k.label.empty() ? "?" : token(k.label)) + k.traits + " " + stats(k.key.first, "vs") + " " +
         stats(k.key.second, "fs") + " unstable=" + (r.unstable ? "1" : "0") + r.fsr + verts(k));
  }
  info("gpu.knock done candidates=" + std::to_string(cands) + " measured=" + std::to_string(res.size()) +
       " partial_frames=" + std::to_string(partialFrames));
  knockFsr.clear();  // the substitutes are not needed any more
}

// ---------------------------------------------------------------------------------------------

MetalBackend::MetalBackend(const ShaderProvider& shaders, LogSink log, MetalOptions options)
    : impl_(std::make_unique<Impl>(shaders, std::move(log), std::move(options))) {
  attr::setLog([m = impl_.get()](const std::string& line) { m->info(line); });  // gpu.bigalloc
  @autoreleasepool {
    impl_->init();
  }
  // The app went to the background: the save job flushes the whole archive queue at once (maybeSaveArchive). Never blocks.
  if (impl_->archiveOn)
    impl_->pauseHook = vpgpu::backgroundGate().addPauseHook([m = impl_.get()] { m->requestArchiveSave(); });
}

MetalBackend::~MetalBackend() {
  attr::setLog(nullptr);
  if (impl_->pauseHook) vpgpu::backgroundGate().removePauseHook(impl_->pauseHook);  // waits for a running call
  impl_->closing = true;
  impl_->waitArchiveSaveJob();  // the save job uses impl_; the final save was done by saveArchive()
  for (dispatch_queue_t q : impl_->prewarmQueues)
    if (q) dispatch_sync(q, ^{});  // queued prewarms see `closing`; the running one finishes
  if (impl_->knockFsrQueue) dispatch_sync(impl_->knockFsrQueue, ^{});  // same for the knock's substitutes
  if (ok()) {
    vpgpu::backgroundGate().release();  // a commit parked for the background holds a slot drain() waits for
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

// The public entry points below run on guest threads, which have no autorelease pool (long-lived pthreads): every one
// that touches Objective-C objects opens its own, or the autoreleased temporaries (descriptors, getter returns) are
// never freed. nativeTexture matters most: texture() hands its MTLTexture back autoreleased (clang ignores
// __unsafe_unretained on a return type), which kept every texture alive past destroyTexture.
void MetalBackend::createBuffer(BufferId handle, const BufferDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  ScopedPool pool;
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
  attr::track(d.guestMemory ? "m2.buf.nocopy" : "m2.buf.own", b.buffer);  // nocopy wraps guest memory
  std::lock_guard<std::recursive_mutex> tables(m.tableMutex);  // the allocation above does not need the tables
  if (m.residency) {
    [m.residency addAllocation:b.buffer];
    Impl::bump(m.cnt.resAdd);
    m.residencyDirty = true;
  }
  b.bytes = b.buffer.length;
  b.host = reinterpret_cast<uintptr_t>(b.buffer.contents) + uintptr_t(b.base);
  b.span = uint64_t(b.buffer.length) > b.base ? uint64_t(b.buffer.length) - b.base : 0;
  b.guest = d.guestMemory != nullptr;
  if (!b.guest) m.rangeAdd(b.buffer.contents, b.bytes);
  Impl::bump(b.guest ? m.cnt.bufNoCopy : m.cnt.bufNew);
  Impl::bump(b.guest ? m.cnt.bufNoCopyBytes : m.cnt.bufNewBytes, b.bytes);
  (b.guest ? m.bufGuestBytes : m.bufOwnBytes) += b.bytes;
  m.buffers[handle.index] = b;
}

uint64_t MetalBackend::bufferAddress(BufferId handle) const {
  Impl& m = *impl_;
  ScopedPool pool;
  std::lock_guard<std::recursive_mutex> tables(impl_->tableMutex);
  auto b = m.buffers.find(handle.index);
  if (b == m.buffers.end() || b->second.generation != handle.generation) return 0;
  if (!m.residency) {  // the encoders make it resident (useBound)
    std::lock_guard<std::mutex> lock(m.addressMutex);
    if (std::find(m.addressTaken.begin(), m.addressTaken.end(), b->second.buffer) == m.addressTaken.end())
      m.addressTaken.push_back(b->second.buffer);
  }
  return b->second.buffer.gpuAddress + b->second.base;
}

void MetalBackend::createTexture(TextureId handle, const TextureDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  ScopedPool pool;
  std::lock_guard<std::recursive_mutex> tables(m.tableMutex);
  m.textures[handle.index] = {handle.generation, nil, d, false};
  Impl::bump(m.cnt.texCreated);
  // Render targets wait for their first use: one that only ever lives on tile never gets memory.
  const bool deferred = !d.persistent && m.tileMemory && (d.usage & (usage::colorTarget | usage::depthTarget));
  if (deferred) {
    // Any deferred render target (colour or depth, 1x or MSAA): it has no memory until a frame needs it, and
    // msaa_memoryless says how much it would take if it had.
    m.textures[handle.index].deferredRT = true;
    const uint64_t logical = m.logicalBytes(d);
    m.textures[handle.index].msaaLogical = logical;
    m.msaaDeferredBytes += logical;
  }
  if (!deferred) m.texture(handle);
}

void MetalBackend::createSampler(SamplerId handle, const SamplerDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  ScopedPool pool;
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
  std::lock_guard<std::recursive_mutex> tables(m.tableMutex);
  m.samplers[handle.index] = {handle.generation, st};
}

void MetalBackend::destroyBuffer(BufferId handle) {
  std::lock_guard<std::recursive_mutex> tables(impl_->tableMutex);
  auto it = impl_->buffers.find(handle.index);
  if (it == impl_->buffers.end() || it->second.generation != handle.generation) return;
  ScopedPool pool;
  if (!it->second.guest) impl_->rangeDel(it->second.buffer.contents);
  {
    std::lock_guard<std::mutex> lock(impl_->addressMutex);
    auto& at = impl_->addressTaken;
    at.erase(std::remove(at.begin(), at.end(), it->second.buffer), at.end());
  }
  impl_->retire(it->second.buffer);
  (it->second.guest ? impl_->bufGuestBytes : impl_->bufOwnBytes) -= it->second.bytes;
  impl_->buffers.erase(it);
}

void MetalBackend::destroyTexture(TextureId handle) {
  std::lock_guard<std::recursive_mutex> tables(impl_->tableMutex);
  auto it = impl_->textures.find(handle.index);
  if (it == impl_->textures.end() || it->second.generation != handle.generation) return;
  ScopedPool pool;
  if (id<MTLTexture> t = it->second.texture)
    for (auto f = impl_->formatViews.begin(); f != impl_->formatViews.end();) {
      if (f->first.base == t) {
        if (f->second != t) impl_->retire(f->second);
        Impl::bump(impl_->cnt.fmtViewDrop);
        f = impl_->formatViews.erase(f);
      } else {
        ++f;
      }
    }
  if (it->second.pinned && it->second.texture) {  // the host holds a raw pointer: see Impl::promotePinned
    uint64_t done;
    {
      std::lock_guard lk(impl_->doneMutex);
      done = impl_->finishedBelow;
    }
    impl_->pinnedHold.push_back({it->second.texture, steadyNs(), done});
  } else
    impl_->retire(it->second.texture);
  if (it->second.texture) impl_->texLive(it->second.texture, false);
  if (it->second.bytes) impl_->texBytes[it->second.cat] -= it->second.bytes;
  if (it->second.msaaLogical)
    impl_->msaaDeferredBytes -= std::min<uint64_t>(impl_->msaaDeferredBytes.load(), it->second.msaaLogical);
  impl_->backedMsaa.erase(handle.index);
  impl_->textures.erase(it);
  Impl::bump(impl_->cnt.texDestroyed);
}

void MetalBackend::destroySampler(SamplerId handle) {
  std::lock_guard<std::recursive_mutex> tables(impl_->tableMutex);
  auto it = impl_->samplers.find(handle.index);
  if (it == impl_->samplers.end() || it->second.generation != handle.generation) return;
  ScopedPool pool;
  impl_->retire(it->second.sampler);
  impl_->samplers.erase(it);
}

bool MetalBackend::compilePipeline(const PipelineState& s, std::string& error) {
  if (!ok()) {
    error = "no Metal device";
    return false;
  }
  @autoreleasepool {
    const ActivityGauge::Scope inCompile(impl_->compileGauge);  // prewarm and the archive harvest wait while pipelines compile
    Impl::Pipeline p;
    const bool built = s.computeShader ? impl_->buildCompute(s, p, error) : impl_->buildRender(s, p, error);
    if (!built) {
      impl_->err("what=pipeline key=%016llx label=%s error=%s", (unsigned long long)s.key().value,
                 token(s.label).c_str(), token(error).c_str());
      if (impl_->psoFailLogged.fetch_add(1) < 64) {  // vk.pso.census failed=: why, for the first 64
        std::string why = token(error);
        for (char& c : why)
          if (c == '\r' || c == '\t') c = '_';
        if (why.size() > 200) why.resize(200);
        char key[24];
        std::snprintf(key, sizeof key, "%016llx", (unsigned long long)s.key().value);
        impl_->info("vk.pso.fail reason=metal:" + why + " key=" + key);
      }
      return false;
    }
    {
      std::lock_guard lock(impl_->pipelineMutex);
      impl_->pipelineDrops.erase(s.key().value);  // wanted again: the pipeline still published is the same one
      const uint64_t fk[2] = {p.fnKeys[0], p.fnKeys[1]};
      // First one wins: entries never move (until dropPipelines, on the submit thread).
      if (impl_->pipelines.emplace(s.key().value, std::move(p)).second)
        for (uint64_t k : fk)
          if (k) ++impl_->functionUsers[k];
    }
    // The pipeline is published: the caller marks it Ready right after this returns. Archiving (which
    // compiles it again and serializes to disk, seconds per chunk) runs on its own queue.
    impl_->requestArchiveSave();
    return true;
  }
}

void MetalBackend::destroyPipeline(PipelineKey key) {
  if (!ok()) return;
  std::lock_guard lock(impl_->pipelineMutex);
  if (impl_->pipelines.count(key.value)) impl_->pipelineDrops.insert(key.value);
}

void MetalBackend::Impl::dropPipelines() {
  std::lock_guard lock(pipelineMutex);
  for (uint64_t key : pipelineDrops) {
    auto it = pipelines.find(key);
    if (it == pipelines.end()) continue;
    Pipeline& p = it->second;
    // Command buffers in flight hold the states they bound; the graveyard covers the rest.
    retire(p.render);
    retire(p.compute);
    retire(p.depth);
    for (auto& md : p.maskedDepth) retire(md.second);
    for (uint64_t k : p.fnKeys) {
      auto u = functionUsers.find(k);
      if (!k || u == functionUsers.end() || --u->second) continue;
      functionUsers.erase(u);
      functions.erase(k);
    }
    pipelines.erase(it);
  }
  pipelineDrops.clear();
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
  ScopedPool pool;
  bool drained = false;
  return impl_->drain(timeoutNs, drained);
}

MetalBackend::WaitResult MetalBackend::waitFrames(uint64_t frames, uint64_t timeoutNs) {
  Impl& m = *impl_;
  ScopedPool pool;
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
    if (c != seen || vpgpu::backgroundGate().active() || t - lastPass > kPauseNs) {
      seen = c;
      quietSince = t;
    } else if (t - quietSince > kStallNs && m.finishedBelow < frames) {
      lock.unlock();
      m.markLost("gpu_stuck (fence wait, no command buffer completed for 20 s)");
      lock.lock();
    } else if (m.finishedBelow < frames) {
      lock.unlock();  // the report takes other locks
      m.stallWarn(c, t - quietSince, "fence_wait");
      lock.lock();
    }
    lastPass = t;
  }
}

// Memory warning / critical (any thread). Frees right now what can be dropped without the submit thread:
// MTLFunctions (each keeps its MTLLibrary) that are not cached for a live pipeline. Both levels do the same:
// the old level 2 also dropped the functions of live pipelines, which freed 0 MiB (a built pipeline keeps its
// own code) and only forced recompiles later (AC Nexus, 1535 libraries). Everything the submit thread owns
// (transient MSAA backings, chained argument buffers of idle slots, the graveyard) is queued and done by the
// next submits; the graveyard is also drained by drainRetired (MetalBackend) when the caller can serialize.
// Pipeline states themselves are NOT dropped:
// the pipeline cache above considers them Ready and a draw with a missing PSO is skipped. Appends
// "key=MiB ..." to `report`; returns bytes freed immediately.
size_t MetalBackend::trimMemory(int level, std::string& report) {
  if (!ok()) return 0;
  Impl& m = *impl_;
  size_t freed = 0;
  char b[512];
  @autoreleasepool {
    const uint64_t before = m.device.currentAllocatedSize;
    size_t dropped = 0, total = 0;
    {
      std::lock_guard lock(m.pipelineMutex);
      total = m.functions.size();
      for (auto it = m.functions.begin(); it != m.functions.end();) {
        auto u = m.functionUsers.find(it->first);
        if (u == m.functionUsers.end() || !u->second) {
          it = m.functions.erase(it);
          ++dropped;
        } else {
          ++it;
        }
      }
    }
    const uint64_t after = m.device.currentAllocatedSize;
    const uint64_t delta = before > after ? before - after : 0;
    freed += delta;
    std::snprintf(b, sizeof b, "libs_dropped=%zu libs_kept=%zu libs=%.1f", dropped, total - dropped, double(delta) / 1048576.0);
    report += b;
  }
  std::snprintf(b, sizeof b, " msaa_queued=%.1f args_queued=%.1f msaa_freed_total=%.1f args_freed_total=%.1f",
                double(m.texBytes[Impl::kTexMsaa].load()) / 1048576.0, double(m.argsMoreBytes.load()) / 1048576.0,
                double(m.trimmedMsaaBytes.load()) / 1048576.0, double(m.trimmedArgsBytes.load()) / 1048576.0);
  report += b;
  m.trimSlotsLeft.store(uint32_t(m.slots.size()));
  m.trimLevel.store(level < 1 ? 1 : level);
  m.info("gpu.metal.info what=trim level=" + std::to_string(level) + " " + report);
  return freed;
}

// Gives back the objects destroyed by the guest that no command buffer can still read (completed frames' retired
// lists and the graveyard, see drainIdle). `minIdleMs` is ignored. Caller rules of submit/destroyTexture
// (serialized, never concurrently with them); never blocks. Bytes released.
size_t MetalBackend::drainRetired(uint32_t /*minIdleMs: no longer gates the drain*/, std::string& report) {
  if (!ok()) return 0;
  @autoreleasepool {
    return impl_->drainIdle(report);
  }
}

void MetalBackend::diagRanges(std::vector<std::pair<uintptr_t, uint64_t>>& out) const {
  if (!impl_) return;
  std::lock_guard lock(impl_->rangesMutex);
  out.reserve(out.size() + impl_->ownRanges.size());
  for (const auto& [a, n] : impl_->ownRanges) out.emplace_back(a, n);
}

void MetalBackend::diagCounters(std::vector<std::pair<const char*, uint64_t>>& out) const {
  if (!impl_) return;
  const Impl::DiagCounters& c = impl_->cnt;
  auto add = [&out](const char* k, const std::atomic<uint64_t>& v) { out.emplace_back(k, v.load(std::memory_order_relaxed)); };
  add("buf_new", c.bufNew);
  add("buf_new_b", c.bufNewBytes);
  add("buf_nocopy", c.bufNoCopy);
  add("buf_nocopy_b", c.bufNoCopyBytes);
  add("buf_int", c.bufInternal);
  add("buf_int_b", c.bufInternalBytes);
  add("tex_new", c.texNew);
  add("tex_new_b", c.texNewBytes);
  add("tex_views", c.texViews);
  add("cb", c.cbs);
  add("enc_render", c.encRender);
  add("enc_compute", c.encCompute);
  add("enc_blit", c.encBlit);
  add("cut_enc", c.cutEnc);
  add("chunk_wait", c.chunkWaits);
  add("chunk_wait_us", c.chunkWaitUs);
  add("res_add", c.resAdd);
  add("res_remove", c.resRemove);
  add("res_commit", c.resCommit);
  add("drain_runs", c.drainRuns);
  add("drained", c.drainedObjs);
  add("drained_b", c.drainedBytes);
  // Gauges (not rates): retired objects still pinned (graveyard + slots) and what they weigh.
  add("g_held", c.heldObjs);
  add("g_held_b", c.heldBytes);
  // Texture lifetime (see DiagCounters): the cumulative pairs show create/destroy balance, the gauges what is alive now.
  add("tex_created", c.texCreated);
  add("tex_destroyed", c.texDestroyed);
  add("view_destroyed", c.viewsDestroyed);
  add("fmtview_new", c.fmtViewNew);
  add("fmtview_drop", c.fmtViewDrop);
  const uint64_t created = c.texCreated.load(std::memory_order_relaxed), destroyed = c.texDestroyed.load(std::memory_order_relaxed);
  out.emplace_back("g_tex_live", created - std::min(created, destroyed));
  out.emplace_back("g_views", impl_->liveViews.load(std::memory_order_relaxed));
}

bool MetalBackend::lost() const { return impl_->lostFlag.load(); }
void MetalBackend::dumpInFlight(const char* why) {
  if (impl_) impl_->dumpInFlight(why, true);
}
// Why the backend was lost (empty while healthy). Metal cannot recover in-process: revoked/ignored/blacklisted
// submissions refuse every queue of the process, and a queue stuck 20 s still owns the in-flight frames whose
// fences, residency and guest memory the rest of the stack waits on; the runtime closes the game cleanly
// (vpvk_gpu_lost) instead of pretending to continue.
std::string MetalBackend::lostReason() const {
  std::lock_guard lk(impl_->lostWhyMutex);
  return impl_->lostWhy;
}

// Age of the oldest frame command buffer still in flight (committed, completion handler not yet run).
uint32_t MetalBackend::gpuBacklogMs() const {
  if (!impl_) return 0;
  std::lock_guard g(impl_->inFlightMutex);
  int64_t oldest = 0;
  const int64_t now = steadyNs();
  for (const auto& [f, c] : impl_->inFlightCbs) oldest = std::max(oldest, now - c.committedNs);
  return uint32_t(std::min<int64_t>(oldest / 1000000, UINT32_MAX));
}

uint32_t MetalBackend::lastFrameHangSplits() const { return impl_ ? impl_->lastFrameSplits.load() : 0; }

void* MetalBackend::contents(BufferId handle) const {
  ScopedPool pool;
  std::lock_guard<std::recursive_mutex> tables(impl_->tableMutex);
  auto it = impl_->buffers.find(handle.index);
  if (it == impl_->buffers.end() || it->second.generation != handle.generation) return nullptr;
  return static_cast<uint8_t*>(it->second.buffer.contents) + it->second.base;
}

TextureId MetalBackend::createView(TextureId base, const ViewDesc& d) {
  Impl& m = *impl_;
  ScopedPool pool;
  std::lock_guard<std::recursive_mutex> tables(m.tableMutex);
  const uint32_t index = Impl::kViewBit | m.nextView++;
  Impl::View v;
  v.generation = 1, v.base = base, v.desc = d;
  m.views[index] = v;
  m.liveViews.fetch_add(1, std::memory_order_relaxed);
  return TextureId{index, 1};
}

void MetalBackend::destroyView(TextureId t) {
  std::lock_guard<std::recursive_mutex> tables(impl_->tableMutex);
  auto it = impl_->views.find(t.index);
  if (it == impl_->views.end()) return;
  ScopedPool pool;
  if (it->second.texture) impl_->retire(it->second.texture);  // may still be in flight
  impl_->views.erase(it);
  impl_->liveViews.fetch_sub(1, std::memory_order_relaxed);
  Impl::bump(impl_->cnt.viewsDestroyed);
}

void* MetalBackend::nativeDevice() const { return (__bridge void*)impl_->device; }
void* MetalBackend::nativeQueue() const { return (__bridge void*)impl_->queue; }

void* MetalBackend::nativeTexture(TextureId handle) {
  ScopedPool pool;  // texture() returns its MTLTexture autoreleased: without a pool the texture is never freed
  std::lock_guard<std::recursive_mutex> tables(impl_->tableMutex);
  if (auto it = impl_->textures.find(handle.index); it != impl_->textures.end() && it->second.generation == handle.generation)
    it->second.pinned = true;  // client keeps it (a stale handle must not pin the texture that reused its slot)
  return (__bridge void*)impl_->texture(handle);
}

// The block layout of what a texture really holds: the first core format whose Metal pixel format is `pf`. The texture's
// own pixel format rules the size of a block, not the core format it was created from (D24UnormS8 is held as
// Depth32Float_Stencil8). `fallback` when no core format maps to `pf`.
static const FormatInfo& heldFormatInfo(MTLPixelFormat pf, const FormatInfo& fallback) {
  for (int i = 1; i < int(Format::Count); ++i)
    if (pixelFormat(Format(i)) == pf) return formatInfo(Format(i));
  return fallback;
}

// One subresource of a texture to the host (Backend::readTexture): private storage, so a blit into a shared buffer, band by
// band (whole block rows) to keep the staging buffer small. Nothing is allocated for a texture that has no memory.
ReadResult MetalBackend::readTexture(const TextureSpan& s, std::vector<uint8_t>& out) {
  out.clear();
  if (!ok() || impl_->lostFlag.load()) return ReadResult::Failed;
  if (!s.width || !s.height || !s.depth || s.layerCount != 1 || (s.texture.index & Impl::kViewBit)) return ReadResult::Unsupported;
  Impl& m = *impl_;
  ScopedPool pool;
  id<MTLTexture> tex = nil;  // strong: another thread's destroyTexture cannot free it under the blit
  TextureDesc desc;
  {
    std::lock_guard<std::recursive_mutex> tables(m.tableMutex);
    const auto it = m.textures.find(s.texture.index);
    if (it == m.textures.end() || it->second.generation != s.texture.generation) return ReadResult::Failed;
    if (!it->second.texture) return it->second.failed ? ReadResult::Failed : ReadResult::Memoryless;
    tex = it->second.texture;
    desc = it->second.desc;
  }
  if (tex.storageMode == MTLStorageModeMemoryless) return ReadResult::Memoryless;
  if (desc.samples > 1 || tex.sampleCount > 1) return ReadResult::Unsupported;  // a multisample texture cannot be blitted out
  if (s.mip >= desc.mips || s.layer >= desc.layers) return ReadResult::Failed;
  const FormatInfo& fi = heldFormatInfo(tex.pixelFormat, formatInfo(desc.format));  // what the texture holds, not what it was made from
  NSUInteger texel = fi.bytesPerBlock;  // bytes of one block of the aspect read
  MTLBlitOption options = MTLBlitOptionNone;
  if (fi.depth() && fi.stencil()) {  // D24UnormS8 / D32FloatS8: one aspect at a time, through the combined texture
    if (s.aspect == Aspect::Depth) {
      if (desc.format == Format::D24UnormS8) return ReadResult::Unsupported;
      texel = 4, options = MTLBlitOptionDepthFromDepthStencil;
    } else if (s.aspect == Aspect::Stencil) {
      texel = 1, options = MTLBlitOptionStencilFromDepthStencil;
    } else {
      return ReadResult::Unsupported;
    }
  }
  const NSUInteger cols = (s.width + fi.blockSize - 1) / fi.blockSize, rows = (s.height + fi.blockHeight - 1) / fi.blockHeight;
  const NSUInteger rowBytes = cols * texel;
  if (!rowBytes) return ReadResult::Unsupported;
  constexpr NSUInteger kStage = 16u << 20;
  constexpr int64_t kReadTimeoutNs = 2 * int64_t(NSEC_PER_SEC);  // per band
  const NSUInteger bandRows = std::max<NSUInteger>(1, std::min<NSUInteger>(rows, kStage / rowBytes));
  id<MTLBuffer> stage = [m.device newBufferWithLength:bandRows * rowBytes options:MTLResourceStorageModeShared];
  if (!stage) return ReadResult::Failed;
  out.resize(size_t(rowBytes) * rows * s.depth);
  for (NSUInteger z = 0; z < s.depth; ++z) {
    for (NSUInteger r = 0; r < rows; r += bandRows) {
      const NSUInteger n = std::min(bandRows, rows - r);
      const NSUInteger y = r * fi.blockHeight, h = std::min<NSUInteger>(n * fi.blockHeight, s.height - y);
      id<MTLCommandBuffer> cb = [m.queue commandBuffer];
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
      if (!bl) {
        out.clear();
        return ReadResult::Failed;
      }
      cb.label = @"vpgpu.readTexture";
      [bl copyFromTexture:tex
               sourceSlice:s.layer
               sourceLevel:s.mip
              sourceOrigin:MTLOriginMake(0, y, z)
                sourceSize:MTLSizeMake(s.width, h, 1)
                  toBuffer:stage
         destinationOffset:0
    destinationBytesPerRow:rowBytes
  destinationBytesPerImage:rowBytes * n
                   options:options];
      [bl endEncoding];
      // Bounded wait (GPU waits never block forever). The handler holds the staging buffer and the texture: on a timeout the
      // blit may still run, and they are freed only once the command buffer is done with them.
      dispatch_semaphore_t done = dispatch_semaphore_create(0);
      [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        (void)stage;
        (void)tex;
        dispatch_semaphore_signal(done);
      }];
      [cb commit];
      if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, kReadTimeoutNs)) != 0) {
        m.err("what=read_texture_timeout tex=%u mip=%u layer=%u", unsigned(s.texture.index), unsigned(s.mip), unsigned(s.layer));
        out.clear();
        return ReadResult::Failed;
      }
      if (cb.status != MTLCommandBufferStatusCompleted) {
        out.clear();
        return ReadResult::Failed;
      }
      std::memcpy(out.data() + (size_t(z) * rows + r) * rowBytes, stage.contents, size_t(n) * rowBytes);
    }
  }
  return ReadResult::Ok;
}

// ---------------------------------------------------------------------------------------------
// Pipeline binary archives

namespace {

// Files loaded per session; the oldest are evicted. Must hold a game's whole pipeline set
// (files of kMinPackCap..32 pipelines): 64 files (2048) against 4101 pipelines evicted exactly the
// files the next start needed, so every start recompiled and rewrote half of the cache.
constexpr size_t kMaxArchiveFiles = 512;
constexpr size_t kMaxDepthArchiveFiles = 64;  // depth-only files (small, a few pipelines each); the oldest are evicted
constexpr size_t kMinPackCap = 8;           // archivePackCap never goes below this
constexpr int kArchiveDrops = 4;            // culprits dropped per save before splitting
constexpr int kArchiveSplitDepth = 4;       // halvings of the unsaved entries (<= 16 files)

// phys_footprint of this process (what jetsam counts), 0 = unknown.
uint64_t physFootprint() {
  task_vm_info_data_t vm{};
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &cnt) == KERN_SUCCESS ? uint64_t(vm.phys_footprint) : 0;
}

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
  // Functions are identified by their label (the Impl::functions key: shader source, fetch inputs,
  // macros), not by address: one is dropped with its last pipeline and may be compiled again.
  void addFunction(id<MTLFunction> fn) {
    NSString* l = fn ? fn.label : nil;
    if (l.length) {
      const char* c = l.UTF8String;
      for (; *c; ++c) add(uint64_t(uint8_t(*c)));
    } else {
      add((id)fn);
    }
  }
};

// Bit i = buffer slot i is declared immutable (VP_BUF_IMMUTABLE).
uint64_t immutableSlots(MTLPipelineBufferDescriptorArray* buffers) {
  uint64_t m = 0;
  for (NSUInteger i = 0; i < 31; ++i)
    if (buffers[i].mutability == MTLMutabilityImmutable) m |= uint64_t(1) << i;
  return m;
}

// The function label (see DescHash::addFunction) identifies the compiled variant.
uint64_t descriptorKey(id desc, bool compute) {
  DescHash k;
  k.add(uint64_t(compute));
  if (compute) {
    MTLComputePipelineDescriptor* c = (MTLComputePipelineDescriptor*)desc;
    k.addFunction(c.computeFunction);
    k.add(uint64_t(c.maxTotalThreadsPerThreadgroup));
    k.add(uint64_t(c.threadGroupSizeIsMultipleOfThreadExecutionWidth));
    // Only when set: descriptors without immutable buffers keep their key (and the archives written for it).
    if (const uint64_t im = immutableSlots(c.buffers)) k.add(im);
    return k.h;
  }
  MTLRenderPipelineDescriptor* d = (MTLRenderPipelineDescriptor*)desc;
  k.addFunction(d.vertexFunction);
  k.addFunction(d.fragmentFunction);
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
  if (d.maxVertexAmplificationCount > 1) k.add(uint64_t(d.maxVertexAmplificationCount));  // only when set: other keys keep their value
  // Equal keys are taken for one pipeline (keyFile: any file with the key serves), so a descriptor with immutable buffers
  // must not share the key of the same one without: an old file would answer for it, miss and compile on every session.
  if (const uint64_t vim = immutableSlots(d.vertexBuffers), fim = immutableSlots(d.fragmentBuffers); vim | fim)
    k.add(vim), k.add(fim);
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

// "<pre><n>.metalar" -> n (n >= 1); pre is "pipelines-" or, for the depth-only archives, "pipelines-depth-".
bool archiveNumber(const std::string& name, uint64_t& n, const std::string& pre = "pipelines-") {
  static const std::string suf = ".metalar";
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

std::filesystem::path MetalBackend::Impl::depthPath(uint64_t n) const {
  return options.cacheRoot / ("pipelines-depth-" + std::to_string(n) + ".metalar");
}

id<MTLBinaryArchive> MetalBackend::Impl::newArchive() {
  NSError* e = nil;
  const attr::Probe probe(device);
  id<MTLBinaryArchive> ar = [device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
  if (!ar) err("what=archive_create error=%s", token(str(e)).c_str());
  attr::trackObject("m2.archive.new", ar, probe);
  return ar;
}

void MetalBackend::Impl::inflightEnter(const char* what) {
  if (options.cacheRoot.empty()) return;
  std::lock_guard lock(inflightMutex);
  if (inflightDepth++) return;
  std::ofstream f(options.cacheRoot / "archive.inflight", std::ios::trunc);
  f << what << '\n';
}

void MetalBackend::Impl::inflightLeave() {
  if (options.cacheRoot.empty()) return;
  std::lock_guard lock(inflightMutex);
  if (--inflightDepth) return;
  std::error_code ec;
  std::filesystem::remove(options.cacheRoot / "archive.inflight", ec);
}

void MetalBackend::Impl::openArchives() {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(options.cacheRoot, ec);
  // Not while any backend of the process adds to / serializes an archive (see processArchiveMutex): also the stale
  // ".tmp" removal below must not take a file another backend is writing.
  std::lock_guard archiveLock(archiveMutex);
  Inflight guard(this, "open");
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
  std::vector<uint64_t> found, depthFound;
  std::vector<fs::path> stale;  // left by an interrupted save or probe
  for (fs::directory_iterator it(options.cacheRoot, ec), end; !ec && it != end; it.increment(ec)) {
    const std::string name = it->path().filename().string();
    uint64_t n = 0;
    if (name == "pipelines.metalar") found.push_back(0);
    else if (archiveNumber(name, n)) found.push_back(n);
    else if (archiveNumber(name, n, "pipelines-depth-")) depthFound.push_back(n);
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
    fs::remove(archive_index::keysPath(p), ec);
    info("gpu.pso.cache evicted file=" + token(p.filename().string()));
    found.erase(found.begin());
    ++evicted;
  }
  archiveEvicted = uint32_t(evicted);
  NSMutableArray<id<MTLBinaryArchive>>* loaded = [NSMutableArray array];
  NSMutableArray* keyed = [NSMutableArray array];
  NSMutableArray<id<MTLBinaryArchive>>* legacy = [NSMutableArray array];
  uint32_t pipelinesLoaded = 0;
  // Budget (MetalOptions::archiveBudgetBytes): most recently modified files first (ties: most entries), since recent
  // sessions hold the pipelines the game uses now; each one's cost predicted from
  // the ratio measured on the ones already open (1x its size before any). `used` is the sum of the footprint growth
  // measured across each kept open (not the process's whole growth, which includes what other threads allocate between
  // the opens), so it is the very quantity the budget limits. The prediction can be low, so it is checked again after
  // the open: a file that pushed `used` over the budget is released and left unopened, never kept. A file left
  // unopened stays on disk (counted, kept for the next session / eviction): its pipelines miss once and compile.
  struct Cand {
    uint64_t n, bytes;
    fs::file_time_type mtime;
    double ageH;  // hours since last modification, for the log
  };
  std::vector<Cand> cands;
  for (uint64_t n : found) {
    const uint64_t b = fs::file_size(archivePath(n), ec);
    const uint64_t bytes = ec ? 0 : b;
    const fs::file_time_type mt = fs::last_write_time(archivePath(n), ec);
    cands.push_back({n, bytes, ec ? fs::file_time_type::min() : mt,
                     ec ? -1.0 : std::chrono::duration<double, std::ratio<3600>>(fs::file_time_type::clock::now() - mt).count()});
  }
  const uint64_t budget = options.archiveBudgetBytes;
  if (budget)
    std::stable_sort(cands.begin(), cands.end(), [&counts](const Cand& a, const Cand& b) {
      if (a.mtime != b.mtime) return a.mtime > b.mtime;
      const auto ca = counts.find(a.n), cb = counts.find(b.n);
      return (ca == counts.end() ? 0u : ca->second) > (cb == counts.end() ? 0u : cb->second);
    });
  const uint64_t foot0 = physFootprint();
  constexpr double kMaxRatio = 32.0;
  double ratio = 1.0;  // footprint growth per archive byte, measured as files open
  uint64_t used = 0, openedBytes = 0, skippedBytes = 0;
  uint32_t opened = 0, skipped = 0;
  for (const Cand& cd : cands) {
    const uint64_t n = cd.n;
    const fs::path p = archivePath(n);
    const auto c = counts.find(n);
    const uint32_t pc = c == counts.end() ? 0 : c->second;
    const uint64_t predicted = uint64_t(double(cd.bytes) * ratio);
    auto skip = [&](const char* reason, uint64_t cost = 0) {
      ++skipped;
      skippedBytes += cd.bytes;
      archiveFileCounts[n] = pc;  // still on disk: counted, kept for the next session / eviction
      info("gpu.pso.cache skipped file=" + token(p.filename().string()) + " bytes=" + std::to_string(cd.bytes) +
           " entries=" + std::to_string(pc) + " age_h=" + std::to_string(int(cd.ageH)) + " reason=" + reason + " predicted_mb=" + std::to_string(predicted >> 20) +
           " cost_mb=" + std::to_string(cost >> 20) + " used_mb=" + std::to_string(used >> 20) + " budget_mb=" + std::to_string(budget >> 20));
    };
    if (budget && used + predicted > budget) {
      skip("budget");
      continue;
    }
    MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
    ad.url = [NSURL fileURLWithPath:@(p.c_str())];
    NSError* ae = nil;
    const attr::Probe probe(device);
    const uint64_t footBefore = physFootprint();
    const auto t0 = std::chrono::steady_clock::now();
    id<MTLBinaryArchive> ar = [device newBinaryArchiveWithDescriptor:ad error:&ae];
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t footAfter = physFootprint();
    attr::trackObject("m2.archive.file", ar, probe);
    if (!ar) {  // stale or from another OS version: remove it, its pipelines are rebuilt
      err("what=archive_load file=%s error=%s", token(p.filename().string()).c_str(), token(str(ae)).c_str());
      fs::remove(p, ec);
      fs::remove(archive_index::keysPath(p), ec);
      continue;
    }
    // A valid reading (both present, growth > 0) is the real cost (Alyx: ~13x the file) and is always used as measured;
    // a negative, zero or missing one is noise from other threads: the prediction stands in for it.
    const uint64_t delta = (footBefore && footAfter > footBefore) ? footAfter - footBefore : predicted;
    if (budget && used + delta > budget) {  // predicted to fit and did not: give it back
      ar = nil;
      // The ratio is not inflated by the rejected file: it stays the mean of the ones opened (or as is while none), so later files that fit are tried.
      if (openedBytes) ratio = std::min(std::max(1.0, double(used) / double(openedBytes)), kMaxRatio);
      skip("budget_after_open", delta);
      continue;
    }
    used += delta;
    openedBytes += cd.bytes;
    ++opened;
    if (openedBytes) ratio = std::min(std::max(1.0, double(used) / double(openedBytes)), kMaxRatio);
    [loaded addObject:ar];
    std::vector<uint64_t> keys;
    if (n && archive_index::read(archive_index::keysPath(p), keys)) {
      const uint32_t idx = uint32_t(keyed.count);
      [keyed addObject:@[ ar ]];
      for (uint64_t k : keys) keyFile[k] = idx;  // the last one opened wins: with a budget that is the oldest (newest-first order), else the newest; equal keys are one pipeline, any serves
    } else {
      [legacy addObject:ar];
    }
    archiveFileCounts[n] = pc;
    pipelinesLoaded += pc;
    info("gpu.pso.cache open file=" + token(p.filename().string()) + " bytes=" + std::to_string(cd.bytes) +
         " entries=" + std::to_string(pc) + " age_h=" + std::to_string(int(cd.ageH)) + " mem_delta_mb=" + std::to_string(delta >> 20) +
         " used_mb=" + std::to_string(used >> 20) + " ratio=" + std::to_string(ratio).substr(0, 4) +
         " ms=" + std::to_string(int(ms)));
  }
  if (!cands.empty()) {
    const uint64_t foot1 = physFootprint();
    info("gpu.pso.cache open_total files=" + std::to_string(opened) + " skipped=" + std::to_string(skipped) +
         " bytes=" + std::to_string(openedBytes) + " skipped_bytes=" + std::to_string(skippedBytes) +
         " used_mb=" + std::to_string(used >> 20) + " proc_mb=" + std::to_string((int64_t(foot1) - int64_t(foot0)) / (1 << 20)) +
         " budget_mb=" + std::to_string(budget >> 20));
  }
  // Depth-only archives: opened like the main ones (their .keys sidecar makes them keyed), outside the memory budget
  // (small files); the oldest beyond kMaxDepthArchiveFiles are evicted. The newest one that is not sealed (fewer than
  // archivePackCap entries, with its .keys sidecar) is continued by this session instead of starting a file per session.
  std::sort(depthFound.begin(), depthFound.end());
  while (depthFound.size() > kMaxDepthArchiveFiles) {
    const fs::path p = depthPath(depthFound.front());
    fs::remove(p, ec);
    fs::remove(archive_index::keysPath(p), ec);
    depthFound.erase(depthFound.begin());
  }
  if (!depthFound.empty()) depthSession = depthFound.back() + 1;
  for (uint64_t n : depthFound) {
    const fs::path p = depthPath(n);
    MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
    ad.url = [NSURL fileURLWithPath:@(p.c_str())];
    NSError* ae = nil;
    const attr::Probe probe(device);
    id<MTLBinaryArchive> ar = [device newBinaryArchiveWithDescriptor:ad error:&ae];
    attr::trackObject("m2.archive.file", ar, probe);
    if (!ar) {  // stale or from another OS version: remove it, its pipelines are rebuilt
      err("what=archive_load file=%s error=%s", token(p.filename().string()).c_str(), token(str(ae)).c_str());
      fs::remove(p, ec);
      fs::remove(archive_index::keysPath(p), ec);
      continue;
    }
    [loaded addObject:ar];
    std::vector<uint64_t> keys;
    const bool haveKeys = archive_index::read(archive_index::keysPath(p), keys);
    if (haveKeys) {
      const uint32_t idx = uint32_t(keyed.count);
      [keyed addObject:@[ ar ]];
      for (uint64_t k : keys) keyFile[k] = idx;
    } else {
      [legacy addObject:ar];
    }
    info("gpu.pso.cache open file=" + token(p.filename().string()) + " depth_only=1 entries=" + std::to_string(keys.size()));
    if (n == depthFound.back() && haveKeys && !keys.empty() && keys.size() < archivePackCap) {
      // A second instance of the file takes the additions (the one in lookupArchives is only read, from compile workers);
      // saveDepthLocked serializes it over the same name through a .tmp rename. Its entries are known by key only.
      NSError* ce = nil;
      const attr::Probe cprobe(device);
      id<MTLBinaryArchive> cont = [device newBinaryArchiveWithDescriptor:ad error:&ce];
      attr::trackObject("m2.archive.file", cont, cprobe);
      if (cont) {
        depthArchive = cont;
        depthSession = n;
        for (uint64_t k : keys) depthEntries.push_back(ArchiveEntry{nil, false, std::string(), k});
        info("gpu.pso.cache continue file=" + token(p.filename().string()) + " depth_only=1 entries=" + std::to_string(keys.size()));
      }
    }
  }
  lookupArchives = [loaded copy];
  keyedArchives = [keyed copy];
  legacyArchives = [legacy copy];
  archiveSession = next;
  archive = newArchive();
  archiveOn = archive != nil;  // without it, lookups in the loaded files still work
  info("gpu.pso.cache loaded=" + std::to_string(pipelinesLoaded) + " path=" + token(options.cacheRoot.string()) +
       " archive=" + std::to_string(archive != nil));
  info("gpu.pso.cache archives=" + std::to_string(lookupArchives.count) + " session=" + std::to_string(archiveSession) +
       " evicted=" + std::to_string(evicted) + " keyed=" + std::to_string(keyed.count) + " keys=" +
       std::to_string(keyFile.size()) + " legacy=" + std::to_string(legacy.count) + " max_files=" +
       std::to_string(kMaxArchiveFiles));
}

void MetalBackend::Impl::writeKeysLocked(uint64_t n, const ArchiveEntry* entries, size_t count) {
  std::vector<uint64_t> keys;
  keys.reserve(count);
  for (size_t i = 0; i < count; ++i) keys.push_back(entries[i].key);
  if (!archive_index::write(archive_index::keysPath(archivePath(n)), keys))
    err("what=archive_keys file=%s error=write_failed", token(archivePath(n).filename().string()).c_str());
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
  size_t queued = 0;
  bool spill = false;
  {
    std::lock_guard lock(archiveQueueMutex);
    spill = archiveQueue.size() >= kArchiveFuse;
    if (!spill) {
      archiveQueue.push_back(ArchiveEntry{desc, compute, label, descriptorKey(desc, compute), std::chrono::steady_clock::now()});
      queued = archiveQueue.size();
      archivePendingN.store(uint32_t(queued), std::memory_order_relaxed);
    }
  }
  if (spill) {
    // The harvest is that far behind (only a fuse against unbounded memory): this one is not archived.
    if (archiveSpilled.fetch_add(1) == 0)
      info("gpu.archive.spill fuse=" + std::to_string(kArchiveFuse) + " spilled=1 (harvest cannot keep up)");
    return;
  }
  if (queued % vpgpu::kHarvestBatch == 0) requestArchiveSave();  // a batch is ready (the job also runs after every compile)
}

// True when the harvest must step aside for a game thread that is blocked on a pipeline (vpgpu::demandWaiters). `oldest`
// is the queue time of the entry about to be added (the queue is FIFO) and `inHand` the entries of its batch still to add.
// Never in the background (the flush must finish) and never when the queue is starving (archiveHarvestYields).
bool MetalBackend::Impl::harvestShouldYield(std::chrono::steady_clock::time_point oldest, size_t inHand) {
  const uint32_t waiters = vpgpu::demandWaiters().load(std::memory_order_acquire);
  if (!waiters || vpgpu::backgroundGate().active()) return false;
  size_t pending = inHand;
  {
    std::lock_guard lock(archiveQueueMutex);
    pending += archiveQueue.size();
  }
  return vpgpu::archiveHarvestYields(waiters, pending, std::chrono::steady_clock::now() - oldest);
}

size_t MetalBackend::Impl::drainArchiveLocked(bool yield, size_t max, bool* yielded) {
  size_t processed = 0;
  while (processed < max) {
    std::vector<ArchiveEntry> batch;
    {
      std::lock_guard lock(archiveQueueMutex);
      const size_t n = std::min({vpgpu::kHarvestBatch, max - processed, archiveQueue.size()});
      batch.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        batch.push_back(std::move(archiveQueue.front()));
        archiveQueue.pop_front();
      }
      archivePendingN.store(uint32_t(archiveQueue.size()), std::memory_order_relaxed);
    }
    if (batch.empty()) break;
    Inflight guard(this, "add");
    for (size_t bi = 0; bi < batch.size(); ++bi) {
      if (yield && harvestShouldYield(batch[bi].queuedAt, batch.size() - bi)) {
        // A game thread is waiting for a pipeline: the rest goes back to the front of the queue for the next window.
        {
          std::lock_guard lock(archiveQueueMutex);
          archiveQueue.insert(archiveQueue.begin(), std::make_move_iterator(batch.begin() + std::ptrdiff_t(bi)),
                              std::make_move_iterator(batch.end()));
          archivePendingN.store(uint32_t(archiveQueue.size()), std::memory_order_relaxed);
        }
        ++archiveYields;
        if (yielded) *yielded = true;
        return processed;
      }
      harvestEntryLocked(batch[bi]);
      ++processed;
    }
  }
  return processed;
}

void MetalBackend::Impl::harvestEntryLocked(ArchiveEntry& q) {
  // A pipeline without its main function cannot be packed ("missing 'vertex' stage").
  const bool complete = q.compute ? [(MTLComputePipelineDescriptor*)q.desc computeFunction] != nil
                                  : [(MTLRenderPipelineDescriptor*)q.desc vertexFunction] != nil;
  if (!complete) {
    ++archiveSkipOther;
    if (archiveWarned.insert(q.label).second)
      info("gpu.pso.cache dropped label=" + token(q.label) +
           " reason=" + (q.compute ? "no_compute_function" : "no_vertex_function"));
    return;
  }
  // Depth-only pipelines (no fragment function) break packing of the whole archive
  // ("expecting 'fragment' stage in pipeline no. N"): they go to an archive of their own (saveDepthLocked), so
  // that failing costs only that one. Without it (it failed, or could not be created) they are not archived.
  if (!q.compute && [(MTLRenderPipelineDescriptor*)q.desc fragmentFunction] == nil) {
    if (archiveSeen.count(q.key)) {
      ++archiveDup;
      return;
    }
    std::string error;
    if (depthEntries.size() >= archivePackCap) saveDepthLocked(true);  // a burst must not make one file too big to pack
    if (!depthOff && !depthArchive) depthArchive = newArchive();
    const auto t0 = std::chrono::steady_clock::now();
    const bool added = !depthOff && depthArchive && addToArchive(depthArchive, q, error);
    archiveAddMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!added) {
      ++archiveSkipDepth;
      if (archiveWarned.insert(q.label).second) {
        info("gpu.pso.cache dropped label=" + token(q.label) + " reason=no_fragment_function");
        if (!error.empty()) info("gpu.archive.depth_fail err=add:" + token(error) + " label=" + token(q.label));
      }
      return;
    }
    archiveSeen.insert(q.key);  // only once it is in: a failed add may be tried again
    depthEntries.push_back(std::move(q));
    ++depthUnsaved;
    ++archiveAdded;
    return;
  }
  if (archiveSeen.count(q.key)) {  // same content already in
    ++archiveDup;
    return;
  }
  if (!archive) archive = buildArchiveLocked(0, archiveEntries.size(), true);
  if (!archive) {  // newArchive logged why; this pipeline is simply not cached
    ++archiveSkipOther;
    return;
  }
  std::string error;
  const auto t0 = std::chrono::steady_clock::now();
  const bool added = addToArchive(archive, q, error);
  archiveAddMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (!added) {
    ++archiveSkipOther;
    err("what=archive_add label=%s error=%s", token(q.label).c_str(), token(error).c_str());
    if (archiveWarned.insert(q.label).second)
      info("gpu.pso.cache dropped label=" + token(q.label) + " reason=add_failed");
    return;
  }
  archiveSeen.insert(q.key);  // only once it is in: a failed add may be tried again
  archiveEntries.push_back(std::move(q));
  ++archiveUnsaved;
  ++archiveAdded;
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
                                          std::string& error, size_t entries) {
  std::filesystem::path tmp = dst;
  tmp += ".tmp";
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  NSError* e = nil;
  const auto t0 = std::chrono::steady_clock::now();
  const bool serialized = [ar serializeToURL:[NSURL fileURLWithPath:@(tmp.c_str())] error:&e];
  const double serMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (serialized) std::filesystem::rename(tmp, dst, ec);
  const double addMs = archiveAddMs;  // adds since the previous line (archiveMutex is held by every caller)
  archiveAddMs = 0;
  // ms = this whole step (serialize + rename), ser_ms = the serialize alone, add_ms = addToArchive time since the last
  // line, pending = entries queued and not added yet, yields = times the harvest stepped aside for a demand wait.
  // First 64 always, then every 8th up to 1024 lines, and the slow ones (> 1 s).
  const uint32_t n = saveLogged.fetch_add(1);
  if (!serialized || ec || n < 64 || (n < 1024 && n % 8 == 0) || (serMs > 1000 && n < 1024)) {
    char b[256];
    std::snprintf(b, sizeof b, "gpu.archive.save ms=%.0f entries=%zu add_ms=%.0f ser_ms=%.0f pending=%u yields=%u ok=%d file=%s",
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), entries, addMs,
                  serMs, archivePendingN.load(std::memory_order_relaxed), archiveYields.load(std::memory_order_relaxed),
                  int(serialized && !ec), token(dst.filename().string()).c_str());
    info(b);
  }
  if (!serialized) {
    error = str(e);
    std::filesystem::remove(tmp, ec);
    return false;
  }
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
  const bool good = serializeArchive(ar, probe, error, end - begin);
  std::error_code ec;
  std::filesystem::remove(probe, ec);
  return good;
}

void MetalBackend::Impl::dropArchiveEntryLocked(size_t i, const char* reason) {
  info("gpu.pso.cache dropped label=" + token(archiveEntries[i].label) + " reason=" + reason);
  --archiveAdded, ++archiveSkipOther;  // it was counted as archived when it was added
  archiveEntries.erase(archiveEntries.begin() + std::ptrdiff_t(i));
  if (i < archiveSavedEntries) {
    --archiveSavedEntries;
    if (archiveSerialized) --archiveSerialized;  // it was on disk and counted there
  }
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

bool MetalBackend::Impl::saveArchiveLocked(bool drain) {
  if (!archiveOn) return false;
  Inflight guard(this, "save");
  if (drain) drainArchiveLocked(false, SIZE_MAX);
  if (!archiveUnsaved) return true;
  if (archiveEntries.size() > archivePackCap) return chunkSaveLocked();
  const auto file = archivePath(archiveSession);
  std::string error;
  for (int drops = 0;; ++drops) {
    // After any failure the archive object is rebuilt from the kept descriptors: a failed
    // serialize can leave it empty ("Nothing to serialize").
    if (!archive && !archiveEntries.empty()) archive = buildArchiveLocked(0, archiveEntries.size(), true);
    if (!archive || archiveEntries.empty()) break;
    if (serializeArchive(archive, file, error, archiveEntries.size())) {
      archiveSerialized += uint32_t(archiveEntries.size() - std::min(archiveSavedEntries, archiveEntries.size()));
      archiveSavedEntries = archiveEntries.size();
      archiveFileCounts[archiveSession] = uint32_t(archiveSavedEntries);
      writeKeysLocked(archiveSession, archiveEntries.data(), archiveEntries.size());
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
    // With an earlier good save the entries added since are saved by the split below (it isolates and drops a single bad
    // one itself): the candidate probes and the bisection are ~8 more serializations on the compiler service, which stalled
    // the game (one pipeline compile waited 849 ms), and found nothing when the entries failed only in combination.
    if (archiveSavedEntries > 0 || drops >= kArchiveDrops || !dropCulpritLocked(error)) break;
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
  return saveArchiveLocked(false) && lost == 0;
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
  if (built && serializeArchive(ar, file, error, n)) {
    archiveFileCounts[archiveSession] = uint32_t(n);
    writeKeysLocked(archiveSession, entries, n);
    ++archiveSession;
    ++files;
    saved += n;
    archiveSerialized += uint32_t(n);
    info("gpu.pso.cache saved_split file=" + token(file.filename().string()) + " entries=" + std::to_string(n) +
         " depth=" + std::to_string(depth));
    return;
  }
  if (n == 1 || depth >= kArchiveSplitDepth || !ar) {
    for (size_t i = 0; i < n; ++i)
      info("gpu.pso.cache dropped label=" + token(entries[i].label) +
           " reason=" + (n == 1 ? "serialize_failed" : "gave_up") + " error=" + token(error));
    lost += n;
    archiveAdded -= uint32_t(n), archiveSkipOther += uint32_t(n);  // counted as archived when added
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

// Saves the depth-only archive (see depthArchive) as pipelines-depth-<n>.metalar + .keys: every 32 additions or 30 s,
// `force` at the final flush. A file holds at most archivePackCap entries (then sealed, the next one starts a new number).
// A failed serialize drops this archive only: the entries added since the last good save count as not archived, the
// session stops archiving depth-only pipelines, and the main archive is untouched.
void MetalBackend::Impl::saveDepthLocked(bool force) {
  if (!depthArchive || depthEntries.empty() || !depthUnsaved) return;
  if (!force && depthUnsaved < 32 && std::chrono::steady_clock::now() - depthSavedAt < std::chrono::seconds(30)) return;
  Inflight guard(this, "save_depth");
  const auto file = depthPath(depthSession);
  std::string error;
  if (!serializeArchive(depthArchive, file, error, depthEntries.size())) {
    info("gpu.archive.depth_fail err=" + token(error) + " file=" + token(file.filename().string()) +
         " entries=" + std::to_string(depthEntries.size()));
    archiveAdded -= depthUnsaved, archiveSkipDepth += depthUnsaved;  // counted as archived when added
    depthArchive = nil;
    depthEntries.clear();
    depthUnsaved = 0;
    depthOff = true;
    return;
  }
  std::vector<uint64_t> keys;
  for (const ArchiveEntry& e : depthEntries) keys.push_back(e.key);
  if (!archive_index::write(archive_index::keysPath(file), keys))
    err("what=archive_keys file=%s error=write_failed", token(file.filename().string()).c_str());
  archiveSerialized += depthUnsaved;
  depthUnsaved = 0;
  depthSavedAt = std::chrono::steady_clock::now();
  info("gpu.pso.cache saved_depth file=" + token(file.filename().string()) + " entries=" + std::to_string(depthEntries.size()));
  if (depthEntries.size() >= archivePackCap) {  // seal: the next depth-only pipelines start a new file
    ++depthSession;
    depthEntries.clear();
    depthArchive = nil;
  }
}

// On the archive save queue (never a compile worker, utility QoS): adds this session's misses to the archive, vpgpu::kHarvestBatch
// at a time, until the queue is empty, and saves every 32 additions or 30 s. It does not wait for the compiles to calm down (UE
// never does): it steps aside only while a game thread is blocked on a pipeline (vpgpu::demandWaiters), unless the queue is
// starving (archiveHarvestYields), also before a serialize. archiveMutex is taken per batch, so a foreground saveArchive() gets
// in (archiveFlushWaiters also makes the job stop). In the background it flushes (flushArchiveLocked, 4 s at most) and never
// yields. Entries not added are lost if the process dies; saveArchive() at exit and the pause hook flush them. True when it
// yielded: the save job looks again shortly.
bool MetalBackend::Impl::maybeSaveArchive() {
  if (!archiveOn) return false;
  if (vpgpu::backgroundGate().active()) {
    std::lock_guard lock(archiveMutex);
    flushArchiveLocked(std::chrono::seconds(4));
    return false;
  }
  for (;;) {
    if (closing || archiveFlushWaiters.load(std::memory_order_acquire)) return false;
    std::lock_guard lock(archiveMutex);
    bool yielded = false;
    const size_t added = drainStepLocked(true, &yielded);
    if (yielded) return true;  // a game thread waits for a pipeline: no serialize now either
    const bool depthDue = depthUnsaved && (depthUnsaved >= 32 || std::chrono::steady_clock::now() - depthSavedAt >= std::chrono::seconds(30));
    if (depthDue || archiveSaveDue()) {
      // Serializing is the heavy part: it waits for the demand wait too (the entries stay added, the next pass saves).
      std::chrono::steady_clock::time_point oldest = std::chrono::steady_clock::now();
      {
        std::lock_guard ql(archiveQueueMutex);
        if (!archiveQueue.empty()) oldest = archiveQueue.front().queuedAt;
      }
      if (harvestShouldYield(oldest, 0)) {
        ++archiveYields;
        return true;
      }
      saveDepthLocked(false);
      if (archiveSaveDue()) saveArchiveLocked(false);
    }
    if (!added) return false;
  }
}

// One harvest step: up to kHarvestBatch entries, never more than fit in the session file (archivePackCap), so that a save
// always finds a single file's worth and never has to re-add entries by chunking / splitting.
size_t MetalBackend::Impl::drainStepLocked(bool yield, bool* yielded) {
  const size_t room = archiveEntries.size() < archivePackCap ? archivePackCap - archiveEntries.size() : 1;
  return drainArchiveLocked(yield, std::min(vpgpu::kHarvestBatch, room), yielded);
}

// A serialize is due: 32 added since the last one, the session file full, or 30 s.
bool MetalBackend::Impl::archiveSaveDue() const {
  return archiveUnsaved &&
         (archiveUnsaved >= 32 || archiveEntries.size() >= archivePackCap ||
          std::chrono::steady_clock::now() - archiveSavedAt >= std::chrono::seconds(30));
}

// Final / background flush: adds the queue and serializes as it goes (a step, then a save when due), never yielding, until it is
// empty or `budget` is spent (what is left stays queued; the recipes bring it back next run). Everything added is serialized
// before it returns. True when nothing failed.
bool MetalBackend::Impl::flushArchiveLocked(std::chrono::milliseconds budget) {
  if (!archiveOn) return false;
  Inflight guard(this, "flush");
  const auto t0 = std::chrono::steady_clock::now();
  bool ok = true, cut = false;
  for (;;) {
    const size_t n = drainStepLocked(false, nullptr);
    if (archiveSaveDue()) ok = saveArchiveLocked(false) && ok;
    saveDepthLocked(false);
    if (!n) break;
    if (std::chrono::steady_clock::now() - t0 >= budget) {
      cut = true;
      break;
    }
  }
  ok = saveArchiveLocked(false) && ok;
  saveDepthLocked(true);
  if (cut) {
    const uint32_t left = archivePendingN.load(std::memory_order_relaxed);
    if (left)
      info("gpu.archive.flush_cut pending=" + std::to_string(left) + " budget_ms=" + std::to_string(budget.count()));
  }
  return ok;
}

void MetalBackend::Impl::requestArchiveSave() {
  if (!archiveOn) return;
  {
    std::lock_guard lock(saveMutex);
    if (closing) return;  // read under the lock: waitArchiveSaveJob() runs after closing is set
    saveWant = true;
    if (saveRunning) return;  // the job loops again when it finishes
    saveRunning = true;
    if (!saveQueue) {
      dispatch_queue_attr_t a = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0);
      saveQueue = dispatch_queue_create("vpgpu.archive.save", a);
    }
  }
  dispatch_async(saveQueue, ^{ runArchiveSaveJob(); });
}

void MetalBackend::Impl::runArchiveSaveJob() {
  for (;;) {
    {
      std::lock_guard lock(saveMutex);
      if (!saveWant || closing) {
        saveWant = false;
        saveRunning = false;
        saveCv.notify_all();
        return;
      }
      saveWant = false;
    }
    bool postponed = false;
    @autoreleasepool { postponed = maybeSaveArchive(); }
    if (postponed) {
      // Compiles are busy: look again shortly, from the queue and without holding a thread. saveRunning stays set, so
      // requests that come meanwhile only mark saveWant, and waitArchiveSaveJob() returns once this block has run
      // (closing is set by then: the block ends the job).
      dispatch_after(dispatch_time(DISPATCH_TIME_NOW, int64_t(250) * int64_t(NSEC_PER_MSEC)), saveQueue, ^{
        {
          std::lock_guard lock(saveMutex);
          saveWant = true;
        }
        runArchiveSaveJob();
      });
      return;
    }
  }
}

void MetalBackend::Impl::waitArchiveSaveJob() {
  std::unique_lock lock(saveMutex);
  saveWant = false;
  saveCv.wait(lock, [&] { return !saveRunning; });
}

MetalBackend::MemStats MetalBackend::memStats() const {
  MemStats s;
  Impl& m = *impl_;
  if (!ok()) return s;
  ScopedPool pool;
  s.deviceAllocated = m.device.currentAllocatedSize;
  s.workingSetMax = m.device.recommendedMaxWorkingSetSize;
  s.texAstc = m.texBytes[Impl::kTexAstc], s.texColor = m.texBytes[Impl::kTexColor];
  s.texDepth = m.texBytes[Impl::kTexDepth], s.texMsaa = m.texBytes[Impl::kTexMsaa];
  s.msaaMemoryless = m.msaaDeferredBytes;
  {
    std::lock_guard lock(m.ledgerMutex);
    s.ledger = m.ledgerLast;
  }
  s.bufOwn = m.bufOwnBytes, s.bufGuest = m.bufGuestBytes;
  s.heldBytes = m.cnt.heldBytes.load(std::memory_order_relaxed);
  s.views = m.liveViews.load(std::memory_order_relaxed);
  s.argsBytes = m.argsRingBytes.load(std::memory_order_relaxed) + m.argsMoreBytes.load(std::memory_order_relaxed);
  s.bufIntBytes = m.bufIntLiveBytes.load(std::memory_order_relaxed);
  s.fovBytes = m.fovShadowBytes.load(std::memory_order_relaxed);
  s.tileRejected = m.tileRejectedTotal;
  {
    std::lock_guard lock(m.pipelineMutex);
    s.pipelines = uint32_t(m.pipelines.size());
    s.functions = s.libraries = uint32_t(m.functions.size());
  }
  {
    std::lock_guard lock(m.libStatsMutex);
    s.libCompiled = m.libCompiled;
  }
  // The archive lock is held across Metal compiles and serializes: never wait for it here.
  if (std::unique_lock lock(m.archiveMutex, std::try_to_lock); lock.owns_lock())
    m.archiveEntriesSeen = uint32_t(m.archiveEntries.size());
  s.archiveEntries = m.archiveEntriesSeen;
  s.archiveFiles = uint32_t(m.lookupArchives.count);
  s.archiveHits = m.psoArchived, s.archiveMisses = m.psoHot;
  s.archiveAdded = m.archiveAdded, s.archiveSkipDepth = m.archiveSkipDepth;
  s.archiveSkipOther = m.archiveSkipOther, s.archiveDup = m.archiveDup;
  s.archiveSerialized = m.archiveSerialized, s.archiveSpilled = m.archiveSpilled;
  s.archivePending = m.archivePendingN + (s.archiveAdded > s.archiveSerialized ? s.archiveAdded - s.archiveSerialized : 0);
  s.archiveKeyed = uint32_t(m.keyedArchives.count), s.archiveEvicted = m.archiveEvicted;
  return s;
}

// Final / background flush: adds everything still queued (never yielding to demand waits) and serializes it.
bool MetalBackend::saveArchive() {
  @autoreleasepool {
    // The harvest job lets this in: it stops between its batches while a flush waits for archiveMutex.
    impl_->archiveFlushWaiters.fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard lock(impl_->archiveMutex);
    impl_->archiveFlushWaiters.fetch_sub(1, std::memory_order_acq_rel);
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t pending = impl_->archivePendingN.load(std::memory_order_relaxed);
    // Step by step (add, serialize when due), 8 s at most: what is left stays queued and the recipes bring it back next run.
    const bool saved = impl_->flushArchiveLocked(std::chrono::seconds(8));
    if (pending)
      impl_->info("gpu.archive.flush pending=" + std::to_string(pending) + " ms=" +
                  std::to_string(int64_t(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count())));
    return saved;
  }
}

}  // namespace vpgpu::metal2

// Found by name with dlsym (the host resolves them at run time and falls back when they are missing).
// The next real frame is dumped pass by pass, like VP_DUMP_FRAME (needs VP_DUMP_DIR; the request stays pending without it).
extern "C" __attribute__((visibility("default"))) void vpgpu_metal2_request_dump(void) {
  vpgpu::metal2::gDumpRequest.store(true);
}
// Splits the two-layer vertex-amplified passes into one encoder per view (CompileOptions::splitViews; 0 = off, the default, or
// VPVK_SPLIT_VIEWS=1 at start). Atomic, read by the next frame that is submitted; no pipeline is built or rebuilt by a change.
extern "C" __attribute__((visibility("default"))) void vpgpu_metal2_set_split_views(int on) {
  vpgpu::splitViewsSwitch().store(on != 0 ? 1 : 0, std::memory_order_relaxed);
}
// Heavy-geometry vertex limit (CompileOptions::tileMaxVertices; VPVK_TILE_MAX_VERTICES at start) in hot: 0 = back to the start value.
// Atomic, read by the next frame that is submitted. A value above the start value is first tried on ONE frame (the canary: the first
// one with an encoder only that value keeps light) and applied for good if its command buffers complete; a memoryless / "Too much
// geometry" error there (or later under it) puts the start value back until the next set. vpgpu_metal2_tile_reverted() is 1 then.
extern "C" __attribute__((visibility("default"))) void vpgpu_metal2_set_tile_max_vertices(uint64_t n) {
  vpgpu::setTileMaxVertices(n);
}
extern "C" __attribute__((visibility("default"))) int vpgpu_metal2_tile_reverted(void) {
  return vpgpu::tileMaxVerticesReverted().load(std::memory_order_relaxed);
}
// VP_KNOCK=auto starts idle: this starts the run (warm-up) at the start of the next frame, when the knock is enabled and idle or
// done (a run in progress ignores it). Without VP_KNOCK it does nothing.
extern "C" __attribute__((visibility("default"))) void vpgpu_metal2_knock_arm(void) {
  using namespace vpgpu::metal2;
  if (gKnockEnv.load()) gKnockArmReq.store(true);
}
// 0 when VP_KNOCK is off; else 1 with *phase (0 idle, 1 warm-up, 2 measure, 3 drain, 4 done, 5 done by timeout) and *done / *total
// (warm-up frames counted / 300 in warm-up; blocks finished / total blocks in measure and drain). Any pointer may be null.
extern "C" __attribute__((visibility("default"))) int vpgpu_metal2_knock_progress(int* phase, int* done, int* total) {
  using namespace vpgpu::metal2;
  if (!gKnockEnv.load()) return 0;
  if (phase) *phase = gKnockPhase.load();
  if (done) *done = gKnockDone.load();
  if (total) *total = gKnockTotal.load();
  return 1;
}
// GPU hangs this process caused so far (command buffers the watchdog killed; refused submissions and victims of another client's
// hang do not count). The host's A/B test reads it to see whether a setting hangs the GPU.
extern "C" __attribute__((visibility("default"))) uint32_t vpgpu_metal2_gpu_hangs(void) {
  return vpgpu::metal2::gGpuHangs.load();
}
// The exclusive GPU time (ms) of a command buffer that completed on our queue, consumed: 1 and *ms set when known, 0 when
// none of its encoders was sampled, -1 when the command buffer is not (or no longer) in the ring. Call it from a completed
// handler added after ours (the host hooks -commit, which runs after we add ours). The first call turns the sampling on:
// frames encoded before it answer -1.
extern "C" __attribute__((visibility("default"))) int vpgpu_metal2_cb_busy_ms(const void* cb, double* ms) {
  using namespace vpgpu::metal2;
  gBusyOn.store(true, std::memory_order_relaxed);
  if (!cb) return -1;  // an empty or consumed entry holds a null cb
  std::lock_guard lock(gBusyMutex);
  for (size_t i = 0; i < kBusyRing; ++i) {
    BusyEntry& e = gBusyRing[i];
    if (e.cb != cb || e.msTaken) continue;
    const double v = e.ms;
    e.msTaken = true;
    if (e.ivTaken) e.cb = nullptr;
    if (v < 0) return 0;
    if (ms) *ms = v;
    return 1;
  }
  return -1;
}
// The GPU timestamp pairs (start, end, interleaved, one per sampled encoder in encoder order; stage-adjusted on render
// encoders like the ones vpgpu_metal2_cb_busy_ms sums, not clipped against the previous command buffers) of a command
// buffer that completed on our queue, for the host's overlap accounting; *ns_per_tick is the calibration
// vpgpu_metal2_cb_busy_ms uses. Pairs beyond cap_pairs / 64 are dropped silently. Returns the number of pairs written
// (<= cap_pairs), 0 when unknown (none sampled, not in the ring, or already asked for). Consumed like
// vpgpu_metal2_cb_busy_ms, independently of it (either order works); the first call turns the sampling on too. A null
// out_pairs (or cap_pairs 0) still consumes the entry and returns 0.
extern "C" __attribute__((visibility("default"))) size_t vpgpu_metal2_cb_busy_intervals(const void* cb, uint64_t* out_pairs, size_t cap_pairs,
                                                                                          double* ns_per_tick) {
  using namespace vpgpu::metal2;
  gBusyOn.store(true, std::memory_order_relaxed);
  if (!cb) return 0;  // an empty or consumed entry holds a null cb
  std::lock_guard lock(gBusyMutex);
  for (size_t i = 0; i < kBusyRing; ++i) {
    BusyEntry& e = gBusyRing[i];
    if (e.cb != cb || e.ivTaken) continue;
    e.ivTaken = true;
    const size_t n = out_pairs ? std::min<size_t>(e.nPairs, cap_pairs) : 0;
    for (size_t k = 0; k < n; ++k) out_pairs[2 * k] = e.pairs[k][0], out_pairs[2 * k + 1] = e.pairs[k][1];
    if (ns_per_tick) *ns_per_tick = e.nsPerTick;
    if (e.msTaken) e.cb = nullptr;
    return n;
  }
  return 0;
}
// Command buffers accounted so far and how many had no sampled encoder (the host logs them).
extern "C" __attribute__((visibility("default"))) void vpgpu_metal2_cb_busy_stats(uint64_t* cbs, uint64_t* unsampled) {
  vpgpu::metal2::gBusyOn.store(true, std::memory_order_relaxed);
  if (cbs) *cbs = vpgpu::metal2::gBusyCbs.load();
  if (unsampled) *unsampled = vpgpu::metal2::gBusyUnsampled.load();
}
