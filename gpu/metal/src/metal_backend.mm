// Metal backend. Classic Metal 3 command encoding (MTLCommandQueue, render/compute/blit encoders,
// MTLFence) with Metal 3 bindless (gpuAddress / gpuResourceID argument tables) and a residency set.
// All resources are hazard-untracked: ordering comes only from the barriers the frame compiler
// placed, so Metal adds no implicit synchronization of its own.
// Metal 4 replacement points are marked "M4:".
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "vpgpu/metal/metal_backend.h"

#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

#include "vpgpu/core/pipeline_cache.h"

namespace vpgpu::metal {

void ShaderTable::add(uint64_t id, MslShader shader) {
  std::lock_guard lock(mutex_);
  shaders_[id] = std::move(shader);
}

bool ShaderTable::find(uint64_t id, MslShader& out) const {
  std::lock_guard lock(mutex_);
  auto it = shaders_.find(id);
  if (it == shaders_.end()) return false;
  out = it->second;
  return true;
}

namespace {

constexpr MTLResourceOptions kSharedUntracked = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
constexpr uint64_t kArgRingBytes = 1 << 20;   // per frame slot
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
      MTLBlendFactorOneMinusDestinationAlpha, MTLBlendFactorBlendColor};
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
MTLStoreAction storeAction(StoreAction a) { return a == StoreAction::Store ? MTLStoreActionStore : MTLStoreActionDontCare; }

std::string str(NSError* e) {
  return e ? std::string(e.localizedDescription.UTF8String ?: "?") : std::string("unknown error");
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
    id<MTLTexture> texture;
    Format format = Format::Undefined;
  };
  struct Sampler {
    uint32_t generation = 0;
    id<MTLSamplerState> sampler;
  };
  struct Pipeline {
    id<MTLRenderPipelineState> render;
    id<MTLDepthStencilState> depth;
    MTLCullMode cull = MTLCullModeNone;
    id<MTLComputePipelineState> compute;
    MTLSize threads{1, 1, 1};
    uint32_t argIndex = 30;
  };
  // Per frame in flight: things the GPU may still read until that frame completes.
  struct Slot {
    id<MTLBuffer> args;
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

  std::mutex pipelineMutex;  // guards pipelines, functions, archive
  std::unordered_map<uint64_t, Pipeline> pipelines;
  std::unordered_map<uint64_t, id<MTLFunction>> functions;
  id<MTLBinaryArchive> archive;

  std::mutex doneMutex;
  std::deque<Done> done;

  // Fences of encoders not yet ordered before any later encoder. A barrier waits on all of them
  // and leaves only itself, so ordering is transitive and each barrier costs few waits. It carries
  // over between frames: a frame's first encoder waits on the previous frame's tail.
  std::vector<id<MTLFence>> frontier;

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
  id<MTLFunction> function(uint64_t id, MslShader& shader, std::string& error);
  bool buildRender(const PipelineState&, Pipeline&, std::string& error);
  bool buildCompute(const PipelineState&, Pipeline&, std::string& error);
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&);
};

void MetalBackend::Impl::init() {
  device = MTLCreateSystemDefaultDevice();
  if (!device) {
    err("what=no_device");
    return;
  }
  queue = [device newCommandQueue];
  queue.label = @"vpgpu";

  MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
  rd.label = @"vpgpu.resources";
  NSError* e = nil;
  residency = [device newResidencySetWithDescriptor:rd error:&e];
  if (residency) [queue addResidencySet:residency];
  else err("what=residency_set error=%s", token(str(e)).c_str());

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

  if (!options.cacheRoot.empty()) {
    NSError* ae = nil;
    std::error_code ec;
    std::filesystem::create_directories(options.cacheRoot, ec);
    const auto path = options.cacheRoot / "pipelines.metalar";
    MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
    if (std::filesystem::exists(path, ec)) ad.url = [NSURL fileURLWithPath:@(path.c_str())];
    archive = [device newBinaryArchiveWithDescriptor:ad error:&ae];
    if (!archive && ad.url) {  // stale or from another OS version: start empty
      err("what=archive_load error=%s", token(str(ae)).c_str());
      ad.url = nil;
      archive = [device newBinaryArchiveWithDescriptor:ad error:&ae];
    }
    if (!archive) err("what=archive_create error=%s", token(str(ae)).c_str());
  }

  info("gpu.metal device=" + token(device.name.UTF8String) + " stage_timestamps=" +
       std::to_string(stageSamples) + " residency=" + std::to_string(residency != nil) +
       " archive=" + std::to_string(archive != nil));
}

void MetalBackend::Impl::calibrate() {
  MTLTimestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
  [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
  [NSThread sleepForTimeInterval:0.002];
  [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
  if (gpu1 > gpu0) gpuTickNs = double(cpu1 - cpu0) / double(gpu1 - gpu0);  // CPU side is in ns
}

void MetalBackend::Impl::retire(id object) { graveyard.push_back(object); }

id<MTLFunction> MetalBackend::Impl::function(uint64_t id, MslShader& shader, std::string& error) {
  if (!shaders.find(id, shader)) {
    error = "shader " + std::to_string(id) + " not found";
    return nil;
  }
  {
    std::lock_guard lock(pipelineMutex);
    auto it = functions.find(id);
    if (it != functions.end()) return it->second;
  }
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.mathMode = MTLMathModeFast;  // the emitter marks precise math with precise:: explicitly
  o.languageVersion = MTLLanguageVersion3_2;
  NSError* e = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:@(shader.source.c_str()) options:o error:&e];
  if (!lib) {
    error = "msl compile: " + str(e);
    return nil;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:@(shader.entry.c_str())];
  if (!fn) {
    error = "entry " + shader.entry + " not in library";
    return nil;
  }
  std::lock_guard lock(pipelineMutex);
  functions[id] = fn;
  return fn;
}

bool MetalBackend::Impl::buildRender(const PipelineState& s, Pipeline& p, std::string& error) {
  MslShader vs, fs;
  MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
  d.label = @(s.label.c_str());
  d.vertexFunction = function(s.vertexShader, vs, error);
  if (!d.vertexFunction) return false;
  if (s.fragmentShader) {  // none = depth-only
    d.fragmentFunction = function(s.fragmentShader, fs, error);
    if (!d.fragmentFunction) return false;
  }
  for (int i = 0; i < kMaxColorAttachments; ++i) {
    MTLRenderPipelineColorAttachmentDescriptor* c = d.colorAttachments[i];
    const BlendState& b = s.blend[i];
    c.pixelFormat = pixelFormat(s.colorFormats[i]);
    c.writeMask = writeMask(b.writeMask);
    c.blendingEnabled = b.enabled;
    c.sourceRGBBlendFactor = blendFactor(b.srcColor);
    c.destinationRGBBlendFactor = blendFactor(b.dstColor);
    c.sourceAlphaBlendFactor = blendFactor(b.srcAlpha);
    c.destinationAlphaBlendFactor = blendFactor(b.dstAlpha);
    c.rgbBlendOperation = blendOp(b.colorOp);
    c.alphaBlendOperation = blendOp(b.alphaOp);
  }
  const FormatInfo& df = formatInfo(s.depthFormat);
  const MTLPixelFormat dsf = pixelFormat(s.depthFormat);
  if (df.depth()) d.depthAttachmentPixelFormat = dsf;
  if (df.stencil()) d.stencilAttachmentPixelFormat = dsf;
  d.rasterSampleCount = s.samples;
  d.alphaToCoverageEnabled = s.alphaToCoverage;
  d.inputPrimitiveTopology = topologyClass(s.primitive);

  NSError* e = nil;
  {
    std::lock_guard lock(pipelineMutex);  // MTLBinaryArchive is not documented as thread-safe
    if (archive) {
      d.binaryArchives = @[ archive ];
      if (![archive addRenderPipelineFunctionsWithDescriptor:d error:&e])
        err("what=archive_add label=%s error=%s", token(s.label).c_str(), token(str(e)).c_str());
    }
  }
  p.render = [device newRenderPipelineStateWithDescriptor:d error:&e];
  if (!p.render) {
    error = "render pipeline: " + str(e);
    return false;
  }
  MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
  ds.depthCompareFunction = s.depthTest ? compareFn(s.depthCompare) : MTLCompareFunctionAlways;
  ds.depthWriteEnabled = s.depthTest && s.depthWrite;
  p.depth = [device newDepthStencilStateWithDescriptor:ds];
  p.cull = s.cull == CullMode::Front ? MTLCullModeFront : s.cull == CullMode::Back ? MTLCullModeBack : MTLCullModeNone;
  p.argIndex = vs.argumentBufferIndex;
  if (s.fragmentShader && fs.argumentBufferIndex != vs.argumentBufferIndex) {
    error = "vertex and fragment argument table indices differ";
    return false;
  }
  return true;
}

bool MetalBackend::Impl::buildCompute(const PipelineState& s, Pipeline& p, std::string& error) {
  MslShader cs;
  MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
  d.label = @(s.label.c_str());
  d.computeFunction = function(s.computeShader, cs, error);
  if (!d.computeFunction) return false;
  NSError* e = nil;
  {
    std::lock_guard lock(pipelineMutex);
    if (archive) {
      d.binaryArchives = @[ archive ];
      if (![archive addComputePipelineFunctionsWithDescriptor:d error:&e])
        err("what=archive_add label=%s error=%s", token(s.label).c_str(), token(str(e)).c_str());
    }
  }
  p.compute = [device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&e];
  if (!p.compute) {
    error = "compute pipeline: " + str(e);
    return false;
  }
  p.threads = MTLSizeMake(cs.threadsPerGroup[0], cs.threadsPerGroup[1], cs.threadsPerGroup[2]);
  p.argIndex = cs.argumentBufferIndex;
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

}  // namespace

void MetalBackend::Impl::submit(uint64_t frameIndex, const Frame& frame, const CompiledFrame& compiled) {
  if (compiled.encoders.empty()) {  // nothing for the GPU; keeps the frontier's fences out of reuse
    std::lock_guard lock(doneMutex);
    done.push_back({frameIndex, {}});
    return;
  }
  dispatch_semaphore_wait(inFlight, DISPATCH_TIME_FOREVER);
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

  MTLCommandBufferDescriptor* cbd = [MTLCommandBufferDescriptor new];
  cbd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;  // per-encoder fault in the log
  id<MTLCommandBuffer> cb = [queue commandBufferWithDescriptor:cbd];
  cb.label = [NSString stringWithFormat:@"vpgpu.f%llu", (unsigned long long)frameIndex];

  const auto& cmds = frame.commands();
  uint8_t* argBase = static_cast<uint8_t*>(slot.args.contents);
  uint64_t argUsed = 0;
  uint32_t missingPipelines = 0, missingResources = 0, argOverflow = 0;
  size_t fenceUsed = 0;
  NSUInteger sampleUsed = 0;
  // Per core encoder: first start sample and last end sample (NSNotFound = unsampled).
  std::vector<std::pair<NSUInteger, NSUInteger>> encSamples(compiled.encoders.size(), {NSNotFound, NSNotFound});

  // Writes one draw's/dispatch's argument table; returns its offset in slot.args or -1.
  auto table = [&](const std::vector<Access>& bindings) -> int64_t {
    const uint64_t bytes = std::max<uint64_t>(8, bindings.size() * 8);
    if (argUsed + bytes > kArgRingBytes) {
      ++argOverflow;
      return -1;
    }
    uint64_t* t = reinterpret_cast<uint64_t*>(argBase + argUsed);
    for (size_t i = 0; i < bindings.size(); ++i) {
      const ResourceRange& r = bindings[i].range;
      t[i] = 0;
      if (r.kind == ResourceRange::Kind::Buffer) {
        auto it = buffers.find(r.index);
        if (it != buffers.end() && it->second.generation == r.generation)
          t[i] = it->second.buffer.gpuAddress + it->second.base + r.begin;
      } else {
        auto it = textures.find(r.index);
        if (it != textures.end() && it->second.generation == r.generation) {
          const MTLResourceID rid = it->second.texture.gpuResourceID;
          std::memcpy(&t[i], &rid, sizeof rid);
        }
      }
      missingResources += t[i] == 0;
    }
    const int64_t at = static_cast<int64_t>(argUsed);
    argUsed += (bytes + kArgAlign - 1) / kArgAlign * kArgAlign;
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
  auto texture = [&](TextureId t) -> id<MTLTexture> {
    auto it = textures.find(t.index);
    return it == textures.end() || it->second.generation != t.generation ? nil : it->second.texture;
  };
  auto formatOf = [&](TextureId t) { return formatInfo(textures.at(t.index).format); };
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
  auto close = [&]() {
    if (open.kind == Open::None) return;
    if (open.render) [open.render updateFence:openFence afterStages:MTLRenderStageFragment];
    else if (open.compute) [open.compute updateFence:openFence];
    else [open.blit updateFence:openFence];
    [open.any() endEncoding];
    frontier.push_back(openFence);
    open = {};
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
    close();

    if (enc.kind == Encoder::Kind::Render) {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      auto attach = [&](MTLRenderPassAttachmentDescriptor* a, const Attachment& src) {
        a.texture = texture(src.texture);
        a.level = src.mip;
        if (a.texture.textureType == MTLTextureType3D) a.depthPlane = src.layer;
        else a.slice = src.layer;
        a.loadAction = loadAction(src.load);
        a.storeAction = storeAction(src.store);
      };
      for (int i = 0; i < kMaxColorAttachments; ++i) {
        const Attachment& c = enc.pass.colors[i];
        if (!c.texture) continue;
        attach(rp.colorAttachments[i], c);
        rp.colorAttachments[i].clearColor = MTLClearColorMake(c.clear.color[0], c.clear.color[1], c.clear.color[2], c.clear.color[3]);
      }
      if (const Attachment& d = enc.pass.depth; d.texture) {
        id<MTLTexture> t = texture(d.texture);
        const MTLPixelFormat pf = t.pixelFormat;
        const bool hasStencil = pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatStencil8;
        if (pf != MTLPixelFormatStencil8) {
          attach(rp.depthAttachment, d);
          rp.depthAttachment.clearDepth = d.clear.depth;
        }
        if (hasStencil) {
          attach(rp.stencilAttachment, d);
          rp.stencilAttachment.clearStencil = d.clear.stencil;
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
      open.kind = Open::Render;
      open.render = [cb renderCommandEncoderWithDescriptor:rp];
      if (!open.render) {
        err("what=render_encoder f=%llu enc=%u", (unsigned long long)frameIndex, e);
        open = {};
        continue;
      }
      opened([NSString stringWithFormat:@"render%u", e], boundary, before);
      for (uint32_t ci : enc.commands) {
        for (const Draw& d : std::get<RenderPass>(cmds[ci]).draws) {
          const Pipeline* p = pipeline(d.pipeline);
          if (!p || !p->render) {
            ++missingPipelines;
            continue;
          }
          const int64_t at = table(d.bindings);
          if (at < 0) continue;
          [open.render setRenderPipelineState:p->render];
          [open.render setDepthStencilState:p->depth];
          [open.render setCullMode:p->cull];
          [open.render setVertexBuffer:slot.args offset:at atIndex:p->argIndex];
          [open.render setFragmentBuffer:slot.args offset:at atIndex:p->argIndex];
          if (d.index) {
            uint64_t off = 0;
            BufferSpan span{d.index->buffer, d.index->offset, 0, 0, 0};
            id<MTLBuffer> ib = buffer(span, off);
            if (!ib) {
              ++missingResources;
              continue;
            }
            const uint64_t stride = d.index->wide ? 4 : 2;
            [open.render drawIndexedPrimitives:primitiveType(d.primitive)
                                    indexCount:d.count
                                     indexType:d.index->wide ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                                   indexBuffer:ib
                             indexBufferOffset:off + uint64_t{d.first} * stride
                                 instanceCount:d.instances
                                    baseVertex:d.baseVertex
                                  baseInstance:0];
          } else {
            [open.render drawPrimitives:primitiveType(d.primitive)
                            vertexStart:d.first
                            vertexCount:d.count
                          instanceCount:d.instances];
          }
        }
      }
      continue;
    }

    // Compute encoder: dispatches and copies in order, barriers between items where placed.
    for (uint32_t k = 0; k < enc.commands.size(); ++k) {
      const Command& cmd = cmds[enc.commands[k]];
      const bool isDispatch = std::holds_alternative<Dispatch>(cmd);
      bool wait = k == 0 && boundary;
      StageMask waitBefore = before;
      if (k > 0 && b < compiled.barriers.size() && compiled.barriers[b].encoder == e && compiled.barriers[b].item == k) {
        waitBefore = compiled.barriers[b++].before;
        // A memory barrier only orders work inside this encoder; earlier unordered encoders
        // (frontier) need a fence wait, which means a new encoder.
        if (open.kind == Open::Compute && isDispatch && frontier.empty())
          [open.compute memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
        else wait = true;
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
        const int64_t at = table(d.bindings);
        if (at < 0) continue;
        [open.compute setComputePipelineState:p->compute];
        [open.compute setBuffer:slot.args offset:at atIndex:p->argIndex];
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
        const uint32_t bw = (t.width + fi.blockSize - 1) / fi.blockSize, bh = (t.height + fi.blockSize - 1) / fi.blockSize;
        row = s.bytesPerRow ? s.bytesPerRow : bw * fi.bytesPerBlock;
        image = row * (s.rowsPerImage ? s.rowsPerImage : bh);
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
          for (uint16_t l = 0; l < s.layerCount; ++l)
            [open.blit copyFromTexture:st
                           sourceSlice:s.layer + l
                           sourceLevel:s.mip
                          sourceOrigin:MTLOriginMake(s.x, s.y, s.z)
                            sourceSize:MTLSizeMake(s.width, s.height, s.depth)
                             toTexture:dt
                      destinationSlice:d.layer + l
                      destinationLevel:d.mip
                     destinationOrigin:MTLOriginMake(d.x, d.y, d.z)];
          continue;
        }
        case CopyKind::BufferToTexture: {
          if (!sb || !dt) break;
          const TextureSpan& d = c.dstTexture;
          NSUInteger row, image;
          pitch(c.srcBuffer, d, row, image);
          for (uint16_t l = 0; l < d.layerCount; ++l)
            [open.blit copyFromBuffer:sb
                         sourceOffset:so + l * image * d.depth
                    sourceBytesPerRow:row
                  sourceBytesPerImage:image
                           sourceSize:MTLSizeMake(d.width, d.height, d.depth)
                            toTexture:dt
                     destinationSlice:d.layer + l
                     destinationLevel:d.mip
                    destinationOrigin:MTLOriginMake(d.x, d.y, d.z)];
          continue;
        }
        case CopyKind::TextureToBuffer: {
          if (!st || !db) break;
          const TextureSpan& s = c.srcTexture;
          NSUInteger row, image;
          pitch(c.dstBuffer, s, row, image);
          for (uint16_t l = 0; l < s.layerCount; ++l)
            [open.blit copyFromTexture:st
                           sourceSlice:s.layer + l
                           sourceLevel:s.mip
                          sourceOrigin:MTLOriginMake(s.x, s.y, s.z)
                            sourceSize:MTLSizeMake(s.width, s.height, s.depth)
                              toBuffer:db
                     destinationOffset:dO + l * image * s.depth
                destinationBytesPerRow:row
              destinationBytesPerImage:image];
          continue;
        }
      }
      ++missingResources;
    }
  }
  close();

  if (missingPipelines || missingResources || argOverflow)
    err("what=submit f=%llu missing_pipelines=%u missing_resources=%u arg_overflow=%u", (unsigned long long)frameIndex,
        missingPipelines, missingResources, argOverflow);

  id<MTLCounterSampleBuffer> samples = slot.samples;
  const NSUInteger sampleCount = sampleUsed;
  dispatch_semaphore_t sem = inFlight;
  Impl* impl = this;  // the destructor waits for every frame, so the handler never outlives *this
  [cb addCompletedHandler:^(id<MTLCommandBuffer> finished) {
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
    dispatch_semaphore_signal(sem);
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
  if (ok()) waitIdle();
}

bool MetalBackend::ok() const { return impl_->device != nil && impl_->queue != nil; }

void MetalBackend::createBuffer(BufferId id, const BufferDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  Impl::Buffer b{id.generation, nil, 0};
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
    m.err("what=buffer index=%u size=%llu guest=%d label=%s", id.index, (unsigned long long)d.size,
          d.guestMemory != nullptr, token(d.label).c_str());
    return;
  }
  if (!d.label.empty()) b.buffer.label = @(d.label.c_str());
  if (m.residency) {
    [m.residency addAllocation:b.buffer];
    m.residencyDirty = true;
  }
  m.buffers[id.index] = b;
}

void MetalBackend::createTexture(TextureId id, const TextureDesc& d) {
  Impl& m = *impl_;
  if (!ok()) return;
  MTLTextureDescriptor* t = [MTLTextureDescriptor new];
  const bool array = d.layers > 1;
  switch (d.type) {
    case TextureType::Tex1D: t.textureType = array ? MTLTextureType1DArray : MTLTextureType1D; break;
    case TextureType::Tex2D:
      t.textureType = d.samples > 1 ? (array ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample)
                                    : (array ? MTLTextureType2DArray : MTLTextureType2D);
      break;
    case TextureType::Tex3D: t.textureType = MTLTextureType3D; break;
    case TextureType::Cube: t.textureType = d.layers > 6 ? MTLTextureTypeCubeArray : MTLTextureTypeCube; break;
  }
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
  if (d.usage & usage::storage) u |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  if (d.usage & (usage::colorTarget | usage::depthTarget)) u |= MTLTextureUsageRenderTarget;
  t.usage = u;
  id<MTLTexture> tex = [m.device newTextureWithDescriptor:t];
  if (!tex) {
    m.err("what=texture index=%u format=%s size=%ux%u label=%s", id.index, formatInfo(d.format).name, d.width,
          d.height, token(d.label).c_str());
    return;
  }
  if (!d.label.empty()) tex.label = @(d.label.c_str());
  if (m.residency) {
    [m.residency addAllocation:tex];
    m.residencyDirty = true;
  }
  m.textures[id.index] = {id.generation, tex, d.format};
}

void MetalBackend::createSampler(SamplerId id, const SamplerDesc& d) {
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
  s.supportArgumentBuffers = YES;  // reachable through gpuResourceID in argument tables
  if (!d.label.empty()) s.label = @(d.label.c_str());
  id<MTLSamplerState> st = [m.device newSamplerStateWithDescriptor:s];
  if (!st) {
    m.err("what=sampler index=%u label=%s", id.index, token(d.label).c_str());
    return;
  }
  m.samplers[id.index] = {id.generation, st};
}

void MetalBackend::destroyBuffer(BufferId id) {
  auto it = impl_->buffers.find(id.index);
  if (it == impl_->buffers.end() || it->second.generation != id.generation) return;
  impl_->retire(it->second.buffer);
  impl_->buffers.erase(it);
}

void MetalBackend::destroyTexture(TextureId id) {
  auto it = impl_->textures.find(id.index);
  if (it == impl_->textures.end() || it->second.generation != id.generation) return;
  impl_->retire(it->second.texture);
  impl_->textures.erase(it);
}

void MetalBackend::destroySampler(SamplerId id) {
  auto it = impl_->samplers.find(id.index);
  if (it == impl_->samplers.end() || it->second.generation != id.generation) return;
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
    std::lock_guard lock(impl_->pipelineMutex);
    impl_->pipelines.emplace(s.key().value, std::move(p));  // first one wins: entries never move
    return true;
  }
}

void MetalBackend::submit(uint64_t frameIndex, const Frame& frame, const CompiledFrame& compiled) {
  if (!ok()) return;
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

void MetalBackend::waitIdle() {
  // Holding every slot means every completion handler has run (not only that the GPU finished).
  for (size_t i = 0; i < impl_->slots.size(); ++i) dispatch_semaphore_wait(impl_->inFlight, DISPATCH_TIME_FOREVER);
  for (size_t i = 0; i < impl_->slots.size(); ++i) dispatch_semaphore_signal(impl_->inFlight);
}

void* MetalBackend::contents(BufferId id) const {
  auto it = impl_->buffers.find(id.index);
  if (it == impl_->buffers.end() || it->second.generation != id.generation) return nullptr;
  return static_cast<uint8_t*>(it->second.buffer.contents) + it->second.base;
}

bool MetalBackend::saveArchive() {
  std::lock_guard lock(impl_->pipelineMutex);
  if (!impl_->archive) return false;
  NSError* e = nil;
  const auto path = impl_->options.cacheRoot / "pipelines.metalar";
  if ([impl_->archive serializeToURL:[NSURL fileURLWithPath:@(path.c_str())] error:&e]) return true;
  impl_->err("what=archive_save error=%s", token(str(e)).c_str());
  return false;
}

}  // namespace vpgpu::metal
