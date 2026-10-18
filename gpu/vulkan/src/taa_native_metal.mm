// VPKTAA nativo en Metal: los kernels de gpu/visionos/shaders/vpktaa_native.metal (embebidos en
// vpktaa_native_msl.h por CMake), su estado (historia por ojo, poses del frame anterior) y la
// medida de GPU. Lo llama MetalPlatform dentro de vpvk_native_submit (submitMutex tomado: un hilo).
#import <Metal/Metal.h>

#include "taa_native_metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>

#include "taa_native.h"
#include "vpktaa_native_msl.h"
#include "vpvk.h"
#include "vpvk/config.h"

namespace vpvk {
namespace {

// Mismo layout que VpknEye / VpknParams del MSL (todo en múltiplos de 16 bytes).
struct alignas(16) GpuEye {
  float rot[3][4];
  float trans[4];
  float tanCur[4], tanPrev[4];
  float crop[4];
  float depthAB[4];
  float mvAB[4];
  uint32_t slices[4];
  float maskAB[4];
  uint32_t maskInfo[4];
};
struct alignas(16) GpuParams {
  GpuEye eye[2];
  float w0[4], w1[4], w2[4];
  int32_t base[4];
  float depthP[4];
  float keep[4];
  float mvP[4];
  uint32_t size[4];
};
static_assert(sizeof(GpuEye) == 192 && sizeof(GpuParams) == 512, "layout del MSL");

enum : uint32_t { kHistory = 1, kDepth = 2, kRevZ = 4, kMotion = 8, kOutSrgb = 16, kMask = 32 };

id<MTLTexture> asArray(id<MTLTexture> t) {
  if (!t) return nil;
  if (t.textureType == MTLTextureType2DArray || t.textureType == MTLTextureType2DMultisampleArray) return t;
  if (t.textureType == MTLTextureType2D)
    return [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2DArray levels:NSMakeRange(0, 1)
                                     slices:NSMakeRange(0, 1)];
  if (t.textureType == MTLTextureType2DMultisample)
    return [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2DMultisampleArray
                                     levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
  return nil;
}
bool isDepthFormat(MTLPixelFormat f) {
  return f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float || f == MTLPixelFormatDepth32Float_Stencil8;
}
}  // namespace

struct TaaNativeMetal::Impl {
  id<MTLDevice> device;
  id<MTLLibrary> lib;
  id<MTLComputePipelineState> resolve[2];  // [depth MSAA]
  id<MTLComputePipelineState> rcas;
  id<MTLRenderPipelineState> depthResolve;
  id<MTLDepthStencilState> depthWrite;
  bool failed = false;
  // Historia: rgba16Float 2DArray de 2 capas (una por ojo), ida y vuelta.
  id<MTLTexture> hist[2];
  id<MTLTexture> resolved;  // historia nueva leída por RCAS (= hist[cur])
  uint32_t histW = 0, histH = 0;
  int cur = 0;
  bool histValid[2] = {false, false};
  taa::EyeView prev[2];
  id<MTLTexture> dummyDepth, dummyMotion, dummyMask;
  // Profundidad MSAA resuelta, por clave.
  struct Resolved {
    id<MTLTexture> tex;
  } depthOut[8];
  // Medida.
  std::mutex statMutex;
  double msSum = 0, msMax = 0;
  uint32_t msN = 0, frames = 0, refusals = 0;
  uint64_t lastLogFrame = 0;

  bool init() {
    if (lib || failed) return !failed;
    const auto t0 = std::chrono::steady_clock::now();
    NSError* e = nil;
    // Dos librerías: profundidad de 1 muestra y MSAA (VPKN_DEPTH_MS); la segunda solo da el resolve.
    for (int ms = 0; ms < 2; ++ms) {
      MTLCompileOptions* o = [MTLCompileOptions new];
      o.preprocessorMacros = @{@"VPKN_DEPTH_MS" : @(ms)};
      id<MTLLibrary> l = [device newLibraryWithSource:@(kVpktaaNativeMsl) options:o error:&e];
      if (!l) {
        logf("vk.taa native err what=library ms=%d error=\"%s\"", ms, e ? e.localizedDescription.UTF8String : "?");
        if (ms == 0) return !(failed = true);
        continue;
      }
      if (ms == 0) lib = l;
      id<MTLFunction> f = [l newFunctionWithName:@"vpktaa_native_resolve"];
      resolve[ms] = f ? [device newComputePipelineStateWithFunction:f error:&e] : nil;
      if (!resolve[ms]) logf("vk.taa native err what=resolve ms=%d error=\"%s\"", ms, e ? e.localizedDescription.UTF8String : "?");
    }
    id<MTLFunction> fr = [lib newFunctionWithName:@"vpktaa_native_rcas"];
    rcas = fr ? [device newComputePipelineStateWithFunction:fr error:&e] : nil;
    MTLRenderPipelineDescriptor* rd = [MTLRenderPipelineDescriptor new];
    rd.vertexFunction = [lib newFunctionWithName:@"vpktaa_native_depth_vertex"];
    rd.fragmentFunction = [lib newFunctionWithName:@"vpktaa_native_depth_fragment"];
    rd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    rd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    depthResolve = [device newRenderPipelineStateWithDescriptor:rd error:&e];
    MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionAlways;
    dd.depthWriteEnabled = YES;
    depthWrite = [device newDepthStencilStateWithDescriptor:dd];
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                  width:1 height:1 mipmapped:NO];
    td.textureType = MTLTextureType2DArray;
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    dummyDepth = [device newTextureWithDescriptor:td];
    td.pixelFormat = MTLPixelFormatRG16Float;
    dummyMotion = [device newTextureWithDescriptor:td];
    td.pixelFormat = MTLPixelFormatR8Unorm;
    dummyMask = [device newTextureWithDescriptor:td];
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    failed = !resolve[0] || !rcas || !dummyDepth || !dummyMotion || !dummyMask;
    logf("vk.taa native init ok=%d resolve=%d resolve_ms=%d rcas=%d depth_resolve=%d compile_ms=%.1f", failed ? 0 : 1,
         resolve[0] ? 1 : 0, resolve[1] ? 1 : 0, rcas ? 1 : 0, depthResolve ? 1 : 0, ms);
    return !failed;
  }

  bool ensureHistory(uint32_t w, uint32_t h) {
    if (hist[0] && histW == w && histH == h) return true;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                  width:w height:h mipmapped:NO];
    td.textureType = MTLTextureType2DArray;
    td.arrayLength = 2;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    for (auto& t : hist) t = [device newTextureWithDescriptor:td];
    histW = w, histH = h;
    histValid[0] = histValid[1] = false;
    logf("vk.taa native history %ux%u rgba16f x2 layers x2 (%.1f MiB)", w, h, double(w) * h * 8 * 2 * 2 / 1048576.0);
    return hist[0] && hist[1];
  }

  void* resolveDepth(id<MTLCommandBuffer> cb, id<MTLTexture> src, bool rev, int mode, int key) {
    if (!init() || !depthResolve || !src || src.sampleCount <= 1 || !isDepthFormat(src.pixelFormat)) return nullptr;
    id<MTLTexture> s = asArray(src);
    if (!s) return nullptr;
    Resolved& r = depthOut[std::clamp(key, 0, 7)];
    if (!r.tex || r.tex.width != s.width || r.tex.height != s.height || r.tex.arrayLength != s.arrayLength) {
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                    width:s.width height:s.height
                                                                                mipmapped:NO];
      td.textureType = MTLTextureType2DArray;
      td.arrayLength = s.arrayLength;
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
      td.storageMode = MTLStorageModePrivate;
      r.tex = [device newTextureWithDescriptor:td];
      logf("vk.depth resolve key=%d %lux%lu layers=%lu samples=%lu mode=%s", key, (unsigned long)s.width,
           (unsigned long)s.height, (unsigned long)s.arrayLength, (unsigned long)s.sampleCount, mode ? "nearest" : "sample0");
    }
    if (!r.tex) return nullptr;
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.depthAttachment.texture = r.tex;
    rp.depthAttachment.loadAction = MTLLoadActionDontCare;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    rp.renderTargetArrayLength = s.arrayLength;
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    enc.label = @"vpktaa.depth_resolve";
    [enc setRenderPipelineState:depthResolve];
    [enc setDepthStencilState:depthWrite];
    [enc setFragmentTexture:s atIndex:0];
    const uint32_t q[4] = {uint32_t(s.sampleCount), uint32_t(mode ? 1 : 0), rev ? 1u : 0u, 0};
    [enc setFragmentBytes:q length:sizeof q atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:s.arrayLength];
    [enc endEncoding];
    return (__bridge void*)r.tex;
  }

  bool encode(const Platform::TaaNativeArgs& a, id<MTLCommandBuffer> cb) {
    const auto* f = static_cast<const vpvk_taa_native_frame*>(a.frame);
    if (!init()) return false;
    id<MTLTexture> out = (__bridge id<MTLTexture>)f->output;
    id<MTLTexture> color[2], depth[2], motion[2], mask[2];
    for (int e = 0; e < 2; ++e) {
      // Máscara de UI del motor (taa_native.h): R8 del tamaño del pase de ojo; null = sin UI este frame.
      mask[e] = asArray((__bridge id<MTLTexture>)a.mask[e].texture);
      if (mask[e] && (mask[e].pixelFormat != MTLPixelFormatR8Unorm || mask[e].sampleCount > 1)) mask[e] = nil;
      color[e] = asArray((__bridge id<MTLTexture>)f->eye[e].color);
      depth[e] = asArray((__bridge id<MTLTexture>)f->eye[e].depth);
      motion[e] = asArray((__bridge id<MTLTexture>)f->eye[e].motion);
      if (depth[e] && !isDepthFormat(depth[e].pixelFormat)) depth[e] = nil;
    }
    auto refuse = [&](const char* why) {
      if (refusals++ < 8) logf("vk.taa native refused why=%s", why);
      return false;
    };
    if (!color[0] || !color[1] || !out) return refuse("missing_texture");
    const uint32_t W = uint32_t(color[0].width), H = uint32_t(color[0].height);
    if (color[1].width != W || color[1].height != H) return refuse("eyes_differ");
    if (out.textureType != MTLTextureType2DArray || out.arrayLength < 2 || out.width != W || out.height != H)
      return refuse("output_shape");
    if (color[0].sampleCount > 1 || color[1].sampleCount > 1) return refuse("color_msaa(resolve first)");
    if (!ensureHistory(W, H)) return refuse("history_alloc");
    if (f->reset) histValid[0] = histValid[1] = false;

    GpuParams p{};
    // Desplazamiento del contenido en texels del color (KleptonTAA.makeInput: target de otro tamaño
    // = búfer intermedio que acaba en el recorte del ojo).
    float sx = 0, sy = 0;
    if (a.jittered) {
      float kx = 1, ky = 1;
      if (a.targetW && a.targetH && (a.targetW != W || a.targetH != H)) {
        kx = f->eye[0].crop[0] * W / float(a.targetW);
        ky = f->eye[0].crop[1] * H / float(a.targetH);
      }
      sx = a.jx * kx, sy = a.jy * ky;
    }
    const taa::Reconstruction rc = taa::reconstruction(sx, sy);
    std::memcpy(p.w0, &rc.weights[0], 16);
    std::memcpy(p.w1, &rc.weights[4], 16);
    p.w2[0] = rc.weights[8];
    p.base[0] = rc.base[0], p.base[1] = rc.base[1], p.base[2] = rc.nearest[0], p.base[3] = rc.nearest[1];
    p.depthP[0] = f->near_z > 0 ? f->near_z : 0.1f;
    p.depthP[1] = std::max(f->far_z, 0.0f);
    p.depthP[2] = 60000.0f;
    p.depthP[3] = 0.05f;
    const float pct = std::clamp(a.percent, 1.0f, 100.0f);
    auto keepAt = [&](float k) { return std::clamp(1 - (1 - k) * 80 / pct, 0.2f, 0.97f); };
    p.keep[0] = keepAt(0.93f), p.keep[1] = keepAt(0.80f);
    p.keep[2] = a.jittered ? 0.97f : 0.5f;  // sin jitter solo suaviza
    p.keep[3] = std::clamp(a.sharpness, 0.0f, 1.0f);
    p.size[0] = W, p.size[1] = H;

    taa::EyeView now[2];
    for (int e = 0; e < 2; ++e) {
      const vpvk_taa_native_eye& in = f->eye[e];
      GpuEye& g = p.eye[e];
      now[e].orientation = {in.orientation[0], in.orientation[1], in.orientation[2], in.orientation[3]};
      now[e].position = {in.position[0], in.position[1], in.position[2]};
      for (int i = 0; i < 4; ++i) now[e].tangents[i] = in.tangents[i];
      const taa::EyeView& pv = histValid[e] ? prev[e] : now[e];
      const taa::Reprojection m = taa::reprojection(now[e], pv);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) g.rot[r][c] = m.r[r * 3 + c];
      g.trans[0] = m.t.x, g.trans[1] = m.t.y, g.trans[2] = m.t.z;
      for (int i = 0; i < 4; ++i) g.tanCur[i] = in.tangents[i], g.tanPrev[i] = pv.tangents[i];
      const bool fullCrop = !(in.crop[0] > 0 && in.crop[1] > 0);
      g.crop[0] = fullCrop ? 1 : std::min(in.crop[0], 1.0f), g.crop[1] = fullCrop ? 1 : std::min(in.crop[1], 1.0f);
      g.crop[2] = fullCrop ? 0 : in.crop[2], g.crop[3] = fullCrop ? 0 : in.crop[3];
      uint32_t fl = histValid[e] ? kHistory : 0;
      if (depth[e]) {
        fl |= kDepth;
        // Profundidad del tamaño del color: su layout; otro tamaño: cubre el recorte del ojo.
        const bool same = depth[e].width == W && depth[e].height == H;
        const float dw = float(depth[e].width), dh = float(depth[e].height);
        g.depthAB[0] = same ? 1 : dw / (g.crop[0] * W), g.depthAB[1] = same ? 1 : dh / (g.crop[1] * H);
        g.depthAB[2] = same ? 0 : -g.crop[2] * W * g.depthAB[0], g.depthAB[3] = same ? 0 : -g.crop[3] * H * g.depthAB[1];
      }
      if (f->reversed_z) fl |= kRevZ;
      if (motion[e]) {
        fl |= kMotion;
        const bool same = motion[e].width == W && motion[e].height == H;
        const float mw = float(motion[e].width), mh = float(motion[e].height);
        g.mvAB[0] = same ? 1 : mw / (g.crop[0] * W), g.mvAB[1] = same ? 1 : mh / (g.crop[1] * H);
        g.mvAB[2] = same ? 0 : -g.crop[2] * W * g.mvAB[0], g.mvAB[3] = same ? 0 : -g.crop[3] * H * g.mvAB[1];
      }
      if (mask[e] && a.mask[e].slice < mask[e].arrayLength) {
        fl |= kMask;
        // Mismo mapeo que la profundidad: del tamaño del color, o cubriendo el recorte del ojo.
        const bool same = mask[e].width == W && mask[e].height == H;
        const float mw = float(mask[e].width), mh = float(mask[e].height);
        g.maskAB[0] = same ? 1 : mw / (g.crop[0] * W), g.maskAB[1] = same ? 1 : mh / (g.crop[1] * H);
        g.maskAB[2] = same ? 0 : -g.crop[2] * W * g.maskAB[0], g.maskAB[3] = same ? 0 : -g.crop[3] * H * g.maskAB[1];
        g.maskInfo[0] = a.mask[e].slice;
      } else {
        mask[e] = nil;
      }
      if (f->output_srgb) fl |= kOutSrgb;
      g.slices[0] = in.color_slice, g.slices[1] = in.depth_slice, g.slices[2] = in.motion_slice, g.slices[3] = fl;
    }
    p.mvP[0] = f->eye[0].motion_scale[0], p.mvP[1] = f->eye[0].motion_scale[1];

    const int src = cur, dst = cur ^ 1;
    // Un dispatch con z = 2 cuando ambos ojos son capas de las mismas texturas (multiview).
    const bool shared = color[0] == color[1] && depth[0] == depth[1] && motion[0] == motion[1] &&
                        f->eye[0].color_slice == 0 && f->eye[1].color_slice == 1 &&
                        (!depth[0] || (f->eye[0].depth_slice == 0 && f->eye[1].depth_slice == 1)) &&
                        (!motion[0] || (f->eye[0].motion_slice == 0 && f->eye[1].motion_slice == 1)) &&
                        mask[0] == mask[1] && (!mask[0] || (a.mask[0].slice == 0 && a.mask[1].slice == 1));
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    enc.label = @"vpktaa.native";
    const MTLSize tg = MTLSizeMake(16, 16, 1);
    for (int e = 0; e < (shared ? 1 : 2); ++e) {
      const bool ms = depth[e] && depth[e].sampleCount > 1;
      if (ms && !resolve[1]) depth[e] = nil, p.eye[e].slices[3] &= ~kDepth;
      p.size[2] = depth[e] ? uint32_t(depth[e].sampleCount) : 1;
      p.size[3] = uint32_t(e);
      [enc setComputePipelineState:resolve[ms && depth[e] ? 1 : 0]];
      [enc setTexture:color[e] atIndex:0];
      [enc setTexture:(depth[e] ? depth[e] : dummyDepth) atIndex:1];
      [enc setTexture:(mask[e] ? mask[e] : dummyMask) atIndex:2];
      [enc setTexture:(motion[e] ? motion[e] : dummyMotion) atIndex:3];
      [enc setTexture:hist[src] atIndex:4];
      [enc setTexture:hist[dst] atIndex:5];
      [enc setBytes:&p length:sizeof p atIndex:0];
      [enc dispatchThreads:MTLSizeMake(W, H, shared ? 2 : 1) threadsPerThreadgroup:tg];
    }
    // RCAS: ambos ojos en un dispatch (z = 2) salvo que cada ojo tenga su máscara.
    [enc memoryBarrierWithScope:MTLBarrierScopeTextures];
    [enc setComputePipelineState:rcas];
    [enc setTexture:hist[dst] atIndex:0];
    [enc setTexture:out atIndex:1];
    const bool sharedRcas = mask[0] == mask[1] && (!mask[0] || (a.mask[0].slice == 0 && a.mask[1].slice == 1));
    for (int e = 0; e < (sharedRcas ? 1 : 2); ++e) {
      p.size[3] = sharedRcas ? 0u : uint32_t(e);
      [enc setTexture:(mask[sharedRcas ? 0 : e] ? mask[sharedRcas ? 0 : e] : dummyMask) atIndex:2];
      [enc setBytes:&p length:sizeof p atIndex:0];
      [enc dispatchThreads:MTLSizeMake(W, H, sharedRcas ? 2 : 1) threadsPerThreadgroup:tg];
    }
    [enc endEncoding];
    cur = dst;
    for (int e = 0; e < 2; ++e) prev[e] = now[e], histValid[e] = true;

    // Medida: el command buffer nativo solo lleva este pase cuando el host no codifica nada más.
    const uint64_t frame = ++frames;
    const uint32_t w = W, h = H;
    const bool jit = a.jittered;
    const uint64_t phase = a.phase;
    const int maskEyes = (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0);
    Impl* impl = this;  // vive tanto como el MetalPlatform (más que sus command buffers: waitIdle)
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
      if (c.status != MTLCommandBufferStatusCompleted) return;
      const double ms = (c.GPUEndTime - c.GPUStartTime) * 1000.0;
      std::lock_guard<std::mutex> l(impl->statMutex);
      impl->msSum += ms, impl->msMax = std::max(impl->msMax, ms), ++impl->msN;
      if (frame <= 3 || frame - impl->lastLogFrame >= 600) {
        logf("vk.taa native gpu_ms mean=%.3f max=%.3f per_eye=%.3f frames=%u size=%ux%u jitter=%d phase=%llu "
             "uimask_eyes=%d (command buffer completo: incluye lo que el host codifique en él)",
             impl->msSum / impl->msN, impl->msMax, impl->msSum / impl->msN / 2, impl->msN, w, h, jit ? 1 : 0,
             static_cast<unsigned long long>(phase % taa::kPhases), maskEyes);
        impl->msSum = impl->msMax = 0, impl->msN = 0;
        impl->lastLogFrame = frame;
      }
    }];
    return true;
  }
};

TaaNativeMetal::TaaNativeMetal(void* device) : impl_(new Impl) { impl_->device = (__bridge id<MTLDevice>)device; }
TaaNativeMetal::~TaaNativeMetal() { delete impl_; }
bool TaaNativeMetal::encode(const Platform::TaaNativeArgs& a, void* cb) {
  return impl_->device && cb && a.frame && impl_->encode(a, (__bridge id<MTLCommandBuffer>)cb);
}
void* TaaNativeMetal::resolveDepth(void* cb, void* depthMS, bool reversedZ, int mode, int key) {
  return impl_->device && cb ? impl_->resolveDepth((__bridge id<MTLCommandBuffer>)cb, (__bridge id<MTLTexture>)depthMS,
                                                   reversedZ, mode, key)
                             : nullptr;
}

}  // namespace vpvk
