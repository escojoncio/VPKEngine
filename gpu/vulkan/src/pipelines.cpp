// VPVulkan phase 3: shader modules, descriptor set layouts / pools / sets / update templates,
// pipeline layouts, graphics and compute pipelines (SPIR-V -> VPIR -> MSL through gpu/shader/spirv
// and gpu/shader/msl), the bind / draw / dispatch commands, the remaining copies and clears,
// secondary command buffers, dynamic rendering and query pools (answered without GPU queries).
//
// Slots: a pipeline layout numbers its descriptors once, in (set, binding) order, per argument-table
// kind (Buffer, ConstantBuffer, Texture, Sampler, StorageTexture). Push constants take the next
// ConstantBuffer slot and vertex buffers the Buffer slots after the descriptors. Every stage of
// every pipeline made from the layout agrees on them.
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "commands.h"
#include "taa_native.h"
#include "vpir/passes.h"
#include "vpgpu/core/archive_harvest.h"
#include "vpgpu/core/background.h"
#include "vpgpu/core/perf.h"
#include "vpmsl/msl.h"
#include "vpvk.h"

#define VPVK_EXPORT extern "C" __attribute__((visibility("default")))

namespace vpvk {

// ---- MSAA of a pass -----------------------------------------------------------------------------
void PassSamples::add(const ImageView* v, bool depth) {
  if (!v || !v->image) return;
  const Image* im = v->image;
  const uint8_t eff = std::max<uint8_t>(1, im->desc.samples);
  if (!effective) effective = eff;
  else if (eff != effective) mismatch = true;
  game = std::max(game, std::max<uint8_t>(1, im->gameSamples));
  if (depth) {
    hasDepth = hasDepth || vpgpu::formatInfo(im->desc.format).depth();
  } else if (!hasColor) {
    hasColor = true;
    w = std::max(1u, im->desc.width >> v->range.baseMipLevel);
    h = std::max(1u, im->desc.height >> v->range.baseMipLevel);
  }
}

void PassSamples::finish(uint32_t viewMask, const char* what) {
  if (mismatch) {
    // Images created with different counts (the override changed between them): Metal refuses
    // the pass. Logged; the frame compiler / backend report the failure itself.
    static std::atomic<unsigned> n{0};
    if (n.fetch_add(1) < 8) logf("vk.err what=msaa_mismatch pass=%s size=%ux%u", what, w, h);
  }
  msaaObservePass(game, w, h, viewMask, hasColor, hasDepth);
}

uint8_t subpassSamples(const RenderPassObj& rp, const Framebuffer& fb, uint32_t sub, bool observe) {
  if (sub >= rp.subpasses.size()) return 0;
  const RenderPassObj::Subpass& sp = rp.subpasses[sub];
  PassSamples ps;
  auto view = [&](uint32_t a) -> const ImageView* {
    return a != VK_ATTACHMENT_UNUSED && a < fb.views.size() ? fb.views[a] : nullptr;
  };
  for (const auto& c : sp.colors) ps.add(view(c.attachment), false);
  if (sp.hasDepth) ps.add(view(sp.depth.attachment), true);
  // The game's own count is in the render pass (the framebuffer may be imageless-like wild handles).
  for (const auto& c : sp.colors)
    if (c.attachment != VK_ATTACHMENT_UNUSED && c.attachment < rp.attachments.size())
      ps.game = std::max<uint8_t>(ps.game, uint8_t(rp.attachments[c.attachment].samples));
  if (observe) ps.finish(sp.viewMask, "render_pass");
  return ps.effective;
}

namespace {
vpgpu::Primitive primitive(VkPrimitiveTopology t, bool& ok);  // defined below
}
// Rasterizer discard: the vertex stage runs (its stores count), nothing reaches the attachments.
// Metal has the switch only in the render pipeline; without a fragment function and with every
// write masked off the draw leaves colour, depth and stencil alone.
static void discardRaster(vpgpu::PipelineState& st) {
  st.fragmentShader = 0;
  for (vpgpu::BlendState& b : st.blend) b.writeMask = 0;
  st.depthTest = st.depthWrite = st.stencilTest = false;
  st.alphaToCoverage = false;
}
// Core cull mode of the Vulkan value.
static vpgpu::CullMode coreCull(VkCullModeFlags m) {
  return m == VK_CULL_MODE_FRONT_BIT ? vpgpu::CullMode::Front : m == VK_CULL_MODE_BACK_BIT ? vpgpu::CullMode::Back
                                                                                          : vpgpu::CullMode::None;
}
static bool isTriangleTopology(VkPrimitiveTopology t) {
  return t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
         t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
}

// Metal has no polygon fill modes beyond lines: VK_POLYGON_MODE_POINT draws the vertices of the
// triangles (one point per vertex, as the spec's "vertices of the polygon") with the point primitive
// class, which also makes the backend keep the vertex shader's PointSize (VP_PSZ). Differences from the
// spec that remain: cull mode does not apply (it applies to polygons, not to the points they become),
// and a strip's restart index in an indexed draw is read as a vertex.
static vpgpu::Primitive primitiveOf(const PipelineObj& p, VkPrimitiveTopology t, vpgpu::Primitive prim) {
  return p.pointMode && isTriangleTopology(t) ? vpgpu::Primitive::Points : prim;
}

// The topology a draw uses: the pipeline's, or the one vkCmdSetPrimitiveTopology gave.
static VkPrimitiveTopology effectiveTopology(const PipelineObj& p, const BoundState::Eds* e) {
  return e && e->topology >= 0 && p.isDynamic(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY) ? VkPrimitiveTopology(e->topology)
                                                                                   : p.topology;
}

// Applies the extended dynamic state of a draw (vkCmdSetCullMode & co.) to `st`, a copy of the
// pipeline's state; the result is a hash of the values that were replaced (0 = nothing, `st` is
// untouched). Metal keeps cull mode, depth-stencil state and the primitive class outside the
// shaders, so the variant costs a pipeline entry, not a shader translation.
static uint64_t applyEds(const PipelineObj& p, const BoundState::Eds& e, vpgpu::PipelineState& st) {
  uint64_t h = 0;
  auto note = [&](uint64_t tag, uint64_t v) {
    h = ((h ? h : 0xcbf29ce484222325ull) ^ (tag << 32 | (v & 0xFFFFFFFFull))) * 0x100000001b3ull;
  };

  if (p.isDynamic(VK_DYNAMIC_STATE_CULL_MODE) && e.cullMode >= 0) {
    st.cull = coreCull(VkCullModeFlags(e.cullMode));
    note(1, uint64_t(e.cullMode));
  }
  if (p.isDynamic(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY) && e.topology >= 0) {
    bool ok;
    const vpgpu::Primitive prim = primitive(VkPrimitiveTopology(e.topology), ok);
    st.primitive = primitiveOf(p, VkPrimitiveTopology(e.topology), prim);  // fans stay triangles (converted per draw)
    note(2, uint64_t(e.topology));
  }
  if (st.depthFormat != vpgpu::Format::Undefined) {
    const bool dt = p.isDynamic(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE) && e.depthTest >= 0;
    const bool dw = p.isDynamic(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE) && e.depthWrite >= 0;
    const bool dc = p.isDynamic(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP) && e.depthCompare >= 0;
    if (dt || dw || dc) {
      const bool test = dt ? e.depthTest != 0 : p.rawDepth.test;
      const bool write = dw ? e.depthWrite != 0 : p.rawDepth.write;
      const VkCompareOp op = dc ? VkCompareOp(e.depthCompare) : p.rawDepth.compare;
      st.depthTest = test;
      st.depthWrite = test && write;  // depth writes are off whenever the depth test is
      st.depthCompare = test ? static_cast<vpgpu::CompareOp>(op) : vpgpu::CompareOp::Always;
      note(3, uint64_t(test) | uint64_t(write) << 1 | uint64_t(op) << 2);
    }
    const bool dst = p.isDynamic(VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE) && e.stencilTest >= 0;
    const bool dso = p.isDynamic(VK_DYNAMIC_STATE_STENCIL_OP);
    if (dst) {
      st.stencilTest = e.stencilTest != 0;
      note(4, uint64_t(st.stencilTest));
    }
    if (dso) {
      vpgpu::StencilFace* faces[2] = {&st.stencilFront, &st.stencilBack};
      for (int f = 0; f < 2; ++f) {
        if (!e.stencilOp[f]) continue;
        faces[f]->fail = static_cast<vpgpu::StencilOp>(e.fail[f]);
        faces[f]->pass = static_cast<vpgpu::StencilOp>(e.pass[f]);
        faces[f]->depthFail = static_cast<vpgpu::StencilOp>(e.depthFail[f]);
        faces[f]->compare = static_cast<vpgpu::CompareOp>(e.compare[f]);
        note(5 + uint64_t(f), uint64_t(e.fail[f]) | uint64_t(e.pass[f]) << 4 | uint64_t(e.depthFail[f]) << 8 |
                                  uint64_t(e.compare[f]) << 12);
      }
    }
  }
  // VK_EXT_extended_dynamic_state3 colour write mask. An output the shader does not write stays masked off.
  if (p.isDynamic(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT))
    for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c) {
      if (e.colorWrite[c] < 0) continue;
      const uint8_t m = (p.fragmentOutputs >> c & 1) ? uint8_t(e.colorWrite[c] & 0xF) : 0;
      if (m == st.blend[c].writeMask) continue;
      st.blend[c].writeMask = m;
      note(16 + uint64_t(c), m);
    }
  // Last: it switches off what the states above may have switched on.
  if (p.isDynamic(VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE) && (e.rasterDiscard >= 0 ? e.rasterDiscard != 0 : p.rasterDiscard)) {
    discardRaster(st);  // the pipeline itself is built un-discarded when the state is dynamic
    note(9, 1);
  }
  return h;
}

// Takes a share of the core pipeline `k` for `p` (once per key), before the request() that builds it:
// vkDestroyPipeline gives the shares back, and the last one frees the native pipeline.
void retainKey(Device* dev, const PipelineObj& p, vpgpu::PipelineKey k) {
  if (std::find(p.ownedKeys.begin(), p.ownedKeys.end(), k.value) != p.ownedKeys.end()) return;
  p.ownedKeys.push_back(k.value);
  dev->gpu->pipelines().retain(k);
}

// ---- compile policy -------------------------------------------------------------------------------
// Two modes, chosen per device (Device::translator, asyncPipelinesOn()):
//
// * Asynchronous creation (default; VPVK_ASYNC_PIPELINES=0 turns it off). Games create thousands of pipelines in a burst (AC Nexus ShaderWarmup: 3313
//   stages / 4101 pipelines) and draw with them in the same breath; translating or compiling on the calling thread
//   froze it for seconds. So:
//   - vkCreate{Graphics,Compute}Pipelines gathers the state the create info decides, queues the rest (SPIR-V -> MSL
//     translation, shader registration, the Metal compile request) on the translate pool and RETURNS. Nothing waits.
//   - A pipeline is resolved when its job finished. A draw / dispatch / variant / destroy that needs the pipeline's
//     translation (its key, the bindings it uses) boosts the job to the front and waits for it (resolvePipeline):
//     that is a few ms of one translation, bounded by VPVK_TRANSLATE_WAIT_MS (default 2000, a safety valve), and
//     never a skipped draw: a command buffer may be replayed, a draw dropped at record time would stay missing.
//   - The Metal compile (or binary-archive lookup) runs on the core's workers; the game's own pipelines are requested at
//     Visible priority, so a variant made at a draw (Blocking) and a pipeline a submitted draw waits for (boost: Urgent)
//     overtake the creation backlog. At vkQueueSubmit, before the frame goes
//     to the backend, gateFramePipelines() decides what to do with draws whose pipeline is still compiling:
//       soft passes (the colour destination is an image of the eye layer, marked by the host with vpvk_mark_eye_image:
//         the game redraws it every frame):
//         wait at most VPVK_PIPELINE_FRAME_WAIT_MS (default 3) for all of them together, then drop the draws still
//         pending from this frame (counted, `vk.pipeline_skip`). Never drawn with another shader.
//         With VPVK_PIPELINE_EYE_HARD_WAIT_MS > 0 (default 0 = the above; 1500 for vpgles, eyeHardWaitMs) the eye is treated
//         as hard: the frame waits up to that budget for its pipelines (already boosted), `vk.pipeline_wait what=eye_target`,
//         and only what is still pending after it is dropped. Each pipeline gets that wait once (later frames drop its
//         draws as before until it is ready) and all eye waits together take at most twice the budget per second.
//       hard passes (they store a texture that outlives the frame and is not the eye: a render-to-texture, a shadow
//         map, a probe; and every dispatch): a draw dropped there can stay missing for good, so the frame waits for
//         them, at most VPVK_PIPELINE_PERSIST_WAIT_MS (default 1500; 10000 for vpgles, persistWaitMs), `vk.pipeline_wait what=persistent_target`.
//     A draw dropped from one submission is drawn again when the command buffer is submitted again.
//   - There is no variant that may stand in for a pipeline that is not ready: MSAA, extended-dynamic-state, texel
//     buffer and UI-mask variants differ from the base pipeline in attachment layout, depth state or shader, so the
//     base would draw wrongly (or be rejected by Metal). The only "compatible" pipeline is one with the same key,
//     which the core cache already shares.
//   - The core's frame hold (VP_HOLD_MS) is off in this mode: the gate replaces it.
//   - vpgles (the internal GLES client) uses the same mode: a GL thread that creates a pipeline at its first draw no longer
//     sits through that pipeline's Metal compile (100-900 ms in a Demeo log, one after the other); the pipelines of a
//     frame compile together on the workers. Its hard passes wait longer (persistWaitMs): GL gives no redraw guarantee.
//     It stays in this mode even with VPVK_ASYNC_PIPELINES=0 (it has no PSO precache and would serialize every compile
//     under the GL device lock) unless VPVK_GLES_SYNC_PIPELINES=1 (asyncPipelinesOn).
//
// * Synchronous creation (opt-in: VPVK_ASYNC_PIPELINES=0, VQP's games and the CTS run): the convention of every desktop / mobile driver (Quest's compiles
//   inside the call): vkCreate*Pipelines returns with the pipelines translated and compiled. An engine that counts a
//   created pipeline as warm (UE4's PSO precache, >8000 of them) is told so only when it is. It waits for the whole
//   call (every pipeline of the batch, boosted to Urgent while it waits), at most VPVK_PIPELINE_CREATE_WAIT_MS
//   (default 60000, a safety valve; the pipelines still pending then count in vk.pso.census timed_out), and variants
//   made at the first draw wait for their compile once (VPVK_PIPELINE_DRAW_WAIT_MS, default 2000; the VPKTAA UI-mask
//   variants, built while a submission is translated under submitMutex, are collected and awaited once before the frame
//   is sent, awaitFreshVariants). No wait while the device is lost or the app is in the background, and after a wait
//   timed out the next ones are capped at 500 ms for 5 s (awaitPipelines). A pipeline the
//   recipe store / disk cache already holds costs no wait. Game threads translate on their own and the Metal compiles run
//   in parallel on the core's workers (PipelineCache::workerLoop compiles outside its mutex; the waiters sleep on a
//   condition variable): no lock is held across the wait. With VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT a
//   pipeline the cache / recipe store does not hold is not requested: VK_PIPELINE_COMPILE_REQUIRED, handle VK_NULL_HANDLE
//   (not honoured in the asynchronous mode, where the key is not known when the call returns).
uint32_t waitBudgetMs(const char* name, uint32_t fallback) {
  const char* e = std::getenv(name);
  return e && *e ? uint32_t(std::strtoul(e, nullptr, 10)) : fallback;
}

// Per-boot counters (struct in commands.h), all threads: the "vk.boot_summary" line, pipelineSummary().
PipelineCounters& pipelineCounters() {
  static PipelineCounters c;
  return c;
}
static uint64_t usSince(std::chrono::steady_clock::time_point t0) {
  return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

// A pipeline whose SPIR-V did not translate (it never reaches the Metal compile): counted in vk.pso.census failed=, and
// the reason logged as "vk.pso.fail reason=translate:<why> key=" for the first 64 (the backend logs its own compile failures).
void noteTranslateFail(const std::string& why, uint64_t key, const std::string& label) {
  ++pipelineCounters().translateFailed;
  static std::atomic<unsigned> n{0};
  if (n.fetch_add(1) >= 64) return;
  std::string r = why.empty() ? "unknown" : why;
  for (char& c : r)
    if (c == ' ' || c == '"' || c == '=' || c == '\n' || c == '\r' || c == '\t') c = '_';
  if (r.size() > 200) r.resize(200);
  logf("vk.pso.fail reason=translate:%s key=%016llx label=%s", r.c_str(), (unsigned long long)key, label.c_str());
}

// Synchronous mode: blocks until every key that is still queued or compiling is Ready / Failed, or the budget passes.
// Returns how many of them are still queued or compiling when it gives up (0 when all came back). Does not wait at all
// while the device is lost or the app is in the background (nothing compiles that anyone could use, the thread would only
// sit out the whole budget); after a wait timed out the next ones are capped at kBackoffWaitMs for kBackoffMs.
unsigned awaitPipelines(Device* dev, const std::vector<vpgpu::PipelineKey>& keys, uint32_t budgetMs, const char* what) {
  if (!dev || !dev->gpu || !budgetMs || dev->translator) return 0;
  if ((dev->platform && dev->platform->lost()) || vpgpu::backgroundGate().active()) return 0;
  constexpr uint32_t kBackoffWaitMs = 500;
  constexpr int64_t kBackoffMs = 5000;
  auto nowNs = [] {
    return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
  };
  if (nowNs() < dev->pipelineBackoffUntilNs.load(std::memory_order_relaxed)) budgetMs = std::min(budgetMs, kBackoffWaitMs);
  std::vector<vpgpu::PipelineKey> pending;
  for (vpgpu::PipelineKey k : keys) {
    if (!k.value) continue;
    const auto st = dev->gpu->pipelines().status(k);
    if (st == vpgpu::PipelineCache::Status::Queued || st == vpgpu::PipelineCache::Status::Compiling) pending.push_back(k);
  }
  if (pending.empty()) return 0;
  for (vpgpu::PipelineKey k : pending) dev->gpu->pipelines().boost(k);  // a synchronous wait: ahead of every queued job
  const auto t0 = std::chrono::steady_clock::now();
  bool complete;
  {
    const vpgpu::DemandWait demand;  // the binary-archive harvest steps aside while a thread waits here
    complete = dev->gpu->pipelines().waitReady(pending, std::chrono::milliseconds(budgetMs));
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  pipelineCounters().blockedCreateUs += usSince(t0);
  static std::atomic<unsigned> n{0};
  const bool overshoot = ms > double(budgetMs) + 250.0;  // waitReady uses one absolute deadline: only a blocked thread gets here
  if (!complete || overshoot || n.fetch_add(1) < 24)
    logf("vk.pipeline_wait what=%s pipelines=%zu ms=%.0f complete=%d budget_ms=%u%s", what, pending.size(), ms,
         int(complete), budgetMs, overshoot ? " overshoot=1" : "");
  // Every wait over 100 ms, aggregated per second (sum and max), whatever the 24-line cap above says.
  if (ms > 100.0) {
    static std::mutex aggMutex;
    static auto windowStart = std::chrono::steady_clock::now();
    static double sum = 0, mx = 0;
    static unsigned count = 0;
    std::lock_guard<std::mutex> lock(aggMutex);
    sum += ms, mx = std::max(mx, ms), ++count;
    const auto now = std::chrono::steady_clock::now();
    if (now - windowStart >= std::chrono::seconds(1)) {
      logf("vk.pipeline_wait_agg window_ms=%.0f waits=%u sum_ms=%.0f max_ms=%.0f last=%s",
           std::chrono::duration<double, std::milli>(now - windowStart).count(), count, sum, mx, what);
      windowStart = now, sum = mx = 0, count = 0;
    }
  }
  unsigned left = 0;
  if (!complete) {
    dev->pipelineBackoffUntilNs.store(nowNs() + kBackoffMs * 1'000'000, std::memory_order_relaxed);
    for (vpgpu::PipelineKey k : pending) {
      const auto st = dev->gpu->pipelines().status(k);
      if (st == vpgpu::PipelineCache::Status::Queued || st == vpgpu::PipelineCache::Status::Compiling) ++left;
    }
  }
  return left;
}

void awaitPipeline(Device* dev, vpgpu::PipelineKey k, const char* what) {
  static const uint32_t budget = waitBudgetMs("VPVK_PIPELINE_DRAW_WAIT_MS", 2000);
  awaitPipelines(dev, {k}, budget, what);
}

// Asynchronous creation: waits (boosted to the front of the pool) for the translation of `p`. True when it is
// resolved (ok / key / bindings valid); false only past VPVK_TRANSLATE_WAIT_MS.
bool resolvePipeline(Device* dev, const PipelineObj& p) {
  if (p.resolved.load(std::memory_order_acquire)) return true;
  if (!dev || !dev->translator || !p.job) return true;
  static const uint32_t budget = waitBudgetMs("VPVK_TRANSLATE_WAIT_MS", 2000);
  PipelineCounters& c = pipelineCounters();
  const auto t0 = std::chrono::steady_clock::now();
  dev->translator->boost(p.job);
  const bool done = dev->translator->wait(p.job, budget) && p.resolved.load(std::memory_order_acquire);
  c.blockedRecordUs += usSince(t0);
  ++c.recordWaits;
  if (!done) {
    ++c.recordTimeouts;
    static std::atomic<unsigned> n{0};
    if (n.fetch_add(1) < 8) logf("vk.pipeline_wait what=translate pipeline=%s timeout_ms=%u", p.state.label.c_str(), budget);
  }
  return done;
}

// Where a render pass draws. Only DestClass::Eye is soft (a draw dropped there is drawn again next frame, because the
// game redraws the eye every frame); everything else is hard: the frame waits for its pipelines. The eye is identified
// without guessing: the host marks the images of the eye layer's swapchain (vpvk_mark_eye_image, Klepton's
// kl_vulkan_layer_image) and the pass is soft when one of its colour attachments, or the resolve target of one, is
// such an image. Size, layer count and the persistent flag are NOT used to call something the eye: a panel's render
// target (UI text drawn once), a persistent RT of the eye's size and a shadow map all looked like it and lost their
// content for good when a draw was dropped.
enum class DestClass : uint8_t { Eye, NoColor, TransientRT, PersistentRT, EyeSizedNotEye };

static DestClass classifyPass(Device* dev, const vpgpu::RenderPassDesc& rd) {
  bool anyColor = false, persists = false;
  uint32_t w = 0, h = 0;
  for (const vpgpu::Attachment& a : rd.colors) {
    if (!a.texture) continue;
    anyColor = true;
    if (isEyeTexture(*dev, a.texture) || (a.resolve && isEyeTexture(*dev, a.resolve))) return DestClass::Eye;
    for (vpgpu::TextureId id : {a.texture, a.resolve}) {
      vpgpu::TextureDesc d;  // a copy taken under the table lock (loading threads create and destroy textures)
      if (id && dev->gpu->textureDesc(id, d) && d.persistent) persists = true, w = std::max(w, d.width), h = std::max(h, d.height);
    }
  }
  if (!anyColor) return DestClass::NoColor;
  if (!persists) return DestClass::TransientRT;
  bool known = false;
  return eyeSizedPass(w, h, &known) && rd.layerCount <= 2 ? DestClass::EyeSizedNotEye : DestClass::PersistentRT;
}

uint32_t persistWaitMs(bool internalClient) {
  // Read once: 0 is a valid value (no wait), so "unset" is told apart from it.
  static const char* const env = std::getenv("VPVK_PIPELINE_PERSIST_WAIT_MS");
  if (env && *env) return uint32_t(std::strtoul(env, nullptr, 10));
  // A Vulkan game redraws what it needs and tolerates a dropped draw after 1.5 s; a GL render-to-texture drawn once (a
  // shadow map, a probe) would stay wrong, so vpgles waits as long as the legacy create did.
  return internalClient ? 10000 : 1500;
}

// How long a submit waits for the pipelines of the eye's passes (soft otherwise). 0 = no extra wait (the soft frame wait
// alone). Read at every call that has eye draws pending, so a test can change it.
uint32_t eyeHardWaitMs(bool internalClient) {
  const char* env = std::getenv("VPVK_PIPELINE_EYE_HARD_WAIT_MS");
  if (env && *env) return uint32_t(std::strtoul(env, nullptr, 10));
  return internalClient ? 1500 : 0;
}

// The eye's wait budget is spent once per pipeline key (`waited`: forgotten when the pipeline is no longer pending) and
// at most twice the budget per 1 s window (`spentMs`), whatever the number of frames and devices.
struct EyeWaitState {
  std::mutex m;
  std::unordered_set<uint64_t> waited;
  std::chrono::steady_clock::time_point windowStart{};
  double spentMs = 0;
};
static EyeWaitState& eyeWaitState() {
  static EyeWaitState s;
  return s;
}
void eyeWaitReset() {  // tests
  EyeWaitState& st = eyeWaitState();
  std::lock_guard<std::mutex> lock(st.m);
  st.waited.clear(), st.windowStart = {}, st.spentMs = 0;
}

GateResult gateFramePipelines(Device* dev, vpgpu::Frame& frame, uint64_t submitNo) {
  GateResult r;
  if (!dev || !dev->gpu || !dev->translator) return r;  // legacy mode: the core's frame hold (VP_HOLD_MS) decides
  PipelineCounters& c = pipelineCounters();
  ++c.gateFrames;
  vpgpu::PipelineCache& cache = dev->gpu->pipelines();
  if (cache.idle()) return r;  // the common case: nothing is queued or compiling
  static const uint32_t softMs = waitBudgetMs("VPVK_PIPELINE_FRAME_WAIT_MS", 3);
  const uint32_t hardMs = persistWaitMs(dev->internalClient);
  using Status = vpgpu::PipelineCache::Status;
  std::unordered_map<uint64_t, bool> pendingOf;  // key -> still queued / compiling (each looked up once)
  auto pending = [&](vpgpu::PipelineKey k) {
    if (!k.value) return false;
    auto [it, fresh] = pendingOf.emplace(k.value, false);
    if (fresh) {
      const Status st = cache.status(k);
      it->second = st == Status::Queued || st == Status::Compiling;
    }
    return it->second;
  };
  std::vector<vpgpu::PipelineKey> hard, soft;
  std::unordered_set<uint64_t> inHard, inSoft;
  std::vector<uint8_t> classOfDraw;  // every non-clear draw, in the order eraseDraws offers them (statistics only)
  for (const vpgpu::Command& cmd : frame.commands()) {
    if (const auto* rp = std::get_if<vpgpu::RenderPass>(&cmd)) {
      const DestClass cls = classifyPass(dev, rp->desc);
      const bool mustRun = cls != DestClass::Eye;
      for (const vpgpu::Draw& d : rp->draws) {
        if (d.clear) continue;
        ++r.draws;
        classOfDraw.push_back(uint8_t(cls));
        if (!pending(d.pipeline)) continue;
        ++r.pendingByClass[size_t(cls)];
        if (mustRun) {
          if (inHard.insert(d.pipeline.value).second) hard.push_back(d.pipeline);
        } else if (inSoft.insert(d.pipeline.value).second) {
          soft.push_back(d.pipeline);
        }
      }
    } else if (const auto* dp = std::get_if<vpgpu::Dispatch>(&cmd)) {
      if (pending(dp->pipeline) && inHard.insert(dp->pipeline.value).second) hard.push_back(dp->pipeline);
    }
  }
  soft.erase(std::remove_if(soft.begin(), soft.end(), [&](vpgpu::PipelineKey k) { return inHard.count(k.value) != 0; }),
             soft.end());
  r.hardPending = uint32_t(hard.size()), r.softPending = uint32_t(soft.size());
  if (hard.empty() && soft.empty()) return r;
  for (vpgpu::PipelineKey k : hard) cache.boost(k);
  for (vpgpu::PipelineKey k : soft) cache.boost(k);
  if (!hard.empty() && hardMs) {
    const auto t0 = std::chrono::steady_clock::now();
    bool complete;
    {
      const vpgpu::DemandWait demand;  // the binary-archive harvest steps aside while the frame waits
      complete = cache.waitReady(hard, std::chrono::milliseconds(hardMs));
    }
    const uint64_t us = usSince(t0);
    c.blockedGateUs += us, ++c.hardWaits;
    r.hardWaitedMs = double(us) / 1000.0;
    if (!complete) ++c.hardTimeouts;
    static std::atomic<unsigned> n{0};
    if (!complete || n.fetch_add(1) < 12)
      logf("vk.pipeline_wait what=persistent_target submit=%llu pipelines=%zu ms=%.0f complete=%d budget_ms=%u "
           "pending_draws_by_dest{eye=%u no_color=%u transient_rt=%u persistent_rt=%u eye_sized_not_marked=%u}",
           (unsigned long long)submitNo, hard.size(), r.hardWaitedMs, int(complete), hardMs, r.pendingByClass[0],
           r.pendingByClass[1], r.pendingByClass[2], r.pendingByClass[3], r.pendingByClass[4]);
  }
  // The eye budget is spent once per pipeline (a slow one would otherwise freeze every frame for the whole budget) and at
  // most twice the budget per second in all: eyeWait are the pipelines that still get their wait, eyeBudgetMs its length.
  const uint32_t eyeMs = soft.empty() ? 0 : eyeHardWaitMs(dev->internalClient);
  std::vector<vpgpu::PipelineKey> eyeWait;
  uint32_t eyeBudgetMs = 0;
  if (eyeMs) {
    EyeWaitState& st = eyeWaitState();
    std::lock_guard<std::mutex> lock(st.m);
    for (auto it = st.waited.begin(); it != st.waited.end();) it = pending(vpgpu::PipelineKey{*it}) ? std::next(it) : st.waited.erase(it);
    const auto now = std::chrono::steady_clock::now();
    if (now - st.windowStart >= std::chrono::seconds(1)) st.windowStart = now, st.spentMs = 0;
    for (vpgpu::PipelineKey k : soft)
      if (!st.waited.count(k.value)) eyeWait.push_back(k);
    const double left = 2.0 * eyeMs - st.spentMs;
    eyeBudgetMs = left >= 1.0 ? uint32_t(std::min<double>(eyeMs, left)) : 0;
    if (!eyeBudgetMs) eyeWait.clear();
    for (vpgpu::PipelineKey k : eyeWait) st.waited.insert(k.value);
  }
  const bool eyeWaits = !eyeWait.empty();
  if (!soft.empty() && (eyeWaits || softMs)) {
    const auto t0 = std::chrono::steady_clock::now();
    bool complete;
    {
      const vpgpu::DemandWait demand;  // the binary-archive harvest steps aside while the frame waits
      complete = eyeWaits ? cache.waitReady(eyeWait, std::chrono::milliseconds(eyeBudgetMs))
                          : cache.waitReady(soft, std::chrono::milliseconds(softMs));
    }
    const uint64_t us = usSince(t0);
    c.blockedGateUs += us;
    r.waitedMs = double(us) / 1000.0;
    if (eyeWaits) {
      {
        EyeWaitState& st = eyeWaitState();
        std::lock_guard<std::mutex> lock(st.m);
        st.spentMs += r.waitedMs;
      }
      static std::atomic<unsigned> lines{0};
      if (const unsigned n = lines.fetch_add(1); n < 48 || n % 120 == 0)
        logf("vk.pipeline_wait what=eye_target submit=%llu pipelines=%zu ms=%.0f complete=%d budget_ms=%u",
             (unsigned long long)submitNo, eyeWait.size(), r.waitedMs, int(complete), eyeBudgetMs);
    }
  }
  // What is still not ready is dropped from this frame (soft passes; a hard pass that timed out too: its pipelines are
  // logged above). Compute pipelines cannot be dropped here: the backend skips such a dispatch (missing_pipelines).
  pendingOf.clear();
  size_t cursor = 0;
  r.skipped = uint32_t(frame.eraseDraws([&](const vpgpu::Draw& d) {
    const bool drop = pending(d.pipeline);
    if (drop && cursor < classOfDraw.size()) ++r.skippedByClass[classOfDraw[cursor]];
    ++cursor;
    return drop;
  }));
  if (r.skipped) {
    ++c.gateSkipFrames;
    c.gateSkippedDraws += r.skipped;
    static std::atomic<unsigned> lines{0};
    const unsigned n = lines.fetch_add(1);
    if (n < 48 || n % 120 == 0)
      logf("vk.pipeline_skip submit=%llu draws_skipped=%u of %u pipelines_pending=%u waited_ms=%.1f budget_ms=%u "
           "hard_pending=%u hard_waited_ms=%.0f total_skip_frames=%llu total_skipped=%llu "
           "skipped_by_dest{eye=%u no_color=%u transient_rt=%u persistent_rt=%u eye_sized_not_marked=%u}",
           (unsigned long long)submitNo, r.skipped, r.draws, r.softPending + r.hardPending, r.waitedMs, softMs,
           r.hardPending, r.hardWaitedMs, (unsigned long long)c.gateSkipFrames.load(),
           (unsigned long long)c.gateSkippedDraws.load(), r.skippedByClass[0], r.skippedByClass[1], r.skippedByClass[2],
           r.skippedByClass[3], r.skippedByClass[4]);
  }
  return r;
}

// One line per boot (after 20 s of device life, then every 60 s while something changed, and at vkDestroyDevice):
// what the cache did and what the game's threads paid for it.
void pipelineSummary(Device* dev, bool final) {
  if (!dev) return;
  PipelineCounters& c = pipelineCounters();
  const uint64_t ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - dev->created).count());
  if (!final) {
    if (ms < dev->nextSummaryMs) return;
    dev->nextSummaryMs = ms + 60000;
  }
  const vpgpu::PipelineStats t = dev->gpu ? dev->gpu->pipelines().totals() : vpgpu::PipelineStats{};
  const std::string archive = dev->platform ? dev->platform->archiveSummary() : std::string();
  logf("vk.boot_summary%s t_s=%.0f mode=%s translator=%s | stages cached=%llu translated=%llu translate_ms=%.0f "
       "miss{new_content=%llu new_options=%llu lower=%llu emit=%llu driver=%llu baseline=%llu no_msl=%llu uncached=%llu} "
       "| pso compiled=%u failed=%u hits=%u compile_ms=%.0f max_ms=%.0f %s | pipelines created=%llu async=%llu "
       "| draws_skipped=%llu skip_frames=%llu | blocked_ms{create=%.0f record=%.0f gate=%.0f} record_waits=%llu "
       "record_timeouts=%llu hard_waits=%llu hard_timeouts=%llu indirect_patch_fb=%llu",
       final ? " final=1" : "", double(ms) / 1000.0, dev->translator ? "async" : "sync", translatorTag().c_str(),
       (unsigned long long)c.stagesCached.load(), (unsigned long long)c.stagesTranslated.load(),
       double(c.translateUs.load()) / 1000.0, (unsigned long long)c.missNewContent.load(),
       (unsigned long long)c.missNewOptions.load(), (unsigned long long)c.missLower.load(),
       (unsigned long long)c.missEmit.load(), (unsigned long long)c.missDriver.load(),
       (unsigned long long)c.missBaseline.load(), (unsigned long long)c.missNoShader.load(),
       (unsigned long long)c.missUncached.load(), t.compiled, t.failed, t.hits, t.totalMs, t.maxMs,
       archive.empty() ? "ar=none" : archive.c_str(), (unsigned long long)c.pipelinesCreated.load(),
       (unsigned long long)c.pipelinesAsync.load(), (unsigned long long)c.gateSkippedDraws.load(),
       (unsigned long long)c.gateSkipFrames.load(), double(c.blockedCreateUs.load()) / 1000.0,
       double(c.blockedRecordUs.load()) / 1000.0, double(c.blockedGateUs.load()) / 1000.0,
       (unsigned long long)c.recordWaits.load(), (unsigned long long)c.recordTimeouts.load(),
       (unsigned long long)c.hardWaits.load(), (unsigned long long)c.hardTimeouts.load(),
       (unsigned long long)c.indirectPatchFallback.load());
}

// The body of the vk.pso.census line. archived = serialized to disk; archive_pending = queued or added but not on disk yet;
// archive_spilled = refused at the harvest queue's fuse.
std::string psoCensusLine(const PipelineCounters& c, const vpgpu::PipelineStats& t, const Platform::ArchiveCensus& a) {
  char b[448];
  std::snprintf(b, sizeof b,
                "created=%llu compiled=%u failed=%llu timed_out=%llu archived=%u not_archived_depth_only=%u "
                "not_archived_other=%u archive_hits=%u compiled_in_session=%u archive_dup=%u archive_pending=%u "
                "archive_spilled=%u",
                (unsigned long long)c.pipelinesCreated.load(), t.compiled, (unsigned long long)(t.failed + c.translateFailed.load()),
                (unsigned long long)c.createTimedOut.load(), a.archived, a.depthOnly, a.other, a.hits, a.compiled, a.dup,
                a.pending, a.spilled);
  return b;
}

// created = vkCreate*Pipelines objects made; compiled / failed = the cache's backend compiles (failed also counts the
// translations that never got there); timed_out = creations that came back unready after VPVK_PIPELINE_CREATE_WAIT_MS
// (synchronous mode only: the asynchronous mode never waits). archived ... compiled_in_session: Platform::ArchiveCensus
// (depth-only pipelines are archived in files of their own; not_archived_depth_only = those whose archive failed).
// Called from the creation calls and the submit tick (every 10 s while a counter moved), and with `force` at the cache
// flush and device destroy (always printed).
void psoCensus(Device* dev, bool force) {
  if (!dev) return;
  static std::atomic<int64_t> nextNs{0};
  const int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!force && nowNs < nextNs.load(std::memory_order_relaxed)) return;
  const PipelineCounters& c = pipelineCounters();
  const vpgpu::PipelineStats t = dev->gpu ? dev->gpu->pipelines().totals() : vpgpu::PipelineStats{};
  Platform::ArchiveCensus a;
  if (dev->platform) dev->platform->archiveCensus(a);
  const std::string b = psoCensusLine(c, t, a);
  static std::mutex m;
  static std::string last;
  std::lock_guard<std::mutex> lock(m);
  nextNs.store(nowNs + 10'000'000'000LL, std::memory_order_relaxed);
  if (!force && last == b) return;
  last = b;
  logf("vk.pso.census %s", b.c_str());
}

// gpu.amp: at most every 10 s, the multiview draws recorded since the last line (none: no line). inst = multiview by
// instancing, split by why the pipeline is not vertex amplified (PipelineObj::AmpWhy); verts_inst counts the views too.
void ampCensus() {
  static std::atomic<int64_t> nextNs{0};
  const int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (nowNs < nextNs.load(std::memory_order_relaxed)) return;
  nextNs.store(nowNs + 10'000'000'000LL, std::memory_order_relaxed);
  PipelineCounters& c = pipelineCounters();
  const uint64_t amp = c.ampDraws.exchange(0, std::memory_order_relaxed), inst = c.ampInstDraws.exchange(0, std::memory_order_relaxed),
                 verts = c.ampInstVerts.exchange(0, std::memory_order_relaxed);
  uint64_t by[PipelineObj::AmpWhyCount];
  for (int i = 0; i < PipelineObj::AmpWhyCount; ++i) by[i] = c.ampBy[i].exchange(0, std::memory_order_relaxed);
  if (!amp && !inst) return;
  logf("gpu.amp draws_amp=%llu draws_inst=%llu by: clip=%llu,layer=%llu,mask=%llu,off=%llu verts_inst=%llu", (unsigned long long)amp,
       (unsigned long long)inst, (unsigned long long)by[PipelineObj::AmpClip], (unsigned long long)by[PipelineObj::AmpLayer],
       (unsigned long long)by[PipelineObj::AmpMask], (unsigned long long)by[PipelineObj::AmpOff], (unsigned long long)verts);
}

// The pipeline key a draw into a pass of `passSamples` (effective count of its attachments; 0 =
// unknown, e.g. a secondary command buffer) uses: the pipeline itself (or its texel buffer variant
// `tv`) when the counts agree, else a variant with the pass's count, requested and recorded
// (recipes) the first time it is needed. `eds`: the command buffer's extended dynamic state, applied
// on top the same way (a variant per combination of replaced values).
vpgpu::PipelineKey pipelineFor(Device* dev, const PipelineObj& p, uint8_t passSamples, PipelineObj::TexelVariant* tv,
                               const BoundState::Eds* eds) {
  const vpgpu::PipelineState& base = tv ? tv->state : p.state;
  vpgpu::PipelineKey baseKey = tv ? tv->key : p.key;
  vpgpu::PipelineKey fresh{};  // a variant this call requested: awaited after the lock is released
  if (!p.compute && passSamples && passSamples != base.samples) {
    std::lock_guard<std::mutex> lock(p.variantMutex);
    auto& list = tv ? tv->samples : p.variants;
    bool found = false;
    for (const auto& [n, k] : list)
      if (n == passSamples) baseKey = k, found = true;
    if (!found) {
      vpgpu::PipelineState st = base;
      st.samples = passSamples;
      st.label += ".ms" + std::to_string(passSamples);
      const vpgpu::PipelineKey k = st.key();
      list.push_back({passSamples, k});
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 16)
        logf("vk.msaa pipeline_variant pipeline=%s samples=%u->%u (game %ux)", base.label.c_str(),
             unsigned(base.samples), unsigned(passSamples), unsigned(p.gameSamples));
      if (dev && p.ok) {
        retainKey(dev, p, k);
        dev->gpu->pipelines().request(st, k, vpgpu::Priority::Blocking);
        if (dev->recipes) dev->recipes->addPipeline(st, tv ? tv->inputs : p.inputs);
        fresh = k;
      }
      baseKey = k;
    }
  }
  if (fresh.value) awaitPipeline(dev, fresh, "msaa_variant");
  if (!eds || p.compute || p.dynamic.empty()) return baseKey;
  // State the pipeline variant above was made from (its samples differ only by the count).
  vpgpu::PipelineState st = base;
  if (passSamples && passSamples != base.samples) st.samples = passSamples;
  const uint64_t h = applyEds(p, *eds, st);
  if (!h) return baseKey;
  const uint64_t id = h ^ (baseKey.value * 0x9E3779B97F4A7C15ull);
  vpgpu::PipelineKey k;
  {
    std::lock_guard<std::mutex> lock(p.variantMutex);
    auto it = p.edsVariants.find(id);
    if (it != p.edsVariants.end()) return it->second;
    st.label += ".eds" + std::to_string(p.edsVariants.size());
    k = st.key();
    p.edsVariants.emplace(id, k);
    static std::atomic<unsigned> n{0};
    if (n.fetch_add(1) < 16) logf("vk.eds pipeline_variant pipeline=%s -> %s", base.label.c_str(), st.label.c_str());
    if (!(dev && p.ok)) return k;
    retainKey(dev, p, k);
    dev->gpu->pipelines().request(st, k, vpgpu::Priority::Blocking);
    if (dev->recipes) dev->recipes->addPipeline(st, tv ? tv->inputs : p.inputs);
  }
  awaitPipeline(dev, k, "eds_variant");
  return k;
}

// ---- upload arena -------------------------------------------------------------------------------
vpgpu::BufferId UploadArena::put(Device* d, const void* data, VkDeviceSize size, VkDeviceSize& offset) {
  const VkDeviceSize need = (size + 255) & ~VkDeviceSize(255);
  while (current < chunks.size() && chunks[current].used + need > chunks[current].size) ++current;
  if (current == chunks.size()) {
    Chunk c;
    c.size = std::max<VkDeviceSize>(65536, (need + 16383) & ~VkDeviceSize(16383));
    c.host = static_cast<uint8_t*>(std::aligned_alloc(16384, c.size));
    vpgpu::BufferDesc bd;
    bd.size = c.size;
    bd.guestMemory = c.host;
    bd.label = "vk.upload_arena";
    {
      // Recording threads create buffers while vkQueueSubmit compiles a frame on another.
      std::lock_guard<std::mutex> lock(d->submitMutex);
      c.id = d->gpu->createBuffer(bd);
    }
    chunks.push_back(c);
  }
  Chunk& c = chunks[current];
  offset = c.used;
  if (data) std::memcpy(c.host + c.used, data, size);
  else std::memset(c.host + c.used, 0, size);  // reserved for a later fill (decoded 3D ASTC uploads)
  c.used += need;
  return c.id;
}

void UploadArena::release(Device* d) {
  for (auto& c : chunks) {
    if (d && c.id) {
      // Same lock as put(): the backend's buffer map and retire list belong to the submit thread.
      std::lock_guard<std::mutex> lock(d->submitMutex);
      d->gpu->destroy(c.id);
    }
    std::free(c.host);
  }
  chunks.clear();
  current = 0;
}

Recorder::~Recorder() { arena.release(device); }

namespace {

vpgpu::BlendFactor blendFactor(VkBlendFactor f) {
  using B = vpgpu::BlendFactor;
  switch (f) {
    case VK_BLEND_FACTOR_ZERO: return B::Zero;
    case VK_BLEND_FACTOR_ONE: return B::One;
    case VK_BLEND_FACTOR_SRC_COLOR: return B::SrcColor;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return B::OneMinusSrcColor;
    case VK_BLEND_FACTOR_DST_COLOR: return B::DstColor;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return B::OneMinusDstColor;
    case VK_BLEND_FACTOR_SRC_ALPHA: return B::SrcAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return B::OneMinusSrcAlpha;
    case VK_BLEND_FACTOR_DST_ALPHA: return B::DstAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return B::OneMinusDstAlpha;
    case VK_BLEND_FACTOR_CONSTANT_COLOR: return B::Constant;
    case VK_BLEND_FACTOR_CONSTANT_ALPHA: return B::ConstantAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return B::OneMinusConstant;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return B::OneMinusConstantAlpha;
    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return B::SrcAlphaSaturate;
    case VK_BLEND_FACTOR_SRC1_COLOR: return B::Src1Color;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return B::OneMinusSrc1Color;
    case VK_BLEND_FACTOR_SRC1_ALPHA: return B::Src1Alpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return B::OneMinusSrc1Alpha;
    default: return B::One;
  }
}
vpgpu::BlendOp blendOp(VkBlendOp op) {
  switch (op) {
    case VK_BLEND_OP_SUBTRACT: return vpgpu::BlendOp::Subtract;
    case VK_BLEND_OP_REVERSE_SUBTRACT: return vpgpu::BlendOp::ReverseSubtract;
    case VK_BLEND_OP_MIN: return vpgpu::BlendOp::Min;
    case VK_BLEND_OP_MAX: return vpgpu::BlendOp::Max;
    default: return vpgpu::BlendOp::Add;
  }
}
vpgpu::StencilFace stencilFace(const VkStencilOpState& s) {
  vpgpu::StencilFace f;
  f.fail = static_cast<vpgpu::StencilOp>(s.failOp);  // same order as Vulkan
  f.depthFail = static_cast<vpgpu::StencilOp>(s.depthFailOp);
  f.pass = static_cast<vpgpu::StencilOp>(s.passOp);
  f.compare = static_cast<vpgpu::CompareOp>(s.compareOp);
  f.readMask = static_cast<uint8_t>(s.compareMask);
  f.writeMask = static_cast<uint8_t>(s.writeMask);
  return f;
}
vpgpu::Primitive primitive(VkPrimitiveTopology t, bool& ok) {
  ok = true;
  switch (t) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return vpgpu::Primitive::Points;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return vpgpu::Primitive::Lines;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return vpgpu::Primitive::LineStrip;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: return vpgpu::Primitive::TriangleStrip;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return vpgpu::Primitive::Triangles;
    default: ok = false; return vpgpu::Primitive::Triangles;  // fans, adjacency, patches
  }
}

vpgpu::BindKind kindOf(VkDescriptorType t) {
  switch (t) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:  // a uniform block whose bytes live in the set
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: return vpgpu::BindKind::ConstantBuffer;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return vpgpu::BindKind::Buffer;
    case VK_DESCRIPTOR_TYPE_SAMPLER: return vpgpu::BindKind::Sampler;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return vpgpu::BindKind::StorageTexture;
    default: return vpgpu::BindKind::Texture;  // sampled, combined, input attachment
  }
}

// Assigns argument-table slots to a layout.
void assignSlots(PipelineLayoutObj& pl) {
  uint16_t next[6] = {0, 0, 0, 0, 0, 0};
  for (uint32_t s = 0; s < pl.sets.size(); ++s) {
    if (!pl.sets[s]) continue;
    for (const SetLayoutBinding& b : pl.sets[s]->bindings) {
      LayoutSlot ls;
      ls.set = s, ls.binding = b.binding, ls.count = std::max<uint32_t>(1, b.count), ls.type = b.type;
      if (b.type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) ls.count = 1;  // descriptorCount is its size in bytes
      ls.kind = kindOf(b.type);
      ls.slot = next[int(ls.kind)];
      next[int(ls.kind)] = uint16_t(next[int(ls.kind)] + ls.count);
      if (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        ls.samplerSlot = next[int(vpgpu::BindKind::Sampler)];
        next[int(vpgpu::BindKind::Sampler)] = uint16_t(next[int(vpgpu::BindKind::Sampler)] + ls.count);
        // Y'CbCr conversion (immutable sampler): planes 1 and 2 are further textures of the binding.
        if (!b.immutable.empty() && b.immutable[0] && b.immutable[0]->ycbcr) {
          ls.ycbcr = b.immutable[0]->ycbcr;
          for (unsigned k = 0; k + 1 < ls.ycbcr->spec.planes; ++k) {
            ls.planeSlot[k] = next[int(vpgpu::BindKind::Texture)];
            next[int(vpgpu::BindKind::Texture)] = uint16_t(next[int(vpgpu::BindKind::Texture)] + ls.count);
          }
        }
      }
      pl.slots.push_back(ls);
    }
  }
  if (pl.pushSize) pl.pushSlot = next[int(vpgpu::BindKind::ConstantBuffer)]++;
  pl.nextBuffer = next[int(vpgpu::BindKind::Buffer)];
}

std::mutex gTranslateLog;

// Vertex formats: the texture formats, plus the three-component ones read as their four-component
// twin with the fourth left out.
spirv::VertexAttribute vertexAttribute(const VkVertexInputAttributeDescription& a) {
  spirv::VertexAttribute v;
  v.location = a.location, v.binding = a.binding, v.offset = a.offset;
  switch (a.format) {
    case VK_FORMAT_R32G32B32_SFLOAT: v.format = vpgpu::Format::RGBA32Float, v.components = 3; break;
    case VK_FORMAT_R32G32B32_UINT: v.format = vpgpu::Format::RGBA32Uint, v.components = 3; break;
    case VK_FORMAT_R32G32B32_SINT: v.format = vpgpu::Format::RGBA32Sint, v.components = 3; break;
    case VK_FORMAT_R16G16B16_SFLOAT: v.format = vpgpu::Format::RGBA16Float, v.components = 3; break;
    case VK_FORMAT_R16G16B16_UNORM: v.format = vpgpu::Format::RGBA16Unorm, v.components = 3; break;
    case VK_FORMAT_R16G16B16_SNORM: v.format = vpgpu::Format::RGBA16Snorm, v.components = 3; break;
    case VK_FORMAT_R8G8B8_UNORM: v.format = vpgpu::Format::RGBA8Unorm, v.components = 3; break;
    case VK_FORMAT_R8G8B8_SNORM: v.format = vpgpu::Format::RGBA8Snorm, v.components = 3; break;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: v.format = vpgpu::Format::RGB10A2Unorm; break;
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: v.format = vpgpu::Format::RGB10A2Uint; break;
    // Stored as R/B-swapped twins (isSwappedRB): the fetch has no swap, so they are not vertex formats (no vertex-buffer feature).
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_UINT_PACK32: v.format = vpgpu::Format::Undefined; break;
    default: v.format = toCoreFormat(a.format);
  }
  if (v.format == vpgpu::Format::Undefined) logf("vk.err what=vertex_format value=%d", int(a.format));
  return v;
}

}  // namespace
}  // namespace vpvk

using namespace vpvk;

// =================================================================================================
// Shader modules
// =================================================================================================
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice dv, const VkShaderModuleCreateInfo* ci,
                                                                const VkAllocationCallbacks*, VkShaderModule* out) {
  auto* m = new ShaderModuleObj;
  m->module = std::shared_ptr<spirv::SpirvModule>(spirv::parse(ci->pCode, ci->codeSize / 4, m->error), spirv::release);
  if (!m->module) logf("vk.err what=spirv_parse error=\"%s\"", m->error.c_str());
  const Device* d = dv ? from<Device>(dv) : nullptr;
  if (m->module && d && d->recipes && d->recipes->enabled()) {
    auto w = std::make_shared<std::vector<uint32_t>>(ci->pCode, ci->pCode + ci->codeSize / 4);
    m->hash = spirvHash(w->data(), w->size());
    m->words = std::move(w);
  }
  *out = to<VkShaderModule>(m);
  return VK_SUCCESS;  // the error surfaces at pipeline creation
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice, VkShaderModule m, const VkAllocationCallbacks*) {
  delete from<ShaderModuleObj>(m);
}

// =================================================================================================
// Descriptor set layouts, pipeline layouts
// =================================================================================================
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo* ci,
                                                                       const VkAllocationCallbacks*,
                                                                       VkDescriptorSetLayout* out) {
  auto data = std::make_shared<SetLayoutData>();
  data->push = (ci->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) != 0;
  for (uint32_t i = 0; i < ci->bindingCount; ++i) {
    const VkDescriptorSetLayoutBinding& b = ci->pBindings[i];
    SetLayoutBinding x;
    x.binding = b.binding, x.count = b.descriptorCount, x.type = b.descriptorType, x.stages = b.stageFlags;
    if (b.pImmutableSamplers)
      for (uint32_t k = 0; k < b.descriptorCount; ++k) x.immutable.push_back(from<Sampler>(b.pImmutableSamplers[k]));
    if (x.count == 0) continue;
    data->bindings.push_back(std::move(x));
  }
  std::sort(data->bindings.begin(), data->bindings.end(),
            [](const SetLayoutBinding& a, const SetLayoutBinding& b) { return a.binding < b.binding; });
  auto* l = new DescriptorSetLayoutObj;
  l->data = data;
  *out = to<VkDescriptorSetLayout>(l);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout l,
                                                                    const VkAllocationCallbacks*) {
  delete from<DescriptorSetLayoutObj>(l);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDescriptorSetLayoutSupport(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                                       VkDescriptorSetLayoutSupport* s) {
  s->supported = VK_TRUE;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo* ci,
                                                                  const VkAllocationCallbacks*, VkPipelineLayout* out) {
  auto pl = std::make_shared<PipelineLayoutObj>();
  for (uint32_t i = 0; i < ci->setLayoutCount; ++i)
    pl->sets.push_back(ci->pSetLayouts[i] ? from<DescriptorSetLayoutObj>(ci->pSetLayouts[i])->data : nullptr);
  for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i)
    pl->pushSize = std::max(pl->pushSize, ci->pPushConstantRanges[i].offset + ci->pPushConstantRanges[i].size);
  assignSlots(*pl);
  *out = reinterpret_cast<VkPipelineLayout>(new std::shared_ptr<const PipelineLayoutObj>(pl));
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice, VkPipelineLayout l, const VkAllocationCallbacks*) {
  delete reinterpret_cast<std::shared_ptr<const PipelineLayoutObj>*>(l);
}

namespace {
std::shared_ptr<const PipelineLayoutObj> layoutOf(VkPipelineLayout l) {
  return l ? *reinterpret_cast<std::shared_ptr<const PipelineLayoutObj>*>(l) : nullptr;
}
}  // namespace

// =================================================================================================
// Descriptor pools and sets
// =================================================================================================
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                                  const VkAllocationCallbacks*, VkDescriptorPool* out) {
  *out = to<VkDescriptorPool>(new DescriptorPoolObj);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice, VkDescriptorPool p, VkDescriptorPoolResetFlags) {
  auto* pool = from<DescriptorPoolObj>(p);
  for (auto* s : pool->sets) delete s;
  pool->sets.clear();
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice d, VkDescriptorPool p, const VkAllocationCallbacks*) {
  if (!p) return;
  vkResetDescriptorPool(d, p, 0);
  delete from<DescriptorPoolObj>(p);
}

namespace {
// An empty set of `layout` (every descriptor unwritten, immutable samplers in place).
void initDescriptorSet(DescriptorSetObj& s, std::shared_ptr<const SetLayoutData> layout) {
  s.layout = std::move(layout);
  s.slots.clear(), s.inlineData.clear();
  s.slots.resize(s.layout->bindings.size());
  s.inlineData.resize(s.layout->bindings.size());
  for (size_t b = 0; b < s.layout->bindings.size(); ++b) {
    const SetLayoutBinding& lb = s.layout->bindings[b];
    if (lb.type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
      s.slots[b].resize(1);
      s.slots[b][0].type = lb.type;
      s.inlineData[b].assign(lb.count, 0);
      continue;
    }
    s.slots[b].resize(lb.count);
    for (uint32_t k = 0; k < lb.immutable.size() && k < lb.count; ++k) {
      s.slots[b][k].sampler = lb.immutable[k];
      s.slots[b][k].type = lb.type;
      s.slots[b][k].immutable = true;
    }
  }
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* ai,
                                                                    VkDescriptorSet* out) {
  auto* pool = from<DescriptorPoolObj>(ai->descriptorPool);
  for (uint32_t i = 0; i < ai->descriptorSetCount; ++i) {
    auto* s = new DescriptorSetObj;
    initDescriptorSet(*s, from<DescriptorSetLayoutObj>(ai->pSetLayouts[i])->data);
    s->pool = pool;
    pool->sets.push_back(s);
    out[i] = to<VkDescriptorSet>(s);
  }
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice, VkDescriptorPool p, uint32_t n,
                                                                const VkDescriptorSet* sets) {
  auto* pool = from<DescriptorPoolObj>(p);
  for (uint32_t i = 0; i < n; ++i) {
    if (!sets[i]) continue;
    auto* s = from<DescriptorSetObj>(sets[i]);
    pool->sets.erase(std::remove(pool->sets.begin(), pool->sets.end(), s), pool->sets.end());
    delete s;
  }
  return VK_SUCCESS;
}

namespace {
// Element `e` of binding `binding` of a set, following Vulkan's rule that writes past a binding's
// count continue into the next binding.
Descriptor* element(DescriptorSetObj* s, uint32_t binding, uint32_t e) {
  for (size_t b = 0; b < s->layout->bindings.size(); ++b) {
    if (s->layout->bindings[b].binding < binding) continue;
    if (s->layout->bindings[b].binding == binding || e < s->slots[b].size()) {
      if (e < s->slots[b].size()) return &s->slots[b][e];
      e -= uint32_t(s->slots[b].size());
      binding = b + 1 < s->layout->bindings.size() ? s->layout->bindings[b + 1].binding : binding + 1;
    }
  }
  return nullptr;
}

void writeImage(Descriptor& d, VkDescriptorType t, const VkDescriptorImageInfo& ii) {
  d.type = t;
  if (t == VK_DESCRIPTOR_TYPE_SAMPLER || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
    if (ii.sampler && !d.immutable) d.sampler = from<Sampler>(ii.sampler);  // an immutable sampler stays
  }
  if (t != VK_DESCRIPTOR_TYPE_SAMPLER) d.view = ii.imageView ? from<ImageView>(ii.imageView) : nullptr;
}
void writeBuffer(Descriptor& d, VkDescriptorType t, const VkDescriptorBufferInfo& bi) {
  d.type = t;
  d.buffer = bi.buffer ? from<Buffer>(bi.buffer) : nullptr;
  d.offset = bi.offset;
  d.range = bi.range == VK_WHOLE_SIZE && d.buffer ? d.buffer->size - bi.offset : bi.range;
}
}  // namespace

namespace {
// The inline block bytes of binding number `binding` of a set (null: not an inline block).
std::vector<uint8_t>* inlineBlock(DescriptorSetObj* s, uint32_t binding) {
  for (size_t b = 0; b < s->layout->bindings.size(); ++b)
    if (s->layout->bindings[b].binding == binding)
      return s->layout->bindings[b].type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK && b < s->inlineData.size()
                 ? &s->inlineData[b]
                 : nullptr;
  return nullptr;
}
void writeInline(DescriptorSetObj* s, uint32_t binding, uint32_t offset, uint32_t size, const void* data) {
  std::vector<uint8_t>* blk = inlineBlock(s, binding);
  if (!blk || offset > blk->size() || size > blk->size() - offset) return;
  std::memcpy(blk->data() + offset, data, size);
}
}  // namespace

namespace {
// One write into `s` (vkUpdateDescriptorSets and vkCmdPushDescriptorSetKHR share it; the set is w.dstSet or the push set).
void applyWrite(DescriptorSetObj* s, const VkWriteDescriptorSet& w) {
  if (w.descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
    // dstArrayElement is a byte offset, descriptorCount a byte count; the data is in the pNext chain.
    for (auto* n = static_cast<const VkBaseInStructure*>(w.pNext); n; n = n->pNext)
      if (n->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK) {
        auto* iu = reinterpret_cast<const VkWriteDescriptorSetInlineUniformBlock*>(n);
        writeInline(s, w.dstBinding, w.dstArrayElement, std::min(w.descriptorCount, iu->dataSize), iu->pData);
      }
    return;
  }
  for (uint32_t k = 0; k < w.descriptorCount; ++k) {
    Descriptor* d = element(s, w.dstBinding, w.dstArrayElement + k);
    if (!d) break;
    switch (w.descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        writeImage(*d, w.descriptorType, w.pImageInfo[k]);
        break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        d->type = w.descriptorType;
        d->texel = w.pTexelBufferView[k] ? from<BufferView>(w.pTexelBufferView[k]) : nullptr;
        break;
      default: writeBuffer(*d, w.descriptorType, w.pBufferInfo[k]); break;
    }
  }
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice, uint32_t nw, const VkWriteDescriptorSet* w,
                                                              uint32_t nc, const VkCopyDescriptorSet* c) {
  for (uint32_t i = 0; i < nw; ++i) applyWrite(from<DescriptorSetObj>(w[i].dstSet), w[i]);
  for (uint32_t i = 0; i < nc; ++i) {
    auto* src = from<DescriptorSetObj>(c[i].srcSet);
    auto* dst = from<DescriptorSetObj>(c[i].dstSet);
    if (std::vector<uint8_t>* sb = inlineBlock(src, c[i].srcBinding)) {
      if (c[i].srcArrayElement <= sb->size() && c[i].descriptorCount <= sb->size() - c[i].srcArrayElement)
        writeInline(dst, c[i].dstBinding, c[i].dstArrayElement, c[i].descriptorCount, sb->data() + c[i].srcArrayElement);
      continue;
    }
    for (uint32_t k = 0; k < c[i].descriptorCount; ++k) {
      Descriptor* a = element(src, c[i].srcBinding, c[i].srcArrayElement + k);
      Descriptor* b = element(dst, c[i].dstBinding, c[i].dstArrayElement + k);
      if (a && b) *b = *a;
    }
  }
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorUpdateTemplate(VkDevice, const VkDescriptorUpdateTemplateCreateInfo* ci,
                                                                            const VkAllocationCallbacks*,
                                                                            VkDescriptorUpdateTemplate* out) {
  auto* t = new DescriptorUpdateTemplateObj;
  t->entries.assign(ci->pDescriptorUpdateEntries, ci->pDescriptorUpdateEntries + ci->descriptorUpdateEntryCount);
  t->push = ci->templateType == VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
  t->bindPoint = ci->pipelineBindPoint, t->set = ci->set;
  *out = to<VkDescriptorUpdateTemplate>(t);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorUpdateTemplate(VkDevice, VkDescriptorUpdateTemplate t,
                                                                         const VkAllocationCallbacks*) {
  delete from<DescriptorUpdateTemplateObj>(t);
}

namespace {
void applyTemplate(DescriptorSetObj* s, VkDescriptorUpdateTemplate tpl, const void* data) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (const VkDescriptorUpdateTemplateEntry& e : from<DescriptorUpdateTemplateObj>(tpl)->entries) {
    if (e.descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {  // bytes at `offset`, count = size
      writeInline(s, e.dstBinding, e.dstArrayElement, e.descriptorCount, bytes + e.offset);
      continue;
    }
    for (uint32_t k = 0; k < e.descriptorCount; ++k) {
      Descriptor* d = element(s, e.dstBinding, e.dstArrayElement + k);
      if (!d) break;
      const uint8_t* p = bytes + e.offset + k * e.stride;
      switch (e.descriptorType) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
          VkDescriptorImageInfo ii;
          std::memcpy(&ii, p, sizeof ii);
          writeImage(*d, e.descriptorType, ii);
          break;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
          VkBufferView bv;
          std::memcpy(&bv, p, sizeof bv);
          d->type = e.descriptorType;
          d->texel = bv ? from<BufferView>(bv) : nullptr;
          break;
        }
        default: {
          VkDescriptorBufferInfo bi;
          std::memcpy(&bi, p, sizeof bi);
          writeBuffer(*d, e.descriptorType, bi);
        }
      }
    }
  }
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSetWithTemplate(VkDevice, VkDescriptorSet set,
                                                                         VkDescriptorUpdateTemplate tpl, const void* data) {
  applyTemplate(from<DescriptorSetObj>(set), tpl, data);
}

// =================================================================================================
// Pipelines
// =================================================================================================
namespace {

// Maximum scalar components that can pass between stages on Apple GPUs. We count Position
// conservatively (+4), so this is where Metal can really fail, not the Vulkan-reported limit.
constexpr uint32_t kMetalMaxVaryingComponents = 124;

struct StageOut {
  bool ok = false;
  std::string why;  // when !ok: the first error (vk.pso.fail)
  uint64_t id = 0;
  uint64_t input = 0;  // StageInput hash when recipes are recorded
  uint32_t varyingComponents = 0;  // vertex: scalar components written to varyings (+4 for Position)
  spirv::SpirvInterface iface;
  vpmsl::MslShader msl;
};

std::vector<std::pair<uint32_t, uint64_t>> specialization(const VkSpecializationInfo* si) {
  std::vector<std::pair<uint32_t, uint64_t>> out;
  if (!si) return out;
  for (uint32_t i = 0; i < si->mapEntryCount; ++i) {
    const VkSpecializationMapEntry& e = si->pMapEntries[i];
    uint64_t v = 0;
    std::memcpy(&v, static_cast<const uint8_t*>(si->pData) + e.offset, std::min<size_t>(e.size, 8));
    out.push_back({e.constantID, v});
  }
  return out;
}

vpir::Stage vpirStage(VkShaderStageFlagBits s) {
  switch (s) {
    case VK_SHADER_STAGE_FRAGMENT_BIT: return vpir::Stage::Fragment;
    case VK_SHADER_STAGE_COMPUTE_BIT: return vpir::Stage::Compute;
    default: return vpir::Stage::Vertex;
  }
}

}  // namespace
namespace vpvk {
void noteShaderLib(uint64_t mslId);  // precompile.cpp
}
namespace {

// Diagnostic for the per-invocation memory of a translated shader: how many private slots (one
// per SPIR-V Function/Private variable) it has and how many are indexed with a non-constant
// address. A dynamically indexed slot cannot live in registers on Apple GPUs (stack memory), so
// this is the first thing to look at when one material is far slower than on Quest.
namespace {
struct PrivateStats {
  uint32_t slots = 0, bytes = 0, dynSlots = 0, dynBytes = 0, loops = 0;
};

void privateWalk(const vpir::Block& b, const std::unordered_map<vpir::ValueId, const vpir::Inst*>& defs,
                 std::vector<uint8_t>& dyn, PrivateStats& st) {
  for (const vpir::Node& n : b) {
    if (n.kind == vpir::NodeKind::Inst) {
      const vpir::Inst& in = n.inst;
      if ((in.op == vpir::Op::Load || in.op == vpir::Op::Store) && vpir::isPrivateSlot(in.sym) && !in.args.empty()) {
        auto d = defs.find(in.args[0]);
        bool isConst = d != defs.end() && d->second->op == vpir::Op::Const;
        uint32_t slot = vpir::privateSlotIndex(in.sym);
        if (!isConst && slot < dyn.size()) dyn[slot] = 1;
      }
    }
    if (n.kind == vpir::NodeKind::Loop) ++st.loops;
    privateWalk(n.body, defs, dyn, st);
    privateWalk(n.elseBody, defs, dyn, st);
  }
}

void collectDefs(const vpir::Block& b, std::unordered_map<vpir::ValueId, const vpir::Inst*>& defs) {
  for (const vpir::Node& n : b) {
    if (n.kind == vpir::NodeKind::Inst && n.inst.result != vpir::kNoValue) defs[n.inst.result] = &n.inst;
    collectDefs(n.body, defs);
    collectDefs(n.elseBody, defs);
  }
}

void logPrivateStats(const vpir::Module& mod, uint64_t id, vpir::Stage stage, const std::string& label) {
  if (mod.privateSlots.empty()) return;
  static std::mutex m;
  static std::unordered_set<uint64_t> seen;
  static uint32_t lines = 0;
  {
    std::lock_guard<std::mutex> lock(m);
    if (lines >= 400 || !seen.insert(id).second) return;
  }
  PrivateStats st;
  std::vector<uint8_t> dyn(mod.privateSlots.size(), 0);
  for (const vpir::Function& f : mod.functions) {
    std::unordered_map<vpir::ValueId, const vpir::Inst*> defs;
    collectDefs(f.body, defs);
    privateWalk(f.body, defs, dyn, st);
  }
  st.slots = uint32_t(mod.privateSlots.size());
  for (size_t i = 0; i < dyn.size(); ++i) {
    st.bytes += mod.privateSlots[i];
    if (dyn[i]) ++st.dynSlots, st.dynBytes += mod.privateSlots[i];
  }
  std::lock_guard<std::mutex> lock(m);
  // Every shader with a dynamically indexed slot, plus the first few others for scale.
  if (st.dynSlots == 0 && lines >= 24) return;
  if (lines >= 400) return;
  ++lines;
  logf("gpu.shader.mem id=%016llx stage=%s pipe=%s slots=%u bytes=%u dyn_slots=%u dyn_bytes=%u loops=%u",
       (unsigned long long)id, vpir::toString(stage), label.c_str(), st.slots, st.bytes, st.dynSlots, st.dynBytes,
       st.loops);
}
}  // namespace

// SPIR-V -> VPIR -> MSL, shared by pipeline creation and recipe retranslation so the same input
// gives the same MSL (and id = content hash of the source) in both.
std::atomic<uint64_t> gStageTranslations{0};  // vpvk::stageTranslations()

enum class TranslateFail { None, Lower, Msl };
TranslateFail lowerAndEmit(const spirv::SpirvModule& m, const spirv::SpirvLowerOptions& opt, vpir::Module& mod,
                           spirv::SpirvInterface& iface, vpmsl::MslShader& msl, uint64_t& id,
                           std::vector<std::string>& errors) {
  std::string error;
  ++gStageTranslations;
  if (!spirv::lower(m, opt, mod, iface, error)) {
    errors.push_back(std::move(error));
    return TranslateFail::Lower;
  }
  vpmsl::EmitOptions eo;
  eo.positionalNames = true;  // MSL text independent of absolute slots (precompile == runtime)
  eo.loopGuard = kVulkanLoopGuard;
  eo.robustBuffers = opt.robustBuffers;  // robustBufferAccess (VPVK_ROBUST): OOB / null storage buffers cannot fault the GPU
  vpmsl::EmitResult e = vpmsl::emitMsl(mod, eo);
  if (!e.errors.empty()) {
    errors = std::move(e.errors);
    return TranslateFail::Msl;
  }
  msl = std::move(e.shader);
  id = vpmsl::shaderId(msl);  // source + argument slots
  if (std::string dd; dumpDirFor(id, dd)) {  // harness: every translated shader (VPVK_DUMP_MSL_FILTER: only those)
    char name[64];
    std::snprintf(name, sizeof name, "/%016llx.%d.metal", (unsigned long long)id, int(opt.stage));
    if (FILE* f = std::fopen((dd + name).c_str(), "w")) { std::fputs(msl.source.c_str(), f); std::fclose(f); }
  }
  return TranslateFail::None;
}

// What a recorded translation has to give back besides the MSL (see StageIface).
StageIface ifaceOf(const vpir::Module& mod, const spirv::SpirvInterface& si, const vpmsl::MslShader& msl,
                   vpir::Stage stage) {
  StageIface f;
  for (const spirv::SpirvBinding& b : si.bindings) {
    spirv::SpirvBinding x;
    x.set = b.set, x.binding = b.binding, x.written = b.written, x.texelFormat = b.texelFormat;
    f.bindings.push_back(x);
  }
  f.usesPushConstants = si.usesPushConstants;
  f.colorOutputs = si.colorOutputs;
  f.mslColorOutputs = msl.colorOutputs;
  f.dualSource = msl.dualSource;
  f.localSize = si.localSize;
  f.amplified = si.amplified;
  if (stage == vpir::Stage::Vertex)
    for (const vpir::IoVar& v : mod.outputs) {
      if (v.builtin == vpir::Builtin::None) f.varyingComponents += v.type.width;
      else if (v.builtin == vpir::Builtin::Position) f.varyingComponents += 4;
    }
  return f;
}

ShaderBlob blobOf(const vpmsl::MslShader& msl) {
  ShaderBlob blob;
  blob.source = msl.source;
  blob.threadsPerGroup = msl.workgroupSize;
  blob.colorOutputs = msl.colorOutputs;
  for (const auto& a : msl.arguments)
    blob.arguments.push_back({static_cast<vpgpu::BindKind>(int(a.kind) + 1), uint16_t(a.index), a.offset});
  return blob;
}

// VPVK_STAGE_CACHE=0: always translate (diagnosis).
bool stageCacheOn() {
  static const bool on = [] {
    const char* e = getenv("VPVK_STAGE_CACHE");
    return !(e && e[0] == '0' && !e[1]);
  }();
  return on;
}

// What a pipeline stage is translated from (kept by pipelines with texel buffers, see
// PipelineObj::texel, so a variant can be translated after the module is destroyed).
PipelineObj::StageSrc stageSrc(const VkPipelineShaderStageCreateInfo& st) {
  PipelineObj::StageSrc s;
  if (const ShaderModuleObj* sm = from<ShaderModuleObj>(st.module)) s.module = sm->module, s.words = sm->words, s.hash = sm->hash;
  s.entry = st.pName ? st.pName : "main";
  s.stage = st.stage;
  s.spec = specialization(st.pSpecializationInfo);
  return s;
}

// `texel`: texel buffer bindings lowered with these element formats (the bound views'), else the
// SPIR-V image format.
StageOut translateStage(Device* d, const PipelineObj::StageSrc& src, const PipelineLayoutObj& pl,
                        uint32_t viewCount, const std::vector<spirv::VertexAttribute>& attrs,
                        const std::vector<spirv::VertexBufferBinding>& vbs, const std::string& label,
                        uint32_t viewMask = 0, const std::vector<PipelineObj::TexelBinding>* texel = nullptr,
                        uint16_t colorIntKinds = 0, int uiMaskSlot = -1, bool uiMaskAlpha = false, bool a2cTest = false,
                        uint32_t sampleMask = ~0u) {
  StageOut o;
  if (!src.module) {
    logf("vk.err what=shader pipeline=%s stage=%u error=\"no module\"", label.c_str(), unsigned(src.stage));
    o.why = "no_module";
    return o;
  }
  spirv::SpirvLowerOptions opt;
  opt.entry = src.entry;
  opt.stage = vpirStage(src.stage);
  opt.specConstants = src.spec;
  opt.viewCount = viewCount;
  opt.viewMask = viewMask;
  // MSAA images are then created 1x (vkCreateImage). Only a module with multisampled images translates differently,
  // so only then does the setting belong to the stage input (toggling it keeps every other stage's cache entry).
  opt.multisampleAsSingle = forceSingleSample() && (!src.words || spirvUsesMultisample(src.words->data(), src.words->size()));
  if (opt.stage == vpir::Stage::Fragment) {
    opt.colorIntKinds = colorIntKinds;
    opt.uiMaskSlot = int8_t(uiMaskSlot), opt.uiMaskAlpha = uiMaskAlpha;
    opt.alphaTest = a2cTest;
    opt.sampleMask = sampleMask;
  }
  // Foveation: multiview stages of exactly two views (rate map layer = view) carry the viewport index / the
  // FragCoord mapping, in every pass they may run in (outside a foveated pass the backend binds an identity map
  // and sets the same viewport for both views). Decided here, at creation: the setting is made before the game runs.
  if (foveationActive() && (opt.stage == vpir::Stage::Vertex || opt.stage == vpir::Stage::Fragment) &&
      (viewMask ? viewMask == 3u : viewCount == fov::kViews)) {
    if (opt.stage == vpir::Stage::Vertex) opt.viewportIndex = true;
    else opt.rateMapPosition = true;
  }
  for (const LayoutSlot& s : pl.slots) {
    spirv::SpirvBinding b;
    b.set = s.set, b.binding = s.binding, b.arraySize = s.count, b.kind = s.kind, b.slot = s.slot, b.samplerSlot = s.samplerSlot;
    if (s.ycbcr) {
      b.ycbcr = s.ycbcr->spec;
      b.ycbcr.planeSlot[0] = s.planeSlot[0], b.ycbcr.planeSlot[1] = s.planeSlot[1];
    }
    if (texel)
      for (const PipelineObj::TexelBinding& t : *texel)
        if (t.set == s.set && t.binding == s.binding) b.texelFormat = t.lowered;
    opt.layout.push_back(b);
  }
  opt.pushConstantSlot = pl.pushSlot;
  applyTranslatorMode(opt);  // process-wide modes: not part of the recipe input (the store's translator id carries them)
  opt.robustBuffers = robustBuffersEffective();  // part of the input: another value is another MSL under another key
  if (opt.stage == vpir::Stage::Vertex) {
    opt.vertexAttributes = attrs;
    opt.vertexBuffers = vbs;
  }
  // A stage the recipe store has already translated (this run or an earlier one) is not translated
  // again: the store hands back the shader id and the interface, and the MSL is registered with the
  // backend once per device. Translating 3000+ stages at ShaderWarmup froze the game thread for ~5 s.
  StageInput in;
  const bool recordable = d->recipes && src.words && !src.words->empty();
  if (recordable) {
    in.spirv = src.hash;
    in.opt = opt;
    o.input = in.hash();
    RecipeStore::StageHit hit;
    RecipeStore::MissInfo miss;
    const bool cacheOn = stageCacheOn();
    if (cacheOn && d->recipes->lookupStage(o.input, hit, &miss, src.hash)) {
      // Registered with the backend under the lock and marked after: a thread that sees the id as
      // added can rely on the backend already having it.
      bool have;
      {
        std::lock_guard<std::mutex> lock(d->shaderMutex);
        have = d->shadersAdded.count(hit.mslId) != 0;
        if (!have) {
          ShaderBlob blob;
          if (d->recipes->shaderBlob(hit.mslId, blob)) {
            d->platform->addShader(hit.mslId, std::move(blob));
            d->shadersAdded.insert(hit.mslId);
            have = true;
          }
        }
      }
      if (have) {
        o.id = hit.mslId;
        o.iface.bindings = std::move(hit.iface.bindings);
        o.iface.usesPushConstants = hit.iface.usesPushConstants;
        o.iface.colorOutputs = hit.iface.colorOutputs;
        o.iface.localSize = hit.iface.localSize;
        o.iface.amplified = hit.iface.amplified;
        o.varyingComponents = hit.iface.varyingComponents;
        o.msl.colorOutputs = hit.iface.mslColorOutputs;
        o.msl.dualSource = hit.iface.dualSource;
        noteShaderLib(hit.sourceHash);
        ++pipelineCounters().stagesCached;
        static std::atomic<uint32_t> nCached{0};
        if (const uint32_t n = ++nCached; n <= 4 || n % 256 == 0) logf("gpu.shader.msl cached=%u (recipe hit, not translated)", n);
        o.ok = true;
        return o;
      }
      // the blob vanished: translate and register again
      miss.why = RecipeStore::StageMiss::NoShader;
    }
    PipelineCounters& pc = pipelineCounters();
    using M = RecipeStore::StageMiss;
    switch (cacheOn ? miss.why : M::Disabled) {
      case M::NewContent: ++pc.missNewContent; break;
      case M::NewOptions: ++pc.missNewOptions; break;
      case M::NoShader: ++pc.missNoShader; break;
      case M::Version:  // one count per component that moved
        if (miss.components & kDiffLower) ++pc.missLower;
        if (miss.components & kDiffEmit) ++pc.missEmit;
        if (miss.components & kDiffDriver) ++pc.missDriver;
        if (miss.components & kDiffBaseline) ++pc.missBaseline;
        break;
      default: ++pc.missUncached; break;
    }
  } else {
    ++pipelineCounters().missUncached;
  }
  const auto t0 = std::chrono::steady_clock::now();
  vpir::Module mod;
  std::vector<std::string> errors;
  switch (lowerAndEmit(*src.module, opt, mod, o.iface, o.msl, o.id, errors)) {
    case TranslateFail::Lower:
      logf("vk.err what=shader pipeline=%s stage=%s error=\"%s\"", label.c_str(), vpir::toString(opt.stage), errors[0].c_str());
      o.why = std::string("lower:") + errors[0];
      return o;
    case TranslateFail::Msl: {
      std::lock_guard<std::mutex> lock(gTranslateLog);
      for (auto& x : errors) logf("vk.err what=msl pipeline=%s stage=%s error=\"%s\"", label.c_str(), vpir::toString(opt.stage), x.c_str());
      o.why = std::string("msl:") + (errors.empty() ? "" : errors[0]);
      return o;
    }
    case TranslateFail::None: break;
  }
  if (std::string dd; src.words && !src.words->empty() && dumpDirFor(o.id, dd)) {  // harness: its SPIR-V too
    char name[64];
    std::snprintf(name, sizeof name, "/%016llx.spv", (unsigned long long)o.id);
    if (FILE* f = std::fopen((dd + name).c_str(), "wb")) { std::fwrite(src.words->data(), 4, src.words->size(), f); std::fclose(f); }
  }
  const StageIface stored = ifaceOf(mod, o.iface, o.msl, opt.stage);
  o.varyingComponents = stored.varyingComponents;
  noteShaderLib(vpmsl::sourceHash(o.msl.source));  // gpu.shader.lib hit/miss vs the first-run index
  {
    static std::atomic<uint32_t> nKeys{0};
    if (nKeys.fetch_add(1) < 6)
      logf("gpu.shader.key stage=%s views=%u specs=%zu slots=%zu %s", vpir::toString(opt.stage), viewCount,
           opt.specConstants.size(), opt.layout.size(), vpmsl::describeKey(o.msl).c_str());
  }
  logPrivateStats(mod, o.id, opt.stage, label);
  {  // translation cost (SPIR-V -> VPIR -> MSL), to tell it apart from the Metal library compile
    static std::mutex m;
    static uint32_t n = 0;
    static double total = 0, worst = 0;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++pipelineCounters().stagesTranslated;
    pipelineCounters().translateUs += uint64_t(ms * 1000.0);
    std::lock_guard<std::mutex> lock(m);
    ++n, total += ms, worst = std::max(worst, ms);
    if (n <= 4 || n % 32 == 0) logf("gpu.shader.msl translated=%u ms_total=%.0f ms_max=%.1f", n, total, worst);
  }
  ShaderBlob blob = blobOf(o.msl);
  if (recordable)  // translator input, so a later translator can redo it
    d->recipes->addStage(in, o.input, *src.words, o.id, blob, &stored);
  {
    std::lock_guard<std::mutex> lock(d->shaderMutex);
    d->platform->addShader(o.id, std::move(blob));
    d->shadersAdded.insert(o.id);
  }
  o.ok = true;
  return o;
}

void collectUsed(PipelineObj& p, const StageOut& s, vpgpu::StageMask stage) {
  for (const spirv::SpirvBinding& b : s.iface.bindings) {
    if (b.set == spirv::kVertexBufferSet) continue;
    bool found = false;
    for (auto& u : p.used)
      if (u.slot.set == b.set && u.slot.binding == b.binding) {
        u.stages |= stage;
        u.written = u.written || b.written;
        found = true;
      }
    if (found) continue;
    for (const LayoutSlot& ls : p.layout->slots)
      if (ls.set == b.set && ls.binding == b.binding) {
        PipelineObj::Used u;
        u.slot = ls, u.stages = stage, u.written = b.written;
        p.used.push_back(u);
      }
  }
  if (s.iface.usesPushConstants) p.pushStages |= stage;
  // Texel buffer bindings and the element format each was lowered with.
  for (const spirv::SpirvBinding& b : s.iface.bindings) {
    if (b.set == spirv::kVertexBufferSet || b.texelFormat == vpgpu::Format::Undefined) continue;
    bool texelSlot = false;
    for (const LayoutSlot& ls : p.layout->slots)
      if (ls.set == b.set && ls.binding == b.binding)
        texelSlot = ls.type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || ls.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    if (!texelSlot) continue;
    bool have = false;
    for (const PipelineObj::TexelBinding& t : p.texel) have = have || (t.set == b.set && t.binding == b.binding);
    if (!have) p.texel.push_back({b.set, b.binding, b.texelFormat});
  }
}

// Attachment formats of the subpass (or of dynamic rendering).
// The mask is kept only when it is not the low popcount bits (SpirvLowerOptions::viewMask).
uint32_t irregularViewMask(uint32_t m) { return m && (m & (m + 1)) ? m : 0; }

void targetFormats(const VkGraphicsPipelineCreateInfo& ci, vpgpu::PipelineState& st, uint32_t& viewCount,
                   uint32_t& viewMask) {
  if (ci.renderPass) {
    const RenderPassObj* rp = from<RenderPassObj>(ci.renderPass);
    if (ci.subpass < rp->subpasses.size()) {
      const auto& sp = rp->subpasses[ci.subpass];
      for (size_t i = 0; i < sp.colors.size() && i < vpgpu::kMaxColorAttachments; ++i)
        if (sp.colors[i].attachment != VK_ATTACHMENT_UNUSED)
          st.colorFormats[i] = toCoreFormat(rp->attachments[sp.colors[i].attachment].format);
      if (sp.hasDepth && sp.depth.attachment != VK_ATTACHMENT_UNUSED)
        st.depthFormat = toCoreFormat(rp->attachments[sp.depth.attachment].format);
      viewCount = sp.viewMask ? uint32_t(std::popcount(sp.viewMask)) : 1;
      viewMask = irregularViewMask(sp.viewMask);
    }
    return;
  }
  for (auto* s = static_cast<const VkBaseInStructure*>(ci.pNext); s; s = s->pNext)
    if (s->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
      auto* r = reinterpret_cast<const VkPipelineRenderingCreateInfo*>(s);
      for (uint32_t i = 0; i < r->colorAttachmentCount && i < vpgpu::kMaxColorAttachments; ++i)
        st.colorFormats[i] = toCoreFormat(r->pColorAttachmentFormats[i]);
      VkFormat df = r->depthAttachmentFormat ? r->depthAttachmentFormat : r->stencilAttachmentFormat;
      st.depthFormat = toCoreFormat(df);
      viewCount = r->viewMask ? uint32_t(std::popcount(r->viewMask)) : 1;
      viewMask = irregularViewMask(r->viewMask);
    }
}

// Was this key among the recipes of earlier runs (or already requested on this device)? Checked before the game's request
// records it. The boot log's last "gpu.pso.recipe game" line is what AVP reads to trust (or drop) warm.stamp.
void noteRecipeKey(Device* d, vpgpu::PipelineKey key, const std::string& label) {
  const bool hit = d->gpu->pipelines().status(key) != vpgpu::PipelineCache::Status::Missing ||
                   (d->recipes && d->recipes->has(key.value));
  if (!hit) ++d->recipesFresh;  // vpvk_recipe_stats: also when nothing was stored (a first run misses everything)
  if (!d->prewarmedRecipes) return;
  const uint32_t h = hit ? ++d->recipeHits : d->recipeHits.load(), m = hit ? d->recipeMisses.load() : ++d->recipeMisses;
  if (!hit && m <= 8) logf("gpu.pso.recipe game miss key=%016llx pipeline=%s", (unsigned long long)key.value, label.c_str());
  if (h + m <= 4 || (h + m) % 64 == 0)
    logf("gpu.pso.recipe game hits=%u misses=%u prewarmed=%u", h, m, d->prewarmedRecipes);
}

// Would creating `key` compile nothing new: the core cache already has it (any status) or an earlier run's recipe does (its
// prewarm / the Metal binary archive serve it)? What VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT tests.
bool pipelineKnown(Device* d, vpgpu::PipelineKey key) {
  return d->gpu->pipelines().status(key) != vpgpu::PipelineCache::Status::Missing ||
         (d->recipes && d->recipes->has(key.value));
}

// The part of a pipeline's creation that translates and registers shaders: everything that needs the SPIR-V. Runs on
// the calling thread (legacy mode) or on the translate pool; writes the fields PipelineObj::resolved documents.
void finishGraphics(Device* d, PipelineObj* p, std::vector<PipelineObj::StageSrc> srcs,
                    std::vector<spirv::VertexAttribute> attrs, bool ok) {
  vpgpu::PipelineState& st = p->state;
  std::array<uint64_t, 3> inputs{};  // recipe stage inputs: vertex, fragment, compute
  std::string why = ok ? std::string() : "stage_unsupported";  // vk.pso.fail: the first reason this pipeline is not ok
  for (PipelineObj::StageSrc& src : srcs) {
    p->sources.push_back(std::move(src));
    const VkShaderStageFlagBits stage = p->sources.back().stage;
    StageOut o = translateStage(d, p->sources.back(), *p->layout, p->viewCount, attrs, p->vertexBuffers, st.label,
                                p->viewMask, nullptr, p->colorIntKinds, -1, false, p->a2cTest, p->sampleMask);
    if (!o.ok) {
      if (ok) why = o.why;
      ok = false;
      continue;
    }
    if (stage == VK_SHADER_STAGE_VERTEX_BIT) {
      st.vertexShader = o.id;
      inputs[0] = o.input;
      // Varyings beyond what Metal can carry between stages (the reported maxVertexOutputComponents is 128
      // with VPVK_REPORT_GEOM, which lets Unity pick its `target 5.0` subshaders): visible here instead of
      // silent garbage in the fragment stage.
      if (o.varyingComponents > kMetalMaxVaryingComponents) {
        static std::atomic<unsigned> k{0};
        if (k.fetch_add(1) < 8)
          logf("vk.err what=varyings count=%u max=%u reported=%u pipeline=%s", o.varyingComponents,
               kMetalMaxVaryingComponents, deviceProperties().limits.maxVertexOutputComponents,
               st.label.c_str());
      }
      collectUsed(*p, o, vpgpu::stage::vertex);
      // Vertex amplification: one instance per Vulkan instance (the views come from the amplification).
      p->amplified = o.iface.amplified;
      p->instanceViews = o.iface.amplified ? 1 : p->viewCount;
      if (p->viewCount > 1 && !o.iface.amplified) {  // derived here, not kept in the recipes: layer is what SpirvLowering refuses first, clip what is left
        const std::shared_ptr<spirv::SpirvModule>& mod = p->sources.back().module;
        p->ampWhy = !(translatorMode() & 2)            ? PipelineObj::AmpOff
                    : p->viewCount != 2 || p->viewMask ? PipelineObj::AmpMask
                    : mod && spirv::declaresLayer(*mod) ? PipelineObj::AmpLayer
                                                        : PipelineObj::AmpClip;
      }
    } else {
      st.fragmentShader = o.id;
      inputs[1] = o.input;
      collectUsed(*p, o, vpgpu::stage::fragment);
      // Outputs the shader does not write keep their contents (no blend, write mask none).
      for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c)
        if (!(o.msl.colorOutputs & (1u << c))) st.blend[c].writeMask = 0;
      p->fragmentOutputs = uint8_t(o.msl.colorOutputs & 0xFF);
      p->dualSource = o.msl.dualSource;
      degradeSrc1Blend(st, o.msl.dualSource);
    }
  }
  if (!st.vertexShader) {
    if (ok) why = "no_vertex_stage";
    ok = false;
  }
  if (p->rasterDiscard && !p->isDynamic(VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE)) discardRaster(st);
  p->ok = ok;
  p->key = st.key();
  if (!ok) noteTranslateFail(why, p->key.value, st.label);
  p->inputs = inputs;
  // Only texel buffer / stride variants and the VPKTAA UI-mask variants (maskPipelineKey) retranslate.
  if (p->texel.empty() && !p->dynStride && !taaUiMaskOn()) p->sources.clear();
  else p->attrs = std::move(attrs);
  if (ok && p->failOnCompile && !pipelineKnown(d, p->key)) {
    p->compileRequired = true;
    return;
  }
  if (ok) {
    noteRecipeKey(d, p->key, st.label);
    retainKey(d, *p, p->key);
    d->gpu->pipelines().request(st, p->key, vpgpu::Priority::Visible);
    if (d->recipes) d->recipes->addPipeline(st, inputs);
  }
}

// Runs `work` now, or on the translate pool while the pipeline is unresolved (see "compile policy").
void runCreation(Device* d, PipelineObj* p, std::function<void()> work) {
  ++pipelineCounters().pipelinesCreated;
  if (!d->translator) {
    work();
    return;
  }
  ++pipelineCounters().pipelinesAsync;
  p->resolved.store(false, std::memory_order_release);
  p->job = d->translator->submit([p, work = std::move(work)] {
    work();
    p->resolved.store(true, std::memory_order_release);
  });
}

PipelineObj* createGraphics(Device* d, const VkGraphicsPipelineCreateInfo& ci) {
  auto* p = new PipelineObj;
  p->failOnCompile = !d->translator && (ci.flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT);
  p->layout = layoutOf(ci.layout);
  if (!p->layout) p->layout = std::make_shared<PipelineLayoutObj>();
  vpgpu::PipelineState& st = p->state;
  targetFormats(ci, st, p->viewCount, p->viewMask);
  p->instanceViews = p->viewCount;
  static std::atomic<unsigned> counter{0};
  st.label = "vk.pipe" + std::to_string(counter++);

  // Vertex input.
  std::vector<spirv::VertexAttribute> attrs;
  if (const auto* vi = ci.pVertexInputState) {
    for (uint32_t i = 0; i < vi->vertexBindingDescriptionCount; ++i) {
      const auto& b = vi->pVertexBindingDescriptions[i];
      spirv::VertexBufferBinding vb;
      vb.binding = b.binding, vb.stride = b.stride, vb.perInstance = b.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
      for (auto* e = static_cast<const VkBaseInStructure*>(vi->pNext); e; e = e->pNext)  // VK_EXT_vertex_attribute_divisor
        if (e->sType == VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT) {
          const auto* dv = reinterpret_cast<const VkPipelineVertexInputDivisorStateCreateInfoEXT*>(e);
          for (uint32_t k = 0; k < dv->vertexBindingDivisorCount; ++k)
            if (dv->pVertexBindingDivisors[k].binding == b.binding) vb.divisor = dv->pVertexBindingDivisors[k].divisor;
        }
      vb.slot = uint16_t(p->layout->nextBuffer + b.binding);
      p->vertexBuffers.push_back(vb);
    }
    for (uint32_t i = 0; i < vi->vertexAttributeDescriptionCount; ++i) {
      const auto& a = vi->pVertexAttributeDescriptions[i];
      attrs.push_back(vertexAttribute(a));
    }
  }
  if (const auto* ia = ci.pInputAssemblyState) {
    p->topology = ia->topology;
    p->primitiveRestart = ia->primitiveRestartEnable;
    bool ok;
    st.primitive = primitive(ia->topology, ok);
    if (!ok && ia->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)  // fans: converted per draw (makeDraw)
      logf("vk.err what=topology value=%d pipeline=%s", int(ia->topology), st.label.c_str());
  }
  if (const auto* rs = ci.pRasterizationState) {
    st.cull = rs->cullMode == VK_CULL_MODE_FRONT_BIT ? vpgpu::CullMode::Front
              : rs->cullMode == VK_CULL_MODE_BACK_BIT ? vpgpu::CullMode::Back : vpgpu::CullMode::None;
    p->cullBoth = rs->cullMode == VK_CULL_MODE_FRONT_AND_BACK;
    p->cullMode = rs->cullMode;
    p->rasterDiscard = rs->rasterizerDiscardEnable;
    p->wireframe = rs->polygonMode == VK_POLYGON_MODE_LINE;
    p->pointMode = rs->polygonMode == VK_POLYGON_MODE_POINT;
    if (p->pointMode) st.primitive = primitiveOf(*p, p->topology, st.primitive);
    p->frontFace = rs->frontFace;
    p->depthBiasEnable = rs->depthBiasEnable;
    p->depthClamp = rs->depthClampEnable;  // Metal's clamp mode = no clipping
    for (auto* e = static_cast<const VkBaseInStructure*>(rs->pNext); e; e = e->pNext)  // VK_EXT_depth_clip_enable: clipping is its own switch
      if (e->sType == VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT)
        p->depthClamp = !reinterpret_cast<const VkPipelineRasterizationDepthClipStateCreateInfoEXT*>(e)->depthClipEnable;
    p->depthBias[0] = rs->depthBiasConstantFactor, p->depthBias[1] = rs->depthBiasClamp, p->depthBias[2] = rs->depthBiasSlopeFactor;
  }
  if (const auto* ms = ci.pMultisampleState) {
    // The game's count, then forcing to 1x / the override / the device clamp, the same rule as
    // vkCreateImage. A draw into a pass whose images were created with another count gets a
    // variant (pipelineFor), so a setting changed in between never mismatches.
    p->gameSamples = std::max<uint8_t>(1, static_cast<uint8_t>(ms->rasterizationSamples));
    st.samples = effectiveSamples(p->gameSamples);
    st.alphaToCoverage = ms->alphaToCoverageEnable;
    // VkSampleMask: Metal has no pipeline coverage mask; the fragment shader ANDs it into its [[sample_mask]] output.
    // Only for the game's own sample count (forced counts have other sample positions) and when it clears a sample.
    if (ms->pSampleMask && p->gameSamples > 1 && p->gameSamples <= 32 && st.samples == p->gameSamples) {
      const uint32_t all = p->gameSamples >= 32 ? ~0u : (1u << p->gameSamples) - 1u;
      if ((ms->pSampleMask[0] & all) != all) p->sampleMask = ms->pSampleMask[0] & all;
    }
    // Native identity: also a pipeline the game made 1x (Alyx: alpha-to-coverage on a single-sample pass): Apple
    // hardware A2C with one sample dithers, the fragment stage tests alpha >= 0.5 as the Adreno does.
    if (st.alphaToCoverage && (p->gameSamples > 1 || nativeExtras()) && st.samples <= 1) {
      // MSAA forced off: no coverage to modulate, the fragment stage tests alpha >= 0.5 instead.
      st.alphaToCoverage = false;
      p->a2cTest = true;
    }
  }
  if (const auto* ds = ci.pDepthStencilState; ds && st.depthFormat != vpgpu::Format::Undefined) {
    p->rawDepth = {bool(ds->depthTestEnable), bool(ds->depthWriteEnable), bool(ds->stencilTestEnable), ds->depthCompareOp};
    st.depthTest = ds->depthTestEnable;
    st.depthWrite = ds->depthTestEnable && ds->depthWriteEnable;
    st.depthCompare = ds->depthTestEnable ? static_cast<vpgpu::CompareOp>(ds->depthCompareOp) : vpgpu::CompareOp::Always;
    st.stencilTest = ds->stencilTestEnable;
    st.stencilFront = stencilFace(ds->front);
    st.stencilBack = stencilFace(ds->back);
    p->stencilRef[0] = ds->front.reference;
    p->stencilRef[1] = ds->back.reference;
  }
  if (const auto* cb = ci.pColorBlendState) {
    for (uint32_t i = 0; i < cb->attachmentCount && i < vpgpu::kMaxColorAttachments; ++i) {
      const auto& a = cb->pAttachments[i];
      vpgpu::BlendState& b = st.blend[i];
      b.enabled = a.blendEnable;
      b.srcColor = blendFactor(a.srcColorBlendFactor), b.dstColor = blendFactor(a.dstColorBlendFactor);
      b.srcAlpha = blendFactor(a.srcAlphaBlendFactor), b.dstAlpha = blendFactor(a.dstAlphaBlendFactor);
      b.colorOp = blendOp(a.colorBlendOp), b.alphaOp = blendOp(a.alphaBlendOp);
      b.writeMask = static_cast<uint8_t>(a.colorWriteMask & 0xF);
    }
    std::memcpy(p->blendConstants, cb->blendConstants, sizeof p->blendConstants);
  }
  if (const auto* vp = ci.pViewportState) {
    if (vp->pViewports && vp->viewportCount) p->hasViewport = true, p->viewport = vp->pViewports[0];
    if (vp->pScissors && vp->scissorCount) p->hasScissor = true, p->scissor = vp->pScissors[0];
  }
  if (const auto* dy = ci.pDynamicState) p->dynamic.assign(dy->pDynamicStates, dy->pDynamicStates + dy->dynamicStateCount);
  p->dynStride = p->isDynamic(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);

  // Integer colour attachments by signedness (fragment outputs take the attachment's type).
  for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c) {
    const vpgpu::FormatInfo& cf = vpgpu::formatInfo(st.colorFormats[c]);
    if (!cf.integer() || !cf.name) continue;
    p->colorIntKinds |= uint16_t(std::strstr(cf.name, "Sint") ? (1u << c) : (0x100u << c));
  }
  bool ok = true;
  std::vector<PipelineObj::StageSrc> srcs;  // read from the create info now: it does not outlive the call
  for (uint32_t i = 0; i < ci.stageCount; ++i) {
    const auto& s = ci.pStages[i];
    if (s.stage != VK_SHADER_STAGE_VERTEX_BIT && s.stage != VK_SHADER_STAGE_FRAGMENT_BIT) {
      logf("vk.err what=stage value=%u pipeline=%s (geometry/tessellation unsupported)", unsigned(s.stage), st.label.c_str());
      ok = false;
      continue;
    }
    srcs.push_back(stageSrc(s));
  }
  runCreation(d, p, [d, p, srcs = std::move(srcs), attrs = std::move(attrs), ok]() mutable {
    finishGraphics(d, p, std::move(srcs), std::move(attrs), ok);
  });
  return p;
}

PipelineObj* createCompute(Device* d, const VkComputePipelineCreateInfo& ci) {
  auto* p = new PipelineObj;
  p->compute = true;
  p->failOnCompile = !d->translator && (ci.flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT);
  p->layout = layoutOf(ci.layout);
  if (!p->layout) p->layout = std::make_shared<PipelineLayoutObj>();
  static std::atomic<unsigned> counter{0};
  p->state.label = "vk.comp" + std::to_string(counter++);
  runCreation(d, p, [d, p, src = stageSrc(ci.stage)]() mutable {
    p->sources.push_back(std::move(src));
    StageOut o = translateStage(d, p->sources.back(), *p->layout, 1, {}, {}, p->state.label);
    if (!o.ok) {
      noteTranslateFail(o.why, 0, p->state.label);
      return;
    }
    p->state.computeShader = o.id;
    collectUsed(*p, o, vpgpu::stage::compute);
    if (p->texel.empty()) p->sources.clear();
    p->ok = true;
    p->key = p->state.key();
    if (p->failOnCompile && !pipelineKnown(d, p->key)) {
      p->compileRequired = true;
      return;
    }
    noteRecipeKey(d, p->key, p->state.label);
    retainKey(d, *p, p->key);
    d->gpu->pipelines().request(p->state, p->key, vpgpu::Priority::Visible);
    if (d->recipes) d->recipes->addPipeline(p->state, {0, 0, o.input});
  });
  return p;
}

}  // namespace

// Recipe retranslation (recipes.cpp RecipeStore::refresh): quiet, no live-translation diagnostics.
uint64_t vpvk::stageTranslations() { return gStageTranslations.load(); }

bool vpvk::translateRecipeStage(const spirv::SpirvModule& m, const spirv::SpirvLowerOptions& opt, uint64_t& mslId,
                                ShaderBlob& blob, std::string& error, StageIface* iface) {
  vpir::Module mod;
  spirv::SpirvInterface ifc;
  vpmsl::MslShader msl;
  std::vector<std::string> errors;
  spirv::SpirvLowerOptions moded = opt;  // stored inputs do not carry the modes: this process's apply
  applyTranslatorMode(moded);
  if (lowerAndEmit(m, moded, mod, ifc, msl, mslId, errors) != TranslateFail::None) {
    error = errors.empty() ? "translate" : errors[0];
    return false;
  }
  blob = blobOf(msl);
  if (iface) *iface = ifaceOf(mod, ifc, msl, opt.stage);
  return true;
}

namespace {
// VK_EXT_pipeline_creation_feedback (core 1.3): the call's own duration for the pipeline (what a caller measures around
// vkCreate*Pipelines); stages carry no figure of their own (VALID with a zero duration). A pipeline that was not created
// (COMPILE_REQUIRED) reports nothing.
void fillCreationFeedback(const void* pNext, uint32_t stageCount, uint64_t ns, bool created) {
  for (auto* n = static_cast<const VkBaseInStructure*>(pNext); n; n = n->pNext) {
    if (n->sType != VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO) continue;
    auto* fb = reinterpret_cast<const VkPipelineCreationFeedbackCreateInfo*>(n);
    const VkPipelineCreationFeedbackFlags f = created ? VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT : 0;
    if (fb->pPipelineCreationFeedback) *fb->pPipelineCreationFeedback = {f, created ? ns : 0};
    for (uint32_t k = 0; k < fb->pipelineStageCreationFeedbackCount && fb->pPipelineStageCreationFeedbacks; ++k)
      fb->pPipelineStageCreationFeedbacks[k] = {k < stageCount ? f : 0, 0};
  }
}
uint64_t nsSince(std::chrono::steady_clock::time_point t) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count());
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice dv, VkPipelineCache, uint32_t n,
                                                                     const VkGraphicsPipelineCreateInfo* ci,
                                                                     const VkAllocationCallbacks*, VkPipeline* out) {
  vpgpu::HitchSpan hitch(vpgpu::hitchPerf().psoCreateUs);  // vk.hitch pso_create_ms
  Device* d = from<Device>(dv);
  std::vector<vpgpu::PipelineKey> keys;
  VkResult result = VK_SUCCESS;
  for (uint32_t i = 0; i < n; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    PipelineObj* p = createGraphics(d, ci[i]);
    fillCreationFeedback(ci[i].pNext, ci[i].stageCount, nsSince(t0), !p->compileRequired);
    if (p->compileRequired) {  // FAIL_ON_PIPELINE_COMPILE_REQUIRED: not requested; the others of the batch still are
      delete p;
      out[i] = VK_NULL_HANDLE;
      result = VK_PIPELINE_COMPILE_REQUIRED;
      if (ci[i].flags & VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT) {  // the rest of the batch is not created
        for (uint32_t j = i + 1; j < n; ++j) out[j] = VK_NULL_HANDLE;
        break;
      }
      continue;
    }
    out[i] = to<VkPipeline>(p);
    if (!d->translator && p->ok) keys.push_back(p->key);  // synchronous mode: translated already
  }
  static const uint32_t budget = waitBudgetMs("VPVK_PIPELINE_CREATE_WAIT_MS", 60000);
  pipelineCounters().createTimedOut += awaitPipelines(d, keys, budget, "create_graphics");  // see "compile policy"
  psoCensus(d, false);
  return result;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice dv, VkPipelineCache, uint32_t n,
                                                                    const VkComputePipelineCreateInfo* ci,
                                                                    const VkAllocationCallbacks*, VkPipeline* out) {
  vpgpu::HitchSpan hitch(vpgpu::hitchPerf().psoCreateUs);  // vk.hitch pso_create_ms
  Device* d = from<Device>(dv);
  std::vector<vpgpu::PipelineKey> keys;
  VkResult result = VK_SUCCESS;
  for (uint32_t i = 0; i < n; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    PipelineObj* p = createCompute(d, ci[i]);
    fillCreationFeedback(ci[i].pNext, 1, nsSince(t0), !p->compileRequired);
    if (p->compileRequired) {
      delete p;
      out[i] = VK_NULL_HANDLE;
      result = VK_PIPELINE_COMPILE_REQUIRED;
      if (ci[i].flags & VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT) {
        for (uint32_t j = i + 1; j < n; ++j) out[j] = VK_NULL_HANDLE;
        break;
      }
      continue;
    }
    out[i] = to<VkPipeline>(p);
    if (!d->translator && p->ok) keys.push_back(p->key);
  }
  static const uint32_t budget = waitBudgetMs("VPVK_PIPELINE_CREATE_WAIT_MS", 60000);
  pipelineCounters().createTimedOut += awaitPipelines(d, keys, budget, "create_compute");
  psoCensus(d, false);
  return result;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice dv, VkPipeline p, const VkAllocationCallbacks*) {
  auto* o = from<PipelineObj>(p);
  Device* d = from<Device>(dv);
  if (o && d && d->translator && o->job) {
    d->translator->boost(o->job);
    d->translator->wait(o->job, UINT32_MAX);  // the job writes into `o` (and holds its pointer)
  }
  if (o && d && d->gpu)
    for (uint64_t k : o->ownedKeys) d->gpu->pipelines().release({k});
  delete o;
}

// Pipeline caches: the core keeps its own (compiled pipelines + binary archive + recipe store).
// The Vulkan object is a key list: vkGetPipelineCacheData returns a VkPipelineCacheHeaderVersionOne
// header (our vendor/device/UUID) followed by "VQPC", a count and the recipe keys, so a game that
// saves and reloads it (Unity) hands the keys back; vkCreatePipelineCache then queues those recipes.
namespace {
constexpr uint32_t kCacheMagic = 0x43505156;  // "VQPC"
struct PipelineCacheObj {
  Device* device = nullptr;
};
void cacheHeader(uint8_t (&h)[32]) {
  std::memset(h, 0, sizeof h);
  uint32_t len = 32, ver = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
  const VkPhysicalDeviceProperties& p = deviceProperties();
  std::memcpy(h, &len, 4);
  std::memcpy(h + 4, &ver, 4);
  std::memcpy(h + 8, &p.vendorID, 4);
  std::memcpy(h + 12, &p.deviceID, 4);
  std::memcpy(h + 16, p.pipelineCacheUUID, 16);
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(VkDevice dv, const VkPipelineCacheCreateInfo* ci,
                                                                 const VkAllocationCallbacks*, VkPipelineCache* out) {
  Device* d = from<Device>(dv);
  auto* c = new PipelineCacheObj;
  c->device = d;
  *out = reinterpret_cast<VkPipelineCache>(c);
  if (!ci || !ci->initialDataSize || !ci->pInitialData) return VK_SUCCESS;
  const auto* b = static_cast<const uint8_t*>(ci->pInitialData);
  const size_t n = ci->initialDataSize;
  uint8_t h[32];
  cacheHeader(h);
  uint32_t magic = 0, count = 0;
  if (n < 40 || std::memcmp(b, h, 32)) {
    logf("vk.pipelinecache initial size=%zu foreign=1 (ignored)", n);
    return VK_SUCCESS;
  }
  std::memcpy(&magic, b + 32, 4);
  std::memcpy(&count, b + 36, 4);
  if (magic != kCacheMagic || n < 40 + size_t(count) * 8) {
    logf("vk.pipelinecache initial size=%zu foreign=1 (ignored)", n);
    return VK_SUCCESS;
  }
  // The keys name recipes of the device-wide store; they are compiled ahead (a prewarm), so not
  // when prewarm is off (VPVK_PREWARM=0: the conformance run). A recipe is queued only with the
  // MSL of all its stages registered first — a pipeline whose shader is unknown cannot be built
  // (CTS log 60: thousands queued without their shaders aborted the process in Metal).
  if (!prewarmEnabled()) {
    logf("vk.pipelinecache initial size=%zu keys=%u skipped=prewarm_off", n, count);
    return VK_SUCCESS;
  }
  uint32_t queued = 0, known = 0, noShader = 0;
  std::vector<std::pair<uint64_t, ShaderBlob>> shaders;
  std::unordered_set<uint64_t> registered;  // each shader once per call
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t k;
    std::memcpy(&k, b + 40 + size_t(i) * 8, 8);
    vpgpu::PipelineState s;
    if (!d->recipes) break;
    if (!d->recipes->stateFor(k, s)) continue;
    ++known;
    if (d->gpu->pipelines().status({k}) != vpgpu::PipelineCache::Status::Missing) continue;
    if (!d->recipes->stateFor(k, s, &shaders)) {  // the MSL only for what will be queued
      ++noShader;
      continue;
    }
    for (auto& [id, blob] : shaders)
      if (registered.insert(id).second) d->platform->addShader(id, std::move(blob));
    d->gpu->pipelines().request(s, {k}, vpgpu::Priority::Prefetch);
    ++queued;
  }
  if (noShader) logf("vk.pipelinecache recipes_without_shaders=%u (not queued)", noShader);
  logf("vk.pipelinecache initial size=%zu keys=%u known=%u queued=%u", n, count, known, queued);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice, VkPipelineCache c, const VkAllocationCallbacks*) {
  delete reinterpret_cast<PipelineCacheObj*>(c);
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice dv, VkPipelineCache, size_t* size, void* data) {
  Device* d = from<Device>(dv);
  std::vector<uint64_t> keys;
  if (d && d->recipes) keys = d->recipes->keys();
  const size_t total = 40 + keys.size() * 8;
  if (!data) {
    *size = total;
    return VK_SUCCESS;
  }
  uint8_t h[32];
  cacheHeader(h);
  if (*size < total) {
    // Spec: write as much as fits (whole keys only); 0 if not even the header fits.
    if (*size < 40) {
      *size = 0;
      return VK_INCOMPLETE;
    }
    keys.resize((*size - 40) / 8);
  }
  auto* o = static_cast<uint8_t*>(data);
  const uint32_t magic = kCacheMagic, count = uint32_t(keys.size());
  std::memcpy(o, h, 32);
  std::memcpy(o + 32, &magic, 4);
  std::memcpy(o + 36, &count, 4);
  if (count) std::memcpy(o + 40, keys.data(), keys.size() * 8);
  const bool partial = 40 + keys.size() * 8 < total;
  *size = 40 + keys.size() * 8;
  logf("vk.pipelinecache data size=%zu keys=%u%s", *size, count, partial ? " incomplete=1" : "");
  return partial ? VK_INCOMPLETE : VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMergePipelineCaches(VkDevice, VkPipelineCache, uint32_t, const VkPipelineCache*) {
  return VK_SUCCESS;  // one device-wide recipe store: every cache object already sees every key
}

// =================================================================================================
// Binding commands
// =================================================================================================
namespace {
Recorder& rec(VkCommandBuffer cb) { return *from<CommandBuffer>(cb)->rec; }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
  BoundState& b = rec(cb).bound;
  const PipelineObj* po = from<PipelineObj>(p);
  if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) {
    b.compute = po;
    return;
  }
  b.graphics = po;
  // Static state comes with the pipeline; dynamic state keeps what was set.
  // The "with count" forms (extended dynamic state, core 1.3) make the pipeline's rectangles dynamic too.
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_VIEWPORT) && !po->isDynamic(VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT) && po->hasViewport)
    b.hasViewport = true, b.viewport = po->viewport;
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_SCISSOR) && !po->isDynamic(VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT) && po->hasScissor)
    b.hasScissor = true, b.scissor = po->scissor;
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_STENCIL_REFERENCE)) b.stencilRef[0] = po->stencilRef[0], b.stencilRef[1] = po->stencilRef[1];
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK)) b.stencilCompareMask[0] = b.stencilCompareMask[1] = -1;
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_STENCIL_WRITE_MASK)) b.stencilWriteMask[0] = b.stencilWriteMask[1] = -1;
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_BLEND_CONSTANTS)) std::memcpy(b.blend, po->blendConstants, sizeof b.blend);
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_DEPTH_BIAS)) std::memcpy(b.depthBias, po->depthBias, sizeof b.depthBias);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout,
                                                               uint32_t first, uint32_t n, const VkDescriptorSet* sets,
                                                               uint32_t nd, const uint32_t* dyn) {
  BoundState& b = rec(cb).bound;
  auto& slots = bp == VK_PIPELINE_BIND_POINT_COMPUTE ? b.computeSets : b.graphicsSets;
  uint32_t used = 0;
  for (uint32_t i = 0; i < n && first + i < slots.size(); ++i) {
    auto& s = slots[first + i];
    s.set = sets[i] ? from<DescriptorSetObj>(sets[i]) : nullptr;
    s.dynamicOffsets.clear();
    if (!s.set) continue;
    // Dynamic offsets are consumed in binding order of the dynamic descriptors.
    for (const SetLayoutBinding& lb : s.set->layout->bindings)
      if (lb.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || lb.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
        for (uint32_t k = 0; k < lb.count; ++k) s.dynamicOffsets.push_back(used < nd ? dyn[used++] : 0);
  }
}

namespace {
// The push set of (bind point, set number), made (or reused) for the layout's set layout and bound in place of whatever
// was bound there. Null when the layout has no such set.
DescriptorSetObj* pushSetFor(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout pl, uint32_t set) {
  const std::shared_ptr<const PipelineLayoutObj> lay = layoutOf(pl);
  Recorder& r = rec(cb);
  if (!lay || set >= lay->sets.size() || !lay->sets[set] || set >= r.pushSets[0].size()) return nullptr;
  const bool compute = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
  auto& own = r.pushSets[compute ? 1 : 0][set];
  if (!own || own->layout != lay->sets[set]) {
    own = std::make_unique<DescriptorSetObj>();
    initDescriptorSet(*own, lay->sets[set]);
  }
  auto& bound = (compute ? r.bound.computeSets : r.bound.graphicsSets)[set];
  bound.set = own.get();
  bound.dynamicOffsets.clear();
  return own.get();
}
}  // namespace

// VK_KHR_push_descriptor: the writes go to a set the command buffer owns; draws copy descriptors when recorded,
// so rewriting it in place for the next push is safe.
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout pl,
                                                                 uint32_t set, uint32_t n, const VkWriteDescriptorSet* w) {
  DescriptorSetObj* s = pushSetFor(cb, bp, pl, set);
  if (!s) return;
  for (uint32_t i = 0; i < n; ++i) applyWrite(s, w[i]);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer cb, VkDescriptorUpdateTemplate tpl,
                                                                             VkPipelineLayout pl, uint32_t set, const void* data) {
  const auto* t = from<DescriptorUpdateTemplateObj>(tpl);
  DescriptorSetObj* s = pushSetFor(cb, t->bindPoint, pl, set);
  if (s) applyTemplate(s, tpl, data);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout, VkShaderStageFlags,
                                                          uint32_t offset, uint32_t size, const void* data) {
  BoundState& b = rec(cb).bound;
  if (offset + size <= b.push.size()) std::memcpy(b.push.data() + offset, data, size);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n,
                                                              const VkBuffer* bufs, const VkDeviceSize* offs) {
  BoundState& b = rec(cb).bound;
  for (uint32_t i = 0; i < n && first + i < b.vertex.size(); ++i)
    b.vertex[first + i] = {bufs[i] ? from<Buffer>(bufs[i]) : nullptr, offs[i]};
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n,
                                                               const VkBuffer* bufs, const VkDeviceSize* offs,
                                                               const VkDeviceSize* sizes, const VkDeviceSize* strides) {
  BoundState& b = rec(cb).bound;
  for (uint32_t i = 0; i < n && first + i < b.vertex.size(); ++i)
    b.vertex[first + i] = {bufs[i] ? from<Buffer>(bufs[i]) : nullptr, offs[i], sizes ? sizes[i] : VK_WHOLE_SIZE,
                           strides ? int64_t(strides[i]) : -1};
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, VkIndexType t) {
  BoundState& b = rec(cb).bound;
  b.index = buf ? from<Buffer>(buf) : nullptr;
  b.indexOffset = off;
  b.indexWide = t == VK_INDEX_TYPE_UINT32;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport* v) {
  if (first == 0 && n) {
    BoundState& b = rec(cb).bound;
    b.hasViewport = true;
    b.viewport = v[0];
  }
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport* v) {
  // maxViewports = 1: el contador no puede pasar de 1 (VUID de la spec); se usa el primero y se avisa.
  if (n > 1) {
    static std::atomic<unsigned> k{0};
    if (k.fetch_add(1) < 4) logf("vk.err what=viewport_count count=%u max=1", n);
  }
  vkCmdSetViewport(cb, 0, std::min(n, 1u), v);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D* r) {
  if (first == 0 && n) {
    BoundState& b = rec(cb).bound;
    b.hasScissor = true;
    b.scissor = r[0];
  }
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetScissorWithCount(VkCommandBuffer cb, uint32_t n, const VkRect2D* r) {
  if (n > 1) {
    static std::atomic<unsigned> k{0};
    if (k.fetch_add(1) < 4) logf("vk.err what=scissor_count count=%u max=1", n);
  }
  vkCmdSetScissor(cb, 0, std::min(n, 1u), r);
}

// VK_EXT_extended_dynamic_state / VK_EXT_extended_dynamic_state2 (core in Vulkan 1.3). A draw applies
// the value only when its pipeline lists the state as dynamic (makeDraw, pipelineFor).
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetCullMode(VkCommandBuffer cb, VkCullModeFlags m) {
  rec(cb).bound.eds.cullMode = int32_t(m);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetFrontFace(VkCommandBuffer cb, VkFrontFace f) {
  rec(cb).bound.eds.frontFace = int32_t(f);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetPrimitiveTopology(VkCommandBuffer cb, VkPrimitiveTopology t) {
  rec(cb).bound.eds.topology = int32_t(t);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthTestEnable(VkCommandBuffer cb, VkBool32 e) {
  rec(cb).bound.eds.depthTest = e ? 1 : 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthWriteEnable(VkCommandBuffer cb, VkBool32 e) {
  rec(cb).bound.eds.depthWrite = e ? 1 : 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthCompareOp(VkCommandBuffer cb, VkCompareOp op) {
  rec(cb).bound.eds.depthCompare = int32_t(op);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBoundsTestEnable(VkCommandBuffer, VkBool32) {}  // depthBounds = 0
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilTestEnable(VkCommandBuffer cb, VkBool32 e) {
  rec(cb).bound.eds.stencilTest = e ? 1 : 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilOp(VkCommandBuffer cb, VkStencilFaceFlags faces, VkStencilOp fail,
                                                         VkStencilOp pass, VkStencilOp depthFail, VkCompareOp compare) {
  BoundState::Eds& e = rec(cb).bound.eds;
  for (int f = 0; f < 2; ++f) {
    if (!(faces & (f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT))) continue;
    e.stencilOp[f] = true, e.fail[f] = fail, e.pass[f] = pass, e.depthFail[f] = depthFail, e.compare[f] = compare;
  }
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetRasterizerDiscardEnable(VkCommandBuffer cb, VkBool32 e) {
  rec(cb).bound.eds.rasterDiscard = e ? 1 : 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBiasEnable(VkCommandBuffer cb, VkBool32 e) {
  rec(cb).bound.eds.depthBias = e ? 1 : 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetPrimitiveRestartEnable(VkCommandBuffer cb, VkBool32 e) {
  rec(cb).bound.eds.restart = e ? 1 : 0;
}
// VK_EXT_extended_dynamic_state3: only the colour write mask is offered (Alyx declares it on its pipelines).
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetColorWriteMaskEXT(VkCommandBuffer cb, uint32_t first, uint32_t count,
                                                                  const VkColorComponentFlags* masks) {
  BoundState::Eds& e = rec(cb).bound.eds;
  for (uint32_t i = 0; i < count && first + i < uint32_t(vpgpu::kMaxColorAttachments); ++i)
    e.colorWrite[first + i] = int8_t(masks[i] & 0xF);
}
// Not offered by the device (extendedDynamicState2LogicOp / PatchControlPoints are false): Metal has
// no logic ops and the driver no tessellation. Accepted so a loader table is never left with holes.
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetLogicOpEXT(VkCommandBuffer, VkLogicOp) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetPatchControlPointsEXT(VkCommandBuffer, uint32_t) {}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilReference(VkCommandBuffer cb, VkStencilFaceFlags faces, uint32_t ref) {
  BoundState& b = rec(cb).bound;
  if (faces & VK_STENCIL_FACE_FRONT_BIT) b.stencilRef[0] = ref;
  if (faces & VK_STENCIL_FACE_BACK_BIT) b.stencilRef[1] = ref;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) {
  std::memcpy(rec(cb).bound.blend, c, sizeof(float) * 4);
}
// Baked into the pipeline by the core (masks, bias) or not expressible: accepted and ignored.
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetLineWidth(VkCommandBuffer, float) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBias(VkCommandBuffer cb, float constant, float clamp, float slope) {
  float* d = rec(cb).bound.depthBias;
  d[0] = constant, d[1] = clamp, d[2] = slope;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBounds(VkCommandBuffer, float, float) {}
// Dynamic stencil masks: the backend builds a depth-stencil state with them (Draw::stencil*Mask).
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilCompareMask(VkCommandBuffer cb, VkStencilFaceFlags faces, uint32_t m) {
  BoundState& b = rec(cb).bound;
  if (faces & VK_STENCIL_FACE_FRONT_BIT) b.stencilCompareMask[0] = int32_t(m & 0xFF);
  if (faces & VK_STENCIL_FACE_BACK_BIT) b.stencilCompareMask[1] = int32_t(m & 0xFF);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilWriteMask(VkCommandBuffer cb, VkStencilFaceFlags faces, uint32_t m) {
  BoundState& b = rec(cb).bound;
  if (faces & VK_STENCIL_FACE_FRONT_BIT) b.stencilWriteMask[0] = int32_t(m & 0xFF);
  if (faces & VK_STENCIL_FACE_BACK_BIT) b.stencilWriteMask[1] = int32_t(m & 0xFF);
}

// =================================================================================================
// Draws and dispatches
// =================================================================================================
namespace vpvk {
void logSkip(const char* why, const PipelineObj* p, const char* fmt, ...) {
  // 32 lines per reason (a frequent empty_scissor must not hide the others); `why` is always a string literal.
  struct Slot {
    std::atomic<const char*> why{nullptr};
    std::atomic<unsigned> n{0};
  };
  static Slot slots[24];
  static std::atomic<unsigned> overflow{0};  // more reasons than slots: they share one cap
  std::atomic<unsigned>* count = &overflow;
  for (Slot& sl : slots) {
    const char* cur = sl.why.load();
    if (!cur) {
      const char* none = nullptr;
      cur = sl.why.compare_exchange_strong(none, why) ? why : none;  // lost the race: the winner's reason
    }
    if (std::strcmp(cur, why) == 0) {
      count = &sl.n;
      break;
    }
  }
  if (count->fetch_add(1) >= 32) return;
  char detail[256] = "";
  if (fmt) {
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(detail, sizeof detail, fmt, args);
    va_end(args);
  }
  logf("vk.skip what=%s pipeline=%s%s%s", why, p ? p->state.label.c_str() : "none", *detail ? " " : "", detail);
}

namespace {

bool resolveBindings(Recorder& r, const PipelineObj& p, const std::array<BoundState::Set, 8>& sets,
                     std::vector<vpgpu::Access>& out, std::vector<vpgpu::ViewBinding>& views,
                     std::vector<vpgpu::TextureId>& zeroReads) {
  for (const PipelineObj::Used& u : p.used) {
    const LayoutSlot& ls = u.slot;
    if (ls.set >= sets.size() || !sets[ls.set].set) return false;
    const DescriptorSetObj* set = sets[ls.set].set;
    size_t bi = SIZE_MAX;
    uint32_t dynBase = 0;
    for (size_t k = 0; k < set->layout->bindings.size(); ++k) {
      const SetLayoutBinding& lb = set->layout->bindings[k];
      if (lb.binding == ls.binding) {
        bi = k;
        break;
      }
      if (lb.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || lb.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
        dynBase += lb.count;
    }
    if (bi == SIZE_MAX) return false;
    const auto& elems = set->slots[bi];
    for (uint32_t e = 0; e < ls.count && e < elems.size(); ++e) {
      const Descriptor& d = elems[e];
      switch (ls.type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
          if (!d.buffer || !d.buffer->id) return false;
          VkDeviceSize off = d.offset, range = d.range;
          bool dynamic = ls.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || ls.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
          if (dynamic && dynBase + e < sets[ls.set].dynamicOffsets.size()) off += sets[ls.set].dynamicOffsets[dynBase + e];
          if (off >= d.buffer->size) return false;
          range = std::min<VkDeviceSize>(range, d.buffer->size - off);
          vpgpu::Access a{vpgpu::ResourceRange::bytes(d.buffer->id, off, range), u.stages, u.written};
          a.kind = ls.kind, a.slot = uint16_t(ls.slot + e);
          out.push_back(a);
          break;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
          if (!d.texel || !d.texel->buffer || !d.texel->buffer->id) return false;
          VkDeviceSize range = d.texel->range == VK_WHOLE_SIZE ? d.texel->buffer->size - d.texel->offset : d.texel->range;
          vpgpu::Access a{vpgpu::ResourceRange::bytes(d.texel->buffer->id, d.texel->offset, range), u.stages, u.written};
          a.kind = vpgpu::BindKind::Buffer, a.slot = uint16_t(ls.slot + e);
          out.push_back(a);
          break;
        }
        case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
          // The block's bytes are copied into the command buffer's upload memory now: later
          // vkUpdateDescriptorSets calls do not reach a draw already recorded.
          const std::vector<uint8_t>& bytes = set->inlineData[bi];
          std::vector<uint8_t> padded(bytes);
          padded.resize(std::max<size_t>(16, (bytes.size() + 15) & ~size_t(15)), 0);
          VkDeviceSize off;
          vpgpu::BufferId id = r.arena.put(r.device, padded.data(), padded.size(), off);
          vpgpu::Access a{vpgpu::ResourceRange::bytes(id, off, padded.size()), u.stages, false};
          a.kind = vpgpu::BindKind::ConstantBuffer, a.slot = uint16_t(ls.slot);
          out.push_back(a);
          break;
        }
        case VK_DESCRIPTOR_TYPE_SAMPLER: {
          if (!d.sampler || !d.sampler->id) return false;
          out.push_back(vpgpu::samplerBinding(d.sampler->id, u.stages, uint16_t(ls.slot + e)));
          break;
        }
        default: {  // images
          if (!d.view || !d.view->image || !d.view->image->id) return false;
          vpgpu::Access a{vpgpu::ResourceRange::wholeTexture(d.view->image->id), u.stages, u.written};
          a.kind = ls.kind, a.slot = uint16_t(ls.slot + e);
          if (ls.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && d.view->storageView)
            views.push_back({ls.kind, a.slot, d.view->storageView});  // cube storage: texture2d_array
          else if (d.view->view) views.push_back({ls.kind, a.slot, d.view->view});
          out.push_back(a);
          if (zeroPendingImage(d.view->image)) zeroReads.push_back(d.view->image->id);
          if (ls.ycbcr) {  // planes 1 and 2 of the multi-planar image
            const unsigned others = ls.ycbcr->spec.planes - 1u;
            if (d.view->planeViews.size() < others) return false;
            for (unsigned k = 0; k < others; ++k) {
              const ImageView* pv = d.view->planeViews[k];
              if (!pv->image || !pv->image->id) return false;
              vpgpu::Access pa{vpgpu::ResourceRange::wholeTexture(pv->image->id), u.stages, false};
              pa.kind = vpgpu::BindKind::Texture, pa.slot = uint16_t(ls.planeSlot[k] + e);
              if (pv->view) views.push_back({pa.kind, pa.slot, pv->view});
              out.push_back(pa);
              if (zeroPendingImage(pv->image)) zeroReads.push_back(pv->image->id);
            }
          }
          if (ls.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            if (!d.sampler || !d.sampler->id) return false;
            out.push_back(vpgpu::samplerBinding(d.sampler->id, u.stages, uint16_t(ls.samplerSlot + e)));
          }
        }
      }
    }
  }
  if (p.pushStages && p.layout->pushSlot >= 0) {
    VkDeviceSize off;
    vpgpu::BufferId id = r.arena.put(r.device, r.bound.push.data(), std::max<uint32_t>(16, p.layout->pushSize), off);
    vpgpu::Access a{vpgpu::ResourceRange::bytes(id, off, std::max<uint32_t>(16, p.layout->pushSize)), p.pushStages, false};
    a.kind = vpgpu::BindKind::ConstantBuffer, a.slot = uint16_t(p.layout->pushSlot);
    out.push_back(a);
  }
  return true;
}

// Texel buffer variant for the views bound now (PipelineObj::texel): null when every binding's
// view has the format the pipeline was lowered with (or none is bound yet: resolveBindings
// rejects the draw then). Element 0 of an arrayed binding decides for the whole array.
PipelineObj::TexelVariant* texelVariant(Device* dev, const PipelineObj& p, const std::array<BoundState::Set, 8>& sets,
                                        const BoundState& bound) {
  if (!dev || (p.texel.empty() && !p.dynStride) || p.sources.empty()) return nullptr;
  std::vector<vpgpu::Format> want(p.texel.size());
  bool differs = false;
  for (size_t i = 0; i < p.texel.size(); ++i) {
    const PipelineObj::TexelBinding& t = p.texel[i];
    want[i] = t.lowered;
    if (t.set >= sets.size() || !sets[t.set].set) continue;
    const DescriptorSetObj* set = sets[t.set].set;
    for (size_t k = 0; k < set->layout->bindings.size(); ++k)
      if (set->layout->bindings[k].binding == t.binding && !set->slots[k].empty() && set->slots[k][0].texel) {
        const vpgpu::Format f = toCoreFormat(set->slots[k][0].texel->format);
        if (f != vpgpu::Format::Undefined) want[i] = f;
        break;
      }
    differs = differs || want[i] != t.lowered;
  }
  // Vertex strides from vkCmdBindVertexBuffers2 (VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE): the
  // vertex fetch is in the shader, so the stride is part of the translation.
  std::vector<uint32_t> strides;
  for (const spirv::VertexBufferBinding& vb : p.vertexBuffers) {
    const int64_t s = p.dynStride ? bound.vertex[vb.binding < bound.vertex.size() ? vb.binding : 0].stride : -1;
    strides.push_back(s >= 0 ? uint32_t(s) : vb.stride);
    differs = differs || strides.back() != vb.stride;
  }
  if (!differs) return nullptr;
  std::unique_lock<std::mutex> lock(p.variantMutex);
  for (PipelineObj::TexelVariant& v : p.texelVariants)
    if (v.formats == want && v.strides == strides) return v.ok ? &v : nullptr;
  PipelineObj::TexelVariant& v = p.texelVariants.emplace_back();
  v.formats = want;
  v.strides = strides;
  v.state = p.state;
  std::vector<PipelineObj::TexelBinding> tb = p.texel;
  std::string fmts;
  for (size_t i = 0; i < tb.size(); ++i) {
    tb[i].lowered = want[i];
    fmts += std::string(fmts.empty() ? "" : ",") + vpgpu::formatInfo(want[i]).name;
  }
  std::vector<spirv::VertexBufferBinding> vbs = p.vertexBuffers;
  std::string strs;
  for (size_t i = 0; i < vbs.size(); ++i) {
    vbs[i].stride = strides[i];
    if (strides[i] != p.vertexBuffers[i].stride) strs += std::string(strs.empty() ? "" : ",") + std::to_string(strides[i]);
  }
  v.state.label += (strs.empty() ? ".tb" : ".st") + std::to_string(p.texelVariants.size());
  v.ok = true;
  for (const PipelineObj::StageSrc& src : p.sources) {
    StageOut o = translateStage(dev, src, *p.layout, p.viewCount, p.attrs, vbs, v.state.label, p.viewMask, &tb,
                                p.colorIntKinds, -1, false, p.a2cTest, p.sampleMask);
    if (!o.ok) {
      v.ok = false;
      break;
    }
    switch (src.stage) {
      case VK_SHADER_STAGE_VERTEX_BIT: v.state.vertexShader = o.id, v.inputs[0] = o.input; break;
      case VK_SHADER_STAGE_FRAGMENT_BIT: v.state.fragmentShader = o.id, v.inputs[1] = o.input; break;
      default: v.state.computeShader = o.id, v.inputs[2] = o.input; break;
    }
  }
  static std::atomic<unsigned> lines{0};
  if (lines.fetch_add(1) < 16)
    logf("vk.texel_variant pipeline=%s formats=%s strides=%s ok=%d", v.state.label.c_str(), fmts.c_str(), strs.c_str(),
         int(v.ok));
  if (!v.ok) return nullptr;  // the base pipeline is used (and the failure logged by translateStage)
  v.key = v.state.key();
  retainKey(dev, p, v.key);
  dev->gpu->pipelines().request(v.state, v.key, vpgpu::Priority::Blocking);
  if (dev->recipes) dev->recipes->addPipeline(v.state, v.inputs);
  const vpgpu::PipelineKey fresh = v.key;
  lock.unlock();  // (texelVariants is a deque-like list of stable elements: &v stays valid)
  awaitPipeline(dev, fresh, "texel_variant");
  return &v;
}

// Does colour 0's alpha say how much of the pixel the draw covers? Alpha blending (straight or
// premultiplied); additive and opaque draws mark the whole pixel.
bool uiMaskUsesAlpha(const vpgpu::BlendState& b) {
  return b.enabled && (b.srcColor == vpgpu::BlendFactor::SrcAlpha || b.dstColor == vpgpu::BlendFactor::OneMinusSrcAlpha ||
                       b.srcAlpha == vpgpu::BlendFactor::SrcAlpha);
}

}  // namespace

bool maskPipelineKey(Device* dev, const PipelineObj& p, PipelineObj::TexelVariant* tv, uint8_t passSamples, bool ui,
                     vpgpu::PipelineKey& out) {
  if (!dev || p.compute || !p.ok) return false;
  const vpgpu::PipelineState& base = tv ? tv->state : p.state;
  const uint8_t samples = passSamples ? passSamples : base.samples;
  std::unique_lock<std::mutex> lock(p.variantMutex);
  for (const PipelineObj::MaskVariant& v : p.maskVariants)
    if (v.tv == tv && v.samples == samples && v.ui == ui) {
      out = v.key;
      return v.ok;
    }
  constexpr int slot = taa::kUiMaskSlot;
  PipelineObj::MaskVariant mv;
  mv.tv = tv, mv.samples = samples, mv.ui = ui;
  vpgpu::PipelineState st = base;
  st.samples = samples;
  std::array<uint64_t, 3> inputs = tv ? tv->inputs : p.inputs;
  bool translated = true;
  if (ui) {
    // The fragment stage again, writing the mask (colour 0's alpha when the draw blends by alpha).
    std::vector<PipelineObj::TexelBinding> tb = p.texel;
    if (tv)
      for (size_t i = 0; i < tb.size() && i < tv->formats.size(); ++i) tb[i].lowered = tv->formats[i];
    translated = false;
    for (const PipelineObj::StageSrc& src : p.sources) {
      if (src.stage != VK_SHADER_STAGE_FRAGMENT_BIT) continue;
      StageOut o = translateStage(dev, src, *p.layout, p.viewCount, p.attrs, p.vertexBuffers, st.label + ".um", p.viewMask,
                                  tv ? &tb : nullptr, p.colorIntKinds, slot, uiMaskUsesAlpha(base.blend[0]), p.a2cTest, p.sampleMask);
      if (o.ok) st.fragmentShader = o.id, inputs[1] = o.input, translated = true;
      break;
    }
    if (!translated) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 8)
        logf("vk.taa uimask variant_failed pipeline=%s (sources=%zu): the draw does not mark the mask", st.label.c_str(),
             p.sources.size());
    }
  }
  const bool writes = ui && translated;
  st.colorFormats[slot] = vpgpu::Format::R8Unorm;
  vpgpu::BlendState& b = st.blend[slot];
  b = vpgpu::BlendState{};
  if (writes) {  // coverage accumulates: the largest value over the draws that touch the pixel
    b.enabled = true;
    b.colorOp = b.alphaOp = vpgpu::BlendOp::Max;
    b.srcColor = b.srcAlpha = b.dstColor = b.dstAlpha = vpgpu::BlendFactor::One;
    b.writeMask = 1;
  } else {
    b.writeMask = 0;
  }
  st.label += writes ? ".umu" : ".ums";
  mv.key = st.key();
  mv.ok = true;
  mv.fragment = st.fragmentShader;
  p.maskVariants.push_back(mv);
  out = mv.key;
  static std::atomic<unsigned> n{0};
  if (n.fetch_add(1) < 12)
    logf("vk.taa uimask pipeline_variant pipeline=%s ui=%d samples=%u", st.label.c_str(), writes ? 1 : 0, unsigned(samples));
  retainKey(dev, p, mv.key);
  dev->gpu->pipelines().request(st, mv.key, vpgpu::Priority::Blocking);
  if (dev->recipes) dev->recipes->addPipeline(st, inputs);
  // Built while a submission is translated (submitMutex held): in synchronous mode the key is only noted and runBatches
  // waits for all of them once (awaitFreshVariants), not one compile at a time under the mutex.
  if (!dev->translator) dev->uiMask.freshVariants.push_back(mv.key);
  return true;
}

void awaitFreshVariants(Device* dev) {
  if (!dev || dev->uiMask.freshVariants.empty()) return;
  static const uint32_t budget = waitBudgetMs("VPVK_PIPELINE_DRAW_WAIT_MS", 2000);
  std::vector<vpgpu::PipelineKey> keys;
  keys.swap(dev->uiMask.freshVariants);
  awaitPipelines(dev, keys, budget, "uimask_variant");
}

OpDraw makeDraw(Recorder& r, uint32_t first, uint32_t count, uint32_t instances, uint32_t firstInstance,
                int32_t baseVertex, bool indexed) {
  OpDraw op;
  const PipelineObj* p = r.bound.graphics;
  if (p && !resolvePipeline(r.device, *p)) {
    op.ok = false, op.reason = SkipReason::TranslateTimeout;
    logSkip("translate_timeout", p);
    return op;
  }
  if (!p || !p->ok) {
    op.ok = false, op.reason = p ? SkipReason::PipelineFailed : SkipReason::NoPipeline;
    logSkip(p ? "pipeline_failed" : "no_pipeline", p);
    return op;
  }
  vpgpu::Draw& d = op.draw;
  const BoundState::Eds& eds = r.bound.eds;
  PipelineObj::TexelVariant* tv = texelVariant(r.device, *p, r.bound.graphicsSets, r.bound);
  d.pipeline = pipelineFor(r.device, *p, r.bound.passSamples, tv, &eds);
  const VkPrimitiveTopology topo = effectiveTopology(*p, &eds);
  op.pipe = p, op.texelVariant = tv, op.passSamples = r.bound.passSamples;
  op.depthWrite = p->isDynamic(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE) && eds.depthWrite >= 0 ? eds.depthWrite != 0
                                                                                         : p->state.depthWrite;
  op.edsVariant = eds.cullMode >= 0 || eds.frontFace >= 0 || eds.topology >= 0 || eds.depthCompare >= 0 ||
                  eds.depthTest >= 0 || eds.depthWrite >= 0 || eds.stencilTest >= 0 || eds.rasterDiscard >= 0 ||
                  eds.depthBias >= 0 || eds.restart >= 0 || eds.stencilOp[0] || eds.stencilOp[1];
  for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c)
    op.edsVariant = op.edsVariant || (eds.colorWrite[c] >= 0 && p->isDynamic(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT));
  for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c)
    op.blended = op.blended || (p->state.colorFormats[c] != vpgpu::Format::Undefined && p->state.blend[c].enabled &&
                                p->state.blend[c].writeMask);
  d.primitive = p->state.primitive;
  if (topo != p->topology) {
    bool known;
    d.primitive = primitiveOf(*p, topo, primitive(topo, known));  // fans and unknown classes: triangles (fans are converted below)
  }
  d.first = first, d.count = count;
  // Multiview by instancing: every Vulkan instance is drawn once per view (instanceViews = 1 when the
  // vertex shader multiviews through vertex amplification instead).
  d.instances = instances * p->instanceViews;
  d.baseInstance = firstInstance * p->instanceViews;
  d.instanceViews = p->instanceViews;  // the frame compiler's vertex count must not multiply by the views again
  d.amplified = p->amplified;
  d.baseVertex = baseVertex;
  op.viewCount = p->instanceViews;
  if (p->viewCount > 1) {  // gpu.amp: multiview draws, vertex amplified or by instancing (and why not)
    PipelineCounters& pc = pipelineCounters();
    constexpr auto kRelaxed = std::memory_order_relaxed;
    if (p->amplified) {
      pc.ampDraws.fetch_add(1, kRelaxed);
    } else {
      pc.ampInstDraws.fetch_add(1, kRelaxed), pc.ampBy[p->ampWhy].fetch_add(1, kRelaxed);
      pc.ampInstVerts.fetch_add(uint64_t(count) * d.instances, kRelaxed);
    }
  }
  if (!resolveBindings(r, *p, r.bound.graphicsSets, d.bindings, d.views, op.zeroReads)) {
    op.ok = false, op.reason = SkipReason::Unbound;
    logSkip("unbound_descriptor", p);
    return op;
  }
  if (r.bound.localRead && !r.bound.passImages.empty())  // attachments read after an in-pass barrier (commands.cpp localReadSplit)
    for (const vpgpu::Access& a : d.bindings)
      for (const Image* im : r.bound.passImages)
        if (a.range.kind == vpgpu::ResourceRange::Kind::Texture && im->id && a.range.index == im->id.index &&
            a.range.generation == im->id.generation && std::find(op.images.begin(), op.images.end(), im) == op.images.end())
          op.images.push_back(im);
  for (const spirv::VertexBufferBinding& vb : p->vertexBuffers) {
    const BoundState::VB& b = r.bound.vertex[vb.binding < r.bound.vertex.size() ? vb.binding : 0];
    if (!b.buffer || !b.buffer->id) continue;  // unused by the shader or not bound
    VkDeviceSize avail = b.offset < b.buffer->size ? b.buffer->size - b.offset : 0;
    if (b.size != VK_WHOLE_SIZE) avail = std::min(avail, b.size);  // vkCmdBindVertexBuffers2 pSizes
    vpgpu::Access a{vpgpu::ResourceRange::bytes(b.buffer->id, b.offset, avail), vpgpu::stage::vertex, false};
    a.kind = vpgpu::BindKind::Buffer, a.slot = vb.slot;
    d.bindings.push_back(a);
    op.vertexBuffers = true;
  }
  op.depthTested = p->isDynamic(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE) && eds.depthTest >= 0
                       ? eds.depthTest != 0
                       : p->state.depthTest || p->state.depthWrite;
  if (op.depthTested) {
    const VkCompareOp cmp = p->isDynamic(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP) && eds.depthCompare >= 0
                                ? VkCompareOp(eds.depthCompare)
                                : p->rawDepth.compare;
    op.depthCmp = cmp == VK_COMPARE_OP_GREATER || cmp == VK_COMPARE_OP_GREATER_OR_EQUAL ? 1
                  : cmp == VK_COMPARE_OP_LESS || cmp == VK_COMPARE_OP_LESS_OR_EQUAL ? -1 : 0;
  }
  if (indexed) {
    if (!r.bound.index || !r.bound.index->id) {
      op.ok = false, op.reason = SkipReason::NoIndex;
      logSkip("no_index_buffer", p);
      return op;
    }
    d.index = vpgpu::IndexBuffer{r.bound.index->id, r.bound.indexOffset, r.bound.indexWide};
  }
  // Triangle fans (Metal has none): an indexed triangle list (v0, vk, vk+1) in the command
  // buffer's upload memory; an indexed fan reads its indices now (unified memory: the buffer the
  // app filled before recording). Indirect fans become direct draws (recordIndirectDraw).
  if (topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN && count >= 3) {
    const uint8_t* src = nullptr;
    if (indexed) {
      const Buffer* ib = r.bound.index;
      const VkDeviceSize stride = r.bound.indexWide ? 4 : 2;
      const VkDeviceSize avail = ib->size > r.bound.indexOffset ? (ib->size - r.bound.indexOffset) / stride : 0;
      if (ib->memory && ib->memory->host && uint64_t(first) + count <= avail)  // in bounds only
        src = ib->memory->host + ib->offset + r.bound.indexOffset;
    }
    if (!indexed || src) {
      auto at = [&](uint32_t i) -> uint32_t {
        if (!indexed) return i;  // relative: drawn with baseVertex = first (gl_BaseVertex stays right)
        if (r.bound.indexWide) {
          uint32_t v;
          std::memcpy(&v, src + size_t(first + i) * 4, 4);
          return v;
        }
        uint16_t v;
        std::memcpy(&v, src + size_t(first + i) * 2, 2);
        return v;
      };
      std::vector<uint32_t> tri;
      tri.reserve(size_t(count - 2) * 3);
      // Primitive restart (indexed): the restart index ends a fan and the next one starts after it.
      const uint32_t restart = r.bound.indexWide ? 0xFFFFFFFFu : 0xFFFFu;
      const bool splits = indexed && (p->isDynamic(VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE) && eds.restart >= 0
                                          ? eds.restart != 0
                                          : p->primitiveRestart);
      for (uint32_t s0 = 0; s0 < count;) {
        uint32_t e = s0;
        while (e < count && !(splits && at(e) == restart)) ++e;  // [s0, e) is one fan
        if (e - s0 >= 3) {
          const uint32_t v0 = at(s0);
          for (uint32_t k = s0 + 1; k + 1 < e; ++k) tri.insert(tri.end(), {v0, at(k), at(k + 1)});
        }
        s0 = e + 1;
      }
      if (tri.empty()) {
        op.ok = false, op.reason = SkipReason::Fan;  // nothing but restarts / short fans: nothing to draw
        logSkip("fan_empty", p, "count=%u indexed=%d restart=%d", count, int(indexed), int(splits));
        return op;
      }
      VkDeviceSize off = 0;
      const vpgpu::BufferId id = r.arena.put(r.device, tri.data(), tri.size() * 4, off);
      d.index = vpgpu::IndexBuffer{id, off, true};
      d.first = 0, d.count = uint32_t(tri.size());
      if (!indexed) d.baseVertex = int32_t(first), op.fanConverted = true;  // list ids are relative to `first`
      d.primitive = primitiveOf(*p, topo, vpgpu::Primitive::Triangles);
    } else {
      static std::atomic<int> k{0};
      if (k.fetch_add(1) < 4) logf("vk.err what=fan_unconverted reason=index_buffer_not_host_visible_or_out_of_range pipeline=%s", p->state.label.c_str());
    }
  }
  if (r.bound.hasViewport) {
    const VkViewport& v = r.bound.viewport;
    d.viewport = {v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth};
    // Diagnostic (one-shot): viewport of multiview draws and any negative-height (flipped) viewport,
    // which Metal does not honour like Vulkan does.
    static bool mvLogged = false, negLogged = false;
    if ((p->viewCount > 1 && !mvLogged) || (v.height < 0 && !negLogged)) {
      (p->viewCount > 1 ? mvLogged : negLogged) = true;
      if (v.height < 0) negLogged = true;
      std::fprintf(stderr, "gpu.diag viewport views=%u x=%.1f y=%.1f w=%.1f h=%.1f z=%.2f..%.2f scissor=%d\n",
                   p->viewCount, v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth, int(r.bound.hasScissor));
    }
  }
  if (r.bound.hasScissor) {
    // Clipped to the render area: Metal rejects a scissor outside the attachment.
    const VkRect2D& s = r.bound.scissor;
    int64_t x0 = std::max<int64_t>(0, s.offset.x), y0 = std::max<int64_t>(0, s.offset.y);
    int64_t x1 = int64_t(s.offset.x) + s.extent.width, y1 = int64_t(s.offset.y) + s.extent.height;
    if (r.bound.targetWidth) x1 = std::min<int64_t>(x1, r.bound.targetWidth), y1 = std::min<int64_t>(y1, r.bound.targetHeight);
    if (x1 <= x0 || y1 <= y0) {
      op.ok = false, op.reason = SkipReason::Scissor;  // nothing visible
      logSkip("empty_scissor", p,
              "scissor=%d,%d,%lld,%lld clamped=%lld,%lld,%lld,%lld target=%ux%u secondary=%d area=%d,%d,%ux%u fb=%ux%ux%u",
              int(s.offset.x), int(s.offset.y), (long long)s.offset.x + s.extent.width,
              (long long)s.offset.y + s.extent.height, (long long)x0, (long long)y0, (long long)x1, (long long)y1,
              r.bound.targetWidth, r.bound.targetHeight, int(!r.inPass), int(r.bound.renderArea.offset.x),
              int(r.bound.renderArea.offset.y), r.bound.renderArea.extent.width, r.bound.renderArea.extent.height,
              r.bound.fbWidth, r.bound.fbHeight, r.bound.fbLayers);
      return op;
    }
    d.scissor = {uint32_t(x0), uint32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0)};
  }
  const bool biasOn = p->isDynamic(VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE) && eds.depthBias >= 0 ? eds.depthBias != 0
                                                                                              : p->depthBiasEnable;
  if (biasOn) d.depthBias = r.bound.depthBias[0], d.depthBiasClamp = r.bound.depthBias[1], d.depthBiasSlope = r.bound.depthBias[2];
  d.depthClamp = p->depthClamp;
  d.wireframe = p->wireframe;
  // Culling front and back faces removes every triangle (lines and points are not culled).
  const bool cullAll = p->isDynamic(VK_DYNAMIC_STATE_CULL_MODE) && eds.cullMode >= 0
                           ? VkCullModeFlags(eds.cullMode) == VK_CULL_MODE_FRONT_AND_BACK
                           : p->cullBoth;
  if (cullAll && isTriangleTopology(topo)) {
    op.ok = false, op.reason = SkipReason::Cull;
    logSkip("cull_front_and_back", p, "pipeline_cull_both=%d dynamic_cull=%d eds_cull=%d topology=%d",
            int(p->cullBoth), int(p->isDynamic(VK_DYNAMIC_STATE_CULL_MODE)), int(eds.cullMode), int(topo));
  }

  d.stencilRef = static_cast<uint8_t>(r.bound.stencilRef[0]);
  d.stencilRefBack = static_cast<int16_t>(r.bound.stencilRef[1] & 0xFF);
  for (int f = 0; f < 2; ++f)
    d.stencilReadMask[f] = int16_t(r.bound.stencilCompareMask[f]), d.stencilWriteMask[f] = int16_t(r.bound.stencilWriteMask[f]);
  std::memcpy(d.blendColor.data(), r.bound.blend, sizeof r.bound.blend);
  const VkFrontFace front = p->isDynamic(VK_DYNAMIC_STATE_FRONT_FACE) && eds.frontFace >= 0 ? VkFrontFace(eds.frontFace)
                                                                                               : p->frontFace;
  d.frontFace = front == VK_FRONT_FACE_CLOCKWISE ? vpgpu::Winding::Clockwise : vpgpu::Winding::CounterClockwise;
  if (QueryPoolObj* q = r.bound.queryPool; q && q->device && q->device->visibility.id)
    d.visibility = vpgpu::BufferSpan{q->device->visibility.id, uint64_t(q->base + r.bound.query) * 8, 8};
  return op;
}

bool gpuIndirectEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VPVK_GPU_INDIRECT");
    return !(e && e[0] == '0');
  }();
  return on;
}

// The argument patch kernel: copies `n` indirect records (srcStride bytes apart) into a packed array
// after a 32-byte parameter block, multiplying instanceCount and firstInstance by the view count
// (multiview draws instances x views, see spirv lower.cpp) and zeroing the instance count of the
// records at or past the GPU-side draw count (vkCmdDraw*IndirectCount). A record with a garbage count or instance count
// (past kMaxIndirectCount / kMaxIndirectInstances in commands.cpp: an uninitialised or stale buffer) becomes an empty draw:
// it would run for seconds and the GPU watchdog would kill the command buffer.
// Parameters (uints): n, srcStride/4, words (4 or 5), views, hasCount, countWord, 0, 0.
constexpr uint64_t kIndirectPatchShader = 0x7650564b49505431ull;  // "vPVKIPT1"
const char* kIndirectPatchMsl = R"(#include <metal_stdlib>
using namespace metal;
struct VpArgs {
    const device uint* src;
    device uint* dst;
    const device uint* count;
};
[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& a [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    device uint* prm = a.dst;
    const uint n = prm[0];
    if (i >= n) return;
    const uint words = prm[2], views = max(prm[3], 1u);
    const device uint* s = a.src + i * prm[1];
    device uint* o = a.dst + 8u + i * words;
    for (uint k = 0; k < words; ++k) o[k] = s[k];
    const uint cnt = prm[4] != 0u ? a.count[prm[5]] : n;
    if (i >= cnt) o[1] = 0u;
    if (o[0] > 33554432u || o[1] > 1048576u) o[0] = 0u, o[1] = 0u;
    o[1] = o[1] * views;
    o[words - 1u] = o[words - 1u] * views;
}
)";

// The same logic in C++ for record `i` (tests; keep in step with vp_main above): `prm` is the 8-word parameter block, `src` the app's
// records (prm[1] words apart), `count` the GPU draw count buffer; writes prm[2] words to `out` (the packed record i).
void indirectPatchRef(uint32_t i, const uint32_t* prm, const uint32_t* src, const uint32_t* count, uint32_t* out) {
  const uint32_t n = prm[0];
  if (i >= n) return;
  const uint32_t words = prm[2], views = std::max(prm[3], 1u);
  const uint32_t* s = src + size_t(i) * prm[1];
  for (uint32_t k = 0; k < words; ++k) out[k] = s[k];
  const uint32_t cnt = prm[4] != 0u ? count[prm[5]] : n;
  if (i >= cnt) out[1] = 0u;
  if (out[0] > 33554432u || out[1] > 1048576u) out[0] = 0u, out[1] = 0u;
  out[1] = out[1] * views;
  out[words - 1u] = out[words - 1u] * views;
}
const char* indirectPatchMsl() { return kIndirectPatchMsl; }

// Registers and requests the patch kernel (once per device), at Urgent priority: vkCreateDevice calls it so the compile is under way
// before the first indirect draw.
void requestIndirectPatch(Device* d) {
  std::call_once(d->indirectPatchOnce, [d] {
    ShaderBlob blob;
    blob.source = kIndirectPatchMsl;
    blob.threadsPerGroup = {64, 1, 1};
    blob.arguments = {{vpgpu::BindKind::Buffer, 0, 0}, {vpgpu::BindKind::Buffer, 1, 8}, {vpgpu::BindKind::Buffer, 2, 16}};
    d->platform->addShader(kIndirectPatchShader, std::move(blob));
    vpgpu::PipelineState st;
    st.computeShader = kIndirectPatchShader;
    st.label = "vk.indirect_patch";
    d->indirectPatchKey = st.key();
    d->gpu->pipelines().request(st, d->indirectPatchKey, vpgpu::Priority::Blocking);
    d->gpu->pipelines().boost(d->indirectPatchKey);  // Urgent: ahead of the prewarm and every queued compile
  });
}

// True when the patch kernel is compiled and the draw may use it. Waits (bounded, 2 s: it is a tiny kernel) while it is queued or
// compiling, rather than recording a dispatch the backend would skip and leave the patched arguments zero (the draws vanish). A
// kernel that failed or did not come back in time sends the draw to the CPU path (or skips a draw-count draw, which has no CPU path),
// loudly: vk.indirect_patch fail, counted in PipelineCounters::indirectPatchFallback.
bool indirectPatch(Device* d, vpgpu::PipelineKey& key) {
  requestIndirectPatch(d);
  key = d->indirectPatchKey;
  using Status = vpgpu::PipelineCache::Status;
  using Clock = std::chrono::steady_clock;
  const auto nowNs = [] { return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()); };
  vpgpu::PipelineCache& cache = d->gpu->pipelines();
  Status st = cache.status(key);
  if ((st == Status::Queued || st == Status::Compiling) && nowNs() >= d->indirectPatchBackoffNs.load(std::memory_order_relaxed)) {
    cache.boost(key);
    const auto t0 = Clock::now();
    cache.waitReady({key}, std::chrono::milliseconds(d->indirectPatchWaitMs));
    st = cache.status(key);
    if (st == Status::Queued || st == Status::Compiling)  // timed out: the next draws do not wait again for a while
      d->indirectPatchBackoffNs.store(nowNs() + int64_t(d->indirectPatchBackoffMs) * 1'000'000, std::memory_order_relaxed);
    static std::atomic<unsigned> w{0};
    if (w.fetch_add(1) < 4)
      logf("vk.indirect_patch wait ms=%.0f status=%d", std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), int(st));
  }
  if (st == Status::Ready) return true;
  const uint64_t total = ++pipelineCounters().indirectPatchFallback;
  // Loud while it lasts: the first 8, then at least one line per second with the cumulative count.
  static std::atomic<int64_t> lastLogNs{0};
  int64_t last = lastLogNs.load(std::memory_order_relaxed);
  const int64_t now = nowNs();
  if (total <= 8 || (now - last >= 1'000'000'000 && lastLogNs.compare_exchange_strong(last, now)))
    logf("vk.indirect_patch fail status=%d (0 missing 1 queued 2 compiling 3 ready 4 failed) fallback=cpu_or_skip total=%llu", int(st),
         (unsigned long long)total);
  return false;
}

// vkCmdDraw(Indexed)Indirect(Count). GPU-read arguments whenever the draw is recorded inside a
// render pass of this command buffer (the common case): plain draws read the app's buffer
// directly; multiview and draw-count draws read a copy that the patch kernel writes before the
// pass begins (Vulkan forbids writing the argument buffer inside the pass, so it is final there).
// Secondary command buffers, VPVK_GPU_INDIRECT=0 or a missing kernel: arguments read at submit (stale if the GPU writes
// them: known gap for multiview in a secondary buffer, whose patch dispatch would have to go before the primary's pass;
// counted by TranslateStats::indirectCpu / indirectCpuSecondary, vk.perf ind{cpu cpu_sec}).
void recordIndirectDraw(Recorder& r, const Buffer* b, VkDeviceSize off, uint32_t n, uint32_t stride, bool indexed,
                        const Buffer* countBuf, VkDeviceSize countOff) {
  OpDraw op = makeDraw(r, 0, 1, 1, 0, 0, indexed);
  op.indirect = b, op.indirectOffset = off, op.indirectCount = n, op.indirectStride = stride;
  if (!op.ok || !b || !b->id || n == 0) {
    if (countBuf && op.ok) {  // no GPU count without the GPU path (logged by the skip)
      op.ok = false, op.reason = SkipReason::Indirect;
      logSkip("indirect_count_no_gpu_path", op.pipe, "n=%u buffer=%d", n, int(b && b->id));
    }
    r.ops.emplace_back(std::move(op));
    return;
  }
  // Triangle fans have no Metal topology: each record becomes a direct draw converted like vkCmdDraw (makeDraw), the
  // arguments read now from the host-visible buffer (stale if the GPU writes them later in the submission).
  if (!countBuf && op.pipe && b->memory && b->memory->host &&
      effectiveTopology(*op.pipe, &r.bound.eds) == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
    const uint32_t size = indexed ? 20 : 16;
    for (uint32_t i = 0; i < n; ++i) {
      const VkDeviceSize at = off + VkDeviceSize(i) * stride;
      if (at + size > b->size) break;
      const uint8_t* src = b->memory->host + b->offset + at;
      uint32_t count, instances, first, firstInstance;
      int32_t vertexOffset = 0;
      if (indexed) {
        VkDrawIndexedIndirectCommand c;
        std::memcpy(&c, src, sizeof c);
        count = c.indexCount, instances = c.instanceCount, first = c.firstIndex, firstInstance = c.firstInstance;
        vertexOffset = c.vertexOffset;
      } else {
        VkDrawIndirectCommand c;
        std::memcpy(&c, src, sizeof c);
        count = c.vertexCount, instances = c.instanceCount, first = c.firstVertex, firstInstance = c.firstInstance;
      }
      if (!count || !instances) continue;
      if (count > (1u << 25) || instances > (1u << 20)) {  // garbage arguments (commands.cpp kMaxIndirectCount)
        logSkip("indirect_clamped", op.pipe, "fan draw=%u of %u count=%u instances=%u", i, n, count, instances);
        continue;
      }
      OpDraw d = makeDraw(r, first, count, instances, firstInstance, vertexOffset, indexed);
      d.draw.drawIndex = i;
      r.ops.emplace_back(std::move(d));
    }
    return;
  }
  const uint32_t words = indexed ? 5 : 4;
  if (indexed && r.bound.index) {  // hazards cover the whole bound index range
    const VkDeviceSize ib = r.bound.index->size > r.bound.indexOffset ? r.bound.index->size - r.bound.indexOffset : 0;
    op.draw.first = 0, op.draw.count = uint32_t(std::min<VkDeviceSize>(ib / (r.bound.indexWide ? 4 : 2), 0xFFFFFFFFu));
  } else {
    op.draw.first = 0, op.draw.count = 0xFFFF;  // unknown on the CPU: never a "fullscreen" pass to VPKTAA
  }
  const bool patch = op.viewCount > 1 || countBuf;
  vpgpu::PipelineKey pk{};
  if (!gpuIndirectEnabled() || (patch && (r.passBegin == SIZE_MAX || r.passBegin >= r.ops.size() || n > 65536 ||
                                          (countBuf && !countBuf->id) || !indirectPatch(r.device, pk)))) {
    if (countBuf) {
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 4) logf("vk.err what=draw_indirect_count_unsupported reason=outside_render_pass_or_disabled");
      op.ok = false, op.reason = SkipReason::Indirect;
      logSkip("indirect_count_unsupported", op.pipe, "in_pass=%d n=%u", int(r.passBegin != SIZE_MAX), n);
    }
    r.ops.emplace_back(std::move(op));
    return;
  }
  op.gpuIndirect = true;
  if (!patch) {
    op.gpuBuffer = b->id, op.gpuOffset = off;
    r.ops.emplace_back(std::move(op));
    return;
  }
  // Parameters + room for the patched records, in the command buffer's arena.
  std::vector<uint32_t> blob(8 + size_t(n) * words, 0);
  blob[0] = n, blob[1] = stride / 4, blob[2] = words, blob[3] = std::max<uint32_t>(op.viewCount, 1);
  blob[4] = countBuf ? 1 : 0, blob[5] = 0;
  VkDeviceSize aoff = 0;
  const vpgpu::BufferId arena = r.arena.put(r.device, blob.data(), blob.size() * 4, aoff);
  OpDispatch dp;
  dp.dispatch.pipeline = pk;
  dp.dispatch.groups = {(n + 63) / 64, 1, 1};
  dp.dispatch.label = "vk.indirect_patch";
  dp.dispatch.required = true;
  const VkDeviceSize srcBytes = VkDeviceSize(n - 1) * stride + words * 4;
  auto acc = [](vpgpu::ResourceRange rr, bool w, uint16_t slot) {
    vpgpu::Access a{rr, vpgpu::stage::compute, w};
    a.kind = vpgpu::BindKind::Buffer, a.slot = slot;
    return a;
  };
  dp.dispatch.bindings.push_back(acc(vpgpu::ResourceRange::bytes(b->id, off, srcBytes), false, 0));
  dp.dispatch.bindings.push_back(acc(vpgpu::ResourceRange::bytes(arena, aoff, blob.size() * 4), true, 1));
  if (countBuf) dp.dispatch.bindings.push_back(acc(vpgpu::ResourceRange::bytes(countBuf->id, countOff, 4), false, 2));
  else dp.dispatch.bindings.push_back(acc(vpgpu::ResourceRange::bytes(b->id, off, 4), false, 2));
  r.ops.insert(r.ops.begin() + std::ptrdiff_t(r.passBegin), Op{std::move(dp)});
  ++r.passBegin;
  op.gpuBuffer = arena, op.gpuOffset = aoff + 32;
  op.indirectStride = words * 4;
  r.ops.emplace_back(std::move(op));
}

OpDispatch makeDispatch(Recorder& r, uint32_t x, uint32_t y, uint32_t z) {
  OpDispatch op;
  const PipelineObj* p = r.bound.compute;
  if (p && !resolvePipeline(r.device, *p)) {
    op.ok = false;
    logSkip("translate_timeout", p);
    return op;
  }
  if (!p || !p->ok) {
    op.ok = false;
    logSkip(p ? "compute_failed" : "no_compute_pipeline", p);
    return op;
  }
  const PipelineObj::TexelVariant* tv = texelVariant(r.device, *p, r.bound.computeSets, r.bound);
  op.dispatch.pipeline = tv ? tv->key : p->key;
  op.dispatch.groups = {x, y, z};
  op.dispatch.label = p->state.label;
  if (!resolveBindings(r, *p, r.bound.computeSets, op.dispatch.bindings, op.dispatch.views, op.zeroReads)) {
    op.ok = false;
    logSkip("unbound_descriptor", p);
  }
  return op;
}

}  // namespace vpvk

namespace vpvk {
size_t recordedOps(VkCommandBuffer cb) { return rec(cb).ops.size(); }
// The draw recorded after `before` ops (none when an empty draw was skipped) gets `index` as its
// gl_DrawID (vkCmdDrawMulti*EXT).
void setDrawIndexSince(VkCommandBuffer cb, size_t before, uint32_t index) {
  Recorder& r = rec(cb);
  if (r.ops.size() <= before) return;
  if (auto* d = std::get_if<OpDraw>(&r.ops.back())) d->draw.drawIndex = index;
}
}  // namespace vpvk

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer cb, uint32_t count, uint32_t instances, uint32_t first,
                                                 uint32_t firstInstance) {
  Recorder& r = rec(cb);
  if (!count || !instances) return;
  r.ops.emplace_back(makeDraw(r, first, count, instances, firstInstance, 0, false));
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t count, uint32_t instances,
                                                        uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
  Recorder& r = rec(cb);
  if (!count || !instances) return;
  r.ops.emplace_back(makeDraw(r, firstIndex, count, instances, firstInstance, vertexOffset, true));
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, uint32_t n,
                                                         uint32_t stride) {
  recordIndirectDraw(rec(cb), from<Buffer>(buf), off, n, stride ? stride : 16, false, nullptr, 0);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off,
                                                                uint32_t n, uint32_t stride) {
  recordIndirectDraw(rec(cb), from<Buffer>(buf), off, n, stride ? stride : 20, true, nullptr, 0);
}

// VK_KHR_draw_indirect_count (core 1.2): the draw count is read by the GPU from countBuffer, at most
// maxDrawCount draws (the argument patch zeroes the instance count of the draws past it).
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off,
                                                              VkBuffer countBuf, VkDeviceSize countOff,
                                                              uint32_t maxDrawCount, uint32_t stride) {
  recordIndirectDraw(rec(cb), from<Buffer>(buf), off, maxDrawCount, stride ? stride : 16, false, from<Buffer>(countBuf),
                     countOff);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off,
                                                                     VkBuffer countBuf, VkDeviceSize countOff,
                                                                     uint32_t maxDrawCount, uint32_t stride) {
  recordIndirectDraw(rec(cb), from<Buffer>(buf), off, maxDrawCount, stride ? stride : 20, true, from<Buffer>(countBuf),
                     countOff);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
  Recorder& r = rec(cb);
  r.ops.emplace_back(makeDispatch(r, x, y, z));
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off) {
  Recorder& r = rec(cb);
  OpDispatch op = makeDispatch(r, 1, 1, 1);
  const Buffer* b = from<Buffer>(buf);
  // The GPU reads the group counts when it gets to the dispatch (a compute pass may have written
  // them earlier in the same submission); VPVK_GPU_INDIRECT=0 reads them at submit as before.
  if (b && b->id && gpuIndirectEnabled()) op.dispatch.indirect = vpgpu::Indirect{b->id, off};
  else op.indirect = b, op.indirectOffset = off;
  r.ops.emplace_back(std::move(op));
}

// =================================================================================================
// Passes: subpasses, dynamic rendering, clears inside a pass, secondary command buffers
// =================================================================================================
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass(VkCommandBuffer cb, VkSubpassContents) {
  Recorder& r = rec(cb);
  r.ops.emplace_back(OpNextSubpass{});
  ++r.bound.subpass;
  if (r.bound.pass && r.bound.framebuffer) r.bound.passSamples = subpassSamples(*r.bound.pass, *r.bound.framebuffer, r.bound.subpass);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass2(VkCommandBuffer cb, const VkSubpassBeginInfo*, const VkSubpassEndInfo*) {
  vkCmdNextSubpass(cb, VK_SUBPASS_CONTENTS_INLINE);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  Recorder& r = rec(cb);
  OpBeginRendering op;
  vpgpu::RenderPassDesc& d = op.desc;
  std::vector<std::pair<const ImageView*, const ImageView*>> forcedPairs, intResolves;
  // Colour clears a render area smaller than the attachment cannot express as a load action: the
  // attachment loads and the clear becomes a quad over the area (op.areaClear).
  auto cr = std::make_shared<vpgpu::ClearRects>();
  bool depthForced = false;  // the depth/stencil store was forced to Store (forced 1x, integer resolve)
  // A multiview mask other than views 0..n-1 (0b1000, 0b101): the pass has bit_width(mask) layers
  // and only the mask's ones are this pass's.
  const bool irregular = ri->viewMask && (ri->viewMask & (ri->viewMask + 1)) != 0;
  // Suspend/resume: the chunks of a suspended render pass instance are ONE instance (Source 2 records its eye pass
  // over several command buffers: flags 0x2 / 0x6 / 0x4). Vulkan requires every chunk to repeat the same
  // VkRenderingInfo, loadOp CLEAR included, and runs the load ops only at the first chunk and the store/resolve ops
  // only at the last. Taken literally, every resumed chunk would clear and only the last chunk's draws would survive.
  const bool resuming = (ri->flags & VK_RENDERING_RESUMING_BIT) != 0;
  const bool suspending = (ri->flags & VK_RENDERING_SUSPENDING_BIT) != 0;
  if (resuming || suspending) {
    static std::atomic<unsigned> said{0};
    if (said.fetch_add(1) < 4)
      logf("vk.rendering suspend_resume flags=0x%x: loads kept from the first chunk, stores/resolves at the last",
           unsigned(ri->flags));
  }
  auto fill = [&](const VkRenderingAttachmentInfo& a, vpgpu::Attachment& t, bool depth) {
    const ImageView* v = a.imageView ? from<ImageView>(a.imageView) : nullptr;
    if (!v || !v->image || !v->image->id) return;
    t.texture = v->image->id;
    t.mip = uint16_t(v->range.baseMipLevel), t.layer = uint16_t(v->range.baseArrayLayer);
    t.load = resuming ? vpgpu::LoadAction::Load : toLoad(a.loadOp);
    // A view of another format (mutable-format image): rendered as the view's format.
    const vpgpu::Format cf = depth ? v->image->desc.format : attachmentFormat(*v);
    if (cf != v->image->desc.format) t.format = cf;
    // Outside the render area, and in the layers an irregular view mask leaves out, the attachment
    // keeps its contents: Metal's DontCare / Clear act on the whole attachment, so load (and clear
    // with quads) instead.
    const bool keepOthers = partialArea(ri->renderArea, v) || irregular;
    if (t.load == vpgpu::LoadAction::DontCare && keepOthers) t.load = vpgpu::LoadAction::Load;
    if (depth) t.clear.depth = a.clearValue.depthStencil.depth, t.clear.stencil = uint8_t(a.clearValue.depthStencil.stencil);
    else t.clear.color = clearColor(a.clearValue.color, cf);
    if (!depth && t.load == vpgpu::LoadAction::Clear && keepOthers) {
      const int slot = int(&t - d.colors.data());
      t.load = vpgpu::LoadAction::Load;
      cr->colorMask |= uint8_t(1u << slot);
      for (int c = 0; c < 4; ++c) cr->colorBits[slot][c] = a.clearValue.color.uint32[c];
    }
    bool resolve = false;
    // A resolve (also the forced-1x and integer copies) happens when the instance ends, not at a suspension.
    if (a.resolveMode != VK_RESOLVE_MODE_NONE && a.resolveImageView && !suspending) {
      const ImageView* rv = from<ImageView>(a.resolveImageView);
      if (rv && rv->image && rv->image->id && resolveAsCopy(v)) {
        // Game MSAA forced to 1x, or a 1x attachment the game "resolves" (Source 2's depth with vr_msaa 0): draw into the
        // 1x source, store it, copy it into the resolve target after the pass (see forcedResolveCopy in commands.cpp for
        // why not render direct). A Metal resolve of a 1x texture is undefined: it hung the GPU on every eye pass.
        t.store = vpgpu::StoreAction::Store;
        depthForced = depthForced || depth;
        if (resolveCopyUsable(v, rv)) forcedPairs.push_back({v, rv});
        return;
      }
      if (rv && rv->image && rv->image->id && !depth && integerResolve(v)) {
        // Integer MSAA (Metal resolves only float formats): store the samples, blit sample 0
        // into the resolve target after the pass (commands.cpp sampleZeroResolve).
        t.store = vpgpu::StoreAction::Store;
        depthForced = depthForced || depth;
        intResolves.push_back({v, rv});
        return;
      }
      if (rv && rv->image && rv->image->id) {
        t.resolve = rv->image->id;
        t.resolveMip = uint16_t(rv->range.baseMipLevel), t.resolveLayer = uint16_t(rv->range.baseArrayLayer);
        resolve = true;
      }
    }
    t.store = toStore(a.storeOp, resolve);
    // A suspending chunk keeps everything for the chunk that resumes it, whatever storeOp says.
    if (suspending) t.store = vpgpu::StoreAction::Store;
    if (keepOthers && t.store == vpgpu::StoreAction::DontCare) t.store = vpgpu::StoreAction::Store;
    if (keepOthers && t.store == vpgpu::StoreAction::Resolve) t.store = vpgpu::StoreAction::StoreAndResolve;
    if (t.store == vpgpu::StoreAction::StoreAndResolve) t.keepSamples = true;  // see attach() in commands.cpp
    if (!depth && !keepOthers && !suspending && !v->image->eye && t.store == vpgpu::StoreAction::Store)
      op.dropColor |= 1u << unsigned(&t - d.colors.data());
  };
  for (uint32_t i = 0; i < ri->colorAttachmentCount && i < vpgpu::kMaxColorAttachments; ++i)
    fill(ri->pColorAttachments[i], d.colors[i], false);
  const VkRenderingAttachmentInfo* da = ri->pDepthAttachment && ri->pDepthAttachment->imageView ? ri->pDepthAttachment : nullptr;
  const VkRenderingAttachmentInfo* sa =
      ri->pStencilAttachment && ri->pStencilAttachment->imageView ? ri->pStencilAttachment : nullptr;
  if (da) fill(*da, d.depth, true);
  else if (sa) fill(*sa, d.depth, true);
  // Depth/stencil resolve: one target for the image, a mode per aspect (an aspect not attached or
  // with mode NONE is not resolved). The resolve may come from the stencil side only.
  auto resolving = [](const VkRenderingAttachmentInfo* a) {
    return a && a->resolveMode != VK_RESOLVE_MODE_NONE && a->resolveImageView ? a : nullptr;
  };
  // (Not at a suspending chunk: the resolve is the last chunk's.)
  if (!suspending && d.depth.texture && resolveAsCopy(from<ImageView>((da ? da : sa)->imageView)) &&
      !resolving(da) && resolving(sa)) {
    // Forced 1x (or a 1x image) with only the stencil resolved: fill() saw no resolve on the depth side.
    const ImageView* v = from<ImageView>(sa->imageView);
    const ImageView* rv = from<ImageView>(sa->resolveImageView);
    if (rv && rv->image && rv->image->id) {
      d.depth.store = vpgpu::StoreAction::Store;
      depthForced = true;
      if (resolveCopyUsable(v, rv)) forcedPairs.push_back({v, rv});
    }
  } else if (!suspending && d.depth.texture && !resolveAsCopy(from<ImageView>((da ? da : sa)->imageView))) {
    const VkRenderingAttachmentInfo* ra = resolving(da) ? resolving(da) : resolving(sa);
    const ImageView* rv = ra ? from<ImageView>(ra->resolveImageView) : nullptr;
    if (!d.depth.resolve && rv && rv->image && rv->image->id) {
      d.depth.resolve = rv->image->id;
      d.depth.resolveMip = uint16_t(rv->range.baseMipLevel), d.depth.resolveLayer = uint16_t(rv->range.baseArrayLayer);
      d.depth.store = toStore((da ? da : sa)->storeOp, true);
    }
    d.depth.depthResolve = resolving(da) ? resolveMode(da->resolveMode) : vpgpu::ResolveMode::None;
    d.depth.stencilResolve = resolving(sa) ? resolveMode(sa->resolveMode) : vpgpu::ResolveMode::None;
  }
  // One depth/stencil image, two load ops (VkRenderingInfo has one per aspect): what the core's
  // single load action cannot say becomes Load plus a quad clear of the cleared aspect(s).
  if (d.depth.texture && (da || sa)) {
    const ImageView* v = from<ImageView>((da ? da : sa)->imageView);
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(v->image->desc.format);
    const bool clearD = !resuming && da && fi.depth() && da->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    const bool clearS = !resuming && sa && fi.stencil() && sa->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    // An aspect of the image that is not attached keeps its contents (it is loaded, and stored below).
    auto loads = [](VkAttachmentLoadOp op) { return op == VK_ATTACHMENT_LOAD_OP_LOAD || op == VK_ATTACHMENT_LOAD_OP_NONE_KHR; };
    const bool loadD = fi.depth() && (da ? loads(da->loadOp) : true);
    const bool loadS = fi.stencil() && (sa ? loads(sa->loadOp) : true);
    if (da) d.depth.clear.depth = da->clearValue.depthStencil.depth;
    if (sa) d.depth.clear.stencil = uint8_t(sa->clearValue.depthStencil.stencil);
    const bool keepOthers = partialArea(ri->renderArea, v) || irregular;
    if ((clearD || clearS) && (loadD || loadS || keepOthers)) {
      d.depth.load = vpgpu::LoadAction::Load;
      cr->depth = clearD;
      cr->stencil = clearS;
      cr->depthValue = d.depth.clear.depth;
      cr->stencilValue = d.depth.clear.stencil;
    } else if (clearD || clearS) {
      d.depth.load = vpgpu::LoadAction::Clear;
    } else if (loadD || loadS || keepOthers || resuming) {
      d.depth.load = vpgpu::LoadAction::Load;
    } else {
      d.depth.load = vpgpu::LoadAction::DontCare;
    }
    // One store action for both aspects: keep the image when either aspect stores, or when an
    // aspect of it is not attached at all (its contents must survive the pass).
    auto stores = [](const VkRenderingAttachmentInfo* a) {
      return a && (a->storeOp == VK_ATTACHMENT_STORE_OP_STORE || a->storeOp == VK_ATTACHMENT_STORE_OP_NONE);
    };
    const bool keep = stores(da) || stores(sa) || (fi.depth() && !da) || (fi.stencil() && !sa) || keepOthers;
    if (keep && d.depth.store == vpgpu::StoreAction::DontCare) d.depth.store = vpgpu::StoreAction::Store;
    if (keep && d.depth.store == vpgpu::StoreAction::Resolve)
      d.depth.store = vpgpu::StoreAction::StoreAndResolve, d.depth.keepSamples = true;
    if (!depthForced && !keepOthers && !suspending && !v->image->eye && !(fi.depth() && !da) && !(fi.stencil() && !sa) &&
        d.depth.store == vpgpu::StoreAction::Store)
      op.dropDepth = uint8_t((stores(da) ? 1 : 0) | (stores(sa) ? 2 : 0));
  }
  if (cr->colorMask || cr->depth || cr->stencil) {
    const uint32_t x = uint32_t(std::max(0, ri->renderArea.offset.x)), y = uint32_t(std::max(0, ri->renderArea.offset.y));
    if (ri->viewMask) {  // the views' layers only
      for (uint32_t m = ri->viewMask; m; m &= m - 1)
        cr->rects.push_back({x, y, ri->renderArea.extent.width, ri->renderArea.extent.height,
                             uint16_t(std::countr_zero(m)), 1});
    } else {
      cr->rects.push_back({x, y, ri->renderArea.extent.width, ri->renderArea.extent.height, 0,
                           uint16_t(std::max<uint32_t>(1, ri->layerCount))});
    }
    op.areaClear = std::move(cr);
  }
  op.area = ri->renderArea;
  op.viewMask = ri->viewMask;
  // A stencil-only attachment is not a depth attachment for the VPKTAA jitter rule.
  if (ri->pDepthAttachment && ri->pDepthAttachment->imageView && d.depth.texture &&
      vpgpu::formatInfo(from<ImageView>(ri->pDepthAttachment->imageView)->image->desc.format).depth()) {
    const ImageView* v = from<ImageView>(ri->pDepthAttachment->imageView);
    op.depthWidth = std::max(1u, v->image->desc.width >> v->range.baseMipLevel);
    op.depthHeight = std::max(1u, v->image->desc.height >> v->range.baseMipLevel);
  }
  d.layerCount = ri->viewMask ? uint16_t(std::bit_width(ri->viewMask)) : uint16_t(std::max<uint32_t>(1, ri->layerCount));
  d.label = "vk.rendering";
  // Attachmentless dynamic rendering: the render area bounds the target (only then: see beginPass).
  bool attached = bool(d.depth.texture);
  for (const vpgpu::Attachment& a : d.colors) attached = attached || bool(a.texture);
  if (!attached) {
    d.width = uint32_t(std::max(0, ri->renderArea.offset.x)) + ri->renderArea.extent.width;
    d.height = uint32_t(std::max(0, ri->renderArea.offset.y)) + ri->renderArea.extent.height;
  }
  {
    // MSAA: the pass's effective count (pipelines must match it) and, for the eye-pass detection,
    // the count the game asked for and the size of its first colour attachment.
    PassSamples ps;
    for (uint32_t i = 0; i < ri->colorAttachmentCount; ++i)
      ps.add(ri->pColorAttachments[i].imageView ? from<ImageView>(ri->pColorAttachments[i].imageView) : nullptr, false);
    if (ri->pDepthAttachment) ps.add(ri->pDepthAttachment->imageView ? from<ImageView>(ri->pDepthAttachment->imageView) : nullptr, true);
    if (ri->pStencilAttachment) ps.add(ri->pStencilAttachment->imageView ? from<ImageView>(ri->pStencilAttachment->imageView) : nullptr, true);
    ps.finish(ri->viewMask, "rendering");
    r.bound.passSamples = ps.effective;
  }
  for (const auto& [src, dst] : forcedPairs) op.resolveCopies.push_back(forcedResolveCopy(src, dst, std::max<uint16_t>(d.layerCount, 1)));
  for (const auto& [src, dst] : intResolves) op.resolveCopies.push_back(sampleZeroResolve(src, dst, std::max<uint16_t>(d.layerCount, 1)));
  r.bound.targetWidth = uint32_t(ri->renderArea.offset.x) + ri->renderArea.extent.width;
  r.bound.targetHeight = uint32_t(ri->renderArea.offset.y) + ri->renderArea.extent.height;
  r.bound.renderArea = ri->renderArea;
  r.bound.fbWidth = r.bound.fbHeight = 0;
  r.bound.fbLayers = ri->layerCount;
  if (ri->colorAttachmentCount && ri->pColorAttachments[0].imageView) {
    const ImageView* v = from<ImageView>(ri->pColorAttachments[0].imageView);
    if (v && v->image)
      r.bound.fbWidth = std::max(1u, v->image->desc.width >> v->range.baseMipLevel),
      r.bound.fbHeight = std::max(1u, v->image->desc.height >> v->range.baseMipLevel);
  }
  d.renderWidth = r.bound.targetWidth, d.renderHeight = r.bound.targetHeight;
  // Attached images and in-pass barriers: draws after a barrier that read an attachment are told apart (makeDraw).
  r.bound.passImages.clear();
  r.bound.localRead = false;
  auto keepImage = [&](const VkRenderingAttachmentInfo* a) {
    if (a && a->imageView && from<ImageView>(a->imageView)->image) r.bound.passImages.push_back(from<ImageView>(a->imageView)->image);
  };
  for (uint32_t i = 0; i < ri->colorAttachmentCount; ++i) keepImage(&ri->pColorAttachments[i]);
  keepImage(ri->pDepthAttachment);
  keepImage(ri->pStencilAttachment);
  r.passBegin = r.ops.size();
  r.ops.emplace_back(std::move(op));
  r.inPass = true;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRendering(VkCommandBuffer cb) {
  Recorder& r = rec(cb);
  r.ops.emplace_back(OpEndPass{});
  r.inPass = false;
  r.passBegin = SIZE_MAX;
  r.bound.passImages.clear();
  r.bound.localRead = false;
}

// VK_KHR_dynamic_rendering_local_read. Attachment locations / input indices: shaders address the attachments by the
// order given at vkCmdBeginRendering, the identity mapping these default to; a remap is logged, not applied. Input
// attachments are read through their descriptors (the image view bound to the input attachment binding), so the index
// mapping has nothing to change.
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetRenderingAttachmentLocationsKHR(VkCommandBuffer,
                                                                               const VkRenderingAttachmentLocationInfoKHR* li) {
  for (uint32_t i = 0; li && li->pColorAttachmentLocations && i < li->colorAttachmentCount; ++i)
    if (li->pColorAttachmentLocations[i] != i && li->pColorAttachmentLocations[i] != VK_ATTACHMENT_UNUSED) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 8) logf("vk.unimpl attachment_location_remap att=%u loc=%u", i, li->pColorAttachmentLocations[i]);
    }
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetRenderingInputAttachmentIndicesKHR(VkCommandBuffer,
                                                                                  const VkRenderingInputAttachmentIndexInfoKHR*) {}

// Private hook of vpgles (reached through vkGetDeviceProcAddr): the pipeline-recipe counters of this device for its
// gles.pso.recipe line. out[0] loaded (recipes read from disk when the device was created), out[1] saved (recipes this run
// added), out[2] prewarmed (recipes ready after the prewarm), out[3] miss_in_game (pipelines the game created that no
// recipe covered, so they compiled at their first draw). All 0 without a recipe store (VP_PIPELINE_CACHE_DIR unset).
VPVK_EXPORT void vpvk_recipe_stats(VkDevice dv, uint32_t out[4]) {
  const Device* d = dv ? from<Device>(dv) : nullptr;
  out[0] = out[1] = out[2] = out[3] = 0;
  if (!d || !d->recipes || !d->recipes->enabled()) return;
  out[0] = d->recipesLoaded, out[1] = uint32_t(d->recipes->storedThisRun()), out[2] = d->recipesPrewarmed.load(),
  out[3] = d->recipesFresh.load();
}

// Private hook of vpgles (reached through vkGetDeviceProcAddr): glInvalidateFramebuffer decides after the begin which
// attachments of the open dynamic-rendering pass are undefined when it ends. `colorMask` (bit per colour slot), `depth`
// and `stencil` are the complete set to drop: they replace an earlier call for this pass (an attachment written again
// is passed as not dropped and stores as the begin said). A combined depth/stencil attachment drops only when every
// aspect it stores is dropped. Resolved, forced-Store and eye attachments keep their store (see OpBeginRendering).
// No-op without an open dynamic-rendering pass.
VPVK_EXPORT void vpvk_cmd_pass_drop(VkCommandBuffer cb, uint32_t colorMask, int depth, int stencil) {
  Recorder& r = rec(cb);
  if (!r.inPass || r.passBegin >= r.ops.size()) return;
  OpBeginRendering* op = std::get_if<OpBeginRendering>(&r.ops[r.passBegin]);
  if (!op) return;
  vpgpu::RenderPassDesc& d = op->desc;
  for (unsigned i = 0; i < d.colors.size() && i < 32; ++i)
    if (op->dropColor >> i & 1u)
      d.colors[i].store = colorMask >> i & 1u ? vpgpu::StoreAction::DontCare : vpgpu::StoreAction::Store;
  if (op->dropDepth) {
    const unsigned want = (depth ? 1u : 0u) | (stencil ? 2u : 0u);
    d.depth.store = (op->dropDepth & ~want) ? vpgpu::StoreAction::Store : vpgpu::StoreAction::DontCare;
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearAttachments(VkCommandBuffer cb, uint32_t n, const VkClearAttachment* a,
                                                             uint32_t nr, const VkClearRect* rects) {
  rec(cb).ops.emplace_back(OpClearAttachments{std::vector<VkClearAttachment>(a, a + n),
                                              std::vector<VkClearRect>(rects, rects + nr)});
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage im, VkImageLayout,
                                                                   const VkClearDepthStencilValue* v, uint32_t n,
                                                                   const VkImageSubresourceRange* ranges) {
  Recorder& r = rec(cb);
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpClearDepth{from<Image>(im), *v, ranges[i]});
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* cbs) {
  Recorder& r = rec(cb);
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpExecute{from<CommandBuffer>(cbs[i])->rec.get()});
}

// =================================================================================================
// Copies
// =================================================================================================
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize off,
                                                         VkDeviceSize size, const void* data) {
  Recorder& r = rec(cb);
  OpUpload u;
  u.src = r.arena.put(r.device, data, size, u.srcOffset);
  u.dst = from<Buffer>(dst), u.dstOffset = off, u.size = size;
  r.ops.emplace_back(u);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src, VkImageLayout, VkBuffer dst,
                                                              uint32_t n, const VkBufferImageCopy* regions) {
  Recorder& r = rec(cb);
  for (uint32_t i = 0; i < n; ++i) {
    VkBufferImageCopy c = regions[i];
    Image* im = planeOf(from<Image>(src), c.imageSubresource.aspectMask);  // a plane of a multi-planar image
    if (im != from<Image>(src)) c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.ops.emplace_back(OpCopyImageToBuffer{im, from<Buffer>(dst), c});
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout, VkImage dst,
                                                      VkImageLayout, uint32_t n, const VkImageCopy* regions) {
  Recorder& r = rec(cb);
  for (uint32_t i = 0; i < n; ++i) {
    VkImageCopy c = regions[i];
    Image* s = planeOf(from<Image>(src), c.srcSubresource.aspectMask);  // planes of multi-planar images
    Image* d = planeOf(from<Image>(dst), c.dstSubresource.aspectMask);
    if (s != from<Image>(src)) c.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if (d != from<Image>(dst)) c.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.ops.emplace_back(OpCopyImage{s, d, c});
  }
}

// A blit region is a plain copy only when the bytes can move unchanged: same format and same size
// (integer formats of one texel size but other channel layouts, RGBA8 -> RGB10A2, convert per
// channel: not raw). Anything else
// (compressed or sRGB source into another format, scaling, mirroring) is a sampled blit drawn by
// the backend, except a mip chain over one image (Unity's GenerateMips), which the backend builds.
namespace {
bool rawBlitCompatible(vpgpu::Format a, vpgpu::Format b) { return a == b; }
}  // namespace

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                                      VkImageLayout dl, uint32_t n, const VkImageBlit* regions,
                                                      VkFilter filter) {
  const Image* s = from<Image>(src);
  const Image* d = from<Image>(dst);
  for (uint32_t i = 0; i < n; ++i) {
    const VkImageBlit& b = regions[i];
    const int sw = b.srcOffsets[1].x - b.srcOffsets[0].x, sh = b.srcOffsets[1].y - b.srcOffsets[0].y;
    const int dw = b.dstOffsets[1].x - b.dstOffsets[0].x, dh = b.dstOffsets[1].y - b.dstOffsets[0].y;
    const int sd = b.srcOffsets[1].z - b.srcOffsets[0].z, dd = b.dstOffsets[1].z - b.dstOffsets[0].z;
    if (!sw || !sh || !dw || !dh) continue;
    if (sw == dw && sh == dh && sd == dd && sw > 0 && sh > 0 && rawBlitCompatible(s->desc.format, d->desc.format) &&
        isSwappedRB(s->info.format) == isSwappedRB(d->info.format)) {
      VkImageCopy c{};
      c.srcSubresource = b.srcSubresource, c.dstSubresource = b.dstSubresource;
      c.srcOffset = b.srcOffsets[0], c.dstOffset = b.dstOffsets[0];
      c.extent = {uint32_t(sw), uint32_t(sh), uint32_t(std::max(1, sd))};
      vkCmdCopyImage(cb, src, sl, dst, dl, 1, &c);
      continue;
    }
    Recorder& r = rec(cb);
    // Mip chain: mip m -> m + 1 of one image (Unity's GenerateMips). One request builds the chain
    // (Metal's box filter: linear only, float/normalized 2D colour formats; nearest, integer and 3D
    // chains are blitted level by level below).
    const vpgpu::FormatInfo& mfi = vpgpu::formatInfo(s->desc.format);
    if (src == dst && b.dstSubresource.mipLevel == b.srcSubresource.mipLevel + 1 && filter == VK_FILTER_LINEAR &&
        !mfi.integer() && !mfi.depth() && !mfi.stencil() && s->desc.type != vpgpu::TextureType::Tex3D) {
      bool repeat = false;
      if (!r.ops.empty())
        if (const auto* prev = std::get_if<OpCopyImage>(&r.ops.back())) repeat = prev->genMips && prev->src == s;
      if (!repeat) {
        OpCopyImage op;
        op.src = op.dst = s, op.genMips = true;
        op.region.srcSubresource = b.srcSubresource;
        r.ops.emplace_back(op);
      }
      continue;
    }
    r.ops.emplace_back(OpBlitImage{s, d, b, filter});
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                                         VkImageLayout dl, uint32_t n, const VkImageResolve* regions) {
  // A source forced to 1x (game MSAA off under VPKTAA), or one the game made 1x, resolves as a plain copy (a Metal resolve of
  // a 1x texture is undefined behaviour).
  if (src && (from<Image>(src)->forcedFrom > 1 || from<Image>(src)->desc.samples <= 1) && regions) {
    for (uint32_t i = 0; i < n; ++i) {
      VkImageCopy c{regions[i].srcSubresource, regions[i].srcOffset, regions[i].dstSubresource, regions[i].dstOffset,
                    regions[i].extent};
      vkCmdCopyImage(cb, src, sl, dst, dl, 1, &c);
    }
    return;
  }
  Recorder& r = rec(cb);
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpResolveImage{from<Image>(src), from<Image>(dst), regions[i]});
}

// =================================================================================================
// Events in command buffers, queries (no GPU queries: zero results)
// =================================================================================================
// Set / reset when the submission finishes (vkQueueSubmit stamps the frame; Event).
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags) {
  rec(cb).ops.emplace_back(OpEvent{from<Event>(e), true});
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags) {
  rec(cb).ops.emplace_back(OpEvent{from<Event>(e), false});
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(VkCommandBuffer cb, uint32_t, const VkEvent*, VkPipelineStageFlags,
                                                       VkPipelineStageFlags, uint32_t, const VkMemoryBarrier*, uint32_t,
                                                       const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
  ++rec(cb).barriers;
}

// ---- Queries -------------------------------------------------------------------------------
// Occlusion: Metal visibility results (counting mode) in one device-wide buffer; a pool takes a
// range of 8-byte counters. Timestamps and pipeline statistics are not measured: timestamps read
// 0, always available.
namespace {
constexpr uint32_t kVisibilitySlots = 65536;  // 512 KiB per device

// VPVK_OCCLUSION=0: occlusion queries are not measured (every one reads visible: the pool stays unbacked).
// Unset or anything else: measured. Read per pool, so a host may set it before the device exists.
bool occlusionOn() {
  const char* e = std::getenv("VPVK_OCCLUSION");
  return !(e && *e == '0');
}

uint32_t allocVisibility(Device* d, uint32_t count) {
  Device::Visibility& v = d->visibility;
  std::lock_guard<std::mutex> l(v.m);
  if (!v.id) {
    const size_t bytes = size_t(kVisibilitySlots) * 8;
    v.host = static_cast<uint8_t*>(std::aligned_alloc(16384, bytes));
    if (!v.host) return UINT32_MAX;
    std::memset(v.host, 0, bytes);
    vpgpu::BufferDesc bd;
    bd.size = bytes;
    bd.guestMemory = v.host;
    bd.label = "vk.visibility";
    {
      std::lock_guard<std::mutex> lock(d->submitMutex);
      v.id = d->gpu->createBuffer(bd);
    }
    if (!v.id) {
      std::free(v.host);
      v.host = nullptr;
      return UINT32_MAX;
    }
  }
  uint32_t at = 0;  // first fit between the sorted ranges
  auto it = v.used.begin();
  for (; it != v.used.end(); ++it) {
    if (it->first - at >= count) break;
    at = it->first + it->second;
  }
  if (count > kVisibilitySlots || at > kVisibilitySlots - count) return UINT32_MAX;
  v.used.insert(it, {at, count});
  return at;
}
}  // namespace

void vpvk::releaseVisibility(Device* d) {
  Device::Visibility& v = d->visibility;
  std::lock_guard<std::mutex> l(v.m);
  if (v.id) {
    std::lock_guard<std::mutex> lock(d->submitMutex);
    d->gpu->destroy(v.id);
  }
  v.id = {};
  std::free(v.host);
  v.host = nullptr;
  v.used.clear();
}

void vpvk::applyQueryEvents(const std::vector<TranslateStats::QueryEvent>& events, uint64_t frame) {
  for (const TranslateStats::QueryEvent& e : events) {
    std::lock_guard<std::mutex> l(e.pool->m);
    for (uint32_t i = 0; i < e.count && e.first + i < e.pool->frame.size(); ++i)
      e.pool->frame[e.first + i] = e.end ? frame + 1 : 0;  // +1: 0 stays "not ended" (frame may be 0)
  }
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice dev, const VkQueryPoolCreateInfo* ci,
                                                             const VkAllocationCallbacks*, VkQueryPool* out) {
  auto* q = new QueryPoolObj;
  q->type = ci->queryType, q->count = ci->queryCount;
  q->device = from<Device>(dev);
  q->frame.assign(ci->queryCount, 0);
  if (((q->type == VK_QUERY_TYPE_OCCLUSION && occlusionOn()) || q->type == VK_QUERY_TYPE_TIMESTAMP) && q->count &&
      q->device && q->device->gpu) {
    q->base = allocVisibility(q->device, q->count);
    if (q->base == UINT32_MAX) logf("vk.err what=query_pool_unbacked count=%u (occlusion reads as visible)", q->count);
  }
  pendingTrack(q);
  *out = to<VkQueryPool>(q);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice, VkQueryPool qp, const VkAllocationCallbacks*) {
  QueryPoolObj* q = from<QueryPoolObj>(qp);
  if (!q) return;
  if (q->base != UINT32_MAX && q->device) {
    Device::Visibility& v = q->device->visibility;
    std::lock_guard<std::mutex> l(v.m);
    for (auto it = v.used.begin(); it != v.used.end(); ++it)
      if (it->first == q->base) {
        v.used.erase(it);
        break;
      }
  }
  pendingUntrack(q);
  delete q;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice, VkQueryPool qp, uint32_t first, uint32_t n,
                                                                 size_t size, void* data, VkDeviceSize stride,
                                                                 VkQueryResultFlags flags) {
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  const bool wide = flags & VK_QUERY_RESULT_64_BIT;
  const size_t word = wide ? 8 : 4;
  const bool withAvail = flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
  const bool backed = (p->type == VK_QUERY_TYPE_OCCLUSION || p->type == VK_QUERY_TYPE_TIMESTAMP) &&
                      p->base != UINT32_MAX && p->device && p->device->visibility.host;
  auto put = [&](uint8_t* at, uint64_t v) {
    if (wide) std::memcpy(at, &v, 8);
    else {
      const uint32_t x = v > UINT32_MAX ? UINT32_MAX : uint32_t(v);
      std::memcpy(at, &x, 4);
    }
  };
  bool notReady = false;
  for (uint32_t i = 0; i < n; ++i) {
    uint8_t* at = static_cast<uint8_t*>(data) + size_t(i) * stride;
    if (size_t(i) * stride + word * (withAvail ? 2 : 1) > size) break;
    const uint32_t q = first + i;
    if (!backed) {  // see vkCreateQueryPool: visible / time 0, available
      put(at, p->type == VK_QUERY_TYPE_OCCLUSION ? 1 : 0);
      if (withAvail) put(at + word, 1);
      continue;
    }
    uint64_t f = 0;
    {
      std::lock_guard<std::mutex> l(p->m);
      f = q < p->frame.size() ? p->frame[q] : 0;
    }
    bool avail = false;
    if (f) {
      const auto w = p->device->platform->waitFrames(f - 1, (flags & VK_QUERY_RESULT_WAIT_BIT) ? UINT64_MAX : 0);
      if (w == Platform::WaitResult::Lost) return VK_ERROR_DEVICE_LOST;
      avail = w == Platform::WaitResult::Done;
    }
    // (WAIT on a query never ended would hang forever in Vulkan; it reports not ready instead.)
    if (avail || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
      uint64_t v;
      std::memcpy(&v, p->device->visibility.host + size_t(p->base + q) * 8, 8);
      put(at, v);
    }
    if (withAvail) put(at + word, avail ? 1 : 0);
    if (!avail) notReady = true, p->lastNotReady = q;
  }
  p->poll.note(notReady ? VK_NOT_READY : VK_SUCCESS);  // vk.pending (pending.cpp)
  return notReady ? VK_NOT_READY : VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkResetQueryPool(VkDevice, VkQueryPool qp, uint32_t first, uint32_t n) {
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  std::lock_guard<std::mutex> l(p->m);
  for (uint32_t i = 0; i < n && first + i < p->frame.size(); ++i) p->frame[first + i] = 0;
  if (p->base != UINT32_MAX && p->device && p->device->visibility.host && first < p->count)
    std::memset(p->device->visibility.host + size_t(p->base + first) * 8, 0, size_t(std::min(n, p->count - first)) * 8);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer cb, VkQueryPool qp, uint32_t first, uint32_t n) {
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  if (first >= p->count) return;
  rec(cb).ops.emplace_back(OpQueryReset{p, first, std::min(n, p->count - first)});
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(VkCommandBuffer cb, VkQueryPool qp, uint32_t q, VkQueryControlFlags) {
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  if (p->type == VK_QUERY_TYPE_OCCLUSION && p->base != UINT32_MAX && q < p->count) {
    rec(cb).bound.queryPool = p;
    rec(cb).bound.query = q;
  }
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(VkCommandBuffer cb, VkQueryPool qp, uint32_t q) {
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  Recorder& r = rec(cb);
  if (r.bound.queryPool == p && r.bound.query == q) r.bound.queryPool = nullptr;
  if (q < p->count) r.ops.emplace_back(OpQueryEnd{p, q});
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits, VkQueryPool qp,
                                                           uint32_t q) {
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  if (q < p->count) rec(cb).ops.emplace_back(OpQueryEnd{p, q, true});
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool qp, uint32_t first, uint32_t n,
                                                                 VkBuffer dst, VkDeviceSize offset, VkDeviceSize stride,
                                                                 VkQueryResultFlags flags) {
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  if (first >= p->count || !dst) return;
  Recorder& r = rec(cb);
  OpQueryCopy op;
  op.pool = p, op.first = first, op.count = std::min(n, p->count - first);
  op.dst = from<Buffer>(dst), op.offset = offset, op.stride = stride, op.flags = flags;
  static const uint64_t kConstants[2] = {0, 1};
  op.constants = r.arena.put(r.device, kConstants, sizeof kConstants, op.constantsOffset);
  r.ops.emplace_back(std::move(op));
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkTrimCommandPool(VkDevice, VkCommandPool, VkCommandPoolTrimFlags) {}

// =================================================================================================
// Proc-address table of this file (entry.cpp's lookup asks it before giving up)
// =================================================================================================
namespace vpvk {
PFN_vkVoidFunction lookupPhase3(const char* name) {
  struct E {
    const char* name;
    PFN_vkVoidFunction fn;
  };
#define F(f) {#f, reinterpret_cast<PFN_vkVoidFunction>(&f)}
  static const E table[] = {
      F(vkCreateShaderModule), F(vkDestroyShaderModule), F(vkCreateDescriptorSetLayout), F(vkDestroyDescriptorSetLayout),
      F(vkGetDescriptorSetLayoutSupport), F(vkCreatePipelineLayout), F(vkDestroyPipelineLayout), F(vkCreateDescriptorPool),
      F(vkResetDescriptorPool), F(vkDestroyDescriptorPool), F(vkAllocateDescriptorSets), F(vkFreeDescriptorSets),
      F(vkUpdateDescriptorSets), F(vkCreateDescriptorUpdateTemplate), F(vkDestroyDescriptorUpdateTemplate),
      F(vkUpdateDescriptorSetWithTemplate), F(vkCreateGraphicsPipelines), F(vkCreateComputePipelines), F(vkDestroyPipeline),
      F(vkCreatePipelineCache), F(vkDestroyPipelineCache), F(vkGetPipelineCacheData), F(vkMergePipelineCaches),
      F(vkCmdBindPipeline), F(vkCmdBindDescriptorSets), F(vkCmdPushDescriptorSetKHR), F(vkCmdPushDescriptorSetWithTemplateKHR), F(vkCmdPushConstants), F(vkCmdBindVertexBuffers),
      F(vkCmdBindVertexBuffers2), F(vkCmdBindIndexBuffer), F(vkCmdSetViewport), F(vkCmdSetScissor),
      F(vkCmdSetStencilReference), F(vkCmdSetBlendConstants), F(vkCmdSetLineWidth), F(vkCmdSetDepthBias),
      F(vkCmdSetDepthBounds), F(vkCmdSetViewportWithCount), F(vkCmdSetScissorWithCount), F(vkCmdSetCullMode),
      F(vkCmdSetFrontFace), F(vkCmdSetPrimitiveTopology), F(vkCmdSetDepthTestEnable), F(vkCmdSetDepthWriteEnable),
      F(vkCmdSetDepthCompareOp), F(vkCmdSetDepthBoundsTestEnable), F(vkCmdSetStencilTestEnable), F(vkCmdSetStencilOp),
      F(vkCmdSetRasterizerDiscardEnable), F(vkCmdSetDepthBiasEnable), F(vkCmdSetPrimitiveRestartEnable),
      F(vkCmdSetLogicOpEXT), F(vkCmdSetPatchControlPointsEXT), F(vkCmdSetColorWriteMaskEXT), F(vkCmdSetStencilCompareMask), F(vkCmdSetStencilWriteMask), F(vkCmdDraw), F(vkCmdDrawIndexed),
      F(vkCmdDrawIndirect), F(vkCmdDrawIndexedIndirect), F(vkCmdDispatch), F(vkCmdDispatchIndirect), F(vkCmdNextSubpass),
      F(vkCmdDrawIndirectCount), F(vkCmdDrawIndexedIndirectCount),
      {"vkCmdDrawIndirectCountKHR", reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDrawIndirectCount)},
      {"vkCmdDrawIndexedIndirectCountKHR", reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDrawIndexedIndirectCount)},
      F(vkCmdNextSubpass2), F(vkCmdBeginRendering), F(vkCmdEndRendering), F(vkCmdSetRenderingAttachmentLocationsKHR),
      F(vkCmdSetRenderingInputAttachmentIndicesKHR), F(vpvk_cmd_pass_drop), F(vpvk_recipe_stats), F(vkCmdClearAttachments),
      F(vkCmdClearDepthStencilImage), F(vkCmdExecuteCommands), F(vkCmdUpdateBuffer), F(vkCmdCopyImageToBuffer),
      F(vkCmdCopyImage), F(vkCmdBlitImage), F(vkCmdResolveImage), F(vkCmdSetEvent), F(vkCmdResetEvent), F(vkCmdWaitEvents),
      F(vkCreateQueryPool), F(vkDestroyQueryPool), F(vkGetQueryPoolResults), F(vkResetQueryPool), F(vkCmdResetQueryPool),
      F(vkCmdBeginQuery), F(vkCmdEndQuery), F(vkCmdWriteTimestamp), F(vkCmdCopyQueryPoolResults), F(vkTrimCommandPool),
  };
#undef F
  static const std::pair<const char*, const char*> aliases[] = {
      {"vkCreateDescriptorUpdateTemplateKHR", "vkCreateDescriptorUpdateTemplate"},
      {"vkDestroyDescriptorUpdateTemplateKHR", "vkDestroyDescriptorUpdateTemplate"},
      {"vkUpdateDescriptorSetWithTemplateKHR", "vkUpdateDescriptorSetWithTemplate"},
      {"vkGetDescriptorSetLayoutSupportKHR", "vkGetDescriptorSetLayoutSupport"},
      {"vkCmdNextSubpass2KHR", "vkCmdNextSubpass2"},
      {"vkCmdBeginRenderingKHR", "vkCmdBeginRendering"},
      {"vkCmdEndRenderingKHR", "vkCmdEndRendering"},
      {"vkCmdBindVertexBuffers2EXT", "vkCmdBindVertexBuffers2"},
      {"vkCmdSetViewportWithCountEXT", "vkCmdSetViewportWithCount"},
      {"vkCmdSetScissorWithCountEXT", "vkCmdSetScissorWithCount"},
      {"vkCmdSetCullModeEXT", "vkCmdSetCullMode"},
      {"vkCmdSetFrontFaceEXT", "vkCmdSetFrontFace"},
      {"vkCmdSetPrimitiveTopologyEXT", "vkCmdSetPrimitiveTopology"},
      {"vkCmdSetDepthTestEnableEXT", "vkCmdSetDepthTestEnable"},
      {"vkCmdSetDepthWriteEnableEXT", "vkCmdSetDepthWriteEnable"},
      {"vkCmdSetDepthCompareOpEXT", "vkCmdSetDepthCompareOp"},
      {"vkCmdSetDepthBoundsTestEnableEXT", "vkCmdSetDepthBoundsTestEnable"},
      {"vkCmdSetStencilTestEnableEXT", "vkCmdSetStencilTestEnable"},
      {"vkCmdSetStencilOpEXT", "vkCmdSetStencilOp"},
      {"vkCmdSetRasterizerDiscardEnableEXT", "vkCmdSetRasterizerDiscardEnable"},
      {"vkCmdSetDepthBiasEnableEXT", "vkCmdSetDepthBiasEnable"},
      {"vkCmdSetPrimitiveRestartEnableEXT", "vkCmdSetPrimitiveRestartEnable"},
      {"vkResetQueryPoolEXT", "vkResetQueryPool"},
      {"vkTrimCommandPoolKHR", "vkTrimCommandPool"},
  };
  for (const auto& [alias, real] : aliases)
    if (!std::strcmp(name, alias)) name = real;
  for (const E& e : table)
    if (!std::strcmp(e.name, name)) return e.fn;
  return nullptr;
}
}  // namespace vpvk
