// Trimmed capture (VPCAP, include/vpvk/vpcap.h): a GFXReconstruct-style capture of a few frames of the Vulkan calls the
// game (or vpgles, which goes through the same entries) makes into this driver.
//
// How it works
//  - vpvk::lookup (entry.cpp) hands out the hooks below instead of the driver's own entry points (captureWrap). A hook
//    costs one relaxed atomic load while neither tracking nor a capture is on.
//  - Tracking (VPVK_CAPTURE=1 / vpvk_capture_track) keeps, per live object, the serialized creation call (a complete VPCAP
//    call record: the create info, the SPIR-V of shader modules, the memory binding), plus the ids of the objects it
//    references. A destroyed object stays while a live one still references it (a layout or module destroyed after the
//    pipelines were made), as in any trimmed capture. Costs memory (the create infos of everything alive), not time.
//  - A capture starts when the next frame boundary (a vkQueueSubmit that carries command buffers) returns: the GPU is
//    drained, the STATE section is written (every tracked object in creation order, then the current contents of the descriptor
//    sets, read from the driver's own descriptor state, so templates and copies are covered) and from then on every hooked
//    call is appended to the FRAME section. Command buffers are recorded only if their vkBeginCommandBuffer happened
//    during the capture (one recorded earlier has no content in the file and is dropped from the recorded submit).
//  - Buffer contents: at each vkQueueSubmit the buffers its command buffers reference (bound vertex / index / indirect /
//    copy buffers, and the buffers of the descriptor sets they bind, read from the driver's descriptor state) are diffed
//    against what the file already holds: one 64-bit hash per 4 KiB page of the memory they live in. Pages that differ
//    (first sight: pages that are not all zero) are written as vpcapMemoryData right before the submit, so the host writes
//    through vkMapMemory between frames and the GPU's own writes of an earlier frame are both captured. The GPU is
//    drained after every captured submit so the contents read are consistent (a captured frame is serialised).
//  - Image contents: at the vkQueueSubmit where an image is first read in the capture (a render pass attachment with loadOp
//    LOAD, an image in a descriptor set the submit binds, the source of a copy / blit / resolve), with the GPU idle, every
//    mip, layer and aspect of it is read (Platform::readTexture) and written as vpcapImageData right before that submit
//    (all-zero subresources are left out: a new image reads as zero). Images without memory (transient / lazily allocated /
//    still on tile), multisample ones and the depth of D24S8 are not read, only counted. An image is read once per capture.
//    Not covered: an image only partly overwritten by a copy / blit before it is ever read.
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "vpvk.h"
#include "vpvk/vpcap_calls.h"

namespace vpvk {
namespace {

using namespace vpcap;

constexpr uint32_t kTrack = 1, kCapture = 2;
constexpr uint64_t kPage = 4096;

enum class Kind : uint8_t {
  None, Instance, Device, Queue, Memory, Buffer, BufferView, Image, ImageView, Sampler, SetLayout, PipelineLayout,
  DescPool, DescSet, RenderPass, Framebuffer, Shader, Pipeline, CmdPool, CmdBuf, Fence, Semaphore, Event, QueryPool
};

// What a call does to the object registry. out / obj / bind: index of the argument (output handle / destroyed handle /
// handle the call binds memory to), -1 = none.
struct Meta {
  Kind kind = Kind::None;
  int out = -1, obj = -1, bind = -1;
};
constexpr Meta cr(Kind k, int out) { return {k, out, -1, -1}; }
constexpr Meta de(int obj) { return {Kind::None, -1, obj, -1}; }
constexpr Meta metaOf(uint16_t i) {
  switch (i) {
    case id::vkCreateInstance: return cr(Kind::Instance, 2);
    case id::vkDestroyInstance: return de(0);
    case id::vkCreateDevice: return cr(Kind::Device, 3);
    case id::vkDestroyDevice: return de(0);
    case id::vkGetDeviceQueue: return cr(Kind::Queue, 3);
    case id::vkAllocateMemory: return cr(Kind::Memory, 3);
    case id::vkFreeMemory: return de(1);
    case id::vkCreateBuffer: return cr(Kind::Buffer, 3);
    case id::vkDestroyBuffer: return de(1);
    case id::vkBindBufferMemory: return {Kind::None, -1, -1, 1};
    case id::vkCreateBufferView: return cr(Kind::BufferView, 3);
    case id::vkDestroyBufferView: return de(1);
    case id::vkCreateImage: return cr(Kind::Image, 3);
    case id::vkDestroyImage: return de(1);
    case id::vkBindImageMemory: return {Kind::None, -1, -1, 1};
    case id::vkCreateImageView: return cr(Kind::ImageView, 3);
    case id::vkDestroyImageView: return de(1);
    case id::vkCreateSampler: return cr(Kind::Sampler, 3);
    case id::vkDestroySampler: return de(1);
    case id::vkCreateDescriptorSetLayout: return cr(Kind::SetLayout, 3);
    case id::vkDestroyDescriptorSetLayout: return de(1);
    case id::vkCreatePipelineLayout: return cr(Kind::PipelineLayout, 3);
    case id::vkDestroyPipelineLayout: return de(1);
    case id::vkCreateDescriptorPool: return cr(Kind::DescPool, 3);
    case id::vkDestroyDescriptorPool: return de(1);
    case id::vkCreateRenderPass:
    case id::vkCreateRenderPass2: return cr(Kind::RenderPass, 3);
    case id::vkDestroyRenderPass: return de(1);
    case id::vkCreateFramebuffer: return cr(Kind::Framebuffer, 3);
    case id::vkDestroyFramebuffer: return de(1);
    case id::vkCreateShaderModule: return cr(Kind::Shader, 3);
    case id::vkDestroyShaderModule: return de(1);
    case id::vkDestroyPipeline: return de(1);
    case id::vkCreateCommandPool: return cr(Kind::CmdPool, 3);
    case id::vkDestroyCommandPool: return de(1);
    case id::vkCreateFence: return cr(Kind::Fence, 3);
    case id::vkDestroyFence: return de(1);
    case id::vkCreateSemaphore: return cr(Kind::Semaphore, 3);
    case id::vkDestroySemaphore: return de(1);
    case id::vkCreateEvent: return cr(Kind::Event, 3);
    case id::vkDestroyEvent: return de(1);
    case id::vkCreateQueryPool: return cr(Kind::QueryPool, 3);
    case id::vkDestroyQueryPool: return de(1);
    default: return {};
  }
}

struct Rec {
  uint64_t id = 0;
  const void* handle = nullptr;
  Kind kind = Kind::None;
  uint64_t seq = 0, bindSeq = 0, dev = 0, parent = 0;
  std::vector<uint8_t> create, bind;  // complete call records
  std::vector<uint64_t> deps;         // ids the blobs reference (each holds one `refs` of its target)
  uint32_t refs = 0;                  // live blobs of other objects referencing this one
  bool destroyed = false;
  uint32_t epoch = 0;                 // CmdBuf: recorded under capture number `epoch`
  std::vector<uint64_t> cbRefs;       // CmdBuf: ids the recorded commands touched
  std::vector<uint64_t> cbImgReads;   // CmdBuf: ids of the images the recorded commands read (loads, copy / blit sources)
};

struct Capture {
  std::unique_ptr<Writer> w;
  std::string path;
  uint32_t target = 0, frames = 0;
  uint64_t submits = 0, stateBytes = 0, startSubmit = 0;
  uint64_t memPages = 0, memBytes = 0, droppedNextAtStart = 0;
  uint32_t cbsDropped = 0;
  bool instanceTracked = false;  // false: tracking began after the instance (the state has no vkCreateInstance)
  // The capture is of one device: the one that made the boundary submit (and its instance). What other devices (the game's
  // and vpgles' internal one live side by side) create, call or submit is left out and counted.
  const void* devHandle = nullptr;  // that VkDevice
  uint64_t devId = 0, instId = 0, foreignCalls = 0, foreignSubmits = 0;
  // Image contents over the per-submit budget wait here for the next submit.
  std::vector<uint64_t> deferredImages;
  uint32_t imgDeferred = 0;
  std::map<std::string, uint32_t> unhookedUsed;  // calls with no recording hook that were made during the capture
  std::set<std::string> invalid;                 // why the file does not reproduce the frames exactly
  std::unordered_set<uint64_t> images;      // image ids the frames touched
  std::unordered_set<uint64_t> imagesDone;  // image ids whose contents were looked at (written, or counted as unreadable)
  // Image contents: images looked at, subresources / bytes written, subresources all zero (left out), and the images that
  // could not be read: no memory, a format the backend cannot copy out, a failed copy.
  uint64_t imgBytes = 0;
  uint32_t imgNew = 0, imgSubres = 0, imgZero = 0, imgMemoryless = 0, imgUnsupported = 0, imgFailed = 0;
  std::unordered_map<uint64_t, std::vector<uint64_t>> shadow;  // memory id -> page hashes (0 = unseen, assumed zero)
};

struct G {
  std::atomic<uint32_t> mode{0};
  std::atomic<bool> pending{false};
  std::mutex mu;
  // The descriptor tables of the driver have no lock of their own: vkUpdateDescriptorSets (and the template update) run under
  // a shared hold of this, the snapshots of a set's contents under an exclusive one (taken with mu held, never the reverse).
  std::shared_mutex setLock;
  std::unordered_map<const void*, Rec> recs;
  std::unordered_map<uint64_t, const void*> byId;
  uint64_t nextId = 1, seq = 1;
  uint64_t droppedNext = 0;  // pNext structs outside VPCAP_PNEXT, in every call serialized (state included)
  uint32_t epoch = 0;
  Capture cap;
  uint32_t reqFrames = 1;
  std::string reqPath;
  std::set<std::string> unhooked;  // resolved by the app, no hook
  uint64_t frameCount = 0;         // submits with command buffers since tracking began (VPVK_CAPTURE_FRAME)
  uint64_t autoFrame = 0;
  uint32_t autoFrames = 1;
  std::string autoDir;
};
G& g() {
  static G s;
  return s;
}

uint32_t threadIndex() {
  static std::atomic<uint32_t> next{1};
  thread_local uint32_t t = next.fetch_add(1);
  return t;
}

Rec& recOf(const void* h) {
  G& s = g();
  auto it = s.recs.find(h);
  if (it != s.recs.end()) return it->second;
  Rec& r = s.recs[h];
  r.id = s.nextId++, r.handle = h;
  s.byId[r.id] = h;
  return r;
}
// The record of an object a call just created at `h`. A destroyed object that live blobs still reference (a zombie) may
// have had this very address: it moves under a key of its own so the new object does not overwrite it.
Rec& freshRec(const void* h) {
  G& s = g();
  auto it = s.recs.find(h);
  if (it != s.recs.end() && it->second.destroyed) {
    Rec z = std::move(it->second);
    s.recs.erase(it);
    z.handle = reinterpret_cast<const void*>(static_cast<uintptr_t>(z.id) | (uintptr_t(1) << 62));
    s.byId[z.id] = z.handle;
    s.recs.emplace(z.handle, std::move(z));
  }
  return recOf(h);
}
Rec* find(uint64_t id) {
  G& s = g();
  auto it = s.byId.find(id);
  if (it == s.byId.end()) return nullptr;
  auto r = s.recs.find(it->second);
  return r == s.recs.end() ? nullptr : &r->second;
}
// A tracked, not destroyed object of that kind lives at `h`. The kind matters: the address of a destroyed object may have
// been reused by one of another kind, and a pointer held by a descriptor or a framebuffer is only safe to follow when the
// registry says what is behind it.
bool live(const void* h, Kind k) {
  if (!h) return false;
  auto it = g().recs.find(h);
  return it != g().recs.end() && !it->second.destroyed && it->second.kind == k;
}

struct Ids final : HandleIds {
  std::vector<uint64_t> touched;
  uint64_t idOf(const void* h) override {
    if (!h) return 0;
    const uint64_t i = recOf(h).id;
    touched.push_back(i);
    return i;
  }
};

void maybeFree(Rec& r) {
  G& s = g();
  if (!r.destroyed || r.refs) return;
  const std::vector<uint64_t> deps = std::move(r.deps);
  const void* h = r.handle;
  s.byId.erase(r.id);
  s.recs.erase(h);
  for (uint64_t d : deps)
    if (Rec* t = find(d)) {
      if (t->refs) --t->refs;
      maybeFree(*t);
    }
}

// The object was destroyed (or freed with its pool): the registry keeps it while a live blob references it.
void retire(const void* h) {
  G& s = g();
  auto it = s.recs.find(h);
  if (it == s.recs.end()) return;
  Rec& r = it->second;
  if (r.destroyed) return;
  r.destroyed = true;
  if (r.kind == Kind::DescPool || r.kind == Kind::CmdPool) {
    std::vector<const void*> kids;
    for (auto& [k, c] : s.recs)
      if (c.parent == r.id && !c.destroyed) kids.push_back(k);
    for (const void* k : kids) retire(k);
    auto again = s.recs.find(h);  // the last child may have taken the pool with it
    if (again != s.recs.end()) maybeFree(again->second);
    return;
  }
  if (r.kind == Kind::Device) {  // everything of the device goes with it
    std::vector<const void*> all;
    for (auto& [k, c] : s.recs)
      if (c.dev == r.id && k != h) all.push_back(k);
    for (const void* k : all) {
      auto i = s.recs.find(k);
      if (i == s.recs.end()) continue;
      s.byId.erase(i->second.id);
      s.recs.erase(i);
    }
    s.byId.erase(r.id);
    s.recs.erase(h);
    return;
  }
  maybeFree(r);
}

void addDeps(Rec& r, const std::vector<uint64_t>& touched) {
  for (uint64_t d : touched) {
    if (d == r.id) continue;
    Rec* t = find(d);
    if (!t || t->kind == Kind::Device || t->kind == Kind::Instance || t->kind == Kind::None) continue;
    if (std::find(r.deps.begin(), r.deps.end(), d) != r.deps.end()) continue;
    r.deps.push_back(d);
    ++t->refs;
  }
}

std::vector<uint8_t> pack(uint16_t callId, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> v(sizeof(CallHeader) + payload.size());
  CallHeader h{callId, 0, threadIndex(), static_cast<uint32_t>(payload.size())};
  std::memcpy(v.data(), &h, sizeof h);
  if (!payload.empty()) std::memcpy(v.data() + sizeof h, payload.data(), payload.size());
  return v;
}

void appendFrame(const std::vector<uint8_t>& rec) {
  Capture& c = g().cap;
  c.w->begin(Section::Frame);
  c.w->append(rec.data(), rec.size());
}

bool cbOk(const void* cb) {
  auto it = g().recs.find(cb);
  return it != g().recs.end() && it->second.epoch == g().epoch && it->second.dev == g().cap.devId;
}

// ---- the images a command reads (g().mu held) ----
// Touching is not reading: what a command reads is what the file must hold from before the capture (the loads of a render
// pass, the source of a copy). The images of descriptor sets are resolved at the submit (resolveRef).
void noteRead(Rec& cb, const Image* im) {
  if (im && live(im, Kind::Image)) cb.cbImgReads.push_back(recOf(im).id);
}
void noteReadView(Rec& cb, VkImageView v) {
  if (v && live(v, Kind::ImageView)) noteRead(cb, from<ImageView>(v)->image);
}
// loadOp rules the colour / depth aspects, stencilLoadOp the stencil one (the other is ignored for a format without it).
bool loadsContents(const VkAttachmentDescription2& d) {
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(toCoreFormat(d.format));
  return (!(fi.stencil() && !fi.depth()) && d.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) ||
         (fi.stencil() && d.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
}
void noteRenderPass(Rec& cb, const VkRenderPassBeginInfo* bi) {
  if (!bi || !live(bi->renderPass, Kind::RenderPass) || !live(bi->framebuffer, Kind::Framebuffer)) return;
  const auto* rp = from<RenderPassObj>(bi->renderPass);
  const auto* fb = from<Framebuffer>(bi->framebuffer);
  for (size_t k = 0; k < fb->views.size() && k < rp->attachments.size(); ++k) {
    if (loadsContents(rp->attachments[k]) && live(fb->views[k], Kind::ImageView)) noteRead(cb, fb->views[k]->image);
  }
}
void noteAttachment(Rec& cb, const VkRenderingAttachmentInfo* ai) {
  if (ai && ai->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) noteReadView(cb, ai->imageView);
}
template <uint16_t ID, class A>
void noteReads(Rec& cb, const A& a) {
  if constexpr (ID == id::vkCmdBeginRenderPass || ID == id::vkCmdBeginRenderPass2) {
    noteRenderPass(cb, std::get<1>(a));
  } else if constexpr (ID == id::vkCmdBeginRendering) {
    if (const VkRenderingInfo* ri = std::get<1>(a)) {
      for (uint32_t i = 0; i < ri->colorAttachmentCount; ++i) noteAttachment(cb, ri->pColorAttachments + i);
      noteAttachment(cb, ri->pDepthAttachment);
      noteAttachment(cb, ri->pStencilAttachment);
    }
  } else if constexpr (ID == id::vkCmdCopyImage || ID == id::vkCmdCopyImageToBuffer || ID == id::vkCmdBlitImage ||
                       ID == id::vkCmdResolveImage) {
    noteRead(cb, from<Image>(std::get<1>(a)));  // the source image
  } else if constexpr (ID == id::vkCmdCopyImage2 || ID == id::vkCmdCopyImageToBuffer2 || ID == id::vkCmdBlitImage2 ||
                       ID == id::vkCmdResolveImage2) {
    if (const auto* info = std::get<1>(a)) noteRead(cb, from<Image>(info->srcImage));
  }
}

// The file does not reproduce the captured frames exactly (g().mu held, a capture running).
void markInvalid(const std::string& why) { g().cap.invalid.insert(why); }

// ---- tracking + recording of one call (g().mu held) ----
// Whether a call belongs to the captured device. The first object it touches that is bound to a device decides; a call that
// makes an instance or a device is never part of it (the captured ones exist before the capture starts).
template <uint16_t ID>
bool ofCapturedDevice(const std::vector<uint64_t>& touched) {
  const Capture& c = g().cap;
  for (uint64_t t : touched) {
    const Rec* r = find(t);
    if (!r) continue;
    if (r->kind == Kind::Instance) return r->id == c.instId;
    const uint64_t d = r->kind == Kind::Device ? r->id : r->dev;
    if (d) return d == c.devId;
  }
  return !(ID == id::vkCreateInstance || ID == id::vkCreateDevice);
}

// Appends the record of a call to the FRAME section when the capture runs and the call is of its device.
template <uint16_t ID>
void appendIfCaptured(const std::vector<uint8_t>& rec, const std::vector<uint64_t>& touched) {
  G& s = g();
  if (!(s.mode.load(std::memory_order_relaxed) & kCapture)) return;
  if (ofCapturedDevice<ID>(touched)) appendFrame(rec);
  else ++s.cap.foreignCalls;
}

template <uint16_t ID, class A>
void emit(A& a) {
  G& s = g();
  constexpr Meta M = metaOf(ID);
  if constexpr (M.out >= 0) freshRec(*std::get<M.out>(a));  // before the id is written into the record
  Ids ids;
  Cod c(&ids);
  CallDef<ID>::cod(c, a);
  s.droppedNext += c.droppedNext;
  const std::vector<uint8_t> rec = pack(ID, c.out);
  appendIfCaptured<ID>(rec, ids.touched);
  if constexpr (ID >= 0x0200) {  // commands: what the command buffer touches, for the contents resolved at submit
    Rec& cb = recOf(std::get<0>(a));
    for (uint64_t t : ids.touched)
      if (t != cb.id) cb.cbRefs.push_back(t);
    noteReads<ID>(cb, a);
  }
  if constexpr (M.out >= 0) {
    Rec& r = recOf(*std::get<M.out>(a));
    r.kind = M.kind, r.seq = s.seq++, r.create = rec;
    for (uint64_t t : ids.touched)
      if (Rec* d = find(t); d && d->kind == Kind::Device && ID != id::vkCreateDevice) r.dev = d->id;
    if constexpr (ID == id::vkCreateDevice) r.dev = r.id;
    addDeps(r, ids.touched);
  }
  if constexpr (M.bind >= 0) {
    Rec& r = recOf(std::get<M.bind>(a));
    r.bindSeq = s.seq++, r.bind = rec;
    addDeps(r, ids.touched);
  }
  if constexpr (ID == id::vkFreeMemory) {
    s.cap.shadow.erase(recOf(std::get<1>(a)).id);
  }
  if constexpr (M.obj >= 0) retire(std::get<M.obj>(a));
  if constexpr (ID == id::vkFreeCommandBuffers) {
    const auto n = std::get<2>(a);
    for (uint32_t i = 0; i < n; ++i) retire(std::get<3>(a)[i]);
  }
  if constexpr (ID == id::vkFreeDescriptorSets) {
    const auto n = std::get<2>(a);
    for (uint32_t i = 0; i < n; ++i) retire(std::get<3>(a)[i]);
  }
  if constexpr (ID == id::vkResetDescriptorPool) {
    const uint64_t pool = recOf(std::get<1>(a)).id;
    std::vector<const void*> kids;
    for (auto& [k, r] : s.recs)
      if (r.parent == pool && r.kind == Kind::DescSet && !r.destroyed) kids.push_back(k);
    for (const void* k : kids) retire(k);
  }
  if constexpr (ID == id::vkResetCommandPool) {
    const uint64_t pool = recOf(std::get<1>(a)).id;
    for (auto& [k, r] : s.recs)
      if (r.parent == pool && r.kind == Kind::CmdBuf) r.epoch = 0, r.cbRefs.clear(), r.cbImgReads.clear();
  }
  if constexpr (ID == id::vkResetCommandBuffer) {
    Rec& cb = recOf(std::get<0>(a));
    cb.epoch = 0, cb.cbRefs.clear(), cb.cbImgReads.clear();
  }
  if constexpr (ID == id::vkBeginCommandBuffer) {
    Rec& cb = recOf(std::get<0>(a));
    cb.epoch = (s.mode.load(std::memory_order_relaxed) & kCapture) ? s.epoch : 0;
    cb.cbRefs.clear(), cb.cbImgReads.clear();
  }
}

// A single-output record of an object a call with several outputs made (command buffers, sets, pipelines).
template <uint16_t ID, class... Args>
void emitOne(Kind kind, uint64_t parent, Args... args) {
  using T = ArgsOf<ID>;
  T a{args...};
  freshRec(*std::get<std::tuple_size_v<T> - 1>(a));  // before the id is written into the record
  Ids ids;
  Cod c(&ids);
  CallDef<ID>::cod(c, a);
  G& s = g();
  s.droppedNext += c.droppedNext;
  const std::vector<uint8_t> rec = pack(ID, c.out);
  appendIfCaptured<ID>(rec, ids.touched);
  // the created handle is the one behind the output pointer (the last argument)
  Rec& r = recOf(*std::get<std::tuple_size_v<T> - 1>(a));
  r.kind = kind, r.seq = s.seq++, r.create = rec, r.parent = parent;
  for (uint64_t t : ids.touched)
    if (Rec* d = find(t); d && d->kind == Kind::Device) r.dev = d->id;
  addDeps(r, ids.touched);
}

// ---- state of the descriptor sets, from the driver's own tables ----
// The contents of one set as a vkUpdateDescriptorSets record. `state`: the record goes to the STATE section (counted in
// stateBytes); otherwise the caller has opened the FRAME section (the set was changed by a call with no record of its own).
void emitSetState(Writer& w, Rec& setRec, bool state) {
  if (setRec.kind != Kind::DescSet || setRec.destroyed) return;
  std::unique_lock<std::shared_mutex> tables(g().setLock);  // held until the record is encoded (it points into the set)
  const auto* set = from<DescriptorSetObj>(reinterpret_cast<VkDescriptorSet>(const_cast<void*>(setRec.handle)));
  Rec* dr = find(setRec.dev);
  if (!set || !set->layout || !dr) return;
  std::vector<VkWriteDescriptorSet> writes;
  std::deque<VkDescriptorBufferInfo> bufs;
  std::deque<VkDescriptorImageInfo> imgs;
  std::deque<VkBufferView> texels;
  std::deque<VkWriteDescriptorSetInlineUniformBlock> inl;
  const VkDescriptorSet hs = reinterpret_cast<VkDescriptorSet>(const_cast<void*>(setRec.handle));
  const auto& bindings = set->layout->bindings;
  for (size_t b = 0; b < bindings.size() && b < set->slots.size(); ++b) {
    const SetLayoutBinding& lb = bindings[b];
    if (lb.type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
      if (b >= set->inlineData.size() || set->inlineData[b].empty()) continue;
      VkWriteDescriptorSetInlineUniformBlock ib{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK};
      ib.dataSize = static_cast<uint32_t>(set->inlineData[b].size());
      ib.pData = set->inlineData[b].data();
      inl.push_back(ib);
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &inl.back()};
      wr.dstSet = hs, wr.dstBinding = lb.binding, wr.descriptorCount = ib.dataSize, wr.descriptorType = lb.type;
      writes.push_back(wr);
      continue;
    }
    for (uint32_t e = 0; e < set->slots[b].size(); ++e) {
      const Descriptor& d = set->slots[b][e];
      if (d.type == VK_DESCRIPTOR_TYPE_MAX_ENUM) continue;  // never written
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      wr.dstSet = hs, wr.dstBinding = lb.binding, wr.dstArrayElement = e, wr.descriptorCount = 1, wr.descriptorType = d.type;
      switch (d.type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
          if (!live(d.buffer, Kind::Buffer)) continue;
          bufs.push_back({to<VkBuffer>(const_cast<Buffer*>(d.buffer)), d.offset, d.range});
          wr.pBufferInfo = &bufs.back();
          break;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
          if (!live(d.texel, Kind::BufferView)) continue;
          texels.push_back(to<VkBufferView>(const_cast<BufferView*>(d.texel)));
          wr.pTexelBufferView = &texels.back();
          break;
        }
        default: {  // sampler / combined / sampled image / storage image / input attachment
          if ((d.view && !live(d.view, Kind::ImageView)) || (d.sampler && !live(d.sampler, Kind::Sampler))) continue;
          VkDescriptorImageInfo ii{};
          ii.sampler = to<VkSampler>(const_cast<Sampler*>(d.sampler));
          ii.imageView = to<VkImageView>(const_cast<ImageView*>(d.view));
          ii.imageLayout = d.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? VK_IMAGE_LAYOUT_GENERAL
                                                                      : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          imgs.push_back(ii);
          wr.pImageInfo = &imgs.back();
        }
      }
      writes.push_back(wr);
    }
  }
  if (writes.empty()) return;
  ArgsOf<id::vkUpdateDescriptorSets> a{to<VkDevice>(const_cast<void*>(dr->handle)), static_cast<uint32_t>(writes.size()),
                                       writes.data(), 0u, nullptr};
  Ids ids;
  Cod c(&ids);
  CallDef<id::vkUpdateDescriptorSets>::cod(c, a);
  g().droppedNext += c.droppedNext;
  const auto rec = pack(id::vkUpdateDescriptorSets, c.out);
  w.append(rec.data(), rec.size());
  if (state) g().cap.stateBytes += rec.size();
}

// ---- buffer contents ----
uint64_t hashPage(const uint8_t* p, size_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t v;
    std::memcpy(&v, p + i, 8);
    h = (h ^ v) * 0xFF51AFD7ED558CCDull;
    h ^= h >> 32;
  }
  for (; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
  return h | 1;  // never 0: 0 is "unseen"
}
bool zeroPage(const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (p[i]) return false;
  return true;
}

void writeMemory(Writer& w, uint64_t memId, const uint8_t* host, uint64_t off, uint64_t len) {
  std::vector<uint8_t> payload(24 + len);
  std::memcpy(payload.data(), &memId, 8);
  std::memcpy(payload.data() + 8, &off, 8);
  std::memcpy(payload.data() + 16, &len, 8);
  std::memcpy(payload.data() + 24, host + off, len);
  const auto rec = pack(id::vpcapMemoryData, payload);
  w.begin(Section::Frame);
  w.append(rec.data(), rec.size());
}

// [off, off + len) of a memory object: write the pages that differ from what the file holds.
void diffMemory(uint64_t memId, const DeviceMemory& m, uint64_t off, uint64_t len) {
  Capture& c = g().cap;
  if (!m.host || off >= m.size) return;
  len = std::min<uint64_t>(len, m.size - off);
  std::vector<uint64_t>& sh = c.shadow[memId];
  if (sh.empty()) sh.assign((m.size + kPage - 1) / kPage, 0);
  constexpr uint64_t kMaxRun = 4u << 20;
  uint64_t runStart = UINT64_MAX;
  auto flush = [&](uint64_t endByte) {
    if (runStart == UINT64_MAX) return;
    for (uint64_t at = runStart; at < endByte; at += kMaxRun) {
      const uint64_t n = std::min<uint64_t>(kMaxRun, endByte - at);
      writeMemory(*c.w, memId, m.host, at, n);
      c.memBytes += n;
    }
    runStart = UINT64_MAX;
  };
  const uint64_t p0 = off / kPage, p1 = (off + len + kPage - 1) / kPage;
  for (uint64_t p = p0; p < p1 && p < sh.size(); ++p) {
    const uint64_t at = p * kPage, n = std::min<uint64_t>(kPage, m.size - at);
    bool dirty;
    if (sh[p] == 0 && zeroPage(m.host + at, n)) {
      dirty = false;
    } else {
      const uint64_t h = hashPage(m.host + at, n);
      dirty = h != sh[p];
      sh[p] = h;
    }
    if (dirty) {
      if (runStart == UINT64_MAX) runStart = at;
      ++c.memPages;
    } else {
      flush(at);
    }
  }
  flush(std::min<uint64_t>(p1 * kPage, m.size));
}

using RangeSet = std::set<std::tuple<uint64_t, uint64_t, uint64_t>>;
// [lo, hi) of the buffer, each distinct range once per submit.
void bufferRange(const Buffer* b, uint64_t lo, uint64_t hi, RangeSet& done) {
  if (!b || !b->memory) return;
  auto it = g().recs.find(b->memory);
  if (it == g().recs.end() || it->second.destroyed || it->second.kind != Kind::Memory || lo >= b->size) return;
  hi = std::min<uint64_t>(hi, b->size);
  if (!done.insert({it->second.id, b->offset + lo, b->offset + hi}).second) return;
  diffMemory(it->second.id, *b->memory, b->offset + lo, hi - lo);
}

// `reads` collects the ids of the images the submit reads (captureImage).
void resolveRef(uint64_t refId, RangeSet& done, std::unordered_set<uint64_t>& visitedCbs, std::vector<uint64_t>& reads) {
  Rec* r = find(refId);
  if (!r || r->destroyed) return;
  switch (r->kind) {
    case Kind::Buffer: bufferRange(from<Buffer>(reinterpret_cast<VkBuffer>(const_cast<void*>(r->handle))), 0, UINT64_MAX, done); break;
    case Kind::BufferView: {
      const auto* v = from<BufferView>(reinterpret_cast<VkBufferView>(const_cast<void*>(r->handle)));
      if (v && live(v->buffer, Kind::Buffer)) bufferRange(v->buffer, 0, UINT64_MAX, done);
      break;
    }
    case Kind::Image: g().cap.images.insert(r->id); break;
    case Kind::ImageView: {
      const auto* v = from<ImageView>(reinterpret_cast<VkImageView>(const_cast<void*>(r->handle)));
      if (v && live(v->image, Kind::Image)) g().cap.images.insert(g().recs.find(v->image)->second.id);
      break;
    }
    case Kind::CmdBuf:
      if (visitedCbs.insert(r->id).second) {
        reads.insert(reads.end(), r->cbImgReads.begin(), r->cbImgReads.end());
        for (uint64_t t : std::vector<uint64_t>(r->cbRefs)) resolveRef(t, done, visitedCbs, reads);
      }
      break;
    case Kind::DescSet: {
      // What the set holds is copied under the tables' lock (an update may be writing them on another thread); the buffers
      // are then diffed without it.
      struct BufRef {
        const Buffer* buf;
        uint64_t lo, hi;
      };
      std::vector<BufRef> bufs;
      std::vector<uint64_t> images;
      {
        std::unique_lock<std::shared_mutex> tables(g().setLock);
        const auto* set = from<DescriptorSetObj>(reinterpret_cast<VkDescriptorSet>(const_cast<void*>(r->handle)));
        if (!set || !set->layout) break;
        for (const auto& slot : set->slots)
          for (const Descriptor& d : slot) {
            if (d.type == VK_DESCRIPTOR_TYPE_MAX_ENUM) continue;
            if (d.buffer && live(d.buffer, Kind::Buffer)) {
              const bool dyn = d.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || d.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
              bufs.push_back({d.buffer, d.offset, (dyn || d.range == VK_WHOLE_SIZE) ? UINT64_MAX : d.offset + d.range});
            }
            if (d.texel && live(d.texel, Kind::BufferView) && live(d.texel->buffer, Kind::Buffer))
              bufs.push_back({d.texel->buffer, 0, UINT64_MAX});
            if (d.view && live(d.view, Kind::ImageView) && live(d.view->image, Kind::Image))
              images.push_back(g().recs.find(d.view->image)->second.id);
          }
      }
      for (const BufRef& b : bufs) bufferRange(b.buf, b.lo, b.hi, done);
      for (uint64_t i : images) {
        g().cap.images.insert(i);
        reads.push_back(i);  // sampled / storage / input attachment: read from the contents it holds
      }
      break;
    }
    case Kind::Framebuffer: {
      const auto* fb = from<Framebuffer>(reinterpret_cast<VkFramebuffer>(const_cast<void*>(r->handle)));
      if (fb)
        for (const ImageView* v : fb->views)
          if (live(v, Kind::ImageView) && live(v->image, Kind::Image)) g().cap.images.insert(g().recs.find(v->image)->second.id);
      break;
    }
    default: break;
  }
}

// ---- image contents ----
constexpr size_t kImageChunk = 16u << 20;  // bytes of a subresource per vpcapImageData record

void writeImageData(Capture& c, uint64_t image, const vpgpu::TextureSpan& s, uint32_t aspect, const std::vector<uint8_t>& bytes) {
  for (uint64_t off = 0; off < bytes.size(); off += kImageChunk) {
    const uint64_t n = std::min<uint64_t>(kImageChunk, bytes.size() - off);
    std::vector<uint8_t> payload(kImageDataHeader + n, 0);  // layout: vpcap.h ImageData
    const uint32_t f[6] = {s.mip, s.layer, aspect, s.width, s.height, s.depth};
    const uint64_t t[3] = {bytes.size(), off, n};
    std::memcpy(payload.data(), &image, 8);
    std::memcpy(payload.data() + 8, f, sizeof f);
    std::memcpy(payload.data() + 40, t, sizeof t);
    std::memcpy(payload.data() + kImageDataHeader, bytes.data() + off, n);
    const auto rec = pack(id::vpcapImageData, payload);
    c.w->begin(Section::Frame);
    c.w->append(rec.data(), rec.size());
  }
}

// One image whose contents go to the file at a submit. prepareImage decides what it takes (g().mu held); readImage reads it
// from the idle GPU (no capture lock: the wait for the GPU must not stop the other threads); writeJob writes it (mu again).
struct ImageJob {
  struct Sub {
    vpgpu::TextureSpan span;
    uint32_t aspect = 0;         // VkImageAspectFlagBits as stored in the file
    std::vector<uint8_t> bytes;  // filled by readImage; left empty when the subresource is all zero
  };
  uint64_t id = 0;
  std::vector<Sub> subs;
  uint32_t zero = 0, memoryless = 0, unsupported = 0, failed = 0;  // what readImage found
  bool deferred = false;                                           // over the budget: waits for the next submit
};

// Bytes of image contents read at one submit (VPVK_CAPTURE_IMG_BUDGET_MB, default 512; 0 = no limit).
uint64_t imgBudgetBytes() {
  const char* e = std::getenv("VPVK_CAPTURE_IMG_BUDGET_MB");
  const double mb = e && *e ? std::atof(e) : 512.0;
  return mb > 0 ? static_cast<uint64_t>(mb * 1048576.0) : 0;
}

// The first read of image `id` in this capture (g().mu held): what can be said without the GPU is counted now, the rest
// becomes a job.
void prepareImage(Device* dev, uint64_t id, std::vector<ImageJob>& jobs) {
  Capture& c = g().cap;
  Rec* r = find(id);
  if (!r || r->destroyed || r->kind != Kind::Image) {
    c.imagesDone.insert(id);
    return;
  }
  const Image* im = from<Image>(reinterpret_cast<VkImage>(const_cast<void*>(r->handle)));
  std::lock_guard<std::mutex> submit(dev->submitMutex);  // the image's texture and memory binding belong to the submit thread
  if (!im->id) {  // not bound to memory: nothing to hold
    c.imagesDone.insert(id);
    return;
  }
  const bool lazy = im->memory && live(im->memory, Kind::Memory) && im->memory->lazy;  // the memory may be gone: the registry says
  if ((im->info.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) || im->desc.transient || lazy) {
    c.imagesDone.insert(id);
    ++c.imgNew, ++c.imgMemoryless;  // no memory behind it (tile only)
    return;
  }
  if (zeroPendingImage(im)) {  // nothing has written it yet: it reads as zero in the replay too
    c.imagesDone.insert(id);
    ++c.imgNew, ++c.imgZero;
    return;
  }
  ImageJob j;  // counted when its contents are written
  j.id = id;
  std::vector<const Image*> parts{im};  // a multi-planar image: planes 1 and 2 are textures of their own
  for (const auto& p : im->planes) parts.push_back(p.get());
  for (size_t k = 0; k < parts.size(); ++k) {
    const vpgpu::TextureDesc& d = parts[k]->desc;
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(d.format);
    struct Asp {
      vpgpu::Aspect aspect;
      uint32_t vk;
    };
    std::vector<Asp> aspects;
    constexpr uint32_t kColor = VK_IMAGE_ASPECT_COLOR_BIT, kDepth = VK_IMAGE_ASPECT_DEPTH_BIT, kStencil = VK_IMAGE_ASPECT_STENCIL_BIT;
    if (parts.size() > 1) aspects = {{vpgpu::Aspect::All, uint32_t(VK_IMAGE_ASPECT_PLANE_0_BIT) << k}};
    else if (fi.depth() && fi.stencil()) aspects = {{vpgpu::Aspect::Depth, kDepth}, {vpgpu::Aspect::Stencil, kStencil}};
    else aspects = {{vpgpu::Aspect::All, fi.depth() ? kDepth : fi.stencil() ? kStencil : kColor}};
    for (const Asp& a : aspects)
      for (uint32_t mip = 0; mip < d.mips; ++mip)
        for (uint32_t layer = 0; layer < d.layers; ++layer) {
          ImageJob::Sub sb;
          vpgpu::TextureSpan& s = sb.span;
          s.texture = parts[k]->id, s.mip = static_cast<uint16_t>(mip), s.layer = static_cast<uint16_t>(layer), s.aspect = a.aspect;
          s.width = std::max(1u, d.width >> mip), s.height = std::max(1u, d.height >> mip);
          s.depth = d.type == vpgpu::TextureType::Tex3D ? std::max(1u, d.depth >> mip) : 1;
          sb.aspect = a.vk;
          j.subs.push_back(std::move(sb));
        }
  }
  jobs.push_back(std::move(j));
}

// Reads the subresources of a job (the GPU idle, no capture lock). Returns the bytes read.
uint64_t readImage(Device* dev, ImageJob& j) {
  uint64_t total = 0;
  std::lock_guard<std::mutex> submit(dev->submitMutex);  // the backend's tables belong to the submit thread
  for (ImageJob::Sub& sb : j.subs) {
    std::vector<uint8_t> bytes;
    switch (dev->platform->readTexture(sb.span, bytes)) {
      case vpgpu::ReadResult::Ok: break;
      case vpgpu::ReadResult::Memoryless: ++j.memoryless; return total;
      case vpgpu::ReadResult::Unsupported: ++j.unsupported; return total;
      case vpgpu::ReadResult::Failed: ++j.failed; return total;
    }
    total += bytes.size();
    if (bytes.empty() || zeroPage(bytes.data(), bytes.size())) {
      ++j.zero;
      continue;
    }
    sb.bytes = std::move(bytes);
  }
  return total;
}

// A job read without the lock goes to the file (g().mu held). Another submit may have written the image meanwhile.
void writeJob(Capture& c, ImageJob& j) {
  if (!c.imagesDone.insert(j.id).second) return;
  ++c.imgNew;
  c.imgZero += j.zero, c.imgMemoryless += j.memoryless, c.imgUnsupported += j.unsupported, c.imgFailed += j.failed;
  for (ImageJob::Sub& sb : j.subs) {
    if (sb.bytes.empty()) continue;
    writeImageData(c, j.id, sb.span, sb.aspect, sb.bytes);
    ++c.imgSubres, c.imgBytes += sb.bytes.size();
    std::vector<uint8_t>().swap(sb.bytes);
  }
}

// ---- start / finish ----
// frame_unit: what "a frame" is in this file (and for VPVK_CAPTURE_FRAME / _FRAMES): a vkQueueSubmit / vkQueueSubmit2 that carries
// command buffers, the same unit vk.perf and vk.hitch count. The driver sees no present (the host composites the eye images) and
// has no end-of-frame call every host makes (vpvk_native_submit only runs with VPKTAA / RCAS on), so a game frame made of
// several submits is several frames here.
std::string meta(const Capture& c, size_t unhooked, const std::string& list) {
  return "format=VPCAP\nversion=" + std::to_string(kVersion) + "\nabi=lp64-le\ncompress=zlib\ndevice=" + platformDeviceName() +
         "\nframes_requested=" + std::to_string(c.target) + "\nframe_unit=submit\nstart_submit=" + std::to_string(c.startSubmit) +
         "\ninstance_in_state=" + std::string(c.instanceTracked ? "1" : "0") + "\nimage_contents=first_read\nunhooked_count=" + std::to_string(unhooked) + "\nunhooked=" + list + "\n";
}

std::string makePath(const std::string& req, const std::string& dir) {
  std::string p = req.empty() ? dir : req;
  if (p.empty()) p = ".";
  struct stat st;
  const bool isDir = p.back() == '/' || (::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
  if (!isDir) return p;
  if (p.back() != '/') p += '/';
  return p + "vpcap_" + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                           std::chrono::system_clock::now().time_since_epoch()).count()) +
         "_" + std::to_string(::getpid()) + ".vpcap";
}

void finishCapture(const char* why) {  // g().mu held
  G& s = g();
  Capture& c = s.cap;
  if (!c.w) return;
  s.mode.fetch_and(~kCapture);
  std::string reasons, used;
  uint32_t usedCalls = 0;
  for (const std::string& r : c.invalid) reasons += (reasons.empty() ? "" : ",") + r;
  for (const auto& [name, n] : c.unhookedUsed) used += (used.empty() ? "" : ",") + name + ":" + std::to_string(n), usedCalls += n;
  char b[1024];
  std::snprintf(b, sizeof b,
                "frames=%u\nframe_unit=submit\nsubmits=%llu\ncalls=%llu\nimages_seen=%u\nimage_subresources=%u\nimage_bytes=%llu\nimage_zero=%u\n"
                "image_memoryless=%u\nimage_unsupported=%u\nimage_failed=%u\nimg_deferred=%u\nimg_deferred_pending=%zu\n"
                "other_device_calls=%llu\nother_device_submits=%llu\npnext_dropped=%llu\nunhooked_used=%u\nunhooked_used_calls=%s\n"
                "invalid=%d\nreason=%s\nend=%s\n",
                c.frames, static_cast<unsigned long long>(c.submits), static_cast<unsigned long long>(c.w->calls()), c.imgNew,
                c.imgSubres, static_cast<unsigned long long>(c.imgBytes), c.imgZero, c.imgMemoryless, c.imgUnsupported,
                c.imgFailed, c.imgDeferred, c.deferredImages.size(), static_cast<unsigned long long>(c.foreignCalls),
                static_cast<unsigned long long>(c.foreignSubmits), static_cast<unsigned long long>(s.droppedNext - c.droppedNextAtStart),
                usedCalls, used.c_str(), c.invalid.empty() ? 0 : 1, reasons.c_str(), why);
  const bool ok = c.w->close(b);
  logf("vk.capture done frames=%u calls=%llu bytes=%llu file_bytes=%llu mem_pages=%llu mem_bytes=%llu cbs_dropped=%u "
       "pnext_dropped=%llu images_touched=%zu images_seen=%u img_subres=%u img_bytes=%llu img_zero=%u img_memoryless=%u "
       "img_unsupported=%u img_failed=%u img_deferred=%u other_device_calls=%llu other_device_submits=%llu unhooked_used=%u "
       "invalid=%d reason=%s ok=%d end=%s file=%s",
       c.frames, static_cast<unsigned long long>(c.w->calls()), static_cast<unsigned long long>(c.w->rawBytes()),
       static_cast<unsigned long long>(c.w->fileBytes()), static_cast<unsigned long long>(c.memPages),
       static_cast<unsigned long long>(c.memBytes), c.cbsDropped, static_cast<unsigned long long>(s.droppedNext - c.droppedNextAtStart),
       c.images.size(), c.imgNew, c.imgSubres, static_cast<unsigned long long>(c.imgBytes), c.imgZero, c.imgMemoryless,
       c.imgUnsupported, c.imgFailed, c.imgDeferred, static_cast<unsigned long long>(c.foreignCalls),
       static_cast<unsigned long long>(c.foreignSubmits), usedCalls, c.invalid.empty() ? 0 : 1, reasons.c_str(), ok ? 1 : 0, why,
       c.path.c_str());
  c = Capture{};
}

// Whether a record belongs to the captured device: its objects, the device and the instance it was made from.
bool ofCapture(const Rec& r) {
  const Capture& c = g().cap;
  return r.kind == Kind::Instance ? r.id == c.instId : r.kind == Kind::Device ? r.id == c.devId : r.dev == c.devId;
}

// g().mu held, the GPU idle. False: the file could not be opened (the request is dropped).
bool startCapture(Device* dev) {
  G& s = g();
  Capture& c = s.cap;
  c = Capture{};
  const auto di = s.recs.find(to<VkDevice>(dev));
  if (di == s.recs.end() || di->second.destroyed) {  // the submit's device was created before tracking began
    logf("vk.capture error what=untracked_device");
    return false;
  }
  c.devHandle = di->first, c.devId = di->second.id;
  if (dev->physical && dev->physical->instance) {
    const auto ii = s.recs.find(to<VkInstance>(dev->physical->instance));
    if (ii != s.recs.end() && !ii->second.destroyed) c.instId = ii->second.id;
  }
  c.instanceTracked = c.instId != 0;
  c.path = makePath(s.reqPath, s.autoDir);
  c.target = s.reqFrames, c.startSubmit = dev->submits, c.droppedNextAtStart = s.droppedNext;
  c.w = std::make_unique<Writer>();
  if (!c.w->open(c.path)) {
    logf("vk.capture error what=open file=%s", c.path.c_str());
    c = Capture{};
    return false;
  }
  std::string list;
  for (const auto& n : s.unhooked) list += (list.empty() ? "" : ",") + n;
  c.w->meta(meta(c, s.unhooked.size(), list));
  c.w->begin(Section::State);
  std::vector<std::pair<uint64_t, const std::vector<uint8_t>*>> items;
  size_t objs = 0;
  for (auto& [h, r] : s.recs) {
    if (!ofCapture(r)) continue;  // the other devices' objects stay out
    if (!r.create.empty()) items.push_back({r.seq, &r.create}), ++objs;
    if (!r.bind.empty()) items.push_back({r.bindSeq, &r.bind});
  }
  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& it : items) {
    c.w->append(it.second->data(), it.second->size());
    c.stateBytes += it.second->size();
  }
  for (auto& [h, r] : s.recs)
    if (r.kind == Kind::DescSet && !r.destroyed && ofCapture(r)) emitSetState(*c.w, r, true);
  ++s.epoch;
  s.mode.fetch_or(kCapture);
  static const bool exitHook = std::atexit([] {  // the process ends mid-capture: close the file properly
    std::lock_guard<std::mutex> lk(g().mu);
    finishCapture("process_exit");
  }) == 0;
  (void)exitHook;
  logf("vk.capture start frames=%u objs=%zu bytes=%llu file=%s unhooked=%zu pnext_dropped_state=%llu", c.target, objs,
       static_cast<unsigned long long>(c.stateBytes), c.path.c_str(), s.unhooked.size(),
       static_cast<unsigned long long>(s.droppedNext));
  return true;
}

// ---- vkQueueSubmit / vkQueueSubmit2: the frame boundary ----
// The two submit calls differ only in how a batch lists its command buffers.
template <class I>
struct Sub;
template <>
struct Sub<VkSubmitInfo> {
  using Elem = VkCommandBuffer;
  static constexpr uint16_t kId = id::vkQueueSubmit;
  static uint32_t count(const VkSubmitInfo& s) { return s.commandBufferCount; }
  static Elem at(const VkSubmitInfo& s, uint32_t j) { return s.pCommandBuffers[j]; }
  static VkCommandBuffer cb(const Elem& e) { return e; }
  static void set(VkSubmitInfo& s, const std::vector<Elem>& k) { s.commandBufferCount = static_cast<uint32_t>(k.size()), s.pCommandBuffers = k.data(); }
  static VkResult real(VkQueue q, uint32_t n, const VkSubmitInfo* si, VkFence f) { return vkQueueSubmit(q, n, si, f); }
};
template <>
struct Sub<VkSubmitInfo2> {
  using Elem = VkCommandBufferSubmitInfo;
  static constexpr uint16_t kId = id::vkQueueSubmit2;
  static uint32_t count(const VkSubmitInfo2& s) { return s.commandBufferInfoCount; }
  static Elem at(const VkSubmitInfo2& s, uint32_t j) { return s.pCommandBufferInfos[j]; }
  static VkCommandBuffer cb(const Elem& e) { return e.commandBuffer; }
  static void set(VkSubmitInfo2& s, const std::vector<Elem>& k) { s.commandBufferInfoCount = static_cast<uint32_t>(k.size()), s.pCommandBufferInfos = k.data(); }
  static VkResult real(VkQueue q, uint32_t n, const VkSubmitInfo2* si, VkFence f) { return vkQueueSubmit2(q, n, si, f); }
};

template <class I>
VkResult submitHook(VkQueue q, uint32_t n, const I* si, VkFence fence) {
  using T = Sub<I>;
  G& s = g();
  const uint32_t m0 = s.mode.load(std::memory_order_relaxed);
  if (!m0) return T::real(q, n, si, fence);
  Device* dev = from<Queue>(q)->device;
  const void* devH = to<VkDevice>(dev);
  bool hasCmds = false;
  for (uint32_t i = 0; i < n; ++i) hasCmds = hasCmds || T::count(si[i]);
  bool start = false;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if (hasCmds && !(s.mode.load() & kCapture)) {
      if (s.autoFrame && ++s.frameCount == s.autoFrame && !s.pending.load()) {  // VPVK_CAPTURE_FRAME
        s.reqFrames = s.autoFrames, s.reqPath.clear();
        s.pending = true;
      }
      start = s.pending.load() && s.recs.count(devH);  // the first tracked device to submit is the one captured
    }
  }
  // The recorded submit, in three steps so that nothing waits for the GPU while the capture's lock is held (another thread
  // may have to record a call before it can signal what a parked batch of this one's idle wait is waiting for):
  //  A (mu): what the submit references is diffed into the file, the images it reads for the first time are listed;
  //  B (no lock): the GPU is drained and those images are read;
  //  C (mu): their contents go to the file, then the record of the submit (the contents must precede it).
  bool recorded = false;
  uint32_t epoch0 = 0;
  std::vector<std::vector<typename T::Elem>> kept(n);  // the recorded copy: only command buffers recorded in the capture
  std::vector<I> infos;
  std::vector<ImageJob> jobs;
  uint32_t imgNew = 0, imgSubres = 0, imgZero = 0, imgMem = 0, imgUns = 0, imgFail = 0;
  uint64_t imgBytes = 0;
  {
    std::lock_guard<std::mutex> lk(s.mu);
    if ((s.mode.load() & kCapture) && s.cap.devHandle == devH) {
      recorded = true;
      epoch0 = s.epoch;
      Capture& c = s.cap;
      RangeSet done;
      std::unordered_set<uint64_t> visited;
      std::vector<uint64_t> reads;  // images the submit reads
      infos.assign(si, si + n);
      for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < T::count(si[i]); ++j) {
          const typename T::Elem e = T::at(si[i], j);
          const VkCommandBuffer cb = T::cb(e);
          if (!cbOk(cb)) {
            ++c.cbsDropped;
            continue;
          }
          kept[i].push_back(e);
          resolveRef(recOf(cb).id, done, visited, reads);
        }
        T::set(infos[i], kept[i]);
      }
      imgNew = c.imgNew, imgSubres = c.imgSubres, imgZero = c.imgZero, imgMem = c.imgMemoryless, imgUns = c.imgUnsupported,
      imgFail = c.imgFailed, imgBytes = c.imgBytes;
      // Images the budget pushed past an earlier submit come first.
      std::vector<uint64_t> want;
      want.swap(c.deferredImages);
      want.insert(want.end(), reads.begin(), reads.end());
      std::unordered_set<uint64_t> queued;
      for (uint64_t i : want)
        if (!c.imagesDone.count(i) && queued.insert(i).second) prepareImage(dev, i, jobs);
    } else if (s.mode.load() & kCapture) {
      ++s.cap.foreignSubmits;
    }
  }
  if (recorded) {
    uint32_t deferred = 0;
    if (!jobs.empty()) {
      vkDeviceWaitIdle(to<VkDevice>(dev));  // a native submit (TAA) may have run since the last drain
      const uint64_t budget = imgBudgetBytes();
      uint64_t readBytes = 0;
      for (ImageJob& j : jobs) {
        if (budget && readBytes >= budget) {
          j.deferred = true;
          ++deferred;
        } else {
          readBytes += readImage(dev, j);
        }
      }
    }
    std::lock_guard<std::mutex> lk(s.mu);
    if (!(s.mode.load() & kCapture) || s.epoch != epoch0 || !s.cap.w) {  // the capture ended meanwhile
      recorded = false;
    } else {
      Capture& c = s.cap;
      for (ImageJob& j : jobs) {
        if (j.deferred) c.deferredImages.push_back(j.id);
        else writeJob(c, j);
      }
      if (deferred) {
        c.imgDeferred += deferred;
        markInvalid("img_budget_deferred");
      }
      if (c.imgNew != imgNew || deferred)
        logf("vk.capture images submit=%llu new=%u subres=%u bytes=%llu zero=%u memoryless=%u unsupported=%u failed=%u deferred=%u",
             static_cast<unsigned long long>(c.submits + 1), c.imgNew - imgNew, c.imgSubres - imgSubres,
             static_cast<unsigned long long>(c.imgBytes - imgBytes), c.imgZero - imgZero, c.imgMemoryless - imgMem,
             c.imgUnsupported - imgUns, c.imgFailed - imgFail, deferred);
      ++c.submits;
      ArgsOf<T::kId> a{q, n, infos.data(), fence};
      emit<T::kId>(a);
    }
  }
  const VkResult r = T::real(q, n, si, fence);
  if (start) {  // the boundary: the capture begins here, so the next frame is the first one recorded whole
    vkDeviceWaitIdle(to<VkDevice>(dev));  // outside g().mu: the contents the snapshot reads must be final
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.pending.exchange(false) && !(s.mode.load() & kCapture)) startCapture(dev);
  }
  if (recorded && hasCmds) {
    vkDeviceWaitIdle(to<VkDevice>(dev));  // the next submit's diff reads what this one wrote
    std::lock_guard<std::mutex> lk(s.mu);
    Capture& c = s.cap;
    if ((s.mode.load() & kCapture) && s.epoch == epoch0 && c.w) {
      uint64_t payload[2] = {c.frames, c.submits};
      const auto rec = pack(id::vpcapFrameEnd, std::vector<uint8_t>(reinterpret_cast<uint8_t*>(payload),
                                                                      reinterpret_cast<uint8_t*>(payload) + 16));
      appendFrame(rec);
      c.w->sync();
      if (++c.frames >= c.target) finishCapture("frames_done");
    }
  }
  return r;
}

// ---- the hooks ----
constexpr bool isPre(uint16_t i) {
  return metaOf(i).obj >= 0 || i == id::vkFreeCommandBuffers || i == id::vkFreeDescriptorSets ||
         i == id::vkResetDescriptorPool || i == id::vkResetCommandPool || i == id::vkResetCommandBuffer;
}

// vkDestroyDevice (g().mu held): the capture ends only when the device it is of goes.
void endIfCapturedDevice(VkDevice dv) {
  G& s = g();
  const auto it = s.recs.find(dv);
  if ((s.mode.load() & kCapture) && it != s.recs.end() && it->second.id == s.cap.devId) finishCapture("device_destroyed");
}

template <uint16_t ID, auto Real, class = decltype(Real)>
struct HookT;
template <uint16_t ID, auto Real, class R, class... P>
struct HookT<ID, Real, R (*)(P...)> {
  static R VKAPI_CALL fn(P... p) {
    // Recorded before the call: commands (only for command buffers recorded during the capture) and the calls that destroy
    // or reset (the registry still knows the handles). Everything else after it, when it succeeded.
    constexpr bool kCmd = ID >= 0x0200 || ID == id::vkEndCommandBuffer;
    const uint32_t m = g().mode.load(std::memory_order_relaxed);
    if (!(m & (kCmd ? kCapture : (kTrack | kCapture)))) return Real(p...);
    ArgsOf<ID> a{p...};
    if constexpr (kCmd) {
      {
        std::lock_guard<std::mutex> lk(g().mu);
        if (cbOk(std::get<0>(a))) emit<ID>(a);
      }
      return Real(p...);
    } else if constexpr (isPre(ID)) {
      {
        std::lock_guard<std::mutex> lk(g().mu);
        if constexpr (ID == id::vkDestroyDevice) endIfCapturedDevice(std::get<0>(a));
        emit<ID>(a);
      }
      return Real(p...);
    } else if constexpr (std::is_void_v<R>) {
      if constexpr (ID == id::vkUpdateDescriptorSets) {
        std::shared_lock<std::shared_mutex> tables(g().setLock);  // a snapshot of the sets waits for the write, not the reverse
        Real(p...);
      } else {
        Real(p...);
      }
      std::lock_guard<std::mutex> lk(g().mu);
      emit<ID>(a);
    } else {
      const R r = Real(p...);
      if constexpr (std::is_same_v<R, VkResult>) {
        if (r < 0) return r;
      }
      std::lock_guard<std::mutex> lk(g().mu);
      emit<ID>(a);
      return r;
    }
  }
};

// vkBeginCommandBuffer is not a "cmd" for cbOk (it is what makes a command buffer ok): it takes the generic path.
// Calls that make several objects: one record per object (so each can be retired on its own).
VkResult VKAPI_CALL allocCbsHook(VkDevice dv, const VkCommandBufferAllocateInfo* ai, VkCommandBuffer* cbs) {
  const VkResult r = vkAllocateCommandBuffers(dv, ai, cbs);
  if (!(g().mode.load(std::memory_order_relaxed) & (kTrack | kCapture)) || r != VK_SUCCESS) return r;
  std::lock_guard<std::mutex> lk(g().mu);
  for (uint32_t i = 0; i < ai->commandBufferCount; ++i) {
    VkCommandBufferAllocateInfo one = *ai;
    one.commandBufferCount = 1;
    VkCommandBuffer cb = cbs[i];
    emitOne<id::vkAllocateCommandBuffers>(Kind::CmdBuf, recOf(ai->commandPool).id, dv, &one, &cb);
  }
  return r;
}
VkResult VKAPI_CALL allocSetsHook(VkDevice dv, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* sets) {
  const VkResult r = vkAllocateDescriptorSets(dv, ai, sets);
  if (!(g().mode.load(std::memory_order_relaxed) & (kTrack | kCapture)) || r != VK_SUCCESS) return r;
  std::lock_guard<std::mutex> lk(g().mu);
  for (uint32_t i = 0; i < ai->descriptorSetCount; ++i) {
    VkDescriptorSetAllocateInfo one = *ai;
    one.descriptorSetCount = 1, one.pSetLayouts = ai->pSetLayouts + i, one.pNext = nullptr;  // variable counts: not stored
    VkDescriptorSet set = sets[i];
    emitOne<id::vkAllocateDescriptorSets>(Kind::DescSet, recOf(ai->descriptorPool).id, dv, &one, &set);
  }
  return r;
}
template <uint16_t ID, class Info, class Fn>
VkResult pipelinesHook(Fn real, VkDevice dv, VkPipelineCache cache, uint32_t n, const Info* infos,
                       const VkAllocationCallbacks* ac, VkPipeline* out) {
  const VkResult r = real(dv, cache, n, infos, ac, out);
  if (!(g().mode.load(std::memory_order_relaxed) & (kTrack | kCapture)) || r < 0) return r;
  std::lock_guard<std::mutex> lk(g().mu);
  for (uint32_t i = 0; i < n; ++i) {
    if (!out[i]) continue;  // a failed one of a batch (VK_PIPELINE_COMPILE_REQUIRED / early return)
    Info one = infos[i];
    one.basePipelineIndex = -1;  // an index into the batch means nothing for a single create
    VkPipeline p = out[i];
    emitOne<ID>(Kind::Pipeline, 0, dv, VkPipelineCache{}, 1u, &one, ac, &p);
  }
  return r;
}
VkResult VKAPI_CALL gfxPipelinesHook(VkDevice dv, VkPipelineCache c, uint32_t n, const VkGraphicsPipelineCreateInfo* i,
                                     const VkAllocationCallbacks* ac, VkPipeline* o) {
  return pipelinesHook<id::vkCreateGraphicsPipelines>(&vkCreateGraphicsPipelines, dv, c, n, i, ac, o);
}
VkResult VKAPI_CALL computePipelinesHook(VkDevice dv, VkPipelineCache c, uint32_t n, const VkComputePipelineCreateInfo* i,
                                         const VkAllocationCallbacks* ac, VkPipeline* o) {
  return pipelinesHook<id::vkCreateComputePipelines>(&vkCreateComputePipelines, dv, c, n, i, ac, o);
}
// ---- state-changing calls with no VPCAP record of their own ----
// vkUpdateDescriptorSetWithTemplate: the set's contents after the call are recorded as a vkUpdateDescriptorSets.
void VKAPI_CALL updateTemplateHook(VkDevice dv, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void* data) {
  if (!(g().mode.load(std::memory_order_relaxed) & (kTrack | kCapture))) return vkUpdateDescriptorSetWithTemplate(dv, set, t, data);
  {
    std::shared_lock<std::shared_mutex> tables(g().setLock);
    vkUpdateDescriptorSetWithTemplate(dv, set, t, data);
  }
  if (!(g().mode.load(std::memory_order_relaxed) & kCapture)) return;
  std::lock_guard<std::mutex> lk(g().mu);
  if (!(g().mode.load() & kCapture) || !live(set, Kind::DescSet)) return;
  Rec& r = g().recs.find(set)->second;
  if (r.dev != g().cap.devId) return void(++g().cap.foreignCalls);
  g().cap.w->begin(Section::Frame);
  emitSetState(*g().cap.w, r, false);
}
// vkBindBufferMemory2 / vkBindImageMemory2: one plain bind per element.
template <uint16_t ID, class Info, class Fn>
VkResult bind2Hook(Fn real, VkDevice dv, uint32_t n, const Info* bi) {
  const VkResult r = real(dv, n, bi);
  if (!(g().mode.load(std::memory_order_relaxed) & (kTrack | kCapture)) || r < 0) return r;
  std::lock_guard<std::mutex> lk(g().mu);
  for (uint32_t i = 0; i < n; ++i) {
    if constexpr (ID == id::vkBindBufferMemory) {
      ArgsOf<ID> a{dv, bi[i].buffer, bi[i].memory, bi[i].memoryOffset};
      emit<ID>(a);
    } else {
      ArgsOf<ID> a{dv, bi[i].image, bi[i].memory, bi[i].memoryOffset};
      emit<ID>(a);
    }
    if (bi[i].pNext && (g().mode.load() & kCapture)) markInvalid("bind2_pnext");
  }
  return r;
}
VkResult VKAPI_CALL bindBuffer2Hook(VkDevice dv, uint32_t n, const VkBindBufferMemoryInfo* bi) {
  return bind2Hook<id::vkBindBufferMemory>(&vkBindBufferMemory2, dv, n, bi);
}
VkResult VKAPI_CALL bindImage2Hook(VkDevice dv, uint32_t n, const VkBindImageMemoryInfo* bi) {
  return bind2Hook<id::vkBindImageMemory>(&vkBindImageMemory2, dv, n, bi);
}
// vkCmdNextSubpass2 is a vkCmdNextSubpass; vkCmdDispatchBase with a zero base is a vkCmdDispatch (the driver drops the others).
void VKAPI_CALL nextSubpass2Hook(VkCommandBuffer cb, const VkSubpassBeginInfo* bi, const VkSubpassEndInfo* ei) {
  if (g().mode.load(std::memory_order_relaxed) & kCapture) {
    std::lock_guard<std::mutex> lk(g().mu);
    if (cbOk(cb)) {
      ArgsOf<id::vkCmdNextSubpass> a{cb, bi ? bi->contents : VK_SUBPASS_CONTENTS_INLINE};
      emit<id::vkCmdNextSubpass>(a);
    }
  }
  vkCmdNextSubpass2(cb, bi, ei);
}
void VKAPI_CALL dispatchBaseHook(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz, uint32_t gx, uint32_t gy, uint32_t gz) {
  if (g().mode.load(std::memory_order_relaxed) & kCapture) {
    std::lock_guard<std::mutex> lk(g().mu);
    if (cbOk(cb)) {
      if (bx || by || bz) {
        markInvalid("dispatch_base_offset");
      } else {
        ArgsOf<id::vkCmdDispatch> a{cb, gx, gy, gz};
        emit<id::vkCmdDispatch>(a);
      }
    }
  }
  vkCmdDispatchBase(cb, bx, by, bz, gx, gy, gz);
}

// The rest are only counted while a capture runs (unhooked_used); the ones the replay cannot reproduce make it invalid.
#define VPCAP_UNHOOKED(X)                                                                                              \
  X(vkCmdCopyQueryPoolResults, true) X(vkCmdDrawMultiEXT, true) X(vkCmdDrawMultiIndexedEXT, true)                      \
  X(vkCmdSetColorWriteMaskEXT, true) X(vkCmdSetRenderingAttachmentLocationsKHR, true)                                  \
  X(vkCmdSetRenderingInputAttachmentIndicesKHR, true) X(vkCopyImageToImageEXT, true) X(vkCopyMemoryToImageEXT, true)   \
  X(vkTransitionImageLayoutEXT, true) X(vkCreateSamplerYcbcrConversion, true)                                          \
  X(vkCopyImageToMemoryEXT, false) X(vkSetEvent, false) X(vkResetEvent, false) X(vkWaitSemaphores, false)             \
  X(vkSignalSemaphore, false) X(vkResetQueryPool, false) X(vkSetPrivateData, false) X(vkCreatePrivateDataSlot, false)  \
  X(vkDestroyPrivateDataSlot, false) X(vkCreatePipelineCache, false) X(vkDestroyPipelineCache, false)                  \
  X(vkCreateDescriptorUpdateTemplate, false) X(vkDestroyDescriptorUpdateTemplate, false)                               \
  X(vkDestroySamplerYcbcrConversion, false) X(vkCmdSetDeviceMask, false) X(vkCmdSetLogicOpEXT, false)                  \
  X(vkCmdSetPatchControlPointsEXT, false)
struct Unhooked {
  const char* name;
  bool invalid;
};
constexpr Unhooked kUnhooked[] = {
#define VPCAP_X(n, inv) {#n, inv},
    VPCAP_UNHOOKED(VPCAP_X)
#undef VPCAP_X
};
enum UnhookedIdx : size_t {
#define VPCAP_X(n, inv) k_##n,
  VPCAP_UNHOOKED(VPCAP_X)
#undef VPCAP_X
};
void noteUnhooked(size_t idx, VkCommandBuffer cb, VkDevice dv) {
  G& s = g();
  std::lock_guard<std::mutex> lk(s.mu);
  if (!(s.mode.load() & kCapture)) return;
  Capture& c = s.cap;
  if (cb) {
    if (!cbOk(cb)) return;
  } else if (dv) {
    const auto it = s.recs.find(dv);
    if (it == s.recs.end() || it->second.id != c.devId) return;
  }
  ++c.unhookedUsed[kUnhooked[idx].name];
  if (kUnhooked[idx].invalid) markInvalid(std::string("unhooked:") + kUnhooked[idx].name);
}
template <size_t I, auto Real, class = decltype(Real)>
struct CountT;
template <size_t I, auto Real, class R, class... P>
struct CountT<I, Real, R (*)(P...)> {
  static R VKAPI_CALL fn(P... p) {
    if (g().mode.load(std::memory_order_relaxed) & kCapture) {
      VkCommandBuffer cb = nullptr;
      VkDevice dv = nullptr;
      if constexpr (sizeof...(P) > 0) {
        using P0 = std::tuple_element_t<0, std::tuple<P...>>;
        if constexpr (std::is_same_v<P0, VkCommandBuffer>) cb = std::get<0>(std::tuple<P...>(p...));
        else if constexpr (std::is_same_v<P0, VkDevice>) dv = std::get<0>(std::tuple<P...>(p...));
      }
      noteUnhooked(I, cb, dv);
    }
    return Real(p...);
  }
};

VkResult VKAPI_CALL submitEntry(VkQueue q, uint32_t n, const VkSubmitInfo* si, VkFence f) { return submitHook(q, n, si, f); }
VkResult VKAPI_CALL submit2Entry(VkQueue q, uint32_t n, const VkSubmitInfo2* si, VkFence f) { return submitHook(q, n, si, f); }

struct HookEntry {
  PFN_vkVoidFunction real, hook;
};
#define VPCAP_H(f) {reinterpret_cast<PFN_vkVoidFunction>(&::f), reinterpret_cast<PFN_vkVoidFunction>(&HookT<id::f, &::f>::fn)}
#define VPCAP_HX(f, h) {reinterpret_cast<PFN_vkVoidFunction>(&::f), reinterpret_cast<PFN_vkVoidFunction>(&h)}
const std::vector<HookEntry>& hooks() {
  // The hand-written hooks come first: the search takes the first match (the generic ones after them stay unused).
  static const std::vector<HookEntry> t = {
      VPCAP_HX(vkQueueSubmit, submitEntry),
      VPCAP_HX(vkQueueSubmit2, submit2Entry),
      VPCAP_HX(vkAllocateCommandBuffers, allocCbsHook),
      VPCAP_HX(vkAllocateDescriptorSets, allocSetsHook),
      VPCAP_HX(vkCreateGraphicsPipelines, gfxPipelinesHook),
      VPCAP_HX(vkCreateComputePipelines, computePipelinesHook),
      VPCAP_HX(vkUpdateDescriptorSetWithTemplate, updateTemplateHook),
      VPCAP_HX(vkBindBufferMemory2, bindBuffer2Hook),
      VPCAP_HX(vkBindImageMemory2, bindImage2Hook),
      VPCAP_HX(vkCmdNextSubpass2, nextSubpass2Hook),
      VPCAP_HX(vkCmdDispatchBase, dispatchBaseHook),
#define VPCAP_X(name, num) VPCAP_H(name),
      VPCAP_VKCALLS(VPCAP_X)
#undef VPCAP_X
  };
  return t;
}

// Counting wrappers of the calls in VPCAP_UNHOOKED (they stay listed as unhooked in META).
const std::vector<HookEntry>& counters() {
  static const std::vector<HookEntry> t = {
#define VPCAP_X(n, inv) {reinterpret_cast<PFN_vkVoidFunction>(&::n), reinterpret_cast<PFN_vkVoidFunction>(&CountT<k_##n, &::n>::fn)},
      VPCAP_UNHOOKED(VPCAP_X)
#undef VPCAP_X
  };
  return t;
}

void initOnce() {
  static std::once_flag once;
  std::call_once(once, [] {
    G& s = g();
    const char* t = std::getenv("VPVK_CAPTURE");
    const char* f = std::getenv("VPVK_CAPTURE_FRAME");
    if ((t && *t && *t != '0') || (f && std::atoll(f) > 0)) s.mode.fetch_or(kTrack);
    if (f && std::atoll(f) > 0) {
      s.autoFrame = static_cast<uint64_t>(std::atoll(f));
      if (const char* n = std::getenv("VPVK_CAPTURE_FRAMES")) s.autoFrames = std::max(1, std::atoi(n));
      if (const char* d = std::getenv("VPVK_CAPTURE_DIR")) s.autoDir = d;
      logf("vk.capture armed frame=%llu frames=%u dir=%s", static_cast<unsigned long long>(s.autoFrame), s.autoFrames,
           s.autoDir.empty() ? "." : s.autoDir.c_str());
    }
  });
}

}  // namespace

PFN_vkVoidFunction captureWrap(const char* name, PFN_vkVoidFunction real) {
  initOnce();
  if (!real) return real;
  for (const HookEntry& h : hooks())
    if (h.real == real) return h.hook;
  static const char* const kStateful[] = {"vkCmd", "vkCreate", "vkAllocate", "vkQueue", "vkUpdate", "vkBind",
                                          "vkReset", "vkDestroy", "vkFree", "vkWait", "vkSet", "vkCopy", "vkTransition", "vkSignal"};
  for (const char* p : kStateful)
    if (!std::strncmp(name, p, std::strlen(p))) {
      std::lock_guard<std::mutex> lk(g().mu);
      g().unhooked.insert(name);
      break;
    }
  for (const HookEntry& h : counters())
    if (h.real == real) return h.hook;
  return real;
}

}  // namespace vpvk

extern "C" {
__attribute__((visibility("default"))) int vpvk_capture_track(int on) {
  using namespace vpvk;
  initOnce();
  G& s = g();
  std::lock_guard<std::mutex> lk(s.mu);
  const int prev = (s.mode.load() & kTrack) ? 1 : 0;
  if (on && !prev) {
    s.recs.clear(), s.byId.clear();
    s.mode.fetch_or(kTrack);
  } else if (!on) {
    finishCapture("tracking_off");
    s.pending = false;
    s.mode.store(0);
    s.recs.clear(), s.byId.clear();
  }
  return prev;
}

__attribute__((visibility("default"))) int vpvk_capture_request(uint32_t frames, const char* path) {
  using namespace vpvk;
  initOnce();
  G& s = g();
  std::lock_guard<std::mutex> lk(s.mu);
  bool device = false;
  for (const auto& [h, r] : s.recs) device = device || (r.kind == Kind::Device && !r.destroyed);
  if (!(s.mode.load() & kTrack) || !device) {
    logf("vk.capture refused reason=%s", (s.mode.load() & kTrack) ? "no_tracked_device" : "tracking_off (VPVK_CAPTURE=1)");
    return -1;
  }
  if (s.pending.load() || (s.mode.load() & kCapture)) return 0;
  s.reqFrames = frames ? frames : 1;
  s.reqPath = path ? path : "";
  s.pending = true;
  return 1;
}

__attribute__((visibility("default"))) int vpvk_capture_state(void) {
  using namespace vpvk;
  G& s = g();
  return (s.mode.load() & kCapture) ? 2 : s.pending.load() ? 1 : 0;
}
}
