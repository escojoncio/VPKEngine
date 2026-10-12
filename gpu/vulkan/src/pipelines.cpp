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
#include <cstdio>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "commands.h"
#include "vpir/passes.h"
#include "vpmsl/msl.h"
#include "vpvk.h"

#define VPVK_EXPORT extern "C" __attribute__((visibility("default")))

namespace vpvk {

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
  std::memcpy(c.host + c.used, data, size);
  c.used += need;
  return c.id;
}

void UploadArena::release(Device* d) {
  for (auto& c : chunks) {
    if (d && c.id) d->gpu->destroy(c.id);
    std::free(c.host);
  }
  chunks.clear();
  current = 0;
}

Recorder::~Recorder() { arena.release(device); }

namespace {

uint64_t fnv(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
  return h ? h : 1;
}

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
    case VK_BLEND_FACTOR_CONSTANT_COLOR:
    case VK_BLEND_FACTOR_CONSTANT_ALPHA: return B::Constant;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return B::OneMinusConstant;
    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return B::One;  // not in the core: closest
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
      ls.kind = kindOf(b.type);
      ls.slot = next[int(ls.kind)];
      next[int(ls.kind)] = uint16_t(next[int(ls.kind)] + ls.count);
      if (b.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        ls.samplerSlot = next[int(vpgpu::BindKind::Sampler)];
        next[int(vpgpu::BindKind::Sampler)] = uint16_t(next[int(vpgpu::BindKind::Sampler)] + ls.count);
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
  m->module = spirv::parse(ci->pCode, ci->codeSize / 4, m->error);
  if (!m->module) logf("vk.err what=spirv_parse error=\"%s\"", m->error.c_str());
  const Device* d = dv ? from<Device>(dv) : nullptr;
  if (m->module && d && d->recipes && d->recipes->enabled()) {
    m->words.assign(ci->pCode, ci->pCode + ci->codeSize / 4);
    m->hash = spirvHash(m->words.data(), m->words.size());
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

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* ai,
                                                                    VkDescriptorSet* out) {
  auto* pool = from<DescriptorPoolObj>(ai->descriptorPool);
  for (uint32_t i = 0; i < ai->descriptorSetCount; ++i) {
    auto* s = new DescriptorSetObj;
    s->layout = from<DescriptorSetLayoutObj>(ai->pSetLayouts[i])->data;
    s->pool = pool;
    s->slots.resize(s->layout->bindings.size());
    for (size_t b = 0; b < s->layout->bindings.size(); ++b) {
      const SetLayoutBinding& lb = s->layout->bindings[b];
      s->slots[b].resize(lb.count);
      for (uint32_t k = 0; k < lb.immutable.size() && k < lb.count; ++k) {
        s->slots[b][k].sampler = lb.immutable[k];
        s->slots[b][k].type = lb.type;
      }
    }
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
    if (ii.sampler) d.sampler = from<Sampler>(ii.sampler);
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

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice, uint32_t nw, const VkWriteDescriptorSet* w,
                                                              uint32_t nc, const VkCopyDescriptorSet* c) {
  for (uint32_t i = 0; i < nw; ++i) {
    auto* s = from<DescriptorSetObj>(w[i].dstSet);
    for (uint32_t k = 0; k < w[i].descriptorCount; ++k) {
      Descriptor* d = element(s, w[i].dstBinding, w[i].dstArrayElement + k);
      if (!d) break;
      switch (w[i].descriptorType) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
          writeImage(*d, w[i].descriptorType, w[i].pImageInfo[k]);
          break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
          d->type = w[i].descriptorType;
          d->texel = w[i].pTexelBufferView[k] ? from<BufferView>(w[i].pTexelBufferView[k]) : nullptr;
          break;
        default: writeBuffer(*d, w[i].descriptorType, w[i].pBufferInfo[k]); break;
      }
    }
  }
  for (uint32_t i = 0; i < nc; ++i) {
    auto* src = from<DescriptorSetObj>(c[i].srcSet);
    auto* dst = from<DescriptorSetObj>(c[i].dstSet);
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
  *out = to<VkDescriptorUpdateTemplate>(t);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorUpdateTemplate(VkDevice, VkDescriptorUpdateTemplate t,
                                                                         const VkAllocationCallbacks*) {
  delete from<DescriptorUpdateTemplateObj>(t);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSetWithTemplate(VkDevice, VkDescriptorSet set,
                                                                         VkDescriptorUpdateTemplate tpl, const void* data) {
  auto* s = from<DescriptorSetObj>(set);
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (const VkDescriptorUpdateTemplateEntry& e : from<DescriptorUpdateTemplateObj>(tpl)->entries) {
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

// =================================================================================================
// Pipelines
// =================================================================================================
namespace {

struct StageOut {
  bool ok = false;
  uint64_t id = 0;
  uint64_t input = 0;  // StageInput hash when recipes are recorded
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
enum class TranslateFail { None, Lower, Msl };
TranslateFail lowerAndEmit(const spirv::SpirvModule& m, const spirv::SpirvLowerOptions& opt, vpir::Module& mod,
                           spirv::SpirvInterface& iface, vpmsl::MslShader& msl, uint64_t& id,
                           std::vector<std::string>& errors) {
  std::string error;
  if (!spirv::lower(m, opt, mod, iface, error)) {
    errors.push_back(std::move(error));
    return TranslateFail::Lower;
  }
  vpmsl::EmitResult e = vpmsl::emitMsl(mod);
  if (!e.errors.empty()) {
    errors = std::move(e.errors);
    return TranslateFail::Msl;
  }
  msl = std::move(e.shader);
  id = fnv(msl.source);
  return TranslateFail::None;
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

StageOut translateStage(Device* d, const VkPipelineShaderStageCreateInfo& st, const PipelineLayoutObj& pl,
                        uint32_t viewCount, const std::vector<spirv::VertexAttribute>& attrs,
                        const std::vector<spirv::VertexBufferBinding>& vbs, const std::string& label) {
  StageOut o;
  const ShaderModuleObj* sm = from<ShaderModuleObj>(st.module);
  if (!sm || !sm->module) {
    logf("vk.err what=shader pipeline=%s stage=%u error=\"no module\"", label.c_str(), unsigned(st.stage));
    return o;
  }
  spirv::SpirvLowerOptions opt;
  opt.entry = st.pName ? st.pName : "main";
  opt.stage = vpirStage(st.stage);
  opt.specConstants = specialization(st.pSpecializationInfo);
  opt.viewCount = viewCount;
  for (const LayoutSlot& s : pl.slots) {
    spirv::SpirvBinding b;
    b.set = s.set, b.binding = s.binding, b.arraySize = s.count, b.kind = s.kind, b.slot = s.slot, b.samplerSlot = s.samplerSlot;
    opt.layout.push_back(b);
  }
  opt.pushConstantSlot = pl.pushSlot;
  if (opt.stage == vpir::Stage::Vertex) {
    opt.vertexAttributes = attrs;
    opt.vertexBuffers = vbs;
  }
  const auto t0 = std::chrono::steady_clock::now();
  vpir::Module mod;
  std::vector<std::string> errors;
  switch (lowerAndEmit(*sm->module, opt, mod, o.iface, o.msl, o.id, errors)) {
    case TranslateFail::Lower:
      logf("vk.err what=shader pipeline=%s stage=%s error=\"%s\"", label.c_str(), vpir::toString(opt.stage), errors[0].c_str());
      return o;
    case TranslateFail::Msl: {
      std::lock_guard<std::mutex> lock(gTranslateLog);
      for (auto& x : errors) logf("vk.err what=msl pipeline=%s stage=%s error=\"%s\"", label.c_str(), vpir::toString(opt.stage), x.c_str());
      return o;
    }
    case TranslateFail::None: break;
  }
  noteShaderLib(o.id);  // gpu.shader.lib hit/miss against the first-run precompile index
  logPrivateStats(mod, o.id, opt.stage, label);
  {  // translation cost (SPIR-V -> VPIR -> MSL), to tell it apart from the Metal library compile
    static std::mutex m;
    static uint32_t n = 0;
    static double total = 0, worst = 0;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(m);
    ++n, total += ms, worst = std::max(worst, ms);
    if (n <= 4 || n % 32 == 0) logf("gpu.shader.msl translated=%u ms_total=%.0f ms_max=%.1f", n, total, worst);
  }
  ShaderBlob blob = blobOf(o.msl);
  if (d->recipes && !sm->words.empty()) {  // translator input, so a later translator can redo it
    StageInput in;
    in.spirv = sm->hash;
    in.opt = std::move(opt);
    o.input = in.hash();
    d->recipes->addStage(in, o.input, sm->words, o.id, blob);
  }
  d->platform->addShader(o.id, std::move(blob));
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
}

// Attachment formats of the subpass (or of dynamic rendering).
void targetFormats(const VkGraphicsPipelineCreateInfo& ci, vpgpu::PipelineState& st, uint32_t& viewCount) {
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
    }
}

PipelineObj* createGraphics(Device* d, const VkGraphicsPipelineCreateInfo& ci) {
  auto* p = new PipelineObj;
  p->layout = layoutOf(ci.layout);
  if (!p->layout) p->layout = std::make_shared<PipelineLayoutObj>();
  vpgpu::PipelineState& st = p->state;
  targetFormats(ci, st, p->viewCount);
  static std::atomic<unsigned> counter{0};
  st.label = "vk.pipe" + std::to_string(counter++);

  // Vertex input.
  std::vector<spirv::VertexAttribute> attrs;
  if (const auto* vi = ci.pVertexInputState) {
    for (uint32_t i = 0; i < vi->vertexBindingDescriptionCount; ++i) {
      const auto& b = vi->pVertexBindingDescriptions[i];
      spirv::VertexBufferBinding vb;
      vb.binding = b.binding, vb.stride = b.stride, vb.perInstance = b.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
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
    bool ok;
    st.primitive = primitive(ia->topology, ok);
    if (!ok) logf("vk.err what=topology value=%d pipeline=%s", int(ia->topology), st.label.c_str());
  }
  if (const auto* rs = ci.pRasterizationState) {
    st.cull = rs->cullMode == VK_CULL_MODE_FRONT_BIT ? vpgpu::CullMode::Front
              : rs->cullMode == VK_CULL_MODE_BACK_BIT ? vpgpu::CullMode::Back : vpgpu::CullMode::None;
    p->cullBoth = rs->cullMode == VK_CULL_MODE_FRONT_AND_BACK;
    p->frontFace = rs->frontFace;
    p->depthBiasEnable = rs->depthBiasEnable;
    p->depthClamp = rs->depthClampEnable;
    p->depthBias[0] = rs->depthBiasConstantFactor, p->depthBias[1] = rs->depthBiasClamp, p->depthBias[2] = rs->depthBiasSlopeFactor;
  }
  if (const auto* ms = ci.pMultisampleState) {
    st.samples = static_cast<uint8_t>(ms->rasterizationSamples);
    if (st.samples > 1 && forceSingleSample()) st.samples = 1;  // game MSAA -> 1x (VPKTAA does the AA)
    st.alphaToCoverage = ms->alphaToCoverageEnable;
  }
  if (const auto* ds = ci.pDepthStencilState; ds && st.depthFormat != vpgpu::Format::Undefined) {
    st.depthTest = ds->depthTestEnable;
    st.depthWrite = ds->depthTestEnable && ds->depthWriteEnable;
    st.depthCompare = ds->depthTestEnable ? static_cast<vpgpu::CompareOp>(ds->depthCompareOp) : vpgpu::CompareOp::Always;
    st.stencilTest = ds->stencilTestEnable;
    st.stencilFront = stencilFace(ds->front);
    st.stencilBack = stencilFace(ds->back);
    p->stencilRef = ds->front.reference;
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

  bool ok = true;
  std::array<uint64_t, 3> inputs{};  // recipe stage inputs: vertex, fragment, compute
  for (uint32_t i = 0; i < ci.stageCount; ++i) {
    const auto& s = ci.pStages[i];
    if (s.stage != VK_SHADER_STAGE_VERTEX_BIT && s.stage != VK_SHADER_STAGE_FRAGMENT_BIT) {
      logf("vk.err what=stage value=%u pipeline=%s (geometry/tessellation unsupported)", unsigned(s.stage), st.label.c_str());
      ok = false;
      continue;
    }
    StageOut o = translateStage(d, s, *p->layout, p->viewCount, attrs, p->vertexBuffers, st.label);
    if (!o.ok) {
      ok = false;
      continue;
    }
    if (s.stage == VK_SHADER_STAGE_VERTEX_BIT) {
      st.vertexShader = o.id;
      inputs[0] = o.input;
      collectUsed(*p, o, vpgpu::stage::vertex);
    } else {
      st.fragmentShader = o.id;
      inputs[1] = o.input;
      collectUsed(*p, o, vpgpu::stage::fragment);
      // Outputs the shader does not write keep their contents (no blend, write mask none).
      for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c)
        if (!(o.msl.colorOutputs & (1u << c))) st.blend[c].writeMask = 0;
    }
  }
  if (!st.vertexShader) ok = false;
  p->ok = ok;
  p->key = st.key();
  if (ok) {
    d->gpu->pipelines().request(st, p->key, vpgpu::Priority::Blocking);
    if (d->recipes) d->recipes->addPipeline(st, inputs);
  }
  return p;
}

PipelineObj* createCompute(Device* d, const VkComputePipelineCreateInfo& ci) {
  auto* p = new PipelineObj;
  p->compute = true;
  p->layout = layoutOf(ci.layout);
  if (!p->layout) p->layout = std::make_shared<PipelineLayoutObj>();
  static std::atomic<unsigned> counter{0};
  p->state.label = "vk.comp" + std::to_string(counter++);
  StageOut o = translateStage(d, ci.stage, *p->layout, 1, {}, {}, p->state.label);
  if (o.ok) {
    p->state.computeShader = o.id;
    collectUsed(*p, o, vpgpu::stage::compute);
    p->ok = true;
    p->key = p->state.key();
    d->gpu->pipelines().request(p->state, p->key, vpgpu::Priority::Blocking);
    if (d->recipes) d->recipes->addPipeline(p->state, {0, 0, o.input});
  }
  return p;
}

}  // namespace

// Recipe retranslation (recipes.cpp RecipeStore::refresh): quiet, no live-translation diagnostics.
bool vpvk::translateRecipeStage(const spirv::SpirvModule& m, const spirv::SpirvLowerOptions& opt, uint64_t& mslId,
                                ShaderBlob& blob, std::string& error) {
  vpir::Module mod;
  spirv::SpirvInterface iface;
  vpmsl::MslShader msl;
  std::vector<std::string> errors;
  if (lowerAndEmit(m, opt, mod, iface, msl, mslId, errors) != TranslateFail::None) {
    error = errors.empty() ? "translate" : errors[0];
    return false;
  }
  blob = blobOf(msl);
  return true;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice dv, VkPipelineCache, uint32_t n,
                                                                     const VkGraphicsPipelineCreateInfo* ci,
                                                                     const VkAllocationCallbacks*, VkPipeline* out) {
  for (uint32_t i = 0; i < n; ++i) out[i] = to<VkPipeline>(createGraphics(from<Device>(dv), ci[i]));
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice dv, VkPipelineCache, uint32_t n,
                                                                    const VkComputePipelineCreateInfo* ci,
                                                                    const VkAllocationCallbacks*, VkPipeline* out) {
  for (uint32_t i = 0; i < n; ++i) out[i] = to<VkPipeline>(createCompute(from<Device>(dv), ci[i]));
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice, VkPipeline p, const VkAllocationCallbacks*) {
  delete from<PipelineObj>(p);
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
  uint32_t queued = 0, known = 0;
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t k;
    std::memcpy(&k, b + 40 + size_t(i) * 8, 8);
    vpgpu::PipelineState s;
    if (!d->recipes || !d->recipes->stateFor(k, s)) continue;
    ++known;
    if (d->gpu->pipelines().status({k}) == vpgpu::PipelineCache::Status::Missing) {
      d->gpu->pipelines().request(s, {k}, vpgpu::Priority::Prefetch);
      ++queued;
    }
  }
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
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_VIEWPORT) && po->hasViewport) b.hasViewport = true, b.viewport = po->viewport;
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_SCISSOR) && po->hasScissor) b.hasScissor = true, b.scissor = po->scissor;
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_STENCIL_REFERENCE)) b.stencilRef = po->stencilRef;
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
                                                               const VkDeviceSize*, const VkDeviceSize*) {
  vkCmdBindVertexBuffers(cb, first, n, bufs, offs);
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
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D* r) {
  if (first == 0 && n) {
    BoundState& b = rec(cb).bound;
    b.hasScissor = true;
    b.scissor = r[0];
  }
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilReference(VkCommandBuffer cb, VkStencilFaceFlags, uint32_t ref) {
  rec(cb).bound.stencilRef = ref;
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
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilCompareMask(VkCommandBuffer, VkStencilFaceFlags, uint32_t) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilWriteMask(VkCommandBuffer, VkStencilFaceFlags, uint32_t) {}

// =================================================================================================
// Draws and dispatches
// =================================================================================================
namespace vpvk {
namespace {

bool resolveBindings(Recorder& r, const PipelineObj& p, const std::array<BoundState::Set, 8>& sets,
                     std::vector<vpgpu::Access>& out, std::vector<vpgpu::ViewBinding>& views) {
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
        case VK_DESCRIPTOR_TYPE_SAMPLER: {
          if (!d.sampler || !d.sampler->id) return false;
          out.push_back(vpgpu::samplerBinding(d.sampler->id, u.stages, uint16_t(ls.slot + e)));
          break;
        }
        default: {  // images
          if (!d.view || !d.view->image || !d.view->image->id) return false;
          vpgpu::Access a{vpgpu::ResourceRange::wholeTexture(d.view->image->id), u.stages, u.written};
          a.kind = ls.kind, a.slot = uint16_t(ls.slot + e);
          if (d.view->view) views.push_back({ls.kind, a.slot, d.view->view});
          out.push_back(a);
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

void logSkip(const char* why, const PipelineObj* p) {
  static std::atomic<unsigned> n{0};
  if (n.fetch_add(1) < 32) logf("vk.skip what=%s pipeline=%s", why, p ? p->state.label.c_str() : "none");
}

}  // namespace

OpDraw makeDraw(Recorder& r, uint32_t first, uint32_t count, uint32_t instances, uint32_t firstInstance,
                int32_t baseVertex, bool indexed) {
  OpDraw op;
  const PipelineObj* p = r.bound.graphics;
  if (!p || !p->ok) {
    op.ok = false;
    logSkip(p ? "pipeline_failed" : "no_pipeline", p);
    return op;
  }
  vpgpu::Draw& d = op.draw;
  d.pipeline = p->key;
  d.primitive = p->state.primitive;
  d.first = first, d.count = count;
  d.instances = instances * p->viewCount;
  d.baseInstance = firstInstance * p->viewCount;
  d.baseVertex = baseVertex;
  op.viewCount = p->viewCount;
  if (!resolveBindings(r, *p, r.bound.graphicsSets, d.bindings, d.views)) {
    op.ok = false;
    logSkip("unbound_descriptor", p);
    return op;
  }
  for (const spirv::VertexBufferBinding& vb : p->vertexBuffers) {
    const BoundState::VB& b = r.bound.vertex[vb.binding < 16 ? vb.binding : 0];
    if (!b.buffer || !b.buffer->id) continue;  // unused by the shader or not bound
    vpgpu::Access a{vpgpu::ResourceRange::bytes(b.buffer->id, b.offset, b.buffer->size - b.offset), vpgpu::stage::vertex, false};
    a.kind = vpgpu::BindKind::Buffer, a.slot = vb.slot;
    d.bindings.push_back(a);
  }
  if (indexed) {
    if (!r.bound.index || !r.bound.index->id) {
      op.ok = false;
      logSkip("no_index_buffer", p);
      return op;
    }
    d.index = vpgpu::IndexBuffer{r.bound.index->id, r.bound.indexOffset, r.bound.indexWide};
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
      op.ok = false;  // nothing visible
      return op;
    }
    d.scissor = {uint32_t(x0), uint32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0)};
  }
  if (p->depthBiasEnable) d.depthBias = r.bound.depthBias[0], d.depthBiasClamp = r.bound.depthBias[1], d.depthBiasSlope = r.bound.depthBias[2];
  d.depthClamp = p->depthClamp;
  if (p->cullBoth) op.ok = false;
  d.stencilRef = static_cast<uint8_t>(r.bound.stencilRef);
  std::memcpy(d.blendColor.data(), r.bound.blend, sizeof r.bound.blend);
  d.frontFace = p->frontFace == VK_FRONT_FACE_CLOCKWISE ? vpgpu::Winding::Clockwise : vpgpu::Winding::CounterClockwise;
  return op;
}

OpDispatch makeDispatch(Recorder& r, uint32_t x, uint32_t y, uint32_t z) {
  OpDispatch op;
  const PipelineObj* p = r.bound.compute;
  if (!p || !p->ok) {
    op.ok = false;
    logSkip(p ? "compute_failed" : "no_compute_pipeline", p);
    return op;
  }
  op.dispatch.pipeline = p->key;
  op.dispatch.groups = {x, y, z};
  op.dispatch.label = p->state.label;
  if (!resolveBindings(r, *p, r.bound.computeSets, op.dispatch.bindings, op.dispatch.views)) {
    op.ok = false;
    logSkip("unbound_descriptor", p);
  }
  return op;
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
  Recorder& r = rec(cb);
  OpDraw op = makeDraw(r, 0, 1, 1, 0, 0, false);
  op.indirect = from<Buffer>(buf), op.indirectOffset = off, op.indirectCount = n, op.indirectStride = stride ? stride : 16;
  r.ops.emplace_back(std::move(op));
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off,
                                                                uint32_t n, uint32_t stride) {
  Recorder& r = rec(cb);
  OpDraw op = makeDraw(r, 0, 1, 1, 0, 0, true);
  op.indirect = from<Buffer>(buf), op.indirectOffset = off, op.indirectCount = n, op.indirectStride = stride ? stride : 20;
  r.ops.emplace_back(std::move(op));
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
  Recorder& r = rec(cb);
  r.ops.emplace_back(makeDispatch(r, x, y, z));
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off) {
  Recorder& r = rec(cb);
  OpDispatch op = makeDispatch(r, 1, 1, 1);
  op.indirect = from<Buffer>(buf), op.indirectOffset = off;
  r.ops.emplace_back(std::move(op));
}

// =================================================================================================
// Passes: subpasses, dynamic rendering, clears inside a pass, secondary command buffers
// =================================================================================================
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass(VkCommandBuffer cb, VkSubpassContents) {
  Recorder& r = rec(cb);
  r.ops.emplace_back(OpNextSubpass{});
  ++r.bound.subpass;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass2(VkCommandBuffer cb, const VkSubpassBeginInfo*, const VkSubpassEndInfo*) {
  vkCmdNextSubpass(cb, VK_SUBPASS_CONTENTS_INLINE);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  Recorder& r = rec(cb);
  OpBeginRendering op;
  vpgpu::RenderPassDesc& d = op.desc;
  std::vector<std::pair<const ImageView*, const ImageView*>> forcedPairs;
  auto fill = [&](const VkRenderingAttachmentInfo& a, vpgpu::Attachment& t, bool depth) {
    const ImageView* v = a.imageView ? from<ImageView>(a.imageView) : nullptr;
    if (!v || !v->image || !v->image->id) return;
    t.texture = v->image->id;
    t.mip = uint16_t(v->range.baseMipLevel), t.layer = uint16_t(v->range.baseArrayLayer);
    t.load = toLoad(a.loadOp);
    bool resolve = false;
    if (a.resolveMode != VK_RESOLVE_MODE_NONE && a.resolveImageView) {
      const ImageView* rv = from<ImageView>(a.resolveImageView);
      if (rv && rv->image && rv->image->id && v->image->forcedFrom > 1) {
        // Game MSAA forced to 1x: draw into the 1x source, store it, copy it into the resolve
        // target after the pass (see forcedResolveCopy in commands.cpp for why not render direct).
        t.store = vpgpu::StoreAction::Store;
        if (depth) t.clear.depth = a.clearValue.depthStencil.depth, t.clear.stencil = uint8_t(a.clearValue.depthStencil.stencil);
        else
          for (int i = 0; i < 4; ++i) t.clear.color[i] = a.clearValue.color.float32[i];
        forcedPairs.push_back({v, rv});
        return;
      }
      if (rv && rv->image && rv->image->id) {
        t.resolve = rv->image->id;
        t.resolveMip = uint16_t(rv->range.baseMipLevel), t.resolveLayer = uint16_t(rv->range.baseArrayLayer);
        resolve = true;
      }
    }
    t.store = toStore(a.storeOp, resolve);
    if (depth) t.clear.depth = a.clearValue.depthStencil.depth, t.clear.stencil = uint8_t(a.clearValue.depthStencil.stencil);
    else
      for (int i = 0; i < 4; ++i) t.clear.color[i] = a.clearValue.color.float32[i];
  };
  for (uint32_t i = 0; i < ri->colorAttachmentCount && i < vpgpu::kMaxColorAttachments; ++i)
    fill(ri->pColorAttachments[i], d.colors[i], false);
  if (ri->pDepthAttachment) fill(*ri->pDepthAttachment, d.depth, true);
  else if (ri->pStencilAttachment) fill(*ri->pStencilAttachment, d.depth, true);
  d.layerCount = ri->viewMask ? uint16_t(std::popcount(ri->viewMask)) : uint16_t(std::max<uint32_t>(1, ri->layerCount));
  d.label = "vk.rendering";
  for (const auto& [src, dst] : forcedPairs) op.resolveCopies.push_back(forcedResolveCopy(src, dst, std::max<uint16_t>(d.layerCount, 1)));
  r.bound.targetWidth = uint32_t(ri->renderArea.offset.x) + ri->renderArea.extent.width;
  r.bound.targetHeight = uint32_t(ri->renderArea.offset.y) + ri->renderArea.extent.height;
  r.ops.emplace_back(std::move(op));
  r.inPass = true;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRendering(VkCommandBuffer cb) {
  Recorder& r = rec(cb);
  r.ops.emplace_back(OpEndPass{});
  r.inPass = false;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearAttachments(VkCommandBuffer cb, uint32_t n, const VkClearAttachment* a,
                                                             uint32_t, const VkClearRect*) {
  rec(cb).ops.emplace_back(OpClearAttachments{std::vector<VkClearAttachment>(a, a + n)});
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
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpCopyImageToBuffer{from<Image>(src), from<Buffer>(dst), regions[i]});
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout, VkImage dst,
                                                      VkImageLayout, uint32_t n, const VkImageCopy* regions) {
  Recorder& r = rec(cb);
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpCopyImage{from<Image>(src), from<Image>(dst), regions[i]});
}

// A blit region is a plain copy only when the bytes can move unchanged: same format (or integer
// formats of one size, which a blit converts bit-exactly in practice) and same size. Anything else
// (compressed or sRGB source into another format, scaling, mirroring) is a sampled blit drawn by
// the backend, except a mip chain over one image (Unity's GenerateMips), which the backend builds.
namespace {
bool rawBlitCompatible(vpgpu::Format a, vpgpu::Format b) {
  if (a == b) return true;
  const vpgpu::FormatInfo &fa = vpgpu::formatInfo(a), &fb = vpgpu::formatInfo(b);
  return fa.integer() && fb.integer() && fa.blockSize == 1 && fb.blockSize == 1 && fa.bytesPerBlock == fb.bytesPerBlock;
}
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
    if (sw == dw && sh == dh && sd == dd && sw > 0 && sh > 0 && rawBlitCompatible(s->desc.format, d->desc.format)) {
      VkImageCopy c{};
      c.srcSubresource = b.srcSubresource, c.dstSubresource = b.dstSubresource;
      c.srcOffset = b.srcOffsets[0], c.dstOffset = b.dstOffsets[0];
      c.extent = {uint32_t(sw), uint32_t(sh), uint32_t(std::max(1, sd))};
      vkCmdCopyImage(cb, src, sl, dst, dl, 1, &c);
      continue;
    }
    Recorder& r = rec(cb);
    // Mip chain: mip m -> m + 1 of one image (Unity's GenerateMips). One request builds the chain.
    if (src == dst && b.dstSubresource.mipLevel == b.srcSubresource.mipLevel + 1) {
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
    if (s->desc.type == vpgpu::TextureType::Tex3D || d->desc.type == vpgpu::TextureType::Tex3D) {
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 8) logf("vk.unimpl blit_3d %dx%dx%d->%dx%dx%d", sw, sh, sd, dw, dh, dd);
      continue;
    }
    r.ops.emplace_back(OpBlitImage{s, d, b, filter});
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
                                                         VkImageLayout dl, uint32_t n, const VkImageResolve* regions) {
  // A source forced to 1x (game MSAA off under VPKTAA) resolves as a plain copy.
  if (src && from<Image>(src)->forcedFrom > 1 && regions) {
    for (uint32_t i = 0; i < n; ++i) {
      VkImageCopy c{regions[i].srcSubresource, regions[i].srcOffset, regions[i].dstSubresource, regions[i].dstOffset,
                    regions[i].extent};
      vkCmdCopyImage(cb, src, sl, dst, dl, 1, &c);
    }
    return;
  }
  static std::atomic<unsigned> k{0};
  if (k.fetch_add(1) < 8) log("vk.unimpl resolve_image (render-pass resolves are supported)");
}

// =================================================================================================
// Events in command buffers, queries (no GPU queries: zero results)
// =================================================================================================
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(VkCommandBuffer, VkEvent e, VkPipelineStageFlags) {
  from<Event>(e)->set = true;  // in-order queue: the work before it is ordered anyway
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(VkCommandBuffer, VkEvent e, VkPipelineStageFlags) {
  from<Event>(e)->set = false;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(VkCommandBuffer cb, uint32_t, const VkEvent*, VkPipelineStageFlags,
                                                       VkPipelineStageFlags, uint32_t, const VkMemoryBarrier*, uint32_t,
                                                       const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
  ++rec(cb).barriers;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice, const VkQueryPoolCreateInfo* ci,
                                                             const VkAllocationCallbacks*, VkQueryPool* out) {
  auto* q = new QueryPoolObj;
  q->type = ci->queryType, q->count = ci->queryCount;
  *out = to<VkQueryPool>(q);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice, VkQueryPool q, const VkAllocationCallbacks*) {
  delete from<QueryPoolObj>(q);
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice, VkQueryPool, uint32_t, uint32_t n, size_t size,
                                                                 void* data, VkDeviceSize stride, VkQueryResultFlags flags) {
  // Timestamps and occlusion read as zero (visible / no time).
  std::memset(data, 0, size);
  (void)n, (void)stride, (void)flags;
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkResetQueryPool(VkDevice, VkQueryPool, uint32_t, uint32_t) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(VkCommandBuffer, VkQueryPool, uint32_t, VkQueryControlFlags) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(VkCommandBuffer, VkQueryPool, uint32_t) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyQueryPoolResults(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t, VkBuffer,
                                                                 VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {}
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
      F(vkCmdBindPipeline), F(vkCmdBindDescriptorSets), F(vkCmdPushConstants), F(vkCmdBindVertexBuffers),
      F(vkCmdBindVertexBuffers2), F(vkCmdBindIndexBuffer), F(vkCmdSetViewport), F(vkCmdSetScissor),
      F(vkCmdSetStencilReference), F(vkCmdSetBlendConstants), F(vkCmdSetLineWidth), F(vkCmdSetDepthBias),
      F(vkCmdSetDepthBounds), F(vkCmdSetStencilCompareMask), F(vkCmdSetStencilWriteMask), F(vkCmdDraw), F(vkCmdDrawIndexed),
      F(vkCmdDrawIndirect), F(vkCmdDrawIndexedIndirect), F(vkCmdDispatch), F(vkCmdDispatchIndirect), F(vkCmdNextSubpass),
      F(vkCmdNextSubpass2), F(vkCmdBeginRendering), F(vkCmdEndRendering), F(vkCmdClearAttachments),
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
