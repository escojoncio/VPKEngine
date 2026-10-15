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
#include <atomic>
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

// The pipeline key a draw into a pass of `passSamples` (effective count of its attachments; 0 =
// unknown, e.g. a secondary command buffer) uses: the pipeline itself (or its texel buffer variant
// `tv`) when the counts agree, else a variant with the pass's count, requested and recorded
// (recipes) the first time it is needed.
vpgpu::PipelineKey pipelineFor(Device* dev, const PipelineObj& p, uint8_t passSamples, PipelineObj::TexelVariant* tv) {
  const vpgpu::PipelineState& base = tv ? tv->state : p.state;
  const vpgpu::PipelineKey baseKey = tv ? tv->key : p.key;
  if (!passSamples || passSamples == base.samples || p.compute) return baseKey;
  std::lock_guard<std::mutex> lock(p.variantMutex);
  auto& list = tv ? tv->samples : p.variants;
  for (const auto& [n, k] : list)
    if (n == passSamples) return k;
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
    dev->gpu->pipelines().request(st, k, vpgpu::Priority::Blocking);
    if (dev->recipes) dev->recipes->addPipeline(st, tv ? tv->inputs : p.inputs);
  }
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
  std::memcpy(c.host + c.used, data, size);
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
  vpmsl::EmitOptions eo;
  eo.positionalNames = true;  // MSL text independent of absolute slots (precompile == runtime)
  eo.loopGuard = kVulkanLoopGuard;
  vpmsl::EmitResult e = vpmsl::emitMsl(mod, eo);
  if (!e.errors.empty()) {
    errors = std::move(e.errors);
    return TranslateFail::Msl;
  }
  msl = std::move(e.shader);
  id = vpmsl::shaderId(msl);  // source + argument slots
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
                        uint16_t colorIntKinds = 0) {
  StageOut o;
  if (!src.module) {
    logf("vk.err what=shader pipeline=%s stage=%u error=\"no module\"", label.c_str(), unsigned(src.stage));
    return o;
  }
  spirv::SpirvLowerOptions opt;
  opt.entry = src.entry;
  opt.stage = vpirStage(src.stage);
  opt.specConstants = src.spec;
  opt.viewCount = viewCount;
  opt.viewMask = viewMask;
  opt.multisampleAsSingle = forceSingleSample();  // MSAA images are then created 1x (vkCreateImage)
  if (opt.stage == vpir::Stage::Fragment) opt.colorIntKinds = colorIntKinds;
  for (const LayoutSlot& s : pl.slots) {
    spirv::SpirvBinding b;
    b.set = s.set, b.binding = s.binding, b.arraySize = s.count, b.kind = s.kind, b.slot = s.slot, b.samplerSlot = s.samplerSlot;
    if (texel)
      for (const PipelineObj::TexelBinding& t : *texel)
        if (t.set == s.set && t.binding == s.binding) b.texelFormat = t.lowered;
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
  switch (lowerAndEmit(*src.module, opt, mod, o.iface, o.msl, o.id, errors)) {
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
    std::lock_guard<std::mutex> lock(m);
    ++n, total += ms, worst = std::max(worst, ms);
    if (n <= 4 || n % 32 == 0) logf("gpu.shader.msl translated=%u ms_total=%.0f ms_max=%.1f", n, total, worst);
  }
  ShaderBlob blob = blobOf(o.msl);
  if (d->recipes && src.words && !src.words->empty()) {  // translator input, so a later translator can redo it
    StageInput in;
    in.spirv = src.hash;
    in.opt = std::move(opt);
    o.input = in.hash();
    d->recipes->addStage(in, o.input, *src.words, o.id, blob);
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

PipelineObj* createGraphics(Device* d, const VkGraphicsPipelineCreateInfo& ci) {
  auto* p = new PipelineObj;
  p->layout = layoutOf(ci.layout);
  if (!p->layout) p->layout = std::make_shared<PipelineLayoutObj>();
  vpgpu::PipelineState& st = p->state;
  targetFormats(ci, st, p->viewCount, p->viewMask);
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
    p->wireframe = rs->polygonMode == VK_POLYGON_MODE_LINE;
    if (rs->polygonMode == VK_POLYGON_MODE_POINT) {
      static std::atomic<int> k{0};
      if (k.fetch_add(1) < 4) logf("vk.err what=polygon_mode_point_unsupported pipeline=%s (drawn filled)", st.label.c_str());
    }
    p->frontFace = rs->frontFace;
    p->depthBiasEnable = rs->depthBiasEnable;
    p->depthClamp = rs->depthClampEnable;
    p->depthBias[0] = rs->depthBiasConstantFactor, p->depthBias[1] = rs->depthBiasClamp, p->depthBias[2] = rs->depthBiasSlopeFactor;
  }
  if (const auto* ms = ci.pMultisampleState) {
    // The game's count, then forcing to 1x / the override / the device clamp, the same rule as
    // vkCreateImage. A draw into a pass whose images were created with another count gets a
    // variant (pipelineFor), so a setting changed in between never mismatches.
    p->gameSamples = std::max<uint8_t>(1, static_cast<uint8_t>(ms->rasterizationSamples));
    st.samples = effectiveSamples(p->gameSamples);
    st.alphaToCoverage = ms->alphaToCoverageEnable;
  }
  if (const auto* ds = ci.pDepthStencilState; ds && st.depthFormat != vpgpu::Format::Undefined) {
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

  // Integer colour attachments by signedness (fragment outputs take the attachment's type).
  for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c) {
    const vpgpu::FormatInfo& cf = vpgpu::formatInfo(st.colorFormats[c]);
    if (!cf.integer() || !cf.name) continue;
    p->colorIntKinds |= uint16_t(std::strstr(cf.name, "Sint") ? (1u << c) : (0x100u << c));
  }
  bool ok = true;
  std::array<uint64_t, 3> inputs{};  // recipe stage inputs: vertex, fragment, compute
  for (uint32_t i = 0; i < ci.stageCount; ++i) {
    const auto& s = ci.pStages[i];
    if (s.stage != VK_SHADER_STAGE_VERTEX_BIT && s.stage != VK_SHADER_STAGE_FRAGMENT_BIT) {
      logf("vk.err what=stage value=%u pipeline=%s (geometry/tessellation unsupported)", unsigned(s.stage), st.label.c_str());
      ok = false;
      continue;
    }
    p->sources.push_back(stageSrc(s));
    StageOut o = translateStage(d, p->sources.back(), *p->layout, p->viewCount, attrs, p->vertexBuffers, st.label,
                                p->viewMask, nullptr, p->colorIntKinds);
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
  p->inputs = inputs;
  if (p->texel.empty()) p->sources.clear();  // only texel buffer variants retranslate
  else p->attrs = std::move(attrs);
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
  p->sources.push_back(stageSrc(ci.stage));
  StageOut o = translateStage(d, p->sources.back(), *p->layout, 1, {}, {}, p->state.label);
  if (o.ok) {
    p->state.computeShader = o.id;
    collectUsed(*p, o, vpgpu::stage::compute);
    if (p->texel.empty()) p->sources.clear();
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
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_VIEWPORT) && po->hasViewport) b.hasViewport = true, b.viewport = po->viewport;
  if (po && !po->isDynamic(VK_DYNAMIC_STATE_SCISSOR) && po->hasScissor) b.hasScissor = true, b.scissor = po->scissor;
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

// Texel buffer variant for the views bound now (PipelineObj::texel): null when every binding's
// view has the format the pipeline was lowered with (or none is bound yet: resolveBindings
// rejects the draw then). Element 0 of an arrayed binding decides for the whole array.
PipelineObj::TexelVariant* texelVariant(Device* dev, const PipelineObj& p, const std::array<BoundState::Set, 8>& sets) {
  if (!dev || p.texel.empty() || p.sources.empty()) return nullptr;
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
  if (!differs) return nullptr;
  std::lock_guard<std::mutex> lock(p.variantMutex);
  for (PipelineObj::TexelVariant& v : p.texelVariants)
    if (v.formats == want) return v.ok ? &v : nullptr;
  PipelineObj::TexelVariant& v = p.texelVariants.emplace_back();
  v.formats = want;
  v.state = p.state;
  std::vector<PipelineObj::TexelBinding> tb = p.texel;
  std::string fmts;
  for (size_t i = 0; i < tb.size(); ++i) {
    tb[i].lowered = want[i];
    fmts += std::string(fmts.empty() ? "" : ",") + vpgpu::formatInfo(want[i]).name;
  }
  v.state.label += ".tb" + std::to_string(p.texelVariants.size());
  v.ok = true;
  for (const PipelineObj::StageSrc& src : p.sources) {
    StageOut o = translateStage(dev, src, *p.layout, p.viewCount, p.attrs, p.vertexBuffers, v.state.label, p.viewMask, &tb,
                                p.colorIntKinds);
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
    logf("vk.texel_variant pipeline=%s formats=%s ok=%d", v.state.label.c_str(), fmts.c_str(), int(v.ok));
  if (!v.ok) return nullptr;  // the base pipeline is used (and the failure logged by translateStage)
  v.key = v.state.key();
  dev->gpu->pipelines().request(v.state, v.key, vpgpu::Priority::Blocking);
  if (dev->recipes) dev->recipes->addPipeline(v.state, v.inputs);
  return &v;
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
  d.pipeline = pipelineFor(r.device, *p, r.bound.passSamples, texelVariant(r.device, *p, r.bound.graphicsSets));
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
    op.vertexBuffers = true;
  }
  op.depthTested = p->state.depthTest || p->state.depthWrite;
  if (indexed) {
    if (!r.bound.index || !r.bound.index->id) {
      op.ok = false;
      logSkip("no_index_buffer", p);
      return op;
    }
    d.index = vpgpu::IndexBuffer{r.bound.index->id, r.bound.indexOffset, r.bound.indexWide};
  }
  // Triangle fans (Metal has none): an indexed triangle list (v0, vk, vk+1) in the command
  // buffer's upload memory; an indexed fan reads its indices now (unified memory: the buffer the
  // app filled before recording). Indirect fans keep the list topology (not converted).
  if (p->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN && count >= 3) {
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
        if (!indexed) return first + i;
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
      const bool splits = indexed && p->primitiveRestart;
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
        op.ok = false;  // nothing but restarts / short fans: nothing to draw
        return op;
      }
      VkDeviceSize off = 0;
      const vpgpu::BufferId id = r.arena.put(r.device, tri.data(), tri.size() * 4, off);
      d.index = vpgpu::IndexBuffer{id, off, true};
      d.first = 0, d.count = uint32_t(tri.size());
      if (!indexed) d.baseVertex = 0, op.fanConverted = true;  // the list holds absolute vertex ids
      d.primitive = vpgpu::Primitive::Triangles;
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
      op.ok = false;  // nothing visible
      return op;
    }
    d.scissor = {uint32_t(x0), uint32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0)};
  }
  if (p->depthBiasEnable) d.depthBias = r.bound.depthBias[0], d.depthBiasClamp = r.bound.depthBias[1], d.depthBiasSlope = r.bound.depthBias[2];
  d.depthClamp = p->depthClamp;
  d.wireframe = p->wireframe;
  if (p->cullBoth) op.ok = false;
  d.stencilRef = static_cast<uint8_t>(r.bound.stencilRef[0]);
  d.stencilRefBack = static_cast<int16_t>(r.bound.stencilRef[1] & 0xFF);
  for (int f = 0; f < 2; ++f)
    d.stencilReadMask[f] = int16_t(r.bound.stencilCompareMask[f]), d.stencilWriteMask[f] = int16_t(r.bound.stencilWriteMask[f]);
  std::memcpy(d.blendColor.data(), r.bound.blend, sizeof r.bound.blend);
  d.frontFace = p->frontFace == VK_FRONT_FACE_CLOCKWISE ? vpgpu::Winding::Clockwise : vpgpu::Winding::CounterClockwise;
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
// records at or past the GPU-side draw count (vkCmdDraw*IndirectCount).
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
    o[1] = o[1] * views;
    o[words - 1u] = o[words - 1u] * views;
}
)";

bool indirectPatch(Device* d, vpgpu::PipelineKey& key) {
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
    d->indirectPatchOk = true;
  });
  key = d->indirectPatchKey;
  return d->indirectPatchOk;
}

// vkCmdDraw(Indexed)Indirect(Count). GPU-read arguments whenever the draw is recorded inside a
// render pass of this command buffer (the common case): plain draws read the app's buffer
// directly; multiview and draw-count draws read a copy that the patch kernel writes before the
// pass begins (Vulkan forbids writing the argument buffer inside the pass, so it is final there).
// Secondary command buffers, VPVK_GPU_INDIRECT=0 or a missing kernel: arguments read at submit.
void recordIndirectDraw(Recorder& r, const Buffer* b, VkDeviceSize off, uint32_t n, uint32_t stride, bool indexed,
                        const Buffer* countBuf, VkDeviceSize countOff) {
  OpDraw op = makeDraw(r, 0, 1, 1, 0, 0, indexed);
  op.indirect = b, op.indirectOffset = off, op.indirectCount = n, op.indirectStride = stride;
  if (!op.ok || !b || !b->id || n == 0) {
    if (countBuf && op.ok) op.ok = false;  // no GPU count without the GPU path (logged by the skip)
    r.ops.emplace_back(std::move(op));
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
      op.ok = false;
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
  if (!p || !p->ok) {
    op.ok = false;
    logSkip(p ? "compute_failed" : "no_compute_pipeline", p);
    return op;
  }
  const PipelineObj::TexelVariant* tv = texelVariant(r.device, *p, r.bound.computeSets);
  op.dispatch.pipeline = tv ? tv->key : p->key;
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
  // A multiview mask other than views 0..n-1 (0b1000, 0b101): the pass has bit_width(mask) layers
  // and only the mask's ones are this pass's.
  const bool irregular = ri->viewMask && (ri->viewMask & (ri->viewMask + 1)) != 0;
  auto fill = [&](const VkRenderingAttachmentInfo& a, vpgpu::Attachment& t, bool depth) {
    const ImageView* v = a.imageView ? from<ImageView>(a.imageView) : nullptr;
    if (!v || !v->image || !v->image->id) return;
    t.texture = v->image->id;
    t.mip = uint16_t(v->range.baseMipLevel), t.layer = uint16_t(v->range.baseArrayLayer);
    t.load = toLoad(a.loadOp);
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
    if (a.resolveMode != VK_RESOLVE_MODE_NONE && a.resolveImageView) {
      const ImageView* rv = from<ImageView>(a.resolveImageView);
      if (rv && rv->image && rv->image->id && v->image->forcedFrom > 1) {
        // Game MSAA forced to 1x: draw into the 1x source, store it, copy it into the resolve
        // target after the pass (see forcedResolveCopy in commands.cpp for why not render direct).
        t.store = vpgpu::StoreAction::Store;
        forcedPairs.push_back({v, rv});
        return;
      }
      if (rv && rv->image && rv->image->id && !depth && integerResolve(v)) {
        // Integer MSAA (Metal resolves only float formats): store the samples, blit sample 0
        // into the resolve target after the pass (commands.cpp sampleZeroResolve).
        t.store = vpgpu::StoreAction::Store;
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
    if (keepOthers && t.store == vpgpu::StoreAction::DontCare) t.store = vpgpu::StoreAction::Store;
    if (keepOthers && t.store == vpgpu::StoreAction::Resolve) t.store = vpgpu::StoreAction::StoreAndResolve;
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
  if (d.depth.texture && from<ImageView>((da ? da : sa)->imageView)->image->forcedFrom > 1 && !resolving(da) &&
      resolving(sa)) {
    // Forced 1x with only the stencil resolved: fill() saw no resolve on the depth side.
    const ImageView* v = from<ImageView>(sa->imageView);
    const ImageView* rv = from<ImageView>(sa->resolveImageView);
    if (rv && rv->image && rv->image->id) {
      d.depth.store = vpgpu::StoreAction::Store;
      forcedPairs.push_back({v, rv});
    }
  } else if (d.depth.texture && !(from<ImageView>((da ? da : sa)->imageView)->image->forcedFrom > 1)) {
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
    const bool clearD = da && fi.depth() && da->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    const bool clearS = sa && fi.stencil() && sa->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    // An aspect of the image that is not attached keeps its contents (it is loaded, and stored below).
    const bool loadD = fi.depth() && (da ? da->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD : true);
    const bool loadS = fi.stencil() && (sa ? sa->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD : true);
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
    } else if (loadD || loadS || keepOthers) {
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
    if (keep && d.depth.store == vpgpu::StoreAction::Resolve) d.depth.store = vpgpu::StoreAction::StoreAndResolve;
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
  r.passBegin = r.ops.size();
  r.ops.emplace_back(std::move(op));
  r.inPass = true;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRendering(VkCommandBuffer cb) {
  Recorder& r = rec(cb);
  r.ops.emplace_back(OpEndPass{});
  r.inPass = false;
  r.passBegin = SIZE_MAX;
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
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpCopyImageToBuffer{from<Image>(src), from<Buffer>(dst), regions[i]});
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout, VkImage dst,
                                                      VkImageLayout, uint32_t n, const VkImageCopy* regions) {
  Recorder& r = rec(cb);
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpCopyImage{from<Image>(src), from<Image>(dst), regions[i]});
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
    if (sw == dw && sh == dh && sd == dd && sw > 0 && sh > 0 && rawBlitCompatible(s->desc.format, d->desc.format)) {
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
  // A source forced to 1x (game MSAA off under VPKTAA) resolves as a plain copy.
  if (src && from<Image>(src)->forcedFrom > 1 && regions) {
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
  if ((q->type == VK_QUERY_TYPE_OCCLUSION || q->type == VK_QUERY_TYPE_TIMESTAMP) && q->count && q->device &&
      q->device->gpu) {
    q->base = allocVisibility(q->device, q->count);
    if (q->base == UINT32_MAX) logf("vk.err what=query_pool_unbacked count=%u (occlusion reads as visible)", q->count);
  }
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
    if (!avail) notReady = true;
  }
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
      F(vkCmdBindPipeline), F(vkCmdBindDescriptorSets), F(vkCmdPushConstants), F(vkCmdBindVertexBuffers),
      F(vkCmdBindVertexBuffers2), F(vkCmdBindIndexBuffer), F(vkCmdSetViewport), F(vkCmdSetScissor),
      F(vkCmdSetStencilReference), F(vkCmdSetBlendConstants), F(vkCmdSetLineWidth), F(vkCmdSetDepthBias),
      F(vkCmdSetDepthBounds), F(vkCmdSetStencilCompareMask), F(vkCmdSetStencilWriteMask), F(vkCmdDraw), F(vkCmdDrawIndexed),
      F(vkCmdDrawIndirect), F(vkCmdDrawIndexedIndirect), F(vkCmdDispatch), F(vkCmdDispatchIndirect), F(vkCmdNextSubpass),
      F(vkCmdDrawIndirectCount), F(vkCmdDrawIndexedIndirectCount),
      {"vkCmdDrawIndirectCountKHR", reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDrawIndirectCount)},
      {"vkCmdDrawIndexedIndirectCountKHR", reinterpret_cast<PFN_vkVoidFunction>(&vkCmdDrawIndexedIndirectCount)},
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
