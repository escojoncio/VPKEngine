// Contract between VPVulkan (gpu/vulkan) and the SPIR-V reader (gpu/shader/spirv, library `vpspirv`).
// The reader implements the three functions below; until the `vpspirv` target exists the driver links
// src/spirv_stub.cpp (every module fails to lower, logged as `vk.err what=shader`).
//
// Slots are assigned by the DRIVER from the pipeline layout and passed in `SpirvLowerOptions`, so the
// vertex and fragment stages of one pipeline agree on them (the Metal backend fills one argument
// table per stage from one list of draw bindings looked up by (kind, slot)). The reader must use
// exactly these slots for the VPIR resources it declares and report which ones the entry point uses.
//
// Lowering rules the driver relies on:
//  * Vertex inputs: there are no vertex descriptors. For each `Location` input, the reader emits a
//    load from the Buffer slot `vertexBuffers[binding].slot` at
//      (vertex_id or instance_id per `perInstance`) * stride + offset
//    and converts per `format` (vpgpu::Format, as the driver translated the VkFormat). VertexId must
//    already include the draw's base vertex (Metal [[vertex_id]] does).
//  * Push constants: one ConstantBuffer at `pushConstantSlot`, byte offsets as in SPIR-V.
//  * Combined image samplers: Texture at `slot + element`, Sampler at `samplerSlot + element`.
//  * Input attachments (subpassLoad): a Texture at the given slot read with `image_fetch` at
//    `convert(position.xy) : vec2<u32>` (builtin Position), layer 0 / current view. The driver then
//    runs vpir::toFramebufferFetch on that slot, which recognises exactly this pattern.
//  * Multiview: `ViewIndex` becomes the render-target layer index (Builtin::Layer output in the vertex
//    stage when viewCount > 1, and the layer input in the fragment stage).
//  * Texel buffers: Buffer slot, byte-addressed; the element format is `texelFormats` of the binding
//    when known, otherwise the SPIR-V image format.
//  * Specialisation constants are folded before lowering.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "vpgpu/core/commands.h"
#include "vpir/ir.h"

namespace spirv {

struct SpirvModule;  // opaque, owned by the reader

// Parses and validates the words once (vkCreateShaderModule). Null + error on failure.
SpirvModule* parse(const uint32_t* words, size_t wordCount, std::string& error);
void release(SpirvModule*);

// Set value for vertex-buffer bindings in SpirvBinding::set.
inline constexpr uint32_t kVertexBufferSet = 0xFFFFFFFFu;

struct SpirvBinding {
  uint32_t set = 0, binding = 0, arraySize = 1;
  vpgpu::BindKind kind = vpgpu::BindKind::Buffer;  // Buffer, ConstantBuffer, Texture, StorageTexture, Sampler
  uint16_t slot = 0;                                // first slot; element e uses slot + e
  uint16_t samplerSlot = 0xFFFF;                    // combined image samplers only
  bool written = false;                             // reported by the reader: the shader stores to it
  vpgpu::Format texelFormat = vpgpu::Format::Undefined;  // texel buffers when known
};

struct VertexAttribute {
  uint32_t location = 0, binding = 0, offset = 0;
  vpgpu::Format format = vpgpu::Format::Undefined;
  // Components actually present (0 = all of `format`'s). Vertex formats with three components have
  // no texture twin: RGB32 is passed as RGBA32 with components = 3 (the fourth is never read).
  uint8_t components = 0;
};
struct VertexBufferBinding {
  uint32_t binding = 0, stride = 0;
  bool perInstance = false;
  uint16_t slot = 0;  // Buffer slot assigned by the driver
};

struct SpirvLowerOptions {
  std::string entry;  // OpEntryPoint name
  vpir::Stage stage = vpir::Stage::Vertex;
  std::vector<std::pair<uint32_t, uint64_t>> specConstants;  // SpecId -> value bits (zero-extended)
  uint32_t viewCount = 1;                                     // multiview: views in the subpass
  std::vector<SpirvBinding> layout;                           // every (set, binding) of the layout
  int pushConstantSlot = -1;                                  // ConstantBuffer slot, -1 = none
  std::vector<VertexAttribute> vertexAttributes;              // vertex stage only
  std::vector<VertexBufferBinding> vertexBuffers;             // vertex stage only
};

struct SpirvInterface {
  std::vector<SpirvBinding> bindings;  // subset of options.layout the entry point uses (+ written)
  bool usesPushConstants = false;
  uint8_t colorOutputs = 0;            // fragment: bit n = writes Location n
  std::array<uint32_t, 3> localSize{1, 1, 1};  // compute
};

bool lower(const SpirvModule&, const SpirvLowerOptions&, vpir::Module& out, SpirvInterface& iface,
           std::string& error);

}  // namespace spirv
