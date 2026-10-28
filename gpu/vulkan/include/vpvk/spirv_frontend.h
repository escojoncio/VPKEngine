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
//    With a Y'CbCr conversion (SpirvBinding::ycbcr) the other planes are Textures at
//    `ycbcr.planeSlot[k] + element`; every sampling op reconstructs the chroma, applies the
//    conversion and returns RGBA (ycbcr.h).
//  * Input attachments (subpassLoad): a Texture at the given slot read with `image_fetch` at
//    `convert(position.xy) : vec2<u32>` (builtin Position), layer 0 / current view. The driver then
//    runs vpir::toFramebufferFetch on that slot, which recognises exactly this pattern.
//  * Multiview: `ViewIndex` becomes the render-target layer index (Builtin::Layer output in the vertex
//    stage when viewCount > 1, and the layer input in the fragment stage). With `amplify`, the view is
//    [[amplification_id]] (two views) and travels to the fragment stage in a flat varying.
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
#include "vpvk/ycbcr.h"

namespace spirv {

struct SpirvModule;  // opaque, owned by the reader

// Parses and validates the words once (vkCreateShaderModule). Null + error on failure.
SpirvModule* parse(const uint32_t* words, size_t wordCount, std::string& error);
void release(SpirvModule*);
// Does the module decorate a variable or a member BuiltIn Layer? A vertex shader that does cannot use vertex amplification
// (SpirvInterface::amplified): the reason a multiview draw is not amplified is "layer" then, else the ClipDistance it writes.
bool declaresLayer(const SpirvModule&);

// Set value for vertex-buffer bindings in SpirvBinding::set.
inline constexpr uint32_t kVertexBufferSet = 0xFFFFFFFFu;

struct SpirvBinding {
  uint32_t set = 0, binding = 0, arraySize = 1;
  vpgpu::BindKind kind = vpgpu::BindKind::Buffer;  // Buffer, ConstantBuffer, Texture, StorageTexture, Sampler
  uint16_t slot = 0;                                // first slot; element e uses slot + e
  uint16_t samplerSlot = 0xFFFF;                    // combined image samplers only
  bool written = false;                             // reported by the reader: the shader stores to it
  vpgpu::Format texelFormat = vpgpu::Format::Undefined;  // texel buffers when known
  // Combined image sampler whose immutable sampler has a Y'CbCr conversion: sampling reads the
  // planes (Texture at `slot` is plane 0, ycbcr.planeSlot the others) and converts to RGBA.
  YcbcrBinding ycbcr;
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
  // Multiview mask when it is not the low viewCount bits (e.g. 0b101, 0b100): view k renders to
  // layer = index of the mask's k-th set bit. 0 = views 0..viewCount-1. Recipes: trailing, if set.
  uint32_t viewMask = 0;
  std::vector<SpirvBinding> layout;                           // every (set, binding) of the layout
  int pushConstantSlot = -1;                                  // ConstantBuffer slot, -1 = none
  std::vector<VertexAttribute> vertexAttributes;              // vertex stage only
  std::vector<VertexBufferBinding> vertexBuffers;             // vertex stage only
  // Multisampled images were created 1x (VPVulkan forcing the game's MSAA off): read them as
  // plain 2D textures (sample 0), textureSamples() = 1. Recipes store it only when set.
  bool multisampleAsSingle = false;
  // Fragment: colour attachments with an integer format, bit n = location n (low byte signed,
  // high byte unsigned). An integer output of the other signedness is written as the attachment's
  // type (same bits; Metal requires the output type to match the attachment). Recipes: trailing.
  uint16_t colorIntKinds = 0;
  // Fragment, VPKTAA UI mask (gpu/vulkan/src/taa_native.h): an extra float output at [[color(slot)]]
  // (a R8Unorm attachment) that every invocation that reaches the end of main writes with the UI
  // coverage: 1, or with uiMaskAlpha the saturated alpha of colour output 0 times 2. -1 = none (the
  // MSL is then byte-identical to before). Recipes: trailing.
  int8_t uiMaskSlot = -1;
  bool uiMaskAlpha = false;
  // Fragment: alpha-to-coverage drawn at 1x (the game's MSAA forced off): the draw becomes an alpha
  // test, every invocation whose colour output 0 alpha (float vec4) is < 0.5 is discarded at the end
  // of main. false = the MSL is byte-identical to before. Recipes: flags bit 5.
  bool alphaTest = false;
  // Fragment: the pipeline's VkSampleMask (pSampleMask[0], limited to the sample count) when it clears samples: the shader's
  // [[sample_mask]] output (added when it declares none) is ANDed with it. ~0u = none, the MSL is then byte-identical to
  // before. Recipes: emit flags bit 1.
  uint32_t sampleMask = ~0u;
  // Foveation (gpu/vulkan/src/foveation.h): a multiview vertex shader also writes [[viewport_array_index]] = the
  // view (a rasterization rate map's layer follows the viewport index, not the render target layer), and a multiview
  // fragment shader that reads FragCoord sees it mapped back from physical to screen space through the pass's
  // rate map (vpir::Module::rateMapPosition; the shader then takes the map data at vpmsl::kRateMapBuffer).
  // false = the MSL is byte-identical to before. Recipes: flags bits 6 and 7.
  bool viewportIndex = false;
  bool rateMapPosition = false;
  // Translator modes (process-wide, VPVulkan's translatorMode(); not part of a recipe input's hash: the recipe store's
  // translator id carries them). amplify: two-view multiview through Metal vertex amplification (see
  // SpirvInterface::amplified). nativeFetch: half vertex attributes via half2. unpackFetch: unorm/snorm vertex attributes of
  // whole 4-byte words (RGBA8, RG16, RGBA16) through the unpack_* ops (MSL unpack_*_to_float). killExit: a fragment
  // shader leaves main once every lane of its SIMD-group is killed or a helper, checked where the kill's branch has
  // reconverged outside any loop. false = the MSL is byte-identical to before.
  bool amplify = false;
  bool nativeFetch = false;
  bool unpackFetch = false;
  bool killExit = false;
  // robustBufferAccess in the emitted MSL (vpmsl::EmitOptions::robustBuffers): every storage buffer access is checked
  // against the buffer's size. Not a lowering input, but it changes the MSL, so it is part of a recipe input: VPVK_ROBUST
  // (config.cpp) decides it per translation. true = the MSL is byte-identical to before. Recipes: trailing, if false.
  bool robustBuffers = true;
};

struct SpirvInterface {
  std::vector<SpirvBinding> bindings;  // subset of options.layout the entry point uses (+ written)
  bool usesPushConstants = false;
  uint8_t colorOutputs = 0;            // fragment: bit n = writes Location n
  std::array<uint32_t, 3> localSize{1, 1, 1};  // compute
  // Vertex: multiview through vertex amplification (SpirvLowerOptions::amplify accepted for this shader): the draw must
  // NOT multiply instances by the view count, and the backend sets the amplification count.
  bool amplified = false;
};

bool lower(const SpirvModule&, const SpirvLowerOptions&, vpir::Module& out, SpirvInterface& iface,
           std::string& error);

}  // namespace spirv
