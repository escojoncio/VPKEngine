// VPIR -> Metal Shading Language (Metal 3.1, visionOS).
//
// The output is meant to be lean: no bounds checks, constant buffers in the `constant` address
// space, fast math except for instructions marked `precise`, structured control flow emitted as-is
// (loops stay loops), constants folded into the expressions that use them.
//
// Argument buffer contract (the Metal backend writes it, the shader only reads it):
//   * Every stage takes one argument buffer, `constant VpArgs& vp_args [[buffer(0)]]`, omitted when the
//     module declares no resources. Vertex buffers fetched through stage_in use buffer(1) and up.
//   * VpArgs has one 8-byte entry per resource, ordered by (ResourceKind enum order, slot index);
//     MslShader::arguments lists them with their byte offsets.
//       Buffer, ConstantBuffer  64-bit GPU address (MTLBuffer.gpuAddress + byte offset)
//       Texture, StorageTexture MTLResourceID (MTLTexture.gpuResourceID)
//       Sampler                 MTLResourceID (descriptor with supportArgumentBuffers = YES; depth
//                               textures sampled with `compare` need the sampler's compareFunction)
//   * Alignment: constant-buffer addresses must be 16-byte aligned (vector reads from them are
//     emitted as aligned vector loads); device buffers only need 4 (vectors at offsets not known to
//     be aligned go through packed_* types).
//   * The backend makes every referenced resource resident (useResource / residency set).
//   * Compute, object and mesh threadgroup sizes come from MslShader::workgroupSize at dispatch.
//
// Entry point: `vp_main`; builtin inputs become its parameters `in_<name>`, helper functions become
// `fn_<name>`, and `discard` is `discard_fragment()`.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vpir/ir.h"

namespace vpmsl {

struct Argument {
    std::string name;
    vpir::ResourceKind kind;
    uint32_t index;   // VPIR slot index
    uint32_t offset;  // byte offset inside VpArgs
};

struct MslShader {
    std::string source;
    std::vector<Argument> arguments;  // VpArgs layout; size in bytes = 8 * arguments.size()
    std::array<uint32_t, 3> workgroupSize{1, 1, 1};
};

struct EmitResult {
    MslShader shader;
    std::vector<std::string> errors;  // verifier errors; the shader is empty when set
};

// Verifies the module and translates it.
EmitResult emitMsl(const vpir::Module& module);

}  // namespace vpmsl
