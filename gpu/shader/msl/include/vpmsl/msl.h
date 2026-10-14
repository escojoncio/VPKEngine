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
//       BufferSize              ulong: byte size of the range bound to Buffer slot `index` (SPIR-V
//                               OpArrayLength); last kind, so shaders without it keep their layout
//   * Alignment: constant-buffer addresses must be 16-byte aligned (vector reads from them are
//     emitted as aligned vector loads); device buffers only need 4 (vectors at offsets not known to
//     be aligned go through packed_* types).
//   * The backend makes every referenced resource resident (useResource / residency set).
//   * Compute, object and mesh threadgroup sizes come from MslShader::workgroupSize at dispatch.
//
// Framebuffer fetch: each attachment read with framebuffer_fetch becomes an entry parameter
// `T vp_fbN [[color(N)]]` (Apple GPUs read it from tile memory; the render pass must keep that colour
// attachment bound with a matching pixel format class). Only colour attachments can be read: a pass
// that needs depth at its own pixel must also write depth to a colour attachment (e.g. r32Float) in
// the producing pass and fetch that. Under per-sample shading the value is the current sample's.
//
// Subgroups: Metal SIMD groups are 32 lanes, so modules with wave_size 64 are rejected when they use
// subgroup ops or subgroup builtins. Emulating a 64-lane wave from two SIMD groups would need
// threadgroup memory and barriers, which fragment shaders lack and divergent control flow forbids;
// the frontend lowers wave-64 code instead (most GCN wave ops become per-lane or wave-32 forms).
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
    // Fragment: bit n = the shader writes [[color(n)]] (every declared colour output is written).
    // Attachments outside the mask must keep their contents (write mask none): a fetch-only
    // G-buffer attachment stays intact under the lighting pipeline.
    uint8_t colorOutputs = 0;
};

struct EmitResult {
    MslShader shader;
    std::vector<std::string> errors;  // verifier errors; the shader is empty when set
};

// Content hash of the MSL text alone (FNV-1a, never 0): what the Metal compiler cache keys on.
inline uint64_t sourceHash(const std::string& source) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : source) h = (h ^ c) * 0x100000001b3ull;
    return h ? h : 1;
}

// Shader id: the source plus the argument layout. Frontends may name resources by position (the
// SPIR-V one does), so one source can come with different absolute slots; those are different
// shaders to the backend (it binds by slot) even though they share one Metal library.
inline uint64_t shaderId(const MslShader& s) {
    uint64_t h = sourceHash(s.source);
    for (const Argument& a : s.arguments) {
        const uint64_t v = uint64_t(a.kind) << 56 ^ uint64_t(a.index) << 24 ^ a.offset;
        for (int i = 0; i < 8; ++i) h = (h ^ ((v >> (8 * i)) & 0xff)) * 0x100000001b3ull;
    }
    return h ? h : 1;
}

struct EmitOptions {
    // Name argument-buffer members by position within their kind (b0, b1, … in slot order) instead
    // of the module's resource names. The source then depends only on which resources the shader
    // uses and their order, not on their absolute slots (a pipeline layout with bindings of other
    // stages, push constants after every set), so the precompile of a module alone and its use in
    // a pipeline give the same text and the Metal compiler cache hits. Absolute slots stay in
    // MslShader::arguments: a client that sets this must key shaders by shaderId(), not by the
    // source alone, or two layouts of one source would share an entry.
    bool positionalNames = false;
};

// One-line description of what a shader id is made of, for cache-miss diagnostics:
// "src=<sourceHash> bytes=<n> args=<kind>:<slot>,… wg=x,y,z colors=0x..".
std::string describeKey(const MslShader& s);

// Verifies the module (run vpir::toFramebufferFetch before, so its output is checked too), runs
// vpir::optimize on a copy, verifies that again (a pass bug becomes an error, not bad MSL) and
// translates it.
EmitResult emitMsl(const vpir::Module& module, const EmitOptions& options = {});

// Fingerprint of the translator sources this library was built from (SPIR-V frontend, VPIR,
// MSL emitter; hashed at configure time). Equal fingerprints produce equal MSL for equal inputs,
// so caches keyed by MSL (precompile index, recipes) can be checked against it across builds.
const char* translatorFingerprint();

}  // namespace vpmsl
