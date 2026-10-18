// VPKEngine's own GLSL ES compiler (ES 1.00, 3.00, 3.10/3.20 subset) for the GLES layer (gpu/gles).
// Own code: lexer, preprocessor, recursive-descent parser with type checking, SPIR-V emitter. The
// SPIR-V it writes is what gpu/shader/spirv (lower.cpp) reads, so GLES programs go through the same
// VPIR -> MSL path, pipeline cache, MSAA/VPKTAA and foveation as Vulkan games.
//
// Model (what gpu/gles relies on):
//  * glCompileShader = parse(): preprocess + parse + type check. Errors go to the info log.
//  * glLinkProgram = link(): matches varyings by name, assigns attribute / varying / output
//    locations and emits one SPIR-V module per stage with the resource layout below.
//  * Resource layout (descriptor set 0 for everything):
//      binding 0                 the default uniform block (all non-opaque uniforms, std140 offsets
//                                reported in Reflection::uniforms)
//      binding 1 .. S            combined image samplers: one binding per sampler uniform, with one
//                                descriptor per array element
//      binding kBlockBase + i    uniform block i (std140; `shared`/`packed` laid out as std140)
//  * Clip space: GL's z in [-w, w] is remapped to Vulkan's [0, w] in the vertex shader. Y is left
//    alone: the GLES layer keeps GL's row 0 (bottom) as memory row 0, so window coordinates,
//    gl_FragCoord, viewports and texture t coordinates already agree; only presenting the default
//    framebuffer flips (done by whoever presents it).
//  * GL_OVR_multiview(2): layout(num_views = N) in; gl_ViewID_OVR = the SPIR-V ViewIndex.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpglsl {

enum class Stage : uint8_t { Vertex, Fragment, Compute };

// GL type enums (GL_FLOAT_VEC4, GL_SAMPLER_2D...) as glGetActiveUniform reports them.
struct Variable {
    std::string name;      // full GL name ("light.color", "bones[0]" for arrays)
    uint32_t glType = 0;   // GL_FLOAT_VEC3...
    int arraySize = 1;     // elements (1 for non-arrays)
    bool isArray = false;
    int location = -1;     // attributes / fragment outputs: the location; uniforms: see below
    // Default-block uniforms: byte offset of element 0 in binding 0, and the array stride.
    uint32_t offset = 0, arrayStride = 0, matrixStride = 0;
    // Samplers: binding of element 0 (elements are consecutive).
    int binding = -1;
    uint32_t stages = 0;   // bit per Stage that uses it
    bool rowMajor = false;
};

struct UniformBlock {
    std::string name;      // block name (GL name; arrays as "B[0]"...)
    uint32_t size = 0;     // std140 size
    uint32_t binding = 0;  // descriptor binding (kBlockBase + index of the block declaration)
    uint32_t element = 0;  // descriptor array element (block arrays: one GL block per element)
    std::vector<Variable> members;  // offsets within the block
    uint32_t stages = 0;
};

inline constexpr uint32_t kDefaultBlockBinding = 0;
inline constexpr uint32_t kBlockBase = 64;  // uniform blocks: binding 64 + block index

struct Reflection {
    std::vector<Variable> attributes;    // vertex inputs (location assigned)
    std::vector<Variable> uniforms;      // default block + samplers
    std::vector<UniformBlock> blocks;
    std::vector<Variable> outputs;       // fragment outputs (location assigned)
    uint32_t defaultBlockSize = 0;       // bytes of binding 0 (0 = none)
    uint32_t samplerCount = 0;           // bindings 1..samplerCount
    uint32_t numViews = 1;               // GL_OVR_multiview
    std::array<uint32_t, 3> localSize{1, 1, 1};  // compute
};

class Shader;  // a parsed shader (opaque)

struct ParseResult {
    std::shared_ptr<Shader> shader;  // null when compilation failed
    std::string log;                 // info log (errors and warnings)
    int version = 100;
};

// Preprocesses, parses and type-checks one shader source (the concatenated glShaderSource strings).
ParseResult parse(Stage stage, const std::string& source);

struct LinkOptions {
    std::map<std::string, int> attribBindings;  // glBindAttribLocation
};

struct LinkResult {
    bool ok = false;
    std::string log;
    std::vector<uint32_t> vertexSpirv, fragmentSpirv, computeSpirv;
    Reflection reflection;
};

// Links a vertex + fragment pair (or one compute shader: pass it as `vs`, `fs` null).
LinkResult link(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs, const LinkOptions& options);

// Preprocessor only (exposed for tests): the token text after macro expansion, one space between
// tokens and a newline per source line kept.
std::string preprocess(const std::string& source, Stage stage, std::string& log, int& version);

}  // namespace vpglsl
