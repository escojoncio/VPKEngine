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
//      binding kStorageBase + i  shader storage block i (std430 when declared so, else std140); an
//                                unsized last array is an OpTypeRuntimeArray: its length is the
//                                bound range minus the offset
//      binding kAtomicBase + b   atomic counter buffer at GL binding point b: a storage buffer of
//                                uints, counter k at byte offset Variable::offset
//      binding kImageBase + i    image uniform i (one descriptor per array element, storage image)
//      binding kAttachmentBase+l input attachment of colour location l (framebuffer fetch)
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

// Binding point counts per resource kind (the GL_MAX_*_BINDINGS / unit limits vpgles reports; context.h asserts they agree).
constexpr int kBindingSamplers = 96, kBindingImages = 8, kBindingAtomics = 8, kBindingUniform = 84, kBindingStorage = 36;

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
    // Samplers and images: the texture / image unit declared with layout(binding = N), -1 = none
    // (the program must start with that unit as the uniform's value).
    int unit = -1;
    // Uniforms: the location of element 0 (consecutive for the elements and for the leaves of a struct), explicit from
    // layout(location = N) or the lowest free one; -1 for atomic counters (they have none) and non-uniforms.
    int uniformLocation = -1;
    // Atomic counters (glType GL_UNSIGNED_INT_ATOMIC_COUNTER): the GL buffer binding point they live
    // at (their byte offset in it is `offset`); -1 for every other variable. They have no place in
    // the default uniform block.
    int atomicBinding = -1;
    // Shader storage block members: the top-level array of an array of structs / of arrays is listed through its element 0
    // ("s[0].a", "a[0][0]"); its size (0 = unsized) and the stride between its elements. 1 / 0 for every other variable.
    int topLevelArraySize = 1;
    uint32_t topLevelArrayStride = 0;
    // Shader storage block members: the unsized (runtime) last array has arraySize 0; its stride is
    // arrayStride and `offset` the offset of its first element.
};

struct UniformBlock {
    std::string name;      // block name (GL name; arrays as "B[0]"...)
    uint32_t size = 0;     // std140 / std430 size (storage blocks: without the unsized array)
    int initialBinding = -1;  // layout(binding = N): the GL binding point the block starts at, -1 = none
    uint32_t binding = 0;  // descriptor binding (kBlockBase / kStorageBase + index of the block declaration)
    uint32_t element = 0;  // descriptor array element (block arrays: one GL block per element)
    std::vector<Variable> members;  // offsets within the block
    uint32_t stages = 0;
};

inline constexpr uint32_t kDefaultBlockBinding = 0;
inline constexpr uint32_t kBlockBase = 64;  // uniform blocks: binding 64 + block index (up to 84, Quest 3 GL_MAX_UNIFORM_BUFFER_BINDINGS)
inline constexpr uint32_t kStorageBase = 152;     // shader storage blocks (up to 36, Quest 3 GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)
inline constexpr uint32_t kAtomicBase = 192;      // atomic counter buffers: 192 + GL binding point (< 16)
inline constexpr uint32_t kImageBase = 224;       // image uniforms: 224 + image uniform index (< 16)
inline constexpr uint32_t kAttachmentBase = 240;  // framebuffer fetch input attachments: 240 + colour location (< 8)
// Transform feedback capture (the vertex shader variant LinkResult::xfbVertexSpirv): one storage buffer at kXfbBinding laid
// out as uint header[kXfbHeaderWords] + uint data[]. header[0] = vertices per instance (slot span), [1] = the vertex index the
// slots start at, [2..3] reserved. The record of (instance i, vertex index v) is Reflection::xfbRecordWords words at
// data[((i * span) + (v - base)) * xfbRecordWords].
inline constexpr uint32_t kXfbBinding = 252;
inline constexpr uint32_t kXfbHeaderWords = 4;

struct Reflection {
    bool flatVaryings = false;  // a fragment input is `flat` (GL takes it from a primitive's LAST vertex, Vulkan/Metal from the first)
    std::vector<Variable> attributes;    // vertex inputs (location assigned)
    std::vector<Variable> uniforms;      // default block + samplers
    std::vector<UniformBlock> blocks;
    std::vector<UniformBlock> storageBlocks;  // shader storage blocks (binding = kStorageBase + index)
    // Atomic counter buffers in use: GL binding point, descriptor binding (kAtomicBase + point) and
    // byte size (highest counter end).
    struct AtomicBuffer {
        int glBinding = 0;
        uint32_t binding = 0, size = 0;
        uint32_t stages = 0;
    };
    std::vector<AtomicBuffer> atomicBuffers;
    // Image uniforms use binding kImageBase + i (Variable::binding, glType GL_IMAGE_*): descriptor
    // kind StorageTexture. Framebuffer fetch: colour locations read, each an input attachment at
    // kAttachmentBase + location (bit n = location n); 0 = the program does not fetch.
    uint32_t framebufferFetchMask = 0;
    uint32_t sharedBytes = 0;            // compute: workgroup memory used
    // glTransformFeedbackVaryings of the link (in the order given): what the capture record holds.
    struct XfbVarying {
        std::string name;        // as given ("v", "v[1]", "gl_Position")
        uint32_t glType = 0;     // type of one element (GL_FLOAT_VEC3, GL_FLOAT_MAT2...)
        int arraySize = 1;       // elements captured: the array's size; 1 for a single element and for non-arrays
        uint32_t words = 0;      // 32-bit words of one element
        uint32_t offset = 0;     // word offset of the first element within the record
    };
    std::vector<XfbVarying> xfb;
    uint32_t xfbRecordWords = 0;         // words of one captured vertex (all varyings)
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
    // Strict-mode rules the shader breaks that lenient mode (the default) let through: one entry per rule, in
    // order of appearance. Empty in strict mode (they were errors) and for conforming shaders.
    std::vector<std::string> lenient;
};

// Preprocesses, parses and type-checks one shader source (the concatenated glShaderSource strings).
ParseResult parse(Stage stage, const std::string& source);

struct LinkOptions {
    std::map<std::string, int> attribBindings;  // glBindAttribLocation
    std::vector<std::string> xfbVaryings;       // glTransformFeedbackVaryings (a link without them captures nothing)
    bool xfbSeparate = false;                   // GL_SEPARATE_ATTRIBS (limits differ from interleaved)
    bool separable = false;                     // GL_PROGRAM_SEPARABLE: a fragment input no vertex output feeds is not an error
};

struct LinkResult {
    bool ok = false;
    std::string log;
    // Diagnostic, empty unless a fragment input is unwritten or its varying fails to link:
    // "vs_out=[name@location ...] fs_in=[name@location!why ...]" (location "-" = none declared).
    std::string varyingDiag;
    std::vector<std::string> lenient;  // see ParseResult::lenient (link-time rules)
    std::vector<uint32_t> vertexSpirv, fragmentSpirv, computeSpirv;
    std::vector<uint32_t> xfbVertexSpirv;  // the vertex shader that also writes the Reflection::xfb record (empty without varyings)
    Reflection reflection;
};

// Links a vertex + fragment pair (or one compute shader: pass it as `vs`, `fs` null).
LinkResult link(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs, const LinkOptions& options);
// The #version of a parsed shader (100, 300, 310, 320).
int shaderVersion(const Shader& s);

// What the GL layer supports beyond the core language; the extension macros and the `#extension`
// behaviour follow it (an extension the layer cannot back is neither defined nor accepted).
struct Caps {
    // GL_EXT_shader_framebuffer_fetch / GL_ARM_shader_framebuffer_fetch: needs the draw to bind the
    // colour attachment as an input attachment at Reflection::framebufferFetchMask.
    bool framebufferFetch = false;
    // GL_OES_texture_cube_map_array / GL_EXT_texture_cube_map_array in ES 3.10 shaders (ES 3.20
    // shaders always have cube map arrays).
    bool cubeMapArray = false;
};
void setCaps(const Caps& caps);  // process-wide, set before compiling
Caps caps();

// Strict GLSL: the compiler of a real driver (Adreno on the headset) is lenient and games ship shaders a strict
// compiler rejects, so every "this must not compile" rule of the language that is not needed to generate code is
// gated: strict mode (VPGLES_STRICT_GLSL=1 in the environment, read once; the conformance CI sets it) makes it an
// error, lenient mode (the default, games) accepts the shader and records the rule in ParseResult::lenient /
// LinkResult::lenient. setStrictMode overrides the environment: 1 strict, 0 lenient, -1 back to the environment.
bool strictMode();
void setStrictMode(int mode);

// Preprocessor only (exposed for tests): the token text after macro expansion, one space between
// tokens and a newline per source line kept.
std::string preprocess(const std::string& source, Stage stage, std::string& log, int& version);

}  // namespace vpglsl
