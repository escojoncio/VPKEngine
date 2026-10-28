// Internal: what the linker decided for one program, which the SPIR-V emitter applies to each stage.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "ast.h"

namespace vpglsl {

struct DefaultMember {
    std::string name;  // GLSL name of the uniform
    Type type;
};

struct ProgramLayout {
    // Default uniform block: every non-opaque uniform of the program, in this order (std140).
    std::vector<DefaultMember> defaultBlock;
    // Samplers: name -> binding (one binding per sampler uniform; arrays = descriptor arrays).
    std::map<std::string, uint32_t> samplerBinding;
    // Uniform blocks by block name -> binding.
    std::map<std::string, uint32_t> blockBinding;
    // Shader storage blocks by block name -> binding.
    std::map<std::string, uint32_t> storageBinding;
    // Image uniforms by name -> binding (arrays = descriptor arrays).
    std::map<std::string, uint32_t> imageBinding;
    // Atomic counter buffers: GL binding point -> descriptor binding.
    std::map<int, uint32_t> atomicBinding;
    // Framebuffer fetch: colour location -> input attachment descriptor binding, and how many
    // gl_LastFragData elements the shader reads (0 = none).
    std::map<int, uint32_t> fbFetchBinding;
    int lastFragDataCount = 0;
    // Stage interface by name.
    std::map<std::string, int> attribLocation;   // vertex inputs
    std::map<std::string, int> varyingLocation;  // fragment inputs by name
    std::map<std::string, int> vertexOutLocation;  // vertex outputs by name (differs from the fragment name when paired by location)
    std::set<std::string> unwrittenInput;        // fragment inputs no vertex output feeds: read as zero
    std::map<std::string, int> outputLocation;   // fragment outputs
    int fragDataCount = 1;                       // gl_FragData elements emitted
    // Transform feedback capture variant of the vertex shader (empty: the ordinary shader).
    struct XfbItem {
        std::string name;  // variable ("gl_Position", "gl_PointSize" or a vertex output)
        int first = 0;     // first array element captured
        int count = 1;     // elements captured
        bool array = false;  // the variable is an array
        uint32_t offset = 0;  // word offset in the record
    };
    std::vector<XfbItem> xfb;
    uint32_t xfbRecordWords = 0;
};

// One stage of a linked program as SPIR-V 1.3 words (entry point "main").
std::vector<uint32_t> emitSpirv(const Shader& sh, const ProgramLayout& layout, std::string& error);

// Visits every expression of a shader (pre-order).
void forEachExpr(const Shader& sh, const std::function<void(const Expr&)>& fn);

}  // namespace vpglsl
