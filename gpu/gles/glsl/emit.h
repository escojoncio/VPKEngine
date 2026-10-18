// Internal: what the linker decided for one program, which the SPIR-V emitter applies to each stage.
#pragma once

#include <map>
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
    // Stage interface by name.
    std::map<std::string, int> attribLocation;   // vertex inputs
    std::map<std::string, int> varyingLocation;  // vertex outputs = fragment inputs
    std::map<std::string, int> outputLocation;   // fragment outputs
    int fragDataCount = 1;                       // gl_FragData elements emitted
};

// One stage of a linked program as SPIR-V 1.3 words (entry point "main").
std::vector<uint32_t> emitSpirv(const Shader& sh, const ProgramLayout& layout, std::string& error);

// Visits every expression of a shader (pre-order).
void forEachExpr(const Shader& sh, const std::function<void(const Expr&)>& fn);

}  // namespace vpglsl
