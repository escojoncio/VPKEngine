#pragma once

#include <string>
#include <vector>

#include "vpir/ir.h"

namespace vpir {

// Checks a module for structural and type errors. Returns an empty list when the module is valid.
// Rules beyond per-op typing:
//  - every value is defined once and only used after its definition, in the same block or an
//    enclosing one (structured dominance);
//  - break/continue appear only inside a loop; a function's returns match its return type;
//  - stage-interface ops (inputs, outputs, resources, shared/payload memory, mesh ops,
//    interpolation) appear only in the entry point `main`;
//  - ops are legal for the module's stage (e.g. derivatives only in fragment shaders).
std::vector<std::string> verify(const Module& module);

}  // namespace vpir
