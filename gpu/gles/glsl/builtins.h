// Internal: built-in function signatures (resolution by argument types) of GLSL ES 1.00-3.20.
#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "ast.h"

namespace vpglsl {

struct BuiltinCtx {
    Stage stage;
    int version;
    const std::set<std::string>* extensions;
    bool ext(const char* e) const { return extensions && extensions->count(e); }
};

namespace Builtins {

// Return type of `name(args)`, or nullopt (with `why` set when the name is a builtin but the
// arguments do not match, empty when the name is not a builtin at all).
std::optional<Type> resolve(const std::string& name, const std::vector<Type>& args, const BuiltinCtx& ctx,
                            std::string& why);

bool isBuiltinName(const std::string& name);

// Indices of the arguments written by the builtin (modf's i, frexp's exp, uaddCarry's carry,
// umulExtended's msb and lsb, the memory operand of atomic*); empty if none.
std::vector<int> outArguments(const std::string& name);

// Constant folding of scalar builtins with constant arguments (abs/min/max/clamp...).
void fold(Expr& e, const std::vector<ExprP>& args);

// Texture function decomposition shared with the emitter.
struct TexCall {
    bool proj = false, lod = false, grad = false, offset = false, fetch = false, gather = false, size = false,
         bias = false, legacy = false, levels = false;
};
bool textureCall(const std::string& name, TexCall& out);

// Coordinate components for `texture()` on a sampler type (shadow: + the reference).
int coordSize(const Type& sampler);

// atomicAdd/Min/Max/And/Or/Xor/Exchange/CompSwap on a buffer or shared variable.
bool isAtomicMemoryFunction(const std::string& name);

// imageLoad / imageStore / imageSize / imageAtomic* (the emitter decodes the operation from the name).
struct ImageCall {
    enum Kind { Load, Store, Size, Atomic } kind = Load;
};
bool imageCall(const std::string& name, ImageCall& out);

}  // namespace Builtins
}  // namespace vpglsl
