// A VPIR interpreter for tests: runs a module's entry point for one invocation with exact IEEE
// single-precision arithmetic (no fast-math), byte-addressed buffers and captured outputs. Covers
// the ops the GCN lifter emits except textures (answered by Invocation::image when set), interpolation, derivatives and cross-lane ops.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "vp4ir/ir.h"

namespace vpireval {

struct Value {
    vp4ir::Type type;
    uint32_t bits[4] = {};  // per component (bool 0/1, f16 in the low 16 bits)
};

struct Invocation {
    std::map<vp4ir::Builtin, Value> builtins;      // inputs by builtin
    std::map<uint32_t, Value> locations;          // inputs by location
    std::map<uint32_t, std::vector<uint8_t>> memory;  // resource index (module.resources) -> bytes
    // Optional: texture ops (sample, gather, fetch, store, size, levels) are answered by the test; `args` are the
    // instruction's evaluated value operands, `in.sym` the texture and `in.sampler` the sampler resource.
    std::function<bool(const vp4ir::Inst& in, const std::vector<Value>& args, Value& out)> image;
    // Results.
    std::map<uint32_t, Value> outputs;  // output index -> last stored value
    bool discarded = false;
};

// When set, run() prints every instruction's result (%id = component bits) to stderr.
inline bool trace = false;

// Runs `main`; false with `error` set when it meets something it cannot evaluate.
bool run(const vp4ir::Module& m, Invocation& inv, std::string& error);

}  // namespace vpireval
