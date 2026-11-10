// A VPIR interpreter for tests: runs a module's entry point for one invocation with exact IEEE
// single-precision arithmetic (no fast-math), byte-addressed buffers and captured outputs. Covers
// the ops the GCN lifter emits except textures (answered by Invocation::image when set), interpolation, derivatives and cross-lane ops.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "vpir/ir.h"

namespace vpireval {

struct Value {
    vpir::Type type;
    uint32_t bits[4] = {};  // per component (bool 0/1, f16 in the low 16 bits)
};

// A storage image the evaluator holds for `Invocation::images` (keyed by the module's resource index): texels of four
// 32-bit components, level after level (data in memory[kImageMemory + resource]).
struct ImageInfo {
    vpir::TexDim dim = vpir::TexDim::D2;
    uint32_t width = 1, height = 1, depth = 1;  // depth: the 3D depth or the array layers
    uint32_t levels = 1;
    // Texel index of (x, y, z) at `level`, or -1 outside.
    int64_t texel(uint32_t level, int64_t x, int64_t y, int64_t z) const;
    uint32_t levelWidth(uint32_t level) const { return width >> level ? width >> level : 1; }
    uint32_t levelHeight(uint32_t level) const { return height >> level ? height >> level : 1; }
    uint32_t levelDepth(uint32_t level) const {
        return dim == vpir::TexDim::D3 ? (depth >> level ? depth >> level : 1) : depth;
    }
    uint64_t totalTexels() const;
};
constexpr uint32_t kImageMemory = 0x20000;

struct WaveState;  // shared state of the invocations run together by runWorkgroup

struct Invocation {
    std::map<vpir::Builtin, Value> builtins;      // inputs by builtin
    std::map<uint32_t, Value> locations;          // inputs by location
    std::map<uint32_t, std::vector<uint8_t>> memory;  // resource index (module.resources) -> bytes
    std::map<uint32_t, ImageInfo> images;  // storage images: fetch / store / atomics / size / levels are evaluated on them
    // Optional: texture ops (sample, gather, fetch, store, size, levels) are answered by the test; `args` are the
    // instruction's evaluated value operands, `in.sym` the texture and `in.sampler` the sampler resource.
    std::function<bool(const vpir::Inst& in, const std::vector<Value>& args, Value& out)> image;
    // Results.
    std::map<uint32_t, Value> outputs;  // output index -> last stored value
    bool discarded = false;
    // Loads of a storage buffer that reach past the bytes held for it (they read 0 there): a test that must see which
    // addresses a lifted load puts on the bus even when it throws the value away counts them.
    uint64_t outOfBufferReads = 0;
    // Set by runWorkgroup: the invocations that run together, and this one's index among them.
    WaveState* wave = nullptr;
    unsigned lane = 0;
};

// When set, run() prints every instruction's result (%id = component bits) to stderr.
inline bool trace = false;

// Runs `main`; false with `error` set when it meets something it cannot evaluate.
bool run(const vpir::Module& m, Invocation& inv, std::string& error);

// Runs one invocation per element of `invs` as a workgroup, each on its own thread: barriers and subgroup ops
// (groups of module.waveSize consecutive invocations) are rendezvous points, `shared` memory and the buffers of
// invs[0].memory are common to all (the final contents are left in invs[0].memory). Every invocation must reach
// each barrier / subgroup op. False with `error` set when any invocation fails.
bool runWorkgroup(const vpir::Module& m, std::vector<Invocation>& invs, std::string& error);

}  // namespace vpireval
