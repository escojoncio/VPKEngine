// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include "image.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace vpaot {

struct Function {
    uint64_t entry = 0;
    std::set<uint64_t> blocks;        // block start addresses (entry included)
    std::set<uint64_t> extra_entries; // addresses the host may enter at besides `entry`
    uint64_t end = 0;                 // highest decoded address + 1
};

struct Stats {
    uint64_t functions = 0;
    uint64_t instructions = 0;
    uint64_t unsupported = 0;
    uint64_t indirect_calls = 0;
    uint64_t indirect_jumps = 0;
    std::map<std::string, uint64_t> by_mnemonic;
    std::map<std::string, uint64_t> unsupported_by_mnemonic;
};

struct Options {
    bool emit_rip_updates = false; // keep cpu->rip current before every instruction (debug)
    bool trace = false;            // call vp_trace(cpu, rip) before every instruction
    std::string symbol_prefix = "fn_";
    uint64_t max_functions = UINT64_MAX;
};

// Finds every function reachable from the image's entry, code pointers and .eh_frame starts.
std::map<uint64_t, Function> discover(const Image& img, const std::vector<uint64_t>& roots, Stats& stats,
                                      const Options& opt);

// Writes the C translation unit for the discovered functions and the dispatch table.
void emit_c(const Image& img, const std::map<uint64_t, Function>& functions, const Options& opt,
            const std::string& out_path, Stats& stats);

} // namespace vpaot
