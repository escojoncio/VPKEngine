// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <stdexcept>
#include <cstdio>

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
    std::set<uint64_t> resume_points; // return addresses of the function's calls (resumable entries)
    uint64_t end = 0;                 // highest decoded address + 1
    // Blocks reached only by falling through into another function's start (an .eh_frame start or
    // a landing pad, typically after a call that never returns such as _Unwind_Resume): emitted
    // as a dispatch to that function instead of a copy of its code.
    std::set<uint64_t> tail_blocks;
    // Indirect jumps whose targets were read from a jump table: jmp address -> targets.
    std::map<uint64_t, std::vector<uint64_t>> jump_tables;
    // Discovery only: an instruction no user-mode game code has (port I/O, hlt, cli/sti, segment
    // loads): the bytes are data (strings, tables) decoded as code; and what it referenced.
    bool implausible = false;
    std::vector<uint64_t> refs;
};

struct Stats {
    uint64_t functions = 0;
    uint64_t instructions = 0;
    uint64_t unsupported = 0;
    uint64_t indirect_calls = 0;
    uint64_t indirect_jumps = 0;
    uint64_t jump_tables = 0;
    uint64_t landing_pads = 0;
    uint64_t rejected_as_data = 0; // candidate functions dropped: data, not code (see Function::implausible)
    std::map<std::string, uint64_t> by_mnemonic;
    std::map<std::string, uint64_t> unsupported_by_mnemonic;
    // For the report: where things were not handled (the first few hundred of each).
    std::vector<std::pair<uint64_t, std::string>> unsupported_sites; // address, mnemonic + operands
    std::vector<std::pair<uint64_t, std::string>> indirect_jump_sites; // address, the jump and what preceded it
};

struct Options {
    bool reject_data = true; // drop candidate functions that decode as data (--keep-data: keep them)
    bool emit_rip_updates = false; // keep cpu->rip current before every instruction (debug)
    bool trace = false;            // call vp_trace(cpu, rip) before every instruction
    std::string symbol_prefix = "fn_";
    std::string module = "main";   // --module NAME: a C identifier; prefixes symbols and names the VpModule
    uint64_t max_functions = UINT64_MAX;
    bool lazy_flags = true;        // write only the flags a later instruction in the block reads (--no-lazy-flags)
    bool scan_data = false;        // --scan-data: aligned qwords in data that point at code are roots
    bool pic = false;              // --pic: image addresses as vp_image_base + offset (load anywhere)
    bool regcache = true;          // general registers in locals per function (--no-regcache turns it off)
    bool resume_points = true;     // every call's return address is an entry (--no-resume-points)
    bool boundaries = true;        // stop at other functions' starts when falling through (--no-boundaries)
    uint64_t split = 0;
    std::set<uint64_t> natives;    // guest addresses implemented by the host (imports): not translated            // functions per output file (0 = one file); files are <out>_NNN.c
};

// Finds every function reachable from the image's entry, code pointers and .eh_frame starts.
std::map<uint64_t, Function> discover(const Image& img, const std::vector<uint64_t>& roots, Stats& stats,
                                      const Options& opt);

// Writes the C translation unit for the discovered functions and the dispatch table.
// Closes a file vpaot wrote, and fails if anything in it was not written (a full disk): a cut C
// file must not look like a translation.
inline void close_written(FILE* f, const std::string& path) {
    const bool bad = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || bad) throw std::runtime_error("cannot write " + path + " (disk full?)");
}

void emit_c(const Image& img, const std::map<uint64_t, Function>& functions, const Options& opt,
            const std::string& out_path, Stats& stats);

} // namespace vpaot
