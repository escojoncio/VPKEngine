// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
//
// The program image the translator works on: a flat copy of the guest's memory with the list of
// executable ranges, the entry point, and every address that is known to hold code (relocated
// code pointers, .eh_frame function starts). Loaded from a raw blob, an ELF, or a PS4 SELF.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vpaot {

struct Range {
    uint64_t start = 0;
    uint64_t end = 0; // exclusive
    bool contains(uint64_t a, uint64_t size = 1) const { return a >= start && a < end && size <= end - a; }
};

struct Image {
    std::vector<uint8_t> memory; // guest address = base + offset
    uint64_t base = 0;
    std::vector<Range> loaded;     // every mapped range
    std::vector<Range> executable; // the ranges the translator may decode
    uint64_t entry = 0;
    std::vector<uint64_t> code_pointers;   // relocations whose value points into executable memory
    std::vector<uint64_t> eh_frame_starts; // FDE initial locations
    std::vector<uint64_t> landing_pads;    // LSDA landing pads: extra function entries
    struct Import {
        std::string name;   // symbol name (PS4: NID#library#module encoded name as in the file)
        uint64_t slot = 0;  // address of the GOT/PLT slot the runtime fills (relocation target)
        bool function = true;
    };
    std::vector<Import> imports;

    bool mapped(uint64_t a, uint64_t size = 1) const {
        for (const auto& r : loaded) {
            if (r.contains(a, size)) return true;
        }
        return false;
    }
    bool is_code(uint64_t a, uint64_t size = 1) const {
        for (const auto& r : executable) {
            if (r.contains(a, size)) return true;
        }
        return false;
    }
    const uint8_t* at(uint64_t a) const { return memory.data() + (a - base); }
    uint64_t end() const { return base + memory.size(); }
    uint64_t rd64(uint64_t a) const {
        uint64_t v = 0;
        if (mapped(a, 8)) std::memcpy(&v, at(a), 8);
        return v;
    }
};

// Raw blob at `base`; the whole blob is executable.
Image load_raw(const std::string& path, uint64_t base);
// ELF64 x86-64 (executable or shared object) or a PS4 SELF wrapping one. Throws std::runtime_error.
Image load_elf_or_self(const std::string& path);

} // namespace vpaot
