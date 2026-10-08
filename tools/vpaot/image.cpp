// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "image.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vpaot {
namespace {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

template <class T>
T get(const std::vector<uint8_t>& d, size_t off) {
    if (off + sizeof(T) > d.size()) throw std::runtime_error("truncated file");
    T v;
    std::memcpy(&v, d.data() + off, sizeof(T));
    return v;
}

struct Phdr {
    uint32_t type, flags;
    uint64_t offset, vaddr, paddr, filesz, memsz, align;
};

constexpr uint32_t PT_LOAD = 1, PT_DYNAMIC = 2, PT_GNU_EH_FRAME = 0x6474e550;
constexpr uint32_t PT_SCE_DYNLIBDATA = 0x61000000, PT_SCE_RELRO = 0x61000010;

// A PS4 SELF is a table of segments followed by the ELF; the ELF's segment data is at the SELF
// segment offsets, not at the program header offsets. Returns a plain ELF file image.
std::vector<uint8_t> unwrap_self(const std::vector<uint8_t>& self) {
    const uint16_t count = get<uint16_t>(self, 24);
    const size_t base = 32 + size_t(count) * 32;
    if (base + 64 > self.size() || std::memcmp(self.data() + base, "\x7f" "ELF", 4) != 0) {
        throw std::runtime_error("SELF without an ELF header");
    }
    const uint64_t phoff = get<uint64_t>(self, base + 32);
    const uint16_t phnum = get<uint16_t>(self, base + 56);
    std::vector<Phdr> ph(phnum);
    for (uint16_t i = 0; i < phnum; ++i) {
        std::memcpy(&ph[i], self.data() + base + phoff + i * 56, 56);
    }
    size_t end = 0;
    for (const auto& p : ph) end = std::max<size_t>(end, p.offset + p.filesz);
    std::vector<uint8_t> elf(end);
    const size_t header_end = phoff + size_t(phnum) * 56;
    std::memcpy(elf.data(), self.data() + base, header_end);
    for (uint16_t i = 0; i < count; ++i) {
        const uint64_t flags = get<uint64_t>(self, 32 + i * 32);
        const uint64_t off = get<uint64_t>(self, 40 + i * 32);
        const uint64_t size = get<uint64_t>(self, 48 + i * 32);
        if (!(flags & 0x800)) continue; // not a blocked (data-carrying) segment
        if (flags & 10) throw std::runtime_error("encrypted or compressed SELF segment");
        const size_t index = (flags >> 20) & 4095;
        if (index >= ph.size()) throw std::runtime_error("bad SELF segment index");
        const auto& p = ph[index];
        if (size != p.filesz || off + size > self.size()) throw std::runtime_error("bad SELF segment");
        std::memcpy(elf.data() + p.offset, self.data() + off, size);
    }
    return elf;
}

// DWARF pointer-encoded value from .eh_frame_hdr (DW_EH_PE_*).
uint64_t read_encoded(const std::vector<uint8_t>& d, size_t& off, uint8_t enc, uint64_t pc_of_field,
                      uint64_t data_base) {
    if (enc == 0xff) return 0;
    uint64_t v = 0;
    switch (enc & 0x0f) {
    case 0x00: v = get<uint64_t>(d, off); off += 8; break;
    case 0x02: v = get<uint16_t>(d, off); off += 2; break;
    case 0x03: v = get<uint32_t>(d, off); off += 4; break;
    case 0x04: v = get<uint64_t>(d, off); off += 8; break;
    case 0x0a: v = (uint64_t)(int64_t)get<int16_t>(d, off); off += 2; break;
    case 0x0b: v = (uint64_t)(int64_t)get<int32_t>(d, off); off += 4; break;
    case 0x0c: v = get<uint64_t>(d, off); off += 8; break;
    default: throw std::runtime_error("unsupported DWARF pointer encoding");
    }
    switch (enc & 0x70) {
    case 0x00: break;
    case 0x10: v += pc_of_field; break;
    case 0x30: v += data_base; break;
    default: throw std::runtime_error("unsupported DWARF pointer application");
    }
    return v;
}

void parse_eh_frame_hdr(Image& img, uint64_t hdr_vaddr, uint64_t hdr_size) {
    if (!img.mapped(hdr_vaddr, hdr_size) || hdr_size < 4) return;
    std::vector<uint8_t> d(img.at(hdr_vaddr), img.at(hdr_vaddr) + hdr_size);
    size_t off = 0;
    if (d[0] != 1) return;
    const uint8_t enc_ptr = d[1], enc_count = d[2], enc_table = d[3];
    off = 4;
    read_encoded(d, off, enc_ptr, hdr_vaddr + off, hdr_vaddr); // eh_frame_ptr, unused
    const uint64_t count = read_encoded(d, off, enc_count, hdr_vaddr + off, hdr_vaddr);
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t start = read_encoded(d, off, enc_table, hdr_vaddr + off, hdr_vaddr);
        read_encoded(d, off, enc_table, hdr_vaddr + off, hdr_vaddr); // the FDE, unused here
        if (img.is_code(start)) img.eh_frame_starts.push_back(start);
    }
}

} // namespace

Image load_raw(const std::string& path, uint64_t base) {
    Image img;
    img.memory = read_file(path);
    img.base = base;
    img.loaded.push_back({base, base + img.memory.size()});
    img.executable = img.loaded;
    img.entry = base;
    return img;
}

Image load_elf_or_self(const std::string& path) {
    std::vector<uint8_t> elf = read_file(path);
    if (elf.size() >= 4 && std::memcmp(elf.data(), "O\x15=\x1d", 4) == 0) {
        elf = unwrap_self(elf);
    }
    if (elf.size() < 64 || std::memcmp(elf.data(), "\x7f" "ELF\x02\x01", 6) != 0 ||
        get<uint16_t>(elf, 18) != 62) {
        throw std::runtime_error("not a little-endian x86-64 ELF64");
    }
    const uint64_t entry = get<uint64_t>(elf, 24);
    const uint64_t phoff = get<uint64_t>(elf, 32);
    const uint16_t phnum = get<uint16_t>(elf, 56);
    std::vector<Phdr> ph(phnum);
    for (uint16_t i = 0; i < phnum; ++i) {
        if (phoff + (i + 1) * 56 > elf.size()) throw std::runtime_error("truncated program headers");
        std::memcpy(&ph[i], elf.data() + phoff + i * 56, 56);
    }
    uint64_t lo = UINT64_MAX, hi = 0;
    for (const auto& p : ph) {
        if (p.type != PT_LOAD && p.type != PT_SCE_RELRO) continue;
        lo = std::min(lo, p.vaddr);
        hi = std::max(hi, p.vaddr + p.memsz);
    }
    if (lo == UINT64_MAX || hi - lo > (uint64_t(1) << 30)) throw std::runtime_error("no loadable segments or image too large");
    Image img;
    img.base = lo;
    img.memory.assign(hi - lo, 0);
    img.entry = entry;
    for (const auto& p : ph) {
        if (p.type != PT_LOAD && p.type != PT_SCE_RELRO) continue;
        if (p.filesz > p.memsz || p.offset + p.filesz > elf.size()) throw std::runtime_error("bad segment");
        std::memcpy(img.memory.data() + (p.vaddr - lo), elf.data() + p.offset, p.filesz);
        img.loaded.push_back({p.vaddr, p.vaddr + p.memsz});
        if (p.flags & 1) img.executable.push_back({p.vaddr, p.vaddr + p.memsz});
    }
    for (const auto& p : ph) {
        if (p.type == PT_GNU_EH_FRAME) parse_eh_frame_hdr(img, p.vaddr, p.memsz);
    }
    // Relocations: standard ELF (DT_RELA) or PS4 (the dynlib data blob with its own tags).
    const Phdr* dyn = nullptr;
    const Phdr* dynlib = nullptr;
    for (const auto& p : ph) {
        if (p.type == PT_DYNAMIC) dyn = &p;
        if (p.type == PT_SCE_DYNLIBDATA) dynlib = &p;
    }
    if (dyn) {
        std::vector<std::pair<uint64_t, uint64_t>> tags;
        for (uint64_t pos = dyn->offset; pos + 16 <= dyn->offset + dyn->filesz; pos += 16) {
            const uint64_t tag = get<uint64_t>(elf, pos), val = get<uint64_t>(elf, pos + 8);
            if (tag == 0) break;
            tags.emplace_back(tag, val);
        }
        auto find = [&](uint64_t t) -> uint64_t {
            for (auto& [k, v] : tags) if (k == t) return v;
            return 0;
        };
        auto scan = [&](const uint8_t* table, uint64_t size) {
            for (uint64_t pos = 0; pos + 24 <= size; pos += 24) {
                uint64_t target, info;
                int64_t addend;
                std::memcpy(&target, table + pos, 8);
                std::memcpy(&info, table + pos + 8, 8);
                std::memcpy(&addend, table + pos + 16, 8);
                const uint32_t kind = (uint32_t)info;
                // R_X86_64_RELATIVE (8): base + addend is the value. The image is linked at
                // its own addresses, so the value is the addend.
                if (kind == 8 && img.is_code((uint64_t)addend)) img.code_pointers.push_back((uint64_t)addend);
                (void)target;
            }
        };
        if (dynlib) {
            // PS4: offsets are relative to the dynlib data blob.
            const uint64_t rela = find(0x61000029), relasz = find(0x6100002d);
            const uint64_t jmprel = find(0x6100002f), pltrelsz = find(0x61000031);
            if (rela + relasz <= dynlib->filesz) scan(elf.data() + dynlib->offset + rela, relasz);
            if (jmprel + pltrelsz <= dynlib->filesz) scan(elf.data() + dynlib->offset + jmprel, pltrelsz);
        } else {
            const uint64_t rela = find(7), relasz = find(8); // DT_RELA, DT_RELASZ (virtual address)
            if (rela && img.mapped(rela, relasz)) scan(img.at(rela), relasz);
        }
    }
    std::sort(img.code_pointers.begin(), img.code_pointers.end());
    img.code_pointers.erase(std::unique(img.code_pointers.begin(), img.code_pointers.end()), img.code_pointers.end());
    std::sort(img.eh_frame_starts.begin(), img.eh_frame_starts.end());
    img.eh_frame_starts.erase(std::unique(img.eh_frame_starts.begin(), img.eh_frame_starts.end()), img.eh_frame_starts.end());
    return img;
}

} // namespace vpaot
