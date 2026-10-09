// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "image.h"

#include <algorithm>
#include <cctype>
#include <map>
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

uint64_t read_uleb(const uint8_t* d, size_t size, size_t& off) {
    uint64_t v = 0;
    int shift = 0;
    while (off < size && shift < 64) {
        const uint8_t b = d[off++];
        v |= (uint64_t)(b & 0x7f) << shift;
        shift += 7;
        if (!(b & 0x80)) break;
    }
    return v;
}
int64_t read_sleb(const uint8_t* d, size_t size, size_t& off) {
    int64_t v = 0;
    int shift = 0;
    uint8_t b = 0;
    while (off < size && shift < 64) {
        b = d[off++];
        v |= (int64_t)(b & 0x7f) << shift;
        shift += 7;
        if (!(b & 0x80)) break;
    }
    if (shift < 64 && (b & 0x40)) v |= -((int64_t)1 << shift);
    return v;
}

// Reads a DW_EH_PE-encoded value from image memory at `at` (advances it).
uint64_t read_encoded_mem(const Image& img, uint64_t& at, uint8_t enc) {
    const uint64_t field = at;
    uint64_t v = 0;
    auto take = [&](size_t n) -> uint64_t {
        if (!img.mapped(at, n)) throw std::runtime_error("eh_frame out of image");
        uint64_t x = 0;
        std::memcpy(&x, img.at(at), n);
        at += n;
        return x;
    };
    if (enc == 0xff) return 0;
    switch (enc & 0x0f) {
    case 0x00: case 0x04: case 0x0c: v = take(8); break;
    case 0x02: v = take(2); break;
    case 0x03: v = take(4); break;
    case 0x0a: v = (uint64_t)(int64_t)(int16_t)take(2); break;
    case 0x0b: v = (uint64_t)(int64_t)(int32_t)take(4); break;
    case 0x01: { // uleb128
        size_t off = 0;
        const uint8_t* d = img.at(at);
        v = read_uleb(d, 16, off);
        at += off;
        break;
    }
    case 0x09: { // sleb128
        size_t off = 0;
        const uint8_t* d = img.at(at);
        v = (uint64_t)read_sleb(d, 16, off);
        at += off;
        break;
    }
    default: throw std::runtime_error("unsupported DWARF pointer encoding");
    }
    if ((enc & 0x70) == 0x10) v += field;
    else if ((enc & 0x70) != 0) throw std::runtime_error("unsupported DWARF pointer application");
    if (enc & 0x80) v = img.rd64(v); // indirect
    return v;
}

// Walks .eh_frame (found from the hdr's eh_frame_ptr), collects every FDE's function start and,
// through its LSDA, the landing pads of its call sites.
void parse_eh_frame(Image& img, uint64_t eh_frame) {
    struct Cie { uint8_t fde_enc = 0; uint8_t lsda_enc = 0xff; bool has_lsda = false; };
    std::map<uint64_t, Cie> cies;
    uint64_t at = eh_frame;
    for (int guard = 0; guard < 2000000; ++guard) {
        if (!img.mapped(at, 4)) break;
        uint32_t len32;
        std::memcpy(&len32, img.at(at), 4);
        if (len32 == 0) break;
        uint64_t len = len32;
        uint64_t rec = at + 4;
        if (len32 == 0xffffffff) {
            if (!img.mapped(rec, 8)) break;
            std::memcpy(&len, img.at(rec), 8);
            rec += 8;
        }
        const uint64_t next = rec + len;
        if (!img.mapped(rec, 4) || next <= rec) break;
        uint32_t cie_id;
        std::memcpy(&cie_id, img.at(rec), 4);
        try {
            auto need = [&](uint64_t a, size_t n) { if (!img.mapped(a, n)) throw std::runtime_error("eh_frame out of image"); };
            if (cie_id == 0) {
                need(rec + 4, 1);
                Cie cie;
                uint64_t p = rec + 4;
                const uint8_t version = *img.at(p++);
                std::string aug;
                while (img.mapped(p) && *img.at(p)) aug += (char)*img.at(p++);
                ++p;
                if (version >= 4) p += 2; // address_size, segment_size
                size_t off = 0;
                need(p, 48);
                read_uleb(img.at(p), 16, off); p += off; off = 0;   // code alignment
                read_sleb(img.at(p), 16, off); p += off; off = 0;   // data alignment
                if (version == 1) ++p; else { read_uleb(img.at(p), 16, off); p += off; off = 0; } // return register
                if (!aug.empty() && aug[0] == 'z') {
                    read_uleb(img.at(p), 16, off); p += off; // augmentation length
                    for (size_t i = 1; i < aug.size(); ++i) {
                        if (aug[i] == 'L') { cie.lsda_enc = *img.at(p++); cie.has_lsda = true; }
                        else if (aug[i] == 'R') { cie.fde_enc = *img.at(p++); }
                        else if (aug[i] == 'P') { const uint8_t enc = *img.at(p++); read_encoded_mem(img, p, enc); }
                        else if (aug[i] == 'S' || aug[i] == 'B') {}
                        else break;
                    }
                }
                cies[at] = cie; // keyed by the record's length field, which `rec - cie_id` resolves to
            } else {
                const uint64_t cie_at = rec - cie_id;
                auto it = cies.find(cie_at);
                if (it == cies.end()) { at = next; continue; }
                const Cie& cie = it->second;
                uint64_t p = rec + 4;
                const uint64_t start = read_encoded_mem(img, p, cie.fde_enc);
                const uint64_t range = read_encoded_mem(img, p, cie.fde_enc & 0x0f);
                if (img.is_code(start)) img.eh_frame_starts.push_back(start);
                if (cie.has_lsda) {
                    need(p, 16);
                    size_t off = 0;
                    const uint64_t aug_len = read_uleb(img.at(p), 16, off);
                    p += off;
                    const uint64_t aug_end = p + aug_len;
                    const uint64_t lsda = read_encoded_mem(img, p, cie.lsda_enc);
                    p = aug_end;
                    if (lsda && img.mapped(lsda, 32)) {
                        uint64_t l = lsda;
                        const uint8_t lpstart_enc = *img.at(l++);
                        uint64_t lpstart = start;
                        if (lpstart_enc != 0xff) lpstart = read_encoded_mem(img, l, lpstart_enc);
                        const uint8_t ttype_enc = *img.at(l++);
                        if (ttype_enc != 0xff) { size_t o = 0; read_uleb(img.at(l), 16, o); l += o; }
                        const uint8_t cs_enc = *img.at(l++);
                        size_t o = 0;
                        const uint64_t cs_len = read_uleb(img.at(l), 16, o);
                        l += o;
                        const uint64_t cs_end = l + cs_len;
                        if (cs_enc == 0xff || cs_len > (1u << 24) || !img.mapped(l, cs_len)) throw std::runtime_error("bad call-site table");
                        while (l < cs_end) {
                            need(l, 16);
                            read_encoded_mem(img, l, cs_enc);                 // call-site start
                            read_encoded_mem(img, l, cs_enc);                 // length
                            const uint64_t lp = read_encoded_mem(img, l, cs_enc); // landing pad
                            size_t o2 = 0;
                            read_uleb(img.at(l), 16, o2);                     // action
                            l += o2;
                            if (lp) {
                                const uint64_t pad = lpstart + lp;
                                if (pad >= start && pad < start + range) img.landing_pads.push_back(pad);
                            }
                        }
                    }
                }
            }
        } catch (const std::exception&) {
            // A malformed record ends the walk; what was read so far stays.
            break;
        }
        at = next;
    }
    std::sort(img.landing_pads.begin(), img.landing_pads.end());
    img.landing_pads.erase(std::unique(img.landing_pads.begin(), img.landing_pads.end()), img.landing_pads.end());
}

void parse_eh_frame_hdr(Image& img, uint64_t hdr_vaddr, uint64_t hdr_size) {
    if (!img.mapped(hdr_vaddr, hdr_size) || hdr_size < 4) return;
    std::vector<uint8_t> d(img.at(hdr_vaddr), img.at(hdr_vaddr) + hdr_size);
    size_t off = 0;
    if (d[0] != 1) return;
    const uint8_t enc_ptr = d[1], enc_count = d[2], enc_table = d[3];
    off = 4;
    const uint64_t eh_frame = read_encoded(d, off, enc_ptr, hdr_vaddr + off, hdr_vaddr);
    if (img.mapped(eh_frame, 4)) parse_eh_frame(img, eh_frame);
    const uint64_t count = read_encoded(d, off, enc_count, hdr_vaddr + off, hdr_vaddr);
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t start = read_encoded(d, off, enc_table, hdr_vaddr + off, hdr_vaddr);
        read_encoded(d, off, enc_table, hdr_vaddr + off, hdr_vaddr); // the FDE, unused here
        if (img.is_code(start)) img.eh_frame_starts.push_back(start);
    }
}

} // namespace


// ---- Windows PE32+ (x86-64) ------------------------------------------------------------------
//
// Sections at ImageBase + VirtualAddress; executable sections are code. Function starts come
// from the exception directory (.pdata: every function that is not a leaf has a RUNTIME_FUNCTION),
// code pointers from the base relocations (every absolute pointer is a DIR64 entry), the exports
// and the TLS callbacks; SEH scope tables give the __except/__finally targets as landing pads.
// Imports are "dll!name" (DLL in lower case), their slot the IAT entry.

namespace {

uint32_t rd32(const Image& img, uint64_t a) {
    uint32_t v = 0;
    if (img.mapped(a, 4)) std::memcpy(&v, img.at(a), 4);
    return v;
}
uint16_t rd16(const Image& img, uint64_t a) {
    uint16_t v = 0;
    if (img.mapped(a, 2)) std::memcpy(&v, img.at(a), 2);
    return v;
}
std::string rdstr(const Image& img, uint64_t a, size_t max = 512) {
    std::string out;
    while (out.size() < max && img.mapped(a)) {
        const char c = (char)*img.at(a++);
        if (!c) break;
        out += c;
    }
    return out;
}

} // namespace

Image load_pe(const std::vector<uint8_t>& file) {
    if (file.size() < 0x40) throw std::runtime_error("truncated PE");
    const uint32_t lfanew = get<uint32_t>(file, 0x3c);
    if (lfanew > file.size() - 24 || std::memcmp(file.data() + lfanew, "PE\0\0", 4) != 0) throw std::runtime_error("not a PE file");
    const uint16_t machine = get<uint16_t>(file, lfanew + 4);
    if (machine != 0x8664) throw std::runtime_error("not an x86-64 PE (machine " + std::to_string(machine) + ")");
    const uint16_t nsections = get<uint16_t>(file, lfanew + 6);
    const uint16_t opt_size = get<uint16_t>(file, lfanew + 20);
    const size_t opt = lfanew + 24;
    if (get<uint16_t>(file, opt) != 0x20b) throw std::runtime_error("not PE32+");
    const uint32_t entry_rva = get<uint32_t>(file, opt + 16);
    const uint64_t image_base = get<uint64_t>(file, opt + 24);
    const uint32_t size_of_image = get<uint32_t>(file, opt + 56);
    const uint32_t size_of_headers = get<uint32_t>(file, opt + 60);
    const uint32_t ndirs = get<uint32_t>(file, opt + 108);
    auto dir = [&](unsigned i) -> std::pair<uint32_t, uint32_t> {
        if (i >= ndirs || 112 + i * 8 + 8 > opt_size) return {0, 0};
        return {get<uint32_t>(file, opt + 112 + i * 8), get<uint32_t>(file, opt + 112 + i * 8 + 4)};
    };
    if (size_of_image == 0 || size_of_image > (1u << 30)) throw std::runtime_error("bad SizeOfImage");
    Image img;
    img.base = image_base;
    img.memory.assign(size_of_image, 0);
    img.loaded.push_back({image_base, image_base + size_of_image});
    std::memcpy(img.memory.data(), file.data(), std::min<size_t>({size_of_headers, file.size(), (size_t)size_of_image}));
    const size_t sec = opt + opt_size;
    for (uint16_t i = 0; i < nsections; ++i) {
        const size_t h = sec + i * 40;
        const uint32_t vsize = get<uint32_t>(file, h + 8), va = get<uint32_t>(file, h + 12);
        const uint32_t raw_size = get<uint32_t>(file, h + 16), raw_ptr = get<uint32_t>(file, h + 20);
        const uint32_t ch = get<uint32_t>(file, h + 36);
        const uint32_t span = std::max(vsize, raw_size);
        if (va > size_of_image || span > size_of_image - va) throw std::runtime_error("section outside the image");
        const uint32_t n = std::min(raw_size, vsize ? vsize : raw_size);
        if (n && raw_ptr <= file.size() && n <= file.size() - raw_ptr) std::memcpy(img.memory.data() + va, file.data() + raw_ptr, n);
        if (ch & 0x20000000) img.executable.push_back({image_base + va, image_base + va + (vsize ? vsize : raw_size)});
    }
    img.entry = entry_rva ? image_base + entry_rva : 0;

    // Base relocations: DIR64 (type 10) slots hold absolute pointers.
    if (auto [rva, size] = dir(5); rva && size) {
        uint64_t p = image_base + rva;
        const uint64_t end = p + size;
        while (p + 8 <= end) {
            const uint32_t page = rd32(img, p), block = rd32(img, p + 4);
            if (block < 8 || p + block > end) break;
            for (uint64_t e = p + 8; e + 2 <= p + block; e += 2) {
                const uint16_t entry = rd16(img, e);
                if ((entry >> 12) != 10) continue;
                const uint64_t slot = image_base + page + (entry & 0xfff);
                img.reloc_sites.push_back(slot);
                const uint64_t value = img.rd64(slot);
                if (img.is_code(value)) img.code_pointers.push_back(value);
            }
            p += block;
        }
    }
    // Exception directory: RUNTIME_FUNCTION { begin, end, unwind } per function.
    if (auto [rva, size] = dir(3); rva && size) {
        for (uint64_t p = image_base + rva; p + 12 <= image_base + rva + size; p += 12) {
            const uint32_t begin = rd32(img, p), unwind = rd32(img, p + 8);
            if (!begin) continue;
            if (img.is_code(image_base + begin)) img.eh_frame_starts.push_back(image_base + begin);
            // UNWIND_INFO: version/flags, prologue size, code count, frame; codes; then the handler.
            const uint64_t u = image_base + (unwind & ~1u);
            if (!img.mapped(u, 4)) continue;
            const uint8_t flags = *img.at(u) >> 3;
            const uint8_t count = *img.at(u + 2);
            const uint64_t after = u + 4 + ((count + 1u) & ~1u) * 2;
            if (flags & 4) continue; // chained: no handler of its own
            if (flags & 3) {         // EHANDLER / UHANDLER
                const uint32_t handler = rd32(img, after);
                if (img.is_code(image_base + handler)) img.code_pointers.push_back(image_base + handler);
                // __C_specific_handler scope table: count, then { begin, end, handler, target }.
                // A plausible table (begin/end inside the function, target inside it too) gives
                // the __except / __finally targets as landing pads and the filters as functions.
                const uint32_t n = rd32(img, after + 4);
                const uint32_t fend = rd32(img, p + 4);
                if (n && n < 256) {
                    bool plausible = true;
                    for (uint32_t k = 0; k < n && plausible; ++k) {
                        const uint32_t sb = rd32(img, after + 8 + k * 16), se = rd32(img, after + 12 + k * 16);
                        plausible = sb >= begin && se <= fend && sb < se;
                    }
                    for (uint32_t k = 0; k < n && plausible; ++k) {
                        const uint32_t h = rd32(img, after + 16 + k * 16), t = rd32(img, after + 20 + k * 16);
                        if (h > 1 && img.is_code(image_base + h)) img.code_pointers.push_back(image_base + h);
                        if (t && t >= begin && t < fend) img.landing_pads.push_back(image_base + t);
                    }
                }
            }
        }
    }
    // Imports: IMAGE_IMPORT_DESCRIPTOR { ILT, time, forwarder, name, IAT }.
    if (auto [rva, size] = dir(1); rva && size) {
        for (uint64_t d = image_base + rva; img.mapped(d, 20); d += 20) {
            const uint32_t ilt = rd32(img, d), name = rd32(img, d + 12), iat = rd32(img, d + 16);
            if (!ilt && !name && !iat) break;
            std::string dll = rdstr(img, image_base + name);
            for (auto& c : dll) c = (char)std::tolower((unsigned char)c);
            const uint64_t table = image_base + (ilt ? ilt : iat);
            for (uint64_t k = 0; k < 65536; ++k) {
                const uint64_t entry = img.rd64(table + k * 8);
                if (!entry) break;
                std::string fn = (entry >> 63) ? "#" + std::to_string(entry & 0xffff) : rdstr(img, image_base + (uint32_t)entry + 2);
                img.imports.push_back({dll + "!" + fn, image_base + iat + k * 8, true});
            }
        }
    }
    // Exports: their addresses are entry points (a DLL's functions).
    if (auto [rva, size] = dir(0); rva && size) {
        const uint64_t e = image_base + rva;
        const uint32_t nfuncs = rd32(img, e + 20), funcs = rd32(img, e + 28);
        for (uint32_t k = 0; k < nfuncs && k < 65536; ++k) {
            const uint32_t f = rd32(img, image_base + funcs + k * 4);
            if (f && img.is_code(image_base + f)) img.code_pointers.push_back(image_base + f);
        }
    }
    // TLS callbacks: a null-terminated array of absolute addresses.
    if (auto [rva, size] = dir(9); rva && size) {
        const uint64_t callbacks = img.rd64(image_base + rva + 24);
        for (uint64_t k = 0; k < 256 && img.mapped(callbacks + k * 8, 8); ++k) {
            const uint64_t cb = img.rd64(callbacks + k * 8);
            if (!cb) break;
            if (img.is_code(cb)) img.code_pointers.push_back(cb);
        }
    }
    auto uniq = [](std::vector<uint64_t>& v) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); };
    uniq(img.code_pointers);
    uniq(img.eh_frame_starts);
    uniq(img.landing_pads);
    uniq(img.reloc_sites);
    return img;
}

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
    if (elf.size() >= 2 && elf[0] == 'M' && elf[1] == 'Z') return load_pe(elf);
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
        // Symbols, for the imports: PS4 keeps them in the dynlib blob (tags 0x61000039 / 0x6100003f
        // for the table, 0x61000035 / 0x61000037 for the strings); ELF in DT_SYMTAB / DT_STRTAB.
        const uint8_t* symtab = nullptr;
        uint64_t symtab_size = 0;
        const uint8_t* strtab = nullptr;
        uint64_t strtab_size = 0;
        if (dynlib) {
            const uint64_t so = find(0x61000039), ss = find(0x6100003f), to = find(0x61000035), ts = find(0x61000037);
            if (so + ss <= dynlib->filesz && to + ts <= dynlib->filesz) {
                symtab = elf.data() + dynlib->offset + so; symtab_size = ss;
                strtab = elf.data() + dynlib->offset + to; strtab_size = ts;
            }
        } else {
            const uint64_t so = find(6), to = find(5), ts = find(10); // DT_SYMTAB, DT_STRTAB, DT_STRSZ
            if (so && to && img.mapped(so) && img.mapped(to, ts)) {
                symtab = img.at(so); symtab_size = (to > so) ? to - so : 0; // the string table usually follows
                strtab = img.at(to); strtab_size = ts;
            }
        }
        uint64_t sym_value = 0;
        auto symbol = [&](uint32_t index, bool& defined, bool& is_function) -> std::string {
            defined = false; is_function = true; sym_value = 0;
            if (!symtab || (uint64_t)index * 24 + 24 > symtab_size) return "";
            uint32_t name; uint8_t info; uint16_t shndx;
            std::memcpy(&name, symtab + index * 24, 4);
            std::memcpy(&info, symtab + index * 24 + 4, 1);
            std::memcpy(&shndx, symtab + index * 24 + 6, 2);
            std::memcpy(&sym_value, symtab + index * 24 + 8, 8);
            defined = shndx != 0;
            is_function = (info & 15) != 1; // STT_OBJECT = 1
            if (name >= strtab_size) return "";
            const char* p = (const char*)strtab + name;
            const uint64_t max = strtab_size - name;
            return std::string(p, strnlen(p, max));
        };
        // Exports: every defined function symbol in executable memory is reachable from outside
        // (another module, the runtime), so it is a root even when nothing in this image calls it.
        // PS4 modules export their functions this way (by NID-encoded name).
        for (uint64_t index = 1; symtab && index * 24 + 24 <= symtab_size; ++index) {
            uint8_t info; uint16_t shndx; uint64_t value;
            std::memcpy(&info, symtab + index * 24 + 4, 1);
            std::memcpy(&shndx, symtab + index * 24 + 6, 2);
            std::memcpy(&value, symtab + index * 24 + 8, 8);
            const unsigned type = info & 15;
            if (shndx != 0 && (type == 2 /* STT_FUNC */ || type == 0 /* NOTYPE */) && img.is_code(value)) img.code_pointers.push_back(value);
        }
        auto scan = [&](const uint8_t* table, uint64_t size) {
            for (uint64_t pos = 0; pos + 24 <= size; pos += 24) {
                uint64_t target, info;
                int64_t addend;
                std::memcpy(&target, table + pos, 8);
                std::memcpy(&info, table + pos + 8, 8);
                std::memcpy(&addend, table + pos + 16, 8);
                const uint32_t kind = (uint32_t)info;
                const uint32_t sym = (uint32_t)(info >> 32);
                // R_X86_64_RELATIVE (8): base + addend is the value. The image is linked at
                // its own addresses, so the value is the addend.
                if (kind == 8 && img.is_code((uint64_t)addend)) img.code_pointers.push_back((uint64_t)addend);
                if (kind == 8 || kind == 1) img.reloc_sites.push_back(target);
                // R_X86_64_64 (1), GLOB_DAT (6), JUMP_SLOT (7) against an undefined symbol: an import.
                if ((kind == 1 || kind == 6 || kind == 7) && sym) {
                    bool defined, is_function;
                    const std::string name = symbol(sym, defined, is_function);
                    if (!defined && !name.empty()) img.imports.push_back({name, target, is_function});
                    // A defined symbol's slot (its own PLT entry, a vtable pointer): code to translate.
                    if (defined && img.is_code(sym_value + (kind == 1 ? (uint64_t)addend : 0))) img.code_pointers.push_back(sym_value + (kind == 1 ? (uint64_t)addend : 0));
                }
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
            const uint64_t jmprel = find(23), pltrelsz = find(2); // DT_JMPREL, DT_PLTRELSZ
            if (jmprel && img.mapped(jmprel, pltrelsz)) scan(img.at(jmprel), pltrelsz);
        }
    }
    std::sort(img.reloc_sites.begin(), img.reloc_sites.end());
    std::sort(img.code_pointers.begin(), img.code_pointers.end());
    img.code_pointers.erase(std::unique(img.code_pointers.begin(), img.code_pointers.end()), img.code_pointers.end());
    std::sort(img.eh_frame_starts.begin(), img.eh_frame_starts.end());
    img.eh_frame_starts.erase(std::unique(img.eh_frame_starts.begin(), img.eh_frame_starts.end()), img.eh_frame_starts.end());
    return img;
}

} // namespace vpaot
