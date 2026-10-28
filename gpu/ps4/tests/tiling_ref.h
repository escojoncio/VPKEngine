// A slow, literal reference for the gfx7 surface addressing, written in the address library's own
// terms (bit selectors, per-coordinate pipe / bank / rotation steps, byte counts per micro tile)
// instead of the production code's shapes. The tests compare the production layout and addressing
// against it element by element.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ref {

struct Mode {
    const char* array;  // "LINEAR", "1D", "2D", "3D", "PRT", "PRT2D", "PRT3D" + "_THIN" / "_THICK" / "_XTHICK"
    const char* micro;  // "display", "thin", "depth", "thick"
    int pipes;          // 2 or 8
    const char* pipeCfg;
    int sampleSplit, tileSplit;
    bool valid;
};

// GB_TILE_MODE0..31 by name.
inline Mode mode(int i) {
    static const Mode m[32] = {
        {"2D_THIN", "depth", 8, "16x16", 1, 64, true},   {"2D_THIN", "depth", 8, "16x16", 1, 128, true},
        {"2D_THIN", "depth", 8, "16x16", 1, 256, true},  {"2D_THIN", "depth", 8, "16x16", 1, 512, true},
        {"2D_THIN", "depth", 8, "16x16", 1, 1024, true}, {"1D_THIN", "depth", 8, "16x16", 1, 64, true},
        {"PRT2D_THIN", "depth", 8, "16x16", 1, 256, true}, {"PRT2D_THIN", "depth", 8, "16x16", 1, 1024, true},
        {"LINEAR_ALIGNED", "display", 8, "16x16", 1, 64, true}, {"1D_THIN", "display", 8, "16x16", 1, 64, true},
        {"2D_THIN", "display", 8, "16x16", 2, 64, true}, {"PRT_THIN", "display", 8, "8x16", 2, 64, true},
        {"PRT2D_THIN", "display", 8, "16x16", 2, 64, true}, {"1D_THIN", "thin", 8, "16x16", 1, 64, true},
        {"2D_THIN", "thin", 8, "16x16", 2, 64, true},    {"3D_THIN", "thin", 8, "8x16", 2, 64, true},
        {"PRT_THIN", "thin", 8, "8x16", 2, 64, true},    {"PRT2D_THIN", "thin", 8, "16x16", 2, 64, true},
        {"PRT3D_THIN", "thin", 8, "16x16", 2, 64, true}, {"1D_THICK", "thick", 8, "16x16", 1, 64, true},
        {"2D_THICK", "thick", 8, "16x16", 1, 64, true},  {"3D_THICK", "thick", 8, "8x16", 1, 64, true},
        {"PRT_THICK", "thick", 8, "8x16", 1, 64, true},  {"PRT2D_THICK", "thick", 8, "16x16", 1, 64, true},
        {"PRT3D_THICK", "thick", 8, "8x16", 1, 64, true}, {"2D_XTHICK", "thick", 8, "16x16", 1, 64, true},
        {"3D_XTHICK", "thick", 8, "8x16", 1, 64, true},  {"", "", 0, "", 0, 0, false},
        {"", "", 0, "", 0, 0, false},                    {"", "", 0, "", 0, 0, false},
        {"", "", 0, "", 0, 0, false},                    {"LINEAR_GENERAL", "display", 2, "", 1, 64, true},
    };
    return m[i];
}

inline bool has(const std::string& s, const char* t) { return s.find(t) != std::string::npos; }
inline int bit(uint64_t v, int b) { return int((v >> b) & 1); }
inline uint64_t up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
inline uint64_t pow2up(uint64_t v) {
    uint64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}
inline int log2i(uint64_t v) {
    int n = 0;
    while (v > 1) v >>= 1, ++n;
    return n;
}

struct Surface {
    int tile, bpe, samples = 1;
    uint32_t width, height, pitch = 0, slices = 1, levels = 1, block = 1;
    bool volume = false, pow2 = false;
    uint32_t bankSwizzle = 0;
};

struct Level {
    uint32_t w, h, d;         // valid
    uint64_t pitch, rows;     // allocated
    std::string array;        // as stored
    uint64_t offset, bytes;
    int thick;
    // macro geometry (macro tiled levels)
    int bankW = 1, bankH = 1, banks = 1, aspect = 1;
    uint64_t macroW = 0, macroH = 0;
};

inline int thickness(const std::string& array) {
    if (has(array, "XTHICK")) return 8;
    if (has(array, "THICK")) return 4;
    return 1;
}

// GB_MACROTILE_MODE n.
inline void macroMode(int n, int& bw, int& bh, int& banks, int& aspect) {
    static const int t[16][4] = {{1, 4, 16, 4}, {1, 2, 16, 2}, {1, 1, 16, 2}, {1, 1, 16, 2}, {1, 1, 8, 1}, {1, 1, 4, 1},
                                 {1, 1, 2, 1},  {1, 1, 2, 1},  {1, 8, 16, 4}, {1, 4, 16, 4}, {1, 2, 16, 2}, {1, 1, 16, 2},
                                 {1, 1, 8, 1},  {1, 1, 4, 1},  {1, 1, 2, 1},  {1, 1, 2, 1}};
    bw = t[n][0], bh = t[n][1], banks = t[n][2], aspect = t[n][3];
}

inline bool isMacro(const std::string& a) {
    return has(a, "2D") || has(a, "3D") || has(a, "PRT");
}

// Mip chain: sizes the way the library pads them.
inline std::vector<Level> levels(const Surface& s) {
    const Mode m = mode(s.tile);
    std::vector<Level> out;
    uint64_t off = 0;
    const int th = thickness(m.array);
    const bool macro = isMacro(m.array);
    // GB_MACROTILE_MODE index from the bytes a micro tile keeps in one bank row.
    int mtm = 0;
    if (macro) {
        const uint64_t tileBytes = 64ull * th * s.bpe;
        uint64_t split = std::string(m.micro) == "depth" ? uint64_t(m.tileSplit) : std::max<uint64_t>(256, m.sampleSplit * tileBytes);
        split = std::min<uint64_t>(split, 1024);
        const uint64_t keep = std::min<uint64_t>(split, tileBytes * s.samples);
        mtm = log2i(keep / 64) + (has(m.array, "PRT") ? 8 : 0);
    }
    for (uint32_t i = 0; i < s.levels; ++i) {
        Level l;
        uint64_t pw = std::max<uint64_t>(1, uint64_t(s.pitch ? std::max(s.pitch, s.width) : s.width) >> i);
        uint64_t ph = std::max<uint64_t>(1, uint64_t(s.height) >> i);
        uint64_t vw = std::max<uint64_t>(1, uint64_t(s.width) >> i), vh = ph;
        if (s.block > 1) {
            pw = std::max<uint64_t>(1, (pw + s.block - 1) / s.block);
            ph = std::max<uint64_t>(1, (ph + s.block - 1) / s.block);
            vw = std::max<uint64_t>(1, (vw + s.block - 1) / s.block);
            vh = std::max<uint64_t>(1, (vh + s.block - 1) / s.block);
        }
        if (s.pow2) pw = pow2up(pw), ph = pow2up(ph);
        const uint64_t vd = s.volume ? std::max<uint64_t>(1, uint64_t(s.slices) >> i) : s.slices;
        uint64_t depthForSize = s.pow2 && s.volume ? pow2up(vd) : vd;
        l.w = uint32_t(vw), l.h = uint32_t(vh), l.d = uint32_t(vd);
        l.array = m.array;
        l.thick = th;
        if (macro) {
            macroMode(mtm, l.bankW, l.bankH, l.banks, l.aspect);
            l.macroW = 8ull * m.pipes * l.bankW * l.aspect;
            l.macroH = 8ull * l.bankH * l.banks / l.aspect;
            if (i > 0 && (pw < l.macroW || ph < l.macroH)) {  // mip levels below one macro tile degrade to 1D (level 0 never)
                l.array = th > 1 ? "1D_THICK" : "1D_THIN";
                l.thick = th > 1 ? 4 : 1;
            }
        }
        const bool derive = i > 0 || s.pitch == 0;
        uint64_t sliceBytes;
        if (has(l.array, "LINEAR")) {
            const bool general = std::string(l.array) == "LINEAR_GENERAL";
            const uint64_t pa = general ? 1 : std::max<uint64_t>(8, 64 / s.bpe);
            uint64_t p = up(pw, pa);
            if (derive && !general)
                while ((p * ph) % std::max<uint64_t>(64, 256 / s.bpe)) p += pa;
            l.pitch = p, l.rows = ph;
            sliceBytes = p * ph * s.bpe;
        } else if (has(l.array, "1D")) {
            uint64_t p = up(pw, 8);
            const uint64_t r = up(ph, 8);
            if (derive)
                while ((p * r * s.bpe * s.samples * l.thick) % 256) p += 8;
            l.pitch = p, l.rows = r;
            sliceBytes = p * r * s.bpe * s.samples;
        } else {
            l.pitch = up(pw, l.macroW), l.rows = up(ph, l.macroH);
            sliceBytes = l.pitch * l.rows * s.bpe * s.samples;
        }
        l.bytes = sliceBytes * up(depthForSize, l.thick);
        l.offset = off;
        off += l.bytes;
        out.push_back(l);
    }
    return out;
}

// Element index inside the micro tile: one letter per index bit, lowest first
// (a x0 b x1 c x2 d y0 e y1 f y2 g z0 h z1 i z2).
inline std::string microPattern(const std::string& micro, int bpe, int thick) {
    if (micro == "display") {
        switch (bpe) {
            case 1: return "abcedf";  // x0 x1 x2 y1 y0 y2
            case 2: return "abcdef";
            case 4: return "abdcef";  // x0 x1 y0 x2 y1 y2
            case 8: return "adbcef";  // x0 y0 x1 x2 y1 y2
            default: return "dabcef"; // y0 x0 x1 x2 y1 y2
        }
    }
    if (micro == "thick") {
        std::string p = bpe <= 2 ? "adbegh" : bpe == 4 ? "adbgeh" : "adgbeh";  // x0 y0 x1 y1 z0 z1 / x0 y0 x1 z0 y1 z1 / x0 y0 z0 x1 y1 z1
        p += "cf";                                                              // x2 y2
        if (thick == 8) p += "i";
        return p;
    }
    return "adbecf";  // thin, depth: x0 y0 x1 y1 x2 y2
}

inline uint64_t microIndex(const std::string& pattern, uint32_t x, uint32_t y, uint32_t z) {
    uint64_t idx = 0;
    for (size_t k = 0; k < pattern.size(); ++k) {
        int v = 0;
        switch (pattern[k]) {
            case 'a': v = bit(x, 0); break;
            case 'b': v = bit(x, 1); break;
            case 'c': v = bit(x, 2); break;
            case 'd': v = bit(y, 0); break;
            case 'e': v = bit(y, 1); break;
            case 'f': v = bit(y, 2); break;
            case 'g': v = bit(z, 0); break;
            case 'h': v = bit(z, 1); break;
            case 'i': v = bit(z, 2); break;
        }
        idx |= uint64_t(v) << k;
    }
    return idx;
}

// Byte offset of element (x, y, z, sample) of level `li`.
inline uint64_t address(const Surface& s, const std::vector<Level>& lv, uint32_t li, uint32_t x, uint32_t y, uint32_t z,
                        uint32_t sample) {
    const Mode m = mode(s.tile);
    const Level& l = lv[li];
    const std::string micro = m.micro;
    const int bpe = s.bpe;
    if (has(l.array, "LINEAR")) return l.offset + (uint64_t(z) * l.rows * l.pitch + uint64_t(y) * l.pitch + x) * bpe;
    const int th = l.thick;
    const std::string pattern = microPattern(micro, bpe, th);
    const uint64_t pix = microIndex(pattern, x & 7, y & 7, z);
    const uint64_t microBytes = 64ull * th * bpe * s.samples;
    // Position inside the micro tile, in bytes: depth surfaces interleave samples, the rest keep a plane per sample.
    uint64_t inTile = micro == "depth" ? (pix * s.samples + sample) * bpe : sample * (microBytes / s.samples) + pix * bpe;
    if (has(l.array, "1D")) {
        const uint64_t tilesPerRow = l.pitch / 8;
        const uint64_t group = z / th;
        return l.offset + group * (l.pitch * l.rows * th * bpe * s.samples) + ((y / 8) * tilesPerRow + x / 8) * microBytes + inTile;
    }
    // Macro tiled.
    const bool prt = has(l.array, "PRT");
    const bool is3d = has(l.array, "3D") && !prt;
    const bool is2d = !has(l.array, "PRT") && !has(l.array, "3D");  // plain 2D family
    const bool thinSplitRot = has(l.array, "THIN") && (has(l.array, "2D") || has(l.array, "3D"));  // 2D/3D/PRT2D/PRT3D thin
    // Tile split: planes of the micro tile.
    uint64_t split = micro == "depth" ? uint64_t(m.tileSplit) : std::max<uint64_t>(256, m.sampleSplit * 64ull * th * bpe);
    split = std::min<uint64_t>(split, 1024);
    uint64_t planes = 1, plane = 0, planeBytes = microBytes;
    if (th == 1 && microBytes > split) {
        planes = microBytes / split;
        plane = inTile / split;
        inTile %= split;
        planeBytes = split;
    }
    const int pipes = m.pipes;
    const int pipeBits = log2i(pipes);
    const int bankBits = log2i(l.banks);
    const uint64_t macroTiles = (l.macroW / 8) * (l.macroH / 8);
    const uint64_t macroBytes = planeBytes * macroTiles / (uint64_t(pipes) * l.banks);
    const uint64_t perRow = l.pitch / l.macroW, rowsOfMacro = l.rows / l.macroH;
    const uint64_t mIndex = (y / l.macroH) * perRow + x / l.macroW;
    const uint64_t planeSlice = perRow * rowsOfMacro * macroBytes;  // one plane of one slice group
    const uint64_t sliceOff = planeSlice * (plane + planes * (z / th));
    const uint64_t tileOff = ((y / 8) % l.bankH) * planeBytes;
    const uint64_t local = sliceOff + mIndex * macroBytes + inTile + tileOff;
    // Pipe and bank from the tile coordinates (inside the macro tile for PRT).
    const uint32_t px = prt ? uint32_t(x % l.macroW) : x, py = prt ? uint32_t(y % l.macroH) : y;
    const uint32_t tx = px / 8, ty = py / 8;
    int pipe = 0;
    if (std::string(m.pipeCfg) == "8x16")
        pipe = (bit(tx, 1) ^ bit(ty, 0) ^ bit(tx, 2)) | ((bit(tx, 0) ^ bit(ty, 1)) << 1) | ((bit(tx, 2) ^ bit(ty, 2)) << 2);
    else
        pipe = (bit(tx, 0) ^ bit(ty, 0) ^ bit(tx, 1)) | ((bit(tx, 1) ^ bit(ty, 1)) << 1) | ((bit(tx, 2) ^ bit(ty, 2)) << 2);
    if (is3d) pipe ^= int((std::max(1, pipes / 2 - 1) * (z / th)) & (pipes - 1));
    const uint32_t bx = px / 8 / pipes, by = py / 8 / l.bankH;  // bank width 1
    int bank = 0;
    switch (l.banks) {
        case 16: bank = (bit(bx, 0) ^ bit(by, 3)) | ((bit(bx, 1) ^ bit(by, 2) ^ bit(by, 3)) << 1) | ((bit(bx, 2) ^ bit(by, 1)) << 2) | ((bit(bx, 3) ^ bit(by, 0)) << 3); break;
        case 8: bank = (bit(bx, 0) ^ bit(by, 2)) | ((bit(bx, 1) ^ bit(by, 1) ^ bit(by, 2)) << 1) | ((bit(bx, 2) ^ bit(by, 0)) << 2); break;
        case 4: bank = (bit(bx, 0) ^ bit(by, 1)) | ((bit(bx, 1) ^ bit(by, 0)) << 1); break;
        default: bank = bit(bx, 0) ^ bit(by, 0);
    }
    int rot = 0;
    if (is2d) rot = int((l.banks / 2 - 1) * (z / th));
    else if (is3d) rot = int(std::max(1, pipes / 2 - 1) * (z / th) / pipes);
    bank ^= int(s.bankSwizzle) + rot;
    if (thinSplitRot) bank ^= int((l.banks / 2 + 1) * plane);
    bank &= l.banks - 1;
    const uint64_t inter = local & 255;
    const uint64_t rest = local >> 8;
    return l.offset + (inter | (uint64_t(pipe) << 8) | (uint64_t(bank) << (8 + pipeBits)) | (rest << (8 + pipeBits + bankBits)));
}

}  // namespace ref
