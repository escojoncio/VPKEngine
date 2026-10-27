// Reference MIMG instructions on storage images (loads, stores, atomics, resinfo) for the differential tests, written
// from the AMD "Sea Islands Series Instruction Set Architecture" MIMG description; independent of the lifter.
#include <bit>
#include <cstring>
#include <string>

#include "gcn_ref.h"

namespace vpgcnref {

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

struct Img {
    const vpgcn::Instruction& in;
    Wave& w;
    ImageData& img;
    std::string& err;

    bool fail(const std::string& why) {
        err = why;
        return false;
    }
    static float asF(u32 v) { return std::bit_cast<float>(v); }

    // Number of address dwords of the coordinates: 1D x, 2D x y, 3D x y z, arrays/cubes x y layer.
    unsigned coordCount() const {
        switch (img.dim) {
            case ImageData::D1: return 1;
            case ImageData::D2: return 2;
            default: return 3;
        }
    }
    u32 levelDim(unsigned level, u32 base) const { return std::max<u32>(1, base >> level); }
    u32 lw(unsigned l) const { return levelDim(l, img.width); }
    u32 lh(unsigned l) const { return levelDim(l, img.height); }
    u32 ld(unsigned l) const { return img.dim == ImageData::D3 ? levelDim(l, img.depth) : img.depth; }
    // Index of the first dword of the texel, or -1 when outside.
    int64_t at(unsigned level, int64_t x, int64_t y, int64_t z) const {
        if (level >= img.levels) return -1;
        if (x < 0 || y < 0 || z < 0 || x >= lw(level) || y >= lh(level) || z >= ld(level)) return -1;
        int64_t base = 0;
        for (unsigned l = 0; l < level; ++l) base += int64_t(lw(l)) * lh(l) * ld(l);
        return 4 * (base + (z * lh(level) + y) * lw(level) + x);
    }

    bool run() {
        std::string name(vpgcn::mnemonic(in.op));
        const unsigned dmask = in.mem.dmask;
        Wave next = w;
        const unsigned data = in.src[0].index;
        const unsigned addr = in.src[1].index;
        const bool mip = name.find("_mip") != std::string::npos;
        const bool atomic = name.starts_with("image_atomic");
        const bool store = name.starts_with("image_store");
        const unsigned cc = coordCount();
        for (unsigned lane = 0; lane < 64; ++lane) {
            if (!(w.exec >> lane & 1)) continue;
            auto a = [&](unsigned k) { return w.v[addr + k][lane]; };
            if (name == "image_get_resinfo") {
                unsigned level = a(0);
                u32 vals[4] = {level < img.levels ? lw(level) : 0, level < img.levels ? lh(level) : 0,
                               level < img.levels ? ld(level) : 0, img.levels};
                if (img.dim == ImageData::D1) vals[1] = vals[2] = level < img.levels ? 1 : 0;
                if (img.dim == ImageData::D2) vals[2] = level < img.levels ? 1 : 0;
                unsigned k = 0;
                for (unsigned c = 0; c < 4; ++c)
                    if (dmask >> c & 1) next.v[data + k++][lane] = vals[c];
                continue;
            }
            int64_t x = int32_t(a(0)), y = cc > 1 ? int32_t(a(1)) : 0, z = cc > 2 ? int32_t(a(2)) : 0;
            unsigned level = mip ? a(cc) : 0;
            int64_t t = at(level, x, y, z);
            if (atomic) {
                std::string op = name.substr(std::strlen("image_atomic_"));
                u32 old = 0;
                if (t >= 0) {
                    u32 s = w.v[data][lane], c = w.v[data + 1][lane], v = img.texels[t], n = v;
                    old = v;
                    if (op == "swap") n = s;
                    else if (op == "cmpswap") n = v == c ? s : v;
                    else if (op == "add") n = v + s;
                    else if (op == "sub") n = v - s;
                    else if (op == "rsub") n = s - v;
                    else if (op == "smin") n = int32_t(s) < int32_t(v) ? s : v;
                    else if (op == "umin") n = s < v ? s : v;
                    else if (op == "smax") n = int32_t(s) > int32_t(v) ? s : v;
                    else if (op == "umax") n = s > v ? s : v;
                    else if (op == "and") n = v & s;
                    else if (op == "or") n = v | s;
                    else if (op == "xor") n = v ^ s;
                    else if (op == "inc") n = v >= s ? 0 : v + 1;
                    else if (op == "dec") n = (v == 0 || v > s) ? s : v - 1;
                    else if (op == "fcmpswap") n = asF(v) == asF(c) ? s : v;
                    else if (op == "fmin") n = asF(s) < asF(v) ? s : v;
                    else if (op == "fmax") n = asF(s) > asF(v) ? s : v;
                    else return fail("unsupported " + name);
                    img.texels[t] = n;
                }
                if (in.mem.glc) next.v[data][lane] = old;
                continue;
            }
            if (store) {
                if (t < 0) continue;
                unsigned k = 0;
                for (unsigned c = 0; c < 4; ++c)
                    if (dmask >> c & 1) img.texels[t + c] = w.v[data + k++][lane];
                continue;
            }
            // loads
            unsigned k = 0;
            for (unsigned c = 0; c < 4; ++c)
                if (dmask >> c & 1) next.v[data + k++][lane] = t < 0 ? 0 : img.texels[t + c];
            if (in.mem.tfe || in.mem.lwe) next.v[data + k][lane] = 0;
        }
        w = next;
        return true;
    }
};

}  // namespace

bool stepImage(const vpgcn::Instruction& inst, Wave& wave, ImageData& image, std::string& err) {
    Img i{inst, wave, image, err};
    return i.run();
}

}  // namespace vpgcnref
