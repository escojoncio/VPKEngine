// Reference MUBUF instructions (untyped loads, stores, atomics) for the differential tests, written from the AMD "Sea
// Islands Series Instruction Set Architecture" buffer addressing description; independent of the lifter. See gcn_ref.h.
#include <bit>
#include <cmath>
#include <cstring>
#include <string>

#include "gcn_ref.h"

namespace vpgcnref {

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

struct Buf {
    const vpgcn::Instruction& in;
    Wave& w;
    BufferMemory& mem;
    Lds* lds;
    std::string& err;

    bool fail(const std::string& why) {
        err = why;
        return false;
    }

    // Per lane, the descriptor-relative byte address of the access and whether it is inside the buffer; `inside(k)` is
    // asked per dword (k bytes after the address).
    struct Where {
        u64 addr = 0;
        bool strided = false, indexBad = false;
        u64 records = 0;
        bool far = false;  // addr64 with bits above 31
    };

    u32 sgpr(unsigned i) const { return w.s[i]; }

    Where locate(unsigned lane) const {
        Where r;
        const vpgcn::MemoryFields& f = in.mem;
        // soffset: an SGPR or an inline constant
        u32 soff = 0;
        if (in.src[3].kind == vpgcn::OperandKind::Sgpr) soff = sgpr(in.src[3].index);
        else if (in.src[3].kind == vpgcn::OperandKind::Const) soff = in.src[3].value;
        if (f.addr64) {
            u64 a = u64(w.v[in.src[1].index][lane]) | u64(w.v[in.src[1].index + 1][lane]) << 32;
            a += f.offset;
            a += soff;
            r.addr = a & 0xffffffffull;
            r.far = (a >> 32) != 0;
            return r;
        }
        const unsigned rs = in.src[2].index;
        const u32 w1 = sgpr(rs + 1), w3 = sgpr(rs + 3);
        const u32 stride = w1 >> 16 & 0x3fff;
        const bool swizzle = w1 >> 31 & 1;
        const bool addTid = w3 >> 23 & 1;
        const unsigned elemSize = 2u << (w3 >> 19 & 3), indexStride = 8u << (w3 >> 21 & 3);
        u32 vreg = in.src[1].index;
        u32 index = 0, off = f.offset;
        if (f.idxen) index = w.v[vreg++][lane];
        if (f.offen) off += w.v[vreg][lane];
        if (addTid) index += lane;
        u64 bufOffset;
        if (stride && swizzle) {
            u64 im = index / indexStride, il = index % indexStride, om = off / elemSize, ol = off % elemSize;
            bufOffset = (im * stride + om * elemSize) * indexStride + il * elemSize + ol;
        } else {
            bufOffset = u64(index) * stride + off;
        }
        r.addr = (bufOffset + soff) & 0xffffffffull;
        r.strided = stride != 0;
        r.records = sgpr(rs + 2);
        r.indexBad = index >= r.records || ((f.idxen || addTid) && off >= stride);
        return r;
    }
    bool inside(const Where& r, unsigned extra) const {
        if (in.mem.addr64) return !r.far;
        if (r.strided) return !r.indexBad;
        return r.addr + extra < r.records;
    }

    u32 rd(u64 a) const {
        u32 v = 0;
        if (a + 4 <= mem.bytes.size()) std::memcpy(&v, &mem.bytes[a], 4);
        return v;
    }
    void wr(u64 a, u32 v) {
        if (a + 4 <= mem.bytes.size()) std::memcpy(&mem.bytes[a], &v, 4);
    }
    u64 rd64(u64 a) const { return u64(rd(a)) | u64(rd(a + 4)) << 32; }
    void wr64(u64 a, u64 v) {
        wr(a, u32(v));
        wr(a + 4, u32(v >> 32));
    }
    static float asF(u32 v) { return std::bit_cast<float>(v); }
    static double asD(u64 v) { return std::bit_cast<double>(v); }

    bool run() {
        std::string name(vpgcn::mnemonic(in.op));
        const bool isStore = name.find("_store_") != std::string::npos, isAtomic = name.find("_atomic_") != std::string::npos;
        Wave next = w;
        const unsigned data = in.src[0].index;
        for (unsigned lane = 0; lane < 64; ++lane) {
            if (!(w.exec >> lane & 1)) continue;
            Where r = locate(lane);
            if (isAtomic) {
                std::string op = name.substr(std::strlen("buffer_atomic_"));
                const bool wide = op.size() > 3 && op.substr(op.size() - 3) == "_x2";
                if (wide) op.resize(op.size() - 3);
                bool ok = inside(r, 0) && (!wide || inside(r, 7));
                u64 a = r.addr & ~3ull;
                u64 old = 0;
                if (ok) {
                    if (!wide) {
                        u32 t = rd(a), s = w.v[data][lane], c = w.v[data + 1][lane], n = t;
                        if (op == "swap") n = s;
                        else if (op == "cmpswap") n = t == c ? s : t;
                        else if (op == "add") n = t + s;
                        else if (op == "sub") n = t - s;
                        else if (op == "smin") n = int32_t(s) < int32_t(t) ? s : t;
                        else if (op == "umin") n = s < t ? s : t;
                        else if (op == "smax") n = int32_t(s) > int32_t(t) ? s : t;
                        else if (op == "umax") n = s > t ? s : t;
                        else if (op == "and") n = t & s;
                        else if (op == "or") n = t | s;
                        else if (op == "xor") n = t ^ s;
                        else if (op == "inc") n = t >= s ? 0 : t + 1;
                        else if (op == "dec") n = (t == 0 || t > s) ? s : t - 1;
                        else if (op == "fcmpswap") n = asF(t) == asF(c) ? s : t;
                        else if (op == "fmin") n = asF(s) < asF(t) ? s : t;
                        else if (op == "fmax") n = asF(s) > asF(t) ? s : t;
                        else return fail("unsupported " + name);
                        wr(a, n);
                        old = t;
                    } else {
                        u64 t = rd64(a), s = u64(w.v[data][lane]) | u64(w.v[data + 1][lane]) << 32,
                            c = u64(w.v[data + 2][lane]) | u64(w.v[data + 3][lane]) << 32, n = t;
                        if (op == "swap") n = s;
                        else if (op == "cmpswap") n = t == c ? s : t;
                        else if (op == "add") n = t + s;
                        else if (op == "sub") n = t - s;
                        else if (op == "smin") n = int64_t(s) < int64_t(t) ? s : t;
                        else if (op == "umin") n = s < t ? s : t;
                        else if (op == "smax") n = int64_t(s) > int64_t(t) ? s : t;
                        else if (op == "umax") n = s > t ? s : t;
                        else if (op == "and") n = t & s;
                        else if (op == "or") n = t | s;
                        else if (op == "xor") n = t ^ s;
                        else if (op == "inc") n = t >= s ? 0 : t + 1;
                        else if (op == "dec") n = (t == 0 || t > s) ? s : t - 1;
                        else if (op == "fcmpswap") n = asD(t) == asD(c) ? s : t;
                        else if (op == "fmin") n = asD(s) < asD(t) ? s : t;
                        else if (op == "fmax") n = asD(s) > asD(t) ? s : t;
                        else return fail("unsupported " + name);
                        wr64(a, n);
                        old = t;
                    }
                }
                if (in.mem.glc) {
                    next.v[data][lane] = u32(old);
                    if (wide) next.v[data + 1][lane] = u32(old >> 32);
                }
                continue;
            }
            if (isStore) {
                unsigned dw = in.src[0].count;
                if (name.ends_with("_byte") || name.ends_with("_short")) {
                    if (!inside(r, 0)) continue;
                    unsigned bytes = name.ends_with("_byte") ? 1 : 2;
                    u64 a = r.addr & ~u64(bytes - 1);
                    if (a < mem.bytes.size()) {
                        u32 v = w.v[data][lane];
                        for (unsigned k = 0; k < bytes; ++k)
                            if (a + k < mem.bytes.size()) mem.bytes[a + k] = uint8_t(v >> (8 * k));
                    }
                    continue;
                }
                if (name.find("_format_") != std::string::npos) {
                    // (no format known at build time: one 32-bit float per component, the element in or out as a whole)
                    unsigned n = name.back() == 'x' ? 1 : name.ends_with("xy") ? 2 : name.ends_with("xyz") ? 3 : 4;
                    static const unsigned bytesOf[5] = {0, 4, 8, 12, 16};
                    (void)bytesOf;
                    if (!(inside(r, 0) && inside(r, 4 * n - 1))) continue;
                    for (unsigned k = 0; k < n; ++k) wr((r.addr & ~3ull) + 4 * k, w.v[data + k][lane]);
                    continue;
                }
                for (unsigned k = 0; k < dw; ++k)
                    if (inside(r, 4 * k)) wr((r.addr & ~3ull) + 4 * k, w.v[data + k][lane]);
                continue;
            }
            // loads
            std::vector<u32> vals;
            if (name.ends_with("_ubyte") || name.ends_with("_sbyte") || name.ends_with("_ushort") || name.ends_with("_sshort")) {
                bool half = name.ends_with("short"), sgn = name.ends_with("_sbyte") || name.ends_with("_sshort");
                u64 a = half ? r.addr & ~1ull : r.addr;
                u32 v = 0;
                if (inside(r, 0))
                    for (unsigned k = 0; k < (half ? 2u : 1u); ++k)
                        if (a + k < mem.bytes.size()) v |= u32(mem.bytes[a + k]) << (8 * k);
                if (sgn) v = half ? u32(int32_t(int16_t(v))) : u32(int32_t(int8_t(v)));
                vals.push_back(v);
            } else if (name.find("_format_") != std::string::npos) {
                unsigned n = name.back() == 'x' ? 1 : name.ends_with("xy") ? 2 : name.ends_with("xyz") ? 3 : 4;
                bool ok = inside(r, 0) && inside(r, 4 * n - 1);
                for (unsigned k = 0; k < n; ++k) vals.push_back(ok ? rd((r.addr & ~3ull) + 4 * k) : 0);
            } else {
                unsigned dw = name.ends_with("x4") ? 4 : name.ends_with("x3") ? 3 : name.ends_with("x2") ? 2 : 1;
                for (unsigned k = 0; k < dw; ++k) vals.push_back(inside(r, 4 * k) ? rd((r.addr & ~3ull) + 4 * k) : 0);
            }
            if (in.mem.lds) {
                if (!lds) return fail("lds=1 without LDS");
                u64 la = u64(w.m0 & 0xffff) + in.mem.offset + 4ull * lane;
                if (la + 4 <= lds->mem.size()) std::memcpy(&lds->mem[la], &vals[0], 4);
                continue;
            }
            for (unsigned k = 0; k < vals.size(); ++k) next.v[data + k][lane] = vals[k];
            if (in.mem.tfe) next.v[data + vals.size()][lane] = 0;
        }
        w = next;
        return true;
    }
};

}  // namespace

bool stepBuffer(const vpgcn::Instruction& inst, Wave& wave, BufferMemory& mem, Lds* lds, std::string& err) {
    Buf b{inst, wave, mem, lds, err};
    return b.run();
}

}  // namespace vpgcnref
