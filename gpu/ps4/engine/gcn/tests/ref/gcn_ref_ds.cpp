// Reference DS (local data share) instructions for the differential tests, written from the AMD "Sea Islands Series
// Instruction Set Architecture" description of the DS encoding; independent of the lifter. See gcn_ref.h.
#include <bit>
#include <cmath>
#include <cstring>
#include <string>

#include "gcn_ref.h"

namespace vpgcnref {

namespace {

using vpgcn::Op;
using u32 = uint32_t;
using u64 = uint64_t;

struct Ds {
    const vpgcn::Instruction& in;
    Wave& w;
    Lds& lds;
    std::string& err;

    bool fail(const std::string& why) {
        err = why;
        return false;
    }

    bool active(unsigned lane) const { return w.exec >> lane & 1; }
    u32 vreg(unsigned reg, unsigned lane) const { return w.v[reg][lane]; }

    // ds_swizzle_b32: vdst[lane] = vaddr[source lane]. Bit 15 of the offset selects quad permutation mode (a
    // two-bit source per lane of the quad in bits 7:0); otherwise bit-mask mode on the lane id within each group of
    // 32 lanes: src = ((id & and) | or) ^ xor with and = [4:0], or = [9:5], xor = [14:10]. The source lane's value is
    // the register's, active or not.
    bool swizzle() {
        const unsigned off = in.mem.offset;
        Wave next = w;
        for (unsigned lane = 0; lane < 64; ++lane) {
            unsigned src;
            if (off & 0x8000) {
                src = (lane & ~3u) | ((off >> (2 * (lane & 3))) & 3);
            } else {
                unsigned id = lane & 31;
                id = ((id & (off & 31)) | ((off >> 5) & 31)) ^ ((off >> 10) & 31);
                src = (lane & 32) | id;
            }
            if (active(lane)) next.v[in.dst.index][lane] = vreg(in.src[0].index, src);
        }
        w = next;
        return true;
    }

    // ---- LDS memory (bytes; reads outside return 0, writes outside are dropped) ----
    u32 rd32(u64 a) const {
        if (a + 4 > lds.mem.size()) return 0;
        u32 v;
        std::memcpy(&v, &lds.mem[a], 4);
        return v;
    }
    void wr32(u64 a, u32 v) {
        if (a + 4 <= lds.mem.size()) std::memcpy(&lds.mem[a], &v, 4);
    }
    u64 rd64(u64 a) const { return u64(rd32(a)) | u64(rd32(a + 4)) << 32; }
    void wr64(u64 a, u64 v) {
        wr32(a, u32(v));
        wr32(a + 4, u32(v >> 32));
    }
    static float asF(u32 v) { return std::bit_cast<float>(v); }

    // One lane's operation. `a` is the byte address, d0/d1 the data operands. Returns the old value.
    u32 op32(const std::string& op, bool isF, bool isI, u32 a, u32 d0, u32 d1) {
        u32 t = rd32(a), n = t;
        if (op == "add") n = t + d0;
        else if (op == "sub") n = t - d0;
        else if (op == "rsub") n = d0 - t;
        else if (op == "inc") n = t >= d0 ? 0 : t + 1;
        else if (op == "dec") n = (t == 0 || t > d0) ? d0 : t - 1;
        else if (op == "min") n = isF ? (asF(d0) < asF(t) ? d0 : t) : isI ? (int32_t(d0) < int32_t(t) ? d0 : t) : (d0 < t ? d0 : t);
        else if (op == "max") n = isF ? (asF(d0) > asF(t) ? d0 : t) : isI ? (int32_t(d0) > int32_t(t) ? d0 : t) : (d0 > t ? d0 : t);
        else if (op == "and") n = t & d0;
        else if (op == "or") n = t | d0;
        else if (op == "xor") n = t ^ d0;
        else if (op == "mskor") n = (t & ~d0) | d1;
        else if (op == "write" || op == "wrxchg") n = d0;
        else if (op == "cmpst") n = (isF ? asF(t) == asF(d0) : t == d0) ? d1 : t;
        else if (op == "wrap") n = t >= d0 ? t - d0 : t + d1;
        wr32(a, n);
        return t;
    }
    u64 op64(const std::string& op, bool isI, u64 a, u64 d0, u64 d1) {
        u64 t = rd64(a), n = t;
        if (op == "add") n = t + d0;
        else if (op == "sub") n = t - d0;
        else if (op == "rsub") n = d0 - t;
        else if (op == "inc") n = t >= d0 ? 0 : t + 1;
        else if (op == "dec") n = (t == 0 || t > d0) ? d0 : t - 1;
        else if (op == "min") n = isI ? (int64_t(d0) < int64_t(t) ? d0 : t) : (d0 < t ? d0 : t);
        else if (op == "max") n = isI ? (int64_t(d0) > int64_t(t) ? d0 : t) : (d0 > t ? d0 : t);
        else if (op == "and") n = t & d0;
        else if (op == "or") n = t | d0;
        else if (op == "xor") n = t ^ d0;
        else if (op == "mskor") n = (t & ~d0) | d1;
        else if (op == "write" || op == "wrxchg") n = d0;
        else if (op == "cmpst") n = t == d0 ? d1 : t;
        wr64(a, n);
        return t;
    }

    bool run() {
        if (in.op == Op::ds_swizzle_b32) return swizzle();
        std::string name(vpgcn::mnemonic(in.op));
        if (name == "ds_nop") return true;
        if (name.find("gws") != std::string::npos || name == "ds_ordered_count" || name == "ds_append" || name == "ds_consume" || in.mem.gds)
            return fail("unsupported " + name);
        std::string n = name.substr(3);
        const std::string ty = n.substr(n.rfind('_') + 1);
        n.resize(n.rfind('_'));
        const bool wide = ty.size() == 3 && ty.substr(1) == "64", isF = ty[0] == 'f', isI = ty[0] == 'i';
        if (wide && isF) return fail("unsupported " + name);
        bool rtn = false, src2 = false;
        auto strip = [&](const char* tag) {
            size_t at = n.find(tag);
            if (at == std::string::npos) return false;
            n.erase(at, std::strlen(tag));
            return true;
        };
        rtn = strip("_rtn");
        src2 = strip("_src2");
        const unsigned esz = wide ? 8 : 4;
        const unsigned off0 = in.mem.offset & 0xff, off1 = in.mem.offset >> 8;
        Wave next = w;
        auto setRet = [&](unsigned lane, unsigned k, u32 v) { next.v[in.dst.index + k][lane] = v; };
        for (unsigned lane = 0; lane < 64; ++lane) {
            if (!active(lane)) continue;
            u32 addr = vreg(in.src[0].index, lane);
            auto data = [&](const vpgcn::Operand& o) -> u64 {
                if (o.kind != vpgcn::OperandKind::Vgpr) return 0;
                return wide ? u64(vreg(o.index, lane)) | u64(vreg(o.index + 1, lane)) << 32 : vreg(o.index, lane);
            };
            if (n == "read" || n == "read2" || n == "read2st64") {
                unsigned count = n == "read" ? 1 : 2;
                if (n == "read" && (ty == "b96" || ty == "b128")) {
                    unsigned words = ty == "b96" ? 3 : 4;
                    for (unsigned k = 0; k < words; ++k) setRet(lane, k, rd32(addr + in.mem.offset + 4 * k));
                    continue;
                }
                for (unsigned i = 0; i < count; ++i) {
                    u64 a = addr + (n == "read" ? in.mem.offset : (i ? off1 : off0) * esz * (n == "read2st64" ? 64 : 1));
                    if (ty == "i8" || ty == "u8") {
                        u32 b = rd32(a & ~3ull) >> (8 * (a & 3)) & 0xff;
                        setRet(lane, 0, ty == "i8" ? u32(int32_t(int8_t(b))) : b);
                    } else if (ty == "i16" || ty == "u16") {
                        u32 h = rd32(a & ~3ull) >> (8 * (a & 2)) & 0xffff;
                        setRet(lane, 0, ty == "i16" ? u32(int32_t(int16_t(h))) : h);
                    } else if (wide) {
                        u64 v = rd64(a);
                        setRet(lane, 2 * i, u32(v));
                        setRet(lane, 2 * i + 1, u32(v >> 32));
                    } else {
                        setRet(lane, i, rd32(a));
                    }
                }
                continue;
            }
            if (n == "write" && (ty == "b8" || ty == "b16") && !src2) {
                u64 a = addr + in.mem.offset;
                u32 word = rd32(a & ~3ull);
                unsigned sh = 8 * (a & 3);
                u32 mask = (ty == "b8" ? 0xffu : 0xffffu) << sh;
                wr32(a & ~3ull, (word & ~mask) | ((vreg(in.src[1].index, lane) << sh) & mask));
                continue;
            }
            if (n == "write" && (ty == "b96" || ty == "b128") && !src2) {
                unsigned words = ty == "b96" ? 3 : 4;
                for (unsigned k = 0; k < words; ++k) wr32(addr + in.mem.offset + 4 * k, vreg(in.src[1].index + k, lane));
                continue;
            }
            if (n == "write2" || n == "write2st64") {
                u64 mul = esz * (n == "write2st64" ? 64 : 1);
                for (unsigned i = 0; i < 2; ++i) {
                    u64 a = addr + (i ? off1 : off0) * mul, v = data(in.src[1 + i]);
                    wide ? wr64(a, v) : wr32(a, u32(v));
                }
                continue;
            }
            if (n == "wrxchg2" || n == "wrxchg2st64") {
                u64 mul = esz * (n == "wrxchg2st64" ? 64 : 1);
                for (unsigned i = 0; i < 2; ++i) {
                    u64 a = addr + (i ? off1 : off0) * mul, v = data(in.src[1 + i]);
                    if (wide) {
                        u64 old = rd64(a);
                        wr64(a, v);
                        setRet(lane, 2 * i, u32(old));
                        setRet(lane, 2 * i + 1, u32(old >> 32));
                    } else {
                        setRet(lane, i, op32("wrxchg", false, false, u32(a), u32(v), 0));
                    }
                }
                continue;
            }
            if (n == "condxchg32") {
                u32 a = addr + in.mem.offset;
                for (unsigned k = 0; k < 2; ++k) {
                    u32 old = rd32(a + 4 * k), v = vreg(in.src[1].index + k, lane);
                    setRet(lane, k, old);
                    if (v >> 31) wr32(a + 4 * k, v & 0x7fffffff);
                }
                continue;
            }
            // single-address read-modify-write, write
            u64 a, d0, d1;
            if (src2) {
                // A = address register; B = A + 4 * the signed 16-bit count in the offsets (or from A's top bits)
                int32_t cnt;
                if (off1 & 0x80) cnt = int16_t(((addr >> 31) << 15) | ((addr >> 17) & 0x7fff));
                else cnt = int16_t(((off1 & 0x40) << 9) | ((off1 & 0x7f) << 8) | off0);
                u32 b = addr + 4 * cnt;
                a = addr;
                d0 = wide ? rd64(b) : rd32(b);
                d1 = 0;
            } else {
                a = addr + in.mem.offset;
                d0 = data(in.src[1]);
                d1 = data(in.src[2]);
            }
            u64 old = wide ? op64(n, isI, a, d0, d1) : u64(op32(n, isF, isI, u32(a), u32(d0), u32(d1)));
            if (rtn) {
                setRet(lane, 0, u32(old));
                if (wide) setRet(lane, 1, u32(old >> 32));
            }
        }
        // (the return registers of an op that returns nothing stay as they were)
        w = next;
        return true;
    }
};

}  // namespace

bool stepDs(const vpgcn::Instruction& inst, Wave& wave, Lds& lds, std::string& err) {
    Ds ds{inst, wave, lds, err};
    return ds.run();
}

}  // namespace vpgcnref
