// Reference DS (local data share) instructions for the differential tests, written from the AMD "Sea Islands Series
// Instruction Set Architecture" description of the DS encoding; independent of the lifter. See gcn_ref.h.
#include <bit>
#include <cmath>
#include <cstring>

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

    bool run() {
        if (in.op == Op::ds_swizzle_b32) return swizzle();
        return fail(std::string("unsupported ") + std::string(vpgcn::mnemonic(in.op)));
    }
};

}  // namespace

bool stepDs(const vpgcn::Instruction& inst, Wave& wave, Lds& lds, std::string& err) {
    Ds ds{inst, wave, lds, err};
    return ds.run();
}

}  // namespace vpgcnref
