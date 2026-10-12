#include "vpps4/fetch.h"

#include <algorithm>

#include "vpgcn/gcn.h"

namespace vpps4 {

using vpgcn::Op;
using vpgcn::OperandKind;

uint64_t fetchAddress(std::span<const uint32_t> vs, const std::array<uint32_t, 16>& userData, uint8_t userSgprs) {
    // Known SGPR values: user data, and what s_mov copies from them.
    std::array<int64_t, 104> known;
    known.fill(-1);
    for (unsigned i = 0; i < userSgprs && i < 16; ++i) known[i] = userData[i];
    size_t at = 0;
    while (at < vs.size()) {
        vpgcn::Instruction in;
        if (!vpgcn::decode(vs.subspan(at), in) || in.size < 4) return 0;
        at += in.size / 4;
        if (in.op == Op::s_endpgm) return 0;
        if (in.op == Op::s_swappc_b64) {
            const auto& s = in.src[0];
            if (s.kind != OperandKind::Sgpr || s.index + 1u >= known.size()) return 0;
            if (known[s.index] < 0 || known[s.index + 1] < 0) return 0;
            return uint64_t(known[s.index]) | uint64_t(known[s.index + 1]) << 32;
        }
        if ((in.op == Op::s_mov_b32 || in.op == Op::s_mov_b64) && in.dst.kind == OperandKind::Sgpr) {
            const unsigned n = in.op == Op::s_mov_b64 ? 2 : 1;
            for (unsigned k = 0; k < n && in.dst.index + k < known.size(); ++k) {
                int64_t v = -1;
                const auto& s = in.src[0];
                if (s.kind == OperandKind::Sgpr && s.index + k < known.size()) v = known[s.index + k];
                else if ((s.kind == OperandKind::Const || s.kind == OperandKind::Literal) && n == 1) v = s.value;
                known[in.dst.index + k] = v;
            }
        } else if (in.dst.kind == OperandKind::Sgpr) {
            for (unsigned k = 0; k < std::max<unsigned>(1, in.dst.count) && in.dst.index + k < known.size(); ++k)
                known[in.dst.index + k] = -1;
        }
    }
    return 0;
}

size_t fetchLength(std::span<const uint32_t> words) {
    size_t at = 0;
    while (at < words.size()) {
        vpgcn::Instruction in;
        if (!vpgcn::decode(words.subspan(at), in) || in.size < 4) return 0;
        at += in.size / 4;
        if (in.op == Op::s_setpc_b64) return at;
        if (in.op == Op::s_endpgm) return 0;
    }
    return 0;
}

}  // namespace vpps4
