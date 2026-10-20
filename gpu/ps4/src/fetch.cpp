#include "vpps4/fetch.h"

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <utility>

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
            if (known[s.index] < 0 || known[s.index + 1] < 0) {
                // The linear walk saw a write to the pair (often on a path that does not reach
                // the call, the walk ignores branches): fall back to the user-data registers themselves.
                if (s.index + 1u < std::min<unsigned>(userSgprs, 16)) return uint64_t(userData[s.index]) | uint64_t(userData[s.index + 1]) << 32;
                return 0;
            }
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

uint64_t FetchAddressMemo::address(std::span<const uint32_t> vs, const std::array<uint32_t, 16>& userData,
                                   uint8_t userSgprs) {
    Entry& e = entries_[(reinterpret_cast<uintptr_t>(vs.data()) >> 3) % kEntries];
    const auto value = [&](const Source& s) { return s.kind == Source::User ? uint64_t(userData[s.value]) : uint64_t(s.value); };
    if (e.code == vs.data() && e.userSgprs == userSgprs && !e.words.empty() && vs.size() >= e.words.size() &&
        std::memcmp(vs.data(), e.words.data(), 4 * e.words.size()) == 0) {
        if (!e.valid) return 0;
        return value(e.lo) | value(e.hi) << 32;
    }
    // The same walk as fetchAddress(), with each SGPR's origin instead of its value.
    Entry n;
    n.code = vs.data();
    n.userSgprs = userSgprs;
    std::array<Source, 104> known{};
    for (unsigned i = 0; i < userSgprs && i < 16; ++i) known[i] = {Source::User, i};
    size_t at = 0;
    bool done = false;
    while (at < vs.size() && !done) {
        vpgcn::Instruction in;
        if (!vpgcn::decode(vs.subspan(at), in) || in.size < 4) return fetchAddress(vs, userData, userSgprs);  // (not kept)
        at += in.size / 4;
        if (in.op == Op::s_endpgm) {
            done = true;
        } else if (in.op == Op::s_swappc_b64) {
            done = true;
            const auto& s = in.src[0];
            if (s.kind == OperandKind::Sgpr && s.index + 1u < known.size()) {
                if (known[s.index].kind != Source::Unknown && known[s.index + 1].kind != Source::Unknown) {
                    n.valid = true;
                    n.lo = known[s.index], n.hi = known[s.index + 1];
                } else if (s.index + 1u < std::min<unsigned>(userSgprs, 16)) {
                    n.valid = true;  // fall back to the user-data registers themselves
                    n.lo = {Source::User, s.index}, n.hi = {Source::User, s.index + 1u};
                }
            }
        } else if ((in.op == Op::s_mov_b32 || in.op == Op::s_mov_b64) && in.dst.kind == OperandKind::Sgpr) {
            const unsigned cnt = in.op == Op::s_mov_b64 ? 2 : 1;
            for (unsigned k = 0; k < cnt && in.dst.index + k < known.size(); ++k) {
                Source v;
                const auto& s = in.src[0];
                if (s.kind == OperandKind::Sgpr && s.index + k < known.size()) v = known[s.index + k];
                else if ((s.kind == OperandKind::Const || s.kind == OperandKind::Literal) && cnt == 1)
                    v = {Source::Const, uint32_t(s.value)};
                known[in.dst.index + k] = v;
            }
        } else if (in.dst.kind == OperandKind::Sgpr) {
            for (unsigned k = 0; k < std::max<unsigned>(1, in.dst.count) && in.dst.index + k < known.size(); ++k)
                known[in.dst.index + k] = {};
        }
    }
    if (!done) return fetchAddress(vs, userData, userSgprs);  // ran off the end of the span: not kept
    n.words.assign(vs.begin(), vs.begin() + std::ptrdiff_t(at));
    e = std::move(n);
    if (!e.valid) return 0;
    return value(e.lo) | value(e.hi) << 32;
}

}  // namespace vpps4
