#include "xfb.h"

#include <cstring>

namespace vpgles {

namespace {
constexpr uint32_t kLineLoop = 2, kLineStrip = 3, kTriangles = 4, kTriangleStrip = 5, kTriangleFan = 6, kPoints = 0, kLines = 1;

void assembleSequence(uint32_t mode, const uint32_t* s, size_t m, std::vector<uint32_t>& out) {
    switch (mode) {
        case kPoints: out.insert(out.end(), s, s + m); break;
        case kLines:
            for (size_t i = 0; i + 1 < m; i += 2) out.insert(out.end(), {s[i], s[i + 1]});
            break;
        case kLineStrip:
            for (size_t i = 0; i + 1 < m; ++i) out.insert(out.end(), {s[i], s[i + 1]});
            break;
        case kLineLoop:
            for (size_t i = 0; i + 1 < m; ++i) out.insert(out.end(), {s[i], s[i + 1]});
            if (m >= 2) out.insert(out.end(), {s[m - 1], s[0]});
            break;
        case kTriangles:
            for (size_t i = 0; i + 2 < m; i += 3) out.insert(out.end(), {s[i], s[i + 1], s[i + 2]});
            break;
        case kTriangleStrip:
            for (size_t i = 0; i + 2 < m; ++i)
                if (i & 1) out.insert(out.end(), {s[i + 1], s[i], s[i + 2]});
                else out.insert(out.end(), {s[i], s[i + 1], s[i + 2]});
            break;
        case kTriangleFan:
            // GL's fan: triangle i is (v_0, v_i, v_i+1), the provoking vertex (the last) last. (Vulkan's (v_i, v_i+1, v_0) puts the
            // first-vertex provoking vertex first: it is not GL's.) Not checked against dEQP / CTS: their sources are not in the tree.
            for (size_t i = 1; i + 1 < m; ++i) out.insert(out.end(), {s[0], s[i], s[i + 1]});
            break;
        default: break;
    }
}
}  // namespace

void xfbAssemble(uint32_t drawMode, const uint32_t* v, size_t n, bool restart, uint32_t restartValue, std::vector<uint32_t>& out) {
    size_t begin = 0;
    for (size_t i = 0; i <= n; ++i) {
        if (i < n && !(restart && v[i] == restartValue)) continue;
        assembleSequence(drawMode, v + begin, i - begin, out);
        begin = i + 1;
    }
}

uint64_t xfbWrite(const std::vector<uint32_t>& order, uint32_t perPrim, const uint32_t* records, uint32_t W, std::vector<XfbOut>& outs) {
    uint64_t written = 0;
    if (!perPrim) return 0;  // would never end: every zero-vertex "primitive" fits
    for (size_t p = 0; (p + 1) * perPrim <= order.size(); ++p) {
        for (const XfbOut& o : outs)
            if (o.used + uint64_t(perPrim) * o.stride > o.room) return written;
        for (uint32_t k = 0; k < perPrim; ++k) {
            const uint32_t* rec = records + size_t(order[p * perPrim + k]) * W;
            for (XfbOut& o : outs) {
                uint8_t* at = o.dst + o.used + uint64_t(k) * o.stride;
                for (const auto& [offset, words] : o.parts) {
                    std::memcpy(at, rec + offset, size_t(words) * 4);
                    at += size_t(words) * 4;
                }
            }
        }
        for (XfbOut& o : outs) o.used += uint64_t(perPrim) * o.stride;
        ++written;
    }
    return written;
}

}  // namespace vpgles
