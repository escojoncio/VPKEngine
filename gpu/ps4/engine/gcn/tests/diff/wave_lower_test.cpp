// The wave-64 lowering of vp4ir::lowerWave64 against the module's own semantics: a compute module with
// wave_size 64 runs on 64 threads (one per lane, the subgroup ops being rendezvous over all 64), then lowered
// (two 32-lane groups exchanging through shared memory) on the same 64 threads; the output buffer must be equal.
//
// usage: vpgcn_wave_lower_test MODULE.vpir
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "vp4ir/passes.h"
#include "vp4ir/text.h"
#include "vp4ir/verify.h"
#include "vpir_eval.h"

namespace {

bool runModule(const vp4ir::Module& m, std::map<uint32_t, std::vector<uint8_t>>& memory, std::string& err) {
    std::vector<vpireval::Invocation> invs(64);
    for (unsigned lane = 0; lane < 64; ++lane) {
        vpireval::Value v;
        v.type = vp4ir::kU32;
        v.bits[0] = lane;
        invs[lane].builtins[vp4ir::Builtin::LocalInvocationIndex] = v;
        invs[lane].builtins[vp4ir::Builtin::SubgroupInvocation] = v;
        vpireval::Value sz = v;
        sz.bits[0] = m.waveSize;
        invs[lane].builtins[vp4ir::Builtin::SubgroupSize] = sz;
    }
    invs[0].memory = memory;
    if (!vpireval::runWorkgroup(m, invs, err)) return false;
    memory = invs[0].memory;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    std::stringstream ss;
    ss << std::ifstream(argv[1]).rdbuf();
    vp4ir::ParseResult p = vp4ir::parse(ss.str());
    if (!p.module) {
        std::fprintf(stderr, "parse: %s\n", p.error.c_str());
        return 1;
    }
    vp4ir::Module native = *p.module, lowered = *p.module;
    std::string why = vp4ir::lowerWave64(lowered);
    if (!why.empty()) {
        std::fprintf(stderr, "lowering: %s\n", why.c_str());
        return 1;
    }
    for (const std::string& e : vp4ir::verify(lowered)) {
        std::fprintf(stderr, "lowered module does not verify: %s\n", e.c_str());
        return 1;
    }
    std::map<uint32_t, std::vector<uint8_t>> a, b;
    for (uint32_t r = 0; r < native.resources.size(); ++r) a[r].assign(4096, 0xcd);
    b = a;
    std::string err;
    if (!runModule(native, a, err)) {
        std::fprintf(stderr, "native: %s\n", err.c_str());
        return 1;
    }
    if (!runModule(lowered, b, err)) {
        std::fprintf(stderr, "lowered: %s\n", err.c_str());
        return 1;
    }
    for (auto& [slot, bytes] : a) {
        if (slot == vp4ir::kTargetShared) continue;
        if (bytes != b[slot]) {
            for (size_t i = 0; i < bytes.size(); ++i)
                if (bytes[i] != b[slot][i]) {
                    std::fprintf(stderr, "resource %u differs at byte %zu: native 0x%02x lowered 0x%02x\n", slot, i, bytes[i], b[slot][i]);
                    break;
                }
            return 1;
        }
    }
    if (std::getenv("VPGCN_WAVE_PRINT"))
        for (auto& [slot, bytes] : a)
            if (slot != vp4ir::kTargetShared)
                for (unsigned lane : {0u, 1u, 40u, 63u}) {
                    std::printf("resource %u lane %u:", slot, lane);
                    for (unsigned k = 0; k < 8; ++k) std::printf(" %08x", *reinterpret_cast<const uint32_t*>(&bytes[lane * 32 + k * 4]));
                    std::printf("\n");
                }
    std::puts("wave-64 lowering matches the native semantics");
    return 0;
}
