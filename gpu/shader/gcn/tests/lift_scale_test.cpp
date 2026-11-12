// Lifter scaling. `lift_scale_test [diamonds]`: a compute program of N forward branch diamonds (2N basic blocks) must lift
// structured in a time that grows about linearly with N. The post-dominator sets per block made it cubic: 1200 blocks took
// ~15 s, 2000 blocks 143 s. `lift_scale_test nested [levels]`: N branches that all jump to the end (N nested ifs) lift
// (past 200 levels as the dispatch loop) and translate to MSL of a size that grows about linearly with N: the indentation
// grew with the square of the nesting (1000 levels = 28 MB, 5000 levels = 700 MB). Both lift on a 512 KB stack: the cycle
// check recursed once per block in a chain (a segmentation fault at ~8000 blocks), and the structured walk once per level.
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vpgcn/lift.h"
#include "vpmsl/msl.h"

namespace {
// The lifter runs on a worker thread in the client (macOS gives those 512 KB): so does the test.
struct Job {
    vpgcn::LiftInput in;
    vpgcn::LiftResult result;
};
void* runJob(void* p) {
    Job* j = static_cast<Job*>(p);
    j->result = vpgcn::lift(j->in);
    return nullptr;
}
vpgcn::LiftResult liftOnSmallStack(const vpgcn::LiftInput& in, size_t kb = 512) {
    Job job;
    job.in = in;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, kb * 1024);
    pthread_t th;
    if (pthread_create(&th, &attr, runJob, &job) != 0) std::abort();
    pthread_join(th, nullptr);
    pthread_attr_destroy(&attr);
    return std::move(job.result);
}

// Conditions a lane evaluates, at most, to reach a node: the If nodes of a block are tried in sequence (the i-th costs i),
// then one of its branches. The dispatch loop as a chain of `if (blk == k)` costs one per block of the program.
size_t worstPath(const vpir::Block& block) {
    size_t worst = 0, ifs = 0;
    for (const vpir::Node& n : block) {
        if (n.kind == vpir::NodeKind::If) worst = std::max(worst, ++ifs + std::max(worstPath(n.body), worstPath(n.elseBody)));
        else if (n.kind == vpir::NodeKind::Loop) worst = std::max(worst, ifs + worstPath(n.body));
    }
    return worst;
}
}  // namespace

// A fetch subroutine that itself calls the fetch subroutine (s_swappc_b64): lifting it inlined itself until the stack ended.
int fetchCallsItself() {
    const std::vector<uint32_t> program = {0xBE802102, 0xBF810000};  // s_swappc_b64 s[0:1], s[2:3]; s_endpgm
    vpgcn::LiftInput in;
    in.stage = vpgcn::ShaderStage::Vertex;
    in.code = program;
    in.fetch = program;
    const vpgcn::LiftResult r = liftOnSmallStack(in);
    if (r.errors.empty()) {
        std::fprintf(stderr, "no error for a recursive fetch subroutine\n");
        return 1;
    }
    return 0;
}

// Every s_swappc_b64 inlines the whole fetch subroutine: 1000 calls of a 500-instruction fetch is 500000 instructions
// (and ~100 MB of VPIR) from a 4 KB program. It is refused with an error that says so.
int fetchManyCalls() {
    std::vector<uint32_t> program(1000, 0xBE802102);  // s_swappc_b64 s[0:1], s[2:3]
    program.push_back(0xBF810000);                   // s_endpgm
    std::vector<uint32_t> fetch(500, 0xBE800301);    // s_mov_b32 s0, s1
    fetch.push_back(0xBE802002);                     // s_setpc_b64 s[2:3]
    vpgcn::LiftInput in;
    in.stage = vpgcn::ShaderStage::Vertex;
    in.code = program;
    in.fetch = fetch;
    const auto t0 = std::chrono::steady_clock::now();
    const vpgcn::LiftResult r = liftOnSmallStack(in);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    bool said = false;
    for (const std::string& e : r.errors) said |= e.find("is called 1000 times") != std::string::npos;
    if (!said) {
        std::fprintf(stderr, "no error for 1000 calls of a 500-instruction fetch subroutine (%zu errors): %s\n", r.errors.size(), r.errors.empty() ? "" : r.errors[0].c_str());
        return 1;
    }
    if (ms > 5000) {
        std::fprintf(stderr, "refusing took %.0f ms\n", ms);
        return 1;
    }
    // One call of the same fetch still lifts.
    program.assign({0xBE802102, 0xBF810000});
    in.code = program;
    const vpgcn::LiftResult one = liftOnSmallStack(in);
    if (!one.errors.empty()) {
        std::fprintf(stderr, "one call failed: %s\n", one.errors[0].c_str());
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "fetch_recursion") return fetchCallsItself();
    if (argc > 1 && std::string(argv[1]) == "fetch_calls") return fetchManyCalls();
    const bool nested = argc > 1 && std::string(argv[1]) == "nested";
    const int n = argc > (nested ? 2 : 1) ? std::atoi(argv[nested ? 2 : 1]) : 600;
    // `nested N KB`: the thread that lifts has KB kilobytes of stack (the walk checks the bytes left, not only its 200 levels).
    const size_t kb = nested && argc > 3 ? size_t(std::atoi(argv[3])) : 512;
    std::vector<uint32_t> code;
    for (int i = 0; i < n && nested; ++i) {
        code.push_back(0xBF000000);  // s_cmp_eq_i32 s0, s0
        code.push_back(0xBF850000 | uint32_t(3 * n - (3 * i + 2)));  // s_cbranch_scc1 to the s_endpgm
        code.push_back(0x80010101);  // s_add_u32 s1, s1, s1
    }
    for (int i = 0; i < n && !nested; ++i) {
        code.push_back(0xBF000000);  // s_cmp_eq_i32 s0, s0
        code.push_back(0xBF850001);  // s_cbranch_scc1 +1
        code.push_back(0x80010101);  // s_add_u32 s1, s1, s1
    }
    code.push_back(0xBF810000);  // s_endpgm
    vpgcn::LiftInput in;
    in.stage = vpgcn::ShaderStage::Compute;
    in.code = code;
    const auto t0 = std::chrono::steady_clock::now();
    vpgcn::LiftResult r = liftOnSmallStack(in, kb);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (const std::string& e : r.errors) std::fprintf(stderr, "lift: %s\n", e.c_str());
    if (!r.errors.empty()) return 1;
    if (!nested && !r.structured) {
        std::fprintf(stderr, "not structured\n");
        return 1;
    }
    if (ms > 3000) {
        std::fprintf(stderr, "%d %s took %.0f ms\n", n, nested ? "levels" : "diamonds", ms);
        return 1;
    }
    if (nested && !r.structured) {
        // The dispatch loop finds its block by binary search: log2(blocks) comparisons, not one per block.
        const size_t path = worstPath(r.module.functions[0].body);
        if (path > 40) {
            std::fprintf(stderr, "%d blocks: a jump evaluates %zu conditions\n", 2 * n, path);
            return 1;
        }
    }
    if (nested) {
        vpmsl::EmitResult e = vpmsl::emitMsl(r.module);
        for (const std::string& x : e.errors) std::fprintf(stderr, "emit: %s\n", x.c_str());
        if (!e.errors.empty()) return 1;
        if (e.shader.source.size() > 4000000) {
            std::fprintf(stderr, "%d nested levels = %zu bytes of MSL\n", n, e.shader.source.size());
            return 1;
        }
    }
    return 0;
}
