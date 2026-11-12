// Lifter scaling. `lift_scale_test [diamonds]`: a compute program of N forward branch diamonds (2N basic blocks) must lift
// structured in a time that grows about linearly with N. The post-dominator sets per block made it cubic: 1200 blocks took
// ~15 s, 2000 blocks 143 s. `lift_scale_test nested [levels]`: N branches that all jump to the end (N nested ifs) lift
// (past the walk limit, 240 levels, as the dispatch loop) and translate to MSL of a size that grows about linearly with N: the indentation
// grew with the square of the nesting (1000 levels = 28 MB, 5000 levels = 700 MB). Both lift on a 512 KB stack: the cycle
// check recursed once per block in a chain (a segmentation fault at ~8000 blocks), and the structured walk once per level.
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
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

// The dispatch loop of the lifter finds the block to run by binary search on the block number (an `if (blk < mid)` tree
// above chains of at most 8 `if (blk == k)`). Differential test against the plain chain, which runs block k exactly when
// blk == k: the tree is evaluated for every block number (and a few past the end) and must reach the same block.
struct DispatchEval {
    std::map<vpir::ValueId, uint32_t> value;
    std::map<vpir::ValueId, uint32_t> eqBlock;  // value of an `eq blk, k` -> k
    std::set<uint32_t> blocks;                  // every k a leaf tests
    // Every block a leaf of the tree tests, whatever blk is.
    void collect(const vpir::Block& b) {
        for (const vpir::Node& n : b) {
            if (n.kind == vpir::NodeKind::Inst && n.inst.op == vpir::Op::Const) value[n.inst.result] = n.inst.lits.empty() ? 0 : n.inst.lits[0];
            if (n.kind == vpir::NodeKind::Inst && n.inst.op == vpir::Op::Eq) eqBlock[n.inst.result] = value[n.inst.args[1]];
            if (n.kind != vpir::NodeKind::If) continue;
            if (auto e = eqBlock.find(n.value); e != eqBlock.end()) {
                blocks.insert(e->second);
            } else {
                collect(n.body);
                collect(n.elseBody);
            }
        }
    }
    // The block the tree enters for blk == v: -1 none.
    long target(const vpir::Block& b, uint32_t v) {
        for (const vpir::Node& n : b) {
            if (n.kind == vpir::NodeKind::Inst) {
                const vpir::Inst& in = n.inst;
                switch (in.op) {
                    case vpir::Op::LoadVar: value[in.result] = v; break;
                    case vpir::Op::Const: value[in.result] = in.lits.empty() ? 0 : in.lits[0]; break;
                    case vpir::Op::Lt: value[in.result] = value[in.args[0]] < value[in.args[1]]; break;
                    case vpir::Op::Eq:
                        value[in.result] = value[in.args[0]] == value[in.args[1]];
                        eqBlock[in.result] = value[in.args[1]];
                        break;
                    default: break;
                }
            } else if (n.kind == vpir::NodeKind::If) {
                const bool c = value[n.value] != 0;
                if (auto e = eqBlock.find(n.value); e != eqBlock.end()) {  // a leaf: `if (blk == k) { the block }`
                    blocks.insert(e->second);
                    if (c) return long(e->second);
                    continue;
                }
                return target(c ? n.body : n.elseBody, v);  // an `if (blk < mid)` of the tree
            }
        }
        return -1;
    }
};
const vpir::Block* dispatchTree(const vpir::Block& body) {
    for (const vpir::Node& n : body)
        if (n.kind == vpir::NodeKind::Loop) return &n.body;
    return nullptr;
}
int dispatchSearch() {
    int dispatched = 0;
    for (int n : {5, 9, 13, 20, 37, 64, 100, 150, 300}) {
        std::vector<uint32_t> code;
        for (int i = 0; i < n; ++i) {
            code.push_back(0xBF000000);  // s_cmp_eq_i32 s0, s0
            code.push_back(0xBF850000 | uint32_t(3 * n - (3 * i + 2)));  // s_cbranch_scc1 to the s_endpgm
            code.push_back(0x80010101);  // s_add_u32 s1, s1, s1
        }
        code.push_back(0xBF810000);  // s_endpgm
        vpgcn::LiftInput in;
        in.stage = vpgcn::ShaderStage::Compute;
        in.code = code;
        // A stack too small for the structured walk (and a nest deeper than its level limit for 300) takes the dispatch loop.
        const vpgcn::LiftResult r = liftOnSmallStack(in, 48);
        if (!r.errors.empty()) {
            std::fprintf(stderr, "%d: lift: %s\n", n, r.errors[0].c_str());
            return 1;
        }
        if (r.structured) continue;
        const vpir::Block* tree = dispatchTree(r.module.functions[0].body);
        if (!tree) {
            std::fprintf(stderr, "%d: no dispatch loop\n", n);
            return 1;
        }
        DispatchEval probe;
        probe.collect(*tree);
        const uint32_t count = uint32_t(probe.blocks.size());
        if (count > 8) ++dispatched;  // (up to 8 blocks the dispatch is the plain chain)
        // (The chain would have one `blk == k` for every block: 0 .. count - 1, each once.)
        if (probe.blocks.empty() || *probe.blocks.rbegin() != count - 1) {
            std::fprintf(stderr, "%d: the dispatch tests %u blocks, last %u\n", n, count, probe.blocks.empty() ? 0 : *probe.blocks.rbegin());
            return 1;
        }
        for (uint32_t v = 0; v < count + 4; ++v) {
            DispatchEval e;
            const long got = e.target(*tree, v);
            const long want = v < count ? long(v) : -1;
            if (got != want) {
                std::fprintf(stderr, "%d blocks: blk = %u reaches %ld, the chain runs %ld\n", int(count), v, got, want);
                return 1;
            }
        }
    }
    if (dispatched < 3) {
        std::fprintf(stderr, "only %d of the programs took the dispatch search tree (more than 8 blocks)\n", dispatched);
        return 1;
    }
    return 0;
}

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
    if (argc > 1 && std::string(argv[1]) == "dispatch_search") return dispatchSearch();
    if (argc > 1 && std::string(argv[1]) == "fetch_calls") return fetchManyCalls();
    const bool nested = argc > 1 && std::string(argv[1]) == "nested";
    const int n = argc > (nested ? 2 : 1) ? std::atoi(argv[nested ? 2 : 1]) : 600;
    // `nested N KB`: the thread that lifts has KB kilobytes of stack (the walk checks the bytes left, not only its 240 levels).
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
    // `nested N KB structured`: N levels under the walk limit lift as structured control flow (not as the dispatch loop).
    if (nested && argc > 4 && std::string(argv[4]) == "structured" && !r.structured) {
        std::fprintf(stderr, "%d nested levels were lifted as the dispatch loop\n", n);
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
