// Lifts every shader of a shadPS4 dump folder (<stage>_<hash>[_perm].bin, vs_<hash>.fetch.bin) and
// reports what fails: lift errors, verifier errors, the dispatch-loop fallbacks and the notes, with
// a histogram per message. The dump has the code only, so the draw state is inferred: user SGPRs =
// the SGPRs read before they are written (at most 16), pixel inputs = perspective centre (and the
// position when the program reads VGPRs past the barycentrics), compute = all group ids on.
//
// usage: vpgcn_lift_batch DUMP_DIR [OUT_DIR]   (OUT_DIR receives NAME.metal for every success)
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <set>
#include <string>
#include <vector>

#include "vpgcn/gcn.h"
#include "vpgcn/lift.h"
#include "vpmsl/msl.h"

namespace fs = std::filesystem;

namespace {

std::vector<uint32_t> readWords(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> w(bytes.size() / 4);
    std::copy_n(bytes.data(), w.size() * 4, reinterpret_cast<char*>(w.data()));
    return w;
}

// SGPRs (codes < 104) read before any write, in straight-line order up to the first s_endpgm.
unsigned liveInSgprs(std::span<const uint32_t> code) {
    std::set<unsigned> written;
    int highest = -1;
    for (size_t pc = 0; pc < code.size();) {
        vpgcn::Instruction in;
        if (!vpgcn::decode(code.subspan(pc), in)) break;
        auto read = [&](const vpgcn::Operand& o) {
            if (o.kind != vpgcn::OperandKind::Sgpr) return;
            for (unsigned k = 0; k < o.count; ++k)
                if (!written.count(o.code + k)) highest = std::max(highest, int(o.code + k));
        };
        for (const vpgcn::Operand& o : in.src) read(o);
        if (in.encoding == vpgcn::Encoding::SOPK || in.encoding == vpgcn::Encoding::SOPC) read(in.dst);
        if (in.dst.kind == vpgcn::OperandKind::Sgpr && in.encoding != vpgcn::Encoding::SOPC)
            for (unsigned k = 0; k < in.dst.count; ++k) written.insert(in.dst.code + k);
        if (in.op == vpgcn::Op::s_endpgm) break;
        pc += in.size / 4;
    }
    return unsigned(std::min(highest + 1, 16));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: vpgcn_lift_batch DUMP_DIR [OUT_DIR]\n");
        return 2;
    }
    fs::path dir = argv[1];
    std::optional<fs::path> outDir;
    if (argc > 2) outDir = fs::path(argv[2]), fs::create_directories(*outDir);
    std::map<std::string, fs::path> fetchByHash;
    std::vector<fs::path> programs;
    for (const auto& e : fs::directory_iterator(dir)) {
        std::string n = e.path().filename().string();
        if (n.ends_with(".fetch.bin")) fetchByHash[n.substr(3, 18)] = e.path();
        else if (n.ends_with(".bin")) programs.push_back(e.path());
    }
    std::sort(programs.begin(), programs.end());
    std::map<std::string, unsigned> errors, notes;
    std::map<std::string, std::vector<std::string>> errorExamples;
    unsigned ok = 0, liftFail = 0, emitFail = 0, dispatch = 0;
    for (const fs::path& p : programs) {
        std::string name = p.stem().string();
        std::vector<uint32_t> code = readWords(p), fetch;
        vpgcn::LiftInput in;
        std::string stage = name.substr(0, 2);
        in.stage = stage == "vs" ? vpgcn::ShaderStage::Vertex : stage == "fs" ? vpgcn::ShaderStage::Pixel : vpgcn::ShaderStage::Compute;
        if (stage != "vs" && stage != "fs" && stage != "cs") continue;  // (other hardware stages: later)
        if (in.stage == vpgcn::ShaderStage::Vertex) {
            auto it = fetchByHash.find(name.substr(3, 18));
            if (it != fetchByHash.end()) fetch = readWords(it->second);
        }
        in.code = code;
        in.fetch = fetch;
        in.userSgprs = liveInSgprs(code);
        if (in.stage == vpgcn::ShaderStage::Compute) in.ldsBytes = 32768;
        if (in.stage == vpgcn::ShaderStage::Pixel) in.psInputAddr = 0x2 | 0x300;  // centre + position x, y
        vpgcn::LiftResult r = vpgcn::lift(in);
        for (const std::string& n : r.notes) ++notes[n];
        if (!r.errors.empty()) {
            ++liftFail;
            for (const std::string& e : r.errors) {
                ++errors[e];
                if (errorExamples[e].size() < 3) errorExamples[e].push_back(name);
            }
            continue;
        }
        if (!r.structured) ++dispatch;
        vpmsl::EmitResult m = vpmsl::emitMsl(r.module);
        if (!m.errors.empty()) {
            ++emitFail;
            for (const std::string& e : m.errors) {
                std::string key = "verify: " + e;
                ++errors[key];
                if (errorExamples[key].size() < 3) errorExamples[key].push_back(name);
            }
            continue;
        }
        ++ok;
        if (outDir) std::ofstream(*outDir / (name + ".metal")) << m.shader.source;
    }
    std::printf("%zu programs: %u lifted to MSL (%u through the dispatch loop), %u lift errors, %u verifier errors\n",
                programs.size(), ok, dispatch, liftFail, emitFail);
    std::vector<std::pair<unsigned, std::string>> sorted;
    for (auto& [e, n] : errors) sorted.push_back({n, e});
    std::sort(sorted.rbegin(), sorted.rend());
    for (auto& [n, e] : sorted) {
        std::printf("%4u  %s  (", n, e.c_str());
        for (const std::string& x : errorExamples[e]) std::printf(" %s", x.c_str());
        std::printf(" )\n");
    }
    std::printf("notes:\n");
    for (auto& [nt, n] : notes) std::printf("%4u  %s\n", n, nt.c_str());
    return 0;
}
