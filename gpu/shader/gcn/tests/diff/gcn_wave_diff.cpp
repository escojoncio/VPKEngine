// Differential test of the lifter's wave mode: each program of programs_wave.txt (gen_wave.py) is a compute shader
// of one 64-lane wave. The reference interpreter (gcn_ref, a real wave) runs it; the lifted VPIR runs on 64
// invocations twice: as lifted (subgroup ops over the 64-lane wave) and after vpir::lowerWave64 (two 32-lane SIMD
// groups exchanging through shared memory, which is what the Metal backend emits). The stored buffer must match
// the reference lane for lane.
//
// With --per-lane the programs lift as a 128-thread compute workgroup (more than a wave: no wave mode) and every lane
// runs as an independent invocation: the model of graphics stages and large workgroups (programs_spill.txt).
//
// With --buf the programs also address a second buffer (s[16:19], a V# whose four dwords come from the program's
// "vsharp" line; programs_buf.txt, gen_buf.py): the MUBUF reference (gcn_ref_buf.cpp) and the lifted code must leave
// that buffer (and the LDS, for lds=1 loads) identical too.
//
// usage: vpgcn_wave_diff programs_wave.txt [--verbose] [--per-lane] [--lds BYTES] [--buf [--known-layout]] [--img]
// Debugging: VPGCN_DIFF_PRINT=<index> prints that program's VPIR; VPGCN_DIFF_ONLY=<index> runs only that program.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../ref/gcn_ref.h"
#include "vpgcn/gcn.h"
#include "vpgcn/lift.h"
#include "vpir_eval.h"
#include "vpir/builder.h"
#include "vpir/passes.h"
#include "vpir/text.h"
#include "vpir/verify.h"

namespace {

// --img: the storage image behind the T# in s[20:27] ("timage dim width height depth levels" line of the program).
struct ImgSpec {
    bool present = false;
    unsigned dim = 1, width = 1, height = 1, depth = 1, levels = 1;
};
vpir::TexDim texDimOf(unsigned dim) {
    using D = vpir::TexDim;
    static const D map[] = {D::D1, D::D2, D::D3, D::D2Array, D::Cube, D::CubeArray};
    return map[dim];
}

struct Program {
    std::vector<std::string> source;
    std::vector<uint32_t> words;
    uint32_t vsharp[4] = {};  // --buf: the test buffer's V#
    ImgSpec img;              // --img
    bool ifBase = false;      // "ifbase": the module's count of `if` statements is the baseline of the programs after it
    int ifMax = -1;           // "ifmax K": at most K `if` statements more than the baseline
    int oobMax = -1;          // "oobmax K": the lifted code reads at most K times (summed over the lanes) past the test buffer
    bool noRef = false;       // "noref": not compared with the reference (it has no 32-bit wrap of the address; only ifmax / oobmax)
};

constexpr unsigned kTestBytes = 8192;
uint64_t gOutOfBufferReads = 0;  // (of the last runWave, over its invocations)



bool sameBits(uint32_t a, uint32_t b) {
    if (a == b) return true;
    if ((a | b) == 0x80000000u) return true;
    float fa, fb;
    std::memcpy(&fa, &a, 4);
    std::memcpy(&fb, &b, 4);
    return std::isnan(fa) && std::isnan(fb);
}

constexpr unsigned kOutBytes = 24 * 256;

// Runs the module on 64 invocations; the output buffer comes back in `out`.
bool runWave(const vpir::Module& m, const vpgcn::LiftResult& r, const std::vector<uint8_t>& ud, std::vector<uint8_t>& out,
             std::string& err, bool perLane = false, bool buf = false, std::vector<uint8_t>* test = nullptr, std::vector<uint8_t>* shared = nullptr,
             const ImgSpec* img = nullptr, std::vector<uint8_t>* imgData = nullptr) {
    std::vector<vpireval::Invocation> invs(64);
    for (unsigned lane = 0; lane < 64; ++lane) {
        auto u32 = [&](uint32_t v) {
            vpireval::Value x;
            x.type = vpir::kU32;
            x.bits[0] = v;
            return x;
        };
        auto vec3 = [&](uint32_t a, uint32_t b, uint32_t c) {
            vpireval::Value x;
            x.type = vpir::vec(vpir::kU32, 3);
            x.bits[0] = a, x.bits[1] = b, x.bits[2] = c;
            return x;
        };
        invs[lane].builtins[vpir::Builtin::LocalInvocationIndex] = u32(lane);
        invs[lane].builtins[vpir::Builtin::LocalInvocationId] = vec3(lane, 0, 0);
        invs[lane].builtins[vpir::Builtin::WorkgroupId] = vec3(0, 0, 0);
        invs[lane].builtins[vpir::Builtin::SubgroupInvocation] = u32(lane % m.waveSize);
        invs[lane].builtins[vpir::Builtin::SubgroupSize] = u32(m.waveSize);
    }
    if (r.userDataSlot) invs[0].memory[*r.userDataSlot] = ud;
    uint32_t outSlot = ~0u, testSlot = ~0u;
    for (const vpgcn::Binding& bd : r.bindings) {
        if (bd.kind != vpir::ResourceKind::Buffer || bd.source.kind != vpgcn::DescriptorSource::UserData) continue;
        if (buf && bd.source.sgpr == 16) testSlot = bd.slot;
        else outSlot = bd.slot;
    }
    if (buf && test && testSlot != ~0u) invs[0].memory[testSlot] = *test;
    uint32_t imgSlot = ~0u;
    if (img && img->present) {
        for (const vpgcn::Binding& bd : r.bindings)
            if ((bd.kind == vpir::ResourceKind::Texture || bd.kind == vpir::ResourceKind::StorageTexture) && bd.source.sgpr == 20)
                imgSlot = bd.slot;
        if (imgSlot != ~0u) {
            vpireval::ImageInfo info;
            info.dim = m.resources[imgSlot].dim;
            info.width = img->width, info.height = img->height, info.depth = img->depth, info.levels = img->levels;
            for (vpireval::Invocation& inv : invs) inv.images[imgSlot] = info;
            invs[0].memory[vpireval::kImageMemory + imgSlot] = *imgData;
        }
    }
    if (outSlot == ~0u) {
        err = "no output buffer binding";
        return false;
    }
    if (perLane) {
        // (independent invocations sharing the buffers; no rendezvous)
        std::map<uint32_t, std::vector<uint8_t>> memory = invs[0].memory;
        memory[outSlot].assign(kOutBytes, 0xcd);
        for (unsigned lane = 0; lane < 64; ++lane) {
            invs[lane].memory = memory;
            if (!vpireval::run(m, invs[lane], err)) return false;
            memory = invs[lane].memory;
            memory.erase(vpir::kTargetPrivate);  // (private memory is the invocation's own)
        }
        out = memory[outSlot];
        return true;
    }
    invs[0].memory[outSlot].assign(kOutBytes, 0xcd);
    if (!vpireval::runWorkgroup(m, invs, err)) return false;
    gOutOfBufferReads = 0;
    for (const vpireval::Invocation& inv : invs) gOutOfBufferReads += inv.outOfBufferReads;
    out = invs[0].memory[outSlot];
    if (buf && test) {
        *test = testSlot != ~0u ? invs[0].memory[testSlot] : std::vector<uint8_t>();
        test->resize(kTestBytes, 0);
    }
    if (shared) *shared = invs[0].memory[vpir::kTargetShared];
    if (imgSlot != ~0u) *imgData = invs[0].memory[vpireval::kImageMemory + imgSlot];
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: vpgcn_wave_diff programs_wave.txt [--verbose]\n");
        return 2;
    }
    bool verbose = false, perLane = false, bufMode = false, knownLayout = false, imgMode = false;
    unsigned ldsBytes = 0;
    for (int a = 2; a < argc; ++a) {
        if (std::string(argv[a]) == "--lds" && a + 1 < argc) ldsBytes = unsigned(std::atoi(argv[a + 1]));
        verbose |= std::string(argv[a]) == "--verbose";
        perLane |= std::string(argv[a]) == "--per-lane";
        bufMode |= std::string(argv[a]) == "--buf";
        imgMode |= std::string(argv[a]) == "--img";
        knownLayout |= std::string(argv[a]) == "--known-layout";  // (with --buf: tell the lifter the V#'s addressing mode)
    }
    std::ifstream in(argv[1]);
    std::vector<Program> programs;
    std::vector<std::string> pending;
    uint32_t pendingV[4] = {};
    ImgSpec pendingImg;
    bool pendingBase = false, pendingNoRef = false;
    int pendingIfMax = -1, pendingOobMax = -1;
    for (std::string line; std::getline(in, line);) {
        if (line.rfind("# ", 0) == 0 && line.find("generated by") == std::string::npos) pending.push_back(line.substr(2));
        if (line.rfind("vsharp ", 0) == 0) {
            std::istringstream ss(line.substr(7));
            for (unsigned k = 0; k < 4; ++k) {
                std::string t;
                ss >> t;
                pendingV[k] = uint32_t(std::stoul(t, nullptr, 16));
            }
        }
        if (line == "ifbase") pendingBase = true;
        if (line == "noref") pendingNoRef = true;
        if (line.rfind("oobmax ", 0) == 0) pendingOobMax = std::atoi(line.c_str() + 7);
        if (line.rfind("ifmax ", 0) == 0) pendingIfMax = std::atoi(line.c_str() + 6);
        if (line.rfind("timage ", 0) == 0) {
            std::istringstream ss(line.substr(7));
            ss >> pendingImg.dim >> pendingImg.width >> pendingImg.height >> pendingImg.depth >> pendingImg.levels;
            pendingImg.present = true;
        }
        if (line.rfind("prog ", 0) == 0) {
            Program p;
            p.img = pendingImg;
            p.ifBase = pendingBase, p.ifMax = pendingIfMax;
            p.oobMax = pendingOobMax, p.noRef = pendingNoRef;
            pendingBase = false, pendingIfMax = -1, pendingOobMax = -1, pendingNoRef = false;
            std::memcpy(p.vsharp, pendingV, sizeof p.vsharp);
            p.source = pending;
            pending.clear();
            std::istringstream ss(line.substr(5));
            for (std::string w; ss >> w;) p.words.push_back(uint32_t(std::stoul(w, nullptr, 16)));
            programs.push_back(p);
        }
    }
    const char* onlyEnv = std::getenv("VPGCN_DIFF_ONLY");
    const unsigned maxShow = std::getenv("VPGCN_DIFF_SHOW") ? unsigned(std::atoi(std::getenv("VPGCN_DIFF_SHOW"))) : 40;
    std::mt19937 rng(11);
    unsigned baselineIfs = 0;
    unsigned ok = 0, failed = 0, refUnsupported = 0, liftFailed = 0, evalFailed = 0, shown = 0, waveModeUsed = 0;
    std::map<std::string, unsigned> unsupported, liftErrors;
    for (size_t pi = 0; pi < programs.size(); ++pi) {
        if (onlyEnv && std::stoul(onlyEnv) != pi) continue;
        const Program& p = programs[pi];
        rng.seed(11 + unsigned(pi));
        uint32_t ud[28] = {};
        for (unsigned k = 0; k < 12; ++k) {
            unsigned kind = rng() % 4;
            ud[k] = kind == 0 ? rng() : kind == 1 ? rng() % 64 : kind == 2 ? (1u << (rng() % 32)) : uint32_t(-int(rng() % 9));
        }
        ud[12] = 0, ud[13] = 0, ud[14] = kOutBytes, ud[15] = 0;  // V#: raw buffer
        if (bufMode)
            for (unsigned k = 0; k < 4; ++k) ud[16 + k] = p.vsharp[k];
        vpgcnref::Wave w{};
        for (unsigned k = 0; k < 28; ++k) w.s[k] = ud[k];
        for (unsigned lane = 0; lane < 64; ++lane) w.v[0][lane] = lane;
        w.exec = ~0ull;
        std::vector<uint8_t> refMem(kOutBytes, 0xcd);
        vpgcnref::Lds lds;
        lds.mem.assign(ldsBytes, 0);
        vpgcnref::ImageData refImg;
        std::vector<uint8_t> imgInit;
        if (imgMode && p.img.present) {
            static const vpgcnref::ImageData::Dim dims[] = {vpgcnref::ImageData::D1, vpgcnref::ImageData::D2, vpgcnref::ImageData::D3,
                                                            vpgcnref::ImageData::D2Array, vpgcnref::ImageData::Cube, vpgcnref::ImageData::CubeArray};
            refImg.dim = dims[p.img.dim];
            refImg.width = p.img.width, refImg.height = p.img.height, refImg.depth = p.img.depth, refImg.levels = p.img.levels;
            vpireval::ImageInfo info;
            info.dim = texDimOf(p.img.dim);
            info.width = p.img.width, info.height = p.img.height, info.depth = p.img.depth, info.levels = p.img.levels;
            std::mt19937 irng(5000 + unsigned(pi));
            refImg.texels.resize(info.totalTexels() * 4);
            for (uint32_t& t : refImg.texels) t = irng() % 4 == 0 ? uint32_t(irng() % 6) : irng();
            imgInit.resize(refImg.texels.size() * 4);
            std::memcpy(imgInit.data(), refImg.texels.data(), imgInit.size());
        }
        vpgcnref::BufferMemory testMem;
        std::vector<uint8_t> testInit;
        if (bufMode) {
            std::mt19937 trng(1000 + unsigned(pi));
            testInit.resize(kTestBytes);
            for (unsigned i = 0; i < kTestBytes; i += 4) {
                uint32_t v = trng() % 5 == 0 ? uint32_t(trng() % 7) : trng();
                std::memcpy(&testInit[i], &v, 4);
            }
            testMem.bytes = testInit;
        }
        bool refOk = true;
        std::string err;
        unsigned steps = 0;
        for (size_t pc = 0; pc < p.words.size();) {
            vpgcn::Instruction x;
            if (!vpgcn::decode(std::span(p.words).subspan(pc), x)) {
                refOk = false;
                err = "undecodable";
                break;
            }
            pc += x.size / 4;
            if (x.op == vpgcn::Op::s_endpgm) break;
            if (++steps > 200000) {
                refOk = false;
                err = "runs too long";
                break;
            }
            {
                using vpgcn::Op;
                bool branch = true, taken = false;
                switch (x.op) {
                    case Op::s_branch: taken = true; break;
                    case Op::s_cbranch_scc0: taken = !w.scc; break;
                    case Op::s_cbranch_scc1: taken = w.scc; break;
                    case Op::s_cbranch_vccz: taken = w.vcc == 0; break;
                    case Op::s_cbranch_vccnz: taken = w.vcc != 0; break;
                    case Op::s_cbranch_execz: taken = w.exec == 0; break;
                    case Op::s_cbranch_execnz: taken = w.exec != 0; break;
                    default: branch = false; break;
                }
                if (branch) {
                    if (taken) pc = size_t(int64_t(pc) + int16_t(x.simm16));
                    continue;
                }
            }
            if (x.op == vpgcn::Op::s_waitcnt || x.op == vpgcn::Op::s_barrier) continue;  // (one wave, instructions complete in order)
            if (imgMode && x.encoding == vpgcn::Encoding::MIMG) {
                if (!vpgcnref::stepImage(x, w, refImg, err)) {
                    refOk = false;
                    break;
                }
                continue;
            }
            if (bufMode && x.encoding == vpgcn::Encoding::MUBUF && x.src[2].index == 16) {
                if (!vpgcnref::stepBuffer(x, w, testMem, ldsBytes ? &lds : nullptr, err)) {
                    refOk = false;
                    break;
                }
                continue;
            }
            if (x.op == vpgcn::Op::buffer_store_dword) {
                for (unsigned lane = 0; lane < 64; ++lane) {
                    if (!(w.exec >> lane & 1)) continue;
                    uint32_t addr = w.v[x.src[1].index][lane] + x.mem.offset;
                    if (addr + 4 <= kOutBytes) std::memcpy(&refMem[addr], &w.v[x.src[0].index][lane], 4);
                }
                continue;
            }
            if (std::getenv("VPGCN_DIFF_TRACE")) std::printf("ref: %s  v5[0]=%08x v5[1]=%08x\n", vpgcn::disassemble(x).c_str(), w.v[5][0], w.v[5][1]);
            if (x.encoding == vpgcn::Encoding::DS ? !vpgcnref::stepDs(x, w, lds, err) : !vpgcnref::step(x, w, err)) {
                refOk = false;
                break;
            }
        }
        if (!refOk) {
            ++refUnsupported;
            ++unsupported[err];
            continue;
        }
        vpgcn::LiftInput li;
        li.stage = vpgcn::ShaderStage::Compute;
        li.userSgprs = imgMode ? 28 : bufMode ? 20 : 16;
        li.tidigCompCnt = 0;
        li.workgroupSize[0] = perLane ? 128 : 64, li.workgroupSize[1] = 1, li.workgroupSize[2] = 1;
        li.ldsBytes = ldsBytes;
        li.code = p.words;
        if (imgMode && p.img.present) {
            const unsigned dim = p.img.dim;
            li.describe = [=](const vpgcn::DescriptorSource& src) -> std::optional<vpgcn::DescriptorInfo> {
                if (src.kind != vpgcn::DescriptorSource::UserData || src.sgpr != 20) return std::nullopt;
                vpgcn::DescriptorInfo d;
                d.dim = texDimOf(dim);
                d.elem = vpir::Scalar::U32;
                return d;
            };
        }
        if (bufMode && knownLayout) {
            const uint32_t w1 = p.vsharp[1], w3 = p.vsharp[3];
            li.describe = [=](const vpgcn::DescriptorSource& src) -> std::optional<vpgcn::DescriptorInfo> {
                if (src.kind != vpgcn::DescriptorSource::UserData || src.sgpr != 16) return std::nullopt;
                vpgcn::DescriptorInfo d;
                d.layoutKnown = true;
                d.swizzled = (w1 >> 31 & 1) && (w1 >> 16 & 0x3fff);
                d.addTid = w3 >> 23 & 1;
                return d;
            };
        }
        vpgcn::LiftResult r = vpgcn::lift(li);
        if (!r.errors.empty()) {
            ++liftFailed;
            ++liftErrors[r.errors.front()];
            if (verbose) std::printf("program %zu: lift error %s\n", pi, r.errors.front().c_str());
            continue;
        }
        for (const std::string& n : r.notes)
            if (verbose) std::printf("program %zu: note %s\n", pi, n.c_str());
        bool usesWave = false;
        for (const vpir::Function& f : r.module.functions) (void)f;
        if (vpir::print(r.module).find("subgroup_") != std::string::npos) usesWave = true;
        waveModeUsed += usesWave;
        bool tooManyIfs = false;
        if (p.ifBase || p.ifMax >= 0) {
            // (the `if` statements of the lifted module, printed one per line)
            unsigned ifs = 0;
            std::istringstream text(vpir::print(r.module));
            for (std::string l; std::getline(text, l);) {
                const size_t at = l.find_first_not_of(' ');
                ifs += at != std::string::npos && l.compare(at, 3, "if ") == 0;
            }
            if (p.ifBase) {
                baselineIfs = ifs;  // (not run: it touches no test buffer to compare)
                ++ok;
                continue;
            }
            if (ifs > baselineIfs + unsigned(p.ifMax)) {
                tooManyIfs = true;
                std::printf("IF COUNT program %zu: %u `if` statements, at most %u wanted (%u baseline + %d)\n", pi, ifs, baselineIfs + p.ifMax,
                            baselineIfs, p.ifMax);
                for (const std::string& s : p.source) std::printf("    %s\n", s.c_str());
            }
        }
        const char* printEnv = std::getenv("VPGCN_DIFF_PRINT");
        if (printEnv && std::stoul(printEnv) == pi) std::printf("%s\n", vpir::print(r.module).c_str());
        std::vector<uint8_t> udBytes(112);
        std::memcpy(udBytes.data(), ud, 112);
        vpir::Module lowered = r.module;
        std::string why = vpir::lowerWave64(lowered);
        if (!why.empty()) {
            std::printf("program %zu: lowering: %s\n", pi, why.c_str());
            ++liftFailed;
            continue;
        }
        bool mismatch = false, evalOk = true;
        for (int variant = 0; variant < (perLane ? 1 : 2) && !mismatch && evalOk; ++variant) {
            std::vector<uint8_t> got, gotTest = testInit, gotShared, gotImg = imgInit;
            std::string e;
            if (!runWave(variant ? lowered : r.module, r, udBytes, got, e, perLane, bufMode, &gotTest, &gotShared, &p.img, &gotImg)) {
                evalOk = false;
                std::printf("program %zu (%s): eval error %s\n", pi, variant ? "lowered" : "native", e.c_str());
                break;
            }
            if (p.oobMax >= 0 && gOutOfBufferReads > uint64_t(p.oobMax)) {
                mismatch = true;
                std::printf("OOB READS program %zu (%s): %llu reads past the test buffer, at most %d wanted\n", pi, variant ? "lowered" : "native",
                            (unsigned long long)gOutOfBufferReads, p.oobMax);
                for (const std::string& s : p.source) std::printf("    %s\n", s.c_str());
            }
            if (p.noRef) continue;
            for (unsigned i = 0; i < kOutBytes && !mismatch; i += 4) {
                uint32_t want, have;
                std::memcpy(&want, &refMem[i], 4);
                std::memcpy(&have, &got[i], 4);
                if (sameBits(want, have)) continue;
                mismatch = true;
                if (std::getenv("VPGCN_DIFF_ALL")) {
                    for (unsigned j = i; j < kOutBytes; j += 4) {
                        uint32_t w1, h1;
                        std::memcpy(&w1, &refMem[j], 4);
                        std::memcpy(&h1, &got[j], 4);
                        if (!sameBits(w1, h1)) std::printf("  diff v%u lane %u: ref 0x%08x got 0x%08x\n", j / 256, (j % 256) / 4, w1, h1);
                    }
                }
                if (shown++ < maxShow) {
                    unsigned reg = i / 256, lane = (i % 256) / 4;
                    std::printf("MISMATCH program %zu (%s) lane %u v%u: reference 0x%08x, lifted 0x%08x\n", pi,
                                variant ? "lowered" : "native", lane, reg, want, have);
                    std::printf("  user data:");
                    for (unsigned k = 0; k < 12; ++k) std::printf(" s%u=0x%x", k, ud[k]);
                    std::printf("\n");
                    for (const std::string& s : p.source) std::printf("    %s\n", s.c_str());
                }
            }
            if (imgMode && p.img.present && !mismatch) {
                std::vector<uint8_t> want(refImg.texels.size() * 4);
                std::memcpy(want.data(), refImg.texels.data(), want.size());
                for (size_t i = 0; i + 4 <= want.size() && !mismatch; i += 4) {
                    uint32_t a = 0, c = 0;
                    std::memcpy(&a, &want[i], 4);
                    if (i + 4 <= gotImg.size()) std::memcpy(&c, &gotImg[i], 4);
                    if (a == c) continue;
                    mismatch = true;
                    if (shown++ < maxShow) {
                        std::printf("MISMATCH program %zu (%s) image dword %zu: reference 0x%08x, lifted 0x%08x\n", pi,
                                    variant ? "lowered" : "native", i / 4, a, c);
                        std::printf("  image dim=%u %ux%ux%u levels=%u\n", p.img.dim, p.img.width, p.img.height, p.img.depth, p.img.levels);
                        for (const std::string& s : p.source) std::printf("    %s\n", s.c_str());
                    }
                }
            }
            if (bufMode && !mismatch) {
                // the test buffer (and the LDS, for lds=1 loads) must end up the same
                auto compare = [&](const char* what, const std::vector<uint8_t>& want, const std::vector<uint8_t>& have) {
                    for (size_t i = 0; i + 4 <= want.size() && !mismatch; i += 4) {
                        uint32_t a = 0, c = 0;
                        std::memcpy(&a, &want[i], 4);
                        if (i + 4 <= have.size()) std::memcpy(&c, &have[i], 4);
                        if (a == c) continue;
                        mismatch = true;
                        if (shown++ < maxShow) {
                            std::printf("MISMATCH program %zu (%s) %s byte %zu: reference 0x%08x, lifted 0x%08x\n", pi,
                                        variant ? "lowered" : "native", what, i, a, c);
                            std::printf("  V# 0x%x 0x%x 0x%x 0x%x; user data:", p.vsharp[0], p.vsharp[1], p.vsharp[2], p.vsharp[3]);
                            for (unsigned k = 0; k < 12; ++k) std::printf(" s%u=0x%x", k, ud[k]);
                            std::printf("\n");
                            for (const std::string& s : p.source) std::printf("    %s\n", s.c_str());
                        }
                    }
                };
                compare("test buffer", testMem.bytes, gotTest);
                if (ldsBytes) compare("LDS", lds.mem, gotShared);
            }
        }
        if (!evalOk) ++evalFailed;
        else if (mismatch || tooManyIfs) ++failed;
        else ++ok;
    }
    std::printf("%zu programs: %u match (%u with subgroup ops), %u mismatch, %u lift errors, %u eval errors, %u not run (reference)\n",
                programs.size(), ok, waveModeUsed, failed, liftFailed, evalFailed, refUnsupported);
    for (auto& [e, n] : unsupported) std::printf("  reference: %u x %s\n", n, e.c_str());
    for (auto& [e, n] : liftErrors) std::printf("  lift: %u x %s\n", n, e.c_str());
    return failed || liftFailed || evalFailed ? 1 : 0;
}
