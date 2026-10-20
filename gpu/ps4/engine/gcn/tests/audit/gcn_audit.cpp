// Opcode audit of the lifter: every opcode of the tables (vpgcn/opcodes.h) is encoded with generic
// operands, decoded, and lifted as a one-instruction compute program. Classes:
//   implemented   lifts without error or note
//   approximate   lifts, but the lifter noted an approximation
//   rejected      the lifter reports an error (the draw would be skipped)
//   undecodable   the generic encoding used here did not decode (audit recipe problem)
// usage: vpgcn_audit [--list] [--expect-max-rejected N]
// Also a regression guard: with --expect-max-rejected N it fails when more than N ops are rejected
// or any op that decodes becomes undecodable.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "vpgcn/gcn.h"
#include "vpgcn/lift.h"

namespace {

struct Entry {
    const char* name;
    vpgcn::Encoding enc;
    unsigned opcode;
    const char* sig;
};

const Entry kTable[] = {
#define E(enc) \
    [](const char* n, unsigned o, const char* s) { return Entry{n, vpgcn::Encoding::enc, o, s}; }
#define X_SOP2(n, o, s) {#n, vpgcn::Encoding::SOP2, o, s},
#define X_SOP1(n, o, s) {#n, vpgcn::Encoding::SOP1, o, s},
#define X_SOPC(n, o, s) {#n, vpgcn::Encoding::SOPC, o, s},
#define X_SOPP(n, o, s) {#n, vpgcn::Encoding::SOPP, o, s},
#define X_SOPK(n, o, s) {#n, vpgcn::Encoding::SOPK, o, s},
#define X_SMRD(n, o, s) {#n, vpgcn::Encoding::SMRD, o, s},
#define X_VOP2(n, o, s) {#n, vpgcn::Encoding::VOP2, o, s},
#define X_VOP1(n, o, s) {#n, vpgcn::Encoding::VOP1, o, s},
#define X_VOPC(n, o, s) {#n, vpgcn::Encoding::VOPC, o, s},
#define X_VOP3(n, o, s) {#n, vpgcn::Encoding::VOP3, o, s},
#define X_VINTRP(n, o, s) {#n, vpgcn::Encoding::VINTRP, o, s},
#define X_DS(n, o, s) {#n, vpgcn::Encoding::DS, o, s},
#define X_MUBUF(n, o, s) {#n, vpgcn::Encoding::MUBUF, o, s},
#define X_MTBUF(n, o, s) {#n, vpgcn::Encoding::MTBUF, o, s},
#define X_MIMG(n, o, s) {#n, vpgcn::Encoding::MIMG, o, s},
#define X_FLAT(n, o, s) {#n, vpgcn::Encoding::FLAT, o, s},
    VPGCN_SOP2(X_SOP2) VPGCN_SOP1(X_SOP1) VPGCN_SOPC(X_SOPC) VPGCN_SOPP(X_SOPP) VPGCN_SOPK(X_SOPK)
    VPGCN_SMRD(X_SMRD) VPGCN_VOP2(X_VOP2) VPGCN_VOP1(X_VOP1) VPGCN_VOPC(X_VOPC) VPGCN_VOP3(X_VOP3)
    VPGCN_VINTRP(X_VINTRP) VPGCN_DS(X_DS) VPGCN_MUBUF(X_MUBUF) VPGCN_MTBUF(X_MTBUF) VPGCN_MIMG(X_MIMG)
    VPGCN_FLAT(X_FLAT)
};

unsigned digit(char c) { return c >= '0' && c <= '9' ? unsigned(c - '0') : 1; }

// The words of a one-instruction program for `e`, or empty when the recipe has no form for it.
std::vector<uint32_t> encode(const Entry& e) {
    using vpgcn::Encoding;
    std::string sig = e.sig;
    unsigned op = e.opcode;
    switch (e.enc) {
        case Encoding::SOP2: {
            unsigned n0 = digit(sig[1]), n1 = digit(sig[2]);
            // 'k': no literal, 64-bit; sources: SGPR pairs / quads; shifts take a 32-bit count.
            unsigned s0 = n0 >= 2 ? 4 : 4, s1 = n1 >= 2 ? 8 : 9;
            if (op == 43) {
                // cbranch_g_fork mask s[8:9], target s[4:5] built by s_getpc_b64 + s_add_u32 (the next instruction)
                return {0xbe841f00u, 0x80048804u, 0x80000000u | op << 23 | 4u << 8 | 8u};
            }
            return {0x80000000u | op << 23 | 20u << 16 | s1 << 8 | s0};
        }
        case Encoding::SOP1: {
            unsigned dst = sig[0] == '-' ? 0 : 20, s0 = sig[1] == '-' ? 0 : 4;
            if (sig[1] == 'r') s0 = 4;
            return {0xbe800000u | dst << 16 | op << 8 | s0};
        }
        case Encoding::SOPC: return {0xbf000000u | op << 16 | 9u << 8 | 4u};
        case Encoding::SOPP: return {0xbf800000u | op << 16 | 0u};
        case Encoding::SOPK:
            if (op == 21) return {0xb0000000u | op << 23 | (3u << 11 | 4u << 6 | 1u), 0x00000030u};  // s_setreg_imm32_b32 hwreg(MODE,..), literal
            return {0xb0000000u | op << 23 | 20u << 16 | (op == 17 ? 0u : 0x0001u)};
        case Encoding::SMRD: {
            unsigned dst = sig[0] == '-' ? 0 : 40;
            unsigned sbase = sig[1] == '-' ? 0 : sig[1] == '4' ? 2 : 0;  // pair for s_load (user ptr), quad for buffer
            return {0xc0000000u | op << 22 | dst << 15 | sbase << 9 | 0x100u | 0u};
        }
        case Encoding::VOP2:
        case Encoding::VOP1:
        case Encoding::VOPC: {
            std::string ops = sig.substr(0, sig.find('/'));
            unsigned fields[3] = {0, 0, 0}, nf = 0, vdst = 0;
            bool literal = false;
            for (size_t i = 0; i < ops.size(); ++i) {
                char c = ops[i];
                if (i == 0) {
                    if (c == 's') vdst = 30;
                    else if (c != '-') vdst = 8;
                    continue;
                }
                if (c == 'C' || c == '+' || c == 'M' || c == 'c') continue;
                if (c == 'k') { literal = true; continue; }
                unsigned code = (c == 's' || c == 'S') ? 4u : 256u + 1 + 4 * nf;
                if (e.enc != Encoding::VOP1 && nf == 1 && code < 256 && c != 's' && c != 'S') code = 256 + 5;  // vsrc1 is a VGPR
                fields[nf++] = code;
            }
            std::vector<uint32_t> w;
            if (e.enc == Encoding::VOP2) {
                uint32_t src0 = fields[0] ? fields[0] : 257, v1 = fields[1] ? (fields[1] & 0xff) : 5;
                w = {op << 25 | vdst << 17 | v1 << 9 | src0};
            } else if (e.enc == Encoding::VOP1) {
                uint32_t src0 = fields[0] ? fields[0] : 257;
                w = {0x7e000000u | vdst << 17 | op << 9 | src0};
            } else {
                uint32_t src0 = fields[0] ? fields[0] : 257, v1 = fields[1] ? (fields[1] & 0xff) : 5;
                w = {0x7c000000u | op << 17 | v1 << 9 | src0};
            }
            if (literal) w.push_back(0x3f800000u);
            return w;
        }
        case Encoding::VOP3: {
            std::string ops = sig.substr(0, sig.find('/'));
            uint32_t f[3] = {257, 261, 265};
            unsigned nf = 0, vdst = 8;
            bool b = false;
            for (size_t i = 0; i < ops.size(); ++i) {
                char c = ops[i];
                if (i == 0) {
                    if (c == '-') vdst = 0;
                    continue;
                }
                if (c == 'C') { b = true; continue; }
                if (c == '+') continue;
                if (c == 'c') { f[nf++] = 22; continue; }
                f[nf] = 257 + 4 * nf;
                ++nf;
            }
            uint32_t w0 = 0xd0000000u | op << 17 | vdst;
            if (b) w0 |= 20u << 8;
            return {w0, f[0] | f[1] << 9 | f[2] << 18};
        }
        case Encoding::DS: {
            uint32_t w0 = 0xd8000000u | op << 18 | 0;
            uint32_t addr = 1, d0 = 2, d1 = 6, dst = 10;
            return {w0, dst << 24 | d1 << 16 | d0 << 8 | addr};
        }
        case Encoding::MUBUF: {
            uint32_t w0 = 0xe0000000u | op << 18 | 1u << 12;  // offen
            return {w0, 128u << 24 | 1u << 16 | 8u << 8 | 1u};
        }
        case Encoding::MTBUF: {
            uint32_t w0 = 0xe8000000u | op << 16 | 1u << 12 | 14u << 19 | 7u << 23;  // offen, dfmt 14? nfmt 7
            w0 = (w0 & ~(0xfu << 19)) | 4u << 19;                                    // dfmt 4 (32), nfmt 7 float
            return {w0, 128u << 24 | 1u << 16 | 8u << 8 | 1u};
        }
        case Encoding::MIMG: {
            bool atomic = sig[0] == 'A';
            unsigned dmask = atomic ? (sig.size() > 2 && sig[2] == 'C' ? 3u : 1u) : 0xf;
            if (sig[0] == 'R') dmask = 0xf;
            uint32_t w0 = 0xf0000000u | op << 18 | dmask << 8;
            return {w0, 3u << 21 | 1u << 16 | 8u << 8 | 0u};
        }
        case Encoding::FLAT: {
            uint32_t w0 = 0xdc000000u | op << 18;
            return {w0, 12u << 24 | 8u << 8 | 0u};
        }
        case Encoding::VINTRP: return {0xc8000000u | uint32_t(op) << 16 | 1u << 10 | 0u << 8 | 8u << 18 | 1u};
        default: return {};
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool list = false;
    int expectMax = -1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--list")) list = true;
        else if (!std::strcmp(argv[i], "--expect-max-rejected") && i + 1 < argc) expectMax = std::atoi(argv[++i]);
    }
    std::map<std::string, std::vector<std::string>> classes;
    std::set<std::string> seen;
    for (const Entry& e : kTable) {
        std::vector<uint32_t> words = encode(e);
        if (words.empty()) { classes["undecodable"].push_back(e.name); continue; }
        // Round trip: the encoding must decode to this very op.
        std::vector<uint32_t> probe = words;
        probe.resize(probe.size() + 2, 0);
        vpgcn::Instruction in;
        bool good = false;
        for (size_t at = 0; at + 2 < probe.size();) {  // (the last instruction of the recipe is the op)
            if (!vpgcn::decode(std::span(probe).subspan(at), in)) break;
            at += in.size / 4;
            good = vpgcn::mnemonic(in.op) == e.name;
            if (at >= words.size()) break;
        }
        if (!good) {
            classes["undecodable"].push_back(e.name);
            continue;
        }
        words.resize(words.size());
        // Skip the control-flow ops that need a real program around them: they are audited by the
        // programs of the differential test.
        words.push_back(0xbf810000u);  // s_endpgm
        vpgcn::LiftInput li;
        li.stage = vpgcn::ShaderStage::Compute;
        li.userSgprs = 16;
        li.ldsBytes = 4096;
        li.workgroupSize[0] = 64;
        li.code = words;
        vpgcn::LiftResult r = vpgcn::lift(li);
        std::string tag = e.name;
        if (!r.errors.empty()) {
            classes["rejected"].push_back(tag + "  -- " + r.errors.front());
        } else {
            bool approx = false;
            std::string why;
            for (const std::string& n : r.notes) {
                if (n.find("zero padding") != std::string::npos) continue;
                approx = true;
                why = n;
            }
            if (approx) classes["approximate"].push_back(tag + "  -- " + why);
            else classes["implemented"].push_back(tag);
        }
    }
    size_t total = 0;
    for (auto& [k, v] : classes) total += v.size();
    std::printf("opcodes in the table: %zu\n", total);
    for (const char* k : {"implemented", "approximate", "rejected", "undecodable"}) {
        auto it = classes.find(k);
        size_t n = it == classes.end() ? 0 : it->second.size();
        std::printf("%-12s %zu\n", k, n);
        if (list && it != classes.end() && std::strcmp(k, "implemented"))
            for (auto& s : it->second) std::printf("  %s\n", s.c_str());
    }
    size_t rejected = classes["rejected"].size();
    if (expectMax >= 0 && (int(rejected) > expectMax || !classes["undecodable"].empty())) {
        std::printf("audit FAILED: rejected %zu (max %d), undecodable %zu\n", rejected, expectMax,
                    classes["undecodable"].size());
        return 1;
    }
    return 0;
}
