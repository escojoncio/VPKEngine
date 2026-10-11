// Checks the decoder and disassembler against the llvm-mc golden corpus: for every entry, the
// valid/invalid verdict, the instruction length and the exact text.
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "vpgcn/gcn.h"

namespace {

const char* encodingName(vpgcn::Encoding e) {
    static constexpr const char* kNames[] = {"SOP2", "SOP1", "SOPC", "SOPP", "SOPK",
                                                 "SMRD", "VOP2", "VOP1", "VOPC", "VOP3", "VINTRP", "DS",
                                                 "MUBUF", "MTBUF", "MIMG", "FLAT", "EXP"};
    return kNames[size_t(e)];
}

// Field checks where the oracle cannot judge (llvm-mc refuses the spelling) or the text does not
// show what the lifter reads.
unsigned fieldChecks() {
    using namespace vpgcn;
    unsigned failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) std::fprintf(stderr, "FAIL field check: %s\n", what), ++failures;
    };
    Instruction in;
    // buffer_load_dword v10, off, s[8:11], s1 lds: data goes to LDS, no VGPR result.
    check(decode(std::vector<uint32_t>{0xE0310000, 0x01020A00}, in) && in.dst.kind == OperandKind::None &&
              in.mem.lds,
          "MUBUF lds load writes no VGPR");
    // buffer_atomic_cmpswap v[10:11], off, s[8:11], s1 glc: returns v10 only.
    check(decode(std::vector<uint32_t>{0xE0C44000, 0x01020A00}, in) && in.src[0].count == 2 &&
              in.dst.index == 10 && in.dst.count == 1,
          "MUBUF cmpswap returns half its data");
    // image_atomic_cmpswap v[10:11], v[20:21], s[8:15] dmask:0x3 glc: returns v10.
    check(decode(std::vector<uint32_t>{0xF0402300, 0x00020A14}, in) && in.dst.count == 1,
          "MIMG cmpswap returns half its data");
    // image_sample dmask:0x1 lwe: one more dword (the LOD warning status).
    check(decode(std::vector<uint32_t>{0xF0820100, 0x00820A14}, in) && in.src[0].count == 2,
          "MIMG lwe adds a status dword");
    // exp mrt0 v1, off, v3, off: disabled sources are None.
    check(decode(std::vector<uint32_t>{0xF8000005, 0x04030201}, in) && in.src[0].index == 1 &&
              in.src[1].kind == OperandKind::None && in.src[2].index == 3 &&
              in.src[3].kind == OperandKind::None,
          "EXP disabled sources are None");
    // exp mrt0 compr, en 0xc: only the second packed pair (vsrc1) is exported.
    check(decode(std::vector<uint32_t>{0xF800040C, 0x00000201}, in) &&
              in.src[0].kind == OperandKind::None && in.src[1].index == 2 &&
              in.src[2].kind == OperandKind::None,
          "EXP compressed pairs");
    // addr64 with offen: not an addressing mode.
    check(!decode(std::vector<uint32_t>{0xE0309000, 0x01020A14}, in), "MUBUF addr64 + offen rejected");
    // Truncated memory instruction.
    check(!decode(std::vector<uint32_t>{0xE0300000}, in), "truncated MUBUF rejected");
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s corpus.txt\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::map<std::string, unsigned> validPerEncoding;
    unsigned entries = 0, invalid = 0, skipped = 0, failures = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find(' ');
        std::string hex = line.substr(0, sp), expected = line.substr(sp + 1);
        std::vector<uint32_t> words;
        std::stringstream ss(hex);
        for (std::string w; std::getline(ss, w, ',');)
            words.push_back(uint32_t(std::stoul(w, nullptr, 16)));
        if (expected == "?") {  // hardware-defined, but the oracle cannot judge it
            ++skipped;
            continue;
        }
        ++entries;
        bool valid = expected != "!";
        vpgcn::Instruction inst;
        bool ok = vpgcn::decode(words, inst);
        std::string got = ok ? vpgcn::disassemble(inst) : "!";
        bool pass = ok == valid && (!valid || (got == expected && inst.size == words.size() * 4));
        // A literal-carrying instruction cut before its literal must be rejected.
        vpgcn::Instruction cut;
        bool cutAccepted = valid && words.size() > 1 && vpgcn::decode(std::span(words).first(1), cut);
        if ((!pass || cutAccepted) && ++failures <= 40)
            std::fprintf(stderr, "FAIL %s: expected '%s' (%zu bytes), got '%s' (%u bytes)%s\n",
                         hex.c_str(), expected.c_str(), words.size() * 4, got.c_str(),
                         ok ? inst.size : 0u, cutAccepted ? ", truncated literal accepted" : "");
        if (valid && ok) ++validPerEncoding[encodingName(inst.encoding)];
        invalid += !valid;
    }
    for (auto& [enc, n] : validPerEncoding) std::printf("%-5s %u valid\n", enc.c_str(), n);
    failures += fieldChecks();
    std::printf("%u entries (%u invalid, %u unjudged skipped), %u failures\n", entries, invalid, skipped,
                failures);
    return failures ? 1 : 0;
}
