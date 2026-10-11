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
                                                 "SMRD", "VOP2", "VOP1", "VOPC", "VOP3"};
    return kNames[size_t(e)];
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
    std::printf("%u entries (%u invalid, %u unjudged skipped), %u failures\n", entries, invalid, skipped,
                failures);
    return failures ? 1 : 0;
}
