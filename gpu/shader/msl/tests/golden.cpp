// Golden test: golden IN.vpir EXPECTED.metal. With VPMSL_UPDATE=1 it rewrites EXPECTED instead.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "vpir/text.h"
#include "vpmsl/msl.h"

namespace {
std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: vpmsl_golden IN.vpir EXPECTED.metal\n");
        return 2;
    }
    vpir::ParseResult p = vpir::parse(readFile(argv[1]));
    if (!p.module) {
        std::fprintf(stderr, "parse: %s\n", p.error.c_str());
        return 1;
    }
    vpmsl::EmitResult r = vpmsl::emitMsl(*p.module);
    for (const std::string& e : r.errors) std::fprintf(stderr, "verify: %s\n", e.c_str());
    if (!r.errors.empty()) return 1;
    const char* update = std::getenv("VPMSL_UPDATE");
    if (update && *update == '1') {
        std::ofstream(argv[2], std::ios::binary) << r.shader.source;
        return 0;
    }
    if (r.shader.source != readFile(argv[2])) {
        std::fprintf(stderr, "output differs from %s; got:\n%s", argv[2], r.shader.source.c_str());
        return 1;
    }
    return 0;
}
