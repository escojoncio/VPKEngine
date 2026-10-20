// Golden test: golden IN.vpir EXPECTED.metal. With VPMSL_UPDATE=1 it rewrites EXPECTED instead.
// `golden IN.vpir --reject SUBSTRING` expects emission to fail with that error.
// `golden IN.vpir EXPECTED.metal --color-outputs MASK` also checks MslShader::colorOutputs.
// `golden IN.vpir EXPECTED.metal --rate-map` emits with Module::rateMapPosition set (foveation).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "vp4ir/text.h"
#include "vp4msl/msl.h"

namespace {
std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
}  // namespace

int main(int argc, char** argv) {
    const bool reject = argc == 4 && std::string(argv[2]) == "--reject";
    const bool mask = argc == 5 && std::string(argv[3]) == "--color-outputs";
    const bool rateMap = argc == 4 && std::string(argv[3]) == "--rate-map";
    if (argc != 3 && !reject && !mask && !rateMap) {
        std::fprintf(stderr, "usage: vp4msl_golden IN.vpir (EXPECTED.metal [--color-outputs MASK] | --reject SUBSTRING)\n");
        return 2;
    }
    vp4ir::ParseResult p = vp4ir::parse(readFile(argv[1]));
    if (!p.module) {
        std::fprintf(stderr, "parse: %s\n", p.error.c_str());
        return 1;
    }
    if (rateMap) p.module->rateMapPosition = true;
    vp4msl::EmitResult r = vp4msl::emitMsl(*p.module);
    if (reject) {
        for (const std::string& e : r.errors)
            if (e.find(argv[3]) != std::string::npos) return 0;
        std::fprintf(stderr, "expected an error containing '%s'\n", argv[3]);
        return 1;
    }
    for (const std::string& e : r.errors) std::fprintf(stderr, "verify: %s\n", e.c_str());
    if (!r.errors.empty()) return 1;
    if (mask && r.shader.colorOutputs != std::strtoul(argv[4], nullptr, 0)) {
        std::fprintf(stderr, "colorOutputs = 0x%02x, expected %s\n", r.shader.colorOutputs, argv[4]);
        return 1;
    }
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
