// A module with `f64` ops gets the soft-float library once, calls to it and nothing else; one without does not.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "vp4ir/text.h"
#include "vp4msl/msl.h"

static std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static size_t count(const std::string& s, const std::string& what) {
    size_t n = 0;
    for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
    return n;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int bad = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", what);
            ++bad;
        }
    };
    {
        vp4ir::ParseResult p = vp4ir::parse(readFile(argv[1]));
        if (!p.module) {
            std::fprintf(stderr, "parse: %s\n", p.error.c_str());
            return 1;
        }
        vp4msl::EmitResult r = vp4msl::emitMsl(*p.module);
        for (const std::string& e : r.errors) std::fprintf(stderr, "emit: %s\n", e.c_str());
        const std::string& s = r.shader.source;
        check(r.errors.empty(), "emits");
        check(count(s, "inline uint2 vp_f64_fma(") == 1, "library emitted once");
        check(count(s, "inline u64 vpf_pack(") == 1, "library core present once");
        check(s.find("vp_f64_fma(") != std::string::npos && s.find("= vp_f64_fma(") != std::string::npos, "fma call");
        check(s.find("= vp_f64_div_scale(") != std::string::npos, "div_scale call");
        check(s.find("typedef ulong u64;") != std::string::npos, "prelude");
        check(s.find("VPF_TWO_OVER_PI_DATA") != std::string::npos, "table");
        // The library comes before the entry point.
        check(s.find("vp_f64_fma(") < s.find("vp_main"), "library before entry");
    }
    {
        vp4ir::ParseResult p = vp4ir::parse(readFile(argv[2]));
        if (!p.module) {
            std::fprintf(stderr, "parse: %s\n", p.error.c_str());
            return 1;
        }
        vp4msl::EmitResult r = vp4msl::emitMsl(*p.module);
        check(r.errors.empty(), "plain module emits");
        check(r.shader.source.find("vpf_") == std::string::npos, "no library without f64 ops");
    }
    return bad ? 1 : 0;
}
