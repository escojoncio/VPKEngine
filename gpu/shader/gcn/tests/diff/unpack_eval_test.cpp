// The evaluator's packed-normalized unpacks (unpack_unorm4x8 / snorm4x8 / unorm2x16 / snorm2x16) on msl/tests/unpack.vpir.
//
// usage: vpgcn_unpack_eval_test unpack.vpir
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "vpir/text.h"
#include "vpir/verify.h"
#include "vpir_eval.h"

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    std::stringstream ss;
    ss << std::ifstream(argv[1]).rdbuf();
    vpir::ParseResult p = vpir::parse(ss.str());
    if (!p.module) return std::printf("parse: %s\n", p.error.c_str()), 1;
    if (!vpir::verify(*p.module).empty()) return std::printf("verify failed\n"), 1;
    vpireval::Invocation inv;
    std::string err;
    if (!vpireval::run(*p.module, inv, err)) return std::printf("run: %s\n", err.c_str()), 1;

    const uint32_t word = 2164195073u;  // the constant of unpack.vpir
    auto byte = [&](unsigned i) { return (word >> (8 * i)) & 0xffu; };
    auto half = [&](unsigned i) { return (word >> (16 * i)) & 0xffffu; };
    double want[4];
    for (unsigned c = 0; c < 4; ++c) {
        const double snorm8 = std::max(double(int8_t(byte(c))) / 127.0, -1.0);
        const double unorm8 = double(byte(c)) / 255.0;
        const double unorm16 = c < 2 ? double(half(c)) / 65535.0 : double(half(c - 2)) / 65535.0;
        const double snorm16 = c < 2 ? std::max(double(int16_t(half(c))) / 32767.0, -1.0)
                                     : std::max(double(int16_t(half(c - 2))) / 32767.0, -1.0);
        want[c] = snorm8 + unorm8 + unorm16 + snorm16;
    }
    const vpireval::Value& got = inv.outputs.at(0);
    for (unsigned c = 0; c < 4; ++c) {
        const double g = std::bit_cast<float>(got.bits[c]);
        if (std::fabs(g - want[c]) > 1e-5) return std::printf("component %u: got %f want %f\n", c, g, want[c]), 1;
    }
    return 0;
}
