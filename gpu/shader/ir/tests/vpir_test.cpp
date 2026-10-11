// VPIR test driver: `roundtrip FILE`, `reject FILE`, `fbfetch IN [EXPECTED]`, `optimize IN EXPECTED`
// or `builder` (API smoke test).
#include <algorithm>
#include <bit>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

#include "vpir/builder.h"
#include "vpir/passes.h"
#include "vpir/text.h"
#include "vpir/verify.h"

using namespace vpir;

namespace {

std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int fail(const std::string& msg) {
    std::fprintf(stderr, "%s\n", msg.c_str());
    return 1;
}

std::string joined(const std::vector<std::string>& errors) {
    std::string s;
    for (const std::string& e : errors) s += e + "\n";
    return s;
}

// The canonical text of a file: itself minus whole-line comments (the printer emits none).
std::string withoutCommentLines(const std::string& text) {
    std::string out;
    for (size_t pos = 0; pos < text.size();) {
        size_t end = std::min(text.find('\n', pos), text.size() - 1) + 1;
        std::string_view l(text.data() + pos, end - pos);
        if (l.find_first_not_of(" \t") == std::string_view::npos || l[l.find_first_not_of(" \t")] != ';')
            out += l;
        pos = end;
    }
    return out;
}

int roundtrip(const char* path) {
    std::string text = readFile(path);
    ParseResult r = parse(text);
    if (!r.module) return fail("parse: " + r.error);
    if (auto errors = verify(*r.module); !errors.empty()) return fail("verify:\n" + joined(errors));
    std::string printed = print(*r.module);
    if (printed != withoutCommentLines(text)) return fail("not canonical; printed:\n" + printed);
    return 0;
}

int reject(const char* path) {
    std::string text = readFile(path);
    const std::string tag = "; expect: ";
    if (text.rfind(tag, 0) != 0) return fail("missing '; expect:' line");
    std::string want = text.substr(tag.size(), text.find('\n') - tag.size());
    ParseResult r = parse(text);
    std::string got = r.module ? joined(verify(*r.module)) : r.error;
    if (got.empty()) return fail("accepted an invalid module");
    if (got.find(want) == std::string::npos) return fail("expected '" + want + "', got:\n" + got);
    return 0;
}

// Values of "; KEY: ..." comment lines, in order.
std::vector<std::string> directives(const std::string& text, const std::string& key) {
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);)
        if (l.rfind("; " + key + ": ", 0) == 0) out.push_back(l.substr(key.size() + 4));
    return out;
}

// Parses, verifies, transforms and checks the result verifies and prints as EXPECTED.
template <typename Pass>
int transformed(const char* in, const char* expected, Pass pass) {
    std::string text = readFile(in);
    ParseResult r = parse(text);
    if (!r.module) return fail("parse: " + r.error);
    if (auto errors = verify(*r.module); !errors.empty()) return fail("input verify:\n" + joined(errors));
    if (int rc = pass(*r.module, text)) return rc;
    if (auto errors = verify(*r.module); !errors.empty()) return fail("output verify:\n" + joined(errors));
    std::string printed = print(*r.module);
    if (expected && printed != withoutCommentLines(readFile(expected))) return fail("pass output:\n" + printed);
    return 0;
}

// `; fetch-map: SLOT=ATTACHMENT ...`, `; nearest: SLOT ...`, one `; reject: SUBSTRING` per expected
// rejection (in order). Without EXPECTED the module must come out unchanged.
int fbfetch(const char* in, const char* expected) {
    return transformed(in, expected, [&](Module& m, const std::string& text) {
        std::map<uint32_t, uint32_t> att;
        std::set<uint32_t> nearest;
        for (const std::string& d : directives(text, "fetch-map")) {
            std::istringstream ss(d);
            for (uint32_t slot, a; ss >> slot && ss.ignore(1) && ss >> a;) att[slot] = a;
        }
        for (const std::string& d : directives(text, "nearest")) {
            std::istringstream ss(d);
            for (uint32_t slot; ss >> slot;) nearest.insert(slot);
        }
        std::string before = print(m);
        std::vector<FetchRejection> got = toFramebufferFetch(m, att, nearest);
        std::vector<std::string> want = directives(text, "reject");
        std::string list;
        for (const FetchRejection& j : got) list += j.function + " %" + std::to_string(j.value) + ": " + j.reason + "\n";
        bool ok = got.size() == want.size();
        for (size_t i = 0; ok && i < got.size(); ++i) ok = got[i].reason.find(want[i]) != std::string::npos;
        if (!ok) return fail("rejections differ; got:\n" + list);
        if (!expected && print(m) != before) return fail("module changed:\n" + print(m));
        return 0;
    });
}

// Builds a small fragment shader through the API and checks it prints as the expected text.
int builder() {
    Module m;
    m.stage = Stage::Fragment;
    m.inputs.resize(1);
    m.inputs[0].name = "uv";
    m.inputs[0].type = vec(kF32, 2);
    m.outputs.resize(1);
    m.outputs[0].name = "color";
    m.outputs[0].type = vec(kF32, 4);
    m.resources.resize(2);
    m.resources[0].name = "tex";
    m.resources[0].kind = ResourceKind::Texture;
    m.resources[1].name = "smp";
    m.resources[1].kind = ResourceKind::Sampler;

    FunctionBuilder b(m, "main");
    uint32_t acc = b.variable(kF32);
    ValueId uv = b.loadInput(0, vec(kF32, 2));
    ValueId c = b.imageSample(0, 1, uv, 0, {}, vec(kF32, 4));
    ValueId a = b.extract(c, 3, kF32);
    b.storeVar(acc, a);
    ValueId half = b.constF32(0.5f);
    b.beginIf(b.binary(Op::Lt, a, half, kBool));
    b.discard();
    b.beginElse();
    b.storeVar(acc, b.ternary(Op::Fma, a, half, half, kF32, true));
    b.end();
    b.storeOutput(0, c);
    b.finish();

    const char* expected =
        "module fragment\n"
        "\n"
        "input uv : vec2<f32> location(0)\n"
        "output color : vec4<f32> location(0)\n"
        "texture tex : 2d f32 index(0)\n"
        "sampler smp : index(0)\n"
        "\n"
        "func main() {\n"
        "  var $0 : f32\n"
        "  %0 = load_input uv : vec2<f32>\n"
        "  %1 = image_sample tex, smp, %0 : vec4<f32>\n"
        "  %2 = extract %1, 3 : f32\n"
        "  store_var $0, %2\n"
        "  %3 = const 0.5 : f32\n"
        "  %4 = lt %2, %3 : bool\n"
        "  if %4 {\n"
        "    discard\n"
        "  } else {\n"
        "    %5 = precise fma %2, %3, %3 : f32\n"
        "    store_var $0, %5\n"
        "  }\n"
        "  store_output color, %1\n"
        "}\n";
    if (auto errors = verify(m); !errors.empty()) return fail("verify:\n" + joined(errors));
    std::string text = print(m);
    if (text != expected) return fail("printed:\n" + text);
    ParseResult r = parse(text);
    if (!r.module || print(*r.module) != text) return fail("reparse mismatch");

    // Half-precision conversions round to nearest even and keep specials.
    if (floatToHalf(1.0f) != 0x3C00 || floatToHalf(65504.0f) != 0x7BFF || floatToHalf(65520.0f) != 0x7C00 ||
        floatToHalf(std::bit_cast<float>(0x33800000u)) != 0x0001 || halfToFloat(0x0001) != std::bit_cast<float>(0x33800000u) ||
        halfToFloat(0xFC00) != -std::numeric_limits<float>::infinity())
        return fail("half conversion");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "builder") return builder();
    if (argc == 3 && mode == "roundtrip") return roundtrip(argv[2]);
    if (argc == 3 && mode == "reject") return reject(argv[2]);
    if ((argc == 3 || argc == 4) && mode == "fbfetch") return fbfetch(argv[2], argc == 4 ? argv[3] : nullptr);
    if (argc == 4 && mode == "optimize")
        return transformed(argv[2], argv[3], [](Module& m, const std::string&) { return optimize(m), 0; });
    return fail("usage: vpir_test builder | roundtrip FILE | reject FILE | fbfetch IN [EXPECTED] | optimize IN EXPECTED");
}
