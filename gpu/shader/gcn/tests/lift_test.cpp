// Lifter golden test: lift_test NAME.lift NAME.metal NAME.txt. The program is lifted, the module
// verified and translated to MSL (must equal NAME.metal); bindings, structure and notes go to
// NAME.txt. VPGCN_UPDATE=1 rewrites both goldens.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vpgcn/lift.h"
#include "vpir/text.h"
#include "vpmsl/msl.h"

namespace {

std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<uint32_t> hexWords(std::istringstream& in) {
    std::vector<uint32_t> w;
    for (std::string t; in >> t;) w.push_back(uint32_t(std::stoul(t, nullptr, 16)));
    return w;
}

const char* kindName(vpir::ResourceKind k) { return vpir::toString(k); }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: lift_test NAME.lift NAME.metal NAME.txt\n");
        return 2;
    }
    vpgcn::LiftInput input;
    std::vector<uint32_t> code, fetch;
    std::istringstream file(readFile(argv[1]));
    for (std::string line; std::getline(file, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "stage") {
            std::string s;
            ls >> s;
            input.stage = s == "vertex" ? vpgcn::ShaderStage::Vertex : s == "pixel" ? vpgcn::ShaderStage::Pixel : vpgcn::ShaderStage::Compute;
        } else if (key == "user_sgprs") {
            ls >> input.userSgprs;
        } else if (key == "ps_input_addr") {
            std::string v;
            ls >> v;
            input.psInputAddr = uint32_t(std::stoul(v, nullptr, 0));
        } else if (key == "lds_bytes") {
            ls >> input.ldsBytes;
        } else if (key == "code") {
            code = hexWords(ls);
        } else if (key == "fetch") {
            fetch = hexWords(ls);
        }
    }
    input.code = code;
    input.fetch = fetch;
    vpgcn::LiftResult r = vpgcn::lift(input);
    std::ostringstream summary;
    summary << "structured " << (r.structured ? "yes" : "no (dispatch loop)") << "\n";
    if (r.userDataSlot) summary << "user data: slot " << *r.userDataSlot << "\n";
    for (const vpgcn::Binding& b : r.bindings) {
        summary << "binding slot " << b.slot << " " << kindName(b.kind) << " ";
        if (b.source.kind == vpgcn::DescriptorSource::UserData) summary << "user s" << b.source.sgpr;
        else summary << "pointer s[" << b.source.sgpr << ":" << b.source.sgpr + 1 << "] + " << b.source.offset;
        summary << (b.dwords ? " (" + std::to_string(b.dwords) + "-dword descriptor)" : " (memory at that address)") << "\n";
    }
    for (const std::string& n : r.notes) summary << "note: " << n << "\n";
    for (const std::string& e : r.errors) {
        summary << "error: " << e << "\n";
        std::fprintf(stderr, "lift: %s\n", e.c_str());
    }
    if (!r.errors.empty()) return 1;
    vpmsl::EmitResult e = vpmsl::emitMsl(r.module);
    for (const std::string& x : e.errors) std::fprintf(stderr, "verify: %s\n", x.c_str());
    if (!e.errors.empty()) {
        std::fprintf(stderr, "%s", vpir::print(r.module).c_str());
        return 1;
    }
    const char* update = std::getenv("VPGCN_UPDATE");
    if (update && *update == '1') {
        std::ofstream(argv[2], std::ios::binary) << e.shader.source;
        std::ofstream(argv[3], std::ios::binary) << summary.str();
        return 0;
    }
    int status = 0;
    if (e.shader.source != readFile(argv[2])) {
        std::fprintf(stderr, "MSL differs from %s; got:\n%s", argv[2], e.shader.source.c_str());
        status = 1;
    }
    if (summary.str() != readFile(argv[3])) {
        std::fprintf(stderr, "summary differs from %s; got:\n%s", argv[3], summary.str().c_str());
        status = 1;
    }
    return status;
}
