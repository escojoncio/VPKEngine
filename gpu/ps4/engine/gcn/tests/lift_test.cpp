// Lifter golden test: lift_test NAME.lift NAME.metal NAME.txt. The program is lifted, the module
// verified and translated to MSL; bindings, structure and notes must equal NAME.txt. NAME.metal is
// the reference MSL that CI compiles with Apple's Metal compiler: its text depends on the C++
// compiler's argument evaluation order (value numbering, order of reads), so it is not compared
// here. VPGCN_UPDATE=1 rewrites both.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vpgcn/lift.h"
#include "vp4ir/text.h"
#include "vp4msl/msl.h"

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

const char* kindName(vp4ir::ResourceKind k) { return vp4ir::toString(k); }

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
            input.stage = s == "vertex" ? vpgcn::ShaderStage::Vertex
                        : s == "pixel" ? vpgcn::ShaderStage::Pixel
                        : s == "export" ? vpgcn::ShaderStage::Export
                        : s == "geometry" ? vpgcn::ShaderStage::Geometry
                        : s == "copyvertex" ? vpgcn::ShaderStage::CopyVertex
                                            : vpgcn::ShaderStage::Compute;
        } else if (key == "vgpr_comp_cnt") {
            ls >> input.vgprCompCnt;
        } else if (key == "vs_out_cntl") {
            std::string v;
            ls >> v;
            input.vsOutCntl = uint32_t(std::stoul(v, nullptr, 0));
        } else if (key == "rect_list") {
            input.rectList = true;
        } else if (key == "geom") {
            for (std::string k, v; ls >> k >> v;) {
                const uint32_t n = uint32_t(std::stoul(v, nullptr, 0));
                if (k == "esgs_item") input.geom.esgsItemDwords = n;
                else if (k == "gsvs_vert") input.geom.gsvsVertDwords = n;
                else if (k == "max_vert") input.geom.maxVertOut = n;
                else if (k == "in_prim") input.geom.inputPrim = n;
                else if (k == "out_prim") input.geom.outPrim = n;
                else if (k == "gs_instances") input.geom.gsInstances = n;
            }
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
        else if (b.source.kind == vpgcn::DescriptorSource::Code) summary << "program image + " << b.source.path[0];
        else if (b.source.kind == vpgcn::DescriptorSource::Ring) summary << "ring " << b.source.sgpr;
        else {
            summary << "pointer s[" << b.source.sgpr << ":" << b.source.sgpr + 1 << "]";
            for (size_t i = 0; i < b.source.path.size(); ++i)
                summary << (i + 1 < b.source.path.size() ? " -> *(+" : " + ") << b.source.path[i] << (i + 1 < b.source.path.size() ? ")" : "");
        }
        summary << (b.dwords ? " (" + std::to_string(b.dwords) + "-dword descriptor)" : " (memory at that address)") << "\n";
    }
    for (const std::string& n : r.notes) summary << "note: " << n << "\n";
    for (const std::string& e : r.errors) {
        summary << "error: " << e << "\n";
        std::fprintf(stderr, "lift: %s\n", e.c_str());
    }
    if (!r.errors.empty()) return 1;
    vp4msl::EmitResult e = vp4msl::emitMsl(r.module);
    for (const std::string& x : e.errors) std::fprintf(stderr, "verify: %s\n", x.c_str());
    if (!e.errors.empty()) {
        std::fprintf(stderr, "%s", vp4ir::print(r.module).c_str());
        return 1;
    }
    const char* update = std::getenv("VPGCN_UPDATE");
    if (update && *update == '1') {
        std::ofstream(argv[2], std::ios::binary) << e.shader.source;
        std::ofstream(argv[3], std::ios::binary) << summary.str();
        return 0;
    }
    int status = 0;
    if (summary.str() != readFile(argv[3])) {
        std::fprintf(stderr, "summary differs from %s; got:\n%s", argv[3], summary.str().c_str());
        status = 1;
    }
    return status;
}
