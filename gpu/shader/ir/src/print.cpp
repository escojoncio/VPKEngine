#include <bit>
#include <charconv>
#include <cmath>

#include "text_common.h"
#include "vpir/text.h"

namespace vpir {

namespace {

class Printer {
public:
    explicit Printer(const Module& m) : m_(m) {}

    std::string run() {
        line("module ", toString(m_.stage));
        if (m_.stage == Stage::Compute || m_.stage == Stage::Object || m_.stage == Stage::Mesh)
            line("workgroup_size ", std::to_string(m_.workgroupSize[0]), " ",
                 std::to_string(m_.workgroupSize[1]), " ", std::to_string(m_.workgroupSize[2]));
        if (m_.sharedBytes) line("shared_bytes ", std::to_string(m_.sharedBytes));
        if (m_.waveSize != 32) line("wave_size ", std::to_string(m_.waveSize));
        if (m_.payloadBytes) line("payload_bytes ", std::to_string(m_.payloadBytes));
        if (m_.privateBytes) line("private_bytes ", std::to_string(m_.privateBytes));
        for (uint32_t b : m_.privateSlots) line("private_slot ", std::to_string(b));
        if (m_.stage == Stage::Mesh)
            line("mesh ", toString(m_.topology), " max_vertices ", std::to_string(m_.maxVertices),
                 " max_primitives ", std::to_string(m_.maxPrimitives));

        if (!m_.inputs.empty() || !m_.outputs.empty() || !m_.resources.empty()) out_ += '\n';
        for (const IoVar& v : m_.inputs) io("input", v);
        for (const IoVar& v : m_.outputs) io("output", v);
        for (const Resource& r : m_.resources) resource(r);

        for (unsigned i = 0; i < m_.functions.size(); ++i) {
            out_ += '\n';
            function(m_.functions[i]);
        }
        return std::move(out_);
    }

private:
    template <typename... T>
    void line(const T&... parts) {
        out_.append(depth_ * 2, ' ');
        ((out_ += parts), ...);
        out_ += '\n';
    }

    void io(const char* kw, const IoVar& v) {
        std::string s = std::string(kw) + " " + v.name + " : " + toString(v.type);
        if (v.builtin != Builtin::None) {
            s += std::string(" builtin(") + toString(v.builtin) + ")";
        } else {
            s += " location(" + std::to_string(v.location) + ")";
            if (v.interp != Interp::Perspective) s += std::string(" ") + toString(v.interp);
            if (v.sampling != Sampling::Center) s += std::string(" ") + toString(v.sampling);
        }
        if (v.perPrimitive) s += " per_primitive";
        line(s);
    }

    void resource(const Resource& r) {
        std::string s = std::string(toString(r.kind)) + " " + r.name + " :";
        if (r.kind == ResourceKind::Texture || r.kind == ResourceKind::StorageTexture) {
            s += std::string(" ") + toString(r.dim) + " " + toString(Type(r.elem));
            if (r.depth) s += " depth";
            if (r.multisample) s += " multisample";
        }
        s += " index(" + std::to_string(r.index) + ")";
        if (r.kind == ResourceKind::Buffer || r.kind == ResourceKind::StorageTexture)
            s += std::string(" ") + toString(r.access);
        line(s);
    }

    void function(const Function& f) {
        std::string head = "func " + f.name + "(";
        for (unsigned i = 0; i < f.params.size(); ++i) {
            if (i) head += ", ";
            head += "%" + std::to_string(f.params[i]) + " : " + toString(f.typeOf(f.params[i]));
        }
        head += ")";
        if (!f.returnType.isVoid()) head += " -> " + toString(f.returnType);
        line(head, " {");
        ++depth_;
        for (unsigned i = 0; i < f.vars.size(); ++i)
            line("var $", std::to_string(i), " : ", toString(f.vars[i].type));
        block(f.body);
        --depth_;
        line("}");
    }

    void block(const Block& b) {
        for (const Node& n : b) node(n);
    }

    void node(const Node& n) {
        switch (n.kind) {
            case NodeKind::Inst: line(inst(n.inst)); break;
            case NodeKind::If:
                line("if %", std::to_string(n.value), " {");
                nested(n.body);
                if (!n.elseBody.empty()) {
                    line("} else {");
                    nested(n.elseBody);
                }
                line("}");
                break;
            case NodeKind::Loop:
                line("loop {");
                nested(n.body);
                line("}");
                break;
            case NodeKind::Break: line("break"); break;
            case NodeKind::Continue: line("continue"); break;
            case NodeKind::Discard: line("discard"); break;
            case NodeKind::Return:
                if (n.value == kNoValue) line("return");
                else line("return %", std::to_string(n.value));
                break;
        }
    }

    void nested(const Block& b) {
        ++depth_;
        block(b);
        --depth_;
    }

    std::string symbol(char c, const Inst& in) const {
        auto at = [](const auto& list, uint32_t i) -> std::string {
            return i < list.size() ? list[i].name : "?" + std::to_string(i);
        };
        switch (c) {
            case '$': return "$" + std::to_string(in.sym);
            case 'i': return at(m_.inputs, in.sym);
            case 'o': return at(m_.outputs, in.sym);
            case 'r': return at(m_.resources, in.sym);
            case 's': return at(m_.resources, in.sampler);
            case 'f': return at(m_.functions, in.sym);
            case 'm':
                if (in.sym == kTargetShared) return "shared";
                if (in.sym == kTargetPayload) return "payload";
                if (in.sym == kTargetPrivate) return "private";
                if (isPrivateSlot(in.sym)) return "private_" + std::to_string(privateSlotIndex(in.sym));
                return at(m_.resources, in.sym);
        }
        return {};
    }

    std::string inst(const Inst& in) const {
        std::string s;
        if (in.result != kNoValue) s = "%" + std::to_string(in.result) + " = ";
        if (in.precise) s += "precise ";
        s += opInfo(in.op).name;

        std::vector<std::string> ops;
        size_t arg = 0, lit = 0;
        auto value = [&] {
            ops.push_back(arg < in.args.size() ? "%" + std::to_string(in.args[arg]) : "%?");
            ++arg;
        };
        for (const char* p = opInfo(in.op).signature; *p; ++p) {
            switch (*p) {
                case 'v': value(); break;
                case '*':
                    while (arg < in.args.size()) value();
                    break;
                case 'n':
                    if (lit < in.lits.size()) ops.push_back(std::to_string(in.lits[lit++]));
                    break;
                case 'N':
                    while (lit < in.lits.size()) ops.push_back(std::to_string(in.lits[lit++]));
                    break;
                case 'k':
                    for (uint32_t bits : in.lits) ops.push_back(formatLiteral(in.type.scalar, bits));
                    break;
                case 'I':
                    for (const ImageOperandSpec& spec : kImageOperands) {
                        if (!(in.image & spec.bit)) continue;
                        std::string g = std::string(spec.name) + "(";
                        for (unsigned k = 0; k < spec.count; ++k, ++arg) {
                            if (k) g += ", ";
                            g += arg < in.args.size() ? "%" + std::to_string(in.args[arg]) : "%?";
                        }
                        ops.push_back(g + ")");
                    }
                    break;
                default: ops.push_back(symbol(*p, in)); break;
            }
        }
        for (size_t i = 0; i < ops.size(); ++i) s += (i ? ", " : " ") + ops[i];
        if (in.result != kNoValue) s += " : " + toString(in.type);
        return s;
    }

    const Module& m_;
    std::string out_;
    unsigned depth_ = 0;
};

}  // namespace

std::string formatLiteral(Scalar s, uint32_t bits) {
    switch (s) {
        case Scalar::Bool: return bits ? "true" : "false";
        case Scalar::I32: return std::to_string(int32_t(bits));
        case Scalar::U32: return std::to_string(bits);
        case Scalar::F16:
        case Scalar::F32: {
            float f = s == Scalar::F16 ? halfToFloat(uint16_t(bits)) : std::bit_cast<float>(bits);
            if (!std::isfinite(f)) {  // keep the exact bits (NaN payloads, infinities)
                char buf[16];
                auto r = std::to_chars(buf, buf + sizeof buf, bits, 16);
                return "0x" + std::string(buf, r.ptr);
            }
            char buf[32];
            auto r = std::to_chars(buf, buf + sizeof buf, f);
            std::string t(buf, r.ptr);
            if (t.find_first_of(".e") == std::string::npos) t += ".0";
            return t;
        }
        case Scalar::Void: break;
    }
    return "?";
}

std::string print(const Module& module) { return Printer(module).run(); }

}  // namespace vpir
