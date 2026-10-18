// Types: equality, names, std140 layout, GL type enums.
#include <algorithm>

#include "ast.h"

namespace vpglsl {

bool Type::containsSampler() const {
    if (base == Base::Sampler) return true;
    if (base == Base::Struct && st)
        for (const Member& m : st->members)
            if (m.type.containsSampler()) return true;
    return false;
}

bool Type::operator==(const Type& o) const {
    if (base != o.base || dims != o.dims) return false;
    switch (base) {
        case Base::Struct: return st == o.st;
        case Base::Sampler: return dim == o.dim && sampled == o.sampled && shadow == o.shadow && ms == o.ms;
        default: return vec == o.vec && cols == o.cols;
    }
}

std::string Type::str() const {
    std::string s;
    switch (base) {
        case Base::Void: s = "void"; break;
        case Base::Struct: s = st ? st->name : "struct"; break;
        case Base::Sampler: {
            s = sampled == Base::Int ? "isampler" : sampled == Base::Uint ? "usampler" : "sampler";
            static const char* d[] = {"2D", "3D", "Cube", "2DArray", "ExternalOES"};
            s += d[int(dim)];
            if (ms) s += "MS";
            if (shadow) s += "Shadow";
            break;
        }
        default: {
            const char* sc = base == Base::Bool ? "bool" : base == Base::Int ? "int" : base == Base::Uint ? "uint" : "float";
            const char* vp = base == Base::Bool ? "bvec" : base == Base::Int ? "ivec" : base == Base::Uint ? "uvec" : "vec";
            if (cols) s = cols == vec ? "mat" + std::to_string(cols) : "mat" + std::to_string(cols) + "x" + std::to_string(vec);
            else if (vec > 1) s = vp + std::to_string(vec);
            else s = sc;
        }
    }
    for (int d : dims) s += "[" + std::to_string(d) + "]";
    return s;
}

static uint32_t roundUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

uint32_t std140MatrixStride(const Type&) { return 16; }

uint32_t std140Align(const Type& t, bool rowMajor) {
    if (t.isArray()) return std::max<uint32_t>(16, std140Align(t.element(), rowMajor));
    if (t.base == Base::Struct) {
        uint32_t a = 16;
        for (const Member& m : t.st->members) a = std::max(a, std140Align(m.type, m.rowMajor));
        return roundUp(a, 16);
    }
    if (t.cols) return 16;  // columns (or rows) are vec4-aligned arrays
    return t.vec == 1 ? 4 : t.vec == 2 ? 8 : 16;
}

uint32_t std140ArrayStride(const Type& t, bool rowMajor) {
    Type e = t.element();
    return roundUp(std140Size(e, rowMajor), 16);
}

uint32_t std140Size(const Type& t, bool rowMajor) {
    if (t.isArray()) return std140ArrayStride(t, rowMajor) * uint32_t(t.dims[0]);
    if (t.base == Base::Struct) {
        uint32_t size = 0;
        std140Offsets(t.st->members, &size);
        return size;
    }
    if (t.cols) return 16 * (rowMajor ? t.vec : t.cols);
    return 4u * t.vec;
}

std::vector<uint32_t> std140Offsets(const std::vector<Member>& members, uint32_t* size) {
    std::vector<uint32_t> off;
    uint32_t at = 0;
    for (const Member& m : members) {
        at = roundUp(at, std140Align(m.type, m.rowMajor));
        off.push_back(at);
        at += std140Size(m.type, m.rowMajor);
        // A struct member is padded to a multiple of its alignment.
        if (m.type.base == Base::Struct || m.type.isArray()) at = roundUp(at, 16);
    }
    if (size) *size = roundUp(at, 16);
    return off;
}

uint32_t glTypeEnum(const Type& t) {
    // GL_FLOAT 0x1406, GL_INT 0x1404, GL_UNSIGNED_INT 0x1405, GL_BOOL 0x8B56.
    switch (t.base) {
        case Base::Float:
            if (t.cols) {
                static const uint32_t m[3][3] = {{0x8B5A, 0x8B65, 0x8B66}, {0x8B67, 0x8B5B, 0x8B68}, {0x8B69, 0x8B6A, 0x8B5C}};
                return m[t.cols - 2][t.vec - 2];  // matCxR
            }
            return t.vec == 1 ? 0x1406 : 0x8B50 + (t.vec - 2);
        case Base::Int: return t.vec == 1 ? 0x1404 : 0x8B53 + (t.vec - 2);
        case Base::Uint: return t.vec == 1 ? 0x1405 : 0x8DC6 + (t.vec - 2);
        case Base::Bool: return t.vec == 1 ? 0x8B56 : 0x8B57 + (t.vec - 2);
        case Base::Sampler:
            if (t.dim == SDim::External) return 0x8D66;  // GL_SAMPLER_EXTERNAL_OES
            if (t.sampled == Base::Int) {
                if (t.ms) return 0x9109;
                return t.dim == SDim::D2 ? 0x8DCA : t.dim == SDim::D3 ? 0x8DCB : t.dim == SDim::Cube ? 0x8DCC : 0x8DCF;
            }
            if (t.sampled == Base::Uint) {
                if (t.ms) return 0x910A;
                return t.dim == SDim::D2 ? 0x8DD2 : t.dim == SDim::D3 ? 0x8DD3 : t.dim == SDim::Cube ? 0x8DD4 : 0x8DD7;
            }
            if (t.ms) return 0x9108;
            if (t.shadow) return t.dim == SDim::D2 ? 0x8B62 : t.dim == SDim::Cube ? 0x8DC5 : 0x8DC4;
            return t.dim == SDim::D2 ? 0x8B5E : t.dim == SDim::D3 ? 0x8B5F : t.dim == SDim::Cube ? 0x8B60 : 0x8DC1;
        default: return 0;
    }
}

}  // namespace vpglsl
