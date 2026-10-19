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

bool Type::containsOpaque() const {
    if (base == Base::Sampler || base == Base::Image || base == Base::AtomicUint) return true;
    if (base == Base::Struct && st)
        for (const Member& m : st->members)
            if (m.type.containsOpaque()) return true;
    return false;
}

bool Type::operator==(const Type& o) const {
    if (base != o.base || dims != o.dims) return false;
    switch (base) {
        case Base::Struct: return st == o.st;
        case Base::Sampler: return dim == o.dim && sampled == o.sampled && shadow == o.shadow && ms == o.ms;
        case Base::Image: return dim == o.dim && sampled == o.sampled && ifmt == o.ifmt;
        case Base::AtomicUint: return true;
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
            static const char* d[] = {"2D", "3D", "Cube", "2DArray", "ExternalOES", "CubeArray"};
            s += d[int(dim)];
            if (ms) s += "MS";
            if (shadow) s += "Shadow";
            break;
        }
        case Base::Image: {
            s = sampled == Base::Int ? "iimage" : sampled == Base::Uint ? "uimage" : "image";
            static const char* d[] = {"2D", "3D", "Cube", "2DArray", "External", "CubeArray"};
            s += d[int(dim)];
            if (ifmt) s += "#" + std::to_string(ifmt);  // distinct types for distinct formats
            break;
        }
        case Base::AtomicUint: s = "atomic_uint"; break;
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

// Components of the vector a matrix column (row-major: row) holds.
static uint32_t matrixVecLen(const Type& t, bool rowMajor) { return rowMajor ? t.cols : t.vec; }

uint32_t std140MatrixStride(const Type& t, bool rowMajor, bool s430) {
    if (!s430) return 16;
    return matrixVecLen(t, rowMajor) == 2 ? 8 : 16;
}

uint32_t std140Align(const Type& t, bool rowMajor, bool s430) {
    if (t.isArray()) {
        uint32_t a = std140Align(t.element(), rowMajor, s430);
        return s430 ? a : std::max<uint32_t>(16, a);
    }
    if (t.base == Base::Struct) {
        uint32_t a = s430 ? 4 : 16;
        for (const Member& m : t.st->members) a = std::max(a, std140Align(m.type, m.rowMajor, s430));
        return s430 ? a : roundUp(a, 16);
    }
    if (t.cols) return s430 ? std140MatrixStride(t, rowMajor, true) : 16;  // columns (or rows) are vec4-aligned arrays
    return t.vec == 1 ? 4 : t.vec == 2 ? 8 : 16;
}

uint32_t std140ArrayStride(const Type& t, bool rowMajor, bool s430) {
    Type e = t.element();
    return roundUp(std140Size(e, rowMajor, s430), s430 ? std140Align(e, rowMajor, s430) : 16);
}

uint32_t std140Size(const Type& t, bool rowMajor, bool s430) {
    if (t.isArray()) return t.dims[0] == 0 ? 0 : std140ArrayStride(t, rowMajor, s430) * uint32_t(t.dims[0]);
    if (t.base == Base::Struct) {
        uint32_t size = 0;
        std140Offsets(t.st->members, &size, s430);
        return size;
    }
    if (t.cols) return std140MatrixStride(t, rowMajor, s430) * (rowMajor ? t.vec : t.cols);
    return 4u * t.vec;
}

std::vector<uint32_t> std140Offsets(const std::vector<Member>& members, uint32_t* size, bool s430) {
    std::vector<uint32_t> off;
    uint32_t at = 0, maxAlign = s430 ? 4 : 16;
    for (const Member& m : members) {
        const uint32_t al = std140Align(m.type, m.rowMajor, s430);
        maxAlign = std::max(maxAlign, al);
        at = roundUp(at, al);
        off.push_back(at);
        at += std140Size(m.type, m.rowMajor, s430);
        // std140: a struct or array member is padded to a multiple of 16 (std430 sizes already are multiples of their alignment).
        if (!s430 && (m.type.base == Base::Struct || m.type.isArray())) at = roundUp(at, 16);
    }
    if (size) *size = roundUp(at, s430 ? maxAlign : 16);
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
        case Base::AtomicUint: return 0x92DB;  // GL_UNSIGNED_INT_ATOMIC_COUNTER
        case Base::Image: {
            // GL_IMAGE_2D 0x904D, _3D 0x904E, _CUBE 0x9050, _2D_ARRAY 0x9053, _CUBE_MAP_ARRAY 0x9054;
            // GL_INT_IMAGE_* 0x9058.. and GL_UNSIGNED_INT_IMAGE_* 0x9063.. follow the same order.
            const uint32_t base = t.sampled == Base::Int ? 0x9058 : t.sampled == Base::Uint ? 0x9063 : 0x904D;
            switch (t.dim) {
                case SDim::D3: return base + 1;
                case SDim::Cube: return base + 3;
                case SDim::D2Array: return base + 6;
                case SDim::CubeArray: return base + 7;
                default: return base;
            }
        }
        case Base::Sampler:
            if (t.dim == SDim::External) return 0x8D66;  // GL_SAMPLER_EXTERNAL_OES
            if (t.dim == SDim::CubeArray) return t.shadow ? 0x900D : t.sampled == Base::Int ? 0x900E : t.sampled == Base::Uint ? 0x900F : 0x900C;
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
