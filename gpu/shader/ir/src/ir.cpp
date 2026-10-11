#include "vpir/ir.h"

#include <bit>
#include <cstring>

namespace vpir {

namespace {
constexpr OpInfo kOps[] = {
#define VPIR_INFO(id, name, sig) {name, sig},
    VPIR_OPS(VPIR_INFO)
#undef VPIR_INFO
};

constexpr unsigned bit(Stage s) { return 1u << unsigned(s); }
constexpr unsigned kVtx = bit(Stage::Vertex), kFrag = bit(Stage::Fragment);
constexpr unsigned kGrid = bit(Stage::Compute) | bit(Stage::Object) | bit(Stage::Mesh);
}  // namespace

const OpInfo& opInfo(Op op) { return kOps[unsigned(op)]; }

bool opFromName(std::string_view name, Op& out) {
    for (unsigned i = 0; i < std::size(kOps); ++i)
        if (name == kOps[i].name) {
            out = Op(i);
            return true;
        }
    return false;
}

std::string toString(Type t) {
    static const char* names[] = {"void", "bool", "i32", "u32", "f16", "f32"};
    std::string s = names[unsigned(t.scalar)];
    if (t.width == 1) return s;
    return "vec" + std::to_string(t.width) + "<" + s + ">";
}

const char* toString(Stage s) {
    static const char* n[] = {"vertex", "fragment", "compute", "object", "mesh"};
    return n[unsigned(s)];
}
const char* toString(Builtin b) {
    static const char* n[] = {"none",          "position",          "point_size",
                              "vertex_id",     "instance_id",       "front_facing",
                              "sample_id",     "sample_mask",       "frag_depth",
                              "primitive_id",  "layer",             "global_invocation_id",
                              "local_invocation_id", "workgroup_id", "num_workgroups",
                              "local_invocation_index", "subgroup_invocation", "subgroup_size"};
    return n[unsigned(b)];
}
const char* toString(Interp i) {
    static const char* n[] = {"perspective", "linear", "flat"};
    return n[unsigned(i)];
}
const char* toString(Sampling s) {
    static const char* n[] = {"center", "centroid", "sample"};
    return n[unsigned(s)];
}
const char* toString(ResourceKind k) {
    static const char* n[] = {"buffer", "constant_buffer", "texture", "sampler", "storage_texture"};
    return n[unsigned(k)];
}
const char* toString(TexDim d) {
    static const char* n[] = {"1d", "2d", "2d_array", "3d", "cube", "cube_array"};
    return n[unsigned(d)];
}
const char* toString(Access a) {
    static const char* n[] = {"read", "write", "read_write"};
    return n[unsigned(a)];
}
const char* toString(Topology t) {
    static const char* n[] = {"point", "line", "triangle"};
    return n[unsigned(t)];
}

unsigned coordWidth(TexDim d) {
    switch (d) {
        case TexDim::D1: return 1;
        case TexDim::D2:
        case TexDim::D2Array: return 2;
        default: return 3;
    }
}
bool isArrayed(TexDim d) { return d == TexDim::D2Array || d == TexDim::CubeArray; }
unsigned verticesPerPrimitive(Topology t) { return unsigned(t) + 1; }

BuiltinInfo builtinInfo(Builtin b, bool asInput) {
    const Type u3{Scalar::U32, 3}, f4{Scalar::F32, 4};
    switch (b) {
        case Builtin::None: return {kVoid, asInput, 0};
        case Builtin::Position:
            return asInput ? BuiltinInfo{f4, true, kFrag}
                           : BuiltinInfo{f4, false, kVtx | bit(Stage::Mesh)};
        case Builtin::PointSize: return {kF32, false, kVtx | bit(Stage::Mesh)};
        case Builtin::VertexId:
        case Builtin::InstanceId: return {kU32, true, kVtx};
        case Builtin::FrontFacing: return {kBool, true, kFrag};
        case Builtin::SampleId:
        case Builtin::PrimitiveId: return {kU32, true, kFrag};
        case Builtin::SampleMask: return {kU32, asInput, kFrag};
        case Builtin::FragDepth: return {kF32, false, kFrag};
        case Builtin::Layer:
            return asInput ? BuiltinInfo{kU32, true, kFrag}
                           : BuiltinInfo{kU32, false, kVtx | bit(Stage::Mesh)};
        case Builtin::GlobalInvocationId:
        case Builtin::LocalInvocationId:
        case Builtin::WorkgroupId:
        case Builtin::NumWorkgroups: return {u3, true, kGrid};
        case Builtin::LocalInvocationIndex: return {kU32, true, kGrid};
        case Builtin::SubgroupInvocation:
        case Builtin::SubgroupSize: return {kU32, true, kGrid | kFrag};
    }
    return {kVoid, asInput, 0};
}

const Function* Module::entry() const {
    for (const Function& f : functions)
        if (f.name == "main") return &f;
    return nullptr;
}

uint16_t floatToHalf(float f) {
    uint32_t x = std::bit_cast<uint32_t>(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mag = x & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u)  // inf / nan (keep a quiet payload bit for nan)
        return uint16_t(sign | 0x7C00u | (mag > 0x7F800000u ? 0x200u | ((mag >> 13) & 0x3FFu) : 0));
    if (mag >= 0x477FF000u) return uint16_t(sign | 0x7C00u);  // rounds to >= 65520: overflow
    if (mag < 0x38800000u) {  // result is subnormal or zero: align to 2^-24 units
        if (mag < 0x33000000u) return uint16_t(sign);
        uint32_t m = (mag & 0x7FFFFFu) | 0x800000u;
        unsigned shift = 126 - (mag >> 23);  // value = m * 2^(e-150); unit 2^-24
        uint32_t r = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return uint16_t(sign | r);
    }
    uint32_t r = ((mag - 0x38000000u) >> 13);  // rebias exponent 127 -> 15
    uint32_t rem = mag & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;  // carry may roll into the exponent
    return uint16_t(sign | r);
}

float halfToFloat(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0x1F) bits = sign | 0x7F800000u | (man << 13);
    else if (exp != 0) bits = sign | ((exp + 112) << 23) | (man << 13);
    else if (man == 0) bits = sign;
    else {  // subnormal: normalise the mantissa
        int e = -1;
        do { man <<= 1; ++e; } while (!(man & 0x400u));
        bits = sign | uint32_t(112 - e) << 23 | ((man & 0x3FFu) << 13);
    }
    return std::bit_cast<float>(bits);
}

}  // namespace vpir
