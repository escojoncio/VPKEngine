// Built-in functions: signature resolution (GLSL ES 3.20 §8, ES 1.00 §8 and the extensions the
// preprocessor knows) and scalar constant folding.
#include "builtins.h"

#include <cmath>
#include <map>

namespace vpglsl {
namespace Builtins {

namespace {

using T = Type;
T F(int n = 1) { return T::make(Base::Float, n); }
T I(int n = 1) { return T::make(Base::Int, n); }
T U(int n = 1) { return T::make(Base::Uint, n); }
T B(int n = 1) { return T::make(Base::Bool, n); }

bool plain(const T& t) { return !t.isArray() && !t.cols && t.numericOrBool(); }
bool genF(const T& t) { return plain(t) && t.base == Base::Float; }
bool genI(const T& t) { return plain(t) && t.base == Base::Int; }
bool genU(const T& t) { return plain(t) && t.base == Base::Uint; }
bool genB(const T& t) { return plain(t) && t.base == Base::Bool; }
bool vecOf(const T& t, Base b) { return plain(t) && t.base == b && t.vec >= 2; }
bool mat(const T& t) { return !t.isArray() && t.cols && t.base == Base::Float; }

enum Group {
    F1,        // genF f(genF)
    FI1,       // genF/genI (ES3) f(x): abs, sign
    F2,        // genF f(genF, genF)
    F2S,       // genF f(genF, genF | float)  (mod)
    MINMAX,    // genF/I/U f(x, same | scalar)
    CLAMP,
    MIX,
    STEP,
    SMOOTH,
    LEN,       // float f(genF)
    DIST,      // float f(genF, genF)
    DOT,
    CROSS,
    FACEFWD,
    REFLECT,
    REFRACT,
    NORMALIZE,
    MATCOMP,
    OUTER,
    TRANSPOSE,
    DETERMINANT,
    INVERSE,
    CMP,       // bvec f(vec, vec) numeric
    EQ,        // bvec f(vec, vec) any
    ANYALL,
    NOT,
    ISNAN,
    FBITS2I, FBITS2U, IBITS2F, UBITS2F,
    PACK2, PACK4, UNPACK2, UNPACK4, PACKHALF, UNPACKHALF,
    MODF,
    FRAC,      // fract-like F1 alias
    DERIV,     // F1 in fragment (ES1: needs OES_standard_derivatives)
    BITCOUNT,  // genI f(genI|genU)
    FINDLSB,
    BITREV,
    BFEXTRACT,
    BFINSERT,
    FMA,
    ATOMIC,         // atomicAdd... on a buffer/shared int or uint
    ATOMICCOUNTER,  // atomicCounter / atomicCounterIncrement / atomicCounterDecrement
    LDEXP,          // genF ldexp(genF, genI)
    FREXP,          // genF frexp(genF, out genI)
    ADDCARRY,       // genU uaddCarry / usubBorrow (x, y, out carry)
    MULEXT,         // umulExtended / imulExtended (x, y, out msb, out lsb)
    BARRIER,        // barrier(), group/shared memory barriers (compute only)
    MEMBARRIER,     // memoryBarrier(), memoryBarrierBuffer/Image/AtomicCounter (any stage)
};

const std::map<std::string, Group>& groups() {
    static const std::map<std::string, Group> g = {
        {"radians", F1}, {"degrees", F1}, {"sin", F1}, {"cos", F1}, {"tan", F1}, {"asin", F1}, {"acos", F1},
        {"sinh", F1}, {"cosh", F1}, {"tanh", F1}, {"asinh", F1}, {"acosh", F1}, {"atanh", F1}, {"exp", F1},
        {"log", F1}, {"exp2", F1}, {"log2", F1}, {"sqrt", F1}, {"inversesqrt", F1}, {"floor", F1}, {"ceil", F1},
        {"fract", F1}, {"trunc", F1}, {"round", F1}, {"roundEven", F1}, {"abs", FI1}, {"sign", FI1}, {"pow", F2},
        {"atan", F1}, {"mod", F2S}, {"min", MINMAX}, {"max", MINMAX}, {"clamp", CLAMP}, {"mix", MIX}, {"step", STEP},
        {"smoothstep", SMOOTH}, {"length", LEN}, {"distance", DIST}, {"dot", DOT}, {"cross", CROSS},
        {"faceforward", FACEFWD}, {"reflect", REFLECT}, {"refract", REFRACT}, {"normalize", NORMALIZE},
        {"matrixCompMult", MATCOMP}, {"outerProduct", OUTER}, {"transpose", TRANSPOSE}, {"determinant", DETERMINANT},
        {"inverse", INVERSE}, {"lessThan", CMP}, {"lessThanEqual", CMP}, {"greaterThan", CMP},
        {"greaterThanEqual", CMP}, {"equal", EQ}, {"notEqual", EQ}, {"any", ANYALL}, {"all", ANYALL}, {"not", NOT},
        {"isnan", ISNAN}, {"isinf", ISNAN}, {"floatBitsToInt", FBITS2I}, {"floatBitsToUint", FBITS2U},
        {"intBitsToFloat", IBITS2F}, {"uintBitsToFloat", UBITS2F}, {"packSnorm2x16", PACK2}, {"packUnorm2x16", PACK2},
        {"packUnorm4x8", PACK4}, {"packSnorm4x8", PACK4}, {"unpackSnorm2x16", UNPACK2}, {"unpackUnorm2x16", UNPACK2},
        {"unpackUnorm4x8", UNPACK4}, {"unpackSnorm4x8", UNPACK4}, {"packHalf2x16", PACKHALF},
        {"unpackHalf2x16", UNPACKHALF}, {"modf", MODF}, {"dFdx", DERIV}, {"dFdy", DERIV}, {"fwidth", DERIV},
        {"bitCount", BITCOUNT}, {"findLSB", FINDLSB}, {"findMSB", FINDLSB}, {"bitfieldReverse", BITREV},
        {"bitfieldExtract", BFEXTRACT}, {"bitfieldInsert", BFINSERT}, {"fma", FMA},
        {"ldexp", LDEXP}, {"frexp", FREXP}, {"uaddCarry", ADDCARRY}, {"usubBorrow", ADDCARRY},
        {"umulExtended", MULEXT}, {"imulExtended", MULEXT},
        {"atomicAdd", ATOMIC}, {"atomicMin", ATOMIC}, {"atomicMax", ATOMIC}, {"atomicAnd", ATOMIC},
        {"atomicOr", ATOMIC}, {"atomicXor", ATOMIC}, {"atomicExchange", ATOMIC}, {"atomicCompSwap", ATOMIC},
        {"atomicCounter", ATOMICCOUNTER}, {"atomicCounterIncrement", ATOMICCOUNTER},
        {"atomicCounterDecrement", ATOMICCOUNTER},
        {"barrier", BARRIER}, {"groupMemoryBarrier", BARRIER}, {"memoryBarrierShared", BARRIER},
        {"memoryBarrier", MEMBARRIER}, {"memoryBarrierBuffer", MEMBARRIER}, {"memoryBarrierImage", MEMBARRIER},
        {"memoryBarrierAtomicCounter", MEMBARRIER},
    };
    return g;
}

// ES 3.00+ only names.
bool es3Only(const std::string& n) {
    static const std::set<std::string> s = {"sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "trunc", "round",
                                            "roundEven", "transpose", "determinant", "inverse", "outerProduct",
                                            "isnan", "isinf", "floatBitsToInt", "floatBitsToUint", "intBitsToFloat",
                                            "uintBitsToFloat", "packSnorm2x16", "packUnorm2x16", "unpackSnorm2x16",
                                            "unpackUnorm2x16", "packHalf2x16", "unpackHalf2x16", "modf"};
    return s.count(n) != 0;
}
bool es31Only(const std::string& n) {
    static const std::set<std::string> s = {"packUnorm4x8", "packSnorm4x8", "unpackUnorm4x8", "unpackSnorm4x8",
                                            "bitCount", "findLSB", "findMSB", "bitfieldReverse", "bitfieldExtract",
                                            "bitfieldInsert", "fma", "ldexp", "frexp", "uaddCarry", "usubBorrow",
                                            "umulExtended", "imulExtended", "atomicAdd", "atomicMin", "atomicMax", "atomicAnd",
                                            "atomicOr", "atomicXor", "atomicExchange", "atomicCompSwap", "atomicCounter",
                                            "atomicCounterIncrement", "atomicCounterDecrement", "barrier",
                                            "groupMemoryBarrier", "memoryBarrierShared", "memoryBarrier",
                                            "memoryBarrierBuffer", "memoryBarrierImage", "memoryBarrierAtomicCounter"};
    return s.count(n) != 0;
}

std::optional<T> generic(Group g, const std::string& name, const std::vector<T>& a, const BuiltinCtx& c) {
    const bool es3 = c.version >= 300;
    auto n = a.size();
    switch (g) {
        case F1:
            if (name == "atan" && n == 2 && genF(a[0]) && a[0] == a[1]) return a[0];
            if (n == 1 && genF(a[0])) return a[0];
            break;
        case FI1:
            if (n == 1 && (genF(a[0]) || (es3 && genI(a[0])))) return a[0];
            break;
        case F2:
            if (n == 2 && genF(a[0]) && a[0] == a[1]) return a[0];
            break;
        case F2S:
            if (n == 2 && genF(a[0]) && (a[1] == a[0] || a[1] == F())) return a[0];
            break;
        case MINMAX:
            if (n == 2 && (genF(a[0]) || (es3 && (genI(a[0]) || genU(a[0])))) &&
                (a[1] == a[0] || a[1] == a[0].scalarType()))
                return a[0];
            break;
        case CLAMP:
            if (n == 3 && (genF(a[0]) || (es3 && (genI(a[0]) || genU(a[0])))) &&
                ((a[1] == a[0] && a[2] == a[0]) || (a[1] == a[0].scalarType() && a[2] == a[0].scalarType())))
                return a[0];
            break;
        case MIX:
            if (n == 3 && genF(a[0]) && a[1] == a[0] && (a[2] == a[0] || a[2] == F())) return a[0];
            if (n == 3 && es3 && plain(a[0]) && a[1] == a[0] && genB(a[2]) && a[2].vec == a[0].vec) return a[0];
            break;
        case STEP:
            if (n == 2 && genF(a[1]) && (a[0] == a[1] || a[0] == F())) return a[1];
            break;
        case SMOOTH:
            if (n == 3 && genF(a[2]) && ((a[0] == a[2] && a[1] == a[2]) || (a[0] == F() && a[1] == F()))) return a[2];
            break;
        case LEN:
            if (n == 1 && genF(a[0])) return F();
            break;
        case DIST:
        case DOT:
            if (n == 2 && genF(a[0]) && a[0] == a[1]) return F();
            break;
        case CROSS:
            if (n == 2 && a[0] == F(3) && a[1] == F(3)) return F(3);
            break;
        case FACEFWD:
            if (n == 3 && genF(a[0]) && a[1] == a[0] && a[2] == a[0]) return a[0];
            break;
        case REFLECT:
            if (n == 2 && genF(a[0]) && a[1] == a[0]) return a[0];
            break;
        case REFRACT:
            if (n == 3 && genF(a[0]) && a[1] == a[0] && a[2] == F()) return a[0];
            break;
        case NORMALIZE:
            if (n == 1 && genF(a[0])) return a[0];
            break;
        case MATCOMP:
            if (n == 2 && mat(a[0]) && a[0] == a[1]) return a[0];
            break;
        case OUTER:
            if (n == 2 && vecOf(a[0], Base::Float) && vecOf(a[1], Base::Float)) return T::make(Base::Float, a[0].vec, a[1].vec);
            break;
        case TRANSPOSE:
            if (n == 1 && mat(a[0])) return T::make(Base::Float, a[0].cols, a[0].vec);
            break;
        case DETERMINANT:
            if (n == 1 && mat(a[0]) && a[0].cols == a[0].vec) return F();
            break;
        case INVERSE:
            if (n == 1 && mat(a[0]) && a[0].cols == a[0].vec) return a[0];
            break;
        case CMP:
            if (n == 2 && plain(a[0]) && a[0].vec >= 2 && a[0].base != Base::Bool && a[0] == a[1] &&
                (a[0].base != Base::Uint || es3))
                return B(a[0].vec);
            break;
        case EQ:
            if (n == 2 && plain(a[0]) && a[0].vec >= 2 && a[0] == a[1]) return B(a[0].vec);
            break;
        case ANYALL:
            if (n == 1 && vecOf(a[0], Base::Bool)) return B();
            break;
        case NOT:
            if (n == 1 && vecOf(a[0], Base::Bool)) return a[0];
            break;
        case ISNAN:
            if (n == 1 && genF(a[0])) return B(a[0].vec);
            break;
        case FBITS2I:
            if (n == 1 && genF(a[0])) return I(a[0].vec);
            break;
        case FBITS2U:
            if (n == 1 && genF(a[0])) return U(a[0].vec);
            break;
        case IBITS2F:
            if (n == 1 && genI(a[0])) return F(a[0].vec);
            break;
        case UBITS2F:
            if (n == 1 && genU(a[0])) return F(a[0].vec);
            break;
        case PACK2:
        case PACKHALF:
            if (n == 1 && a[0] == F(2)) return U();
            break;
        case PACK4:
            if (n == 1 && a[0] == F(4)) return U();
            break;
        case UNPACK2:
        case UNPACKHALF:
            if (n == 1 && a[0] == U()) return F(2);
            break;
        case UNPACK4:
            if (n == 1 && a[0] == U()) return F(4);
            break;
        case MODF:
            if (n == 2 && genF(a[0]) && a[1] == a[0]) return a[0];
            break;
        case FRAC: break;
        case DERIV:
            if (c.stage != Stage::Fragment) return std::nullopt;
            if (!es3 && !c.ext("GL_OES_standard_derivatives")) return std::nullopt;
            if (n == 1 && genF(a[0])) return a[0];
            break;
        case BITCOUNT:
        case FINDLSB:
            if (n == 1 && (genI(a[0]) || genU(a[0]))) return I(a[0].vec);
            break;
        case BITREV:
            if (n == 1 && (genI(a[0]) || genU(a[0]))) return a[0];
            break;
        case BFEXTRACT:
            if (n == 3 && (genI(a[0]) || genU(a[0])) && a[1] == I() && a[2] == I()) return a[0];
            break;
        case BFINSERT:
            if (n == 4 && (genI(a[0]) || genU(a[0])) && a[1] == a[0] && a[2] == I() && a[3] == I()) return a[0];
            break;
        case FMA:
            if (n == 3 && genF(a[0]) && a[1] == a[0] && a[2] == a[0]) return a[0];
            break;
        case LDEXP:
            if (n == 2 && genF(a[0]) && genI(a[1]) && a[1].vec == a[0].vec) return a[0];
            break;
        case FREXP:
            if (n == 2 && genF(a[0]) && a[1] == I(a[0].vec)) return a[0];
            break;
        case ADDCARRY:
            if (n == 3 && genU(a[0]) && a[1] == a[0] && a[2] == a[0]) return a[0];
            break;
        case MULEXT:
            if (n == 4 && (name == "umulExtended" ? genU(a[0]) : genI(a[0])) && a[1] == a[0] && a[2] == a[0] && a[3] == a[0])
                return T::make(Base::Void);
            break;
        case ATOMIC:
            if (name == "atomicCompSwap") {
                if (n == 3 && (a[0] == I() || a[0] == U()) && a[1] == a[0] && a[2] == a[0]) return a[0];
            } else if (n == 2 && (a[0] == I() || a[0] == U()) && a[1] == a[0]) {
                return a[0];
            }
            break;
        case ATOMICCOUNTER:
            if (n == 1 && a[0].isAtomic()) return U();
            break;
        case BARRIER:
            if (n == 0 && c.stage == Stage::Compute) return T::make(Base::Void);
            break;
        case MEMBARRIER:
            if (n == 0) return T::make(Base::Void);
            break;
    }
    return std::nullopt;
}

bool isSampler(const T& t) { return t.isSampler(); }

// Coordinate size (without projection / shadow reference) of a sampler.
int baseCoord(const T& s) {
    switch (s.dim) {
        case SDim::D2: case SDim::External: return 2;
        case SDim::D3: case SDim::Cube: case SDim::D2Array: return 3;
        case SDim::CubeArray: return 4;
        case SDim::Buffer: return 1;
    }
    return 2;
}

T sampleResult(const T& s) {
    if (s.shadow) return F();
    return T::make(s.sampled, 4);
}

}  // namespace

int coordSize(const T& s) {
    int c = baseCoord(s);
    if (s.shadow) c += 1;
    return std::min(c, 4);
}

bool isAtomicMemoryFunction(const std::string& n) {
    return n == "atomicAdd" || n == "atomicMin" || n == "atomicMax" || n == "atomicAnd" || n == "atomicOr" ||
           n == "atomicXor" || n == "atomicExchange" || n == "atomicCompSwap";
}

bool imageCall(const std::string& n, ImageCall& out) {
    static const std::map<std::string, ImageCall> k = {
        {"imageLoad", {ImageCall::Load}},
        {"imageStore", {ImageCall::Store}},
        {"imageSize", {ImageCall::Size}},
        {"imageAtomicAdd", {ImageCall::Atomic}},
        {"imageAtomicMin", {ImageCall::Atomic}},
        {"imageAtomicMax", {ImageCall::Atomic}},
        {"imageAtomicAnd", {ImageCall::Atomic}},
        {"imageAtomicOr", {ImageCall::Atomic}},
        {"imageAtomicXor", {ImageCall::Atomic}},
        {"imageAtomicExchange", {ImageCall::Atomic}},
        {"imageAtomicCompSwap", {ImageCall::Atomic}},
    };
    auto it = k.find(n);
    if (it == k.end()) return false;
    out = it->second;
    return true;
}

// imageLoad / imageStore / imageSize / imageAtomic*.
static std::optional<T> imageSig(const std::string& name, const ImageCall& ic, const std::vector<T>& a, const BuiltinCtx& c) {
    if (c.version < 310 || a.empty() || !a[0].isImage()) return std::nullopt;
    const T& im = a[0];
    const int cd = im.dim == SDim::Buffer ? 1 : im.dim == SDim::D2 ? 2 : 3;  // image coordinates: int (buffer), ivec2 (2D) or ivec3 (3D, cube, arrays)
    switch (ic.kind) {
        case ImageCall::Size:
            if (a.size() != 1) return std::nullopt;
            return I(im.dim == SDim::Buffer ? 1 : im.dim == SDim::D2 || im.dim == SDim::Cube ? 2 : 3);
        case ImageCall::Load:
            if (a.size() == 2 && a[1] == I(cd)) return T::make(im.sampled, 4);
            return std::nullopt;
        case ImageCall::Store:
            if (a.size() == 3 && a[1] == I(cd) && a[2] == T::make(im.sampled, 4)) return T::make(Base::Void);
            return std::nullopt;
        case ImageCall::Atomic: {
            const T el = T::make(im.sampled);
            // GLSL also has imageAtomicExchange on r32f images; VPIR/Metal texture atomics are 32-bit integer only.
            if (im.sampled == Base::Float) return std::nullopt;
            const size_t n = name == "imageAtomicCompSwap" ? 4 : 3;
            if (a.size() != n || a[1] != I(cd)) return std::nullopt;
            for (size_t i = 2; i < n; ++i)
                if (a[i] != el) return std::nullopt;
            return el;
        }
    }
    return std::nullopt;
}

bool textureCall(const std::string& n, TexCall& t) {
    static const std::map<std::string, TexCall> k = [] {
        std::map<std::string, TexCall> m;
        auto add = [&](const char* name, std::initializer_list<int> f, bool legacy = false) {
            TexCall c;
            for (int x : f) {
                switch (x) {
                    case 0: c.proj = true; break;
                    case 1: c.lod = true; break;
                    case 2: c.grad = true; break;
                    case 3: c.offset = true; break;
                    case 4: c.fetch = true; break;
                    case 5: c.gather = true; break;
                    case 6: c.size = true; break;
                }
            }
            c.legacy = legacy;
            m[name] = c;
        };
        add("texture", {});
        add("textureProj", {0});
        add("textureLod", {1});
        add("textureOffset", {3});
        add("textureProjOffset", {0, 3});
        add("textureLodOffset", {1, 3});
        add("textureProjLod", {0, 1});
        add("textureProjLodOffset", {0, 1, 3});
        add("textureGrad", {2});
        add("textureGradOffset", {2, 3});
        add("textureProjGrad", {0, 2});
        add("textureProjGradOffset", {0, 2, 3});
        add("texelFetch", {4});
        add("texelFetchOffset", {4, 3});
        add("textureGather", {5});
        add("textureGatherOffset", {5, 3});
        add("textureSize", {6});
        // ES 1.00 and extensions.
        add("texture2D", {}, true);
        add("texture2DProj", {0}, true);
        add("texture2DLod", {1}, true);
        add("texture2DProjLod", {0, 1}, true);
        add("textureCube", {}, true);
        add("textureCubeLod", {1}, true);
        add("texture2DLodEXT", {1}, true);
        add("texture2DProjLodEXT", {0, 1}, true);
        add("textureCubeLodEXT", {1}, true);
        add("texture2DGradEXT", {2}, true);
        add("texture2DProjGradEXT", {0, 2}, true);
        add("textureCubeGradEXT", {2}, true);
        add("shadow2DEXT", {}, true);
        add("shadow2DProjEXT", {0}, true);
        add("texture3D", {}, true);
        add("texture3DProj", {0}, true);
        add("texture3DLod", {1}, true);
        return m;
    }();
    auto it = k.find(n);
    if (it == k.end()) return false;
    t = it->second;
    return true;
}

static std::optional<T> textureSig(const std::string& name, const TexCall& tc, const std::vector<T>& a,
                                   const BuiltinCtx& c, std::string& why) {
    const bool es3 = c.version >= 300;
    if (a.empty() || !isSampler(a[0])) return std::nullopt;
    const T& s = a[0];
    if (tc.legacy) {
        if (es3) return std::nullopt;  // removed in ES 3.00
        const bool cube = name.find("Cube") != std::string::npos, shadow = name.rfind("shadow", 0) == 0,
                   d3 = name.find("3D") != std::string::npos;
        if (cube != (s.dim == SDim::Cube) || shadow != s.shadow || d3 != (s.dim == SDim::D3)) return std::nullopt;
        if (!cube && !d3 && s.dim != SDim::D2 && s.dim != SDim::External) return std::nullopt;
        if (name.find("EXT") != std::string::npos && name.rfind("shadow", 0) != 0 && !c.ext("GL_EXT_shader_texture_lod") &&
            !c.lenientAllows("extension_not_enabled")) {
            why = "needs GL_EXT_shader_texture_lod";
            return std::nullopt;
        }
        if (shadow && !c.ext("GL_EXT_shadow_samplers") && !c.lenientAllows("extension_not_enabled")) {
            why = "needs GL_EXT_shadow_samplers";
            return std::nullopt;
        }
        if (tc.lod && name.find("EXT") == std::string::npos && c.stage != Stage::Vertex) {
            why = "only available in the vertex shader";
            return std::nullopt;
        }
    } else if (!es3) {
        return std::nullopt;
    }
    if (s.ms && !tc.fetch && !tc.size) return std::nullopt;
    // Not in the GLSL ES 3.00 / 3.10 tables (3.20 and the shadow cube / array extensions add them).
    if (!tc.legacy && !tc.gather && c.version < 320 && s.shadow && !tc.size) {
        const bool arr = s.dim == SDim::D2Array, cube = s.dim == SDim::Cube;
        if (((arr || cube) && tc.lod) || (arr && tc.offset) || (arr && !tc.lod && !tc.grad && a.size() > 2)) {
            if (!c.lenientAllows("shadow_overload")) return std::nullopt;
        }
    }
    // samplerBuffer: only texelFetch(sampler, int) and textureSize(sampler) (no lod, no filtering).
    if (s.dim == SDim::Buffer) {
        if (tc.size) return a.size() == 1 ? std::optional<T>(I()) : std::nullopt;
        if (tc.fetch && !tc.offset && a.size() == 2 && a[1] == I()) return sampleResult(s);
        return std::nullopt;
    }
    size_t i = 1;
    auto need = [&](const T& t) { return i < a.size() && a[i] == t ? (++i, true) : false; };
    if (tc.size) {
        const int d = s.dim == SDim::D3 || s.dim == SDim::D2Array || s.dim == SDim::CubeArray ? 3 : 2;
        if (s.ms) return a.size() == 1 ? std::optional<T>(I(2)) : std::nullopt;
        if (!need(I()) || i != a.size()) return std::nullopt;
        return I(d);
    }
    if (tc.fetch) {
        if (s.dim == SDim::Cube || s.dim == SDim::CubeArray || s.shadow) return std::nullopt;
        const int d = s.dim == SDim::D3 || s.dim == SDim::D2Array ? 3 : 2;
        if (!need(I(d))) return std::nullopt;
        if (!need(I())) return std::nullopt;  // lod or sample
        if (tc.offset && !need(I(s.dim == SDim::D3 ? 3 : 2))) return std::nullopt;
        return i == a.size() ? std::optional<T>(sampleResult(s)) : std::nullopt;
    }
    if (tc.gather) {
        if (c.version < 310 || s.dim == SDim::D3) return std::nullopt;
        if (!need(F(baseCoord(s)))) return std::nullopt;
        if (s.shadow) {
            if (!need(F())) return std::nullopt;
        }
        if (tc.offset && !need(I(2))) return std::nullopt;
        if (!s.shadow && i < a.size() && a[i] == I()) ++i;  // component
        if (i != a.size()) return std::nullopt;
        return T::make(s.shadow ? Base::Float : s.sampled, 4);
    }
    // Coordinate.
    int cs = baseCoord(s);
    if (tc.proj) {
        if (s.dim == SDim::Cube || s.dim == SDim::D2Array || s.dim == SDim::CubeArray) return std::nullopt;
        if (s.shadow) {
            if (!need(F(4))) return std::nullopt;
        } else if (!(need(F(cs + 1)) || (cs == 2 && need(F(4))))) return std::nullopt;
    } else {
        if (s.dim == SDim::CubeArray && s.shadow) {
            // samplerCubeArrayShadow: vec4 coordinate and a separate reference value.
            if (!need(F(4)) || !need(F())) return std::nullopt;
        } else {
            if (!need(F(s.shadow ? cs + 1 : cs))) return std::nullopt;
        }
    }
    if (tc.lod && !need(F())) return std::nullopt;
    if (tc.grad) {
        int g = s.dim == SDim::Cube || s.dim == SDim::CubeArray || s.dim == SDim::D3 ? 3 : 2;
        if (!need(F(g)) || !need(F(g))) return std::nullopt;
    }
    if (tc.offset) {
        int o = s.dim == SDim::D3 ? 3 : 2;
        if (s.dim == SDim::Cube || s.dim == SDim::CubeArray) return std::nullopt;
        if (!need(I(o))) return std::nullopt;
    }
    // Optional bias (implicit-lod forms, fragment only).
    if (!tc.lod && !tc.grad && i < a.size() && a[i] == F() && c.stage == Stage::Fragment) ++i;
    if (i != a.size()) return std::nullopt;
    if ((!tc.lod && !tc.grad) && c.stage != Stage::Fragment && false) return std::nullopt;
    return sampleResult(s);
}

bool isBuiltinName(const std::string& n) {
    TexCall t;
    ImageCall ic;
    return groups().count(n) || textureCall(n, t) || imageCall(n, ic);
}

std::vector<int> outArguments(const std::string& n) {
    if (n == "modf" || n == "frexp") return {1};
    if (n == "uaddCarry" || n == "usubBorrow") return {2};
    if (n == "umulExtended" || n == "imulExtended") return {2, 3};
    if (isAtomicMemoryFunction(n)) return {0};
    return {};
}

std::optional<T> resolve(const std::string& name, const std::vector<T>& a, const BuiltinCtx& c, std::string& why) {
    TexCall tc;
    if (textureCall(name, tc)) {
        auto r = textureSig(name, tc, a, c, why);
        if (!r && why.empty()) why = "no matching overloaded function found";
        return r;
    }
    ImageCall ic;
    if (imageCall(name, ic)) {
        auto r = imageSig(name, ic, a, c);
        if (!r) why = "no matching overloaded function found";
        return r;
    }
    auto g = groups().find(name);
    if (g == groups().end()) return std::nullopt;
    if (c.version < 300 && es3Only(name)) return std::nullopt;
    if (c.version < 310 && es31Only(name)) return std::nullopt;
    auto r = generic(g->second, name, a, c);
    if (!r) why = "no matching overloaded function found";
    return r;
}

void fold(Expr& e, const std::vector<ExprP>& args) {
    if (!e.type.isScalar()) return;
    for (auto& a : args)
        if (!a->hasConst || !a->type.isScalar()) return;
    const std::string& n = e.op;
    auto f = [&](size_t i) { return args[i]->constF; };
    auto x = [&](size_t i) { return args[i]->constI; };
    const bool isF = e.type.base == Base::Float;
    auto setF = [&](double v) { e.hasConst = true, e.constF = double(float(v)), e.constI = int64_t(v); };
    auto setI = [&](int64_t v) { e.hasConst = true, e.constI = v, e.constF = double(v); };
    if (n == "abs") isF ? setF(std::fabs(f(0))) : setI(x(0) < 0 ? -x(0) : x(0));
    else if (n == "min") isF ? setF(std::min(f(0), f(1))) : setI(std::min(x(0), x(1)));
    else if (n == "max") isF ? setF(std::max(f(0), f(1))) : setI(std::max(x(0), x(1)));
    else if (n == "clamp") isF ? setF(std::min(std::max(f(0), f(1)), f(2))) : setI(std::min(std::max(x(0), x(1)), x(2)));
    else if (isF && n == "sqrt") setF(std::sqrt(f(0)));
    else if (isF && n == "floor") setF(std::floor(f(0)));
    else if (isF && n == "ceil") setF(std::ceil(f(0)));
    else if (isF && n == "pow") setF(std::pow(f(0), f(1)));
    else if (isF && n == "exp2") setF(std::exp2(f(0)));
    else if (isF && n == "log2") setF(std::log2(f(0)));
}

}  // namespace Builtins
}  // namespace vpglsl
