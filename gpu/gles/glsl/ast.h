// Internal: types, symbols and the typed AST of the GLSL ES compiler.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "glsl.h"

namespace vpglsl {

enum class Base : uint8_t { Void, Bool, Int, Uint, Float, Sampler, Struct, Image, AtomicUint };
enum class SDim : uint8_t { D2, D3, Cube, D2Array, External, CubeArray, Buffer };

struct StructDef;

struct Type {
    Base base = Base::Void;
    uint8_t vec = 1;   // components; rows for matrices
    uint8_t cols = 0;  // matrix columns, 0 = not a matrix
    // Samplers and images (images: `sampled` is the texel component type, `ifmt` the SPIR-V ImageFormat
    // of the layout(rgba8...) qualifier, 0 = none)
    SDim dim = SDim::D2;
    Base sampled = Base::Float;  // result component type
    bool shadow = false, ms = false;
    uint8_t ifmt = 0;
    const StructDef* st = nullptr;
    std::vector<int> dims;  // array dimensions, outermost first

    bool isArray() const { return !dims.empty(); }
    bool isMatrix() const { return cols != 0 && dims.empty(); }
    bool isVector() const { return cols == 0 && vec > 1 && dims.empty() && numericOrBool(); }
    bool isScalar() const { return cols == 0 && vec == 1 && dims.empty() && numericOrBool(); }
    bool numericOrBool() const { return base == Base::Bool || base == Base::Int || base == Base::Uint || base == Base::Float; }
    bool isNumeric() const { return base == Base::Int || base == Base::Uint || base == Base::Float; }
    bool isIntegral() const { return base == Base::Int || base == Base::Uint; }
    bool isSampler() const { return base == Base::Sampler && dims.empty(); }
    bool isImage() const { return base == Base::Image && dims.empty(); }
    bool isAtomic() const { return base == Base::AtomicUint && dims.empty(); }
    // Samplers and images: UniformConstant handles passed to functions by pointer.
    bool isHandle() const { return isSampler() || isImage(); }
    bool containsSampler() const;
    // Samplers, images and atomic counters (opaque types: uniform only, no assignment, no blocks).
    bool containsOpaque() const;
    // Element of an array (dims dropped once).
    Type element() const {
        Type t = *this;
        t.dims.erase(t.dims.begin());
        return t;
    }
    Type scalarType() const { return make(base); }
    Type columnType() const { return make(base, vec); }
    int components() const { return (cols ? cols : 1) * vec; }  // non-arrays
    bool operator==(const Type& o) const;
    bool operator!=(const Type& o) const { return !(*this == o); }
    std::string str() const;

    static Type make(Base b, int vec = 1, int cols = 0) {
        Type t;
        t.base = b, t.vec = uint8_t(vec), t.cols = uint8_t(cols);
        return t;
    }
};

// Memory qualifiers (Symbol::mem, Member::mem).
enum MemQual : uint8_t { kReadonly = 1, kWriteonly = 2, kCoherent = 4, kVolatile = 8, kRestrict = 16 };

struct Member {
    std::string name;
    Type type;
    bool rowMajor = false;
    uint8_t mem = 0;  // buffer block members: MemQual
    uint8_t prec = 0;  // declared precision (1 lowp, 2 mediump, 3 highp; explicit or the default in force), 0 = none
};

struct StructDef {
    std::string name;
    std::vector<Member> members;
    int index = 0;  // unique per shader
};

enum class Storage : uint8_t {
    Local, Param, Global, Const, Uniform, BlockMember, BlockInstance, In, Out, Builtin, Shared,
};
enum class ParamDir : uint8_t { In, Out, InOut };
enum class BuiltinVar : uint8_t {
    None, Position, PointSize, VertexID, InstanceID, FragCoord, FrontFacing, PointCoord, FragColor, FragData,
    FragDepth, ViewID, LocalInvocationID, GlobalInvocationID, WorkGroupID, NumWorkGroups, LocalInvocationIndex,
    // Read-only builtins from here on (see Parser::isLValue).
    WorkGroupSize, HelperInvocation, LastFragData, LastFragColor,
};
enum class Interp : uint8_t { Smooth, Flat, Centroid };

struct UniformBlockDecl;

struct Symbol {
    std::string name;
    Type type;
    Storage storage = Storage::Local;
    ParamDir dir = ParamDir::In;
    BuiltinVar builtin = BuiltinVar::None;
    Interp interp = Interp::Smooth;
    int location = -1, binding = -1;
    bool invariant = false;
    bool hasConst = false;     // compile-time constant scalar int/uint/bool/float (folded)
    double constF = 0;
    int64_t constI = 0;
    uint8_t mem = 0;                    // MemQual (images, shared)
    int offset = -1;                    // atomic counters: byte offset within the buffer binding
    bool fbFetch = false;               // fragment `inout` output (GL_EXT_shader_framebuffer_fetch)
    UniformBlockDecl* block = nullptr;  // BlockMember / BlockInstance
    int member = -1;                     // BlockMember: index in the block
    int id = 0;                          // unique per shader (emitter maps)
    bool used = false;                   // referenced by any expression
    uint8_t prec = 0;                    // uniforms: declared precision (Member::prec)
    bool constant = false;               // a `const` whose value is a constant expression (usable in one)
    bool constParam = false;             // a function parameter declared `const`
    // A uniform struct holding samplers: it has no storage of its own, every leaf ("u[1].tex") is a uniform of its own.
    bool split = false;
    std::map<std::string, Symbol*> leaves;  // path relative to the symbol ("[1].tex", ".a") -> leaf uniform
};

struct UniformBlockDecl {
    std::string name, instance;
    std::vector<Member> members;
    std::vector<int> dims;  // instance array
    int binding = -1;
    bool std430 = false;
    uint8_t packing = 0;  // declared layout: 0 none (shared), 1 shared, 2 packed, 3 std140, 4 std430
    bool rowMajor = false;  // the block's default matrix layout
    bool buffer = false;  // shader storage block
    Symbol* instanceSym = nullptr;  // the variable that names the block (instance or hidden)
};

enum class EK : uint8_t {
    IntLit, UintLit, FloatLit, BoolLit, Var, Binary, Unary, PreInc, PreDec, PostInc, PostDec, Assign, Ternary,
    Call, Builtin, Ctor, Index, Field, Swizzle, Comma, Length,
};

struct Function;
struct Expr;
using ExprP = std::unique_ptr<Expr>;

struct Expr {
    EK k;
    Type type;
    int line = 0;
    std::string op;        // Binary/Unary/Assign operator; Builtin: function name
    std::vector<ExprP> a;  // operands
    Symbol* sym = nullptr;
    Function* fn = nullptr;
    double f = 0;
    uint64_t i = 0;
    int field = -1;        // Field: member index
    uint8_t swz[4] = {0, 0, 0, 0}, nswz = 0;
    bool hasConst = false;  // folded scalar (int/uint/bool/float) value
    double constF = 0;
    int64_t constI = 0;
};

enum class SK : uint8_t { Block, Decl, Expr, If, For, While, Do, Return, Break, Continue, Discard, Switch, Case, Default, Empty };

struct Stmt;
using StmtP = std::unique_ptr<Stmt>;
struct Stmt {
    SK k;
    int line = 0;
    ExprP e;                    // Expr / If cond / loop cond / Return value / Switch selector / Case value
    ExprP inc;                  // For
    StmtP init, then, els, body;
    std::vector<StmtP> list;    // Block / Switch body
    std::vector<std::pair<Symbol*, ExprP>> decls;  // Decl
    bool newScope = true;       // Block
};

struct Function {
    std::string name;
    Type ret;
    std::vector<Symbol*> params;
    StmtP body;  // null for a prototype never defined
    int id = 0;
    bool isMain() const { return name == "main"; }
};

struct GlobalDecl {
    Symbol* sym;
    ExprP init;  // Global/Out initialisers (run at entry)
};

class Shader {
public:
    Stage stage = Stage::Vertex;
    int version = 100;
    std::vector<GlobalDecl> globals;  // every global symbol, in declaration order
    std::vector<Function*> functions;
    std::vector<UniformBlockDecl*> blocks;
    std::vector<Symbol*> builtins;  // builtin variables referenced
    int numViews = 0;
    std::array<uint32_t, 3> localSize{1, 1, 1};
    bool hasLocalSize = false;
    bool earlyFragmentTests = false;

    // Arena.
    std::vector<std::unique_ptr<Symbol>> symbols;
    std::vector<std::unique_ptr<Function>> fnStore;
    std::vector<std::unique_ptr<StructDef>> structs;
    std::vector<std::unique_ptr<UniformBlockDecl>> blockStore;
    Symbol* newSymbol() {
        symbols.push_back(std::make_unique<Symbol>());
        symbols.back()->id = int(symbols.size());
        return symbols.back().get();
    }
    Function* main() const {
        for (Function* f : functions)
            if (f->isMain() && f->body) return f;
        return nullptr;
    }
};

// std140 layout (also used for the default uniform block) and, with `s430`, std430 (shader storage
// blocks: arrays and structs are not rounded up to vec4, matrices use their vector's alignment).
// A runtime-sized array (dims[0] == 0) has size 0.
uint32_t std140Align(const Type& t, bool rowMajor = false, bool s430 = false);
uint32_t std140Size(const Type& t, bool rowMajor = false, bool s430 = false);
uint32_t std140ArrayStride(const Type& t, bool rowMajor = false, bool s430 = false);  // stride of t's elements (t is an array)
uint32_t std140MatrixStride(const Type& t, bool rowMajor = false, bool s430 = false);
// Offsets of the members of a struct/block laid out with std140.
std::vector<uint32_t> std140Offsets(const std::vector<Member>& members, uint32_t* size, bool s430 = false, bool rowMajor = false);

uint32_t glTypeEnum(const Type& t);

}  // namespace vpglsl
