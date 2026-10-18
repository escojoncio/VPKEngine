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

enum class Base : uint8_t { Void, Bool, Int, Uint, Float, Sampler, Struct };
enum class SDim : uint8_t { D2, D3, Cube, D2Array, External };

struct StructDef;

struct Type {
    Base base = Base::Void;
    uint8_t vec = 1;   // components; rows for matrices
    uint8_t cols = 0;  // matrix columns, 0 = not a matrix
    // Samplers
    SDim dim = SDim::D2;
    Base sampled = Base::Float;  // result component type
    bool shadow = false, ms = false;
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
    bool containsSampler() const;
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

struct Member {
    std::string name;
    Type type;
    bool rowMajor = false;
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
    UniformBlockDecl* block = nullptr;  // BlockMember / BlockInstance
    int member = -1;                     // BlockMember: index in the block
    int id = 0;                          // unique per shader (emitter maps)
    bool used = false;                   // referenced by any expression
};

struct UniformBlockDecl {
    std::string name, instance;
    std::vector<Member> members;
    std::vector<int> dims;  // instance array
    int binding = -1;
    bool std430 = false;
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

// std140 layout (also used for the default uniform block).
uint32_t std140Align(const Type& t, bool rowMajor = false);
uint32_t std140Size(const Type& t, bool rowMajor = false);
uint32_t std140ArrayStride(const Type& t, bool rowMajor = false);  // stride of t's elements (t is an array)
uint32_t std140MatrixStride(const Type& t);
// Offsets of the members of a struct/block laid out with std140.
std::vector<uint32_t> std140Offsets(const std::vector<Member>& members, uint32_t* size);

uint32_t glTypeEnum(const Type& t);

}  // namespace vpglsl
