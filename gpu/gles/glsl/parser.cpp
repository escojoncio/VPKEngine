// Recursive-descent parser with type checking (GLSL ES 1.00 / 3.00 / 3.10 vertex and fragment).
// Produces the typed AST of ast.h; constant scalars are folded on the way (array sizes, const ints).
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

#include "ast.h"
#include "builtins.h"
#include "pp.h"

namespace vpglsl {

namespace {

struct ParseError : std::runtime_error {
    int line;
    ParseError(int l, const std::string& m) : std::runtime_error(m), line(l) {}
};

struct Qual {
    bool isConst = false, uniform = false, in = false, out = false, inout = false, attribute = false,
         varying = false, centroid = false, flat = false, smooth = false, invariant = false, shared = false,
         buffer = false;
    int location = -1, binding = -1, numViews = 0, offset = -1;
    int localSize[3] = {0, 0, 0};
    bool std430 = false, std140 = false, rowMajor = false, colMajor = false, earlyFragmentTests = false;
    uint8_t mem = 0;     // MemQual
    uint8_t ifmt = 0;    // image format layout qualifier (SPIR-V ImageFormat)
    uint8_t prec = 0;    // precision qualifier: 1 lowp, 2 mediump, 3 highp
    uint8_t packing = 0; // layout(shared|packed|std140|std430): 1 shared, 2 packed, 3 std140, 4 std430
    bool layoutOther = false;  // a layout id other than row_major / column_major
    bool any() const {
        return isConst || uniform || in || out || inout || attribute || varying || centroid || flat || smooth ||
               invariant || shared || buffer || location >= 0 || binding >= 0 || mem != 0;
    }
};

class Parser {
public:
    Parser(Stage stage, PpResult pp, std::string& log) : stage_(stage), pp_(std::move(pp)), log_(log) {
        sh_ = std::make_shared<Shader>();
        sh_->stage = stage;
        sh_->version = pp_.version;
    }

    std::shared_ptr<Shader> run() {
        try {
            pushScope();
            declareBuiltins();
            pushScope();  // user globals may shadow builtins? No (ES): same scope rules; keep separate for lookup
            while (peek().kind != Tok::End) externalDeclaration();
            for (Function* f : sh_->functions)
                if (!f->body) {
                    // Prototype without a definition is only an error when called; checked at link.
                }
            if (!sh_->main()) throw ParseError(0, "missing main()");
            checkRecursion();
            if (stage_ == Stage::Compute) {
                if (pp_.version < 310) throw ParseError(0, "compute shaders need #version 310 es");
                if (!sh_->hasLocalSize) throw ParseError(0, "compute shaders must declare layout(local_size_x...) in");
                const auto& l = sh_->localSize;
                if (l[0] * l[1] * l[2] > 1024 || l[2] > 64 || l[0] > 1024 || l[1] > 1024)
                    throw ParseError(0, "work group size exceeds the limits (1024 invocations, z <= 64)");
            }
        } catch (const ParseError& e) {
            log_ += "ERROR: 0:" + std::to_string(e.line) + ": " + e.what() + "\n";
            return nullptr;
        }
        return sh_;
    }

private:
    // Recursion (static, not even through a chain of calls) cannot be compiled: SPIR-V forbids it and the hardware has
    // no stack. Always an error, in lenient mode too.
    void checkRecursion() {
        std::map<Function*, int> state;  // 1 on the DFS stack, 2 done
        std::function<void(Function*)> visit = [&](Function* f) {
            state[f] = 1;
            for (Function* c : callGraph_[f]) {
                if (state[c] == 1) throw ParseError(0, "'" + c->name + "' : recursion is not allowed");
                if (state[c] == 0) visit(c);
            }
            state[f] = 2;
        };
        if (Function* m = sh_->main()) visit(m);
    }

    // ---- tokens ---------------------------------------------------------------------------------
    const Token& peek(size_t k = 0) const {
        size_t i = std::min(pos_ + k, pp_.tokens.size() - 1);
        return pp_.tokens[i];
    }
    Token next() { return pp_.tokens[std::min(pos_++, pp_.tokens.size() - 1)]; }
    bool isP(const char* p, size_t k = 0) const { return peek(k).kind == Tok::Punct && peek(k).text == p; }
    bool isId(const char* p, size_t k = 0) const { return peek(k).kind == Tok::Ident && peek(k).text == p; }
    bool acceptP(const char* p) {
        if (!isP(p)) return false;
        ++pos_;
        return true;
    }
    void expectP(const char* p) {
        if (!acceptP(p)) err("expected '" + std::string(p) + "' before '" + peek().text + "'");
    }
    std::string expectIdent() {
        if (peek().kind != Tok::Ident) err("expected an identifier before '" + peek().text + "'");
        return next().text;
    }
    [[noreturn]] void err(const std::string& m) const { throw ParseError(peek().line, m); }
    [[noreturn]] void errAt(int line, const std::string& m) const { throw ParseError(line, m); }
    // A rule of the language that a lenient driver lets through (see strictMode()): an error in strict mode, otherwise
    // noted in lenient_ and ignored. Code after it must be valid whether or not the shader broke the rule.
    void strictAt(const char* rule, int line, const std::string& m) {
        if (strictMode()) errAt(line, m);
        if (std::find(lenient_.begin(), lenient_.end(), rule) == lenient_.end()) lenient_.push_back(rule);
    }
    bool es3() const { return pp_.version >= 300; }

    // ---- scopes ---------------------------------------------------------------------------------
    struct Scope {
        std::map<std::string, Symbol*> vars;
        std::map<std::string, StructDef*> structs;
    };
    std::vector<Scope> scopes_;
    std::map<std::string, std::vector<Function*>> functions_;

    void pushScope() { scopes_.emplace_back(); }
    void popScope() { scopes_.pop_back(); }
    Symbol* findVar(const std::string& n) const {
        for (size_t i = scopes_.size(); i-- > 0;) {
            auto it = scopes_[i].vars.find(n);
            if (it != scopes_[i].vars.end()) return it->second;
        }
        return nullptr;
    }
    StructDef* findStruct(const std::string& n) const {
        for (size_t i = scopes_.size(); i-- > 0;) {
            auto it = scopes_[i].structs.find(n);
            if (it != scopes_[i].structs.end()) return it->second;
        }
        return nullptr;
    }
    // Keywords and reserved words cannot name anything the shader declares (GLSL ES 1.00 / 3.00 / 3.10 spec, "Keywords").
    void checkName(const std::string& n, int line) {
        static const std::set<std::string> kw100 = {
            "attribute", "const", "uniform", "varying", "break", "continue", "do", "for", "while", "if", "else", "in", "out", "inout",
            "float", "int", "void", "bool", "true", "false", "lowp", "mediump", "highp", "precision", "invariant", "discard", "return",
            "mat2", "mat3", "mat4", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "sampler2D",
            "samplerCube", "struct",
            // reserved
            "asm", "class", "union", "enum", "typedef", "template", "this", "packed", "goto", "switch", "default", "inline", "noinline",
            "volatile", "public", "static", "extern", "external", "interface", "flat", "long", "short", "double", "half", "fixed",
            "unsigned", "superp", "input", "output", "hvec2", "hvec3", "hvec4", "dvec2", "dvec3", "dvec4", "fvec2", "fvec3", "fvec4",
            "sampler1D", "sampler3D", "sampler1DShadow", "sampler2DShadow", "sampler2DRect", "sampler3DRect", "sampler2DRectShadow",
            "sizeof", "cast", "namespace", "using"};
        static const std::set<std::string> kw300 = {
            "const", "uniform", "layout", "centroid", "flat", "smooth", "break", "continue", "do", "for", "while", "switch", "case",
            "default", "if", "else", "in", "out", "inout", "float", "int", "void", "bool", "true", "false", "invariant", "discard",
            "return", "mat2", "mat3", "mat4", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4",
            "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "uint", "uvec2", "uvec3", "uvec4", "lowp",
            "mediump", "highp", "precision", "sampler2D", "sampler3D", "samplerCube", "sampler2DShadow", "samplerCubeShadow",
            "sampler2DArray", "sampler2DArrayShadow", "isampler2D", "isampler3D", "isamplerCube", "isampler2DArray", "usampler2D",
            "usampler3D", "usamplerCube", "usampler2DArray", "struct",
            // reserved
            "attribute", "varying", "coherent", "restrict", "readonly", "writeonly", "resource", "atomic_uint", "noperspective",
            "patch", "sample", "subroutine", "common", "partition", "active", "asm", "class", "union", "enum", "typedef", "template",
            "this", "goto", "inline", "noinline", "volatile", "public", "static", "extern", "external", "interface", "long", "short",
            "double", "half", "fixed", "unsigned", "superp", "input", "output", "hvec2", "hvec3", "hvec4", "dvec2", "dvec3", "dvec4",
            "fvec2", "fvec3", "fvec4", "sampler3DRect", "filter", "image1D", "image2D", "image3D", "imageCube", "iimage1D", "iimage2D",
            "iimage3D", "iimageCube", "uimage1D", "uimage2D", "uimage3D", "uimageCube", "image1DArray", "image2DArray", "iimage1DArray",
            "iimage2DArray", "uimage1DArray", "uimage2DArray", "imageBuffer", "iimageBuffer", "uimageBuffer", "sampler1D",
            "sampler1DShadow", "sampler1DArray", "sampler1DArrayShadow", "isampler1D", "isampler1DArray", "usampler1D",
            "usampler1DArray", "sampler2DRect", "sampler2DRectShadow", "isampler2DRect", "usampler2DRect", "samplerBuffer",
            "isamplerBuffer", "usamplerBuffer", "sampler2DMS", "isampler2DMS", "usampler2DMS", "sampler2DMSArray", "isampler2DMSArray",
            "usampler2DMSArray", "sizeof", "cast", "namespace", "using"};
        const bool bad = es3() ? (kw300.count(n) != 0 || (pp_.version >= 310 && (n == "buffer" || n == "shared")))
                               : (kw100.count(n) != 0 || n.find("__") != std::string::npos);
        if (bad) strictAt("keyword_as_identifier", line, "'" + n + "' : reserved word or keyword used as an identifier");
        if (n.size() > 1024) strictAt("identifier_length", line, "'" + n.substr(0, 16) + "...' : identifier too long");
    }
    void declare(Symbol* s, int line) {
        if (s->storage != Storage::Builtin) checkName(s->name, line);
        if (s->name.rfind("gl_", 0) == 0 && s->storage != Storage::Builtin)
            errAt(line, "'" + s->name + "' : identifiers starting with \"gl_\" are reserved");
        auto& v = scopes_.back().vars;
        if (v.count(s->name)) errAt(line, "'" + s->name + "' : redefinition");
        if (s->storage != Storage::Builtin && !s->name.empty() &&
            (scopes_.back().structs.count(s->name) || (scopes_.size() == 2 && (functions_.count(s->name) || blockNames_.count(s->name)))))
            strictAt("name_conflict", line, "'" + s->name + "' : name already used");
        v[s->name] = s;
    }

    // ---- builtins -------------------------------------------------------------------------------
    Symbol* builtinVar(const char* name, Type t, BuiltinVar b, Storage st = Storage::Builtin) {
        Symbol* s = sh_->newSymbol();
        s->name = name, s->type = t, s->builtin = b, s->storage = st;
        scopes_.back().vars[name] = s;
        return s;
    }
    void builtinConst(const char* name, int v) {
        Symbol* s = sh_->newSymbol();
        s->name = name, s->type = Type::make(Base::Int), s->storage = Storage::Const;
        s->hasConst = true, s->constI = v, s->constF = v, s->constant = true;
        scopes_.back().vars[name] = s;
    }
    void declareBuiltins() {
        const Type v4 = Type::make(Base::Float, 4), f1 = Type::make(Base::Float), i1 = Type::make(Base::Int);
        builtinConst("gl_MaxVertexAttribs", 16);
        builtinConst("gl_MaxVertexUniformVectors", 256);
        builtinConst("gl_MaxFragmentUniformVectors", 256);
        builtinConst("gl_MaxVaryingVectors", 15);
        builtinConst("gl_MaxVertexOutputVectors", 16);
        builtinConst("gl_MaxFragmentInputVectors", 15);
        builtinConst("gl_MaxVertexTextureImageUnits", 16);
        builtinConst("gl_MaxCombinedTextureImageUnits", 32);
        builtinConst("gl_MaxTextureImageUnits", 16);
        builtinConst("gl_MaxDrawBuffers", 8);
        builtinConst("gl_MinProgramTexelOffset", -8);
        builtinConst("gl_MaxProgramTexelOffset", 7);
        if (pp_.version >= 310) {
            // Match the limits the GL layer reports (GL_MAX_*).
            builtinConst("gl_MaxImageUnits", 8);
            builtinConst("gl_MaxVertexImageUniforms", 8);
            builtinConst("gl_MaxFragmentImageUniforms", 8);
            builtinConst("gl_MaxComputeImageUniforms", 8);
            builtinConst("gl_MaxCombinedImageUniforms", 24);
            builtinConst("gl_MaxCombinedShaderOutputResources", 16);
            builtinConst("gl_MaxComputeUniformComponents", 1024);
            builtinConst("gl_MaxComputeTextureImageUnits", 16);
            builtinConst("gl_MaxComputeAtomicCounters", 8);
            builtinConst("gl_MaxComputeAtomicCounterBuffers", 1);
            builtinConst("gl_MaxVertexAtomicCounters", 0);
            builtinConst("gl_MaxFragmentAtomicCounters", 8);
            builtinConst("gl_MaxCombinedAtomicCounters", 8);
            builtinConst("gl_MaxAtomicCounterBindings", 8);
            builtinConst("gl_MaxVertexAtomicCounterBuffers", 0);
            builtinConst("gl_MaxFragmentAtomicCounterBuffers", 1);
            builtinConst("gl_MaxCombinedAtomicCounterBuffers", 1);
            builtinConst("gl_MaxAtomicCounterBufferSize", 32);
        }
        Type u1 = Type::make(Base::Uint);
        if (stage_ == Stage::Vertex) {
            builtinVar("gl_Position", v4, BuiltinVar::Position);
            builtinVar("gl_PointSize", f1, BuiltinVar::PointSize);
            if (es3()) {
                builtinVar("gl_VertexID", i1, BuiltinVar::VertexID);
                builtinVar("gl_InstanceID", i1, BuiltinVar::InstanceID);
            }
        } else if (stage_ == Stage::Fragment) {
            builtinVar("gl_FragCoord", v4, BuiltinVar::FragCoord);
            builtinVar("gl_FrontFacing", Type::make(Base::Bool), BuiltinVar::FrontFacing);
            builtinVar("gl_PointCoord", Type::make(Base::Float, 2), BuiltinVar::PointCoord);
            if (pp_.version >= 320) builtinVar("gl_HelperInvocation", Type::make(Base::Bool), BuiltinVar::HelperInvocation);
            if (fbFetchExt()) {
                Type fd = v4;
                fd.dims = {8};
                if (!es3()) builtinVar("gl_LastFragData", fd, BuiltinVar::LastFragData);
                if (pp_.extensions.count("GL_ARM_shader_framebuffer_fetch")) builtinVar("gl_LastFragColorARM", v4, BuiltinVar::LastFragColor);
            }
            if (!es3()) {
                builtinVar("gl_FragColor", v4, BuiltinVar::FragColor);
                Type fd = v4;
                fd.dims = {8};
                builtinVar("gl_FragData", fd, BuiltinVar::FragData);
                builtinVar("gl_FragDepthEXT", f1, BuiltinVar::FragDepth);
            } else {
                builtinVar("gl_FragDepth", f1, BuiltinVar::FragDepth);
            }
        } else {
            Type u3 = Type::make(Base::Uint, 3);
            builtinVar("gl_LocalInvocationID", u3, BuiltinVar::LocalInvocationID);
            builtinVar("gl_GlobalInvocationID", u3, BuiltinVar::GlobalInvocationID);
            builtinVar("gl_WorkGroupID", u3, BuiltinVar::WorkGroupID);
            builtinVar("gl_NumWorkGroups", u3, BuiltinVar::NumWorkGroups);
            builtinVar("gl_LocalInvocationIndex", u1, BuiltinVar::LocalInvocationIndex);
            builtinVar("gl_WorkGroupSize", u3, BuiltinVar::WorkGroupSize);
        }
        if (stage_ != Stage::Compute) builtinVar("gl_ViewID_OVR", u1, BuiltinVar::ViewID);
    }

    bool fbFetchExt() const {
        return pp_.extensions.count("GL_EXT_shader_framebuffer_fetch") || pp_.extensions.count("GL_ARM_shader_framebuffer_fetch");
    }

    // ---- types ----------------------------------------------------------------------------------
    static bool precisionWord(const std::string& s) { return s == "highp" || s == "mediump" || s == "lowp"; }
    static uint8_t precisionOf(const std::string& s) { return s == "highp" ? 3 : s == "mediump" ? 2 : 1; }
    // The precision a declaration gets: its own qualifier, else the type's (`mediump float`), else the default in force
    // (`precision` statements; the built-in ones: sampler2D and samplerCube lowp, vertex float / int highp, fragment int mediump).
    uint8_t effectivePrecision(uint8_t qualifier, uint8_t typeWord, const Type& t) const {
        if (qualifier) return qualifier;
        if (typeWord) return typeWord;
        std::string key;
        if (t.base == Base::Float) key = "float";
        else if (t.base == Base::Int || t.base == Base::Uint) key = "int";
        else if (t.base == Base::Sampler) key = t.str();
        else return 0;
        auto it = defaultPrec_.find(key);
        if (it != defaultPrec_.end()) return it->second;
        if (key == "sampler2D" || key == "samplerCube") return 1;
        if (key == "float" || key == "int") return stage_ == Stage::Vertex ? 3 : key == "int" ? 2 : 0;
        return 0;
    }

    std::optional<Type> basicType(const std::string& n) const {
        static const std::map<std::string, Type> kTypes = [] {
            std::map<std::string, Type> m;
            m["void"] = Type::make(Base::Void);
            m["float"] = Type::make(Base::Float), m["int"] = Type::make(Base::Int);
            m["uint"] = Type::make(Base::Uint), m["bool"] = Type::make(Base::Bool);
            for (int n = 2; n <= 4; ++n) {
                m["vec" + std::to_string(n)] = Type::make(Base::Float, n);
                m["ivec" + std::to_string(n)] = Type::make(Base::Int, n);
                m["uvec" + std::to_string(n)] = Type::make(Base::Uint, n);
                m["bvec" + std::to_string(n)] = Type::make(Base::Bool, n);
                m["mat" + std::to_string(n)] = Type::make(Base::Float, n, n);
                for (int r = 2; r <= 4; ++r)
                    m["mat" + std::to_string(n) + "x" + std::to_string(r)] = Type::make(Base::Float, r, n);
            }
            auto smp = [&](const char* name, SDim d, Base b, bool shadow, bool ms) {
                Type t = Type::make(Base::Sampler);
                t.dim = d, t.sampled = b, t.shadow = shadow, t.ms = ms;
                m[name] = t;
            };
            smp("sampler2D", SDim::D2, Base::Float, false, false);
            smp("sampler3D", SDim::D3, Base::Float, false, false);
            smp("samplerCube", SDim::Cube, Base::Float, false, false);
            smp("sampler2DArray", SDim::D2Array, Base::Float, false, false);
            smp("sampler2DShadow", SDim::D2, Base::Float, true, false);
            smp("samplerCubeShadow", SDim::Cube, Base::Float, true, false);
            smp("sampler2DArrayShadow", SDim::D2Array, Base::Float, true, false);
            smp("samplerExternalOES", SDim::External, Base::Float, false, false);
            smp("__samplerExternal2DY2YEXT", SDim::External, Base::Float, false, false);
            smp("sampler2DMS", SDim::D2, Base::Float, false, true);
            smp("samplerCubeArray", SDim::CubeArray, Base::Float, false, false);
            smp("samplerCubeArrayShadow", SDim::CubeArray, Base::Float, true, false);
            smp("isamplerCubeArray", SDim::CubeArray, Base::Int, false, false);
            smp("usamplerCubeArray", SDim::CubeArray, Base::Uint, false, false);
            smp("samplerBuffer", SDim::Buffer, Base::Float, false, false);
            smp("isamplerBuffer", SDim::Buffer, Base::Int, false, false);
            smp("usamplerBuffer", SDim::Buffer, Base::Uint, false, false);
            auto img = [&](const char* name, SDim d, Base b) {
                Type t = Type::make(Base::Image);
                t.dim = d, t.sampled = b;
                m[name] = t;
            };
            img("image2D", SDim::D2, Base::Float), img("iimage2D", SDim::D2, Base::Int), img("uimage2D", SDim::D2, Base::Uint);
            img("image3D", SDim::D3, Base::Float), img("iimage3D", SDim::D3, Base::Int), img("uimage3D", SDim::D3, Base::Uint);
            img("imageCube", SDim::Cube, Base::Float), img("iimageCube", SDim::Cube, Base::Int), img("uimageCube", SDim::Cube, Base::Uint);
            img("image2DArray", SDim::D2Array, Base::Float), img("iimage2DArray", SDim::D2Array, Base::Int),
                img("uimage2DArray", SDim::D2Array, Base::Uint);
            img("imageCubeArray", SDim::CubeArray, Base::Float), img("iimageCubeArray", SDim::CubeArray, Base::Int),
                img("uimageCubeArray", SDim::CubeArray, Base::Uint);
            img("imageBuffer", SDim::Buffer, Base::Float), img("iimageBuffer", SDim::Buffer, Base::Int),
                img("uimageBuffer", SDim::Buffer, Base::Uint);
            m["atomic_uint"] = Type::make(Base::AtomicUint);
            smp("isampler2D", SDim::D2, Base::Int, false, false);
            smp("isampler3D", SDim::D3, Base::Int, false, false);
            smp("isamplerCube", SDim::Cube, Base::Int, false, false);
            smp("isampler2DArray", SDim::D2Array, Base::Int, false, false);
            smp("isampler2DMS", SDim::D2, Base::Int, false, true);
            smp("usampler2D", SDim::D2, Base::Uint, false, false);
            smp("usampler3D", SDim::D3, Base::Uint, false, false);
            smp("usamplerCube", SDim::Cube, Base::Uint, false, false);
            smp("usampler2DArray", SDim::D2Array, Base::Uint, false, false);
            smp("usampler2DMS", SDim::D2, Base::Uint, false, true);
            return m;
        }();
        auto it = kTypes.find(n);
        if (it == kTypes.end()) return std::nullopt;
        if (!es3() && (it->second.base == Base::Uint || (it->second.cols && it->second.cols != it->second.vec) ||
                       (it->second.base == Base::Sampler && (it->second.sampled != Base::Float || it->second.dim == SDim::D2Array || it->second.ms))))
            return std::nullopt;  // ES 1.00 keywords only (uint, matNxM, integer samplers are reserved)
        const Type& ty = it->second;
        if ((ty.base == Base::Image || ty.base == Base::AtomicUint) && pp_.version < 310) return std::nullopt;  // ES 3.1
        if (ty.dim == SDim::CubeArray && (ty.base == Base::Sampler || ty.base == Base::Image) && pp_.version < 320 &&
            !pp_.extensions.count("GL_OES_texture_cube_map_array") && !pp_.extensions.count("GL_EXT_texture_cube_map_array"))
            return std::nullopt;  // ES 3.2 (or the extension)
        if (ty.dim == SDim::Buffer && (pp_.version < 310 || (pp_.version < 320 && !pp_.extensions.count("GL_EXT_texture_buffer"))))
            return std::nullopt;  // ES 3.2 (or GL_EXT_texture_buffer in 3.10)
        return it->second;
    }
    bool atTypeStart(size_t k = 0) const {
        const Token& t = peek(k);
        if (t.kind != Tok::Ident) return false;
        if (t.text == "struct" || precisionWord(t.text)) return true;
        return basicType(t.text).has_value() || (findStruct(t.text) && !findVar(t.text));
    }
    static bool qualifierWord(const std::string& s) {
        static const std::set<std::string> q = {"const", "uniform", "in", "out", "inout", "attribute", "varying",
                                                "centroid", "flat", "smooth", "invariant", "layout", "highp",
                                                "mediump", "lowp", "precise", "shared", "buffer", "readonly",
                                                "writeonly", "coherent", "volatile", "restrict", "noperspective"};
        return q.count(s) != 0;
    }

    int constIntExpr() {
        ExprP e = conditional();
        if (!e->hasConst || !e->type.isIntegral() || !e->type.isScalar())
            errAt(e->line, "constant integer expression required");
        return int(e->constI);
    }

    // SPIR-V ImageFormat of an ES 3.1 image format qualifier (0 = not one).
    static int imageFormat(const std::string& n) {
        static const std::map<std::string, int> m = {
            {"rgba32f", 1}, {"rgba16f", 2}, {"r32f", 3}, {"rgba8", 4}, {"rgba8_snorm", 5},
            {"rgba32i", 21}, {"rgba16i", 22}, {"rgba8i", 23}, {"r32i", 24},
            {"rgba32ui", 30}, {"rgba16ui", 31}, {"rgba8ui", 32}, {"r32ui", 33},
        };
        auto it = m.find(n);
        return it == m.end() ? 0 : it->second;
    }
    // Component type an image format stores: 0 float, 1 int, 2 uint.
    static int imageFormatKind(int f) { return f >= 30 ? 2 : f >= 21 ? 1 : 0; }

    void layoutQualifier(Qual& q) {
        const int qline = peek().line;
        expectP("(");
        if (isP(")")) strictAt("layout_qualifier", qline, "'layout' : empty layout qualifier list");
        static const std::set<std::string> valued = {"location", "binding", "num_views", "offset", "local_size_x", "local_size_y",
                                                     "local_size_z", "index"};
        static const std::set<std::string> flags = {"std140", "std430", "shared", "packed", "row_major", "column_major",
                                                    "early_fragment_tests"};
        while (!isP(")")) {
            const int idLine = peek().line;
            std::string id = expectIdent();
            int v = 0;
            bool hasValue = acceptP("=");
            if (hasValue) v = constIntExpr();
            const bool known = valued.count(id) || flags.count(id) || imageFormat(id) || id.rfind("blend_support_", 0) == 0;
            if (!known) strictAt("layout_qualifier", idLine, "'" + id + "' : unknown layout qualifier id");
            else if (valued.count(id) && !hasValue) strictAt("layout_qualifier", idLine, "'" + id + "' : layout qualifier needs a value");
            else if (!valued.count(id) && hasValue) strictAt("layout_qualifier", idLine, "'" + id + "' : layout qualifier takes no value");
            if (id == "location") q.location = v;
            else if (id == "binding") q.binding = v;
            else if (id == "num_views") q.numViews = v;
            else if (id == "std430") q.std430 = true, q.std140 = false, q.packing = 4;
            else if (id == "std140") q.std140 = true, q.std430 = false, q.packing = 3;
            else if (id == "shared") q.packing = 1;
            else if (id == "packed") q.packing = 2;
            else if (id == "offset") q.offset = v;
            else if (id == "row_major") q.rowMajor = true, q.colMajor = false;
            else if (id == "column_major") q.rowMajor = false, q.colMajor = true;
            else if (id == "local_size_x") q.localSize[0] = v;
            else if (id == "local_size_y") q.localSize[1] = v;
            else if (id == "local_size_z") q.localSize[2] = v;
            else if (id == "early_fragment_tests") q.earlyFragmentTests = true;
            else if (int f = imageFormat(id)) q.ifmt = uint8_t(f);
            // index, blend_support_*: accepted and ignored.
            if (id != "row_major" && id != "column_major") q.layoutOther = true;
            if (!acceptP(",")) break;
        }
        expectP(")");
    }

    // `param`: a function parameter (const, then in/out/inout, then precision). Before GLSL ES 3.10 the order of
    // the qualifiers is fixed: [layout] [invariant] [smooth|flat] [centroid] [storage] [precision].
    Qual qualifiers(bool param = false) {
        Qual q;
        int lastRank = 0;
        bool sawStorage = false;
        while (peek().kind == Tok::Ident && qualifierWord(peek().text)) {
            const int wline = peek().line;
            std::string w = next().text;
            if (pp_.version < 310) {
                int rank;
                if (w == "layout") rank = 0;
                else if (param) rank = w == "const" ? 1 : w == "in" || w == "out" || w == "inout" ? 2 : 3;
                else if (w == "invariant") rank = 1;
                else if (w == "smooth" || w == "flat") rank = 2;
                else if (w == "highp" || w == "mediump" || w == "lowp") rank = 4;
                else rank = 3;  // centroid, const, in, out, uniform, attribute, varying
                if (rank < lastRank || (!param && w == "centroid" && sawStorage))
                    strictAt("qualifier_order", wline, "'" + w + "' : qualifiers in the wrong order");
                lastRank = rank;
                if (!param && rank == 3 && w != "centroid") sawStorage = true;
            }
            if (w == "const") q.isConst = true;
            else if (w == "uniform") q.uniform = true;
            else if (w == "in") q.in = true;
            else if (w == "out") q.out = true;
            else if (w == "inout") q.inout = true;
            else if (w == "attribute") q.attribute = true;
            else if (w == "varying") q.varying = true;
            else if (w == "centroid") q.centroid = true;
            else if (w == "flat") q.flat = true;
            else if (w == "smooth" || w == "noperspective") q.smooth = true;
            else if (w == "invariant") q.invariant = true;
            else if (w == "shared") q.shared = true;
            else if (w == "buffer") q.buffer = true;
            else if (w == "layout") layoutQualifier(q);
            else if (w == "readonly") q.mem |= kReadonly;
            else if (w == "writeonly") q.mem |= kWriteonly;
            else if (w == "coherent") q.mem |= kCoherent;
            else if (w == "volatile") q.mem |= kVolatile;
            else if (w == "restrict") q.mem |= kRestrict;
            else if (precisionWord(w)) q.prec = precisionOf(w);
            // precise: no effect on code generation.
        }
        if (q.invariant && (q.uniform || q.attribute))
            strictAt("invariant_storage", peek().line, "'invariant' : only shader outputs can be invariant");
        if (pp_.version == 300 && q.invariant && q.in)
            strictAt("invariant_input", peek().line, "'invariant' : only shader outputs can be invariant in GLSL ES 3.00");
        return q;
    }

    Type typeSpecifier() {
        uint8_t pr = 0;
        while (peek().kind == Tok::Ident && precisionWord(peek().text)) pr = precisionOf(next().text);
        const Token t = peek();
        if (t.kind != Tok::Ident) err("expected a type");
        if (t.text == "struct") {
            Type ty = structSpecifier();
            tsPrec_ = 0;
            return ty;
        }
        next();
        if (auto b = basicType(t.text)) {
            Type ty = *b;
            arraySuffix(ty);
            tsPrec_ = pr;
            return ty;
        }
        if (StructDef* s = findStruct(t.text)) {
            Type ty;
            ty.base = Base::Struct, ty.st = s;
            arraySuffix(ty);
            tsPrec_ = 0;
            return ty;
        }
        errAt(t.line, "'" + t.text + "' : unknown type");
    }

    // [N][M]... ; `[]` (implicit size) yields 0 and is resolved by the initialiser.
    void arraySuffix(Type& t) {
        std::vector<int> d;
        while (acceptP("[")) {
            if (acceptP("]")) {
                d.push_back(0);
                continue;
            }
            int n = constIntExpr();
            if (n <= 0) err("array size must be a positive integer");
            expectP("]");
            d.push_back(n);
        }
        if (!d.empty() && !es3() && (d.size() > 1 || !t.dims.empty())) err("arrays of arrays need GLSL ES 3.10");
        if (!d.empty() && pp_.version < 310 && (d.size() > 1 || !t.dims.empty()))
            strictAt("arrays_of_arrays", peek().line, "arrays of arrays need GLSL ES 3.10");
        t.dims.insert(t.dims.end(), d.begin(), d.end());
    }

    Type structSpecifier() {
        int line = next().line;  // struct
        auto def = std::make_unique<StructDef>();
        if (peek().kind == Tok::Ident) checkName(peek().text, peek().line), def->name = next().text;
        else def->name = "__anon" + std::to_string(sh_->structs.size());
        expectP("{");
        while (!acceptP("}")) {
            Qual q = qualifiers();
            Type base = typeSpecifier();
            const uint8_t basePrec = tsPrec_;
            do {
                Member m;
                m.name = expectIdent();
                checkName(m.name, peek().line);
                m.type = base;
                arraySuffix(m.type);
                m.prec = effectivePrecision(q.prec, basePrec, m.type);
                needPrecision(m.prec, m.type, line, m.name);
                m.rowMajor = q.rowMajor;
                for (auto& o : def->members)
                    if (o.name == m.name) errAt(line, "'" + m.name + "' : duplicate member");
                def->members.push_back(m);
            } while (acceptP(","));
            expectP(";");
        }
        def->index = int(sh_->structs.size());
        StructDef* p = def.get();
        sh_->structs.push_back(std::move(def));
        if (p->name.rfind("__anon", 0) != 0) {
            if (scopes_.back().structs.count(p->name)) errAt(line, "'" + p->name + "' : redefinition");
            if (scopes_.size() == 2 && (functions_.count(p->name) || blockNames_.count(p->name) || scopes_.back().vars.count(p->name)))
                strictAt("name_conflict", line, "'" + p->name + "' : name already used");
            scopes_.back().structs[p->name] = p;
        }
        Type t;
        t.base = Base::Struct, t.st = p;
        arraySuffix(t);
        return t;
    }

    // ---- declarations ---------------------------------------------------------------------------
    void externalDeclaration() {
        if (acceptP(";")) return;
        if (isId("precision")) {
            precisionStatement();
            return;
        }
        int line = peek().line;
        Qual q = qualifiers();
        // Qualifier-only declarations: layout(...) in; / invariant gl_Position;
        if (acceptP(";")) {
            if (q.numViews > 0) sh_->numViews = q.numViews;
            if (q.localSize[0] || q.localSize[1] || q.localSize[2]) {
                if (stage_ != Stage::Compute) errAt(line, "local_size is only valid in compute shaders");
                for (int i = 0; i < 3; ++i) sh_->localSize[i] = std::max(1, q.localSize[i]);
                sh_->hasLocalSize = true;
            }
            if (q.earlyFragmentTests) sh_->earlyFragmentTests = true;
            // Default layouts of the blocks that follow: `layout(std430) buffer;`, `layout(row_major) uniform;`.
            if (q.buffer) {
                if (q.std430 || q.std140) defaultBufferStd430_ = q.std430;
                if (q.rowMajor || q.colMajor) defaultBufferRowMajor_ = q.rowMajor;
                if (q.packing) defaultBufferPacking_ = q.packing;
            }
            if (q.uniform && (q.rowMajor || q.colMajor)) defaultUniformRowMajor_ = q.rowMajor;
            if (q.uniform && q.packing == 4) strictAt("uniform_std430", line, "std430 is only for shader storage blocks");
            if (q.uniform && q.packing) defaultUniformPacking_ = q.packing;
            return;
        }
        if (stage_ == Stage::Compute && (q.in || q.out || q.attribute || q.varying))
            errAt(line, "compute shaders have no inputs or outputs");
        if (q.invariant && peek().kind == Tok::Ident && !atTypeStart()) {
            // invariant <varying>;
            do expectIdent();
            while (acceptP(","));
            expectP(";");
            return;
        }
        // Interface block: qualifier IDENT {
        if ((q.uniform || q.buffer || q.in || q.out) && peek().kind == Tok::Ident && isP("{", 1)) {
            interfaceBlock(q, line);
            return;
        }
        Type t = typeSpecifier();
        declTypePrec_ = tsPrec_;
        if (acceptP(";")) return;  // struct declaration only
        std::string name = expectIdent();
        if (name != "main") checkName(name, peek().line);
        if (isP("(")) {
            if (q.any() || q.layoutOther)
                strictAt("return_qualifier", line, "'" + name + "' : a function return type takes no qualifiers other than precision");
            needPrecision(effectivePrecision(q.prec, declTypePrec_, t), t, line, name);
            functionDefinition(t, name, line);
            return;
        }
        globalDeclarators(q, t, name, line);
    }

    // precision <highp|mediump|lowp> <type>; sets the default of that type (float, int, or a sampler) from here on.
    void precisionStatement() {
        next();  // precision
        const std::string word = peek().kind == Tok::Ident && precisionWord(peek().text) ? next().text : "";
        if (word.empty()) err("expected a precision qualifier");
        const std::string type = peek().text;
        typeSpecifier();
        defaultPrec_[type == "uint" ? "int" : type] = precisionOf(word);
        expectP(";");
    }

    void interfaceBlock(const Qual& q, int line) {
        if (!es3()) errAt(line, "interface blocks need GLSL ES 3.00");
        if (q.buffer && pp_.version < 310) errAt(line, "shader storage blocks need GLSL ES 3.10");
        if (!q.uniform && !q.buffer) errAt(line, "only uniform and shader storage blocks are supported (I/O blocks are not)");
        auto b = std::make_unique<UniformBlockDecl>();
        checkName(peek().text, peek().line);
        if (peek().text.rfind("gl_", 0) == 0) strictAt("reserved_name", peek().line, "'" + peek().text + "' : identifiers starting with \"gl_\" are reserved");
        b->name = next().text;
        if (blockNames_.count(b->name)) strictAt("repeated_block", line, "'" + b->name + "' : interface block declared twice");
        if (functions_.count(b->name) || scopes_[1].vars.count(b->name) || scopes_[1].structs.count(b->name))
            strictAt("name_conflict", line, "'" + b->name + "' : name already used");
        blockNames_.insert(b->name);
        b->binding = q.binding;
        b->buffer = q.buffer;
        if (!q.buffer && q.packing == 4) strictAt("uniform_std430", line, "std430 is only for shader storage blocks");
        b->packing = q.packing ? q.packing : q.buffer ? defaultBufferPacking_ : defaultUniformPacking_;
        // std430 only for shader storage blocks; `shared`/`packed`/unspecified use std140 (as for uniform blocks).
        b->std430 = q.buffer && (q.std430 || (!q.std140 && defaultBufferStd430_));
        const bool defaultRow = q.buffer ? defaultBufferRowMajor_ : defaultUniformRowMajor_;
        b->rowMajor = q.colMajor ? false : (q.rowMajor || defaultRow);
        expectP("{");
        while (!acceptP("}")) {
            if (isId("struct")) strictAt("block_member_struct", peek().line, "struct definitions are not allowed inside an interface block");
            Qual mq = qualifiers();
            if (mq.isConst || (mq.uniform && !q.uniform) || mq.in || mq.out || mq.inout || mq.attribute || mq.varying || mq.centroid || mq.flat ||
                mq.smooth || mq.invariant || mq.shared || (mq.buffer && !q.buffer) || mq.layoutOther || mq.location >= 0 || mq.binding >= 0)
                strictAt("block_member_qualifier", peek().line, "an interface block member takes only matrix layout, memory and precision qualifiers");
            Type base = typeSpecifier();
            const uint8_t basePrec = tsPrec_;
            do {
                Member m;
                m.name = expectIdent();
                checkName(m.name, peek().line);
                m.type = base;
                arraySuffix(m.type);
                m.prec = effectivePrecision(mq.prec, basePrec, m.type);
                needPrecision(m.prec, m.type, line, m.name);
                m.rowMajor = mq.colMajor || q.colMajor ? false : (mq.rowMajor || q.rowMajor || defaultRow);
                m.mem = uint8_t(q.mem | mq.mem);
                if (m.type.containsOpaque()) errAt(line, "opaque types are not allowed in blocks");
                for (size_t k = 0; k < m.type.dims.size(); ++k)
                    if (m.type.dims[k] == 0 && !(k == 0 && b->buffer && isP(";") ))
                        errAt(line, "'" + m.name + "' : only the last member of a shader storage block can be unsized");
                if (!b->members.empty() && !b->members.back().type.dims.empty() && b->members.back().type.dims[0] == 0)
                    errAt(line, "'" + b->members.back().name + "' : an unsized array must be the last member of the block");
                b->members.push_back(m);
            } while (acceptP(","));
            expectP(";");
        }
        if (b->members.empty()) errAt(line, "'" + b->name + "' : empty block");
        UniformBlockDecl* bp = b.get();
        Symbol* inst = sh_->newSymbol();
        inst->storage = Storage::BlockInstance;
        inst->block = bp;
        inst->type.base = Base::Void;
        if (peek().kind == Tok::Ident) {
            bp->instance = next().text;
            if (bp->instance == b->name) strictAt("name_conflict", line, "'" + b->name + "' : the instance has the name of its block");
            Type at;
            arraySuffix(at);
            bp->dims = at.dims;
            inst->type.dims = at.dims;  // an instance array is indexed like any array (each element is one block)
            inst->name = bp->instance;
            declare(inst, line);
        } else {
            // Members are globals of their own name.
            inst->name = "";
            for (size_t i = 0; i < bp->members.size(); ++i) {
                Symbol* s = sh_->newSymbol();
                s->name = bp->members[i].name, s->type = bp->members[i].type;
                s->storage = Storage::BlockMember, s->block = bp, s->member = int(i);
                declare(s, line);
            }
        }
        bp->instanceSym = inst;
        expectP(";");
        sh_->blocks.push_back(bp);
        sh_->blockStore.push_back(std::move(b));
    }

    void globalDeclarators(const Qual& q, const Type& base, std::string name, int line) {
        for (;;) {
            Symbol* s = sh_->newSymbol();
            s->name = name;
            s->type = base;
            arraySuffix(s->type);
            s->location = q.location;
            s->binding = q.binding;
            s->prec = effectivePrecision(q.prec, declTypePrec_, base);
            needPrecision(s->prec, base, line, name);
            s->invariant = q.invariant;
            s->interp = q.flat ? Interp::Flat : q.centroid ? Interp::Centroid : Interp::Smooth;
            if (q.isConst) s->storage = Storage::Const;
            else if (q.uniform) s->storage = Storage::Uniform;
            else if (q.attribute) {
                if (stage_ != Stage::Vertex || es3()) errAt(line, "'attribute' only in ES 1.00 vertex shaders");
                s->storage = Storage::In;
            } else if (q.varying) {
                if (es3()) errAt(line, "'varying' is reserved in GLSL ES 3.00");
                s->storage = stage_ == Stage::Vertex ? Storage::Out : Storage::In;
            } else if (q.inout) {
                if (stage_ != Stage::Fragment || !es3() || !pp_.extensions.count("GL_EXT_shader_framebuffer_fetch"))
                    errAt(line, "'inout' variables need GL_EXT_shader_framebuffer_fetch in an ES 3.00+ fragment shader");
                s->storage = Storage::Out, s->fbFetch = true;
            } else if (q.in) s->storage = Storage::In;
            else if (q.out) s->storage = Storage::Out;
            else if (q.shared) {
                if (stage_ != Stage::Compute) errAt(line, "'shared' is only valid in compute shaders");
                s->storage = Storage::Shared;
            } else s->storage = Storage::Global;
            s->mem = q.mem;
            if (s->storage == Storage::In && stage_ == Stage::Fragment && s->type.isIntegral() && s->interp != Interp::Flat && es3()) {
                strictAt("integer_input_not_flat", line, "integer fragment inputs must be 'flat'");
                s->interp = Interp::Flat;  // the only interpolation integers have
            }
            if ((s->storage == Storage::In || s->storage == Storage::Out) && !s->fbFetch && !interfaceTypeOk(s->type))
                strictAt("interface_type", line, "'" + name + "' : invalid type for a shader input or output");
            if (s->type.containsOpaque() && s->storage != Storage::Uniform)
                errAt(line, "'" + name + "' : samplers, images and atomic counters must be uniform");
            if (s->type.base == Base::Image) {
                if (q.ifmt) {
                    const int want = s->type.sampled == Base::Float ? 0 : s->type.sampled == Base::Int ? 1 : 2;
                    if (imageFormatKind(q.ifmt) != want) errAt(line, "'" + name + "' : image format does not match the image type");
                    s->type.ifmt = q.ifmt;
                }
            }
            if (s->type.base == Base::AtomicUint) {
                if (q.location >= 0) strictAt("atomic_location", line, "'" + name + "' : atomic counters have no location");
                if (s->binding < 0) s->binding = 0;
                const int count = s->type.isArray() ? s->type.dims[0] : 1;
                int& next = atomicNext_[s->binding];
                if (q.offset >= 0) next = q.offset;
                if (next % 4) errAt(line, "'" + name + "' : atomic counter offset must be a multiple of 4");
                s->offset = next;
                next += 4 * count;
            }
            if (s->type.base == Base::Void) errAt(line, "'" + name + "' : void variable");
            ExprP init;
            if (acceptP("=")) {
                if (s->storage == Storage::Uniform || s->storage == Storage::In || s->storage == Storage::Out ||
                    s->storage == Storage::Shared)
                    errAt(line, "cannot initialise '" + name + "'");
                init = assignment();
                resolveImplicitSize(s->type, init->type, line);
                if (init->type != s->type)
                    errAt(line, "'=' : cannot convert from '" + init->type.str() + "' to '" + s->type.str() + "'");
                if (s->storage == Storage::Const) {
                    if (init->hasConst) {
                        s->hasConst = true, s->constF = init->constF, s->constI = init->constI;
                    }
                    s->constant = isConstExpr(*init);
                    if (!s->constant) strictAt("const_initializer", line, "'" + name + "' : the initializer of a const must be a constant expression");
                }
            } else if (s->storage == Storage::Const) {
                errAt(line, "'" + name + "' : const variables need an initialiser");
            }
            for (int d : s->type.dims)
                if (d == 0) errAt(line, "'" + name + "' : implicitly sized array without initialiser");
            declare(s, line);
            sh_->globals.push_back({s, std::move(init)});
            if (!acceptP(",")) break;
            name = expectIdent();
        }
        expectP(";");
    }

    // Every float / integer / sampler declaration needs a precision (its own or a default in force).
    void needPrecision(uint8_t effective, const Type& t, int line, const std::string& name) {
        if (effective || !(t.base == Base::Float || t.base == Base::Int || t.base == Base::Uint || t.base == Base::Sampler)) return;
        strictAt("unspecified_precision", line, "'" + name + "' : no precision specified and no default precision for '" + t.str() + "'");
    }

    // Types a shader input / output (attribute, varying, in, out) may have: ES 1.00 float-based only; ES 3.00 no bool, no
    // array of arrays, structs of non-array / non-struct / non-bool members and not in arrays.
    bool interfaceTypeOk(const Type& t) const {
        if (t.base == Base::Bool) return false;
        if (!es3()) return t.base == Base::Float;
        if (pp_.version != 300) return true;  // 3.10 / 3.20 relaxed these
        if (t.dims.size() > 1) return false;
        if (t.base == Base::Struct) {
            if (t.isArray()) return false;
            for (const Member& m : t.st->members)
                if (m.type.isArray() || m.type.base == Base::Struct || m.type.base == Base::Bool) return false;
        }
        return true;
    }

    static void resolveImplicitSize(Type& decl, const Type& init, int line) {
        if (!decl.dims.empty() && decl.dims[0] == 0) {
            if (init.dims.size() != decl.dims.size()) throw ParseError(line, "array initialiser shape mismatch");
            decl.dims[0] = init.dims[0];
        }
    }

    void functionDefinition(const Type& ret, const std::string& name, int line) {
        if (ret.containsOpaque()) errAt(line, "functions cannot return samplers, images or atomic counters");
        auto fn = std::make_unique<Function>();
        fn->name = name;
        fn->ret = ret;
        expectP("(");
        std::vector<Symbol*> params;
        if (isId("void") && isP(")", 1)) next();
        while (!isP(")")) {
            Qual q = qualifiers(true);
            if (q.uniform || q.varying || q.attribute || q.centroid || q.flat || q.smooth || q.invariant || q.shared || q.buffer ||
                (q.layoutOther && !q.ifmt) || (q.isConst && (q.out || q.inout)))
                strictAt("param_qualifier", peek().line, "'" + name + "' : invalid parameter qualifier");
            Type t = typeSpecifier();
            if (t.base == Base::Image && q.ifmt) t.ifmt = q.ifmt;
            if (t.base == Base::AtomicUint) errAt(line, "atomic counters cannot be passed to functions");
            Symbol* p = sh_->newSymbol();
            p->storage = Storage::Param;
            p->dir = q.inout ? ParamDir::InOut : q.out ? ParamDir::Out : ParamDir::In;
            if (peek().kind == Tok::Ident) p->name = next().text;
            p->type = t;
            arraySuffix(p->type);
            p->constParam = q.isConst;
            needPrecision(effectivePrecision(q.prec, tsPrec_, t), t, line, p->name.empty() ? name : p->name);
            for (int d : p->type.dims)
                if (d == 0) strictAt("unsized_param_array", line, "'" + name + "' : a parameter array needs a size");
            params.push_back(p);
            if (!acceptP(",")) break;
        }
        expectP(")");
        // Match a prior prototype.
        Function* existing = nullptr;
        for (Function* f : functions_[name]) {
            if (f->params.size() != params.size()) continue;
            bool same = true;
            for (size_t i = 0; i < params.size(); ++i) same &= f->params[i]->type == params[i]->type;
            if (same) existing = f;
        }
        if (existing && existing->ret != ret) errAt(line, "'" + name + "' : return type differs from prototype");
        if (existing)
            for (size_t i = 0; i < params.size(); ++i)
                if (existing->params[i]->dir != params[i]->dir || existing->params[i]->constParam != params[i]->constParam)
                    strictAt("prototype_mismatch", line, "'" + name + "' : parameter qualifiers differ from the prototype");
        if (!es3() && (ret.isArray() || (ret.base == Base::Struct && structHasArray(ret.st))))
            strictAt("return_type", line, "'" + name + "' : GLSL ES 1.00 functions cannot return arrays");
        if (Builtins::isBuiltinName(name) && !existing) {
            bool redefines = es3();  // ES 3.00: no function can share a built-in's name; ES 1.00: only the same signature
            if (!es3()) {
                std::vector<Type> ts;
                for (Symbol* p : params) ts.push_back(p->type);
                BuiltinCtx ctx{stage_, pp_.version, &pp_.extensions};
                std::string why;
                redefines = Builtins::resolve(name, ts, ctx, why).has_value();
            }
            if (redefines) strictAt("redefine_builtin", line, "'" + name + "' : cannot redefine a built-in function");
        }
        if (!existing && (scopes_[1].vars.count(name) || scopes_[1].structs.count(name) || blockNames_.count(name)))
            strictAt("name_conflict", line, "'" + name + "' : name already used");
        if (existing && isP(";") && pp_.version < 300)
            strictAt("function_redeclared", line, "'" + name + "' : function declared twice");
        Function* f = existing;
        if (!f) {
            fn->params = params;
            fn->id = int(sh_->fnStore.size()) + 1;
            f = fn.get();
            sh_->fnStore.push_back(std::move(fn));
            sh_->functions.push_back(f);
            functions_[name].push_back(f);
        }
        if (acceptP(";")) return;  // prototype
        if (f->body) errAt(line, "'" + name + "' : function redefinition");
        if (name == "main" && (!params.empty() || ret.base != Base::Void)) errAt(line, "main() must be void main()");
        f->params = params;  // names come from the definition
        pushScope();
        for (Symbol* p : params)
            if (!p->name.empty()) declare(p, line);
        Function* saved = cur_;
        cur_ = f;
        f->body = compound(false);
        cur_ = saved;
        popScope();
    }

    static bool structHasArray(const StructDef* st) {
        for (const Member& m : st->members)
            if (m.type.isArray() || (m.type.base == Base::Struct && structHasArray(m.type.st))) return true;
        return false;
    }

    // ---- statements -----------------------------------------------------------------------------
    StmtP mk(SK k, int line) {
        auto s = std::make_unique<Stmt>();
        s->k = k, s->line = line;
        return s;
    }

    StmtP compound(bool scope = true) {
        int line = peek().line;
        expectP("{");
        auto s = mk(SK::Block, line);
        s->newScope = scope;
        if (scope) pushScope();
        while (!acceptP("}")) {
            if (peek().kind == Tok::End) err("unexpected end of shader");
            s->list.push_back(statement());
        }
        if (scope) popScope();
        return s;
    }

    bool atDeclaration() {
        const Token& t = peek();
        if (t.kind != Tok::Ident) return false;
        if (t.text == "const" || t.text == "struct" || precisionWord(t.text) || t.text == "precision" || t.text == "invariant")
            return true;
        if (!atTypeStart()) return false;
        // `vec4(...)` / `S(...)` at statement start is an expression (constructor).
        size_t k = 1;
        while (isP("[", k)) {  // T[N] name / T[N](...)
            int depth = 0;
            for (;; ++k) {
                if (isP("[", k)) ++depth;
                else if (isP("]", k) && --depth == 0) break;
                if (peek(k).kind == Tok::End) return false;
            }
            ++k;
        }
        return peek(k).kind == Tok::Ident;
    }

    StmtP statement() {
        int line = peek().line;
        if (isP("{")) return compound();
        if (acceptP(";")) return mk(SK::Empty, line);
        if (isId("if")) {
            next();
            expectP("(");
            auto s = mk(SK::If, line);
            s->e = expression();
            if (s->e->type != Type::make(Base::Bool)) errAt(line, "if condition must be a scalar bool");
            expectP(")");
            s->then = scopedStatement();
            if (isId("else")) {
                next();
                s->els = scopedStatement();
            }
            return s;
        }
        if (isId("while")) {
            next();
            expectP("(");
            auto s = mk(SK::While, line);
            pushScope();
            s->e = condition();
            expectP(")");
            ++loops_;
            s->body = statementNoScope();
            --loops_;
            popScope();
            return s;
        }
        if (isId("do")) {
            next();
            auto s = mk(SK::Do, line);
            ++loops_;
            s->body = scopedStatement();
            --loops_;
            if (!isId("while")) err("expected 'while'");
            next();
            expectP("(");
            s->e = expression();
            if (s->e->type != Type::make(Base::Bool)) errAt(line, "loop condition must be a scalar bool");
            expectP(")");
            expectP(";");
            return s;
        }
        if (isId("for")) {
            next();
            expectP("(");
            auto s = mk(SK::For, line);
            pushScope();
            if (acceptP(";")) s->init = mk(SK::Empty, line);
            else s->init = simpleStatement();
            if (!isP(";")) s->e = condition();
            expectP(";");
            if (!isP(")")) s->inc = expression();
            expectP(")");
            ++loops_;
            s->body = statementNoScope();
            --loops_;
            popScope();
            return s;
        }
        if (isId("switch")) {
            if (!es3()) err("switch needs GLSL ES 3.00");
            next();
            expectP("(");
            auto s = mk(SK::Switch, line);
            s->e = expression();
            if (!s->e->type.isIntegral() || !s->e->type.isScalar()) errAt(line, "switch selector must be int or uint");
            expectP(")");
            expectP("{");
            pushScope();
            ++switches_;
            std::set<int64_t> caseValues;
            bool sawDefault = false;
            while (!acceptP("}")) {
                int l = peek().line;
                if (isId("case")) {
                    next();
                    auto c = mk(SK::Case, l);
                    c->e = conditional();
                    if (!c->e->hasConst || c->e->type != s->e->type) errAt(l, "case label must be a constant of the selector's type");
                    if (!caseValues.insert(c->e->constI).second) errAt(l, "duplicate case label");
                    expectP(":");
                    s->list.push_back(std::move(c));
                } else if (isId("default")) {
                    next();
                    if (sawDefault) errAt(l, "duplicate default label");
                    sawDefault = true;
                    expectP(":");
                    s->list.push_back(mk(SK::Default, l));
                } else {
                    if (s->list.empty()) strictAt("switch_statement_before_label", l, "a statement before the first case label of a switch");
                    s->list.push_back(statement());
                }
            }
            if (!s->list.empty() && (s->list.back()->k == SK::Case || s->list.back()->k == SK::Default))
                strictAt("switch_empty_label", peek().line, "the last label of a switch needs a statement after it");
            --switches_;
            popScope();
            return s;
        }
        if (isId("return")) {
            next();
            auto s = mk(SK::Return, line);
            if (!isP(";")) s->e = expression();
            Type got = s->e ? s->e->type : Type::make(Base::Void);
            if (got != cur_->ret) errAt(line, "'return' : type '" + got.str() + "' does not match '" + cur_->ret.str() + "'");
            expectP(";");
            return s;
        }
        if (isId("break")) {
            next();
            if (!loops_ && !switches_) errAt(line, "'break' outside a loop or switch");
            expectP(";");
            return mk(SK::Break, line);
        }
        if (isId("continue")) {
            next();
            if (!loops_) errAt(line, "'continue' outside a loop");
            expectP(";");
            return mk(SK::Continue, line);
        }
        if (isId("discard")) {
            next();
            if (stage_ != Stage::Fragment) errAt(line, "'discard' only in fragment shaders");
            expectP(";");
            return mk(SK::Discard, line);
        }
        return simpleStatement();
    }

    StmtP scopedStatement() {
        pushScope();
        StmtP s = statement();
        popScope();
        return s;
    }
    StmtP statementNoScope() {
        if (isP("{")) return compound(false);
        return statement();
    }

    // while/for condition: an expression or `type name = init`.
    ExprP condition() {
        if (atDeclaration()) {
            int line = peek().line;
            Type t = typeSpecifier();
            Symbol* s = sh_->newSymbol();
            s->name = expectIdent(), s->type = t, s->storage = Storage::Local;
            expectP("=");
            ExprP init = assignment();
            if (init->type != t) errAt(line, "condition initialiser type mismatch");
            declare(s, line);
            // Represent as an assignment expression to the new local.
            auto v = std::make_unique<Expr>();
            v->k = EK::Var, v->sym = s, v->type = t, v->line = line;
            auto a = std::make_unique<Expr>();
            a->k = EK::Assign, a->op = "=", a->type = t, a->line = line;
            a->a.push_back(std::move(v));
            a->a.push_back(std::move(init));
            return a;
        }
        ExprP e = expression();
        if (e->type != Type::make(Base::Bool)) errAt(e->line, "loop condition must be a scalar bool");
        return e;
    }

    StmtP simpleStatement() {
        int line = peek().line;
        if (isId("precision")) {
            precisionStatement();
            return mk(SK::Empty, line);
        }
        if (atDeclaration()) {
            Qual q = qualifiers();
            Type base = typeSpecifier();
            const uint8_t basePrec = tsPrec_;
            auto s = mk(SK::Decl, line);
            if (acceptP(";")) return mk(SK::Empty, line);  // local struct declaration
            for (;;) {
                Symbol* v = sh_->newSymbol();
                v->name = expectIdent();
                v->type = base;
                arraySuffix(v->type);
                needPrecision(effectivePrecision(q.prec, basePrec, base), base, line, v->name);
                v->storage = q.isConst ? Storage::Const : Storage::Local;
                if (v->type.containsOpaque()) errAt(line, "samplers, images and atomic counters cannot be local variables");
                ExprP init;
                if (acceptP("=")) {
                    init = assignment();
                    resolveImplicitSize(v->type, init->type, line);
                    if (init->type != v->type)
                        errAt(line, "'=' : cannot convert from '" + init->type.str() + "' to '" + v->type.str() + "'");
                    if (q.isConst && init->hasConst) v->hasConst = true, v->constF = init->constF, v->constI = init->constI;
                    if (q.isConst) {
                        v->constant = isConstExpr(*init);
                        if (!v->constant) strictAt("const_initializer", line, "'" + v->name + "' : the initializer of a const must be a constant expression");
                    }
                } else if (q.isConst) {
                    errAt(line, "const variables need an initialiser");
                } else {
                    for (int d : v->type.dims)
                        if (d == 0) errAt(line, "'" + v->name + "' : implicitly sized array without initialiser");
                }
                declare(v, line);  // after the initialiser: `float x = x;` sees the outer x
                s->decls.emplace_back(v, std::move(init));
                if (!acceptP(",")) break;
            }
            expectP(";");
            return s;
        }
        auto s = mk(SK::Expr, line);
        s->e = expression();
        expectP(";");
        return s;
    }

    // ---- expressions ----------------------------------------------------------------------------
    ExprP mkE(EK k, const Type& t, int line) {
        auto e = std::make_unique<Expr>();
        e->k = k, e->type = t, e->line = line;
        return e;
    }

    ExprP expression() {
        ExprP e = assignment();
        while (isP(",")) {
            int line = next().line;
            ExprP r = assignment();
            ExprP c = mkE(EK::Comma, r->type, line);
            c->a.push_back(std::move(e));
            c->a.push_back(std::move(r));
            e = std::move(c);
        }
        return e;
    }

    // A constant expression (GLSL ES 3.00 §4.3.2): literals, `const` variables initialised with one, constructors,
    // operators and built-in calls (but not texture / derivative / image / atomic functions) on constant operands.
    // Parameters, uniforms, user function calls, assignments and the comma operator are not.
    bool isConstExpr(const Expr& e) const {
        switch (e.k) {
            case EK::IntLit: case EK::UintLit: case EK::FloatLit: case EK::BoolLit: return true;
            case EK::Var:
                return e.sym && ((e.sym->storage == Storage::Const && e.sym->constant) ||
                                 (e.sym->builtin == BuiltinVar::WorkGroupSize && sh_->hasLocalSize));
            case EK::Length: return e.a.empty() || isConstExpr(*e.a[0]);
            case EK::Call: case EK::Assign: case EK::Comma: case EK::PreInc: case EK::PreDec: case EK::PostInc: case EK::PostDec:
                return false;
            case EK::Builtin: {
                Builtins::TexCall tc;
                Builtins::ImageCall ic;
                if (Builtins::textureCall(e.op, tc) || Builtins::imageCall(e.op, ic) || Builtins::isAtomicMemoryFunction(e.op) ||
                    e.op == "dFdx" || e.op == "dFdy" || e.op == "fwidth" || e.op == "barrier" || e.op.rfind("memoryBarrier", 0) == 0 ||
                    e.op == "groupMemoryBarrier" || e.op == "interpolateAtCentroid" || e.op == "interpolateAtSample" ||
                    e.op == "interpolateAtOffset")
                    return false;
                break;
            }
            default: break;
        }
        for (const ExprP& a : e.a)
            if (!isConstExpr(*a)) return false;
        return true;
    }

    bool isLValue(const Expr& e, std::string& why) {
        switch (e.k) {
            case EK::Var: {
                const Symbol* s = e.sym;
                if (s->storage == Storage::Const || s->storage == Storage::Uniform || s->storage == Storage::In ||
                    s->storage == Storage::BlockMember || s->storage == Storage::BlockInstance) {
                    why = "l-value required (can't modify '" + s->name + "')";
                    return false;
                }
                if (s->storage == Storage::Builtin &&
                    (s->builtin == BuiltinVar::VertexID || s->builtin == BuiltinVar::InstanceID || s->builtin == BuiltinVar::FragCoord ||
                     s->builtin == BuiltinVar::FrontFacing || s->builtin == BuiltinVar::PointCoord || s->builtin == BuiltinVar::ViewID ||
                     s->builtin >= BuiltinVar::LocalInvocationID)) {
                    why = "l-value required (can't modify '" + s->name + "')";
                    return false;
                }
                if (s->storage == Storage::Param && s->constParam)
                    strictAt("const_param_write", peek().line, "l-value required (can't modify the const parameter '" + s->name + "')");
                if (s->storage == Storage::Param && s->dir == ParamDir::In) return true;  // copies
                return true;
            }
            case EK::Field: {
                // Members of a shader storage block are writable unless readonly.
                const Expr* inst = e.a[0].get();
                if (inst->k == EK::Index) inst = inst->a[0].get();
                if (inst->k == EK::Var && inst->sym->storage == Storage::BlockInstance && inst->sym->block->buffer) {
                    if (inst->sym->block->members[size_t(e.field)].mem & kReadonly) {
                        why = "l-value required (can't modify readonly '" + inst->sym->block->members[size_t(e.field)].name + "')";
                        return false;
                    }
                    return true;
                }
                return isLValue(*e.a[0], why);
            }
            case EK::Index: return isLValue(*e.a[0], why);
            case EK::Swizzle: {
                for (int i = 0; i < e.nswz; ++i)
                    for (int j = i + 1; j < e.nswz; ++j)
                        if (e.swz[i] == e.swz[j]) {
                            why = "l-value swizzle with repeated components";
                            return false;
                        }
                return isLValue(*e.a[0], why);
            }
            default:
                why = "l-value required";
                return false;
        }
    }

    ExprP assignment() {
        ExprP lhs = conditional();
        static const char* const ops[] = {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|="};
        for (const char* op : ops) {
            if (!isP(op)) continue;
            int line = next().line;
            std::string why;
            if (!isLValue(*lhs, why)) errAt(line, "'" + std::string(op) + "' : " + why);
            ExprP rhs = assignment();
            if (std::string(op) == "=") {
                if (lhs->type != rhs->type)
                    errAt(line, "'=' : cannot convert from '" + rhs->type.str() + "' to '" + lhs->type.str() + "'");
                if (lhs->type.containsOpaque()) errAt(line, "cannot assign samplers, images or atomic counters");
            } else {
                std::string bop(op, std::char_traits<char>::length(op) - 1);
                Type rt = binaryType(bop, lhs->type, rhs->type, line);
                if (rt != lhs->type) errAt(line, "'" + std::string(op) + "' : result type mismatch");
            }
            ExprP a = mkE(EK::Assign, lhs->type, line);
            a->op = op;
            a->a.push_back(std::move(lhs));
            a->a.push_back(std::move(rhs));
            return a;
        }
        return lhs;
    }

    ExprP conditional() {
        ExprP c = binary(0);
        if (!isP("?")) return c;
        int line = next().line;
        if (c->type != Type::make(Base::Bool)) errAt(line, "'?:' : condition must be a scalar bool");
        ExprP a = assignment();
        expectP(":");
        ExprP b = assignment();
        if (a->type != b->type) errAt(line, "'?:' : operand types differ");
        ExprP t = mkE(EK::Ternary, a->type, line);
        if (c->hasConst && a->hasConst && b->hasConst) {
            const Expr& s = c->constI ? *a : *b;
            t->hasConst = true, t->constF = s.constF, t->constI = s.constI;
        }
        t->a.push_back(std::move(c));
        t->a.push_back(std::move(a));
        t->a.push_back(std::move(b));
        return t;
    }

    static int precedence(const std::string& op) {
        if (op == "||") return 1;
        if (op == "^^") return 2;
        if (op == "&&") return 3;
        if (op == "|") return 4;
        if (op == "^") return 5;
        if (op == "&") return 6;
        if (op == "==" || op == "!=") return 7;
        if (op == "<" || op == ">" || op == "<=" || op == ">=") return 8;
        if (op == "<<" || op == ">>") return 9;
        if (op == "+" || op == "-") return 10;
        if (op == "*" || op == "/" || op == "%") return 11;
        return -1;
    }

    Type binaryType(const std::string& op, const Type& l, const Type& r, int line) {
        auto bad = [&]() -> Type {
            errAt(line, "'" + op + "' : wrong operand types (" + l.str() + " and " + r.str() + ")");
        };
        if (l.isArray() || r.isArray() || l.base == Base::Struct || r.base == Base::Struct || l.base == Base::Sampler ||
            r.base == Base::Sampler || l.base == Base::Void || r.base == Base::Void) {
            if ((op == "==" || op == "!=") && l == r && !l.containsOpaque() && es3()) return Type::make(Base::Bool);
            if ((op == "==" || op == "!=") && l == r && !l.isArray() && l.base == Base::Struct) return Type::make(Base::Bool);
            return bad();
        }
        if (op == "&&" || op == "||" || op == "^^") {
            if (l != Type::make(Base::Bool) || r != Type::make(Base::Bool)) return bad();
            return l;
        }
        if (op == "==" || op == "!=") {
            if (l != r) return bad();
            return Type::make(Base::Bool);
        }
        if (op == "<" || op == ">" || op == "<=" || op == ">=") {
            if (!l.isScalar() || l != r || !l.isNumeric()) return bad();
            return Type::make(Base::Bool);
        }
        if ((op == "<<" || op == ">>") && es3() && l.isIntegral() && r.isIntegral() && !l.cols && !r.cols) {
            if (r.vec != 1 && r.vec != l.vec) return bad();
            return l;  // int << uint is allowed: the result has the left operand's type
        }
        if (l.base != r.base || !l.isNumeric()) return bad();
        if (op == "%" || op == "&" || op == "|" || op == "^" || op == "<<" || op == ">>") {
            if (!es3() || !l.isIntegral() || l.cols || r.cols) return bad();
            if (op == "<<" || op == ">>") {
                if (r.vec != 1 && r.vec != l.vec) return bad();
                return l;
            }
            if (l.vec == r.vec) return l;
            if (l.vec == 1) return r;
            if (r.vec == 1) return l;
            return bad();
        }
        // + - * /
        if (l.cols || r.cols) {
            if (op == "*") {
                if (l.cols && r.cols) {
                    if (l.cols != r.vec) return bad();
                    return Type::make(Base::Float, l.vec, r.cols);
                }
                if (l.cols && r.vec > 1) {
                    if (l.cols != r.vec) return bad();
                    return Type::make(Base::Float, l.vec);
                }
                if (r.cols && l.vec > 1 && !l.cols) {
                    if (r.vec != l.vec) return bad();
                    return Type::make(Base::Float, r.cols);
                }
            }
            if (l.cols && r.cols) {
                if (l != r) return bad();
                return l;
            }
            if (l.cols && r.isScalar()) return l;
            if (r.cols && l.isScalar()) return r;
            return bad();
        }
        if (l.vec == r.vec) return l;
        if (l.vec == 1) return r;
        if (r.vec == 1) return l;
        return bad();
    }

    static void fold(Expr& e, const std::string& op, const Expr& l, const Expr& r) {
        if (!l.hasConst || !r.hasConst || !e.type.isScalar()) return;
        const bool isF = l.type.base == Base::Float;
        const bool isU = l.type.base == Base::Uint;
        double a = l.constF, b = r.constF;
        int64_t x = l.constI, y = r.constI;
        auto setI = [&](int64_t v) {
            if (isU) v = int64_t(uint32_t(v));
            else v = int64_t(int32_t(v));
            e.hasConst = true, e.constI = v, e.constF = double(v);
        };
        auto setF = [&](double v) {
            e.hasConst = true, e.constF = double(float(v)), e.constI = int64_t(v);
        };
        auto setB = [&](bool v) { e.hasConst = true, e.constI = v, e.constF = v; };
        if (op == "+") isF ? setF(a + b) : setI(x + y);
        else if (op == "-") isF ? setF(a - b) : setI(x - y);
        else if (op == "*") isF ? setF(a * b) : setI(x * y);
        else if (op == "/") {
            if (isF) setF(a / b);
            else if (y != 0) setI(isU ? int64_t(uint32_t(x) / uint32_t(y)) : x / y);
        } else if (op == "%") {
            if (y != 0) setI(isU ? int64_t(uint32_t(x) % uint32_t(y)) : x % y);
        } else if (op == "&") setI(x & y);
        else if (op == "|") setI(x | y);
        else if (op == "^") setI(x ^ y);
        else if (op == "<<") setI(y >= 0 && y < 32 ? x << y : 0);
        else if (op == ">>") setI(y >= 0 && y < 32 ? (isU ? int64_t(uint32_t(x) >> y) : x >> y) : 0);
        else if (op == "<") setB(isF ? a < b : isU ? uint32_t(x) < uint32_t(y) : x < y);
        else if (op == ">") setB(isF ? a > b : isU ? uint32_t(x) > uint32_t(y) : x > y);
        else if (op == "<=") setB(isF ? a <= b : isU ? uint32_t(x) <= uint32_t(y) : x <= y);
        else if (op == ">=") setB(isF ? a >= b : isU ? uint32_t(x) >= uint32_t(y) : x >= y);
        else if (op == "==") setB(isF ? a == b : x == y);
        else if (op == "!=") setB(isF ? a != b : x != y);
        else if (op == "&&") setB(x && y);
        else if (op == "||") setB(x || y);
        else if (op == "^^") setB(bool(x) != bool(y));
    }

    ExprP binary(int minPrec) {
        ExprP l = unary();
        for (;;) {
            if (peek().kind != Tok::Punct) break;
            int p = precedence(peek().text);
            if (p < 0 || p < minPrec) break;
            Token opTok = next();
            ExprP r = binary(p + 1);
            Type t = binaryType(opTok.text, l->type, r->type, opTok.line);
            ExprP b = mkE(EK::Binary, t, opTok.line);
            b->op = opTok.text;
            fold(*b, opTok.text, *l, *r);
            b->a.push_back(std::move(l));
            b->a.push_back(std::move(r));
            l = std::move(b);
        }
        return l;
    }

    ExprP unary() {
        int line = peek().line;
        for (const char* op : {"++", "--"}) {
            if (!isP(op)) continue;
            next();
            ExprP e = unary();
            std::string why;
            if (!isLValue(*e, why)) errAt(line, "'" + std::string(op) + "' : " + why);
            if (!e->type.isNumeric() || e->type.isArray()) errAt(line, "'" + std::string(op) + "' : wrong operand type");
            ExprP u = mkE(op[0] == '+' ? EK::PreInc : EK::PreDec, e->type, line);
            u->a.push_back(std::move(e));
            return u;
        }
        for (const char* op : {"+", "-", "!", "~"}) {
            if (!isP(op)) continue;
            next();
            ExprP e = unary();
            const Type& t = e->type;
            std::string o = op;
            if (o == "!" && t != Type::make(Base::Bool)) errAt(line, "'!' : wrong operand type");
            if ((o == "+" || o == "-") && (!t.isNumeric() || t.isArray())) errAt(line, "'" + o + "' : wrong operand type");
            if (o == "~" && (!es3() || !t.isIntegral() || t.cols || t.isArray())) errAt(line, "'~' : wrong operand type");
            if (o == "+") return e;
            ExprP u = mkE(EK::Unary, t, line);
            u->op = o;
            if (e->hasConst) {
                u->hasConst = true;
                if (o == "-") {
                    u->constF = -e->constF;
                    u->constI = t.base == Base::Uint ? int64_t(uint32_t(-e->constI)) : int64_t(int32_t(-e->constI));
                    if (t.base != Base::Float) u->constF = double(u->constI);
                } else if (o == "!") u->constI = !e->constI, u->constF = double(u->constI);
                else u->constI = t.base == Base::Uint ? int64_t(uint32_t(~e->constI)) : int64_t(int32_t(~e->constI)), u->constF = double(u->constI);
            }
            u->a.push_back(std::move(e));
            return u;
        }
        return postfix(primary());
    }

    static int swizzleIndex(char c, int& set) {
        const char* sets[3] = {"xyzw", "rgba", "stpq"};
        for (int s = 0; s < 3; ++s)
            for (int i = 0; i < 4; ++i)
                if (sets[s][i] == c) {
                    if (set >= 0 && set != s) return -2;
                    set = s;
                    return i;
                }
        return -1;
    }

    ExprP postfix(ExprP e) {
        for (;;) {
            int line = peek().line;
            if (acceptP("[")) {
                ExprP idx = expression();
                expectP("]");
                if (!idx->type.isIntegral() || !idx->type.isScalar()) errAt(line, "array index must be an integer scalar");
                Type t;
                const Type& bt = e->type;
                int limit = 0;
                if (bt.isArray()) t = bt.element(), limit = bt.dims[0];
                else if (bt.cols) t = bt.columnType(), limit = bt.cols;
                else if (bt.vec > 1 && bt.numericOrBool()) t = bt.scalarType(), limit = bt.vec;
                else errAt(line, "'[' : not an array, matrix or vector");
                if (idx->hasConst && (idx->constI < 0 || (limit > 0 && idx->constI >= limit)))
                    errAt(line, "'[' : index " + std::to_string(idx->constI) + " out of range");
                // ES 1.00: gl_FragData has gl_MaxDrawBuffers elements, which is 1 without GL_EXT_draw_buffers.
                if (e->k == EK::Var && e->sym->builtin == BuiltinVar::FragData && !pp_.extensions.count("GL_EXT_draw_buffers") &&
                    !(idx->hasConst && idx->constI == 0))
                    strictAt("fragdata_index", line, "'gl_FragData' : only element 0 exists without GL_EXT_draw_buffers");
                if (bt.isArray() && bt.containsSampler() && !idx->hasConst && !es3())
                    errAt(line, "sampler arrays need constant indices in GLSL ES 1.00");
                ExprP x = mkE(EK::Index, t, line);
                x->a.push_back(std::move(e));
                x->a.push_back(std::move(idx));
                e = std::move(x);
                continue;
            }
            if (acceptP(".")) {
                std::string f = expectIdent();
                if (f == "length" && isP("(")) {
                    next();
                    expectP(")");
                    if (!e->type.isArray()) errAt(line, "length() on a non-array");
                    if (!es3()) errAt(line, "length() needs GLSL ES 3.00");
                    ExprP l = mkE(EK::Length, Type::make(Base::Int), line);
                    if (e->type.dims[0] == 0) {
                        l->a.push_back(std::move(e));  // unsized shader storage array: queried at run time
                    } else {
                        l->hasConst = true, l->constI = e->type.dims[0], l->constF = double(l->constI);
                    }
                    e = std::move(l);
                    continue;
                }
                const Type& bt = e->type;
                if (bt.base == Base::Struct && !bt.isArray()) {
                    int idx = -1;
                    for (size_t i = 0; i < bt.st->members.size(); ++i)
                        if (bt.st->members[i].name == f) idx = int(i);
                    if (idx < 0) errAt(line, "'" + f + "' : no such field in '" + bt.str() + "'");
                    ExprP x = mkE(EK::Field, bt.st->members[size_t(idx)].type, line);
                    x->field = idx;
                    x->a.push_back(std::move(e));
                    e = std::move(x);
                    continue;
                }
                if (e->k == EK::Var && e->sym->storage == Storage::BlockInstance) {
                    UniformBlockDecl* b = e->sym->block;
                    if (!b->dims.empty()) errAt(line, "block array must be indexed");
                    e = blockField(std::move(e), b, f, line);
                    continue;
                }
                if (e->k == EK::Index && e->a[0]->k == EK::Var && e->a[0]->sym->storage == Storage::BlockInstance) {
                    UniformBlockDecl* b = e->a[0]->sym->block;  // before the move: the evaluation order of the arguments is unspecified
                    e = blockField(std::move(e), b, f, line);
                    continue;
                }
                if (!bt.isArray() && !bt.cols && bt.numericOrBool() && f.size() <= 4) {
                    ExprP x = mkE(EK::Swizzle, Type::make(bt.base, int(f.size())), line);
                    int set = -1;
                    for (size_t i = 0; i < f.size(); ++i) {
                        int c = swizzleIndex(f[i], set);
                        if (c < 0 || c >= bt.vec) errAt(line, "'" + f + "' : invalid swizzle");
                        x->swz[i] = uint8_t(c);
                    }
                    x->nswz = uint8_t(f.size());
                    if (e->hasConst && f.size() == 1) x->hasConst = true, x->constF = e->constF, x->constI = e->constI;
                    // gl_WorkGroupSize.x is a constant expression once layout(local_size_*) is declared (ES 3.10).
                    if (e->k == EK::Var && e->sym->builtin == BuiltinVar::WorkGroupSize && f.size() == 1 && sh_->hasLocalSize)
                        x->hasConst = true, x->constI = sh_->localSize[x->swz[0]], x->constF = double(x->constI);
                    x->a.push_back(std::move(e));
                    e = std::move(x);
                    continue;
                }
                errAt(line, "'" + f + "' : field selection on '" + bt.str() + "'");
            }
            for (const char* op : {"++", "--"}) {
                if (!isP(op)) continue;
                next();
                std::string why;
                if (!isLValue(*e, why)) errAt(line, "'" + std::string(op) + "' : " + why);
                if (!e->type.isNumeric() || e->type.isArray()) errAt(line, "'" + std::string(op) + "' : wrong operand type");
                ExprP x = mkE(op[0] == '+' ? EK::PostInc : EK::PostDec, e->type, line);
                x->a.push_back(std::move(e));
                e = std::move(x);
                goto again;
            }
            return e;
        again:;
        }
    }

    // An atomic* operand: a scalar of a shader storage block or a shared variable.
    static bool atomicTarget(const Expr& e) {
        const Expr* x = &e;
        while (x->k == EK::Index || x->k == EK::Field) {
            if (x->k == EK::Field && x->a[0]->k == EK::Var && x->a[0]->sym->storage == Storage::BlockInstance)
                return x->a[0]->sym->block->buffer;
            if (x->k == EK::Field && x->a[0]->k == EK::Index && x->a[0]->a[0]->k == EK::Var &&
                x->a[0]->a[0]->sym->storage == Storage::BlockInstance)
                return x->a[0]->a[0]->sym->block->buffer;
            x = x->a[0].get();
        }
        return x->k == EK::Var && x->sym->storage == Storage::Shared;
    }

    ExprP blockField(ExprP inst, UniformBlockDecl* b, const std::string& f, int line) {
        for (size_t i = 0; i < b->members.size(); ++i)
            if (b->members[i].name == f) {
                ExprP x = mkE(EK::Field, b->members[i].type, line);
                x->field = int(i);
                x->a.push_back(std::move(inst));
                return x;
            }
        errAt(line, "'" + f + "' : no such member in block '" + b->name + "'");
    }

    ExprP primary() {
        const Token t = peek();
        int line = t.line;
        switch (t.kind) {
            case Tok::IntLit: {
                next();
                if (t.ival > 0xFFFFFFFFull) errAt(line, "integer constant overflow");
                ExprP e = mkE(EK::IntLit, Type::make(Base::Int), line);
                e->i = t.ival;
                e->hasConst = true, e->constI = int64_t(int32_t(uint32_t(t.ival))), e->constF = double(e->constI);
                return e;
            }
            case Tok::UintLit: {
                next();
                if (!es3()) errAt(line, "unsigned literals need GLSL ES 3.00");
                ExprP e = mkE(EK::UintLit, Type::make(Base::Uint), line);
                e->i = uint32_t(t.ival);
                e->hasConst = true, e->constI = int64_t(uint32_t(t.ival)), e->constF = double(e->constI);
                return e;
            }
            case Tok::FloatLit: {
                next();
                ExprP e = mkE(EK::FloatLit, Type::make(Base::Float), line);
                e->f = double(float(t.fval));
                e->hasConst = true, e->constF = e->f, e->constI = int64_t(e->f);
                return e;
            }
            case Tok::BoolLit: {
                next();
                ExprP e = mkE(EK::BoolLit, Type::make(Base::Bool), line);
                e->i = t.ival;
                e->hasConst = true, e->constI = int64_t(t.ival), e->constF = double(t.ival);
                return e;
            }
            default: break;
        }
        if (acceptP("(")) {
            ExprP e = expression();
            expectP(")");
            return e;
        }
        if (t.kind != Tok::Ident) err("syntax error, unexpected '" + t.text + "'");
        // Constructors: type names (incl. arrays T[N](...)) and struct names.
        if (atTypeStart() && !findVar(t.text)) {
            if (precisionWord(t.text)) strictAt("precision_in_expression", line, "a precision qualifier cannot start an expression");
            Type ty = typeSpecifier();
            if (!isP("(")) errAt(line, "'" + t.text + "' : unexpected type name");
            return constructor(ty, line);
        }
        next();
        if (isP("(")) return call(t.text, line);
        Symbol* s = findVar(t.text);
        if (!s) errAt(line, "'" + t.text + "' : undeclared identifier");
        if (s->builtin == BuiltinVar::ViewID && !pp_.extensions.count("GL_OVR_multiview") && !pp_.extensions.count("GL_OVR_multiview2"))
            errAt(line, "'gl_ViewID_OVR' needs GL_OVR_multiview");
        if (s->builtin == BuiltinVar::FragDepth && !es3() && !pp_.extensions.count("GL_EXT_frag_depth"))
            errAt(line, "'gl_FragDepthEXT' needs GL_EXT_frag_depth");
        if (s->storage == Storage::Builtin && std::find(sh_->builtins.begin(), sh_->builtins.end(), s) == sh_->builtins.end())
            sh_->builtins.push_back(s);
        s->used = true;
        if (s->builtin == BuiltinVar::FragColor) usesFragColor_ = true;
        if (s->builtin == BuiltinVar::FragData) usesFragData_ = true;
        if (usesFragColor_ && usesFragData_)
            strictAt("fragcolor_and_fragdata", line, "gl_FragColor and gl_FragData cannot both be written");
        if (s->block) s->block->instanceSym->used = true;
        ExprP e = mkE(EK::Var, s->type, line);
        e->sym = s;
        if (s->storage == Storage::BlockMember) {
            // Unnamed block member: rewrite as instance.member.
            ExprP inst = mkE(EK::Var, Type{}, line);
            inst->sym = s->block->instanceSym;
            ExprP f = mkE(EK::Field, s->type, line);
            f->field = s->member;
            f->a.push_back(std::move(inst));
            return f;
        }
        if (s->hasConst) e->hasConst = true, e->constF = s->constF, e->constI = s->constI;
        return e;
    }

    std::vector<ExprP> arguments() {
        std::vector<ExprP> args;
        expectP("(");
        if (isId("void") && isP(")", 1)) next();
        while (!isP(")")) {
            args.push_back(assignment());
            if (!acceptP(",")) break;
        }
        expectP(")");
        return args;
    }

    ExprP constructor(Type t, int line) {
        std::vector<ExprP> args = arguments();
        if (args.empty()) errAt(line, "constructor needs arguments");
        if (t.isArray()) {
            if (!es3()) errAt(line, "array constructors need GLSL ES 3.00");
            if (t.dims[0] == 0) t.dims[0] = int(args.size());
            if (int(args.size()) != t.dims[0]) errAt(line, "array constructor: wrong number of elements");
            Type et = t.element();
            for (auto& a : args)
                if (a->type != et) errAt(line, "array constructor: element type mismatch");
        } else if (t.base == Base::Struct) {
            if (args.size() != t.st->members.size()) errAt(line, "'" + t.str() + "' : wrong number of constructor arguments");
            for (size_t i = 0; i < args.size(); ++i)
                if (args[i]->type != t.st->members[i].type) errAt(line, "'" + t.str() + "' : constructor argument type mismatch");
        } else {
            int have = 0;
            for (auto& a : args) {
                if (a->type.isArray() || !a->type.numericOrBool()) errAt(line, "'" + t.str() + "' : invalid constructor argument");
                have += a->type.components();
            }
            const int need = t.components();
            if (args.size() == 1 && args[0]->type.isScalar()) {
                // splat / diagonal
            } else if (t.cols && args.size() == 1 && args[0]->type.cols) {
                // matrix from matrix
            } else {
                if (t.cols)
                    for (auto& a : args)
                        if (a->type.cols) errAt(line, "matrix constructed from a matrix must be the only argument");
                if (have < need) errAt(line, "'" + t.str() + "' : not enough data provided for construction");
                // Every argument must contribute (the last may be partially used).
                int used = 0;
                for (size_t i = 0; i < args.size(); ++i) {
                    if (used >= need) errAt(line, "'" + t.str() + "' : too many arguments");
                    used += args[i]->type.components();
                }
            }
        }
        ExprP e = mkE(EK::Ctor, t, line);
        if (t.isScalar() && args.size() == 1 && args[0]->hasConst) {
            e->hasConst = true;
            const Expr& s = *args[0];
            switch (t.base) {
                case Base::Float: e->constF = s.type.base == Base::Float ? s.constF : s.type.base == Base::Uint ? double(uint32_t(s.constI)) : double(s.constI); e->constI = int64_t(e->constF); break;
                case Base::Int: e->constI = s.type.base == Base::Float ? int64_t(int32_t(s.constF)) : int64_t(int32_t(s.constI)); e->constF = double(e->constI); break;
                case Base::Uint: e->constI = s.type.base == Base::Float ? int64_t(uint32_t(int64_t(s.constF))) : int64_t(uint32_t(s.constI)); e->constF = double(e->constI); break;
                case Base::Bool: e->constI = s.type.base == Base::Float ? s.constF != 0 : s.constI != 0; e->constF = double(e->constI); break;
                default: e->hasConst = false;
            }
        }
        e->a = std::move(args);
        return e;
    }

    ExprP call(const std::string& name, int line) {
        // A struct name shadowed by nothing: handled in primary(). Here: user functions, builtins.
        std::vector<ExprP> args = arguments();
        std::vector<Type> types;
        for (auto& a : args) types.push_back(a->type);
        auto it = functions_.find(name);
        if (it != functions_.end()) {
            for (Function* f : it->second) {
                if (f->params.size() != args.size()) continue;
                bool ok = true;
                for (size_t i = 0; i < args.size(); ++i) {
                    Type pt = f->params[i]->type;
                    // An image parameter without a format accepts any format.
                    if (pt.base == Base::Image && pt.ifmt == 0 && types[i].base == Base::Image) pt.ifmt = types[i].ifmt;
                    ok &= pt == types[i];
                }
                if (!ok) continue;
                for (size_t i = 0; i < args.size(); ++i) {
                    std::string why;
                    if (f->params[i]->dir != ParamDir::In && !isLValue(*args[i], why))
                        errAt(line, "'" + name + "' : argument " + std::to_string(i + 1) + " must be an l-value");
                }
                ExprP e = mkE(EK::Call, f->ret, line);
                e->fn = f;
                if (cur_) callGraph_[cur_].insert(f);
                e->a = std::move(args);
                return e;
            }
        }
        BuiltinCtx ctx{stage_, pp_.version, &pp_.extensions, &lenient_};
        std::string why;
        std::optional<Type> rt = Builtins::resolve(name, types, ctx, why);
        if (rt) {
            ExprP e = mkE(EK::Builtin, *rt, line);
            e->op = name;
            Builtins::fold(*e, args);
            // out parameters of builtins (modf, frexp, *Extended): l-values.
            for (int outArg : Builtins::outArguments(name)) {
                std::string w;
                if (outArg < int(args.size()) && !isLValue(*args[size_t(outArg)], w)) errAt(line, "'" + name + "' : " + w);
            }
            if (Builtins::isAtomicMemoryFunction(name) && !atomicTarget(*args[0]))
                errAt(line, "'" + name + "' : the first argument must be a buffer or shared variable");
            e->a = std::move(args);
            return e;
        }
        if (!why.empty()) errAt(line, "'" + name + "' : " + why);
        if (it != functions_.end()) errAt(line, "'" + name + "' : no matching overloaded function found");
        errAt(line, "'" + name + "' : no matching overloaded function found");
    }

public:
    std::vector<std::string> lenient_;  // strict-mode rules let through
private:
    std::map<int, int> atomicNext_;  // atomic counter binding -> next free byte offset
    bool defaultBufferStd430_ = false, defaultBufferRowMajor_ = false, defaultUniformRowMajor_ = false;
    uint8_t defaultBufferPacking_ = 0, defaultUniformPacking_ = 0;  // layout(...) buffer; / layout(...) uniform;
    uint8_t declTypePrec_ = 0;                    // tsPrec_ of the declaration being parsed (it survives nested type parsing)
    uint8_t tsPrec_ = 0;                          // precision word typeSpecifier() consumed before the type (0 = none)
    std::map<std::string, uint8_t> defaultPrec_;  // `precision p T;` statements: type keyword -> precision
    std::set<std::string> blockNames_;            // interface block names (a namespace shared with functions and variables)
    std::map<Function*, std::set<Function*>> callGraph_;
    bool usesFragColor_ = false, usesFragData_ = false;

    Stage stage_;
    PpResult pp_;
    std::string& log_;
    std::shared_ptr<Shader> sh_;
    size_t pos_ = 0;
    Function* cur_ = nullptr;
    int loops_ = 0, switches_ = 0;
};

}  // namespace

ParseResult parse(Stage stage, const std::string& source) {
    ParseResult r;
    PpResult pp = runPreprocessor(source, stage, r.log);
    r.version = pp.version;
    r.lenient = pp.lenient;
    if (!pp.ok) return r;
    Parser p(stage, std::move(pp), r.log);
    r.shader = p.run();
    if (r.shader) r.shader->version = r.version;
    for (const std::string& l : p.lenient_)
        if (std::find(r.lenient.begin(), r.lenient.end(), l) == r.lenient.end()) r.lenient.push_back(l);
    return r;
}

}  // namespace vpglsl
