#include <bit>
#include <charconv>
#include <cstdlib>
#include <stdexcept>

#include "text_common.h"
#include "vpir/text.h"

namespace vpir {

namespace {

enum class Tok : uint8_t { End, Newline, Ident, Number, Value, Var, Punct };

struct Token {
    Tok kind = Tok::End;
    std::string_view text;
    unsigned line = 1;
};

struct ParseError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Lexer {
public:
    explicit Lexer(std::string_view s) : s_(s) {}

    Token next() {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ';') {
                while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
            } else if (c == ' ' || c == '\t' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ >= s_.size()) return {Tok::End, {}, line_};
        size_t start = pos_;
        char c = s_[pos_];
        auto take = [&](Tok kind) { return Token{kind, s_.substr(start, pos_ - start), line_}; };
        if (c == '\n') {
            ++pos_;
            return Token{Tok::Newline, s_.substr(start, 1), line_++};
        }
        if (c == '-' && pos_ + 1 < s_.size() && s_[pos_ + 1] == '>') {
            pos_ += 2;
            return take(Tok::Punct);
        }
        if (c == '%' || c == '$') {
            ++pos_;
            while (pos_ < s_.size() && isDigit(s_[pos_])) ++pos_;
            return take(c == '%' ? Tok::Value : Tok::Var);
        }
        if (isDigit(c) || c == '-' || c == '+') {
            ++pos_;
            // Covers integers, hex bits, decimals and exponents ("1e-5", "-inf" is not accepted).
            while (pos_ < s_.size() && (isIdent(s_[pos_]) || s_[pos_] == '.' ||
                                        ((s_[pos_] == '-' || s_[pos_] == '+') &&
                                         (s_[pos_ - 1] == 'e' || s_[pos_ - 1] == 'E'))))
                ++pos_;
            return take(Tok::Number);
        }
        if (isIdent(c)) {
            while (pos_ < s_.size() && isIdent(s_[pos_])) ++pos_;
            return take(Tok::Ident);
        }
        ++pos_;
        return take(Tok::Punct);
    }

private:
    static bool isDigit(char c) { return c >= '0' && c <= '9'; }
    static bool isIdent(char c) {
        return isDigit(c) || c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    std::string_view s_;
    size_t pos_ = 0;
    unsigned line_ = 1;
};

template <typename E>
bool lookupEnum(std::string_view name, E& out, const char* (*str)(E), E last) {
    for (unsigned i = 0; i <= unsigned(last); ++i)
        if (name == str(E(i))) {
            out = E(i);
            return true;
        }
    return false;
}

class Parser {
public:
    explicit Parser(std::string_view text) : lex_(text) { advance(); }

    Module run() {
        skipNewlines();
        expectWord("module");
        m_.stage = parseEnum<Stage>(toString, Stage::Mesh, "stage");
        endLine();
        while (skipNewlines(), tok_.kind != Tok::End) {
            std::string_view kw = expectIdent();
            if (kw == "workgroup_size") {
                for (uint32_t& w : m_.workgroupSize) w = parseU32();
            } else if (kw == "shared_bytes") {
                m_.sharedBytes = parseU32();
            } else if (kw == "wave_size") {
                m_.waveSize = parseU32();
            } else if (kw == "payload_bytes") {
                m_.payloadBytes = parseU32();
            } else if (kw == "private_bytes") {
                m_.privateBytes = parseU32();
            } else if (kw == "private_slot") {
                m_.privateSlots.push_back(parseU32());
            } else if (kw == "mesh") {
                m_.topology = parseEnum<Topology>(toString, Topology::Triangle, "topology");
                expectWord("max_vertices");
                m_.maxVertices = parseU32();
                expectWord("max_primitives");
                m_.maxPrimitives = parseU32();
            } else if (kw == "input" || kw == "output") {
                (kw == "input" ? m_.inputs : m_.outputs).push_back(parseIo());
            } else if (kw == "func") {
                parseFunction();
                continue;  // the closing brace already consumed the line
            } else {
                ResourceKind k;
                if (!lookupEnum<ResourceKind>(kw, k, toString, ResourceKind::StorageTexture))
                    fail("unknown declaration '" + std::string(kw) + "'");
                m_.resources.push_back(parseResource(k));
            }
            endLine();
        }
        return std::move(m_);
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw ParseError("line " + std::to_string(tok_.line) + ": " + msg);
    }

private:
    // ---- token helpers ----
    void advance() { tok_ = lex_.next(); }
    void skipNewlines() {
        while (tok_.kind == Tok::Newline) advance();
    }
    bool isPunct(char c) const { return tok_.kind == Tok::Punct && tok_.text[0] == c; }
    bool acceptPunct(char c) {
        if (!isPunct(c)) return false;
        advance();
        return true;
    }
    void expectPunct(char c) {
        if (!acceptPunct(c)) fail(std::string("expected '") + c + "'");
    }
    bool isWord(std::string_view w) const { return tok_.kind == Tok::Ident && tok_.text == w; }
    bool acceptWord(std::string_view w) {
        if (!isWord(w)) return false;
        advance();
        return true;
    }
    void expectWord(std::string_view w) {
        if (!acceptWord(w)) fail("expected '" + std::string(w) + "'");
    }
    std::string_view expectIdent() {
        if (tok_.kind != Tok::Ident) fail("expected identifier");
        std::string_view t = tok_.text;
        advance();
        return t;
    }
    void endLine() {
        if (tok_.kind != Tok::Newline && tok_.kind != Tok::End) fail("expected end of line");
        skipNewlines();
    }

    template <typename E>
    E parseEnum(const char* (*str)(E), E last, const char* what) {
        E e;
        if (tok_.kind != Tok::Ident || !lookupEnum<E>(tok_.text, e, str, last))
            fail(std::string("expected ") + what);
        advance();
        return e;
    }

    uint32_t parseU32() {
        if (tok_.kind != Tok::Number) fail("expected number");
        uint32_t v = 0;
        auto [p, ec] = std::from_chars(tok_.text.data(), tok_.text.data() + tok_.text.size(), v);
        if (ec != std::errc() || p != tok_.text.data() + tok_.text.size()) fail("bad integer");
        advance();
        return v;
    }

    uint32_t parseParenU32() {
        expectPunct('(');
        uint32_t v = parseU32();
        expectPunct(')');
        return v;
    }

    uint32_t parseId(Tok kind) {
        if (tok_.kind != kind || tok_.text.size() < 2) fail(kind == Tok::Value ? "expected %value" : "expected $var");
        uint32_t v = 0;
        auto [p, ec] = std::from_chars(tok_.text.data() + 1, tok_.text.data() + tok_.text.size(), v);
        if (ec != std::errc() || v == kNoValue) fail("bad id");
        advance();
        return v;
    }

    Scalar parseScalar() {
        static const std::pair<const char*, Scalar> names[] = {
            {"bool", Scalar::Bool}, {"i32", Scalar::I32}, {"u32", Scalar::U32},
            {"f16", Scalar::F16},   {"f32", Scalar::F32}};
        for (auto [n, s] : names)
            if (acceptWord(n)) return s;
        fail("expected scalar type");
    }

    Type parseType() {
        for (uint8_t w = 2; w <= 4; ++w) {
            if (acceptWord("vec" + std::to_string(w))) {
                expectPunct('<');
                Scalar s = parseScalar();
                expectPunct('>');
                return {s, w};
            }
        }
        return {parseScalar()};
    }

    // ---- declarations ----
    IoVar parseIo() {
        IoVar v;
        v.name = expectIdent();
        expectPunct(':');
        v.type = parseType();
        if (acceptWord("builtin")) {
            expectPunct('(');
            v.builtin = parseEnum<Builtin>(toString, Builtin::SubgroupSize, "builtin");
            if (v.builtin == Builtin::None) fail("expected builtin");
            expectPunct(')');
        } else {
            expectWord("location");
            v.location = parseParenU32();
            Interp in;
            if (tok_.kind == Tok::Ident && lookupEnum<Interp>(tok_.text, in, toString, Interp::Flat) &&
                in != Interp::Perspective) {
                v.interp = in;
                advance();
            }
            Sampling sa;
            if (tok_.kind == Tok::Ident &&
                lookupEnum<Sampling>(tok_.text, sa, toString, Sampling::Sample) &&
                sa != Sampling::Center) {
                v.sampling = sa;
                advance();
            }
        }
        v.perPrimitive = acceptWord("per_primitive");
        return v;
    }

    Resource parseResource(ResourceKind k) {
        Resource r;
        r.kind = k;
        r.name = expectIdent();
        expectPunct(':');
        bool image = k == ResourceKind::Texture || k == ResourceKind::StorageTexture;
        if (image) {
            r.dim = parseEnumNumeric();
            Type t = parseType();
            if (t.width != 1) fail("texture element must be scalar");
            r.elem = t.scalar;
            r.depth = acceptWord("depth");
        }
        expectWord("index");
        r.index = parseParenU32();
        if (k == ResourceKind::Buffer || k == ResourceKind::StorageTexture)
            r.access = parseEnum<Access>(toString, Access::ReadWrite, "access");
        return r;
    }

    // Texture dimensions start with a digit ("2d"), so the lexer yields them as numbers.
    TexDim parseEnumNumeric() {
        TexDim d;
        if ((tok_.kind != Tok::Number && tok_.kind != Tok::Ident) ||
            !lookupEnum<TexDim>(tok_.text, d, toString, TexDim::CubeArray))
            fail("expected texture dimension");
        advance();
        return d;
    }

    // ---- functions ----
    void defineValue(ValueId id, Type t) {
        if (t.isVoid()) fail("value of type void");
        if (fn_->valueTypes.size() <= id) {
            if (id > (1u << 24)) fail("value id too large");
            fn_->valueTypes.resize(id + 1);
        }
        // Redefinition is a verifier error; the parser keeps the first type it saw.
        if (fn_->valueTypes[id].isVoid()) fn_->valueTypes[id] = t;
    }

    void parseFunction() {
        m_.functions.emplace_back();
        fn_ = &m_.functions.back();
        fn_->name = expectIdent();
        expectPunct('(');
        if (!acceptPunct(')')) {
            do {
                ValueId id = parseId(Tok::Value);
                expectPunct(':');
                defineValue(id, parseType());
                fn_->params.push_back(id);
            } while (acceptPunct(','));
            expectPunct(')');
        }
        if (acceptPunct('-')) {  // "->" is lexed as one punct token starting with '-'
            fn_->returnType = parseType();
        }
        expectPunct('{');
        endLine();
        while (acceptWord("var")) {
            uint32_t id = parseId(Tok::Var);
            if (id != fn_->vars.size()) fail("variables must be numbered in order");
            expectPunct(':');
            fn_->vars.push_back({parseType()});
            endLine();
        }
        fn_->body = parseBlock();
        endLine();
    }

    // Parses statements up to (and consuming) the closing '}'.
    Block parseBlock() {
        Block b;
        while (!acceptPunct('}')) {
            if (tok_.kind == Tok::End) fail("unexpected end of input");
            b.push_back(parseStatement());
        }
        return b;
    }

    Node parseStatement() {
        Node n;
        if (acceptWord("if")) {
            n.kind = NodeKind::If;
            n.value = parseId(Tok::Value);
            expectPunct('{');
            endLine();
            n.body = parseBlock();
            if (acceptWord("else")) {
                expectPunct('{');
                endLine();
                n.elseBody = parseBlock();
                if (n.elseBody.empty()) fail("empty else block");
            }
        } else if (acceptWord("loop")) {
            n.kind = NodeKind::Loop;
            expectPunct('{');
            endLine();
            n.body = parseBlock();
        } else if (acceptWord("break")) {
            n.kind = NodeKind::Break;
        } else if (acceptWord("continue")) {
            n.kind = NodeKind::Continue;
        } else if (acceptWord("discard")) {
            n.kind = NodeKind::Discard;
        } else if (acceptWord("return")) {
            n.kind = NodeKind::Return;
            if (tok_.kind == Tok::Value) n.value = parseId(Tok::Value);
        } else {
            n.inst = parseInst();
        }
        endLine();
        return n;
    }

    uint32_t lookupSymbol(char c) {
        std::string_view name = expectIdent();
        auto find = [&](const auto& list) -> uint32_t {
            for (uint32_t i = 0; i < list.size(); ++i)
                if (list[i].name == name) return i;
            fail("unknown name '" + std::string(name) + "'");
        };
        switch (c) {
            case 'i': return find(m_.inputs);
            case 'o': return find(m_.outputs);
            case 'f': {
                // Only earlier functions are callable; this also keeps the call graph acyclic.
                for (uint32_t i = 0; i + 1 < m_.functions.size(); ++i)
                    if (m_.functions[i].name == name) return i;
                fail("unknown function '" + std::string(name) + "'");
            }
            case 'm':
                if (name == "shared") return kTargetShared;
                if (name == "payload") return kTargetPayload;
                if (name == "private") return kTargetPrivate;
                if (name.size() > 8 && name.substr(0, 8) == "private_" &&
                    name.find_first_not_of("0123456789", 8) == std::string_view::npos)
                    return kTargetPrivateSlot0 + uint32_t(std::stoul(std::string(name.substr(8))));
                return find(m_.resources);
            default: return find(m_.resources);
        }
    }

    Inst parseInst() {
        Inst in;
        if (tok_.kind == Tok::Value) {
            in.result = parseId(Tok::Value);
            expectPunct('=');
        }
        in.precise = acceptWord("precise");
        std::string_view opName = expectIdent();
        if (!opFromName(opName, in.op)) fail("unknown op '" + std::string(opName) + "'");

        std::vector<std::string_view> rawLits;
        bool first = true;
        auto sep = [&] {
            if (!first) expectPunct(',');
            first = false;
        };
        auto more = [&] { return isPunct(','); };  // optional trailing operands follow a comma
        for (const char* p = opInfo(in.op).signature; *p; ++p) {
            switch (*p) {
                case 'v':
                    sep();
                    in.args.push_back(parseId(Tok::Value));
                    break;
                case '$':
                    sep();
                    in.sym = parseId(Tok::Var);
                    break;
                case 's':
                    sep();
                    in.sampler = lookupSymbol('r');
                    break;
                case 'n':
                    sep();
                    in.lits.push_back(parseU32());
                    break;
                case '*':
                    while (first ? tok_.kind == Tok::Value : more()) {
                        sep();
                        in.args.push_back(parseId(Tok::Value));
                    }
                    break;
                case 'N':
                    while (more()) {
                        sep();
                        in.lits.push_back(parseU32());
                    }
                    break;
                case 'k':
                    while (first ? tok_.kind == Tok::Number || tok_.kind == Tok::Ident : more()) {
                        sep();
                        if (tok_.kind != Tok::Number && tok_.kind != Tok::Ident) fail("expected literal");
                        rawLits.push_back(tok_.text);
                        advance();
                    }
                    break;
                case 'I': parseImageOperands(in, sep, more); break;
                default:
                    sep();
                    in.sym = lookupSymbol(*p);
                    break;
            }
        }
        if (in.result != kNoValue) {
            expectPunct(':');
            in.type = parseType();
            defineValue(in.result, in.type);
        }
        for (std::string_view lit : rawLits) in.lits.push_back(parseLiteral(in.type.scalar, lit));
        return in;
    }

    template <typename Sep, typename More>
    void parseImageOperands(Inst& in, Sep& sep, More& more) {
        unsigned lastBit = 0;
        while (more()) {
            sep();
            std::string_view name = expectIdent();
            const ImageOperandSpec* spec = nullptr;
            for (const ImageOperandSpec& s : kImageOperands)
                if (name == s.name) spec = &s;
            if (!spec) fail("unknown image operand '" + std::string(name) + "'");
            if (spec->bit <= lastBit) fail("image operands out of order or repeated");
            lastBit = spec->bit;
            in.image |= spec->bit;
            expectPunct('(');
            for (unsigned k = 0; k < spec->count; ++k) {
                if (k) expectPunct(',');
                in.args.push_back(parseId(Tok::Value));
            }
            expectPunct(')');
        }
    }

    uint32_t parseLiteral(Scalar s, std::string_view t) {
        const char* b = t.data();
        const char* e = b + t.size();
        auto whole = [&](auto r) {
            if (r.ec != std::errc() || r.ptr != e) fail("bad literal '" + std::string(t) + "'");
        };
        if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
            uint32_t bits = 0;
            whole(std::from_chars(b + 2, e, bits, 16));
            return bits;
        }
        switch (s) {
            case Scalar::Bool:
                if (t == "true") return 1;
                if (t == "false") return 0;
                break;
            case Scalar::I32: {
                int32_t v = 0;
                whole(std::from_chars(b, e, v));
                return uint32_t(v);
            }
            case Scalar::U32: {
                uint32_t v = 0;
                whole(std::from_chars(b, e, v));
                return v;
            }
            case Scalar::F16:
            case Scalar::F32: {
                // Not std::from_chars: its float overload is missing from Apple's libc++ before macOS 26.
                // strtof is correctly rounded; the character filter keeps out its hex/inf/nan forms.
                std::string z(b, e);
                char* end = nullptr;
                if (z.empty() || z.find_first_not_of("0123456789+-.eE") != std::string::npos)
                    fail("bad literal '" + std::string(t) + "'");
                float f = std::strtof(z.c_str(), &end);
                if (end != z.c_str() + z.size()) fail("bad literal '" + std::string(t) + "'");
                return s == Scalar::F16 ? floatToHalf(f) : std::bit_cast<uint32_t>(f);
            }
            case Scalar::Void: break;
        }
        fail("bad literal '" + std::string(t) + "'");
    }

    Lexer lex_;
    Token tok_;
    Module m_;
    Function* fn_ = nullptr;
};

}  // namespace

ParseResult parse(std::string_view text) {
    Parser p(text);
    try {
        return {p.run(), {}};
    } catch (const ParseError& e) {
        return {std::nullopt, e.what()};
    }
}

}  // namespace vpir
