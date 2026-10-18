// Lexer and preprocessor (GLSL ES 3.20 §3, which ES 1.00 and 3.00 are subsets of).
#include "pp.h"

#include <cctype>
#include <cstdlib>
#include <functional>
#include <map>

namespace vpglsl {

namespace {

bool identStart(char c) { return std::isalpha(uint8_t(c)) || c == '_'; }
bool identChar(char c) { return std::isalnum(uint8_t(c)) || c == '_'; }

// Longest-first punctuators.
const char* const kPuncts[] = {
    "<<=", ">>=", "...", "++", "--", "<=", ">=", "==", "!=", "&&", "||", "^^", "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=", "<<", ">>", "##", "+", "-", "*", "/", "%", "<", ">", "=", "!", "~", "&", "|", "^",
    "?", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}", "#",
};

}  // namespace

std::vector<Token> lex(const std::string& in, std::string& log) {
    // Line continuations first (GLSL ES 3.00+; harmless for 1.00 sources, which never use them).
    std::string s;
    s.reserve(in.size());
    std::vector<int> lineOf;  // line of each kept char
    int line = 1;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size() && (in[i + 1] == '\n' || in[i + 1] == '\r')) {
            ++i;
            if (in[i] == '\r' && i + 1 < in.size() && in[i + 1] == '\n') ++i;
            ++line;
            continue;
        }
        if (in[i] == '\r') {
            if (i + 1 < in.size() && in[i + 1] == '\n') continue;
            s += '\n', lineOf.push_back(line++);
            continue;
        }
        s += in[i];
        lineOf.push_back(line);
        if (in[i] == '\n') ++line;
    }
    std::vector<Token> out;
    bool space = false;
    size_t i = 0;
    auto lineAt = [&](size_t p) { return p < lineOf.size() ? lineOf[p] : line; };
    while (i < s.size()) {
        char c = s[i];
        if (c == '\n') {
            Token t;
            t.kind = Tok::Newline, t.line = lineAt(i);
            out.push_back(t);
            ++i, space = false;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\v' || c == '\f') {
            ++i, space = true;
            continue;
        }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '/') {
            while (i < s.size() && s[i] != '\n') ++i;
            space = true;
            continue;
        }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '*') {
            size_t e = s.find("*/", i + 2);
            if (e == std::string::npos) {
                log += "ERROR: 0:" + std::to_string(lineAt(i)) + ": unterminated comment\n";
                e = s.size() - 2;
            }
            // Keep the newlines inside the comment (line numbers, directive ends).
            for (size_t k = i; k < e + 2 && k < s.size(); ++k)
                if (s[k] == '\n') {
                    Token t;
                    t.kind = Tok::Newline, t.line = lineAt(k);
                    out.push_back(t);
                }
            i = e + 2, space = true;
            continue;
        }
        Token t;
        t.line = lineAt(i);
        t.spaceBefore = space;
        space = false;
        if (identStart(c)) {
            size_t b = i;
            while (i < s.size() && identChar(s[i])) ++i;
            t.kind = Tok::Ident, t.text = s.substr(b, i - b);
            if (t.text == "true" || t.text == "false") t.kind = Tok::BoolLit, t.ival = t.text == "true";
            out.push_back(t);
            continue;
        }
        if (std::isdigit(uint8_t(c)) || (c == '.' && i + 1 < s.size() && std::isdigit(uint8_t(s[i + 1])))) {
            size_t b = i;
            bool isFloat = false;
            if (c == '0' && i + 1 < s.size() && (s[i + 1] == 'x' || s[i + 1] == 'X')) {
                i += 2;
                while (i < s.size() && std::isxdigit(uint8_t(s[i]))) ++i;
            } else {
                while (i < s.size() && std::isdigit(uint8_t(s[i]))) ++i;
                if (i < s.size() && s[i] == '.') {
                    isFloat = true, ++i;
                    while (i < s.size() && std::isdigit(uint8_t(s[i]))) ++i;
                }
                if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
                    size_t k = i + 1;
                    if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
                    if (k < s.size() && std::isdigit(uint8_t(s[k]))) {
                        isFloat = true, i = k;
                        while (i < s.size() && std::isdigit(uint8_t(s[i]))) ++i;
                    }
                }
            }
            std::string num = s.substr(b, i - b);
            if (isFloat || (i < s.size() && (s[i] == 'f' || s[i] == 'F') && num.find_first_of("xX") == std::string::npos)) {
                t.kind = Tok::FloatLit;
                t.fval = std::strtod(num.c_str(), nullptr);
                if (i < s.size() && (s[i] == 'f' || s[i] == 'F')) ++i;
                // "hf"/"HF" (half suffix of desktop GLSL) is not ES.
            } else {
                t.kind = Tok::IntLit;
                const bool octal = num.size() > 1 && num[0] == '0' && num.find_first_of("xX") == std::string::npos;
                t.ival = std::strtoull(num.c_str(), nullptr, octal ? 8 : 0);
                if (i < s.size() && (s[i] == 'u' || s[i] == 'U')) t.kind = Tok::UintLit, ++i;
            }
            t.text = s.substr(b, i - b);
            out.push_back(t);
            continue;
        }
        bool matched = false;
        for (const char* p : kPuncts) {
            size_t n = std::char_traits<char>::length(p);
            if (s.compare(i, n, p) == 0) {
                t.kind = std::string(p) == "#" ? Tok::Hash : Tok::Punct;
                t.text = p;
                i += n;
                matched = true;
                break;
            }
        }
        if (!matched) {
            log += "ERROR: 0:" + std::to_string(t.line) + ": '" + std::string(1, c) + "' : unexpected character\n";
            ++i;
            continue;
        }
        out.push_back(t);
    }
    Token end;
    end.kind = Tok::End, end.line = line;
    out.push_back(end);
    return out;
}

namespace {

struct Macro {
    bool function = false;
    std::vector<std::string> params;
    std::vector<Token> body;
};

// Extensions the compiler knows; each defines a macro of its name = 1.
const char* const kExtensions[] = {
    "GL_OES_standard_derivatives", "GL_EXT_shader_texture_lod", "GL_OES_EGL_image_external",
    "GL_OES_EGL_image_external_essl3", "GL_EXT_frag_depth", "GL_EXT_draw_buffers", "GL_OVR_multiview",
    "GL_OVR_multiview2", "GL_EXT_shadow_samplers", "GL_OES_texture_3D", "GL_EXT_shader_framebuffer_fetch",
    "GL_ARM_shader_framebuffer_fetch", "GL_EXT_YUV_target", "GL_OES_sample_variables",
    "GL_OES_shader_multisample_interpolation", "GL_EXT_texture_buffer", "GL_EXT_clip_cull_distance",
    "GL_EXT_shader_io_blocks", "GL_OES_shader_io_blocks", "GL_EXT_gpu_shader5", "GL_OES_texture_storage_multisample_2d_array",
};

class Pp {
public:
    Pp(Stage stage, std::string& log) : stage_(stage), log_(log) {}

    PpResult run(const std::string& source) {
        std::vector<Token> toks = lex(source, log_);
        // Defaults before #version is seen.
        defineSimple("GL_ES", "1");
        defineSimple("GL_FRAGMENT_PRECISION_HIGH", "1");
        for (const char* e : kExtensions) defineSimple(e, "1");
        size_t i = 0;
        bool firstToken = true;
        while (toks[i].kind != Tok::End) {
            // Gather one line.
            std::vector<Token> line;
            while (toks[i].kind != Tok::End && toks[i].kind != Tok::Newline) line.push_back(toks[i++]);
            if (toks[i].kind == Tok::Newline) ++i;
            if (line.empty()) continue;
            if (line[0].kind == Tok::Hash) {
                directive(line, firstToken);
                firstToken = false;
                continue;
            }
            firstToken = false;
            if (!active()) continue;
            pending_.insert(pending_.end(), line.begin(), line.end());
            // A function-like macro call may span lines: expand once the text is complete.
            flushPending(false);
        }
        flushPending(true);
        if (!conds_.empty()) error(toks.back().line, "unterminated #if");
        if (!defines_.count("__VERSION__")) defineSimple("__VERSION__", "100");
        Token end;
        end.kind = Tok::End, end.line = toks.back().line;
        res_.tokens.push_back(end);
        res_.ok = ok_;
        return std::move(res_);
    }

private:
    struct Cond {
        bool active, taken, parentActive, sawElse;
    };

    bool active() const { return conds_.empty() || conds_.back().active; }

    void error(int line, const std::string& m) {
        log_ += "ERROR: 0:" + std::to_string(line) + ": " + m + "\n";
        ok_ = false;
    }

    void defineSimple(const std::string& name, const std::string& value) {
        Macro m;
        std::string l;
        for (const Token& t : lex(value, l))
            if (t.kind != Tok::End && t.kind != Tok::Newline) m.body.push_back(t);
        defines_[name] = m;
    }

    // Expands everything accumulated so far; keeps an unfinished trailing macro call.
    void flushPending(bool final) {
        if (pending_.empty()) return;
        if (!final) {
            // Unbalanced parentheses after a function-like macro name: wait for more lines.
            int depth = 0;
            bool afterFn = false;
            for (const Token& t : pending_) {
                if (t.kind == Tok::Ident) {
                    auto it = defines_.find(t.text);
                    afterFn = it != defines_.end() && it->second.function;
                    continue;
                }
                if (t.kind == Tok::Punct && t.text == "(" && (afterFn || depth > 0)) ++depth;
                else if (t.kind == Tok::Punct && t.text == ")" && depth > 0) --depth;
                afterFn = false;
            }
            if (depth > 0) return;
            if (!pending_.empty() && pending_.back().kind == Tok::Ident) {
                auto it = defines_.find(pending_.back().text);
                if (it != defines_.end() && it->second.function) return;
            }
        }
        std::vector<Token> outv;
        std::set<std::string> hide;
        expand(pending_, outv, hide);
        for (Token& t : outv) res_.tokens.push_back(t);
        pending_.clear();
    }

    // Collects the arguments of a function-like call starting at `in[i]` == "(". Returns false when
    // the call is unterminated.
    bool collectArgs(const std::vector<Token>& in, size_t& i, std::vector<std::vector<Token>>& args) {
        int depth = 0;
        args.assign(1, {});
        for (; i < in.size(); ++i) {
            const Token& t = in[i];
            if (t.kind == Tok::Punct && t.text == "(") {
                if (depth++ == 0) continue;
            } else if (t.kind == Tok::Punct && t.text == ")") {
                if (--depth == 0) {
                    ++i;
                    return true;
                }
            } else if (t.kind == Tok::Punct && t.text == "," && depth == 1) {
                args.emplace_back();
                continue;
            }
            args.back().push_back(t);
        }
        return false;
    }

    void expand(const std::vector<Token>& in, std::vector<Token>& out, std::set<std::string>& hide, int depth = 0) {
        if (depth > 64) {
            error(in.empty() ? 0 : in[0].line, "macro expansion too deep");
            return;
        }
        for (size_t i = 0; i < in.size();) {
            const Token& t = in[i];
            if (t.kind != Tok::Ident || hide.count(t.text)) {
                out.push_back(t), ++i;
                continue;
            }
            if (t.text == "__LINE__") {
                Token n = t;
                n.kind = Tok::IntLit, n.ival = uint64_t(t.line), n.text = std::to_string(t.line);
                out.push_back(n), ++i;
                continue;
            }
            if (t.text == "__FILE__") {
                Token n = t;
                n.kind = Tok::IntLit, n.ival = 0, n.text = "0";
                out.push_back(n), ++i;
                continue;
            }
            auto it = defines_.find(t.text);
            if (it == defines_.end()) {
                out.push_back(t), ++i;
                continue;
            }
            const Macro& m = it->second;
            if (!m.function) {
                std::vector<Token> body = m.body;
                for (Token& b : body) b.line = t.line;
                body = paste(body);
                hide.insert(t.text);
                expand(body, out, hide, depth + 1);
                hide.erase(t.text);
                ++i;
                continue;
            }
            size_t j = i + 1;
            if (j >= in.size() || in[j].kind != Tok::Punct || in[j].text != "(") {
                out.push_back(t), ++i;  // name of a function-like macro without a call
                continue;
            }
            std::vector<std::vector<Token>> args;
            if (!collectArgs(in, j, args)) {
                error(t.line, "unterminated macro call '" + t.text + "'");
                return;
            }
            if (m.params.empty() && args.size() == 1 && args[0].empty()) args.clear();
            if (args.size() != m.params.size()) {
                error(t.line, "macro '" + t.text + "' expects " + std::to_string(m.params.size()) + " arguments");
                i = j;
                continue;
            }
            // Arguments are fully expanded before substitution, except next to ##.
            std::vector<Token> body;
            for (size_t k = 0; k < m.body.size(); ++k) {
                const Token& b = m.body[k];
                int p = -1;
                if (b.kind == Tok::Ident)
                    for (size_t q = 0; q < m.params.size(); ++q)
                        if (m.params[q] == b.text) p = int(q);
                if (p < 0) {
                    Token c = b;
                    c.line = t.line;
                    body.push_back(c);
                    continue;
                }
                const bool nearPaste = (k > 0 && m.body[k - 1].text == "##") ||
                                       (k + 1 < m.body.size() && m.body[k + 1].text == "##");
                if (nearPaste) {
                    for (Token a : args[size_t(p)]) a.line = t.line, body.push_back(a);
                } else {
                    std::vector<Token> ex;
                    expand(args[size_t(p)], ex, hide, depth + 1);
                    for (Token a : ex) a.line = t.line, body.push_back(a);
                }
            }
            body = paste(body);
            hide.insert(t.text);
            // Rescan with the rest of the input so a macro producing a function-like name followed
            // by "(" in the source still works: only the body is rescanned here (common case).
            expand(body, out, hide, depth + 1);
            hide.erase(t.text);
            i = j;
        }
    }

    std::vector<Token> paste(const std::vector<Token>& in) {
        std::vector<Token> out;
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].kind == Tok::Punct && in[i].text == "##" && !out.empty() && i + 1 < in.size()) {
                std::string text = out.back().text + in[i + 1].text;
                std::string l;
                std::vector<Token> re = lex(text, l);
                int line = out.back().line;
                out.pop_back();
                for (Token& r : re)
                    if (r.kind != Tok::End && r.kind != Tok::Newline) r.line = line, out.push_back(r);
                ++i;
                continue;
            }
            out.push_back(in[i]);
        }
        return out;
    }

    // #if expression evaluation (integers only).
    struct Eval {
        const std::vector<Token>& t;
        size_t i = 0;
        bool bad = false;
        const Token* peek() { return i < t.size() ? &t[i] : nullptr; }
        bool is(const char* p) { return peek() && peek()->kind == Tok::Punct && peek()->text == p; }
        int64_t primary() {
            const Token* k = peek();
            if (!k) return bad = true, 0;
            if (is("(")) {
                ++i;
                int64_t v = expr(0);
                if (!is(")")) bad = true;
                else ++i;
                return v;
            }
            if (is("!")) return ++i, !primary();
            if (is("-")) return ++i, -primary();
            if (is("+")) return ++i, primary();
            if (is("~")) return ++i, ~primary();
            if (k->kind == Tok::IntLit || k->kind == Tok::UintLit || k->kind == Tok::BoolLit) return ++i, int64_t(k->ival);
            if (k->kind == Tok::Ident) return ++i, 0;  // undefined identifier: 0
            bad = true;
            return 0;
        }
        static int prec(const std::string& op) {
            if (op == "||") return 1;
            if (op == "&&") return 2;
            if (op == "|") return 3;
            if (op == "^") return 4;
            if (op == "&") return 5;
            if (op == "==" || op == "!=") return 6;
            if (op == "<" || op == ">" || op == "<=" || op == ">=") return 7;
            if (op == "<<" || op == ">>") return 8;
            if (op == "+" || op == "-") return 9;
            if (op == "*" || op == "/" || op == "%") return 10;
            return -1;
        }
        int64_t expr(int minPrec) {
            int64_t l = primary();
            for (;;) {
                const Token* k = peek();
                if (!k || k->kind != Tok::Punct) break;
                int p = prec(k->text);
                if (p < 0 || p < minPrec) break;
                std::string op = k->text;
                ++i;
                int64_t r = expr(p + 1);
                if (op == "||") l = l || r;
                else if (op == "&&") l = l && r;
                else if (op == "|") l |= r;
                else if (op == "^") l ^= r;
                else if (op == "&") l &= r;
                else if (op == "==") l = l == r;
                else if (op == "!=") l = l != r;
                else if (op == "<") l = l < r;
                else if (op == ">") l = l > r;
                else if (op == "<=") l = l <= r;
                else if (op == ">=") l = l >= r;
                else if (op == "<<") l = r >= 0 && r < 64 ? l << r : 0;
                else if (op == ">>") l = r >= 0 && r < 64 ? l >> r : 0;
                else if (op == "+") l += r;
                else if (op == "-") l -= r;
                else if (op == "*") l *= r;
                else if (op == "/") l = r ? l / r : (bad = true, 0);
                else if (op == "%") l = r ? l % r : (bad = true, 0);
            }
            return l;
        }
    };

    bool evalIf(const std::vector<Token>& line, size_t from) {
        // `defined X` / `defined(X)` first, then macro expansion.
        std::vector<Token> pre;
        for (size_t i = from; i < line.size(); ++i) {
            if (line[i].kind == Tok::Ident && line[i].text == "defined") {
                size_t j = i + 1;
                bool paren = j < line.size() && line[j].text == "(";
                if (paren) ++j;
                Token n;
                n.kind = Tok::IntLit, n.line = line[i].line;
                n.ival = j < line.size() && defines_.count(line[j].text) ? 1 : 0;
                pre.push_back(n);
                i = j + (paren ? 1 : 0);
                continue;
            }
            pre.push_back(line[i]);
        }
        std::vector<Token> ex;
        std::set<std::string> hide;
        expand(pre, ex, hide);
        Eval e{ex};
        int64_t v = e.expr(0);
        if (e.bad || e.i != ex.size()) error(line[0].line, "invalid #if expression");
        return v != 0;
    }

    void directive(const std::vector<Token>& line, bool first) {
        const int ln = line[0].line;
        if (line.size() == 1) return;  // null directive
        const std::string d = line[1].text;
        if (d == "if" || d == "ifdef" || d == "ifndef") {
            bool parent = active();
            bool v = false;
            if (parent) {
                if (d == "if") v = evalIf(line, 2);
                else if (line.size() < 3) error(ln, "#" + d + " needs a name");
                else v = (defines_.count(line[2].text) != 0) == (d == "ifdef");
            }
            conds_.push_back({parent && v, v, parent, false});
            return;
        }
        if (d == "elif") {
            if (conds_.empty()) return error(ln, "#elif without #if");
            Cond& c = conds_.back();
            if (c.sawElse) return error(ln, "#elif after #else");
            if (c.taken || !c.parentActive) {
                c.active = false;
                return;
            }
            bool v = evalIf(line, 2);
            c.active = v, c.taken = v;
            return;
        }
        if (d == "else") {
            if (conds_.empty()) return error(ln, "#else without #if");
            Cond& c = conds_.back();
            if (c.sawElse) return error(ln, "duplicate #else");
            c.sawElse = true;
            c.active = c.parentActive && !c.taken;
            c.taken = true;
            return;
        }
        if (d == "endif") {
            if (conds_.empty()) return error(ln, "#endif without #if");
            conds_.pop_back();
            return;
        }
        if (!active()) return;
        flushPending(true);
        if (d == "version") {
            if (!first) error(ln, "#version must be the first directive");
            if (line.size() < 3 || line[2].kind != Tok::IntLit) return error(ln, "bad #version");
            res_.version = int(line[2].ival);
            res_.es = line.size() < 4 || line[3].text == "es";
            if (res_.version != 100 && res_.version != 300 && res_.version != 310 && res_.version != 320)
                return error(ln, "unsupported #version " + std::to_string(res_.version));
            if (res_.version >= 300 && !res_.es) return error(ln, "only ES profiles are supported");
            defineSimple("__VERSION__", std::to_string(res_.version));
            return;
        }
        if (d == "define") {
            if (line.size() < 3 || line[2].kind != Tok::Ident) return error(ln, "#define needs a name");
            const std::string& name = line[2].text;
            if (name.rfind("GL_", 0) == 0 || name.find("__") != std::string::npos) {
                // Reserved names: allowed (warning) for "__" ones by many compilers; GL_ is an error.
                if (name.rfind("GL_", 0) == 0) return error(ln, "'" + name + "' : reserved macro name");
            }
            Macro m;
            size_t i = 3;
            if (i < line.size() && line[i].text == "(" && !line[i].spaceBefore) {
                m.function = true;
                ++i;
                while (i < line.size() && line[i].text != ")") {
                    if (line[i].kind == Tok::Ident) m.params.push_back(line[i].text);
                    ++i;
                }
                ++i;
            }
            for (; i < line.size(); ++i) m.body.push_back(line[i]);
            defines_[name] = m;
            return;
        }
        if (d == "undef") {
            if (line.size() >= 3) defines_.erase(line[2].text);
            return;
        }
        if (d == "extension") {
            // #extension name : behavior
            if (line.size() < 5) return error(ln, "bad #extension");
            const std::string& name = line[2].text;
            const std::string& beh = line[4].text;
            bool known = name == "all";
            for (const char* e : kExtensions) known |= name == e;
            if (!known && beh == "require") return error(ln, "extension '" + name + "' is not supported");
            if (!known) log_ += "WARNING: 0:" + std::to_string(ln) + ": extension '" + name + "' is not supported\n";
            if (beh == "disable") res_.extensions.erase(name);
            else res_.extensions.insert(name);
            return;
        }
        if (d == "pragma" || d == "line") return;
        if (d == "error") {
            std::string msg;
            for (size_t i = 2; i < line.size(); ++i) msg += (i > 2 ? " " : "") + line[i].text;
            return error(ln, "#error " + msg);
        }
        error(ln, "unknown directive #" + d);
    }

    Stage stage_;
    std::string& log_;
    std::map<std::string, Macro> defines_;
    std::vector<Cond> conds_;
    std::vector<Token> pending_;
    PpResult res_;
    bool ok_ = true;
};

}  // namespace

PpResult runPreprocessor(const std::string& source, Stage stage, std::string& log) {
    Pp p(stage, log);
    return p.run(source);
}

std::string preprocess(const std::string& source, Stage stage, std::string& log, int& version) {
    PpResult r = runPreprocessor(source, stage, log);
    version = r.version;
    std::string out;
    for (const Token& t : r.tokens) {
        if (t.kind == Tok::End) break;
        if (!out.empty()) out += ' ';
        out += t.text;
    }
    return out;
}

}  // namespace vpglsl
