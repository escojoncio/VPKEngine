// Lexer and preprocessor (GLSL ES 3.20 §3, which ES 1.00 and 3.00 are subsets of).
#include "pp.h"
#include "vpir/stack.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <new>

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

// A token is 72 bytes and the preprocessor holds a few copies of the source's tokens (lexed, per line, pending a macro call), so
// the source is bounded in tokens (the same number as the expansion's step limit): a hostile 4 MB source (2M tokens) took 530 MB before
// this limit. Real shaders are far below it (1M tokens are 3-4 MB of text).
constexpr size_t kMaxLexTokens = 1'000'000;      // tokens other than line ends, in the whole source (inactive #if blocks count: they are lexed)
// The lexer builds a copy of the source and a line number per byte (about 5 bytes per byte of source) before it makes a token.
constexpr size_t kMaxSourceBytes = size_t(64) << 20;
constexpr size_t kMaxLexNewlines = 1'000'000;    // line-end tokens (kept for the directives)

}  // namespace

std::vector<Token> lex(const std::string& in, std::string& log) {
    if (in.size() > kMaxSourceBytes) {
        log += "ERROR: 0:1: shader source too large (more than 64 MiB)\n";
        Token end;
        end.kind = Tok::End;
        return {end};
    }
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
    size_t newlines = 0;  // Newline tokens in `out`
    bool space = false;
    size_t i = 0;
    auto lineAt = [&](size_t p) { return p < lineOf.size() ? lineOf[p] : line; };
    bool tooBig = false;
    while (i < s.size() && !tooBig) {
        if (out.size() - newlines > kMaxLexTokens || newlines > kMaxLexNewlines) {
            log += "ERROR: 0:" + std::to_string(lineAt(i)) + ": shader source too large (more than " + std::to_string(kMaxLexTokens) + " tokens or " + std::to_string(kMaxLexNewlines) + " lines)\n";
            line = lineAt(i);
            break;
        }
        char c = s[i];
        if (c == '\n') {
            Token t;
            t.kind = Tok::Newline, t.line = lineAt(i);
            out.push_back(t), ++newlines;
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
                    t.kind = Tok::Newline, t.line = lineAt(k), t.ival = 1;  // ival 1: inside a comment
                    out.push_back(t), ++newlines;
                    if (newlines > kMaxLexNewlines) {  // (checked here: a comment of millions of lines is one pass of this loop)
                        log += "ERROR: 0:" + std::to_string(lineAt(k)) + ": shader source too large (more than " + std::to_string(kMaxLexTokens) +
                               " tokens or " + std::to_string(kMaxLexNewlines) + " lines)\n";
                        line = lineAt(k);
                        tooBig = true;
                        break;
                    }
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
            int badKind = 0;  // Bad token after the number: 3 = `1f`, 4 = a digit 8 or 9 in an octal number
            if (isFloat || (i < s.size() && (s[i] == 'f' || s[i] == 'F') && num.find_first_of("xX") == std::string::npos)) {
                t.kind = Tok::FloatLit;
                t.fval = std::strtod(num.c_str(), nullptr);
                if (i < s.size() && (s[i] == 'f' || s[i] == 'F')) {
                    if (!isFloat) badKind = 3;  // "1f": the suffix needs a decimal point or an exponent
                    ++i;
                }
                // "hf"/"HF" (half suffix of desktop GLSL) is not ES.
            } else {
                t.kind = Tok::IntLit;
                const bool octal = num.size() > 1 && num[0] == '0' && num.find_first_of("xX") == std::string::npos;
                if (octal && num.find_first_of("89") != std::string::npos) badKind = 4;
                t.ival = std::strtoull(num.c_str(), nullptr, octal ? 8 : 0);
                if (i < s.size() && (s[i] == 'u' || s[i] == 'U')) t.kind = Tok::UintLit, ++i;
            }
            t.text = s.substr(b, i - b);
            out.push_back(t);
            if (badKind) {
                Token bad;
                bad.kind = Tok::Bad, bad.line = t.line, bad.ival = uint64_t(badKind), bad.text = t.text;
                out.push_back(bad);
            }
            if (i < s.size() && identChar(s[i])) {  // "1e", "12abc": an invalid pp-number (the number, then the junk as a Bad token)
                Token bad;
                bad.kind = Tok::Bad, bad.line = t.line, bad.ival = 1;
                const size_t sb = i;
                while (i < s.size() && identChar(s[i])) ++i;
                bad.text = s.substr(sb, i - sb);
                out.push_back(bad);
            }
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
            t.kind = Tok::Bad, t.text = std::string(1, c);  // an error unless it sits in a #pragma line
            ++i;
            out.push_back(t);
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

// Extensions the compiler implements; each defines a macro of its name = 1 (a shader tests those to
// pick code paths, so only extensions that do something are listed). Not implemented, hence neither
// defined nor accepted by `#extension : require`: GL_EXT_YUV_target, GL_OES_sample_variables,
// GL_OES_shader_multisample_interpolation, GL_EXT_clip_cull_distance,
// GL_EXT_gpu_shader5, GL_OES_texture_storage_multisample_2d_array. (GL_EXT_shader_io_blocks and
// GL_OES_shader_io_blocks: see kAcceptedExtensions.)
const char* const kExtensions[] = {
    "GL_OES_standard_derivatives", "GL_EXT_shader_texture_lod", "GL_OES_EGL_image_external",
    "GL_OES_EGL_image_external_essl3", "GL_EXT_frag_depth", "GL_EXT_draw_buffers", "GL_OVR_multiview",
    "GL_OVR_multiview2", "GL_EXT_shadow_samplers", "GL_OES_texture_3D", "GL_EXT_texture_buffer",
};
// Accepted by `#extension` but not defined as macros: the shader may name them without using what they add.
// I/O blocks (the whole of GL_EXT/OES_shader_io_blocks) are still rejected where they are declared.
const char* const kAcceptedExtensions[] = {"GL_EXT_shader_io_blocks", "GL_OES_shader_io_blocks"};
// Extensions that need the GL layer's help (see Caps).
const char* const kFramebufferFetchExtensions[] = {"GL_EXT_shader_framebuffer_fetch", "GL_ARM_shader_framebuffer_fetch"};
const char* const kCubeArrayExtensions[] = {"GL_OES_texture_cube_map_array", "GL_EXT_texture_cube_map_array"};

std::atomic<bool> gFramebufferFetch{false}, gCubeMapArray{false};

std::vector<const char*> supportedExtensions() {
    std::vector<const char*> v(std::begin(kExtensions), std::end(kExtensions));
    if (gFramebufferFetch) v.insert(v.end(), std::begin(kFramebufferFetchExtensions), std::end(kFramebufferFetchExtensions));
    if (gCubeMapArray) v.insert(v.end(), std::begin(kCubeArrayExtensions), std::end(kCubeArrayExtensions));
    return v;
}

class Pp {
public:
    Pp(Stage, std::string& log) : log_(log) {}

    PpResult run(const std::string& source) {
        const size_t logBefore = log_.size();
        std::vector<Token> toks = lex(source, log_);
        if (log_.size() != logBefore) ok_ = false;  // a character or literal the language does not have
        // Defaults before #version is seen.
        defineSimple("GL_ES", "1");
        defineSimple("__VERSION__", "100");  // a shader without #version is ES 1.00 (#if __VERSION__ ... is valid)
        defineSimple("GL_FRAGMENT_PRECISION_HIGH", "1");
        for (const char* e : supportedExtensions()) defineSimple(e, "1");
        size_t i = 0;
        bool firstToken = true;
        while (toks[i].kind != Tok::End && !outputOverflow_) {
            // Gather one line.
            std::vector<Token> line;
            {  // (reserved: a line of a million tokens must not grow by doubling next to the copy it comes from)
                size_t k = 0;
                for (size_t j = i; toks[j].kind != Tok::End && (toks[j].kind != Tok::Newline || (toks[j].ival == 1 && k > 0 && toks[i].kind == Tok::Hash)); ++j)
                    k += toks[j].kind != Tok::Newline;
                line.reserve(k);
            }
            // (a newline inside a comment does not end a directive: the comment is one space)
            while (toks[i].kind != Tok::End && (toks[i].kind != Tok::Newline || (toks[i].ival == 1 && !line.empty() && line[0].kind == Tok::Hash))) {
                if (toks[i].kind != Tok::Newline) line.push_back(std::move(toks[i]));  // (moved: only `kind` is read again)
                ++i;
            }
            if (toks[i].kind == Tok::Newline) ++i;
            if (line.empty()) continue;
            for (Token& t : line) t.line += lineDelta_;
            if (!(line[0].kind == Tok::Hash && line.size() > 1 && line[1].text == "pragma")) {
                if (active())
                    for (const Token& t : line)
                        if (t.kind == Tok::Bad) {
                            strict("character", t.line,
                                   t.ival == 3   ? "'" + t.text + "' : the f suffix needs a decimal point or an exponent"
                                   : t.ival == 4 ? "'" + t.text + "' : invalid octal digit"
                                   : t.ival      ? "'" + t.text + "' : invalid numeric literal"
                                                 : "'" + t.text + "' : unexpected character");
                            break;
                        }
                line.erase(std::remove_if(line.begin(), line.end(), [](const Token& t) { return t.kind == Tok::Bad; }), line.end());
                if (line.empty()) continue;
            }
            if (line[0].kind == Tok::Hash) {
                directive(line, firstToken);
                firstToken = false;
                continue;
            }
            firstToken = false;
            if (!active()) continue;
            sawToken_ = true;
            if (pending_.empty()) pending_ = std::move(line);  // (no second copy of a long line)
            else pending_.insert(pending_.end(), std::make_move_iterator(line.begin()), std::make_move_iterator(line.end()));
            line = {};
            // A function-like macro call may span lines: expand once the text is complete.
            flushPending(false);
        }
        flushPending(true);
        if (!conds_.empty()) strict("unbalanced_conditional", toks.back().line, "unterminated #if");
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
    // A rule of the language that a lenient driver lets through: an error in strict mode, otherwise noted and ignored.
    // True when it was an error.
    bool strict(const char* rule, int line, const std::string& m) {
        if (strictMode()) {
            error(line, m);
            return true;
        }
        if (std::find(res_.lenient.begin(), res_.lenient.end(), rule) == res_.lenient.end()) res_.lenient.push_back(rule);
        return false;
    }

    void defineSimple(const std::string& name, const std::string& value) {
        Macro m;
        std::string l;
        for (const Token& t : lex(value, l))
            if (t.kind != Tok::End && t.kind != Tok::Newline) m.body.push_back(t);
        defines_[name] = m;
    }

    // True when `name` is a function-like macro, or an object-like macro whose expansion ends with such a name (`#define B A` with
    // `#define A(p, q) ...`): a "(" after it starts a call, which may continue on the next lines.
    bool namesFunction(const std::string& name) const {
        const std::string* cur = &name;
        for (int k = 0; k < 64; ++k) {
            auto it = defines_.find(*cur);
            if (it == defines_.end()) return false;
            if (it->second.function) return true;
            const auto& b = it->second.body;
            if (b.empty() || b.back().kind != Tok::Ident) return false;
            cur = &b.back().text;
        }
        return false;
    }

    // Expands the text accumulated so far, except an unfinished trailing macro call: parentheses left open after a function-like macro
    // name wait for more lines (the scan for that resumes where the previous one stopped: pending_ only grows until it is expanded, so a
    // call open over N lines costs N, not N^2). When the expansion of the lines ends with the NAME of a function-like macro that has no
    // "(" after it (`#define F(y) A` / `F(0)` / `(5)`; `ID(G)` / `(7)`), everything before that name is released and the name alone stays
    // pending, so the "(" of the next line calls it (C99 6.10.3.4, as cpp does). The expansion budget is per flush, so a long run of
    // such lines costs each line its own share.
    size_t scanned_ = 0;
    int scanDepth_ = 0, rawDepth_ = 0;  // rawDepth_: every "(" not yet closed, call or not
    bool scanAfterFn_ = false;

    void flushPending(bool final) {
        if (pending_.empty()) return;
        if (!final) {
            for (; scanned_ < pending_.size(); ++scanned_) {
                const Token& t = pending_[scanned_];
                if (t.kind == Tok::Ident) {
                    scanAfterFn_ = namesFunction(t.text);
                    continue;
                }
                if (t.kind == Tok::Punct && t.text == "(") {
                    ++rawDepth_;
                    if (scanAfterFn_ || scanDepth_ > 0) ++scanDepth_;
                } else if (t.kind == Tok::Punct && t.text == ")") {
                    if (rawDepth_ > 0) --rawDepth_;
                    if (scanDepth_ > 0) --scanDepth_;
                }
                scanAfterFn_ = false;
            }
            // (a "(" left open after the expansion of an earlier call, `A() (1` with `#define A(p) D`, is no call to the scan: the text
            // waits for its ")" all the same; waiting only delays the expansion, which cannot differ for text without a call)
            if (scanDepth_ > 0 || rawDepth_ > 0) return;
        }
        std::vector<Token> outv;
        std::set<std::string> hide;
        const size_t logBefore = log_.size();
        expand(pending_, outv, hide);
        pending_.clear();
        scanned_ = 0, scanDepth_ = rawDepth_ = 0, scanAfterFn_ = false;
        if (!final && log_.size() == logBefore && endsWithName_ && !outv.empty() && outv.back().kind == Tok::Ident && !outv.back().painted) {
            auto fm = defines_.find(outv.back().text);
            if (fm != defines_.end() && fm->second.function) {
                pending_.push_back(std::move(outv.back()));
                outv.pop_back();
            }
        }
        if (res_.tokens.size() + outv.size() > kMaxOutputTokens) {
            if (!outputOverflow_) error(outv.empty() ? 0 : outv.front().line, "macro expansion too large");
            outputOverflow_ = true;
            return;
        }
        for (Token& t : outv) res_.tokens.push_back(std::move(t));
    }
    // The expansion budget is per flush; the tokens that all of them leave are limited too (a line of 260000 tokens, 300 times over,
    // took 3 GB). Same order as the limit on the source.
    static constexpr size_t kMaxOutputTokens = 2'000'000;
    bool outputOverflow_ = false;

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

    size_t expandSteps_ = 0;  // work of the expansion in progress, in tokens (a macro that doubles at each level would never end)
    bool expandOverflow_ = false;
    int topLine_ = 0;  // line of the macro name that the expansion in progress started from (where its errors are reported)
    static constexpr size_t kMaxExpandSteps = 1'000'000;
    // Counts `n` tokens of work (walked, copied, collected or pasted) against kMaxExpandSteps. True once past it: the error is
    // reported once and every caller unwinds.
    bool charge(size_t n, int line) {
        expandSteps_ += n;
        if (expandSteps_ <= kMaxExpandSteps) return false;
        if (!expandOverflow_) error(topLine_ ? topLine_ : line, "macro expansion too large");
        expandOverflow_ = true;
        return true;
    }

    // Set by expand(): the last token it wrote is the name of a function-like macro that had no "(" after it in `in`, so a "(" that
    // follows `in` in the enclosing text calls it. False when anything else (even a macro that expands to nothing) follows the name.
    bool endsWithName_ = false;
    // The levels of nested expansion (arguments, bodies) are bounded by the stack left, and by this count as a backstop.
    static constexpr int kMaxExpandDepth = 1000;

    void expand(const std::vector<Token>& in, std::vector<Token>& out, std::set<std::string>& hide, int depth = 0) {
        struct Ends {
            bool& dst;
            bool v = false;
            ~Ends() { dst = v; }
        } ends{endsWithName_};
        if (depth > kMaxExpandDepth || vpir::stackLow()) {
            error(in.empty() ? 0 : in[0].line, "macro expansion too deep");
            return;
        }
        if (depth == 0) expandSteps_ = 0, expandOverflow_ = false;
        for (size_t i = 0; i < in.size();) {
            ends.v = false;
            if (depth == 0) topLine_ = in[i].line;
            if (charge(1, in[i].line)) return;
            const Token& t = in[i];
            if (t.kind != Tok::Ident) {
                out.push_back(t), ++i;
                continue;
            }
            if (t.painted || hide.count(t.text)) {
                out.push_back(t), out.back().painted = true, ++i;
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
                n.kind = Tok::IntLit, n.ival = uint64_t(fileNo_), n.text = std::to_string(fileNo_);
                out.push_back(n), ++i;
                continue;
            }
            auto it = defines_.find(t.text);
            if (it == defines_.end()) {
                out.push_back(t), ++i;
                continue;
            }
            // `name` is the macro being expanded: the token at in[i], or, below, a function-like macro name that ended the expansion
            // of the previous one and takes the "(" that follows it in the input as its call (C99 6.10.3.4: the rescan goes on with
            // the rest of the source). The loop replaces a recursion over a copy of the rest of the input. `at` is where the "(" of
            // a call would be.
            const Token* name = &t;
            const Macro* mac = &it->second;
            Token carried;  // storage for a name that came out of an expansion
            size_t at = i + 1;
            bool nameLast = false;  // the expansion ended with a function-like name that nothing in `in` follows
            for (;;) {
                const size_t before = out.size();
                nameLast = false;
                if (!mac->function) {
                    if (charge(mac->body.size(), name->line)) return;
                    std::vector<Token> body = mac->body;
                    for (Token& b : body) b.line = name->line;
                    body = paste(body);
                    hide.insert(name->text);
                    expand(body, out, hide, depth + 1);
                    hide.erase(name->text);
                    nameLast = endsWithName_;
                } else {
                    if (at >= in.size() || in[at].kind != Tok::Punct || in[at].text != "(") {
                        out.push_back(*name);  // name of a function-like macro without a call
                        nameLast = at >= in.size();
                        break;
                    }
                    size_t j = at;
                    std::vector<std::vector<Token>> args;
                    const bool closed = collectArgs(in, j, args);
                    if (charge(j - at, name->line)) return;
                    if (!closed) {
                        error(name->line, "unterminated macro call '" + name->text + "'");
                        return;
                    }
                    if (mac->params.empty() && args.size() == 1 && args[0].empty()) args.clear();
                    if (args.size() != mac->params.size()) {
                        error(name->line, "macro '" + name->text + "' expects " + std::to_string(mac->params.size()) + " arguments");
                        at = j;
                        break;
                    }
                    // Arguments are fully expanded before substitution, except next to ##.
                    std::vector<Token> body;
                    for (size_t k = 0; k < mac->body.size(); ++k) {
                        const Token& b = mac->body[k];
                        int p = -1;
                        if (b.kind == Tok::Ident)
                            for (size_t q = 0; q < mac->params.size(); ++q)
                                if (mac->params[q] == b.text) p = int(q);
                        if (p < 0) {
                            Token c = b;
                            c.line = name->line;
                            body.push_back(c);
                            continue;
                        }
                        const bool nearPaste = (k > 0 && mac->body[k - 1].text == "##") ||
                                               (k + 1 < mac->body.size() && mac->body[k + 1].text == "##");
                        if (nearPaste) {
                            if (charge(args[size_t(p)].size(), name->line)) return;
                            if (args[size_t(p)].empty()) {  // an empty argument next to ## is a placemarker (C99 6.10.3.3)
                                Token pm;
                                pm.kind = Tok::Punct, pm.line = name->line, pm.placemarker = true;
                                body.push_back(pm);
                            }
                            for (Token a : args[size_t(p)]) a.line = name->line, body.push_back(a);
                        } else {
                            std::vector<Token> ex;
                            expand(args[size_t(p)], ex, hide, depth + 1);
                            if (charge(ex.size() + 1, name->line)) return;
                            for (Token a : ex) a.line = name->line, body.push_back(a);
                        }
                    }
                    if (charge(body.size(), name->line)) return;
                    body = paste(body);
                    hide.insert(name->text);
                    expand(body, out, hide, depth + 1);
                    hide.erase(name->text);
                    nameLast = endsWithName_;
                    at = j;
                }
                // The expansion ends with the name of a function-like macro (not one met inside its own expansion) and a "(" follows in
                // the input: that name is called with it.
                if (nameLast && out.size() > before && out.back().kind == Tok::Ident && !out.back().painted && !hide.count(out.back().text) &&
                    at < in.size() && in[at].kind == Tok::Punct && in[at].text == "(") {
                    auto fm = defines_.find(out.back().text);
                    if (fm != defines_.end() && fm->second.function) {
                        Token next = std::move(out.back());
                        out.pop_back();
                        carried = std::move(next);
                        name = &carried;
                        mac = &fm->second;
                        continue;
                    }
                }
                break;
            }
            ends.v = nameLast && at >= in.size();
            i = std::max(at, i + 1);
        }
    }

    std::vector<Token> paste(const std::vector<Token>& in) {
        std::vector<Token> out;
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i].kind == Tok::Punct && in[i].text == "##" && !in[i].placemarker && !out.empty() && i + 1 < in.size()) {
                if (out.back().placemarker || in[i + 1].placemarker) {  // a placemarker pastes to the other operand
                    if (out.back().placemarker) out.back() = in[i + 1];
                    ++i;
                    continue;
                }
                std::string text = out.back().text + in[i + 1].text;
                std::string l;
                std::vector<Token> re = lex(text, l);
                int line = out.back().line;
                std::vector<Token> good;
                for (Token& r : re)
                    if (r.kind != Tok::End && r.kind != Tok::Newline) r.line = line, good.push_back(r);
                // The result must be ONE valid token (C99 6.10.3.3): `1 ## F` is "1F", `+ ## -` is two tokens.
                if (good.size() != 1 || good[0].kind == Tok::Bad || !l.empty())
                    error(line, "pasting \"" + out.back().text + "\" and \"" + in[i + 1].text + "\" does not give a valid preprocessing token");
                out.pop_back();
                for (Token& r : good) out.push_back(r);
                ++i;
                continue;
            }
            out.push_back(in[i]);
        }
        std::erase_if(out, [](const Token& x) { return x.placemarker; });
        return out;
    }

    // #if expression evaluation (integers only).
    struct Eval {
        const std::vector<Token>& t;
        size_t i = 0;
        bool bad = false;
        int skip = 0;                    // > 0 inside the operand a && / || short-circuits away: nothing in it is evaluated
        const Token* undef = nullptr;    // first undefined identifier that was evaluated
        const Token* peek() { return i < t.size() ? &t[i] : nullptr; }
        bool is(const char* p) { return peek() && peek()->kind == Tok::Punct && peek()->text == p; }
        int nest = 0;  // depth of primary(): "(((...", "!!!..." in a #if cannot overflow the stack
        struct Nest {
            int& n;
            explicit Nest(int& v) : n(v) { ++n; }
            ~Nest() { --n; }
        };
        int64_t primary() {
            Nest guard(nest);
            if (nest > 256) return bad = true, 0;
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
            if (is("-")) return ++i, int64_t(0 - uint64_t(primary()));
            if (is("+")) return ++i, primary();
            if (is("~")) return ++i, ~primary();
            if (k->kind == Tok::IntLit || k->kind == Tok::UintLit || k->kind == Tok::BoolLit) return ++i, int64_t(k->ival);
            if (k->kind == Tok::Ident) return ++i, (!skip && !undef ? (undef = k) : nullptr), 0;  // undefined identifier: 0
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
                const bool skipped = (op == "&&" && !l) || (op == "||" && l);
                skip += skipped;
                int64_t r = expr(p + 1);
                skip -= skipped;
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
                // Wrapping arithmetic: signed overflow is undefined in C++ (and INT64_MIN / -1 traps).
                else if (op == "<<") l = r >= 0 && r < 64 ? int64_t(uint64_t(l) << r) : 0;
                else if (op == ">>") l = r >= 0 && r < 64 ? l >> r : 0;
                else if (op == "+") l = int64_t(uint64_t(l) + uint64_t(r));
                else if (op == "-") l = int64_t(uint64_t(l) - uint64_t(r));
                else if (op == "*") l = int64_t(uint64_t(l) * uint64_t(r));
                else if (op == "/") l = r == -1 ? int64_t(0 - uint64_t(l)) : r ? l / r : (skip ? 0 : (bad = true, 0));
                else if (op == "%") l = r == -1 ? 0 : r ? l % r : (skip ? 0 : (bad = true, 0));
            }
            return l;
        }
    };

    static bool isPredefined(const std::string& n) { return n == "__LINE__" || n == "__FILE__" || n == "__VERSION__"; }

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
                if (j >= line.size() || line[j].kind != Tok::Ident || (paren && (j + 1 >= line.size() || line[j + 1].text != ")"))) {
                    if (strict("defined", line[0].line, "'defined' needs an identifier" + std::string(paren ? " and a closing ')'" : "")))
                        return false;
                    pre.push_back(n);  // 0
                    i = j - 1;
                    continue;
                }
                n.ival = defines_.count(line[j].text) || isPredefined(line[j].text) ? 1 : 0;
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
        // GLSL ES: an identifier that is not a macro is an error, not 0 (only when it is evaluated: `0 && AAA` is valid).
        if (e.undef && strict("undefined_identifier", line[0].line, "'" + e.undef->text + "' : undefined identifier in a preprocessor expression"))
            return false;
        if (e.bad || e.i != ex.size()) error(line[0].line, "invalid #if expression");
        return v != 0;
    }

    void directive(const std::vector<Token>& line, bool first) {
        const int ln = line[0].line;
        if (line.size() == 1) return;  // null directive
        const std::string d = line[1].text;
        // A conditional ends the wait for the "(" of a call, as every other directive does (cpp: `F(0)` / `#if 1` / `(1)` leaves `A (1)`).
        if (d == "if" || d == "ifdef" || d == "ifndef" || d == "elif" || d == "else" || d == "endif") flushPending(true);
        if (d == "if" || d == "ifdef" || d == "ifndef") {
            bool parent = active();
            bool v = false;
            if (parent) {
                if (d == "if") v = evalIf(line, 2);
                else if (line.size() < 3 || line[2].kind != Tok::Ident) strict("ifdef_name", ln, "#" + d + " needs a name");
                else {
                    if (line.size() > 3) strict("extra_tokens", ln, "#" + d + ": extra tokens after the name");
                    v = (defines_.count(line[2].text) != 0 || isPredefined(line[2].text)) == (d == "ifdef");
                }
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
            if (conds_.empty()) {
                strict("unbalanced_conditional", ln, "#else without #if");
                return;
            }
            Cond& c = conds_.back();
            if (c.sawElse) return error(ln, "duplicate #else");
            if (line.size() > 2 && strict("extra_tokens", ln, "#else: extra tokens")) return;
            c.sawElse = true;
            c.active = c.parentActive && !c.taken;
            c.taken = true;
            return;
        }
        if (d == "endif") {
            if (conds_.empty()) {
                strict("unbalanced_conditional", ln, "#endif without #if");
                return;
            }
            if (line.size() > 2 && strict("extra_tokens", ln, "#endif: extra tokens")) return;
            conds_.pop_back();
            return;
        }
        if (!active()) return;
        flushPending(true);
        if (d == "version") {
            if (!first) strict("version_position", ln, "#version must be the first directive");
            if (line.size() < 3 || line[2].kind != Tok::IntLit) return error(ln, "bad #version");
            res_.version = int(line[2].ival);
            if ((line.size() > 4 || (line.size() == 4 && line[3].text != "es")) &&
                strict("version_profile", ln, "#version: only 'es' may follow the number"))
                return;
            if (line.size() < 4 && res_.version >= 300 &&
                strict("version_profile", ln, "#version " + std::to_string(res_.version) + " needs 'es'"))
                return;
            res_.es = true;
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
                if (name.rfind("GL_", 0) == 0 && strict("reserved_macro", ln, "'" + name + "' : reserved macro name")) return;
            }
            if (isPredefined(name)) {
                strict("predefined_macro", ln, "'" + name + "' : predefined macro");  // lenient: the directive is ignored
                return;
            }
            Macro m;
            size_t i = 3;
            if (i < line.size() && line[i].text == "(" && !line[i].spaceBefore && line[i].kind == Tok::Punct) {
                m.function = true;
                ++i;
                // ( [ident {, ident}] ) : identifiers separated by commas, all different, closed by ')'
                if (i < line.size() && line[i].text == ")") ++i;
                else {
                    for (;;) {
                        if (i >= line.size() || line[i].kind != Tok::Ident) {
                            strict("macro_params", ln, "#define: macro parameter name expected");
                            return;  // lenient: the directive is ignored
                        }
                        for (const std::string& q : m.params)
                            if (q == line[i].text) {
                                strict("macro_params", ln, "#define: duplicate macro parameter '" + q + "'");
                                return;
                            }
                        m.params.push_back(line[i++].text);
                        if (i < line.size() && line[i].text == ",") { ++i; continue; }
                        if (i < line.size() && line[i].text == ")") { ++i; break; }
                        strict("macro_params", ln, "#define: ',' or ')' expected in the macro parameters");
                        return;
                    }
                }
            }
            for (; i < line.size(); ++i) m.body.push_back(line[i]);
            auto prev = defines_.find(name);
            if (prev != defines_.end()) {  // a redefinition must be identical (same parameters, same tokens, same spacing)
                const Macro& o = prev->second;
                bool same = o.function == m.function && o.params == m.params && o.body.size() == m.body.size();
                for (size_t k = 0; same && k < m.body.size(); ++k)
                    same = o.body[k].kind == m.body[k].kind && o.body[k].text == m.body[k].text &&
                           (k == 0 || o.body[k].spaceBefore == m.body[k].spaceBefore);
                if (!same && strict("macro_redefinition", ln, "'" + name + "' : redefinition of a macro with a different definition")) return;
            }
            defines_[name] = m;
            return;
        }
        if (d == "undef") {
            if (line.size() < 3 || line[2].kind != Tok::Ident) {
                strict("undef", ln, "#undef needs a macro name");
                return;
            }
            if (line.size() > 3 && strict("extra_tokens", ln, "#undef: extra tokens after the name")) return;
            if (isPredefined(line[2].text) || line[2].text.rfind("GL_", 0) == 0) {
                strict("undef", ln, "'" + line[2].text + "' : cannot undefine a predefined macro");
                return;
            }
            defines_.erase(line[2].text);
            return;
        }
        if (d == "extension") {
            // #extension name : behavior
            if (line.size() != 5 || line[2].kind != Tok::Ident || line[3].text != ":" || line[4].kind != Tok::Ident) {
                strict("extension_syntax", ln, "#extension: 'name : behavior' expected");
                return;
            }
            const std::string& name = line[2].text;
            const std::string& beh = line[4].text;
            if (beh != "require" && beh != "enable" && beh != "warn" && beh != "disable") {
                strict("extension_syntax", ln, "#extension: unknown behavior '" + beh + "'");
                return;
            }
            if (sawToken_ && strict("extension_position", ln, "#extension must come before any non-preprocessor token")) return;
            if (name == "all" && (beh == "require" || beh == "enable") &&
                strict("extension_all", ln, "#extension all: only 'warn' and 'disable' are allowed"))
                return;
            bool known = name == "all";
            for (const char* e : supportedExtensions()) known |= name == e;
            for (const char* e : kAcceptedExtensions) known |= name == e;
            if (!known && beh == "require" && strict("extension_unsupported", ln, "extension '" + name + "' is not supported")) return;
            if (!known) log_ += "WARNING: 0:" + std::to_string(ln) + ": extension '" + name + "' is not supported\n";
            if (beh == "disable") res_.extensions.erase(name);
            else res_.extensions.insert(name);
            return;
        }
        if (d == "pragma") {
            if (line.size() == 7 && line[2].text == "STDGL" && line[3].text == "invariant" && line[4].text == "(" &&
                line[5].text == "all" && line[6].text == ")")
                res_.invariantAll = true;
            return;
        }
        if (d == "line") {
            // #line line [source-string-number] : plain decimal integers
            // The operands are macro-expanded first (`#define A 4` / `#line A`).
            std::vector<Token> args(line.begin() + 2, line.end()), ex;
            std::set<std::string> hide;
            expand(args, ex, hide);
            auto num = [&](size_t k) { return k < ex.size() && ex[k].kind == Tok::IntLit; };
            if (!num(0) || ex.size() > 2 || (ex.size() == 2 && !num(1))) {
                strict("line_syntax", ln, "#line: 'line [source-string-number]' expected");
                return;
            }
            lineDelta_ = int(ex[0].ival) - (ln - lineDelta_ + 1);
            if (ex.size() == 2) fileNo_ = int(ex[1].ival);
            return;
        }
        if (d == "error") {
            std::string msg;
            for (size_t i = 2; i < line.size(); ++i) msg += (i > 2 ? " " : "") + line[i].text;
            return error(ln, "#error " + msg);
        }
        strict("unknown_directive", ln, "unknown directive #" + d);
    }

    std::string& log_;
    std::map<std::string, Macro> defines_;
    std::vector<Cond> conds_;
    std::vector<Token> pending_;
    PpResult res_;
    bool ok_ = true;
    int lineDelta_ = 0;       // #line: added to the line of every following token
    int fileNo_ = 0;          // #line's source string number: what __FILE__ expands to
    bool sawToken_ = false;   // a non-directive token has been seen (#extension must come before)
};

}  // namespace

void setCaps(const Caps& c) {
    gFramebufferFetch = c.framebufferFetch;
    gCubeMapArray = c.cubeMapArray;
}

namespace {
std::atomic<int> gStrictOverride{-1};
}  // namespace

bool strictMode() {
    const int o = gStrictOverride.load();
    if (o >= 0) return o != 0;
    static const bool env = [] {
        const char* e = std::getenv("VPGLES_STRICT_GLSL");
        return e && e[0] && e[0] != '0';
    }();
    return env;
}

void setStrictMode(int mode) { gStrictOverride = mode; }

Caps caps() {
    Caps c;
    c.framebufferFetch = gFramebufferFetch;
    c.cubeMapArray = gCubeMapArray;
    return c;
}

PpResult runPreprocessor(const std::string& source, Stage stage, std::string& log) {
    vpir::StackGuard stackGuard;  // (the nested expansion stops before the stack of this thread does)
    Pp p(stage, log);
    try {
        return p.run(source);
    } catch (const std::bad_alloc&) {
        log += "ERROR: 0:0: out of memory in the preprocessor\n";
        PpResult r;
        r.ok = false;
        Token end;
        end.kind = Tok::End;
        r.tokens.push_back(end);
        return r;
    }
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
