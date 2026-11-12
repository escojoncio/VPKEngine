// Internal: tokens, lexer and preprocessor of the GLSL ES compiler.
#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "glsl.h"

namespace vpglsl {

enum class Tok : uint8_t { End, Ident, IntLit, UintLit, FloatLit, BoolLit, Punct, Newline, Hash, Bad };  // Bad: a character the language does not have

struct Token {
    Tok kind = Tok::End;
    std::string text;
    int line = 1;
    uint64_t ival = 0;   // IntLit/UintLit (wrapped to 32 bits by the parser)
    double fval = 0;     // FloatLit
    bool spaceBefore = false;
    bool placemarker = false;  // the empty argument next to a ## of a macro body (never leaves the substitution)
    bool painted = false;  // a macro name met inside its own expansion: never expanded again
};

// Splits one source into tokens (comments removed, line continuations joined). Newline tokens are
// kept so the preprocessor can see directive boundaries.
std::vector<Token> lex(const std::string& src, std::string& log);

struct PpResult {
    std::vector<Token> tokens;  // no Newline/Hash tokens
    int version = 100;
    bool es = true;
    std::set<std::string> extensions;  // enabled / required / warned
    int numViews = 0;                  // not set here (layout); kept for symmetry
    bool invariantAll = false;         // #pragma STDGL invariant(all)
    bool ok = true;
    std::vector<std::string> lenient;  // strict-mode rules let through (see ParseResult::lenient)
};

PpResult runPreprocessor(const std::string& source, Stage stage, std::string& log);

}  // namespace vpglsl
