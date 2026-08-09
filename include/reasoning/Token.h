#pragma once
#include <string>

namespace neurocore::reasoning {

enum class TokenType {
    NUMBER,
    IDENTIFIER,
    PLUS,
    MINUS,
    MULTIPLY,
    DIVIDE,
    LPAREN,
    RPAREN,
    EQUALS,
    END_OF_FILE,
    UNKNOWN
};

struct Token {
    TokenType type;
    std::string value;

    Token(TokenType t, const std::string& v = "") : type(t), value(v) {}
};

} // namespace neurocore::reasoning
