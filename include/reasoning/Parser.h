#pragma once
#include "Token.h"
#include <vector>
#include <string>
#include <memory>

namespace neurocore::reasoning {

struct ASTNode {
    Token token;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;

    ASTNode(Token t) : token(t) {}
};

class Parser {
public:
    Parser() = default;
    std::unique_ptr<ASTNode> parse(const std::string& input);
    double evaluate(const ASTNode* node);

private:
    std::vector<Token> tokenize(const std::string& input);
    std::unique_ptr<ASTNode> expression();
    std::unique_ptr<ASTNode> term();
    std::unique_ptr<ASTNode> factor();
    
    void consume(TokenType type);
    Token peek();
    Token advance();

    std::vector<Token> tokens;
    size_t pos = 0;
};

} // namespace neurocore::reasoning
