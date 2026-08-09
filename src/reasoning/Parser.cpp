#include "reasoning/Parser.h"
#include <cctype>
#include <stdexcept>

namespace neurocore::reasoning {

std::vector<Token> Parser::tokenize(const std::string& input) {
    std::vector<Token> result;
    for (size_t i = 0; i < input.length(); ++i) {
        if (std::isspace(input[i])) continue;
        if (std::isdigit(input[i])) {
            std::string val;
            while (i < input.length() && (std::isdigit(input[i]) || input[i] == '.')) {
                val += input[i++];
            }
            --i;
            result.emplace_back(TokenType::NUMBER, val);
        } else if (std::isalpha(input[i])) {
            std::string val;
            while (i < input.length() && std::isalnum(input[i])) {
                val += input[i++];
            }
            --i;
            // Identifiers are tokenized (needed elsewhere, e.g. fact-style
            // text), but the arithmetic parser below will REJECT them as
            // invalid inside a numeric expression rather than silently
            // treating them as 0.
            result.emplace_back(TokenType::IDENTIFIER, val);
        } else if (input[i] == '+') result.emplace_back(TokenType::PLUS, "+");
        else if (input[i] == '-') result.emplace_back(TokenType::MINUS, "-");
        else if (input[i] == '*') result.emplace_back(TokenType::MULTIPLY, "*");
        else if (input[i] == '/') result.emplace_back(TokenType::DIVIDE, "/");
        else if (input[i] == '(') result.emplace_back(TokenType::LPAREN, "(");
        else if (input[i] == ')') result.emplace_back(TokenType::RPAREN, ")");
        else if (input[i] == '=') result.emplace_back(TokenType::EQUALS, "=");
        else {
            // Any character we don't recognize (e.g. '?', '#', '@') must
            // NOT be silently skipped -- that would let malformed input
            // quietly produce a wrong answer. Mark it explicitly UNKNOWN
            // so parse() below will error out instead of guessing.
            result.emplace_back(TokenType::UNKNOWN, std::string(1, input[i]));
        }
    }
    result.emplace_back(TokenType::END_OF_FILE);
    return result;
}

std::unique_ptr<ASTNode> Parser::parse(const std::string& input) {
    tokens = tokenize(input);
    pos = 0;
    auto node = expression();
    if (peek().type != TokenType::END_OF_FILE) {
        throw std::runtime_error("Trailing tokens");
    }
    return node;
}

std::unique_ptr<ASTNode> Parser::expression() {
    auto node = term();
    while (peek().type == TokenType::PLUS || peek().type == TokenType::MINUS) {
        Token t = advance();
        auto newNode = std::make_unique<ASTNode>(t);
        newNode->left = std::move(node);
        newNode->right = term();
        node = std::move(newNode);
    }
    return node;
}

std::unique_ptr<ASTNode> Parser::term() {
    auto node = factor();
    while (peek().type == TokenType::MULTIPLY || peek().type == TokenType::DIVIDE) {
        Token t = advance();
        auto newNode = std::make_unique<ASTNode>(t);
        newNode->left = std::move(node);
        newNode->right = factor();
        node = std::move(newNode);
    }
    return node;
}

std::unique_ptr<ASTNode> Parser::factor() {
    Token t = peek();
    if (t.type == TokenType::NUMBER) {
        // FIX: only NUMBER is a valid arithmetic leaf. IDENTIFIER used to
        // be accepted here too, which let something like "9+a" silently
        // evaluate "a" as if it were a number (defaulting to 0 downstream).
        return std::make_unique<ASTNode>(advance());
    } else if (t.type == TokenType::LPAREN) {
        consume(TokenType::LPAREN);
        auto node = expression();
        consume(TokenType::RPAREN);
        return node;
    } else if (t.type == TokenType::IDENTIFIER) {
        throw std::runtime_error("Unexpected identifier '" + t.value +
            "' in arithmetic expression");
    } else if (t.type == TokenType::UNKNOWN) {
        throw std::runtime_error("Unrecognized character '" + t.value +
            "' in expression");
    }
    throw std::runtime_error("Unexpected token in factor");
}

void Parser::consume(TokenType type) {
    if (peek().type == type) {
        advance();
    } else {
        throw std::runtime_error("Expected token type not found");
    }
}

Token Parser::peek() {
    if (pos < tokens.size()) return tokens[pos];
    return Token(TokenType::END_OF_FILE);
}

Token Parser::advance() {
    if (pos < tokens.size()) return tokens[pos++];
    return Token(TokenType::END_OF_FILE);
}

double Parser::evaluate(const ASTNode* node) {
    if (!node) {
        throw std::runtime_error("Cannot evaluate null expression node");
    }
    if (node->token.type == TokenType::NUMBER) {
        return std::stod(node->token.value);
    }
    if (node->token.type != TokenType::PLUS &&
        node->token.type != TokenType::MINUS &&
        node->token.type != TokenType::MULTIPLY &&
        node->token.type != TokenType::DIVIDE) {
        // Safeguard: should be unreachable now that factor() rejects
        // IDENTIFIER/UNKNOWN leaves, but kept so evaluate() never silently
        // returns 0 for an unexpected node type.
        throw std::runtime_error("Cannot evaluate non-numeric/non-operator node");
    }

    double left = evaluate(node->left.get());
    double right = evaluate(node->right.get());

    switch (node->token.type) {
        case TokenType::PLUS: return left + right;
        case TokenType::MINUS: return left - right;
        case TokenType::MULTIPLY: return left * right;
        case TokenType::DIVIDE:
            if (right == 0.0) {
                throw std::runtime_error("Division by zero");
            }
            return left / right;
        default:
            throw std::runtime_error("Unsupported operator");
    }
}

} // namespace neurocore::reasoning
