#include "reasoning/Parser.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace neurocore::reasoning;

int main() {
    Parser p;

    auto ast1 = p.parse("1 + 2 * 3");
    assert(std::abs(p.evaluate(ast1.get()) - 7.0) < 0.001);

    auto ast2 = p.parse("(1 + 2) * 3");
    assert(std::abs(p.evaluate(ast2.get()) - 9.0) < 0.001);

    auto ast3 = p.parse("10 / 2 - 1");
    assert(std::abs(p.evaluate(ast3.get()) - 4.0) < 0.001);

    std::cout << "Parser tests passed!" << std::endl;
    return 0;
}
