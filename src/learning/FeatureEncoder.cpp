#include "learning/FeatureEncoder.h"
#include <sstream>
#include <cmath>

namespace neurocore::learning {

std::optional<std::vector<double>> FeatureEncoder::encodeExpression(const std::string& expr) {
    std::stringstream ss(expr);
    double a, b;
    std::string op;
    if (ss >> a >> op >> b) {
        // Only hand off to the Learning Engine for the operator it was
        // actually trained on. Anything else (e.g. "100 * 2") must fall
        // through to the Parser for correct real arithmetic instead of
        // being silently mis-answered by a network with no idea how to
        // multiply.
        if (op == "+") {
            return std::vector<double>{a, b};
        }
    }
    return std::nullopt;
}

std::optional<std::vector<double>> FeatureEncoder::encodeTarget(const std::string& value) {
    try {
        double v = std::stod(value);
        return std::vector<double>{v};
    } catch (...) {
        return std::nullopt;
    }
}

std::string FeatureEncoder::decode(const std::vector<double>& output) {
    if (output.empty()) return "?";
    double rounded = std::round(output[0]);
    return std::to_string(static_cast<long long>(rounded));
}

} // namespace neurocore::learning
