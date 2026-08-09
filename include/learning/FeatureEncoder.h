#pragma once
#include <string>
#include <vector>
#include <optional>

namespace neurocore::learning {

// Converts a simple two-operand expression like "1 + 3" into a numeric
// feature vector {1, 3} that the Network can consume, and converts a
// single-value target/result string like "4" into {4}.
//
// LIMITATION (documented on purpose): this v1 encoder only recognizes
// ONE operator -- '+' -- because that's the only pattern the demo
// dataset actually teaches the Network. It deliberately REFUSES to
// encode any other operator (*, -, /) so that expressions like
// "100 * 2" correctly fall through to the Parser for real arithmetic,
// instead of being silently (and wrongly) routed to a network that was
// never trained on multiplication. Extending this to multiple operators
// is a clearly scoped future improvement (would need the operator
// itself encoded as an input feature, e.g. one-hot).
class FeatureEncoder {
public:
    // Returns operands {a, b} ONLY if 'expr' is "<num> + <num>".
    // Returns std::nullopt for any other operator or malformed input.
    static std::optional<std::vector<double>> encodeExpression(const std::string& expr);

    // Parses a plain numeric string like "4" into {4}.
    static std::optional<std::vector<double>> encodeTarget(const std::string& value);

    // Converts a raw network output vector back into a display string
    // (rounded to the nearest integer, since this demo's taught facts
    // use whole numbers).
    static std::string decode(const std::vector<double>& output);
};

} // namespace neurocore::learning
