#include "reasoning/InferenceEngine.h"
#include "learning/FeatureEncoder.h"
#include <sstream>

namespace neurocore::reasoning {

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(' ');
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(' ');
    return s.substr(start, end - start + 1);
}

static std::string joinAnswers(const std::vector<std::string>& answers) {
    std::string result;
    for (size_t i = 0; i < answers.size(); ++i) {
        result += answers[i];
        if (i + 1 < answers.size()) result += ", ";
    }
    return result;
}

static std::string joinPath(const std::vector<std::string>& path) {
    std::string result;
    for (size_t i = 0; i < path.size(); ++i) {
        result += path[i];
        if (i + 1 < path.size()) result += " -> ";
    }
    return result;
}

// Builds the provenance-aware trace text for a successful graph answer.
static std::string describeSource(const std::vector<std::string>& hopPath, const std::string& source) {
    if (hopPath.size() <= 1) {
        // 0-hop / direct match: report where the fact itself came from.
        return "Directly taught fact (source: " + source + ")";
    }
    return "Inferred via " + std::to_string(hopPath.size() - 1) + "-hop chain: " + joinPath(hopPath);
}

std::string InferenceEngine::ask(const std::string& question, explain::ExplanationTrace& trace) {
    std::string trimmedQuestion = trim(question);

    // 1. EXACT GRAPH LOOKUP -- contradiction check, then provenance-aware answer.
    {
        if (graph->hasContradiction(trimmedQuestion, "equals")) {
            auto allAnswers = graph->getAllAnswers(trimmedQuestion, "equals");
            trace.addStep("Contradiction",
                "Conflicting facts taught for '" + trimmedQuestion + "': " +
                joinAnswers(allAnswers) + ". Returning most recently taught value.",
                0.5);
            return allAnswers.back() + "  (CONFLICTING: " + joinAnswers(allAnswers) + ")";
        }

        auto exactWithPath = graph->findMultiHopWithPath(trimmedQuestion, "equals");
        if (exactWithPath) {
            const std::string& answer = exactWithPath->first;
            const std::vector<std::string>& path = exactWithPath->second;

            std::string sourceLabel = "user-taught";
            auto directRel = graph->findDirectRelationship(trimmedQuestion, "equals");
            if (directRel) sourceLabel = directRel->source;

            trace.addStep("GraphLookup",
                trimmedQuestion + " = " + answer + " -- " + describeSource(path, sourceLabel),
                1.0);
            return answer;
        }
    }

    // 2. FACT-STYLE LOOKUP ("Subject Property?") -- same contradiction + provenance handling.
    {
        std::stringstream ss(trimmedQuestion);
        std::string subject, property;
        if (ss >> subject >> property) {
            if (graph->hasContradiction(subject, property)) {
                auto allAnswers = graph->getAllAnswers(subject, property);
                trace.addStep("Contradiction",
                    "Conflicting facts taught for '" + subject + " " + property + "': " +
                    joinAnswers(allAnswers) + ". Returning most recently taught value.",
                    0.5);
                return allAnswers.back() + "  (CONFLICTING: " + joinAnswers(allAnswers) + ")";
            }

            auto resultWithPath = graph->findMultiHopWithPath(subject, property);
            if (resultWithPath) {
                const std::string& answer = resultWithPath->first;
                const std::vector<std::string>& path = resultWithPath->second;

                std::string sourceLabel = "user-taught";
                auto directRel = graph->findDirectRelationship(subject, property);
                if (directRel) sourceLabel = directRel->source;

                trace.addStep("GraphLookup",
                    subject + " " + property + " " + answer + " -- " + describeSource(path, sourceLabel),
                    1.0);
                return answer;
            }
        }
    }

    // 3. LEARNING ENGINE -- real pattern generalization.
    if (network && network->isTrained()) {
        auto features = learning::FeatureEncoder::encodeExpression(trimmedQuestion);
        if (features) {
            std::vector<double> output = network->predict(*features);
            std::string decoded = learning::FeatureEncoder::decode(output);
            trace.addStep("LearningEngine",
                "No exact taught fact; Network predicted a learned pattern for '" +
                trimmedQuestion + "' -> " + decoded,
                0.75);
            return decoded;
        }
    }

    // 4. PARSER (pure arithmetic calculation) -- last resort fallback.
    try {
        auto ast = parser.parse(trimmedQuestion);
        if (ast) {
            double val = parser.evaluate(ast.get());
            trace.addStep("Parser",
                "No taught fact or learned pattern found; evaluated as real arithmetic expression",
                0.6);
            return std::to_string(val);
        }
    } catch (...) {
        // Not a valid arithmetic expression either -- fall through.
    }

    trace.addStep("None", "No answer found in Graph, Network, or Parser", 0.0);
    return "I don't know.";
}

} // namespace neurocore::reasoning
