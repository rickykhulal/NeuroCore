#include "explain/ExplanationTrace.h"
#include <sstream>
#include <iomanip>

namespace neurocore::explain {

void ExplanationTrace::addStep(const std::string& subsystem, const std::string& details, double confidence) {
    steps.push_back({subsystem, details, confidence});
}

void ExplanationTrace::clear() {
    steps.clear();
}

std::string ExplanationTrace::toString() const {
    if (steps.empty()) return "No reasoning trace available.";

    std::stringstream ss;
    ss << "Reasoning Trace:\n";
    for (const auto& step : steps) {
        ss << "  [" << step.subsystem << "] " << step.details 
           << " (Confidence: " << std::fixed << std::setprecision(2) << step.confidence << ")\n";
    }
    return ss.str();
}

} // namespace neurocore::explain
