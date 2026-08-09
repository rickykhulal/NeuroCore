#pragma once
#include <string>
#include <vector>

namespace neurocore::explain {

struct TraceStep {
    std::string subsystem;
    std::string details;
    double confidence;
};

class ExplanationTrace {
public:
    void addStep(const std::string& subsystem, const std::string& details, double confidence);
    void clear();
    std::string toString() const;

private:
    std::vector<TraceStep> steps;
};

} // namespace neurocore::explain
