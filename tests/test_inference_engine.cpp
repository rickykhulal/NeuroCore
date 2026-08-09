#include "reasoning/InferenceEngine.h"
#include "memory/KnowledgeGraph.h"
#include "learning/Network.h"
#include "explain/ExplanationTrace.h"
#include <cassert>
#include <iostream>

using namespace neurocore;

int main() {
    auto bus = std::make_shared<core::EventBus>();
    auto kg = std::make_shared<memory::KnowledgeGraph>(bus);
    auto net = std::make_shared<learning::Network>();
    reasoning::InferenceEngine engine(kg, net);
    explain::ExplanationTrace trace;

    // Test Graph Lookup
    kg->addRelationship("Sky", "isA", "Blue");
    std::string res1 = engine.ask("Sky isA", trace);
    assert(res1 == "Blue");
    assert(trace.toString().find("GraphLookup") != std::string::npos);

    // Test Parser
    trace.clear();
    std::string res2 = engine.ask("2 + 2", trace);
    assert(res2 == "4.000000");
    assert(trace.toString().find("Parser") != std::string::npos);

    // Test Network Fallback
    trace.clear();
    net->train({}, {});
    std::string res3 = engine.ask("Complex Pattern", trace);
    assert(res3.find("LearnedResult") != std::string::npos);
    assert(trace.toString().find("LearningEngine") != std::string::npos);

    std::cout << "InferenceEngine tests passed!" << std::endl;
    return 0;
}
