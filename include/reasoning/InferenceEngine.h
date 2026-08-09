#pragma once
#include "memory/KnowledgeGraph.h"
#include "learning/Network.h"
#include "explain/ExplanationTrace.h"
#include "Parser.h"
#include <memory>
#include <string>

namespace neurocore::reasoning {

class InferenceEngine {
public:
    InferenceEngine(std::shared_ptr<memory::KnowledgeGraph> kg, 
                    std::shared_ptr<learning::Network> ln) 
        : graph(kg), network(ln) {}

    std::string ask(const std::string& question, explain::ExplanationTrace& trace);

private:
    std::shared_ptr<memory::KnowledgeGraph> graph;
    std::shared_ptr<learning::Network> network;
    Parser parser;
};

} // namespace neurocore::reasoning
