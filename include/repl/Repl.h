#pragma once
#include "core/EventBus.h"
#include "memory/KnowledgeGraph.h"
#include "reasoning/InferenceEngine.h"
#include "learning/Network.h"
#include "explain/ExplanationTrace.h"
#include <memory>
#include <string>

namespace neurocore::repl {

class Repl {
public:
    Repl();
    void run();

private:
    void handleCommand(const std::string& line);
    void printHelp();

    std::shared_ptr<core::EventBus> eventBus;
    std::shared_ptr<memory::KnowledgeGraph> graph;
    std::shared_ptr<learning::Network> network;
    std::unique_ptr<reasoning::InferenceEngine> engine;
    explain::ExplanationTrace lastTrace;
};

} // namespace neurocore::repl
