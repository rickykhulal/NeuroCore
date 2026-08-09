#pragma once
#include "core/EventBus.h"
#include "memory/KnowledgeGraph.h"
#include "learning/Network.h"
#include "reasoning/InferenceEngine.h"
#include "explain/ExplanationTrace.h"
#include <memory>
#include <string>

namespace neurocore::app {

// Application is the single place where "what NeuroCore X can do" lives,
// independent of HOW it's presented (console text vs. a web request).
// It owns the real, unmodified subsystems (KnowledgeGraph, Network,
// InferenceEngine, EventBus) -- no reimplementation, no duplicate
// intelligence. Every method returns a plain string of the same output
// that used to go straight to std::cout, so the console REPL's visible
// behavior is fully reproducible, and a web layer can just wrap that
// string in JSON.
class Application {
public:
    Application();

    // Executes one command line (e.g. "ask 2 + 3") and returns the
    // full text output, exactly as the console would have printed it.
    std::string handle(const std::string& line);

    // Small structured snapshot of current system state, for the web
    // dashboard sidebar (concept/relationship/example counts, etc.).
    // Returned as a ready-to-send JSON string to keep the web layer
    // simple.
    std::string statsJson() const;

private:
    std::shared_ptr<core::EventBus> eventBus;
    std::shared_ptr<memory::KnowledgeGraph> graph;
    std::shared_ptr<learning::Network> network;
    std::unique_ptr<reasoning::InferenceEngine> engine;
    explain::ExplanationTrace lastTrace;

    // Same command dispatch as the console Repl, but writes into a
    // string stream instead of std::cout, and returns the result.
    std::string dispatch(const std::string& line);
};

} // namespace neurocore::app
