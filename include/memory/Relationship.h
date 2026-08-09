#pragma once
#include <string>
#include <chrono>

namespace neurocore::memory {

struct Relationship {
    std::string fromConcept;
    std::string toConcept;
    std::string type;
    double confidence;
    std::chrono::system_clock::time_point timestamp;

    // Set by KnowledgeGraph::addRelationship() when this relationship's
    // (fromConcept, type) pair conflicts with a different toConcept
    // already stored. Both relationships are kept, not overwritten.
    bool conflicted = false;

    // NEW: where this fact came from. Currently either "user-taught"
    // (added via the "teach" REPL command) or "loaded" (restored from a
    // save file). Multi-hop INFERRED answers are not stored as new
    // relationships -- their provenance is reported separately by
    // InferenceEngine as the chain of relationships it traversed live.
    std::string source = "user-taught";

    Relationship(const std::string& from, const std::string& relType, const std::string& to,
                 double conf = 1.0, const std::string& src = "user-taught")
        : fromConcept(from), toConcept(to), type(relType), confidence(conf),
          timestamp(std::chrono::system_clock::now()), source(src) {}
};

} // namespace neurocore::memory
