#pragma once
#include "Concept.h"
#include "Relationship.h"
#include "core/EventBus.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <optional>
#include <memory>
#include <utility>

namespace neurocore::memory {

class KnowledgeGraph {
public:
    KnowledgeGraph(std::shared_ptr<core::EventBus> bus) : eventBus(bus) {}

    void addConcept(const std::string& name);
    void addRelationship(const std::string& from, const std::string& type, const std::string& to,
                          double confidence = 1.0, const std::string& source = "user-taught");

    std::optional<std::string> findDirect(const std::string& from, const std::string& type) const;
    std::optional<std::string> findMultiHop(const std::string& from, const std::string& type) const;

    std::optional<Relationship> findDirectRelationship(const std::string& from, const std::string& type) const;

    std::optional<std::pair<std::string, std::vector<std::string>>>
        findMultiHopWithPath(const std::string& from, const std::string& type) const;

    bool hasContradiction(const std::string& from, const std::string& type) const;
    std::vector<std::string> getAllAnswers(const std::string& from, const std::string& type) const;

    // NEW: both now return true on real success, false if the file
    // could not be opened/written/read -- so the caller (Repl) can
    // report an honest message instead of always claiming success.
    bool save(const std::string& filename) const;
    bool load(const std::string& filename);

    const std::unordered_map<std::string, Concept>& getConcepts() const { return concepts; }
    const std::unordered_map<std::string, std::vector<Relationship>>& getAdjacencyList() const { return adjList; }

private:
    std::unordered_map<std::string, Concept> concepts;
    std::unordered_map<std::string, std::vector<Relationship>> adjList;
    std::shared_ptr<core::EventBus> eventBus;
};

} // namespace neurocore::memory
