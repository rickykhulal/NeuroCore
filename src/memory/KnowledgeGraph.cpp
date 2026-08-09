#include "memory/KnowledgeGraph.h"
#include <fstream>
#include <sstream>
#include <queue>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <filesystem>

namespace neurocore::memory {

void KnowledgeGraph::addConcept(const std::string& name) {
    if (concepts.find(name) == concepts.end()) {
        concepts[name] = Concept(name);
    }
}

void KnowledgeGraph::addRelationship(const std::string& from, const std::string& type, const std::string& to,
                                      double confidence, const std::string& source) {
    addConcept(from);
    addConcept(to);

    bool conflictFound = false;
    auto it = adjList.find(from);
    if (it != adjList.end()) {
        for (auto& existingRel : it->second) {
            if (existingRel.type == type && existingRel.toConcept != to) {
                existingRel.conflicted = true;
                conflictFound = true;
            }
        }
    }

    Relationship rel(from, type, to, confidence, source);
    rel.conflicted = conflictFound;
    adjList[from].push_back(rel);

    if (eventBus) {
        core::Event event("FactAdded");
        event.set("from", from);
        event.set("type", type);
        event.set("to", to);
        event.set("confidence", std::to_string(confidence));
        eventBus->publish(event);

        if (conflictFound) {
            core::Event conflictEvent("ContradictionDetected");
            conflictEvent.set("from", from);
            conflictEvent.set("type", type);
            conflictEvent.set("newValue", to);
            eventBus->publish(conflictEvent);
        }
    }
}

std::optional<std::string> KnowledgeGraph::findDirect(const std::string& from, const std::string& type) const {
    auto it = adjList.find(from);
    if (it != adjList.end()) {
        for (const auto& rel : it->second) {
            if (rel.type == type) {
                return rel.toConcept;
            }
        }
    }
    return std::nullopt;
}

std::optional<Relationship> KnowledgeGraph::findDirectRelationship(const std::string& from, const std::string& type) const {
    auto it = adjList.find(from);
    if (it != adjList.end()) {
        for (const auto& rel : it->second) {
            if (rel.type == type) {
                return rel;
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> KnowledgeGraph::findMultiHop(const std::string& from, const std::string& type) const {
    auto direct = findDirect(from, type);
    if (direct) return direct;

    std::queue<std::string> q;
    std::set<std::string> visited;
    q.push(from);
    visited.insert(from);

    while (!q.empty()) {
        std::string current = q.front();
        q.pop();

        auto it = adjList.find(current);
        if (it != adjList.end()) {
            for (const auto& rel : it->second) {
                if (rel.type == type) {
                    return rel.toConcept;
                }
                if (rel.type == "isA" && visited.find(rel.toConcept) == visited.end()) {
                    visited.insert(rel.toConcept);
                    q.push(rel.toConcept);
                }
            }
        }
    }

    return std::nullopt;
}

std::optional<std::pair<std::string, std::vector<std::string>>>
KnowledgeGraph::findMultiHopWithPath(const std::string& from, const std::string& type) const {
    auto direct = findDirect(from, type);
    if (direct) {
        return std::make_pair(*direct, std::vector<std::string>{});
    }

    std::queue<std::string> q;
    std::set<std::string> visited;
    std::unordered_map<std::string, std::string> parent;

    q.push(from);
    visited.insert(from);

    while (!q.empty()) {
        std::string current = q.front();
        q.pop();

        auto it = adjList.find(current);
        if (it != adjList.end()) {
            for (const auto& rel : it->second) {
                if (rel.type == type) {
                    std::vector<std::string> path;
                    std::string node = current;
                    while (node != from) {
                        path.push_back(node);
                        node = parent[node];
                    }
                    path.push_back(from);
                    std::reverse(path.begin(), path.end());
                    return std::make_pair(rel.toConcept, path);
                }
                if (rel.type == "isA" && visited.find(rel.toConcept) == visited.end()) {
                    visited.insert(rel.toConcept);
                    parent[rel.toConcept] = current;
                    q.push(rel.toConcept);
                }
            }
        }
    }

    return std::nullopt;
}

bool KnowledgeGraph::hasContradiction(const std::string& from, const std::string& type) const {
    auto it = adjList.find(from);
    if (it == adjList.end()) return false;

    for (const auto& rel : it->second) {
        if (rel.type == type && rel.conflicted) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> KnowledgeGraph::getAllAnswers(const std::string& from, const std::string& type) const {
    std::vector<std::string> answers;
    auto it = adjList.find(from);
    if (it == adjList.end()) return answers;

    for (const auto& rel : it->second) {
        if (rel.type == type) {
            answers.push_back(rel.toConcept);
        }
    }
    return answers;
}

bool KnowledgeGraph::save(const std::string& filename) const {
    // FIX: ofstream does NOT create missing parent directories on its
    // own. If "data/" doesn't exist relative to the current working
    // directory, the file silently fails to open and nothing gets
    // written -- previously this went completely unnoticed because the
    // caller printed a success message regardless of the outcome.
    std::filesystem::path filePath(filename);
    if (filePath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);
        // ec is intentionally not treated as fatal here -- if the
        // directory already exists, create_directories still succeeds
        // logically; we only truly fail below if the file itself can't
        // be opened for writing.
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& pair : concepts) {
        file << "C|" << pair.first << "\n";
    }
    for (const auto& pair : adjList) {
        for (const auto& rel : pair.second) {
            file << "R|" << rel.fromConcept << "|" << rel.type << "|" << rel.toConcept
                 << "|" << rel.confidence << "|" << (rel.conflicted ? 1 : 0)
                 << "|" << rel.source << "\n";
        }
    }

    return true;
}

bool KnowledgeGraph::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    concepts.clear();
    adjList.clear();

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string part;
        std::vector<std::string> parts;
        while (std::getline(ss, part, '|')) {
            parts.push_back(part);
        }

        if (parts.empty()) continue;

        if (parts[0] == "C" && parts.size() >= 2) {
            addConcept(parts[1]);
        } else if (parts[0] == "R" && parts.size() >= 5) {
            std::string src = (parts.size() >= 7) ? parts[6] : "loaded";
            addRelationship(parts[1], parts[2], parts[3], std::stod(parts[4]), src);
        }
    }

    return true;
}

} // namespace neurocore::memory
