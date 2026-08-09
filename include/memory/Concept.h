#pragma once
#include <string>
#include <unordered_map>
#include <chrono>

namespace neurocore::memory {

struct Concept {
    std::string name;
    std::chrono::system_clock::time_point createdAt;
    std::unordered_map<std::string, std::string> metadata;

    Concept() = default;
    Concept(const std::string& n) : name(n), createdAt(std::chrono::system_clock::now()) {}
};

} // namespace neurocore::memory
