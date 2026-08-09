#pragma once
#include <string>
#include <unordered_map>

namespace neurocore::core {

struct Event {
    std::string type;
    std::unordered_map<std::string, std::string> payload;

    Event(const std::string& t) : type(t) {}
    
    void set(const std::string& key, const std::string& value) {
        payload[key] = value;
    }

    std::string get(const std::string& key) const {
        auto it = payload.find(key);
        return it != payload.end() ? it->second : "";
    }
};

} // namespace neurocore::core
