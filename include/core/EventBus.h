#pragma once
#include "Event.h"
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

namespace neurocore::core {

class EventBus {
public:
    using EventHandler = std::function<void(const Event&)>;

    void subscribe(const std::string& eventType, EventHandler handler);
    void publish(const Event& event);

private:
    std::unordered_map<std::string, std::vector<EventHandler>> subscribers;
};

} // namespace neurocore::core
