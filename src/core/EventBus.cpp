#include "core/EventBus.h"

namespace neurocore::core {

void EventBus::subscribe(const std::string& eventType, EventHandler handler) {
    subscribers[eventType].push_back(handler);
}

void EventBus::publish(const Event& event) {
    auto it = subscribers.find(event.type);
    if (it != subscribers.end()) {
        for (const auto& handler : it->second) {
            handler(event);
        }
    }
}

} // namespace neurocore::core
