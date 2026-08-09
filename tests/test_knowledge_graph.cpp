#include "memory/KnowledgeGraph.h"
#include "core/EventBus.h"
#include <cassert>
#include <iostream>

using namespace neurocore;

int main() {
    auto bus = std::make_shared<core::EventBus>();
    memory::KnowledgeGraph kg(bus);

    bool eventFired = false;
    bus->subscribe("FactAdded", [&](const core::Event& e) {
        eventFired = true;
    });

    kg.addRelationship("Apple", "isA", "Fruit");
    assert(eventFired);
    assert(kg.getConcepts().size() == 2);
    
    auto direct = kg.findDirect("Apple", "isA");
    assert(direct.has_value() && *direct == "Fruit");

    kg.addRelationship("Fruit", "hasProperty", "Sweet");
    auto multi = kg.findMultiHop("Apple", "hasProperty");
    assert(multi.has_value() && *multi == "Sweet");

    kg.save("test_kg.txt");
    memory::KnowledgeGraph kg2(bus);
    kg2.load("test_kg.txt");
    assert(kg2.getConcepts().size() == 3);
    assert(kg2.findMultiHop("Apple", "hasProperty").has_value());

    std::cout << "KnowledgeGraph tests passed!" << std::endl;
    return 0;
}
