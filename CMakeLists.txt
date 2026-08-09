cmake_minimum_required(VERSION 3.10)
project(NeuroCoreX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Source files
set(SOURCES
    src/core/EventBus.cpp
    src/memory/KnowledgeGraph.cpp
    src/reasoning/Parser.cpp
    src/reasoning/InferenceEngine.cpp
    src/learning/DenseLayer.cpp
    src/learning/ActivationLayer.cpp
    src/learning/Network.cpp
    src/learning/FeatureEncoder.cpp
    src/explain/ExplanationTrace.cpp
    src/repl/Repl.cpp
    src/main.cpp
)

# Main executable
add_executable(NeuroCoreX ${SOURCES})

# ------------------------------------------------------------------
# Web UI executable (separate target -- does NOT modify NeuroCoreX
# above in any way; both share the real KnowledgeGraph/Network/Parser/
# InferenceEngine code, wired independently through Application.cpp)
# ------------------------------------------------------------------
set(WEB_SOURCES
    src/core/EventBus.cpp
    src/memory/KnowledgeGraph.cpp
    src/reasoning/Parser.cpp
    src/reasoning/InferenceEngine.cpp
    src/learning/DenseLayer.cpp
    src/learning/ActivationLayer.cpp
    src/learning/Network.cpp
    src/learning/FeatureEncoder.cpp
    src/explain/ExplanationTrace.cpp
    src/app/Application.cpp
    src/web/main_web.cpp
)

add_executable(NeuroCoreXWeb ${WEB_SOURCES})
target_include_directories(NeuroCoreXWeb PRIVATE vendor)

find_package(Threads REQUIRED)
target_link_libraries(NeuroCoreXWeb PRIVATE Threads::Threads)

if(WIN32)
    target_link_libraries(NeuroCoreXWeb PRIVATE ws2_32 crypt32)
endif()

# Tests
enable_testing()

add_executable(test_knowledge_graph tests/test_knowledge_graph.cpp src/memory/KnowledgeGraph.cpp src/core/EventBus.cpp src/explain/ExplanationTrace.cpp)
add_test(NAME KnowledgeGraphTest COMMAND test_knowledge_graph)

add_executable(test_parser tests/test_parser.cpp src/reasoning/Parser.cpp)
add_test(NAME ParserTest COMMAND test_parser)

add_executable(test_inference_engine tests/test_inference_engine.cpp 
    src/reasoning/InferenceEngine.cpp 
    src/reasoning/Parser.cpp 
    src/memory/KnowledgeGraph.cpp 
    src/core/EventBus.cpp 
    src/learning/Network.cpp
    src/learning/DenseLayer.cpp
    src/learning/ActivationLayer.cpp
    src/learning/FeatureEncoder.cpp
    src/explain/ExplanationTrace.cpp
)
add_test(NAME InferenceEngineTest COMMAND test_inference_engine)

add_executable(test_layers tests/test_layers.cpp 
    src/learning/DenseLayer.cpp 
    src/learning/ActivationLayer.cpp 
    src/learning/Network.cpp
)
add_test(NAME LayersTest COMMAND test_layers)
