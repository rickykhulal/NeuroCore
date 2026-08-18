#include "app/Application.h"
#include "learning/DenseLayer.h"
#include "learning/ActivationLayer.h"
#include "learning/FeatureEncoder.h"
#include "memory/KnowledgeGraph.h"
#include <sstream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>

namespace neurocore::app {

namespace fs = std::filesystem;

static const std::string SNAPSHOT_DIR = "data/snapshots";

static bool isValidSnapshotName(const std::string& name) {
    if (name.empty()) return false;
    for (char c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            return false;
        }
    }
    return true;
}

static std::string snapshotPath(const std::string& name) {
    return SNAPSHOT_DIR + "/" + name + ".ngx";
}

static std::unique_ptr<learning::Network> buildEvaluationNetwork() {
    auto net = std::make_unique<learning::Network>();
    net->addLayer(std::make_unique<learning::DenseLayer>(2, 8));
    net->addLayer(std::make_unique<learning::ActivationLayer>("relu"));
    net->addLayer(std::make_unique<learning::DenseLayer>(8, 1));
    net->addLayer(std::make_unique<learning::ActivationLayer>("linear"));
    return net;
}

// Minimal JSON string escaping (quotes, backslashes, newlines) so
// multi-line console-style output can be embedded safely in a JSON
// string value for the web layer.
static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

Application::Application() {
    eventBus = std::make_shared<core::EventBus>();
    graph = std::make_shared<memory::KnowledgeGraph>(eventBus);
    network = std::make_shared<learning::Network>();

    network->addLayer(std::make_unique<learning::DenseLayer>(2, 8));
    network->addLayer(std::make_unique<learning::ActivationLayer>("relu"));
    network->addLayer(std::make_unique<learning::DenseLayer>(8, 1));
    network->addLayer(std::make_unique<learning::ActivationLayer>("linear"));

    engine = std::make_unique<reasoning::InferenceEngine>(graph, network);

    // Note: unlike the console Repl, this does not subscribe a
    // std::cout logger to "FactAdded" -- printing straight to the
    // server process's console on every web request isn't useful. The
    // same event still fires on the EventBus; a future subscriber
    // (e.g. a live-updating web log) could hook into it without any
    // change to KnowledgeGraph itself.
}

bool Application::reset() {
    std::error_code ec;

    // Remove persisted memory and named snapshots as part of a full reset.
    fs::remove("data/memory_save.txt", ec);
    if (ec) return false;

    ec.clear();
    fs::remove_all(SNAPSHOT_DIR, ec);
    if (ec) return false;

    // Recreate the active graph and network so all in-memory facts,
    // training examples, learned weights, and trained state are cleared.
    graph = std::make_shared<memory::KnowledgeGraph>(eventBus);
    network = std::make_shared<learning::Network>();
    network->addLayer(std::make_unique<learning::DenseLayer>(2, 8));
    network->addLayer(std::make_unique<learning::ActivationLayer>("relu"));
    network->addLayer(std::make_unique<learning::DenseLayer>(8, 1));
    network->addLayer(std::make_unique<learning::ActivationLayer>("linear"));
    engine = std::make_unique<reasoning::InferenceEngine>(graph, network);
    lastTrace.clear();

    return true;
}

std::string Application::handle(const std::string& line) {
    return dispatch(line);
}

std::string Application::dispatch(const std::string& line) {
    std::ostringstream out;
    std::stringstream ss(line);
    std::string cmd;
    ss >> cmd;

    if (cmd == "reset") {
        out << (reset() ? "All memory, training data, and snapshots have been reset.\n"
                         : "Reset failed while deleting persisted data.\n");

    } else if (cmd == "help") {
        out << "Available Commands:\n";
        out << "  teach <sub/num> <rel/op> <obj/num>  - Add a fact or training example\n";
        out << "  teach <expr> = <result>             - Add an equation-style fact (e.g. teach 1 + 1 = 2)\n";
        out << "  ask <question/expr>                - Query the system\n";
        out << "  why                                - Explain the last answer\n";
        out << "  train                              - Train the Learning Engine\n";
        out << "  evaluate                           - Held-out generalization test\n";
        out << "  memory                             - Inspect Knowledge Graph\n";
        out << "  save / load                        - Persist/Restore the default memory file\n";
        out << "  snapshot save <name>               - Save current memory as a named snapshot\n";
        out << "  snapshot load <name>               - Replace current memory with a named snapshot\n";
        out << "  snapshot list                      - List all saved snapshots\n";
        out << "  snapshot delete <name>             - Delete a named snapshot\n";
        out << "  snapshot diff <name1> <name2>      - Compare two saved snapshots\n";
        out << "  reset                              - Clear all memory and saved data\n";

    } else if (cmd == "teach") {
        std::string rest;
        std::getline(ss, rest);
        if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);

        if (rest.empty()) {
            out << "Usage: teach <subject> <relation> <object>  OR  teach <expr> = <result>\n";
            return out.str();
        }

        size_t eqPos = rest.find('=');
        if (eqPos != std::string::npos) {
            std::string left = rest.substr(0, eqPos);
            std::string right = rest.substr(eqPos + 1);

            while (!left.empty() && left.back() == ' ') left.pop_back();
            while (!right.empty() && right.front() == ' ') right.erase(0, 1);
            while (!right.empty() && right.back() == ' ') right.pop_back();

            if (left.empty() || right.empty()) {
                out << "Usage: teach <expr> = <result>  (e.g. teach 1 + 1 = 2)\n";
                return out.str();
            }

            graph->addRelationship(left, "equals", right);
            out << "Learned (memory): " << left << " = " << right << "\n";

            auto features = learning::FeatureEncoder::encodeExpression(left);
            auto targetVec = learning::FeatureEncoder::encodeTarget(right);
            if (features && targetVec) {
                network->addExample(*features, *targetVec);
                out << "Learned (pattern): added training example for the Learning Engine ("
                    << network->exampleCount() << " total).\n";
            }
        } else {
            std::stringstream restStream(rest);
            std::string sub, rel, obj;
            if (restStream >> sub >> rel >> obj) {
                graph->addRelationship(sub, rel, obj);
                out << "Learned: " << sub << " " << rel << " " << obj << "\n";
            } else {
                out << "Usage: teach <subject> <relation> <object>  OR  teach <expr> = <result>\n";
            }
        }

    } else if (cmd == "ask") {
        std::string question;
        std::getline(ss, question);
        if (question.empty()) {
            out << "Usage: ask <question>\n";
            return out.str();
        }
        if (question[0] == ' ') question.erase(0, 1);

        lastTrace.clear();
        std::string result = engine->ask(question, lastTrace);
        out << "Answer: " << result << "\n";

    } else if (cmd == "why") {
        out << lastTrace.toString() << "\n";

    } else if (cmd == "train") {
        if (network->exampleCount() == 0) {
            out << "Nothing to train on yet -- teach some equation-style facts first "
                << "(e.g. teach 1 + 1 = 2).\n";
            return out.str();
        }
        out << "Training Learning Engine on " << network->exampleCount() << " example(s)...\n";
        network->train(5000, 0.005);
        out << "Training complete.\n";

    } else if (cmd == "evaluate") {
        const auto& allInputs = network->getTrainingInputs();
        const auto& allTargets = network->getTrainingTargets();
        size_t total = allInputs.size();

        const size_t MIN_TOTAL_FOR_EVAL = 4;
        if (total < MIN_TOTAL_FOR_EVAL) {
            out << "Evaluation unavailable.\n";
            out << "Reason: at least " << MIN_TOTAL_FOR_EVAL
                << " independent taught examples are required for a held-out evaluation.\n";
            out << "Currently available: " << total << "\n";
            return out.str();
        }

        size_t holdOutCount = std::max<size_t>(1, total / 4);
        size_t trainCount = total - holdOutCount;

        std::vector<std::vector<double>> trainInputs(allInputs.begin(), allInputs.begin() + trainCount);
        std::vector<std::vector<double>> trainTargets(allTargets.begin(), allTargets.begin() + trainCount);
        std::vector<std::vector<double>> heldInputs(allInputs.begin() + trainCount, allInputs.end());
        std::vector<std::vector<double>> heldTargets(allTargets.begin() + trainCount, allTargets.end());

        auto evalNet = buildEvaluationNetwork();
        for (size_t i = 0; i < trainInputs.size(); ++i) {
            evalNet->addExample(trainInputs[i], trainTargets[i]);
        }

        out << "Running held-out generalization evaluation...\n";
        out << "(Training a temporary network on " << trainCount
            << " examples; your existing trained network is not affected.)\n";
        evalNet->train(5000, 0.005);

        const double TOLERANCE = 1.0;
        double sumAbsError = 0.0;
        int passCount = 0;
        std::vector<std::tuple<std::string, double, double, double, bool>> rows;

        for (size_t i = 0; i < heldInputs.size(); ++i) {
            std::vector<double> output = evalNet->predict(heldInputs[i]);
            double predicted = output.empty() ? 0.0 : output[0];
            double expected = heldTargets[i].empty() ? 0.0 : heldTargets[i][0];
            double absError = std::fabs(predicted - expected);
            bool passed = absError <= TOLERANCE;

            sumAbsError += absError;
            if (passed) passCount++;

            std::string label = (heldInputs[i].size() >= 2)
                ? std::to_string(static_cast<long long>(heldInputs[i][0])) + " + " +
                  std::to_string(static_cast<long long>(heldInputs[i][1]))
                : "?";
            rows.emplace_back(label, expected, predicted, absError, passed);
        }

        double mae = heldInputs.empty() ? 0.0 : sumAbsError / static_cast<double>(heldInputs.size());
        bool overallPass = (passCount == static_cast<int>(heldInputs.size()));

        out << "==================================================\n";
        out << "             NEUROCORE GENERALIZATION\n";
        out << "==================================================\n";
        out << "Dataset\n";
        out << "  Total examples:       " << total << "\n";
        out << "  Training examples:    " << trainCount << "\n";
        out << "  Held-out examples:    " << holdOutCount << "\n\n";
        out << "Held-Out Evaluation\n";
        out << "--------------------------------------------------\n";
        out << std::left << std::setw(12) << "Input" << std::setw(15) << "Expected"
            << std::setw(15) << "Predicted" << std::setw(12) << "Error" << "Result\n";
        for (const auto& row : rows) {
            out << std::left << std::setw(12) << std::get<0>(row)
                << std::setw(15) << std::fixed << std::setprecision(2) << std::get<1>(row)
                << std::setw(15) << std::fixed << std::setprecision(2) << std::get<2>(row)
                << std::setw(12) << std::fixed << std::setprecision(2) << std::get<3>(row)
                << (std::get<4>(row) ? "PASS" : "FAIL") << "\n";
        }
        out << "--------------------------------------------------\n";
        out << "Evaluation Summary\n";
        out << "  Mean Absolute Error:  " << std::fixed << std::setprecision(3) << mae << "\n";
        out << "  Passed:               " << passCount << "\n";
        out << "  Failed:               " << (heldInputs.size() - passCount) << "\n";
        out << "  Generalization:       " << (overallPass ? "PASS" : "FAIL") << "\n";
        out << "==================================================\n";

    } else if (cmd == "memory") {
        out << "Knowledge Graph Contents:\n";
        for (const auto& pair : graph->getAdjacencyList()) {
            for (const auto& rel : pair.second) {
                out << "  " << rel.fromConcept << " --" << rel.type << "--> " << rel.toConcept
                    << " (conf: " << rel.confidence << ")\n";
            }
        }

    } else if (cmd == "save") {
        if (graph->save("data/memory_save.txt")) {
            out << "Memory saved to data/memory_save.txt\n";
        } else {
            out << "FAILED to save memory.\n";
        }

    } else if (cmd == "load") {
        if (graph->load("data/memory_save.txt")) {
            out << "Memory loaded from data/memory_save.txt\n";
        } else {
            out << "FAILED to load memory -- did you run 'save' first?\n";
        }

    } else if (cmd == "snapshot") {
        std::string subCmd;
        ss >> subCmd;

        if (subCmd == "save") {
            std::string name;
            ss >> name;
            if (!isValidSnapshotName(name)) {
                out << "Usage: snapshot save <name>\n";
            } else if (graph->save(snapshotPath(name))) {
                out << "Snapshot '" << name << "' saved.\n";
            } else {
                out << "FAILED to save snapshot '" << name << "'.\n";
            }

        } else if (subCmd == "load") {
            std::string name;
            ss >> name;
            std::string path = snapshotPath(name);
            if (!isValidSnapshotName(name) || !fs::exists(path)) {
                out << "No snapshot named '" << name << "' found.\n";
            } else if (graph->load(path)) {
                out << "Snapshot '" << name << "' loaded.\n";
            } else {
                out << "FAILED to load snapshot '" << name << "'.\n";
            }

        } else if (subCmd == "list") {
            if (!fs::exists(SNAPSHOT_DIR) || fs::is_empty(SNAPSHOT_DIR)) {
                out << "No snapshots saved yet.\n";
            } else {
                out << "Available snapshots:\n";
                std::vector<fs::directory_entry> entries;
                for (const auto& entry : fs::directory_iterator(SNAPSHOT_DIR)) {
                    if (entry.path().extension() == ".ngx") entries.push_back(entry);
                }
                std::sort(entries.begin(), entries.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
                    return fs::last_write_time(a) > fs::last_write_time(b);
                });
                for (const auto& entry : entries) {
                    out << "  - " << entry.path().stem().string() << "\n";
                }
            }

        } else if (subCmd == "delete") {
            std::string name;
            ss >> name;
            std::string path = snapshotPath(name);
            if (!isValidSnapshotName(name) || !fs::exists(path)) {
                out << "No snapshot named '" << name << "' found.\n";
            } else {
                std::error_code ec;
                fs::remove(path, ec);
                out << (!ec ? "Snapshot '" + name + "' deleted.\n" : "FAILED to delete snapshot.\n");
            }

        } else if (subCmd == "diff") {
            std::string nameA, nameB;
            ss >> nameA >> nameB;
            std::string pathA = snapshotPath(nameA), pathB = snapshotPath(nameB);

            if (!isValidSnapshotName(nameA) || !isValidSnapshotName(nameB)) {
                out << "Usage: snapshot diff <name1> <name2>\n";
                return out.str();
            }
            if (!fs::exists(pathA)) { out << "Snapshot not found: " << nameA << "\n"; return out.str(); }
            if (!fs::exists(pathB)) { out << "Snapshot not found: " << nameB << "\n"; return out.str(); }

            memory::KnowledgeGraph graphA(nullptr);
            memory::KnowledgeGraph graphB(nullptr);
            graphA.load(pathA);
            graphB.load(pathB);

            using Triple = std::tuple<std::string, std::string, std::string>;
            std::set<Triple> triplesA, triplesB;
            for (const auto& lp : graphA.getAdjacencyList())
                for (const auto& rel : lp.second) triplesA.insert({rel.fromConcept, rel.type, rel.toConcept});
            for (const auto& lp : graphB.getAdjacencyList())
                for (const auto& rel : lp.second) triplesB.insert({rel.fromConcept, rel.type, rel.toConcept});

            out << "Snapshot Diff: " << nameA << " -> " << nameB << "\n";
            out << "Added:\n";
            for (const auto& t : triplesB) if (!triplesA.count(t))
                out << "  + " << std::get<0>(t) << " --" << std::get<1>(t) << "--> " << std::get<2>(t) << "\n";
            out << "Removed:\n";
            for (const auto& t : triplesA) if (!triplesB.count(t))
                out << "  - " << std::get<0>(t) << " --" << std::get<1>(t) << "--> " << std::get<2>(t) << "\n";

        } else {
            out << "Usage: snapshot save|load|list|delete|diff <name(s)>\n";
        }

    } else {
        out << "Unknown command: " << cmd << ". Type 'help' for assistance.\n";
    }

    return out.str();
}

std::string Application::statsJson() const {
    size_t conceptCount = graph->getConcepts().size();
    size_t relCount = 0;
    size_t conflictCount = 0;
    for (const auto& pair : graph->getAdjacencyList()) {
        for (const auto& rel : pair.second) {
            relCount++;
            if (rel.conflicted) conflictCount++;
        }
    }

    std::ostringstream j;
    j << "{"
      << "\"concepts\":" << conceptCount << ","
      << "\"relationships\":" << relCount << ","
      << "\"conflicts\":" << conflictCount << ","
      << "\"trainingExamples\":" << network->exampleCount() << ","
      << "\"trained\":" << (network->isTrained() ? "true" : "false")
      << "}";
    return j.str();
}

} // namespace neurocore::app
