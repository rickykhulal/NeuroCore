#include "app/Application.h"
#include "httplib.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

// Minimal JSON string escaping for embedding Application::handle()'s
// plain-text output as a JSON string value.
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

// Extremely small JSON body parser for the one shape we need:
// {"command": "..."} -- avoids pulling in a full JSON library for a
// single field. Not a general-purpose parser; documented as such.
static std::string extractCommandField(const std::string& body) {
    size_t keyPos = body.find("\"command\"");
    if (keyPos == std::string::npos) return "";
    size_t colon = body.find(':', keyPos);
    if (colon == std::string::npos) return "";
    size_t firstQuote = body.find('"', colon);
    if (firstQuote == std::string::npos) return "";
    size_t start = firstQuote + 1;
    std::string result;
    for (size_t i = start; i < body.size(); ++i) {
        char c = body[i];
        if (c == '\\' && i + 1 < body.size()) {
            char next = body[i + 1];
            if (next == 'n') { result += '\n'; i++; continue; }
            if (next == '"' || next == '\\') { result += next; i++; continue; }
            continue;
        }
        if (c == '"') break;
        result += c;
    }
    return result;
}

int main(int argc, char** argv) {
    neurocore::app::Application app;
    httplib::Server server;

    int port = 8080;

    // Cloud platforms like Render assign a port dynamically via the
    // PORT environment variable -- the app must bind to that, not a
    // hardcoded value, or the platform's router can't reach it.
    if (const char* envPort = std::getenv("PORT")) {
        try { port = std::stoi(envPort); } catch (...) {}
    }
    if (argc >= 2) {
        try { port = std::stoi(argv[1]); } catch (...) {}
    }

    // Serve the dashboard page.
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream file("web/index.html", std::ios::binary);
        if (!file.is_open()) {
            res.status = 500;
            res.set_content("index.html not found. Expected at web/index.html relative to the working directory.", "text/plain");
            return;
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        res.set_content(ss.str(), "text/html");
    });

    // Main command endpoint: runs a REPL-equivalent command through the
    // SAME Application (i.e. the same real KnowledgeGraph/Network/Parser/
    // InferenceEngine as the console app) and returns its output as JSON.
    server.Post("/api/command", [&app](const httplib::Request& req, httplib::Response& res) {
        std::string command = extractCommandField(req.body);
        std::string output = command.empty() ? "Empty command." : app.handle(command);

        std::ostringstream j;
        j << "{\"output\":\"" << jsonEscape(output) << "\"}";
        res.set_content(j.str(), "application/json");
    });

    // Reset endpoint: clears active memory, training data, and persisted snapshots.
    server.Post("/api/reset", [&app](const httplib::Request&, httplib::Response& res) {
        bool success = app.reset();
        res.status = success ? 200 : 500;
        res.set_content(success
            ? "{\"ok\":true,\"message\":\"All memory, training data, and snapshots have been reset.\"}"
            : "{\"ok\":false,\"message\":\"Reset failed while deleting persisted data.\"}",
            "application/json");
    });

    // Lightweight stats endpoint for the sidebar (concept/relationship/
    // conflict/training counts), polled periodically by the page.
    server.Get("/api/stats", [&app](const httplib::Request&, httplib::Response& res) {
        res.set_content(app.statsJson(), "application/json");
    });

    // Basic CORS so the page can be opened/tested flexibly during development.
    server.set_default_headers({
        { "Access-Control-Allow-Origin", "*" }
    });

    std::cout << "NeuroCore X Web is running at http://localhost:" << port << std::endl;
    std::cout << "(Serving index.html from ./web/index.html -- run this executable "
              << "from the project root, or adjust the path in main_web.cpp.)" << std::endl;

    server.listen("0.0.0.0", port);
    return 0;
}
