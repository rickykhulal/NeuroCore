# NeuroCore X — A Modular Cognitive Architecture in C++

## What this is

NeuroCore X is a C++17, console-based system that combines a from-scratch
neural network, a knowledge graph, an arithmetic parser, and an inference
engine into a single explainable, interactive architecture. Every
component is built without external ML/graph libraries, as an exercise
in object-oriented design (abstraction, inheritance, polymorphism,
encapsulation) applied to a real hybrid symbolic/neural system.

**Central question this project explores:**
> Can symbolic knowledge (a graph of taught facts) and neural pattern
> learning coexist in one architecture, while remaining fully
> explainable about how each answer was reached, honest about
> contradictions in what it's been taught, and able to measure its own
> ability to generalize to new, unseen inputs?

## Architecture

```
                         EventBus (publish/subscribe)
                                  |
        +---------------+--------+--------+----------------+
        |               |                 |                |
  KnowledgeGraph      Parser        Learning Engine    Explainability
     (Memory)      (Reasoning)      (Neural Network)   (ExplanationTrace)
        |               |                 |                |
        +---------------+--------+--------+----------------+
                                  |
                          InferenceEngine
                    (decides HOW to answer, in order:
                     exact fact -> multi-hop -> learned
                     pattern -> pure computation -> "I don't know")
                                  |
                                REPL
```

### Learning Engine
- `Layer` (abstract) -> `DenseLayer` (weights/biases, real gradient
  descent) + `ActivationLayer` (ReLU / Sigmoid / Linear)
- `Network` composes layers polymorphically; runs real forward
  propagation, MSE loss, and backpropagation -- not simulated.
- `FeatureEncoder` converts arithmetic expressions to/from numeric
  vectors, and intentionally only routes the operator it was actually
  trained on (`+`) to the network; anything else falls through to the
  Parser for correct real arithmetic.

### Knowledge Graph
- `Concept` nodes + typed, directional `Relationship` edges.
- Direct and multi-hop lookup (BFS over `isA` chains).
- **Contradiction Engine**: when a new fact conflicts with an existing
  one for the same (subject, relation), both are kept, flagged, and
  reported honestly instead of one silently overwriting the other.
- **Provenance tracking**: every relationship records its `source`
  (`user-taught` vs `loaded`), and multi-hop answers report the exact
  chain of concepts traversed to reach the answer.

### Parser
- Hand-written recursive-descent tokenizer/parser/evaluator for
  arithmetic expressions with correct operator precedence and
  parentheses. Rejects invalid tokens and division by zero instead of
  silently returning a wrong number.

### InferenceEngine
Answers a question by checking, in this order:
1. Exact taught fact (with contradiction check)
2. Multi-hop inferred fact (with contradiction check)
3. Learned pattern from the Neural Network (only for the trained operator)
4. Pure computation via the Parser
5. Honest `"I don't know."` if none apply

### Explainability
- Every answer can be followed by `why`, which reports which subsystem
  answered, what evidence/chain it used, and a confidence score.

### Persistence
- `save` / `load` for a default memory file.
- **Named Snapshots**: `snapshot save/load/list/delete <name>`, stored
  under `data/snapshots/`, independent of each other.
- **Snapshot Diff**: `snapshot diff <name1> <name2>` performs a real
  structural comparison of two saved knowledge states -- added/removed
  concepts, added/removed/changed relationships (direction-sensitive).

### Generalization Evaluation
- `evaluate` performs a genuine held-out test: it splits the real taught
  examples into a training subset and a held-out subset, trains a
  **separate temporary network** (so the user's real trained network is
  never touched), predicts on the held-out inputs using that temporary
  network, and reports Mean Absolute Error and PASS/FAIL per sample
  against a fixed tolerance. If too few examples exist, it refuses
  honestly rather than fabricating a result.

## Build Instructions

Requires CMake >= 3.10 and a C++17 compiler (MSVC, GCC, or Clang).

```
mkdir build
cd build
cmake ..
cmake --build .
```

Or open the project folder directly in Visual Studio (File -> Open ->
Folder) -- CMake integration will configure it automatically.

Run the resulting `NeuroCoreX` executable.

## Command Reference

```
teach <subject> <relation> <object>   - Add a fact (e.g. teach Apple isA Fruit)
teach <expr> = <result>               - Add an equation-style fact (e.g. teach 1 + 1 = 2)
ask <question/expr>                   - Query the system
why                                   - Explain the last answer
train                                 - Train the Learning Engine on all taught equations
evaluate                              - Held-out generalization test
memory                                - Inspect the Knowledge Graph
save / load                           - Persist/restore the default memory file
snapshot save <name>                  - Save current memory as a named snapshot
snapshot load <name>                  - Replace current memory with a named snapshot
snapshot list                         - List all saved snapshots
snapshot delete <name>                - Delete a named snapshot
snapshot diff <name1> <name2>         - Compare two saved snapshots
help / exit
```

## Demo Script

```
teach 1 + 1 = 4
teach 2 + 2 = 8
teach 3 + 1 = 8
teach 1 + 3 = 8
teach 4 + 4 = 16
teach 5 + 2 = 14
teach 0 + 0 = 0
teach 3 + 3 = 12
train
ask 2 + 3
why
evaluate
ask 100 * 2
why

teach Apple isA Fruit
teach Fruit hasProperty Sweet
ask Apple hasProperty
why

teach Apple isA Vegetable
ask Apple isA
why

ask 9+a

snapshot save v1
teach Dog isA Animal
snapshot save v2
snapshot diff v1 v2

save
```

This sequence demonstrates, in order: real generalization to an untaught
input, a measured held-out evaluation, correct routing between the
neural network and pure arithmetic, multi-hop inference with provenance,
contradiction detection, honest failure on invalid input, and snapshot
comparison.

## Known Limitations (Future Work)

- The Learning Engine's trained weights are not currently saved/restored
  across program restarts -- only the Knowledge Graph persists.
  Re-running `train` is required after reloading.
- `FeatureEncoder` currently only recognizes the `+` operator for neural
  pattern learning; other operators are correctly routed to the Parser
  instead, but are not themselves learnable patterns yet.
- No GUI/visualization layer; this is an intentional scope decision for
  a console-based academic project, not an oversight.
- No plugin system, multi-agent reasoning, or Truth Maintenance System.
  These were deliberately scoped out to keep the implemented system
  small, fully working, and defensible, rather than partially building
  many larger research-scale ideas.

## Team

- Shubham Shah
- Ricky Bd. Khulal
- Pujan Basnet
- Sudin Dhakal

Department of Electronics and Computer Engineering, Purwanchal Campus,
Institute of Engineering, Tribhuvan University.
