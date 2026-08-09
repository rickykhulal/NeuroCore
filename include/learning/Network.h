#pragma once
#include "Layer.h"
#include <vector>
#include <memory>

namespace neurocore::learning {

class Network {
public:
    void addLayer(std::unique_ptr<Layer> layer);
    std::vector<double> predict(const std::vector<double>& input);

    void addExample(const std::vector<double>& input, const std::vector<double>& target);
    void train(int epochs = 2000, double learningRate = 0.05);

    bool isTrained() const { return trained; }
    size_t exampleCount() const { return trainingInputs.size(); }

    // NEW (needed for 'evaluate'): read-only access to the real
    // accumulated training dataset, so a held-out split can be built
    // from actual taught examples instead of anything fabricated.
    const std::vector<std::vector<double>>& getTrainingInputs() const { return trainingInputs; }
    const std::vector<std::vector<double>>& getTrainingTargets() const { return trainingTargets; }

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::vector<double>> trainingInputs;
    std::vector<std::vector<double>> trainingTargets;
    bool trained = false;
};

} // namespace neurocore::learning
