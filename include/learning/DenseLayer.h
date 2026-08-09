#pragma once
#include "Layer.h"
#include <vector>

namespace neurocore::learning {

class DenseLayer : public Layer {
public:
    DenseLayer(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;

    const std::vector<std::vector<double>>& getWeights() const { return weights; }
    const std::vector<double>& getBiases() const { return biases; }

private:
    std::vector<std::vector<double>> weights; // [outputSize][inputSize]
    std::vector<double> biases;               // [outputSize]
    std::vector<double> lastInput;             // cached for backward()
};

} // namespace neurocore::learning
