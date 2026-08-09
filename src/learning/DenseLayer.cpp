#include "learning/DenseLayer.h"
#include <random>

namespace neurocore::learning {

DenseLayer::DenseLayer(int inputSize, int outputSize) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    weights.resize(outputSize, std::vector<double>(inputSize));
    for (auto& row : weights)
        for (auto& w : row)
            w = dist(gen);

    // FIX: biases were previously randomly initialized from the same
    // (-0.5, 0.5) range as weights. With ReLU hidden units, a
    // moderately negative random bias can push a neuron's pre-activation
    // permanently negative from epoch 1 -- ReLU's gradient is exactly 0
    // there, so that neuron never updates again ("dead ReLU"). With a
    // small dataset and several hidden neurons, enough of them can die
    // immediately to explain a loss curve that drops once, then goes
    // completely flat (not oscillating) for the rest of training -- which
    // is exactly the plateau observed (68 -> 17.2 -> flat).
    //
    // Starting biases at a small positive constant keeps ReLU units
    // active at initialization, without touching weight initialization,
    // network architecture, or the training algorithm itself.
    biases.resize(outputSize, 0.01);
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
    lastInput = input;
    std::vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        double sum = biases[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }
    return output;
}

std::vector<double> DenseLayer::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> inputGradient(lastInput.size(), 0.0);

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            double weightGradient = outputGradient[i] * lastInput[j];
            inputGradient[j] += weights[i][j] * outputGradient[i];
            weights[i][j] -= learningRate * weightGradient;
        }
        biases[i] -= learningRate * outputGradient[i];
    }

    return inputGradient;
}

} // namespace neurocore::learning
