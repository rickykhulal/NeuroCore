#include "learning/ActivationLayer.h"
#include <cmath>
#include <algorithm>

namespace neurocore::learning {

ActivationLayer::ActivationLayer(const std::string& type) : type(type) {}

double ActivationLayer::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double ActivationLayer::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

std::vector<double> ActivationLayer::forward(const std::vector<double>& input) {
    lastInput = input;
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        if (type == "sigmoid") {
            output[i] = sigmoid(input[i]);
        } else if (type == "relu") {
            output[i] = relu(input[i]);
        } else { // "linear" (identity) -- used on the FINAL layer for
                 // regression-style outputs (e.g. predicting the number 10),
                 // since squashing through sigmoid (0-1 range) would make it
                 // impossible to ever output values like 10 or 4.
            output[i] = input[i];
        }
    }

    lastOutput = output;
    return output;
}

std::vector<double> ActivationLayer::backward(const std::vector<double>& outputGradient, double /*learningRate*/) {
    std::vector<double> inputGradient(outputGradient.size());

    for (size_t i = 0; i < outputGradient.size(); ++i) {
        double derivative;
        if (type == "sigmoid") {
            derivative = lastOutput[i] * (1.0 - lastOutput[i]);
        } else if (type == "relu") {
            derivative = lastInput[i] > 0.0 ? 1.0 : 0.0;
        } else { // linear: derivative is always 1
            derivative = 1.0;
        }
        inputGradient[i] = outputGradient[i] * derivative;
    }

    return inputGradient;
}

} // namespace neurocore::learning
