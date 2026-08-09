#include "learning/Network.h"
#include <iostream>

namespace neurocore::learning {

void Network::addLayer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

std::vector<double> Network::predict(const std::vector<double>& input) {
    std::vector<double> current = input;
    for (const auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

void Network::addExample(const std::vector<double>& input, const std::vector<double>& target) {
    trainingInputs.push_back(input);
    trainingTargets.push_back(target);
    trained = false;
}

void Network::train(int epochs, double learningRate) {
    if (trainingInputs.empty() || layers.empty()) {
        trained = false;
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t sample = 0; sample < trainingInputs.size(); ++sample) {
            const auto& x = trainingInputs[sample];
            const auto& y = trainingTargets[sample];

            std::vector<double> output = x;
            for (const auto& layer : layers) {
                output = layer->forward(output);
            }

            std::vector<double> outputGradient(output.size());
            for (size_t i = 0; i < output.size(); ++i) {
                double diff = output[i] - y[i];
                totalLoss += diff * diff;
                outputGradient[i] = 2.0 * diff / static_cast<double>(output.size());
            }

            std::vector<double> grad = outputGradient;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backward(grad, learningRate);
            }
        }

        if (epoch % 500 == 0 || epoch == epochs - 1) {
            double avgLoss = totalLoss / static_cast<double>(trainingInputs.size());
            std::cout << "  [Learning Engine] Epoch " << epoch << ", Loss: " << avgLoss << std::endl;
        }
    }

    trained = true;
}

} // namespace neurocore::learning
