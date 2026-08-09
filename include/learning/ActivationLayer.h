#pragma once
#include "Layer.h"
#include <string>
#include <vector>

namespace neurocore::learning {

class ActivationLayer : public Layer {
public:
    ActivationLayer(const std::string& type);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;

private:
    std::string type;
    std::vector<double> lastInput;   // needed for relu/linear derivative
    std::vector<double> lastOutput;  // needed for sigmoid derivative

    double sigmoid(double x);
    double relu(double x);
};

} // namespace neurocore::learning
