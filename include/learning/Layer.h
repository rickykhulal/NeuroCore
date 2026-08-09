#pragma once
#include <vector>

namespace neurocore::learning {

// Abstract interface every layer type must implement.
// forward()  : produces this layer's output from its input.
// backward() : given the gradient of the loss w.r.t. this layer's OUTPUT,
//              updates this layer's own trainable parameters (if any) and
//              returns the gradient of the loss w.r.t. this layer's INPUT,
//              so the previous layer can continue the chain (backprop).
class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) = 0;
};

} // namespace neurocore::learning
