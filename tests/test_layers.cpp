#include "learning/DenseLayer.h"
#include "learning/ActivationLayer.h"
#include "learning/Network.h"
#include <cassert>
#include <iostream>

using namespace neurocore::learning;

int main() {
    DenseLayer dense(2, 3);
    std::vector<double> input = {1.0, 0.5};
    auto output = dense.forward(input);
    assert(output.size() == 3);

    ActivationLayer relu("relu");
    std::vector<double> input2 = {-1.0, 2.0, 0.0};
    auto output2 = relu.forward(input2);
    assert(output2[0] == 0.0);
    assert(output2[1] == 2.0);
    assert(output2[2] == 0.0);

    Network net;
    net.addLayer(std::make_unique<DenseLayer>(2, 2));
    net.addLayer(std::make_unique<ActivationLayer>("sigmoid"));
    auto res = net.predict({0.1, 0.2});
    assert(res.size() == 2);
    assert(res[0] >= 0.0 && res[0] <= 1.0);

    std::cout << "Layers tests passed!" << std::endl;
    return 0;
}
