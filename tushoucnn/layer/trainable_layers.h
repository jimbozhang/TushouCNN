// Copyright 2017-2018  Junbo Zhang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef TUSHOUCNN_LAYER_TRAINABLE_H
#define TUSHOUCNN_LAYER_TRAINABLE_H

#include <string>
#include <vector>
#include <map>
#include "layer/layer.h"

namespace tushoucnn {

typedef std::map<std::string, std::string> LayerHParams;

class FullyConnectedLayer : public Layer {
public:
  Tensor &fprop(Tensor &data) {
    return data;
  }

  void load_params() {
    int in_size = std::stoi(hparams_["in"]);
    int out_size = std::stoi(hparams_["out"]);
    int size = in_size * out_size;
    std::vector<FeatType> weights = read_params_file("weights");
    assert(weights.size() >= size);
    weights.resize(size);
    std::vector<FeatType> biases = read_params_file("biases");
    assert(biases.size() >= out_size);
    biases.resize(out_size);
  }
};

class ConvolutionalLayer : public Layer {
public:
  Tensor &fprop(Tensor &data) {
    return data;
  }

  void load_params() {
    int kh = std::stoi(hparams_["kernel_h"]);
    int kw = std::stoi(hparams_["hernel_w"]);
    int ci = std::stoi(hparams_["channel_in"]);
    int co = std::stoi(hparams_["channel_out"]);
    int size = kh * kw * ci * co;
    std::vector<FeatType> weights = read_params_file("weights");
    assert(weights.size() >= size);
    weights.resize(size);
    std::vector<FeatType> biases = read_params_file("biases");
    assert(biases.size() >= co);
    biases.resize(co);
  }
};

} // namespace tushoucnn
#endif // TUSHOUCNN_LAYER_TRAINABLE_H_
