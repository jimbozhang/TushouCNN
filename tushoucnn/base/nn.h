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

#ifndef TUSHOUCNN_BASE_NN_H
#define TUSHOUCNN_BASE_NN_H

#include "base/types.h"
#include "base/tensor.h"
#include "layer/trainable_layers.h"
#include "layer/static_layers.h"

namespace tushoucnn {

class NN {
public:
  NN() {}
  ~NN() {
    for (auto l = layers_.begin(); l < layers_.end(); ++l) {
      delete *l;
    }
  }

  void load_model(std::string model_dir, bool binary = false) {
    std::string model_file = model_dir + "/model";
    if (binary) {
      std::cerr << "Unsupported format." << std::endl;
      return;
    }
    else {
      std::ifstream fin;
      fin.open(model_file.c_str());
      assert(fin.is_open());
      LayerHParams hparam;
      std::string layer_type;
      while (! fin.eof()) {
        std::string tok_key;
        fin >> tok_key;
        if (tok_key == "{") {
          hparam.clear();
        }
        else if (tok_key == "}") {
          assert(hparam.size() > 0);
          Layer *layer = new_layer(hparam);
          layers_.push_back(layer);
        }
        else {
          std::string tok_val;
          fin >> tok_val;
          if (tok_key == std::string("weights") || tok_key == std::string("biases")) {
            tok_val = model_dir + "/" + tok_val;
          }
          hparam[tok_key] = tok_val;
        }
      }
     fin.close();
    }
  }

  LabelType predict(Tensor &feats) {
    Tensor out = feats;
    for (auto l = layers_.begin(); l < layers_.end(); ++l) {
      out = (*l)->fprop(out);
    }
    return 7;  // lucky number
  }

private:
  std::vector<Layer *> layers_;

  Layer *new_layer(LayerHParams &param) {
    Layer *layer = NULL;
    if (param["layer_type"] == "fc") {
      layer = new FullyConnectedLayer();
    }
    else if (param["layer_type"] == "conv") {
      layer = new ConvolutionalLayer();
    }
    else if (param["layer_type"] == "relu") {
      layer = new ReluLayer();
    }
    else if (param["layer_type"] == "maxpool") {
      layer = new MaxPollingLayer();
    }
    else if (param["layer_type"] == "softmax") {
      layer = new SoftmaxLayer();
    }
    else if (param["layer_type"] == "conv") {
      layer = new ConvolutionalLayer();
    }
    else {
      std::cerr << "Unsupported layer type." << std::endl;
      assert(false); 
    }
    layer->load_hparams(param);
    layer->load_params();

    return layer;
  }
};

} // namespace tushoucnn
#endif // TUSHOUCNN_BASE_NN_H_
