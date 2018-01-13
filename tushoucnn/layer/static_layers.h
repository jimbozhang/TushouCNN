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

#ifndef TUSHOUCNN_LAYER_STATIC_H
#define TUSHOUCNN_LAYER_STATIC_H

#include <string>
#include <vector>
#include <map>
#include "layer/layer.h"

namespace tushoucnn {

typedef std::map<std::string, std::string> LayerHParams;

class ReluLayer : public Layer {
public:
  Tensor &fprop(Tensor &data) {
    return data;
  }
};

class MaxPollingLayer : public Layer {
public:
  Tensor &fprop(Tensor &data) {
    return data;
  }
};

class SoftmaxLayer : public Layer {
public:
  Tensor &fprop(Tensor &data) {
    return data;
  }
};

} // namespace tushoucnn
#endif // TUSHOUCNN_LAYER_STATIC_H_
