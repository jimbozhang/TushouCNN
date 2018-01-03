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

#ifndef TUSHOUCNN_TUSHOU_CNN_H
#define TUSHOUCNN_TUSHOU_CNN_H

#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cassert>

namespace tushoucnn {

typedef float FeatType;
typedef int LabelType;

struct Layer {
  std::string type;
  std::map<std::string, std::string> hparams;
};

class NN {
public:
  NN() {}

  void load_model(std::string model_dir, bool binary = false) {
    std::string model_file = model_dir + "/model";
    if (binary) {
      std::cerr << "Unsupported format." << std::endl;
      return;
    }
    else {
      std::ifstream fin;
      fin.open(model_file.c_str());
      Layer *cur_layer = NULL;
      while (! fin.eof()) {
        std::string tok_key, tok_val;
        fin >> tok_key;
        fin >> tok_val;
        if (tok_key == "layer") {
          // new layer
          if (cur_layer != NULL) {
            layers.push_back(*cur_layer);
            delete cur_layer;
          }
          cur_layer = new Layer;
          cur_layer->type = tok_val;
        }
        else {
          // this layer
          assert(cur_layer != NULL);
          cur_layer->hparams[tok_key] = tok_val;
        }
     }
     if (cur_layer != NULL) {
       layers.push_back(*cur_layer);
       delete cur_layer;
     }
     fin.close();
    }
  }

  LabelType predict(std::vector<FeatType> feats) {
    return 7;  // lucky number
  }

private:
  std::vector<Layer> layers;

};

} // namespace tushoucnn
#endif // TUSHOUCNN_TUSHOU_CNN_H
