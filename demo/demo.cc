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

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "tushou-cnn.h"

using namespace std;
using namespace tushoucnn;

void load_mnist_data(string filename, vector<vector<float> > &images, vector<int> &labels) {
  images.clear();
  labels.clear();
  ifstream fin(filename.c_str());
  while(! fin.eof()) {
    string line;
    getline(fin, line);
    stringstream sline(line);
    vector<float> values;
    int y;
    sline >> y;
    while (! sline.eof()) {
      float x;
      sline >> x;
      x = (x - (255 / 2.0)) / 255;
      values.push_back(x);
    }
    if (values.size() == 784) {
      images.push_back(values);
      labels.push_back(y);
    }
  }
  fin.close();
}

float eval_data(vector<vector<float> > &test_data, vector<int> &ref_labels) {
  assert(test_data.size() == ref_labels.size());
  NN nn;
  nn.load_model("./model");

  int correct_num = 0;
  for (size_t i = 0; i < test_data.size(); i++) {
    Tensor image(test_data[i]);
    Shape image_shape;
    image_shape.push_back(1);
    image_shape.push_back(28);
    image_shape.push_back(28);
    image_shape.push_back(1);
    image.reshape(image_shape);
    int predict_result = nn.predict(image);
    if (predict_result == ref_labels[i])
      correct_num++;
  }
  return (float)correct_num / ref_labels.size();
}

int main()
{
  vector<vector<float> > test_images;
  vector<int> test_labels;
  load_mnist_data("mnist_test.txt", test_images, test_labels);
  float accuracy = eval_data(test_images, test_labels);
  cout << "Accuracy: " << accuracy << endl;

  return 0;
}
