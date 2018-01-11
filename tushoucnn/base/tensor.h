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

#ifndef TUSHOUCNN_BASE_TENSOR_H
#define TUSHOUCNN_BASE_TENSOR_H

namespace tushoucnn {

typedef std::vector<int> Shape;

class Tensor {
public:
  Tensor() {}
  Tensor(std::vector<FeatType> &v) {
    data_ = v;
    shape_.push_back(v.size());
  }

  void Reshape(Shape &shape) {
    int size = 1;
    for (auto it = shape.begin(); it < shape.end(); ++it) {
      size *= *it;
    }
    assert(size == data_.size());
    shape_ = shape;
  }

  Shape & GetShape() {
    return shape_;
  }

  size_t GetSize() {
    return data_.size();
  }

private:
  std::vector<FeatType> data_;
  Shape shape_;
};

} // namespace tushoucnn
#endif // TUSHOUCNN_BASE_TENSOR_H_
