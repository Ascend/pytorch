// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor, Tensor> dropout_with_add_softmax_backward_npu(
    const Tensor& grad_out,
    const Tensor& mask,
    const Tensor& softmax_out,
    Scalar alpha,
    double p,
    int64_t dim){
  Tensor result = OpPreparation::ApplyTensor(softmax_out);
  Tensor grad_res = OpPreparation::ApplyTensor(softmax_out);
  SmallVector<int64_t, N> dimList = {dim};
  double retain = 1. - p;
  Scalar prob = Scalar(retain);

  OpCommand cmd;
  cmd.Name("DropoutWithMulsAndSoftmaxGrad")
     .Input(grad_out)
     .Input(mask)
     .Input(softmax_out)
     .Output(result)
     .Attr("alpha", alpha)
     .Attr("input_keep_prob", prob)
     .Attr("axes", dimList)
     .Run();
  grad_res = grad_out;
  return std::tie(result, grad_res);
}

}
}
