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

Tensor linear_backward_out_npu(
    Tensor& result,
    const Tensor& input,
    const Tensor& weight,
    bool transpose_x1,
    bool transpose_x2) {
  int64_t offset_x = 0;
  OpCommand cmd;
  cmd.Name("MatMulV2")
      .Input(input)
      .Input(weight)
      .Output(result)
      .Attr("transpose_x1", transpose_x1)
      .Attr("transpose_x2", transpose_x2)
      .Attr("offset_x", offset_x)
      .Run();
  return result;
}

tuple<Tensor, Tensor> linear_backward_npu(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& weight) {
  SmallVector<int64_t, SIZE> inputGradOutputSize = {
      grad.size(0), 
      weight.size(1)};
  SmallVector<int64_t, SIZE> weightGradOutputSize = {
      grad.size(1), 
      input.size(1)};
  Tensor inputGrad = OpPreparation::ApplyTensor(input, inputGradOutputSize);
  Tensor weightGrad = OpPreparation::ApplyTensor(weight, weightGradOutputSize);

  linear_backward_out_npu(inputGrad, grad, weight, false, false);
  linear_backward_out_npu(weightGrad, grad, input, true, false);
  
  return std::tie(inputGrad, weightGrad);
}

} // namespace native
} // namespace at
