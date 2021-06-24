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

Tensor softmax_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    const Tensor& self) {
  SmallVector<int64_t, N> dimList = {dim};
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("SoftmaxGrad")
      .Input(output)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("axes", dimList)
      .Run();

  return grad_input;
}

Tensor _softmax_backward_npu(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    const Tensor& self) {
  // calculate the output size
  auto outputSize = input_same_output_size(grad_output);

  // output'format must be same with grad_output
  if (CalcuOpUtil::get_tensor_npu_format(output) != CalcuOpUtil::get_tensor_npu_format(grad_output)) {
    output.npu_format_cast_(CalcuOpUtil::get_tensor_npu_format(grad_output));
  }

  // construct the output tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output, outputSize);

  // calculate the output result of the NPU
  softmax_backward_out_npu(grad_input, grad_output, output, dim, self);

  return grad_input;
}

} // namespace native
} // namespace at