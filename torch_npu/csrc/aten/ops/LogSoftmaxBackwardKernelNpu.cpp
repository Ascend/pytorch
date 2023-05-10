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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& log_softmax_backward_data_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  c10::SmallVector<int64_t, N> dimList = {dim};
  OpCommand cmd;
  cmd.Name("LogSoftmaxGrad")
      .Input(grad_output)
      .Input(output)
      .Output(result)
      .Attr("axis", dimList)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::_log_softmax_backward_data_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    c10::ScalarType input_dtype,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {grad_output, output},
      result,
      grad_output);

  log_softmax_backward_data_out_npu_nocheck(result, grad_output, output, dim, input_dtype);

  return result;
}

at::Tensor NPUNativeFunctions::_log_softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  // calculate the output size
  auto outputSize = input_same_output_size(grad_output);

  // output'format must be same with grad_output
  at::Tensor temp_output = output;
  if (CalcuOpUtil::GetTensorNpuFormat(temp_output) == ACL_FORMAT_NC1HWC0) {
    NPUNativeFunctions::npu_format_cast_(temp_output, CalcuOpUtil::GetTensorNpuFormat(grad_output));
  }

  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(temp_output, outputSize);

  // calculate the output result of the NPU
  log_softmax_backward_data_out_npu_nocheck(grad_input, grad_output, temp_output, dim, input_dtype);

  return grad_input;
}

} // namespace native
} // namespace at_npu
