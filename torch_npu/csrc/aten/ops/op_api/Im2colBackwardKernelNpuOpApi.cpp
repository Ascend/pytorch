// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::im2col_backward_out(const at::Tensor& grad_output, at::IntArrayRef input_size,
                                                         at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                                                         at::IntArrayRef padding, at::IntArrayRef stride,
                                                         at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnIm2colBackward,
                   NPUNativeFunctions::im2col_backward_out(grad_output, input_size, kernel_size,
                                                           dilation, padding, stride, grad_input));

  auto output_size = im2col_backward_npu_output_size(grad_output, input_size, kernel_size);
  OpPreparation::CheckOut({grad_output}, grad_input, grad_input.scalar_type(), output_size);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnIm2colBackward, grad_output, input_size, kernel_size, dilation, padding, stride, grad_input);

  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::im2col_backward(const at::Tensor& grad_output, at::IntArrayRef input_size,
                                                    at::IntArrayRef kernel_size, at::IntArrayRef dilation,
                                                    at::IntArrayRef padding, at::IntArrayRef stride) {
  DO_COMPATIBILITY(aclnnIm2colBackward, NPUNativeFunctions::im2col_backward(grad_output, input_size, kernel_size,
                                                                            dilation, padding, stride));

  auto output_size = im2col_backward_npu_output_size(grad_output, input_size, kernel_size);
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output, output_size);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnIm2colBackward, grad_output, input_size, kernel_size, dilation, padding, stride, grad_input);

  return grad_input;
}
} // namespace native
} // namespace at_npu
