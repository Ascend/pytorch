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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {



at::Tensor& max_unpool3d_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& indices) {
  int64_t N = 1;
  int64_t C = indices.size(0);
  if (grad_output.dim() == 5) {
    N = indices.size(0);
    C = indices.size(1);
  }
  at::Tensor reshape_grad_output = grad_output.reshape({N, C, -1});
  at::Tensor reshape_indices = indices.reshape({N, C, -1});
  grad_input = grad_input.reshape({N, C, -1});

  int64_t dim = 2;
  OpCommand cmd;
  cmd.Name("GatherElements")
     .Input(reshape_grad_output)
     .Input(reshape_indices)
     .Output(grad_input)
     .Attr("dim", dim)
     .Run();
  grad_input = grad_input.reshape(indices.sizes());
  return grad_input;
}

at::Tensor& max_unpool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  OpPreparation::CheckOut(
      {grad_output, self, indices},
      grad_input,
      self);
  if (!NpuUtils::check_match(&grad_input)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(grad_input);

    max_unpool3d_backward_out_npu_nocheck(contiguous_result, grad_output, indices);
    NpuUtils::format_fresh_view(grad_input, contiguous_result);
  } else {
    max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);
  }
  return grad_input;
}

at::Tensor max_unpool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly 3 elements (depth, height, width) in output_size");
  TORCH_CHECK(
      (self.ndimension() == 4 || self.ndimension() == 5),
      "Input to max_unpooling2d should be a 4d or 5d Tensor");
  TORCH_CHECK(
      self.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");
  TORCH_CHECK(self.numel() > 0, "Input must be non-empty");

  at::Tensor grad_input = OpPreparation::ApplyTensor(self);

  max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);

  return grad_input;
}


} // namespace native
} // namespace at_npu
