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


Tensor& max_unpool3d_backward_out_npu_nocheck(
    Tensor& grad_input, 
    const Tensor& grad_output, 
    const Tensor& indices) {
  int64_t N = 1;
  int64_t C = indices.size(0);
  if (grad_output.dim() == 5) {
    N = indices.size(0);
    C = indices.size(1);
  }
  Tensor reshape_grad_output = grad_output.reshape({N, C, -1});
  Tensor reshape_indices = indices.reshape({N, C, -1});
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

Tensor& max_unpool3d_backward_out_npu(
    Tensor& grad_input, 
    const Tensor& grad_output, 
    const Tensor& self, 
    const Tensor& indices, 
    IntArrayRef output_size, 
    IntArrayRef stride, 
    IntArrayRef padding) {
  OpPreparation::CheckOut(
      {grad_output, self, indices},
      grad_input,
      self);

  max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);

  return grad_input;
}

Tensor max_unpool3d_backward_npu(
    const Tensor& grad_output, 
    const Tensor& self, 
    const Tensor& indices, 
    IntArrayRef output_size, 
    IntArrayRef stride, 
    IntArrayRef padding) {
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
  
  Tensor grad_input = OpPreparation::ApplyTensor(self);

  max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);

  return grad_input;
}

} // namespace native
} // namespace at
