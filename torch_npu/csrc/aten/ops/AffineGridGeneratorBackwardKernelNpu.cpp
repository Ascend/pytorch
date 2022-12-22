// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
namespace{
at::Tensor _linspace_from_neg_one(const at::Tensor& grid, int64_t num_steps, bool align_corners) {
  if (num_steps <= 1) {
    return at::tensor(0, grid.options());
  }
  auto range = at::linspace(-1, 1, num_steps, grid.options());
  if (!align_corners && num_steps != 0) {
    range = range * (num_steps - 1) / num_steps;
  }
  return range;
}

at::Tensor& affine_grid_generator_backward_nocheck(
    at::Tensor& result, 
    const at::Tensor& grad,   
    at::IntArrayRef size,
    bool align_corners) {
  at::Tensor assist = OpPreparation::ApplyTensor(grad, {size[0], size[2], size[3], 3});
  assist.select(-1, 0).copy_(_linspace_from_neg_one(grad, size[3], align_corners));
  assist.select(-1, 1).copy_(_linspace_from_neg_one(grad, size[2], align_corners).unsqueeze_(-1));
  assist.select(-1, 2).fill_(1);
  AT_ASSERT(grad.sizes() == at::IntArrayRef({size[0], size[2], size[3], 2})); 

  auto reassist = assist.view({size[0], size[2]*size[3], 3}).transpose(1, 2);
  auto grid = grad.view({size[0], size[2]*size[3], 2});

  OpCommand cmd;
  cmd.Name("BatchMatMul")
      .Input(reassist)
      .Input(grid)
      .Output(result)
      .Attr("bias", (int64_t)0)
      .Attr("adj_x1", (bool)false)
      .Attr("adj_x2", (bool)false)
      .Attr("_allow_hf32", true, at_npu::native::env::allowHF32Matmul())
      .Run();

  return result;
}
} // namespace

at::Tensor NPUNativeFunctions::affine_grid_generator_backward(
    const at::Tensor& grad, 
    at::IntArrayRef size,
    bool align_corners) {
  TORCH_CHECK(size.size() == 4, "AffineGridGeneratorBackward needs 4d (spatial) input.")

  // calculate the output size
  c10::SmallVector<int64_t, SIZE> outputSize = {size[0], 3, 2};

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(grad, outputSize, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  affine_grid_generator_backward_nocheck(
      result, 
      grad, 
      size,
      align_corners);
  auto fresult = result.transpose(1, 2);

  return fresult;
}
} // namespace native
} // namespace at_npu
