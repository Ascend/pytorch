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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& affine_grid_generator_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& theta,
    at::IntArrayRef size,
    bool align_corners) {

  OpCommand cmd;
  cmd.Name("AffineGrid")
      .Input(theta)
      .Input(size, at::kInt)
      .Output(result)
      .Attr("align_corners", align_corners)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::affine_grid_generator(
    const at::Tensor& theta,
    at::IntArrayRef size,
    bool align_corners) {
  TORCH_CHECK(size.size() == 4 || size.size() == 5, 
      "AffineGridGenerator needs 4d or 5d size(input).");
  // calculate the output size
  at::SmallVector<int64_t, SIZE> outputSize = { };
  if(size.size() == 4) {
    outputSize = {size[0], size[2] * size[3], 2};
  } else {
    outputSize = {size[0], size[2] * size[3] * size[4], 3};
  }

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(theta, outputSize);
  // calculate the output result of the NPU
  affine_grid_generator_npu_nocheck(
      result, 
      theta, 
      size,
      align_corners);

  if(size.size() == 4) {
    result = result.view({size[0], size[2], size[3], 2});
  } else {
    result = result.view({size[0], size[2], size[3], size[4], 3});
  }
  return result;
}

} // namespace native
} // namespace at_npu