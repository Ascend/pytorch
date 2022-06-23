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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& affine_grid_generator_npu_nocheck(
    Tensor& result,
    const Tensor& theta,
    IntArrayRef size,
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

Tensor affine_grid_generator_npu(
    const Tensor& theta,
    IntArrayRef size,
    bool align_corners) {
  TORCH_CHECK(size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d or 5d size(input).");
  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = { };
  if(size.size() == 4) {
    outputSize = {size[0], size[2] * size[3], 2};
  } else {
    outputSize = {size[0], size[2] * size[3] * size[4], 3};
  }

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(theta, outputSize);
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
} // namespace at
