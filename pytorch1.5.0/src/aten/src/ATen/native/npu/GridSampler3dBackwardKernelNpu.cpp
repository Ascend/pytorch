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

#include <c10/npu/OptionsManager.h>
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_npu(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  TORCH_CHECK(
      (0 <= interpolation_mode && interpolation_mode <= 2),
      "interpolation_mode must be in range [0~2].")
  TORCH_CHECK(
      (0 <= padding_mode && padding_mode <= 2),
      "padding_mode must be in range [0~2].")
  Tensor formatCastOfGrad = grad;
  Tensor formatCastOfInput = input;
  Tensor formatCastOfGrid = grid;
  if (formatCastOfGrad.scalar_type() == ScalarType::Half) {
    formatCastOfGrad = formatCastOfGrad.npu_dtype_cast(ScalarType::Float);
  }
  if (formatCastOfInput.scalar_type() == ScalarType::Half) {
    formatCastOfInput = formatCastOfInput.npu_dtype_cast(ScalarType::Float);
  }
  if (formatCastOfGrid.scalar_type() == ScalarType::Half) {
    formatCastOfGrid = formatCastOfGrid.npu_dtype_cast(ScalarType::Float);
  }
  // construct the output tensor of the NPU
  Tensor dx = OpPreparation::ApplyTensor(formatCastOfInput);
  Tensor dgrid = OpPreparation::ApplyTensor(formatCastOfGrid);
  std::string interMode[] = {"bilinear", "nearest", "bicubic"};
  std::string paddingMode[] = {"zeros", "border", "reflection"};
  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("GridSampler3DGrad")
      .Input(formatCastOfGrad)
      .Input(formatCastOfInput)
      .Input(formatCastOfGrid)
      .Output(dx)
      .Output(dgrid)
      .Attr("interpolation_mode", interMode[interpolation_mode])
      .Attr("padding_mode", paddingMode[padding_mode])
      .Attr("align_corners", align_corners)
      .Run();
  return std::tuple<Tensor, Tensor>(dx, dgrid);
}
} // namespace native
} // namespace at
