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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::grid_sampler_2d_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  TORCH_CHECK(
      (0 <= interpolation_mode && interpolation_mode <= 2),
      "interpolation_mode must be in range [0~2].")
  TORCH_CHECK(
      (0 <= padding_mode && padding_mode <= 2),
      "padding_mode must be in range [0~2].")

  at::Tensor formatCastOfGrad = grad;
  at::Tensor formatCastOfInput = input;
  at::Tensor formatCastOfGrid = grid;
  if (formatCastOfGrad.scalar_type() == at::ScalarType::Half) {
    formatCastOfGrad = NPUNativeFunctions::npu_dtype_cast(formatCastOfGrad, at::ScalarType::Float);
  }
  if (formatCastOfInput.scalar_type() == at::ScalarType::Half) {
    formatCastOfInput = NPUNativeFunctions::npu_dtype_cast(formatCastOfInput, at::ScalarType::Float);
  }
  if (formatCastOfGrid.scalar_type() == at::ScalarType::Half) {
    formatCastOfGrid = NPUNativeFunctions::npu_dtype_cast(formatCastOfGrid, at::ScalarType::Float);
  }

  at::Tensor dx = OpPreparation::ApplyTensor(formatCastOfInput);
  at::Tensor dgrid = OpPreparation::ApplyTensor(formatCastOfGrid);

  c10::SmallVector<string, SIZE>interMode = {"bilinear", "nearest", "bicubic"};
  c10::SmallVector<string, SIZE>paddingMode = {"zeros", "border", "reflection"};

  OpCommand cmd;
  cmd.Name("GridSampler2DGrad")
      .Input(formatCastOfGrad)
      .Input(formatCastOfInput)
      .Input(formatCastOfGrid)
      .Output(dx)
      .Output(dgrid)
      .Attr("interpolation_mode", interMode[interpolation_mode])
      .Attr("padding_mode", paddingMode[padding_mode])
      .Attr("align_corners", align_corners)
      .Run();

  at::ScalarType inputScalarType(input.scalar_type());
  if (dx.scalar_type() != inputScalarType) {
    dx = NPUNativeFunctions::npu_dtype_cast(dx, inputScalarType);
    dgrid = NPUNativeFunctions::npu_dtype_cast(dgrid, inputScalarType);
  }

  return std::tuple<at::Tensor, at::Tensor>(dx, dgrid);
}
} // namespace native
} // namespace at_npu
