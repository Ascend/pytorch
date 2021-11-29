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

Tensor grid_sampler_2d_npu(
    const Tensor& self,
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

  Tensor formatCastOfSelf = self;
  Tensor formatCastOfGrid = grid;
  if (formatCastOfSelf.scalar_type() == ScalarType::Half) {
    formatCastOfSelf = formatCastOfSelf.npu_dtype_cast(ScalarType::Float);
  }
  if (formatCastOfGrid.scalar_type() == ScalarType::Half) {
    formatCastOfGrid = formatCastOfGrid.npu_dtype_cast(ScalarType::Float);
  }

  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = {formatCastOfSelf.size(0),
                                           formatCastOfSelf.size(1),
                                           formatCastOfGrid.size(1),
                                           formatCastOfGrid.size(2)};
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(formatCastOfSelf, outputSize, ACL_FORMAT_ND);

  OpCommand cmd;
  cmd.Name("GridSampler2D")
      .Input(formatCastOfSelf)
      .Input(formatCastOfGrid)
      .Output(result)
      .Attr("interpolation_mode", interpolation_mode)
      .Attr("padding_mode", padding_mode)
      .Attr("align_corners", align_corners)
      .Run();

  ScalarType selfScalarType(self.scalar_type());
  if (result.scalar_type() != selfScalarType) {
    result = result.npu_dtype_cast(selfScalarType);
  }

  return result;
}
} // namespace native
} // namespace at
