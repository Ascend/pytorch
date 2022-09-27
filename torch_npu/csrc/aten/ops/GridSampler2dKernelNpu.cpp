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

at::Tensor NPUNativeFunctions::grid_sampler_2d(
    const at::Tensor& self, 
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

  at::Tensor dtypeCastOfSelf = self;
  at::Tensor dtypeCastOfGrid = grid;
  if (dtypeCastOfSelf.scalar_type() == c10::ScalarType::Half) {
    dtypeCastOfSelf = NPUNativeFunctions::npu_dtype_cast(dtypeCastOfSelf, c10::ScalarType::Float);
  }
  if (dtypeCastOfGrid.scalar_type() == c10::ScalarType::Half) {
    dtypeCastOfGrid = NPUNativeFunctions::npu_dtype_cast(dtypeCastOfGrid, c10::ScalarType::Float);
  }

  c10::SmallVector<int64_t, SIZE> outputSize = {dtypeCastOfSelf.size(0),
                                                dtypeCastOfSelf.size(1),
                                                dtypeCastOfGrid.size(1),
                                                dtypeCastOfGrid.size(2)};

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(dtypeCastOfSelf, outputSize, ACL_FORMAT_ND);

  OpCommand cmd;
  cmd.Name("GridSampler2D")
      .Input(dtypeCastOfSelf)
      .Input(dtypeCastOfGrid)
      .Output(result)
      .Attr("interpolation_mode", interpolation_mode)
      .Attr("padding_mode", padding_mode)
      .Attr("align_corners", align_corners)
      .Run();

  c10::ScalarType selfScalarType(self.scalar_type());
  if (result.scalar_type() != selfScalarType) {
    result = NPUNativeFunctions::npu_dtype_cast(result,selfScalarType);
  }
  return result;
}
} // namespace native
} // namespace at_npu
