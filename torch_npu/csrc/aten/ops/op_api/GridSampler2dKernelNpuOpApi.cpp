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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::grid_sampler_2d(
    const at::Tensor& self,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  DO_COMPATIBILITY(aclnnGridSampler2D, NPUNativeFunctions::grid_sampler_2d(self, grid, interpolation_mode,
                                                                           padding_mode, align_corners));
  auto outputSize = {self.size(0), self.size(1), grid.size(1), grid.size(2)};
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options());
  EXEC_NPU_CMD(aclnnGridSampler2D, self, grid, interpolation_mode, padding_mode, align_corners, result);
  return result;
}

} // namespace native
} // namespace at_npu
