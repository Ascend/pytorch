// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

at::Tensor& grid_sampler_npu_nocheck(
    const at::Tensor& self, 
    const at::Tensor& grid,
    int64_t interpolation_mode, 
    int64_t padding_mode, 
    bool align_corners,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("GridSampler2D")
      .Input(self)
      .Input(grid)
      .Output(result)
      .Attr("interpolation_mode", interpolation_mode)
      .Attr("padding_mode", padding_mode)
      .Attr("align_corners", align_corners)
      .Run();    
  return result;
}

at::Tensor NPUNativeFunctions::grid_sampler(
    const at::Tensor& self, 
    const at::Tensor& grid,
    int64_t interpolation_mode, 
    int64_t padding_mode, 
    bool align_corners) {
  at::Tensor formatCastOfSelf = NPUNativeFunctions::npu_format_cast(self, ACL_FORMAT_ND);
  at::Tensor formatCastOfGrid = NPUNativeFunctions::npu_format_cast(grid, ACL_FORMAT_ND);
  if (formatCastOfSelf.scalar_type() == at::ScalarType::Half) {
    formatCastOfSelf = NPUNativeFunctions::npu_dtype_cast(formatCastOfSelf, at::ScalarType::Float);
  }
  if (formatCastOfGrid.scalar_type() == at::ScalarType::Half) {
    formatCastOfGrid = NPUNativeFunctions::npu_dtype_cast(formatCastOfGrid, at::ScalarType::Float);
  }
  c10::SmallVector<int64_t, SIZE> outputSize = {formatCastOfSelf.size(0),
                                           formatCastOfSelf.size(1),
                                           formatCastOfGrid.size(1),
                                           formatCastOfGrid.size(2)};
  auto result = OpPreparation::ApplyTensorWithFormat(
      outputSize, formatCastOfSelf.options(), ACL_FORMAT_ND);
  grid_sampler_npu_nocheck(
      formatCastOfSelf, 
      formatCastOfGrid,  
      interpolation_mode,
      padding_mode,
      align_corners,
      result);
  if (result.scalar_type() != self.scalar_type()) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }
  return result;
}

} // namespace native
} // namespace at_npu