// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor &NPUNativeOpApiFunctions::mv_out(const at::Tensor &self, const at::Tensor &vec, at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnMv, NPUNativeFunctions::mv_out(self, vec, result));
  OpPreparation::CheckOut({self, vec}, result, result.scalar_type(), {self.size(0)});
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnMv, self, vec, result, cube_math_type);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::mv(const at::Tensor &self, const at::Tensor &vec)
{
  DO_COMPATIBILITY(aclnnMv, NPUNativeFunctions::mv(self, vec));
  at::Tensor result = OpPreparation::ApplyTensorWithSizes({self.size(0)}, vec.options());
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnMv, self, vec, result, cube_math_type);
  return result;
}

} // namespace native
} // namespace at_npu
