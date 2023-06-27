// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::_s_where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnSWhere, NPUNativeFunctions::_s_where(condition, self, other));
  auto broadcastOutputSize = broadcast_ops_npu_output_size(self, other);
  auto outputSize = broadcast_ops_npu_output_size(condition.sizes(), broadcastOutputSize);

  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  EXEC_NPU_CMD(aclnnSWhere, condition, self, other, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnSWhere, NPUNativeFunctions::where(condition, self, other));
  return at::_s_where(condition, self, other);
}

} // namespace native
} // namespace at_npu