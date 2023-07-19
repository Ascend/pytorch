// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor &NPUNativeOpApiFunctions::logaddexp_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out) {
  DO_COMPATIBILITY(aclnnLogAddExp, NPUNativeFunctions::logaddexp_out(self, other, out));
  EXEC_NPU_CMD(aclnnLogAddExp, self, other, out);

  return out;
}

at::Tensor NPUNativeOpApiFunctions::logaddexp(const at::Tensor &self, const at::Tensor &other) {
  DO_COMPATIBILITY(aclnnLogAddExp, NPUNativeFunctions::logaddexp(self, other));
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());
  EXEC_NPU_CMD(aclnnLogAddExp, self, other, result);

  return result;
}

} // namespace native
} // namespace at_npu
