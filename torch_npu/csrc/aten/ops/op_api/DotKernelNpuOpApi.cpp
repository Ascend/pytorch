// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::dot_out(const at::Tensor& self, const at::Tensor& tensor, at::Tensor& result) {
  c10::SmallVector<int64_t, SIZE> outputSize = dot_npu_output_size(self, tensor);
  OpPreparation::CheckOut({self, tensor}, result, CalcuOpUtil::GetTensorNpuFormat(self), self.scalar_type(), outputSize);
             
  EXEC_NPU_CMD(aclnnDot, self, tensor, result);

  c10::SmallVector<int64_t, N> shape = {};
  result.resize_(shape);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::dot(const at::Tensor& self, const at::Tensor& tensor) {
  c10::SmallVector<int64_t, SIZE> outputSize = dot_npu_output_size(self, tensor);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  NPUNativeOpApiFunctions::dot_out(self, tensor, result);
  return result;
}
} // namespace native
} // namespace at_npu