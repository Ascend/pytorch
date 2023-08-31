// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::gcd_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGcd, NPUNativeFunctions::gcd_out(self, other, result));
  at::IntArrayRef output_size;
  output_size = broadcast_ops_npu_output_size(self, other);
  // Shape of result must be the same as broadcastshape of self and other, dtype has no limitation.
  if (result.sizes() != output_size) {
    result.resize_(output_size);
  }
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnGcd, self, other, result);
  return result;
}

} // namespace native
} // namespace at_npu
