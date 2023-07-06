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

at::Tensor& NPUNativeOpApiFunctions::expm1_out(const at::Tensor& self, at::Tensor& out) {
  DO_COMPATIBILITY(aclnnExpm1, NPUNativeFunctions::expm1_out(self, out));
  // check if out on NPU
  TORCH_CHECK(at_npu::key::isDeviceTensor(out), "out with device ", out.device(),
              " doesn't match the desired device NPU");
  // resize_ the output size when size of out and self don't match with each other.
  if (out.sizes() != self.sizes()) {
    auto output_size = input_same_output_size(self);
    out.resize_(output_size);
  }
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnExpm1, self, out);
  return out;
}

at::Tensor NPUNativeOpApiFunctions::expm1(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnExpm1, NPUNativeFunctions::expm1(self));
  // construct the output tensor of NPU
  at::Tensor out = OpPreparation::ApplyTensorWithoutFormat(self);
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnExpm1, self, out);
  return out;
}

} // namespace native
} // namespace at_npu
