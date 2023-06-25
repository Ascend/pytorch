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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::max_unpool2d_out(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef outputSize,
    at::Tensor& output) {
  DO_COMPATIBILITY(aclnnMaxUnpool2d, NPUNativeFunctions::max_unpool2d_out(self, indices, outputSize, output));
  
  auto output_size = max_pool2d_out_size(self, outputSize);
  OpPreparation::CheckOut({self, indices}, output, self.scalar_type(), output_size);

  EXEC_NPU_CMD(aclnnMaxUnpool2d, self, indices, outputSize, output);
  return output;
};

at::Tensor NPUNativeOpApiFunctions::max_unpool2d(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size) {
  DO_COMPATIBILITY(aclnnMaxUnpool2d, NPUNativeFunctions::max_unpool2d(self, indices, output_size));
  
  auto outputSize = max_pool2d_out_size(self, output_size);
  at::Tensor output = OpPreparation::ApplyTensor(self, outputSize);
  NPUNativeOpApiFunctions::max_unpool2d_out(self, indices, output_size, output);
  return output;
}
} // namespace native
} // namespace at_npu

