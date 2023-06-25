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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {


tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::adaptive_max_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& output,
    at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnAdaptiveMaxPool2d,
                   NPUNativeFunctions::adaptive_max_pool2d_out(self, output_size, output, indices));
  OpPreparation::CheckMemory({self}, {output, indices});
  
  auto outputSize = max_pool2d_out_size(self, output_size);

  OpPreparation::CheckOut({self}, output, self.scalar_type(), outputSize);
  OpPreparation::CheckOut({self}, indices, at::ScalarType::Long, outputSize);
  
  EXEC_NPU_CMD(aclnnAdaptiveMaxPool2d, self, output_size, output, indices);
  return tuple<at::Tensor&, at::Tensor&>(output, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::adaptive_max_pool2d(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  DO_COMPATIBILITY(aclnnAdaptiveMaxPool2d, NPUNativeFunctions::adaptive_max_pool2d(self, output_size));
  
  auto outputSize = max_pool2d_out_size(self, output_size);
  
  at::Tensor output = OpPreparation::ApplyTensor(self, outputSize);
  at::Tensor indices = OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::kLong), self);

  NPUNativeOpApiFunctions::adaptive_max_pool2d_out(self, output_size, output, indices);
  return tuple<at::Tensor, at::Tensor>(output, indices);
}


} // namespace native
} // namespace at_npu

