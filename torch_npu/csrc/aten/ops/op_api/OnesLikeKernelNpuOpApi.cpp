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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor NPUNativeOpApiFunctions::ones_like(const at::Tensor &self,
                                              c10::optional<c10::ScalarType> dtype_opt,
                                              c10::optional<c10::Layout> layout_opt,
                                              c10::optional<c10::Device> device_opt,
                                              c10::optional<bool> pin_memory_opt,
                                              c10::optional<c10::MemoryFormat> optional_memory_format) {
  DO_COMPATIBILITY(aclnnInplaceOne, NPUNativeFunctions::ones_like(self, dtype_opt, layout_opt, device_opt,
                                                                  pin_memory_opt, optional_memory_format));
  c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                                  .device(device_opt)
                                                  .layout(layout_opt)
                                                  .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), option);

  EXEC_NPU_CMD(aclnnInplaceOne, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::one_(at::Tensor &self) {
  DO_COMPATIBILITY(aclnnInplaceOne, NPUNativeFunctions::one_(self));
  EXEC_NPU_CMD(aclnnInplaceOne, self);
  return self;
}
} // namespace native
} // namespace at_npu
