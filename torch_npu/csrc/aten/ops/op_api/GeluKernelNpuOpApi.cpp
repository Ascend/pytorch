// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::gelu_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGelu, NPUNativeFunctions::gelu_out(self, result));
  OpPreparation::CheckOut({self}, result, self);

  EXEC_NPU_CMD(aclnnGelu, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::gelu(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnGelu, NPUNativeFunctions::gelu(self));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  EXEC_NPU_CMD(aclnnGelu, self, result);
  return result;
}
}  // namespace native
}  // namespace at_npu
