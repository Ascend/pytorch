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

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::tril_out(const at::Tensor& self, int64_t diagonal, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnTril, NPUNativeFunctions::tril_out(self, diagonal, result));
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  EXEC_NPU_CMD(aclnnTril, self, diagonal, result);
  
  return result;
}

at::Tensor NPUNativeOpApiFunctions::tril(const at::Tensor& self, int64_t diagonal) {
  DO_COMPATIBILITY(aclnnTril, NPUNativeFunctions::tril(self, diagonal));
  at::Tensor result = OpPreparation::ApplyTensor(self);
  EXEC_NPU_CMD(aclnnTril, self, diagonal, result);

  return result;
}

at::Tensor& NPUNativeOpApiFunctions::tril_(at::Tensor& self, int64_t diagonal) {
  DO_COMPATIBILITY(aclnnTril, NPUNativeFunctions::tril_(self, diagonal));
  NPUNativeOpApiFunctions::tril_out(self, diagonal, self);

  return self;
}
} // namespace native
} // namespace at_npu
