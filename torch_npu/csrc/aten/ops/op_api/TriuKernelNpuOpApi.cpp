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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::triu_out(const at::Tensor& self, int64_t diag, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnTriu, NPUNativeFunctions::triu_out(self, diag, result));
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  EXEC_NPU_CMD(aclnnTriu, self, diag, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::triu(const at::Tensor& self, int64_t diag) {
  DO_COMPATIBILITY(aclnnTriu, NPUNativeFunctions::triu(self, diag));
  at::Tensor result = OpPreparation::ApplyTensor(self);
  EXEC_NPU_CMD(aclnnTriu, self, diag, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::triu_(at::Tensor& self, int64_t diag) {
  DO_COMPATIBILITY(aclnnInplaceTriu, NPUNativeFunctions::triu_(self, diag));
  EXEC_NPU_CMD(aclnnInplaceTriu, self, diag);
  return self;
}

} // namespace native
} // namespace at_npu
