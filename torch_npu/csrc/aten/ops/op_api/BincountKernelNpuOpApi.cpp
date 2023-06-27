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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor NPUNativeOpApiFunctions::bincount(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t minlength) {
  DO_COMPATIBILITY(aclnnBincount,
                   NPUNativeFunctions::bincount(self, weight_opt, minlength));
  // null tensor
  if (self.dim() == 1 && self.numel() == 0) {
      auto result = OpPreparation::ApplyTensorWithSizes(
          {0}, 
          self.options().dtype(at::ScalarType::Long));
      return result;
  } 

  // cheack non-negative
  auto min_value = NPUNativeOpApiFunctions::min(self).item().toLong();
  TORCH_CHECK(min_value >= 0, "bincount only support 1-d non-negative integral inputs.");

  const at::Tensor& weights = c10::value_or_else(weight_opt, [] {return at::Tensor();});

  // calculate output size
  auto sizes = NPUNativeOpApiFunctions::max(self).item().toLong();
  sizes = (sizes < minlength) ? minlength : (sizes + 1);

  // weight convert dtype as same as output defined by torch
  at::Tensor result;
  if (!weights.defined()) {
      result = OpPreparation::ApplyTensorWithSizes({sizes}, self.options().dtype(at::ScalarType::Long));
  } else if (weights.dtype() == at::ScalarType::Float) {
      result = OpPreparation::ApplyTensorWithSizes({sizes}, weights.options().dtype(at::ScalarType::Float));
  } else {
      result = OpPreparation::ApplyTensorWithSizes({sizes}, weights.options().dtype(at::ScalarType::Double));
  }
  
  EXEC_NPU_CMD(aclnnBincount, self, weights, minlength, result);

  return result;
}
} // namespace native
} // namespace at
