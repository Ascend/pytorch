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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::maximum_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMaximum, NPUNativeFunctions::maximum_out(self, other, result));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self, other}, result, result, outputSize);
  EXEC_NPU_CMD(aclnnMaximum, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::maximum(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnMaximum, NPUNativeFunctions::maximum(self, other));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options().dtype(high_type), self);
  EXEC_NPU_CMD(aclnnMaximum, self, other, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
