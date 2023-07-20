// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2013, Facebook CORPORATION.
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

at::Tensor& NPUNativeOpApiFunctions::log1p_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLog1p, NPUNativeFunctions::log1p_out(self, result));
  TORCH_CHECK(!isIntegralType(result.scalar_type(), true),
              "result type Float can't be cast to the desired output type ", toString(self.scalar_type()));
  auto output_size = self.sizes();
  OpPreparation::CheckOut({self}, result, result.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnLog1p, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::log1p_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceLog1p, NPUNativeFunctions::log1p_(self));
  TORCH_CHECK(!isIntegralType(self.scalar_type(), true),
              "result type Float can't be cast to the desired output type ", toString(self.scalar_type()));
  EXEC_NPU_CMD(aclnnInplaceLog1p, self);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::log1p(const at::Tensor& self) {
  auto output_size = self.sizes();
  auto out_dtype = self.dtype();
  DO_COMPATIBILITY(aclnnLog1p, NPUNativeFunctions::log1p(self));
  if (isIntegralType(self.scalar_type(), true)) {
    out_dtype = at::kFloat;
  }
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(out_dtype));
  EXEC_NPU_CMD(aclnnLog1p, self, result);
  return result;
}

} // namespace native
} // namespace at_npu
