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

at::Tensor& NPUNativeOpApiFunctions::trunc_out(const at::Tensor& self, at::Tensor& result) {
    DO_COMPATIBILITY(aclnnTrunc,
        NPUNativeFunctions::trunc_out(self, result));
    auto outputSize = self.sizes();
    OpPreparation::CheckOut(
        {self}, result, self.scalar_type(), outputSize);
    EXEC_NPU_CMD(aclnnTrunc, self, result);
    return result;
}

at::Tensor& NPUNativeOpApiFunctions::trunc_(at::Tensor& self) {
    DO_COMPATIBILITY(aclnnInplaceTrunc,
        NPUNativeFunctions::trunc_(self));
    EXEC_NPU_CMD(aclnnInplaceTrunc, self);
    return self;
}

at::Tensor NPUNativeOpApiFunctions::trunc(const at::Tensor& self) {
    auto outputSize = self.sizes();
    DO_COMPATIBILITY(aclnnTrunc,
        NPUNativeFunctions::trunc(self));
    at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options());
    EXEC_NPU_CMD(aclnnTrunc, self, result);
    return result;
}

} // namespace native
} // namespace at_npu
