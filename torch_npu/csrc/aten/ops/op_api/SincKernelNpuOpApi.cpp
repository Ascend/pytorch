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

at::Tensor& NPUNativeOpApiFunctions::sinc_out(const at::Tensor& self, at::Tensor& result) {
    DO_COMPATIBILITY(aclnnSinc, NPUNativeFunctions::sinc_out(self, result));
    OpPreparation::CheckOut({self}, result, result.scalar_type(), self.sizes());
    EXEC_NPU_CMD(aclnnSinc, self, result);
    return result;
}

at::Tensor& NPUNativeOpApiFunctions::sinc_(at::Tensor& self) {
    DO_COMPATIBILITY(aclnnInplaceSinc, NPUNativeFunctions::sin_(self));
    EXEC_NPU_CMD(aclnnInplaceSinc, self);
    return self;
}

at::Tensor NPUNativeOpApiFunctions::sinc(const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnSinc, NPUNativeFunctions::sinc(self));
    auto output_size = self.sizes();
    auto outDtype = self.dtype();
    if (isIntegralType(self.scalar_type(), true)) {
        outDtype = at::kFloat;
    }
    at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(outDtype));
    EXEC_NPU_CMD(aclnnSinc, self, result);
    return result;
}

} // namespace native
} // namespace at_npu
