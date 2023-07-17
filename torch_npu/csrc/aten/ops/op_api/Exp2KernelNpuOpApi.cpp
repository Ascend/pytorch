// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::exp2_out(const at::Tensor& self, at::Tensor& out) {

    DO_COMPATIBILITY(aclnnExp2, NPUNativeFunctions::exp2_out(self, out));
    OpPreparation::CheckOut({self}, out, out.scalar_type(), self.sizes());

    EXEC_NPU_CMD(aclnnExp2, self, out);

    return out;
}

at::Tensor NPUNativeOpApiFunctions::exp2(const at::Tensor& self) {
    
    DO_COMPATIBILITY(aclnnExp2, NPUNativeFunctions::exp2(self));

    auto out_Dtype = self.dtype();
    if (isIntegralType(self.scalar_type(), true)) {
        out_Dtype = at::ScalarType::Float;
    }
    
    at::Tensor out = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), self.options().dtype(out_Dtype));
    
    EXEC_NPU_CMD(aclnnExp2, self, out);
    return out;
}
    
   
}  // namespace native
}  // namespace at_npu

