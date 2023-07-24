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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor linalg_cross_output(const at::Tensor& self, const at::Tensor& other) {
    bool is_self_wrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
    return is_self_wrapped ? other : self;
}

at::Tensor& NPUNativeOpApiFunctions::linalg_cross_out(const at::Tensor& self, const at::Tensor& other,
    int64_t dim, at::Tensor& result) {
    DO_COMPATIBILITY(aclnnLinalgCross,
        NPUNativeFunctions::linalg_cross_out(self, other, dim, result));
    auto output_size = broadcast_ops_npu_output_size(self, other);
    OpPreparation::CheckOut(
        {self},
        result,
        self.scalar_type(),
        output_size);
    EXEC_NPU_CMD(aclnnLinalgCross, self, other, dim, result);
    return result;
}

at::Tensor NPUNativeOpApiFunctions::linalg_cross(const at::Tensor& self, const at::Tensor& other,
    int64_t dim) {
    DO_COMPATIBILITY(aclnnLinalgCross,
        NPUNativeFunctions::linalg_cross(self, other, dim));
    auto output_size = broadcast_ops_npu_output_size(self, other);
    at::Tensor output_tensor = linalg_cross_output(self, other);
    at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());
    EXEC_NPU_CMD(aclnnLinalgCross, self, other, dim, result);
    return result;
}

} // namespace native
} // namespace at_npu
