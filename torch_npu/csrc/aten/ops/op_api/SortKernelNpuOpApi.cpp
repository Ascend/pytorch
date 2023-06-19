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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor &, at::Tensor &> sort_output(const at::Tensor &self, bool stable, int64_t dim, bool descending,
    at::Tensor &values, at::Tensor &indices) {
    EXEC_NPU_CMD(aclnnSort, self, stable, dim, descending, values, indices);

    return std::tie(values, indices);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::sort(const at::Tensor &self, int64_t dim, bool descending) {
    DO_COMPATIBILITY(aclnnSort, NPUNativeFunctions::sort(self, dim, descending));
    at::Tensor values = OpPreparation::ApplyTensor(self);
    at::Tensor indices = OpPreparation::ApplyTensor(self, self.options().dtype(at::kLong));
    bool stable = false;

    return sort_output(self, stable, dim, descending, values, indices);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::sort(const at::Tensor &self, c10::optional<bool> stable,
    int64_t dim, bool descending) {
    DO_COMPATIBILITY(aclnnSort, NPUNativeFunctions::sort(self, stable, dim, descending));
    at::Tensor values = OpPreparation::ApplyTensor(self);
    at::Tensor indices = OpPreparation::ApplyTensor(self, self.options().dtype(at::kLong));
    bool argStable = c10::value_or_else(stable, [] { return false; });

    return sort_output(self, argStable, dim, descending, values, indices);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::sort(const at::Tensor &self, at::Dimname dim,
    bool descending) {
    DO_COMPATIBILITY(aclnnSort, NPUNativeFunctions::sort(self, dim, descending));
    at::Tensor values = OpPreparation::ApplyTensor(self);
    at::Tensor indices = OpPreparation::ApplyTensor(self, self.options().dtype(at::kLong));
    bool stable = false;
    int64_t argDim = dimname_to_position(self, dim);

    return sort_output(self, stable, argDim, descending, values, indices);
}

std::tuple<at::Tensor &, at::Tensor &> NPUNativeOpApiFunctions::sort_out(const at::Tensor &self, int64_t dim,
    bool descending, at::Tensor &values, at::Tensor &indices) {
    DO_COMPATIBILITY(aclnnSort, NPUNativeFunctions::sort_out(self, dim, descending, values, indices));
    OpPreparation::CheckOut({self}, values, values.scalar_type(), self.sizes());
    OpPreparation::CheckOut({self}, indices, indices.scalar_type(), self.sizes());
    bool stable = false;

    return sort_output(self, stable, dim, descending, values, indices);
}

std::tuple<at::Tensor &, at::Tensor &> NPUNativeOpApiFunctions::sort_out(const at::Tensor &self,
    c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor &values, at::Tensor &indices) {
    DO_COMPATIBILITY(aclnnSort, NPUNativeFunctions::sort_out(self, stable, dim, descending, values, indices));
    OpPreparation::CheckOut({self}, values, values.scalar_type(), self.sizes());
    OpPreparation::CheckOut({self}, indices, indices.scalar_type(), self.sizes());
    bool argStable = c10::value_or_else(stable, [] { return false; });

    return sort_output(self, argStable, dim, descending, values, indices);
}

std::tuple<at::Tensor &, at::Tensor &> NPUNativeOpApiFunctions::sort_out(const at::Tensor &self, at::Dimname dim,
    bool descending, at::Tensor &values, at::Tensor &indices) {
    DO_COMPATIBILITY(aclnnSort, NPUNativeFunctions::sort_out(self, dim, descending, values, indices));
    OpPreparation::CheckOut({self}, values, values.scalar_type(), self.sizes());
    OpPreparation::CheckOut({self}, indices, indices.scalar_type(), self.sizes());
    bool stable = false;
    
    return sort_output(self, stable, dimname_to_position(self, dim), descending, values, indices);
}

} // namespace native
} // namespace at_npu

