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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include <ATen/ops/bucketize_cpu_dispatch.h>

namespace at_npu {
namespace native {

at::Tensor &NPUNativeFunctions::bucketize_out(
    const at::Tensor& self, 
    const at::Tensor& boundaries, 
    bool out_int32, 
    bool right, 
    at::Tensor& result) {
    TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
    const auto self_cpu = self.cpu();
    const auto boundaries_cpu = boundaries.cpu();
    OpPreparation::CheckOut(
        {self},
        result,
        self);
    auto out_cpu = result.cpu();
    at::cpu::bucketize_out(out_cpu, self_cpu, boundaries_cpu, out_int32, right);
    result.copy_(out_cpu);
    return result;
}

at::Tensor NPUNativeFunctions::bucketize(
    const at::Tensor& self, 
    const at::Tensor& boundaries, 
    bool out_int32, 
    bool right) {
    at::ScalarType scalar_type = out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
    c10::TensorOptions options = at::TensorOptions().device(self.options().device()).dtype(scalar_type);
    at::Tensor result = at::empty({0}, options, at::MemoryFormat::Contiguous).to(self.device());
    NPUNativeFunctions::bucketize_out(self, boundaries, out_int32, right, result);
    return result;
}

} // namespace native
} // namespace at_npu
