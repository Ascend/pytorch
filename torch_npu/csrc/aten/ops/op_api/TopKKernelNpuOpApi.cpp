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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::topk_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices) {
  DO_COMPATIBILITY(aclnnTopk, NPUNativeFunctions::topk_out(self, k, dim, largest, sorted, values, indices));
  at::Tensor self_cp = OpPreparation::CastBackToOriFormat(self);
  auto output_size = topk_npu_output_size(self_cp, k, dim, largest, sorted);
  
  OpPreparation::CheckOut(
    {self_cp},
    values,
    self_cp,
    output_size);
  
  OpPreparation::CheckOut(
    {self_cp},
    indices,
    CalcuOpUtil::GetTensorNpuFormat(self_cp),
    at::ScalarType::Long,
    output_size);

  EXEC_NPU_CMD(aclnnTopk, self, k, dim, largest, sorted, values, indices);

  return tuple<at::Tensor&, at::Tensor&>(values, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::topk(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  DO_COMPATIBILITY(aclnnTopk, NPUNativeFunctions::topk(self, k, dim, largest, sorted));
  at::Tensor self_cp = OpPreparation::CastBackToOriFormat(self);
  auto output_size = topk_npu_output_size(self_cp, k, dim, largest, sorted);

  at::Tensor values = OpPreparation::ApplyTensor(output_size, self_cp.options(), self_cp);
  at::Tensor indices = OpPreparation::ApplyTensor(output_size, self_cp.options().dtype(at::kLong), self_cp);

  EXEC_NPU_CMD(aclnnTopk, self_cp, k, dim, largest, sorted, values, indices);
  return tuple<at::Tensor, at::Tensor>(values, indices);
}
} // namespace native
} // namespace at_npu
