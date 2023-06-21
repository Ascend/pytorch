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

namespace at_npu {
namespace native {
void cummax_out_npu_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim) {
  OpCommand cmd;
  cmd.Name("Cummax")
      .Input(self)
      .Output(values)
      .Output(indices)
      .Attr("dim", dim)
      .Run();
}

void NPUNativeFunctions::_cummax_helper(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim) {
  at::Tensor values_temp = OpPreparation::ApplyTensor(self);
  at::Tensor indices_temp = OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options().dtype(at::kLong),
      ACL_FORMAT_ND);
  cummax_out_npu_nocheck(values_temp, indices_temp, self, dim);

  values.copy_(values_temp);
  indices.copy_(indices_temp);
}

} // namespace native
} // namespace at_npu
