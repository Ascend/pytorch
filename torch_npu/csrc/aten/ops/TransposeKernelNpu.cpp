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

#include <ATen/record_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& npu_transpose_real(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef perm,
    bool require_contiguous) {
  OpCommand cmd;
  if (require_contiguous) {
    // Any tensor-view(discontiguous) Input Tensor from users should be transformed to be contiguous here.
    cmd.Name("Transpose")
        .Input(self)
        .Input(perm)
        .Output(result)
        .Run();
  } else {
    // For permute-opt in trans-contiguous, it accepts transposed(discontiguous) Input Tensor.
    cmd.Name("Transpose")
        .InputWithoutContiguous(self)
        .Input(perm)
        .Output(result)
        .Run();
  }
  return result;
}

at::Tensor& NPUNativeFunctions::npu_transpose_out(
    const at::Tensor& self,
    at::IntArrayRef perm,
    bool require_contiguous,
    at::Tensor& result) {
  if (self.is_complex()) {
    auto self_r = at::native::view_as_real(self);
    torch_npu::NPUStorageDesc &self_real_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self_r)->npu_desc_;
    self_real_desc.base_sizes_ = self_r.sizes();
    self_real_desc.base_strides_ = self_r.strides();
    auto real_and_complex = at::chunk(self_r, 2, -1);
    at::Tensor self_real = real_and_complex[0].squeeze(-1);
    at::Tensor self_complex = real_and_complex[1].squeeze(-1);
    auto output_size = transpose_npu_output_size(self_real, perm);
    at::Tensor result_real = OpPreparation::ApplyTensor(self_real, output_size);
    at::Tensor result_complex = OpPreparation::ApplyTensor(self_complex, output_size);
    npu_transpose_real(result_real, self_real, perm, require_contiguous);
    npu_transpose_real(result_complex, self_complex, perm, require_contiguous);
    at::Tensor result_cat = at::stack({result_real, result_complex}, -1);
    auto result_copy = at::native::view_as_complex(result_cat);
    result.copy_(result_copy);
  } else {
    npu_transpose_real(result, self, perm, require_contiguous);
  }
  return result;
}

at::Tensor NPUNativeFunctions::npu_transpose(const at::Tensor& self, at::IntArrayRef perm, bool require_contiguous) {
  at::SmallVector<int64_t, SIZE> new_perm;
  if (self.is_complex()) {
    for (int64_t i = 0; i < perm.size(); i++) {
      new_perm.emplace_back(make_wrap_dim(perm[i], self.dim()));
    }
  } else {
    new_perm = array_to_small_vector(perm);
  }
  
  auto output_size = transpose_npu_output_size(self, new_perm);
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  NPUNativeFunctions::npu_transpose_out(self, new_perm, require_contiguous, result);
  return result;
}

} // namespace native
} // namespace at_npu
