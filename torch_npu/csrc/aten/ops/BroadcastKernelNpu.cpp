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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"

namespace at_npu {
namespace native {

at::Tensor& npu_broadcast_out_nocheck(at::Tensor& result, const at::Tensor& self, at::IntArrayRef size) {
  OpCommand cmd;
  cmd.Name("BroadcastTo")
      .InputWithoutContiguous(self)
      .Input(size, at::kLong)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::npu_broadcast_out(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::Tensor &result) {
  if (!self.is_complex()) {
    npu_broadcast_out_nocheck(result, self, size);
  } else {
    c10::SmallVector<at::Tensor, N> real_and_complex = complex_compute_split(self);
    at::Tensor self_real = real_and_complex[0].squeeze(-1);
    at::Tensor self_complex = real_and_complex[1].squeeze(-1);

    at::Tensor result_real = OpPreparation::ApplyTensor(self_real, size);
    at::Tensor result_complex = OpPreparation::ApplyTensor(self_complex, size);
    at::Tensor self_real_contiguous = self_real.is_contiguous() ? self_real : self_real.contiguous();
    at::Tensor self_complex_contiguous = self_complex.is_contiguous() ? self_complex : self_complex.contiguous();
    change_base_sizes_and_base_strides(self_real_contiguous);
    change_base_sizes_and_base_strides(self_complex_contiguous);

    npu_broadcast_out_nocheck(result_real, self_real_contiguous, size);
    npu_broadcast_out_nocheck(result_complex, self_complex_contiguous, size);

    at::Tensor result_cat = NPUNativeFunctions::stack({result_real, result_complex}, -1);
    at::Tensor result_cp = at::native::view_as_complex(result_cat);
    result.copy_(result_cp);
  }
  return result;
}

at::Tensor NPUNativeFunctions::npu_broadcast(const at::Tensor &self, at::IntArrayRef size) {
  at::Tensor input = self.is_contiguous() ? self : self.contiguous();
  if (self.dtype() == at::kBool) {
    input = NPUNativeFunctions::npu_dtype_cast(input, at::kInt);
  }

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      size,
      input.options(),
      CalcuOpUtil::GetTensorNpuFormat(self));

  NPUNativeFunctions::npu_broadcast_out(input, size, result);

  if (self.dtype() == at::kBool) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }

  return result;
}

} // namespace native
} // namespace at_npu
