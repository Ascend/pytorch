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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& abs_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Abs")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor complex_abs(const at::Tensor& self) {
  at::Tensor real_self = at::native::view_as_real(self);
  auto pow_real_self = real_self.pow(2);
  auto chunk_pow = pow_real_self.chunk(2, -1);
  at::Tensor add_pow = NPUNativeFunctions::add(chunk_pow[0], chunk_pow[1], 1);
  at::Tensor cal_res = NPUNativeFunctions::sqrt(add_pow);
  at::Tensor res = NPUNativeFunctions::squeeze(cal_res, -1);
  return res;
}

at::Tensor& NPUNativeFunctions::abs_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!self.is_complex()) {
    if (!NpuUtils::check_match(&result)) {
      at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
      abs_out_npu_nocheck(contiguous_result, self);
      NpuUtils::format_fresh_view(result, contiguous_result);
    } else {
      abs_out_npu_nocheck(result, self);
    }
  } else {
    at::Tensor result_cp = complex_abs(self);
    result.copy_(result_cp);
  }
  return result;
}

at::Tensor NPUNativeFunctions::abs(const at::Tensor& self) {
  at::Tensor result;
  if (!self.is_complex()) {
    result = OpPreparation::ApplyTensor(self);
    abs_out_npu_nocheck(result, self);
  } else {
    result = complex_abs(self);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::abs_(at::Tensor& self) {
  TORCH_CHECK(
      !self.is_complex(),
      "In-place abs is not supported for complex tensors.");
  abs_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
