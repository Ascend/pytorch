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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& expm1_out_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Expm1")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::expm1_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(result),
      self.scalar_type(),
      self.sizes());

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    expm1_out_nocheck(contiguous_result, self);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    expm1_out_nocheck(result, self);
  }
  return result;
}

at::Tensor NPUNativeFunctions::expm1(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  expm1_out_nocheck(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::expm1_(at::Tensor& self) {
  return NPUNativeFunctions::expm1_out(self, self);
}

} // namespace native
} // namespace at_npu
