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

at::Tensor& inverse_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self) {
  at::Tensor selfCast = self;
  at::Tensor resultCast = result;
  if(self.scalar_type() == at::kHalf) {
    selfCast = self.to(at::kFloat);
  }
  if(result.scalar_type() == at::kHalf) {
    resultCast = resultCast.to(at::kFloat);
  }
  OpCommand cmd;
  cmd.Name("MatrixInverse")
      .Input(selfCast)
      .Output(resultCast)
      .Attr("adjoint", false)
      .Run();
  result.copy_(resultCast);
  return result;
}

at::Tensor& NPUNativeFunctions::inverse_out(
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
 if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor newResult = inverse_out_npu_nocheck(contiguousResult, self);
    NpuUtils::format_fresh_view(result, newResult);
  } else {
    inverse_out_npu_nocheck(result, self);
  }

  return result;
}

at::Tensor NPUNativeFunctions::inverse(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);

  inverse_out_npu_nocheck(result, self);

  return result;
}

} // namespace native
} // namespace at_npu
