// Copyright (c) 2022, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor& logaddexp_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  at::Tensor self_exp = OpPreparation::ApplyTensor(self);
  at::Tensor other_exp = OpPreparation::ApplyTensor(self);

  OpCommand cmd_exp_1, cmd_exp_2, cmd_add, cmd_log;
  cmd_exp_1.Name("Exp")
      .Input(self)
      .Output(self_exp)
      .Run();

  cmd_exp_2.Name("Exp")
      .Input(other)
      .Output(other_exp)
      .Run();

  at::Tensor add_result = OpPreparation::ApplyTensor(self);
  auto unified_result =
      OpPreparation::binary_op_check(add_result, self_exp, other_exp, true);

  cmd_add.Name("Add")
      .Expect(unified_result)
      .Input(self_exp)
      .Input(other_exp)
      .Output(add_result)
      .Run();

  cmd_log.Name("Log")
      .Input(add_result)
      .Output(result)
      .Attr("base", (float)-1)
      .Attr("scale", (float)1)
      .Attr("shift", (float)0)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::logaddexp_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self, other}, result, self, outputSize);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    logaddexp_out_npu_nocheck(self, other, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    logaddexp_out_npu_nocheck(self, other, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::logaddexp(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  logaddexp_out_npu_nocheck(self, other, result);
  return result;
}

}  // namespace native
}  // namespace at_npu