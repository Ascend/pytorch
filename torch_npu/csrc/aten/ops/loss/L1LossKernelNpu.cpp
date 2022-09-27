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

at::Tensor& l1_loss_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    string reductionStr) {
  OpCommand cmd;
  cmd.Name("LpLoss")
      .Input(self)
      .Input(target)
      .Attr("reduction", reductionStr)
      .Attr("p", (int64_t)1)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::l1_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self, target},
      result,
      self);
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  return l1_loss_out_npu_nocheck(result, self, target, reductionStr);
}

at::Tensor NPUNativeFunctions::l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  }
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);

  l1_loss_out_npu_nocheck(result, self, target, reductionStr);
  return result;
}

} // namespace native
} // namespace at_npu
