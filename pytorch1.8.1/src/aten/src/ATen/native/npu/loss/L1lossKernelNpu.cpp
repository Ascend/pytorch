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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& l1_loss_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& target,
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

Tensor& l1_loss_out_npu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& result) {
  OpPreparation::CheckOut(
      {self, target},
      result,
      self);
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  return l1_loss_out_npu_nocheck(result, self, target, reductionStr);
}

Tensor l1_loss_npu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IntArrayRef outputSize;
  if (reduction == Reduction::None) {
    outputSize = input_same_output_size(self);
  }
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);

  l1_loss_out_npu_nocheck(result, self, target, reductionStr);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("l1_loss", TORCH_FN(l1_loss_npu));
  m.impl("l1_loss.out", TORCH_FN(l1_loss_out_npu));
}
} // namespace native
} // namespace at
