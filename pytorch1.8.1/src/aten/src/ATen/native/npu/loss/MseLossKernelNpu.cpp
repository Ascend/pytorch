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

Tensor& mse_loss_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  if (self.numel() == 0 || target.numel() == 0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = result.to(at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }
  auto unified_result = OpPreparation::binary_op_check(result, self, target, true);
  string reductionStr(CalcuOpUtil::get_reduction_str(reduction));
  OpCommand cmd;
  cmd.Name("MseLoss")
      .Expect(unified_result)
      .Input(self)
      .Input(target)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();
  return result;
}

Tensor& mse_loss_out_npu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& result) {
  IntArrayRef outputSize;
  if (reduction == Reduction::None) {
    outputSize = input_same_output_size(self);
  }

  OpPreparation::CheckOut(
      {self, target},
      result,
      self,
      outputSize);

  OpPreparation::CheckMemory({self, target}, {result});
  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}

Tensor mse_loss_npu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IntArrayRef outputSize;
  if (reduction == Reduction::None) {
    outputSize = input_same_output_size(self);
  }
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("mse_loss.out", TORCH_FN(mse_loss_out_npu));
  m.impl("mse_loss", TORCH_FN(mse_loss_npu));
}
} // namespace native
} // namespace at