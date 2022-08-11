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
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& mse_loss_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
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

at::Tensor& XLANativeFunctions::mse_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  }

  OpPreparation::CheckOut(
      {self, target},
      result,
      self,
      outputSize);

  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}

at::Tensor XLANativeFunctions::mse_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  }
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}

} // namespace native
} // namespace at_npu