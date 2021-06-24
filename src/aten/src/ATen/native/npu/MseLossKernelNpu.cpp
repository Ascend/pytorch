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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& mse_loss_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  if (self.numel()==0 || target.numel()==0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = result.to(at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }

  auto unified_result = OpPreparation::binary_op_check(result, self, target, true);
  string reductionStr;
  if (reduction == Reduction::None) {
    reductionStr = "none";
  } else if (reduction == Reduction::Mean) {
    reductionStr = "mean";
  } else if (reduction == Reduction::Sum) {
    reductionStr = "sum";
  }
  
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
    Tensor& result,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  IntArrayRef outputSize;
  if (reduction == Reduction::None) {
    outputSize = input_same_output_size(self);
  }  
  OpPreparation::CheckOut(
    {self, target}, 
    result, 
    CalcuOpUtil::get_tensor_npu_format(self), 
    self.scalar_type(), 
    outputSize);
  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}

Tensor mse_loss_npu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  // calculate the output size
  IntArrayRef outputSize;
  if (reduction == Reduction::None) {
    outputSize = input_same_output_size(self);
  }

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  mse_loss_out_npu_nocheck(result, self, target, reduction);

  return result;
}

} // namespace native
} // namespace at