// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

Tensor& smooth_l1_loss_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  // Check the self empty
  if (self.numel()==0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = result.to(at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }

  string reductionStr;
  if (reduction == Reduction::None) {
    reductionStr = "none";
  } else if (reduction == Reduction::Mean) {
    reductionStr = "mean";
  } else if (reduction == Reduction::Sum) {
    reductionStr = "sum";
  }

  OpCommand cmd;
  cmd.Name("SmoothL1LossV2")
    .Input(self)
    .Input(target)
    .Output(result)
    .Attr("reduction", reductionStr)
    .Run();

  return result;
}

Tensor& smooth_l1_loss_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto outputSize = smooth_l1_loss_npu_output_size(self, target, reduction);
  OpPreparation::CheckOut(
    {self, target}, 
    result, 
    CalcuOpUtil::get_tensor_npu_format(self), 
    self.scalar_type(), 
    outputSize);
  smooth_l1_loss_out_npu_nocheck(result, self, target, reduction); 
  return result;  
}

Tensor smooth_l1_loss_npu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  // calculate the output size
  auto outputSize = smooth_l1_loss_npu_output_size(self, target, reduction);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  smooth_l1_loss_out_npu_nocheck(result, self, target, reduction);

  return result;
}

} // namespace native
} // namespace at