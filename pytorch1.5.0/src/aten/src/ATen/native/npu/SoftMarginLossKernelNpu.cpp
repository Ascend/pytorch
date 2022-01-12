// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> soft_margin_loss_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> soft_margin_loss_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> soft_margin_loss_npu_attr(
    int64_t reduction) {
  std::string reductionStr = NpuUtils::get_reduction_str(reduction);

  NPUAttrDesc npuAttrReduction = NPUAttrDesc("reduction", reductionStr);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrReduction};
  return attrs;
}

Tensor& soft_margin_loss_out_npu(Tensor& result, const Tensor& self, const Tensor& target, int64_t reduction) {
// constructs the input and output NPUTensorDesc
  Tensor target_broadcast = target;
  if(target.sizes() != self.sizes()) {
    target_broadcast = broadcast_npu(target, self.sizes());
  }
  auto inputs = soft_margin_loss_npu_input({self, target_broadcast});
  auto outputs = soft_margin_loss_npu_output({result});

// constructs the attr of the NPUAttrDesc
  auto attrs = soft_margin_loss_npu_attr(reduction);

// executing the NPU operator
  CalcuOpUtil::execute_npu_operate("SoftMarginLoss", inputs, outputs, attrs);
  return result;
}

Tensor soft_margin_loss_npu(const Tensor& self, const Tensor& target, int64_t reduction) {
// calculate the output size
  auto outputSize = soft_margin_loss_npu_output_size(
      self,
      target,
      reduction);

// construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

// calculate the output result of the NPU
  soft_margin_loss_out_npu(result, self, target, reduction);
  if (reduction == Reduction::None) {
    return result;
  } else {
    return result.reshape({});
  }
}

} // namespace native
} // namespace at
