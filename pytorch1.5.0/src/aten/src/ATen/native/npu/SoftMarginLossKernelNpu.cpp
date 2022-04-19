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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& soft_margin_loss_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& target, int64_t reduction) {
// constructs the input and output NPUTensorDesc
  Tensor target_broadcast = target;
  if(target.sizes() != self.sizes()) {
    target_broadcast = broadcast_npu(target, self.sizes());
  }
  std::string reductionStr = NpuUtils::get_reduction_str(reduction);
  OpCommand cmd;
  cmd.Name("SoftMarginLoss")
      .Input(self)
      .Input(target_broadcast)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();
  return result;
}

Tensor& soft_margin_loss_out_npu(Tensor& result, const Tensor& self, const Tensor& target, int64_t reduction) {
  auto outputSize = soft_margin_loss_npu_output_size(
      self,
      target,
      reduction);
  OpPreparation::CheckOut(
      {self, target},
      result,
      self,
      outputSize);
  if (!NpuUtils::check_match(&result)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(result);
    soft_margin_loss_out_npu_nocheck(contiguousResult, self, target, reduction);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    soft_margin_loss_out_npu_nocheck(result, self, target, reduction);
  }
   return result;
}

Tensor soft_margin_loss_npu(const Tensor& self, const Tensor& target, int64_t reduction) {
// calculate the output size
  auto outputSize = soft_margin_loss_npu_output_size(
      self,
      target,
      reduction);

// construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(
      self, outputSize);

// calculate the output result of the NPU
  soft_margin_loss_out_npu_nocheck(result, self, target, reduction);
  if (reduction == Reduction::None) {
    return result;
  } else {
    return result.reshape({});
  }
}

} // namespace native
} // namespace at
