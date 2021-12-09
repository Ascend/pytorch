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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& soft_margin_loss_out_npu(const Tensor& self, const Tensor& target, int64_t reduction, Tensor& result) {
  Tensor target_broadcast = target;
  if(target.sizes() != self.sizes()) {
    target_broadcast = at::npu_broadcast(target, self.sizes());
  }

  OpPreparation::CheckMemory({self, target}, {result});
  string reductionStr(CalcuOpUtil::get_reduction_str(reduction));
  OpCommand cmd;
  cmd.Name("SoftMarginLoss")
      .Input(self)
      .Input(target_broadcast)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();
  return result;
}

Tensor soft_margin_loss_npu(const Tensor& self, const Tensor& target, int64_t reduction) {
  auto outputSize = soft_margin_loss_npu_output_size(
      self,
      target,
      reduction);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  soft_margin_loss_out_npu(self, target, reduction, result);
  if (reduction == Reduction::None) {
    return result;
  } else {
    return result.reshape({});
  }
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("soft_margin_loss", TORCH_FN(soft_margin_loss_npu));
  m.impl("soft_margin_loss.out", TORCH_FN(soft_margin_loss_out_npu));
}
} // namespace native
} // namespace at
