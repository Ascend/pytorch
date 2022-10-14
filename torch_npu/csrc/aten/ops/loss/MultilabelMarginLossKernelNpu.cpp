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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::multilabel_margin_loss_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& output,
    at::Tensor& is_target) {

  OpPreparation::CheckMemory({self, target}, {output, is_target});
  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  OpCommand cmd;
  cmd.Name("MultilabelMarginLoss")
    .Input(self)
    .Input(target)
    .Output(output)
    .Output(is_target)
    .Attr("reduction", reductionStr)
    .Run();
  return std::tuple<at::Tensor&, at::Tensor&>(output, is_target);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::multilabel_margin_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  c10::SmallVector<int64_t, SIZE> outputSize;
  int64_t nframe;
  if (self.dim() <= 1) {
    nframe = 1;
  } else {
    nframe = self.size(0);
  }
  if (reduction == at::Reduction::None) {
    outputSize = {nframe};
  }
  auto output = OpPreparation::ApplyTensor(self, outputSize);
  auto is_target = OpPreparation::ApplyTensor(target);

  NPUNativeFunctions::multilabel_margin_loss_forward_out(
      self, target, reduction, output, is_target);
  return std::make_tuple(output, is_target);
}

} // namespace native
} // namespace at_npu
