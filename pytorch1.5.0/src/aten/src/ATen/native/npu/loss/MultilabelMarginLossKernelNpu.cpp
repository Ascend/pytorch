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

std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out_npu(
    Tensor& output,
    Tensor& is_target,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {

  string reductionStr = CalcuOpUtil::get_reduction_str(reduction);
  OpCommand cmd;
  cmd.Name("MultilabelMarginLoss")
    .Input(self)
    .Input(target)
    .Output(output)
    .Output(is_target)
    .Attr("reduction", reductionStr)
    .Run();

  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_npu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {

  SmallVector<int64_t, SIZE> outputSize;
  const auto ndims = self.dim();
  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = ndims == 0 ? 1 : self.size(0);
  } else {
    nframe = self.size(0);
    dim = self.size(1);
  }

  if (reduction == Reduction::None) {
    outputSize = {nframe};
  }

  auto output = OpPreparation::ApplyTensor(self, outputSize);
  auto is_target = OpPreparation::ApplyTensor(target);

  multilabel_margin_loss_forward_out_npu(
      output, is_target, self, target, reduction);
  return std::make_tuple(output, is_target);
}

} // namespace native
} // namespace at
