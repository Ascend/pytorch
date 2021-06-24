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

SmallVector<NPUTensorDesc, N> multilabel_margin_loss_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> multilabel_margin_loss_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> multilabel_margin_loss_npu_attr(int64_t reduction) {
  string reductionStr;
  if (reduction == Reduction::None) {
    reductionStr = "none";
  } else if (reduction == Reduction::Mean) {
    reductionStr = "mean";
  } else if (reduction == Reduction::Sum) {
    reductionStr = "sum";
  }

  NPUAttrDesc npuAttrReduction = NPUAttrDesc("reduction", reductionStr);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrReduction};

  return attrs;
}

std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out_npu(
    Tensor& output,
    Tensor& is_target,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto inputs = multilabel_margin_loss_npu_input({self, target});
  auto outputs = multilabel_margin_loss_npu_output({output, is_target});
  auto attrs = multilabel_margin_loss_npu_attr(reduction);
  CalcuOpUtil::execute_npu_operate("MultilabelMarginLoss", inputs, outputs, attrs);
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

  auto output = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  auto is_target = at::empty_with_format(target.sizes(), target.options(), CalcuOpUtil::get_tensor_npu_format(target));  

  multilabel_margin_loss_forward_out_npu(
      output, is_target, self, target, reduction);
  return std::make_tuple(output, is_target);
}

Tensor& multilabel_margin_loss_backward_npu_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto inputs = multilabel_margin_loss_npu_input({self, grad_output, target, is_target});
  auto outputs = multilabel_margin_loss_npu_output({grad_input});
  auto attrs = multilabel_margin_loss_npu_attr(reduction);
  CalcuOpUtil::execute_npu_operate("MultilabelMarginLossGrad", inputs, outputs, attrs);
  return grad_input;
}

Tensor multilabel_margin_loss_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto grad_input = at::empty_with_format(self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  multilabel_margin_loss_backward_npu_out(
      grad_input, grad_output, self, target, reduction, is_target);
  return grad_input;
}

} // namespace native
} // namespace at
