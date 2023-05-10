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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;

at::Tensor binary_cross_entropy_with_logits_impl(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& pos_weight,
    int64_t reduction) {
  at::IntArrayRef outputSize;
  int64_t resultformat = CalcuOpUtil::GetTensorNpuFormat(self);

  if (reduction == at::Reduction::None) {
    outputSize = input_same_output_size(self);
  } else {
    outputSize = at::ArrayRef<int64_t>();
    resultformat = ACL_FORMAT_ND;
  }

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, self.options(), resultformat);
  at::Tensor weightTensor;
  if (weight.defined()) {
    weightTensor = NpuUtils::format_contiguous(weight);
    weightTensor = (weight.scalar_type() != self.scalar_type()) ? NPUNativeFunctions::npu_dtype_cast(weightTensor,
        self.scalar_type()) : weightTensor;
  } else {
    weightTensor = at::ones(self.sizes(), self.options());
  }

  at::Tensor posWeightTensor;
  if (pos_weight.defined()) {
    posWeightTensor = NpuUtils::format_contiguous(pos_weight);
    posWeightTensor = (posWeightTensor.scalar_type() != self.scalar_type()) ? NPUNativeFunctions::npu_dtype_cast(posWeightTensor,
        self.scalar_type()) : posWeightTensor;
  } else {
    posWeightTensor = at::ones(self.sizes(), self.options());
  }

  string reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpCommand cmd;
  cmd.Name("SigmoidCrossEntropyWithLogitsV2")
      .Input(self.to(target.dtype()))
      .Input(target)
      .Input(weightTensor)
      .Input(posWeightTensor)
      .Output(result)
      .Attr("reduction", reductionStr)
      .Run();

  return result;
}

at::Tensor binary_cross_entropy_with_logits_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& pos_weight,
    int64_t reduction) {
  at::Tensor gradInput = OpPreparation::ApplyTensor(self);
  at::Tensor weightTensor;
  if (weight.defined()) {
    weightTensor = NpuUtils::format_contiguous(weight);
    weightTensor = (weightTensor.scalar_type() != self.scalar_type()) ?
        NPUNativeFunctions::npu_dtype_cast(weightTensor, self.scalar_type()) : weightTensor;
  } else {
    weightTensor = at::ones(self.sizes(), self.options());
  }

  at::Tensor posWeightTensor;
  if (pos_weight.defined()) {
    posWeightTensor = NpuUtils::format_contiguous(pos_weight);
    posWeightTensor = (posWeightTensor.scalar_type() != self.scalar_type()) ?
        NPUNativeFunctions::npu_dtype_cast(posWeightTensor, self.scalar_type()) : posWeightTensor;
  } else {
    posWeightTensor = at::ones(self.sizes(), self.options());
  }

  at::Tensor doutTensor = NPUNativeFunctions::npu_broadcast(grad_output, self.sizes());
  std::string reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpCommand cmd;
  cmd.Name("SigmoidCrossEntropyWithLogitsGradV2")
      .Input(self)
      .Input(target)
      .Input(doutTensor)
      .Input(weightTensor)
      .Input(posWeightTensor)
      .Output(gradInput)
      .Attr("reduction", reductionStr)
      .Run();

  return gradInput;
}

class NPUBinaryCrossEntropyWithLogitsFunction :
    public torch::autograd::Function<NPUBinaryCrossEntropyWithLogitsFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction) {
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
    const at::Tensor& pos_weight = c10::value_or_else(pos_weight_opt, [] {return at::Tensor();});
    at::AutoNonVariableTypeMode g;
    at::Tensor result = binary_cross_entropy_with_logits_impl(self, target, weight, pos_weight, reduction);
    ctx->save_for_backward({self, target, weight, pos_weight});
    ctx->saved_data["reduction"] = reduction;
    return result;
  }

  static tensor_list backward(AutogradContext *ctx,
      tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto reduction = ctx->saved_data["reduction"].toInt();
    auto self_grad = binary_cross_entropy_with_logits_backward(grad_outputs[0], saved[0], saved[1],saved[2], saved[3], reduction);
    tensor_list output = {self_grad, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::binary_cross_entropy_with_logits(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction) {
    return NPUBinaryCrossEntropyWithLogitsFunction::apply(self, target, weight_opt, pos_weight_opt, reduction);
}

} // namespace native
} // namespace at_npu
