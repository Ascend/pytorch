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

#include "ATen/native/npu/utils/OpAdapter.h"
#include <torch/csrc/autograd/custom_function.h>

namespace at {
namespace native {
using namespace at::native::npu;
using namespace torch::autograd;

tuple<Tensor, Tensor> softmax_cross_entropy_with_logits_npu_nocheck(
    Tensor& result,
    Tensor& backprop,
    const Tensor& self,
    const Tensor& labels) {
  OpCommand cmd;
  cmd.Name("SoftmaxCrossEntropyWithLogits")
    .Input(self)
    .Input(labels)
    .Output(result)
    .Output(backprop)
    .Run();

  return std::make_tuple(result, backprop);
}

tuple<Tensor, Tensor> softmax_cross_entropy_with_logits_impl_npu(
    const Tensor& self,
    const Tensor& labels) {
  auto outputSizes =
      softmax_cross_entropy_with_logits_impl_npu_output_size(self);
  Tensor result = OpPreparation::ApplyTensor(self, std::get<0>(outputSizes));
  Tensor backprop = OpPreparation::ApplyTensor(self, std::get<1>(outputSizes));

  softmax_cross_entropy_with_logits_npu_nocheck(result, backprop, self, labels);

  return std::make_tuple(result, backprop);
}

Tensor softmax_cross_entropy_with_logits_npu(
    const Tensor& self,
    const Tensor& labels) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::NPU);
  return std::get<0>(softmax_cross_entropy_with_logits_impl_npu(self, labels));
}

Tensor softmax_cross_entropy_with_logits_backward_npu(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& labels) {
      Tensor result1 = std::get<1>(softmax_cross_entropy_with_logits_impl_npu(self, labels));
      return result1 * grad.unsqueeze(-1);
}

class NPUSoftmaxCrossEntropyWithLogitsFunction: public torch::autograd::Function<NPUSoftmaxCrossEntropyWithLogitsFunction> {
public:
  static Tensor forward(AutogradContext *ctx,
    const Tensor& self,
    const Tensor& labels) {
    ctx->saved_data["labels"] = labels;
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self});
    return at::native::softmax_cross_entropy_with_logits_npu(self, labels);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto labels = ctx->saved_data["labels"].toTensor();
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];

    Tensor result = at::native::softmax_cross_entropy_with_logits_backward_npu(grad_outputs[0],
        self,
        labels);
    tensor_list output = {result,
        Tensor()};
    return output;
  }
};

Tensor npu_softmax_cross_entropy_with_logits_autograd(const Tensor& self,
    const Tensor& labels) {
    return NPUSoftmaxCrossEntropyWithLogitsFunction::apply(self, labels);
}

TORCH_LIBRARY_IMPL(aten, AutogradNPU, m) {
    m.impl("npu_softmax_cross_entropy_with_logits", npu_softmax_cross_entropy_with_logits_autograd);
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("npu_softmax_cross_entropy_with_logits", TORCH_FN(softmax_cross_entropy_with_logits_npu));
  m.impl("npu_softmax_cross_entropy_with_logits_backward", TORCH_FN(softmax_cross_entropy_with_logits_backward_npu));
}

} // namespace native
} // namespace at