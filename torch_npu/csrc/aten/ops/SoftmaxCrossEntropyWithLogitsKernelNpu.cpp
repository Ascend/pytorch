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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include <torch/csrc/autograd/custom_function.h>

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

tuple<at::Tensor, at::Tensor> softmax_cross_entropy_with_logits_npu_nocheck(
    at::Tensor& result,
    at::Tensor& backprop,
    const at::Tensor& self,
    const at::Tensor& labels) {
  OpCommand cmd;
  cmd.Name("SoftmaxCrossEntropyWithLogits")
    .Input(self)
    .Input(labels)
    .Output(result)
    .Output(backprop)
    .Run();

  return std::make_tuple(result, backprop);
}

tuple<at::Tensor, at::Tensor> softmax_cross_entropy_with_logits_impl_npu(
    const at::Tensor& self,
    const at::Tensor& labels) {
  auto outputSizes =
      softmax_cross_entropy_with_logits_impl_npu_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(self, std::get<0>(outputSizes));
  at::Tensor backprop = OpPreparation::ApplyTensor(self, std::get<1>(outputSizes));

  softmax_cross_entropy_with_logits_npu_nocheck(result, backprop, self, labels);

  return std::make_tuple(result, backprop);
}

at::Tensor softmax_cross_entropy_with_logits_npu(
    const at::Tensor& self,
    const at::Tensor& labels) {
  TORCH_CHECK(self.device().type() == at_npu::key::NativeDeviceType);
  return std::get<0>(softmax_cross_entropy_with_logits_impl_npu(self, labels));
}

at::Tensor NPUNativeFunctions::npu_softmax_cross_entropy_with_logits_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& labels) {
      at::Tensor result1 = std::get<1>(softmax_cross_entropy_with_logits_impl_npu(self, labels));
      return result1 * grad.unsqueeze(-1);
}

class NPUSoftmaxCrossEntropyWithLogitsFunction: public torch::autograd::Function<NPUSoftmaxCrossEntropyWithLogitsFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self,
    const at::Tensor& labels) {
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self, labels});
    return softmax_cross_entropy_with_logits_npu(self, labels);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];
    auto labels = saved[1];

    at::Tensor result = NPUNativeFunctions::npu_softmax_cross_entropy_with_logits_backward(grad_outputs[0],
        self,
        labels);
    tensor_list output = {result, at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_softmax_cross_entropy_with_logits(const at::Tensor& self,
    const at::Tensor& labels) {
    return NPUSoftmaxCrossEntropyWithLogitsFunction::apply(self, labels);
}

} // namespace native
} // namespace at_npu