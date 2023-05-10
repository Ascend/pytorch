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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

at::Tensor dropout_genmask(const at::Tensor& self, at::Scalar prob){
  uint32_t length = (self.numel() + 128 - 1) / 128 * 128;
  at::Tensor mask = OpPreparation::ApplyTensorWithFormat(
      {length},
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);
  at::IntArrayRef selfShape = self.sizes();

  int64_t seed = 2;
  int64_t seed2 = 0;
  OpCommand cmd;
  cmd.Name("DropOutGenMaskV3")
      .Input(selfShape)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(mask)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();
  return mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dropout_with_add_softmax_forward(
    const at::Tensor& self,
    const at::Tensor& x1,
    at::Scalar alpha,
    double p,
    int64_t dim) {
  auto self_type = self.scalar_type();
  auto x1_type = x1.scalar_type();

  at::Tensor self_cp = (self_type == at::kHalf) ? self : NPUNativeFunctions::npu_dtype_cast(self, at::kHalf);
  at::Tensor x1_cp = (x1_type == at::kHalf) ? x1 : NPUNativeFunctions::npu_dtype_cast(x1, at::kHalf);

  at::Tensor result_softmax = OpPreparation::ApplyTensor(x1_cp);
  at::Tensor result_dropout = OpPreparation::ApplyTensor(self_cp);
  c10::SmallVector<int64_t, N> dimList = {dim};
  double retain = 1. - p;
  at::Scalar prob = at::Scalar(retain);
  at::Tensor mask;
  auto original_stream = c10_npu::getCurrentNPUStream();
  {
    c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
    mask = dropout_genmask(x1_cp, prob);
  }
  c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);

  OpCommand cmd;
  cmd.Name("AxpyWithSoftmaxAndDropOutDoMask")
     .Input(x1_cp)
     .Input(self_cp)
     .Input(mask)
     .Output(result_softmax)
     .Output(result_dropout)
     .Attr("alpha", alpha)
     .Attr("input_keep_prob", prob)
     .Attr("axis", dimList)
     .Run();
  if (x1_type != at::kHalf) {
    result_softmax = NPUNativeFunctions::npu_dtype_cast(result_softmax, x1_type);
    result_dropout = NPUNativeFunctions::npu_dtype_cast(result_dropout, x1_type);
  }
  return std::tie(mask, result_softmax, result_dropout);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_dropout_with_add_softmax_backward(
    const at::Tensor& grad_out,
    const at::Tensor& mask,
    const at::Tensor& softmax_out,
    const at::Scalar& alpha,
    double p,
    int64_t dim) {
  auto softmax_dtype = softmax_out.scalar_type();
  at::Tensor grad_out_cp = (grad_out.scalar_type() == at::kHalf) ?
      grad_out : NPUNativeFunctions::npu_dtype_cast(grad_out, at::kHalf);
  at::Tensor softmax_out_cp = (softmax_dtype == at::kHalf) ?
      softmax_out : NPUNativeFunctions::npu_dtype_cast(softmax_out, at::kHalf);

  at::Tensor result = OpPreparation::ApplyTensor(softmax_out_cp);
  c10::SmallVector<int64_t, N> dimList = {dim};
  double retain = 1. - p;
  at::Scalar prob = at::Scalar(retain);

  OpCommand cmd;
  cmd.Name("DropoutWithMulsAndSoftmaxGrad")
     .Input(grad_out_cp)
     .Input(mask)
     .Input(softmax_out_cp)
     .Output(result)
     .Attr("alpha", alpha)
     .Attr("input_keep_prob", prob)
     .Attr("axes", dimList)
     .Run();
  if (softmax_dtype != at::kHalf) {
    result = NPUNativeFunctions::npu_dtype_cast(result, softmax_dtype);
  }
  return std::tie(result, grad_out);
}

class NPUdropoutwasFunction: public torch::autograd::Function<NPUdropoutwasFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& self,
    const at::Tensor& x1,
    at::Scalar alpha,
    double p,
    int64_t dim) {
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["p"] = p;
    ctx->saved_data["dim"] = dim;
    at::AutoNonVariableTypeMode g;
    auto result = npu_dropout_with_add_softmax_forward(self, x1, alpha, p, dim);
    auto result0 = std::get<0>(result);
    auto result1 = std::get<1>(result);
    ctx->save_for_backward({result0, result1});
    tensor_list result_list = {result0, result1, std::get<2>(result)};
    return result_list;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto alpha = ctx->saved_data["alpha"].toScalar();
    auto p = ctx->saved_data["p"].toDouble();
    auto dim = ctx->saved_data["dim"].toInt();
    auto saved = ctx->get_saved_variables();
    auto result0 = saved[0];
    auto result1 = saved[1];
    auto result = NPUNativeFunctions::npu_dropout_with_add_softmax_backward(
        grad_outputs[2],
        result0,
        result1,
        alpha,
        p,
        dim);
    tensor_list output = {std::get<0>(result),
                          std::get<1>(result),
                          at::Tensor(),
                          at::Tensor(),
                          at::Tensor()};
    return output;
  }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_dropout_with_add_softmax(
    const at::Tensor& self,
    const at::Tensor& x1,
    const at::Scalar& alpha,
    double p,
    int64_t dim){
  auto result = NPUdropoutwasFunction::apply(self, x1, alpha, p, dim);
  return std::tie(result[0], result[1], result[2]);
}

} // namespace native
} // namespace at_npu
