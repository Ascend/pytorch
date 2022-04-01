// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpTemplate.h"
#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include <ATen/npu/NPUGeneratorImpl.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
at::Tensor dropout_do_mask_with_byte_mask(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Scalar prob) {
  OpCommand cmd;
  cmd.Name("DropOutDoMaskV3")
      .Input(self)
      .Input(mask)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();
  return result;
}

at::Tensor dropout_gen_byte_mask(const at::Tensor& self, at::Scalar prob) {
  at::IntArrayRef selfShape = self.sizes();
  at::Tensor mask = OpPreparation::ApplyTensorWithFormat(
      selfShape,
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);
  OpCommand cmd;
  // If either seed or seed2 are set to be non-zero, the random number generator
  // is seeded by the given seed. Otherwise, it is seeded by a random seed.
  // DropOutGenMaskV3 use seed and seed2 to generator a seed, like this:
  //  seed2   seed
  // 127~64   63~0
  // so, we set seed2 = 0 to ensure the seed which user set is equal to the seed 
  // used by the operator DropOutGenMaskV3
  const auto gen = at::npu::detail::getDefaultNPUGenerator();
  const int64_t seed = static_cast<int64_t>(gen.current_seed());
  const int64_t seed2 = 0;
  cmd.Name("DropOutGenMaskV3")
      .Input(selfShape)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(mask)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();
  return mask;
}

std::tuple<at::Tensor, at::Tensor> dropout_npu_impl(
    at::Tensor result,
    const at::Tensor& self,
    double p) {
  at::Tensor selfCp = NpuUtils::format_contiguous(self);
  TORCH_CHECK(
      p >= 0 && p <= 1,
      "dropout probability has to be between 0 and 1, but got ", p);
  TORCH_CHECK(
      at::isFloatingType(selfCp.scalar_type()),
      "dropout only supports floating-point dtypes");

  double retain = 1. - p;
  at::Scalar prob = at::Scalar(retain);
  at::Tensor mask;
  auto original_stream = c10::npu::getCurrentNPUStream();
  {
    // During the life cycle of this raii instance, the calcu stream is set as the
    // secondary stream, and tasks are distributed to the secondary stream. At the
    // same time, according to the one-stream-one-pool principle, memory is also
    // alloced from the pool of the secondary stream.
    torch_npu::SecondaryStreamGuard guard(c10::npu::getCurrentSecondaryStream());
    mask = dropout_gen_byte_mask(selfCp, prob);
  }
  // When tasks on multiple streams read and write the same block of memory,
  // recordStream needs to be called to ensure the correctness of memory reuse.
  c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);
  dropout_do_mask_with_byte_mask(result, selfCp, mask, prob);

  return std::tie(result, mask);
}

std::tuple<at::Tensor, at::Tensor> _dropout_with_byte_mask_new(
    const at::Tensor& self,
    double p) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  return dropout_npu_impl(result, self, p);
}

std::tuple<at::Tensor, at::Tensor> _dropout_with_byte_mask_inplace_new(
    at::Tensor& self,
    double p) {
   return dropout_npu_impl(self, self, p);
}

at::Tensor NPUNativeFunctions::_dropout_with_byte_mask_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale)  {
  TORCH_CHECK(
      at::isFloatingType(grad_output.scalar_type()),
      "dropoutbackward only supports floating-point dtypes");
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype");
  double retain =  1. - scale;
  at::Tensor result = OpPreparation::ApplyTensor(grad_output);

  OpCommand cmd;
  cmd.Name("DropOutDoMaskV3")
      .Input(grad_output)
      .Input(mask)
      .Input(retain, grad_output.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();

  return result;
}

class NPUdropoutwbmFunction: public torch::autograd::Function<NPUdropoutwbmFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& self,
    double p) {
    ctx->saved_data["p"] = p;
    at::AutoNonVariableTypeMode g;
    auto result = _dropout_with_byte_mask_new(self, p);
    auto result1 = std::get<1>(result);
    ctx->saved_data["output"] = result1;
    tensor_list result_list = {std::get<0>(result), result1};
    return result_list;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto p = ctx->saved_data["p"].toDouble();
    auto mask = ctx->saved_data["output"].toTensor();
    at::Tensor result = NPUNativeFunctions::_dropout_with_byte_mask_backward(grad_outputs[0], mask, p);
    tensor_list output = {result, at::Tensor()};
    return output;
  }
};

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_dropout_with_byte_mask(
    const at::Tensor& self,
    double p) {
    auto result = NPUdropoutwbmFunction::apply(self, p);
    std::tuple<at::Tensor, at::Tensor> output(result[0], result[1]);
    return output;
}

at::Tensor NPUNativeFunctions::dropout_with_byte_mask(const at::Tensor& self, double p, bool train) {
  TORCH_CHECK(
      self.is_npu(),
      "dropout_with_byte_mask only supports device for NPU!");
  if (p == 0 || !train || self.numel() == 0) {
    return self;
  }
  if (p == 1) {
    return self.mul(at::zeros(self.sizes(), self.options()));
  }
  return std::get<0>(NPUNativeFunctions::_dropout_with_byte_mask(self, p));
}


} // namespace native
} // namespace at
