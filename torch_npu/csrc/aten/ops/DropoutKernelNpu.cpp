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

#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
at::Tensor dropout_do_mask(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Scalar prob) {
  OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(self)
      .Input(mask)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();

  return result;
}

std::tuple<at::Tensor, at::Tensor> dropout_do_mask_npu(
    const at::Tensor& self,
    const at::Tensor& mask,
    double p) {
  at::Scalar prob = at::Scalar(1. - p);
  at::Tensor result = OpPreparation::ApplyTensor(self);
  OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(self)
      .Input(mask)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();

  return std::tie(result, mask);
}

at::Tensor dropout_gen_mask(const at::Tensor& self, at::Scalar prob) {
  bool isFuzzyCompile = env::CheckFuzzyEnable();
  int64_t numels;
  auto desc_ = torch_npu::NPUBridge::GetNpuStorageImpl(self)->get_npu_desc();
  numels = isFuzzyCompile ? c10::multiply_integers(desc_.storage_sizes_) : self.numel();

  uint32_t length = (numels + 128 - 1) / 128 * 128;
  at::Tensor mask = OpPreparation::ApplyTensorWithFormat(
      {length / 8},
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);

  at::IntArrayRef selfShape = isFuzzyCompile ? desc_.storage_sizes_ : self.sizes();

  OpCommand cmd;
  // DropOutGenMask use seed and seed1 to generator a seed, like this:
  //  seed1   seed
  // 127~64   63~0
  // so, we set seed1 = 0 to ensure the seed which user set is equal to the seed
  // used by the operator DropOutGenMask
  const auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto pair = at::check_generator<NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  at::SmallVector<int64_t, N> offsetList = {0, offset};
  const int64_t seed1 = 0;
  cmd.Name("StatelessDropOutGenMask")
      .Input(selfShape)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Input(at::Scalar(seed), at::ScalarType::Int)
      .Input(at::Scalar(seed1), at::ScalarType::Int)
      .Input(offsetList, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(mask)
      .Run();
  return mask;
}

at::Tensor NPUNativeFunctions::npu_dropout_gen_mask(
    at::IntArrayRef size, double p,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                          .device(device_opt)
                                          .layout(layout_opt)
                                          .pinned_memory(pin_memory_opt);

  at::Scalar prob = at::Scalar(1. - p);
  int64_t numels = c10::multiply_integers(size);

  uint32_t length = (numels + 128 - 1) / 128 * 128;
  at::Tensor mask = OpPreparation::ApplyTensorWithFormat(at::IntArrayRef{length / 8}, options.dtype(at::kByte),
                                                         ACL_FORMAT_ND);

  OpCommand cmd;
  // If either seed or seed1 are set to be non-zero, the random number generator
  // is seeded by the given seed. Otherwise, it is seeded by a random seed.
  const auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto pair = at::check_generator<NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  at::SmallVector<int64_t, N> offsetList = {0, offset};
  const int64_t seed1 = 0;
  cmd.Name("StatelessDropOutGenMask")
      .Input(size)
      .Input(prob, c10::typeMetaToScalarType(options.dtype()), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Input(at::Scalar(seed), at::ScalarType::Int)
      .Input(at::Scalar(seed1), at::ScalarType::Int)
      .Input(offsetList, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(mask)
      .Run();
  return mask;
}

std::tuple<at::Tensor, at::Tensor> dropout_v1_npu_impl(
    at::Tensor result,
    const at::Tensor& self,
    double p) {
  at::Tensor selfCp = NpuUtils::format_contiguous(self);
  TORCH_CHECK(
      p >= 0 && p <= 1,
      "dropout probability has to be between 0 and 1, but got ",
      p);
  TORCH_CHECK(
      at::isFloatingType(selfCp.scalar_type()),
      "dropout only supports floating-point dtypes");

  double retain = 1. - p;
  at::Scalar prob = at::Scalar(retain);
  at::Tensor mask;
  auto original_stream = c10_npu::getCurrentNPUStream();
  {
    // During the life cycle of this raii instance, the calcu stream is set as the
    // secondary stream, and tasks are distributed to the secondary stream. At the
    // same time, according to the one-stream-one-pool principle, memory is also
    // alloced from the pool of the secondary stream.
    c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
    mask = dropout_gen_mask(selfCp, prob);
  }
  // When tasks on multiple streams read and write the same block of memory,
  // recordStream needs to be called to ensure the correctness of memory reuse.
  c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);
  dropout_do_mask(result, selfCp, mask, prob);

  return std::tie(result, mask);
}

at::Tensor NPUNativeFunctions::npu_dropout_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale) {
  TORCH_CHECK(
      at::isFloatingType(grad_output.scalar_type()),
      "dropoutbackward only supports floating-point dtypes");
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype");
  double retain =  1. - scale;
  at::Tensor result = OpPreparation::ApplyTensor(grad_output);

  OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(grad_output)
      .Input(mask)
      .Input(at::Scalar(retain), grad_output.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();

  return result;
}

std::tuple<at::Tensor, at::Tensor> _dropout_npu_com(
    const at::Tensor& self,
    double p) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  return dropout_v1_npu_impl(result, self, p);
}

class NPUdropoutFunction: public torch::autograd::Function<NPUdropoutFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& self,
    double p) {
    ctx->saved_data["p"] = p;
    at::AutoNonVariableTypeMode g;
    auto result = _dropout_npu_com(self, p);
    auto result1 = std::get<1>(result);
    ctx->save_for_backward({result1});
    tensor_list result_list = {std::get<0>(result), result1};
    return result_list;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto p = ctx->saved_data["p"].toDouble();
    auto saved = ctx->get_saved_variables();
    auto mask = saved[0];
    at::Tensor result = NPUNativeFunctions::npu_dropout_backward(grad_outputs[0], mask, p);
    tensor_list output = {result, at::Tensor()};
    return output;
  }
};

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_npu_dropout(
    const at::Tensor& self,
    double p) {
    auto result = NPUdropoutFunction::apply(self, p);
    std::tuple<at::Tensor, at::Tensor> output(result[0], result[1]);
  return output;
}

class NPUDropoutDoMaskFunction: public torch::autograd::Function<NPUDropoutDoMaskFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& self,
    const at::Tensor& mask,
    double p) {
    ctx->saved_data["p"] = p;
    at::AutoNonVariableTypeMode g;
    auto result = dropout_do_mask_npu(self, mask, p);
    auto result1 = std::get<1>(result);
    ctx->save_for_backward({result1});
    tensor_list result_list = {std::get<0>(result), result1};
    return result_list;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto p = ctx->saved_data["p"].toDouble();
    auto saved = ctx->get_saved_variables();
    auto mask = saved[0];
    at::Tensor result = NPUNativeFunctions::npu_dropout_backward(grad_outputs[0], mask, p);
    tensor_list output = {result, at::Tensor(), at::Tensor()};
    return output;
  }
};

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_dropout_do_mask(
    const at::Tensor& self,
    const at::Tensor& mask,
    double p) {
    auto result = NPUDropoutDoMaskFunction::apply(self, mask, p);
    std::tuple<at::Tensor, at::Tensor> output(result[0], result[1]);
  return output;
}

at::Tensor NPUNativeFunctions::dropout(const at::Tensor& self, double p, bool train) {
  if (p == 0 || !train || self.numel() == 0) {
    return self;
  }
  if (p == 1) {
    return self.mul(at::zeros(self.sizes(), self.options()));
  }
  at::Tensor result = std::get<0>(NPUNativeFunctions::_npu_dropout(self, p));
  return result;
}

} // namespace native
} // namespace at_npu
