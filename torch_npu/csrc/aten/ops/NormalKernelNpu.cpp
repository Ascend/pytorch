// Copyright (c) 2022, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace at_npu {
namespace native {

at::Tensor &normal_out_npu_nocheck(
    at::Tensor& result,
    c10::optional<at::Generator> gen_) {
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  at::SmallVector<int64_t, N> key = {seed}; 
  at::SmallVector<int64_t, N> counter = {0, offset}; 
  const int32_t alg = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandomNormalV2")
      .Input(result.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(key, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(counter, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int)
      .Output(result)
      .Attr("dtype", result.scalar_type())
      .Run();
  return result;
}

at::Tensor &NPUNativeFunctions::normal_out(
    const at::Tensor &mean, 
    double std, 
    c10::optional<at::Generator> generator,
    at::Tensor &result) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);

  OpPreparation::CheckOut({mean}, result, mean);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

at::Tensor &NPUNativeFunctions::normal_out(
    double mean, 
    const at::Tensor &std, 
    c10::optional<at::Generator> generator,
    at::Tensor &result) {
  OpPreparation::CheckOut({std}, result, std);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

at::Tensor &NPUNativeFunctions::normal_out(
    const at::Tensor &mean, 
    const at::Tensor &std,
    c10::optional<at::Generator> generator, 
    at::Tensor &result) {
  at::SmallVector<int64_t, SIZE> outputSize = broadcast_ops_npu_output_size(mean, std);
  OpPreparation::CheckOut({mean, std}, result, mean, outputSize);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

at::Tensor &NPUNativeFunctions::normal_out(
    double mean, 
    double std, 
    at::IntArrayRef size,
    c10::optional<at::Generator> generator, 
    at::Tensor &result) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);
  OpPreparation::CheckOut({}, result, result, size);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

at::Tensor NPUNativeFunctions::normal(
    const at::Tensor &mean, 
    double std,
    c10::optional<at::Generator> generator) {
  at::Tensor result = OpPreparation::ApplyTensor(mean);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);

  return result;
}

at::Tensor NPUNativeFunctions::normal(
    double mean, 
    const at::Tensor &std,
    c10::optional<at::Generator> generator) {
  at::Tensor result = OpPreparation::ApplyTensor(std);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);

  return result;
}

at::Tensor NPUNativeFunctions::normal(
    const at::Tensor &mean,
    const at::Tensor &std,
    c10::optional<at::Generator> generator) {
  at::Tensor result = OpPreparation::ApplyTensor(mean);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);

  return result;
}

at::Tensor NPUNativeFunctions::normal(
    double mean, double std,
    at::IntArrayRef size,
    c10::optional<at::Generator> generator,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                                  .device(device_opt)
                                                  .layout(layout_opt)
                                                  .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(size, option, ACL_FORMAT_ND);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);

  return result;
}

at::Tensor &NPUNativeFunctions::normal_(
    at::Tensor &self, 
    double mean, 
    double std,
    c10::optional<at::Generator> generator) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    NPUNativeFunctions::normal_out(mean, std, contiguousSelf.sizes(), generator, contiguousSelf);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    NPUNativeFunctions::normal_out(mean, std, self.sizes(), generator, self);
  }

  return self;
}

}  // namespace native
}  // namespace at_npu
