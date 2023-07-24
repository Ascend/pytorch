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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::normal_(at::Tensor& self, double mean, double std,
                                             c10::optional<at::Generator> generator) {
  DO_COMPATIBILITY(aclnnInplaceNormal, NPUNativeFunctions::normal_(self, mean, std, generator));
  TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
  OpPreparation::CheckOut({}, self, self, self.sizes());
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float mean_cast = static_cast<float>(mean);
  float rstd_cast = static_cast<float>(std);
  EXEC_NPU_CMD(aclnnInplaceNormal, self, mean_cast, rstd_cast, seed, offset);
  return self;
}

/* TensorTensor */
at::Tensor& NPUNativeOpApiFunctions::normal_out(const at::Tensor &mean, const at::Tensor &std,
                                                c10::optional<at::Generator> generator, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnNormalTensorTensor, NPUNativeFunctions::normal_out(mean, std, generator, result));
  at::SmallVector<int64_t, SIZE> output_size = broadcast_ops_npu_output_size(mean, std);
  OpPreparation::CheckOut({mean, std}, result, result, output_size);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  EXEC_NPU_CMD(aclnnNormalTensorTensor, mean, std, seed, offset, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::normal(const at::Tensor &mean, const at::Tensor &std,
                                           c10::optional<at::Generator> generator) {
  DO_COMPATIBILITY(aclnnNormalTensorTensor, NPUNativeFunctions::normal(mean, std, generator));
  at::SmallVector<int64_t, SIZE> output_size = broadcast_ops_npu_output_size(mean, std);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(mean, output_size);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  EXEC_NPU_CMD(aclnnNormalTensorTensor, mean, std, seed, offset, result);
  return result;
}

/* TensorFloat */
at::Tensor& NPUNativeOpApiFunctions::normal_out(const at::Tensor &mean, double std,
                                                c10::optional<at::Generator> generator, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnNormalTensorFloat, NPUNativeFunctions::normal_out(mean, std, generator, result));
  TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
  OpPreparation::CheckOut({mean}, result, result);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float rstd_cast = static_cast<float>(std);
  EXEC_NPU_CMD(aclnnNormalTensorFloat, mean, rstd_cast, seed, offset, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::normal(const at::Tensor &mean, double std,
                                           c10::optional<at::Generator> generator) {
  DO_COMPATIBILITY(aclnnNormalTensorFloat, NPUNativeFunctions::normal(mean, std, generator));
  TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(mean);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float rstd_cast = static_cast<float>(std);
  EXEC_NPU_CMD(aclnnNormalTensorFloat, mean, rstd_cast, seed, offset, result);
  return result;
}

/* FloatTensor */
at::Tensor& NPUNativeOpApiFunctions::normal_out(double mean, const at::Tensor &std,
                                                c10::optional<at::Generator> generator, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnNormalFloatTensor, NPUNativeFunctions::normal_out(mean, std, generator, result));
  OpPreparation::CheckOut({std}, result, result);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float mean_cast = static_cast<float>(mean);
  EXEC_NPU_CMD(aclnnNormalFloatTensor, mean_cast, std, seed, offset, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::normal(double mean, const at::Tensor &std,
                                           c10::optional<at::Generator> generator) {
  DO_COMPATIBILITY(aclnnNormalFloatTensor, NPUNativeFunctions::normal(mean, std, generator));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(std);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float mean_cast = static_cast<float>(mean);
  EXEC_NPU_CMD(aclnnNormalFloatTensor, mean_cast, std, seed, offset, result);
  return result;
}

/* FloatFloat */
at::Tensor& NPUNativeOpApiFunctions::normal_out(double mean, double std, at::IntArrayRef size,
                                                c10::optional<at::Generator> generator, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnNormalFloatFloat, NPUNativeFunctions::normal_out(mean, std, size, generator, result));
  TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
  OpPreparation::CheckOut({}, result, result, size);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float mean_cast = static_cast<float>(mean);
  float rstd_cast = static_cast<float>(std);
  EXEC_NPU_CMD(aclnnNormalFloatFloat, mean_cast, rstd_cast, seed, offset, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::normal(double mean, double std,
                                           at::IntArrayRef size,
                                           c10::optional<at::Generator> generator,
                                           c10::optional<at::ScalarType> dtype_opt,
                                           c10::optional<c10::Layout> layout_opt,
                                           c10::optional<c10::Device> device_opt,
                                           c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnNormalFloatFloat, NPUNativeFunctions::normal(mean, std, size, generator, dtype_opt, 
                                                                     layout_opt, device_opt, pin_memory_opt));
  c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                                  .device(device_opt)
                                                  .layout(layout_opt)
                                                  .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(size, option);
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float mean_cast = static_cast<float>(mean);
  float rstd_cast = static_cast<float>(std);
  EXEC_NPU_CMD(aclnnNormalFloatFloat, mean_cast, rstd_cast, seed, offset, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
