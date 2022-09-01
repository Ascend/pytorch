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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "ATen/npu/NPUGenerator.h"
#include "ATen/Utils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& normal_out_npu_nocheck(
    Tensor& result,
    Generator* gen_) {
  auto gen = get_generator_or_default<NPUGenerator>(gen_, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  SmallVector<int64_t, N> seed_list = {seed};
  SmallVector<int64_t, N> offset_list = {0, offset};
  const int32_t alg = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandomNormalV2")
      .Input(result.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(seed_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(offset_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int)
      .Output(result)
      .Attr("dtype", result.scalar_type())
      .Run();
  return result;
}

Tensor& normal_out_npu(
    Tensor& result,
    const Tensor& mean, 
    double std, 
    Generator* generator) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);

  OpPreparation::CheckOut({mean}, result, mean);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor& normal_out_npu(
    Tensor& result,
    double mean, 
    const Tensor& std, 
    Generator* generator) {
  OpPreparation::CheckOut({std}, result, std);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor& normal_out_npu(
    Tensor& result,
    const Tensor& mean, 
    const Tensor& std, 
    Generator* generator) {
  SmallVector<int64_t, SIZE> outputSize = broadcast_ops_npu_output_size(mean, std);
  OpPreparation::CheckOut({mean, std}, result, mean, outputSize);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor& normal_out_npu(
    Tensor& result,
    double mean, 
    double std, 
    IntArrayRef size,
    Generator* generator) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);
  OpPreparation::CheckOut({}, result, result, size);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor normal_npu(
    const Tensor& mean, 
    double std, 
    Generator* generator) {
  Tensor result = OpPreparation::ApplyTensor(mean);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor normal_npu(
    double mean, 
    const Tensor& std, 
    Generator* generator) {
  Tensor result = OpPreparation::ApplyTensor(std);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor normal_npu(
    const Tensor& mean, 
    const Tensor& std, 
    Generator* generator) {
  Tensor result = OpPreparation::ApplyTensor(mean);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor normal_npu(
    double mean, 
    double std, 
    IntArrayRef size,
    Generator* generator,
    const TensorOptions& options) {
  Tensor result = OpPreparation::ApplyTensorWithFormat(size, options, ACL_FORMAT_ND);
  normal_out_npu_nocheck(result, generator);
  result.mul_(std).add_(mean);
  return result;
}

Tensor& normal_npu_(
    Tensor& self,
    double mean,
    double std,
    Generator* generator) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    normal_out_npu(contiguousSelf, mean, std, contiguousSelf.sizes(), generator);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    normal_out_npu(self, mean, std, self.sizes(), generator);
  }

  return self;
}

} // namespace native
} // namespace at
