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
#include "ATen/native/npu/utils/OpTemplate.h"
#include "ATen/npu/NPUGenerator.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& multinomial_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    Generator* gen) {
  auto gen_ = get_generator_or_default<NPUGenerator>(gen, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  OpCommand cmd;
  cmd.Name("MultinomialWithReplacement")
      .Input(self)
      .Input(at::Scalar(seed), at::ScalarType::Long)
      .Input(at::Scalar(offset), at::ScalarType::Long)
      .Output(result)
      .Attr("numsamples", num_samples)
      .Attr("replacement", replacement)
      .Run();
  return result;
}

Tensor& multinomial_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    Generator* gen) {
  auto input_dim = self.dim();
  TORCH_CHECK(input_dim==1 || input_dim==2, "dim of input tensor only can be 1 or 2.");

  auto outputSize = array_to_small_vector(self.sizes());
  outputSize[input_dim - 1] = num_samples;
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      at::kLong,
      outputSize);
  multinomial_out_npu_nocheck(result, self, num_samples, replacement, gen);
  return result;
}

Tensor multinomial_npu(
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    Generator* gen) {
  auto dim = self.dim();
  TORCH_CHECK(dim==1 || dim==2, "dim of input tensor only can be 1 or 2.");

  auto shape = array_to_small_vector(self.sizes());
  shape[dim-1] = num_samples;
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      shape, self.options().dtype(at::kLong), CalcuOpUtil::get_tensor_npu_format(self));
  multinomial_out_npu(result, self, num_samples, replacement, gen);
  return result;
}
} // namespace native
} // namespace at
