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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

namespace at_npu {
namespace native {

at::Tensor& multinomial_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen) {
  auto gen_ = at::get_generator_or_default<NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
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

at::Tensor& NPUNativeFunctions::multinomial_out(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen,
    at::Tensor& result) {
  auto input_dim = self.dim();
  TORCH_CHECK(input_dim==1 || input_dim==2, "dim of input tensor only can be 1 or 2.");

  auto outputSize = array_to_small_vector(self.sizes());
  outputSize[input_dim - 1] = num_samples;
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      at::ScalarType::Long,
      outputSize);
  multinomial_out_npu_nocheck(result, self, num_samples, replacement, gen);
  return result;
}

at::Tensor NPUNativeFunctions::multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen) {
  auto dim = self.dim();
  TORCH_CHECK(dim==1 || dim==2, "dim of input tensor only can be 1 or 2.");

  auto shape = array_to_small_vector(self.sizes());
  shape[dim-1] = num_samples;
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      shape, self.options().dtype(at::kLong), CalcuOpUtil::get_tensor_npu_format(self));
  multinomial_out_npu_nocheck(result, self, num_samples, replacement, gen);
  return result;
}
} // namespace native
} // namespace at_npu
