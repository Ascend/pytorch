// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

namespace at_npu {
namespace native {

at::Tensor& multinomial_op_api(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen) {
  auto gen_ = at::get_generator_or_default<NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  EXEC_NPU_CMD(aclnnMultinomial, self, num_samples, replacement, seed, offset, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::multinomial_out(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMultinomial, NPUNativeFunctions::multinomial_out(self, num_samples, replacement, gen, result));
  auto input_dim = self.dim();
  auto output_size = array_to_small_vector(self.sizes());
  output_size[input_dim - 1] = num_samples;
  OpPreparation::CheckOut(
      {self},
      result,
      at::ScalarType::Long,
      output_size);
  multinomial_op_api(result, self, num_samples, replacement, gen);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnMultinomial, NPUNativeFunctions::multinomial(self, num_samples, replacement, gen));
  auto dim = self.dim();
  auto shape = array_to_small_vector(self.sizes());
  shape[dim-1] = num_samples;
  at::Tensor result = OpPreparation::ApplyTensor(
      shape, self.options().dtype(at::kLong), self);
  multinomial_op_api(result, self, num_samples, replacement, gen);
  return result;
}
} // namespace native
} // namespace at_npu
