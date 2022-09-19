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

Tensor& randperm_out_nocheck(Tensor& result, int64_t n, Generator* gen_) {
  auto gen = get_generator_or_default<NPUGenerator>(gen_, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  const int64_t layout = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandperm")
      .Input(at::Scalar(n), at::kLong)
      .Input(at::Scalar(seed), at::kLong)
      .Input(at::Scalar(offset), at::kLong)
      .Output(result)
      .Attr("layout", layout)
      .Attr("dtype", result.scalar_type())
      .Run();
  return result;
}

Tensor randperm_npu(int64_t n, const TensorOptions& options) {
  return native::randperm(n, nullptr, options);
}

Tensor randperm_npu(
    int64_t n,
    Generator* generator,
    const TensorOptions& options) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  Tensor result = OpPreparation::ApplyTensorWithFormat({n}, options, ACL_FORMAT_ND);
  return at::randperm_out(result, n, generator);
}

Tensor& randperm_out_npu(Tensor& result, int64_t n) {
  return at::randperm_out(result, n, nullptr);
}

Tensor& randperm_out_npu(Tensor& result, int64_t n, Generator* generator) {
  OpPreparation::CheckOut({}, result, result, {n});
  randperm_out_nocheck(result, n, generator);
  return result;
}

} // namespace native
} // namespace at
