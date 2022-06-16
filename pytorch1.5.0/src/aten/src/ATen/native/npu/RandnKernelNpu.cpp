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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/npu/NPUGenerator.h"
#include "ATen/Utils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& randn_out_npu_nocheck(Tensor& result, IntArrayRef size, int64_t seed, int64_t seed2) {
  OpCommand cmd;
  cmd.Name("RandomStandardNormal")
    .Input(size)
    .Output(result)
    .Attr("dtype", result.scalar_type())
    .Attr("seed", seed)
    .Attr("seed2", seed2)
    .Run();
  return result;
}

Tensor& randn_out_npu(Tensor& result, IntArrayRef size) {
  const auto gen = at::npu::detail::getDefaultNPUGenerator();
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t seed2 = pair.second;
  OpPreparation::CheckOut(
      {},
      result,
      result,
      size);
  randn_out_npu_nocheck(result, size, seed, seed2);
  return result;
}

Tensor& randn_out_npu(Tensor& result, IntArrayRef size, Generator* gen_) {
  auto gen = get_generator_or_default<NPUGenerator>(gen_, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t seed2 = pair.second;
  OpPreparation::CheckOut(
      {},
      result,
      result,
      size);
  randn_out_npu_nocheck(result, size, seed, seed2);
  return result;
}

Tensor randn_npu(IntArrayRef size, const TensorOptions& options) {
  const auto gen = at::npu::detail::getDefaultNPUGenerator();
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t seed2 = pair.second;
  Tensor result = OpPreparation::ApplyTensorWithFormat(size, options, ACL_FORMAT_ND);
  randn_out_npu_nocheck(result, size, seed, seed2);
  return result;
}

Tensor randn_npu(IntArrayRef size, Generator* gen_, const TensorOptions& options) {
  auto gen = get_generator_or_default<NPUGenerator>(gen_, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t seed2 = pair.second;
  Tensor result = OpPreparation::ApplyTensorWithFormat(size, options, ACL_FORMAT_ND);
  randn_out_npu_nocheck(result, size, seed, seed2);
  return result;
}

Tensor randn_npu(IntArrayRef size, optional<DimnameList> names, const TensorOptions& options) {
  return randn_npu(size, options);
}

Tensor randn_npu(IntArrayRef size, Generator* gen_, optional<DimnameList> names, const TensorOptions& options) {
  return randn_npu(size, gen_, options);
}

} // namespace native
} // namespace at