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
#include "c10/npu/NPUCachingAllocator.h"
#include "ATen/npu/NPUGenerator.h"
#include "ATen/Utils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bernoulli_npu_nocheck(Tensor& y, double prob, int64_t seed, int64_t offset) {
  OpCommand cmd;
  cmd.Name("StatelessBernoulli")
    .Input(y.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
    .Input(Scalar(prob), ScalarType::Float)
    .Input(Scalar(seed), ScalarType::Long)
    .Input(Scalar(offset), ScalarType::Long)
    .Output(y)
    .Attr("dtype", y.scalar_type())
    .Run();
  return y;
}

Tensor& bernoulli_npu_nocheck(Tensor& y, const Tensor& prob, int64_t seed, int64_t offset) {
  OpCommand cmd;
  cmd.Name("StatelessBernoulli")
    .Input(y.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
    .Input(prob)
    .Input(Scalar(seed), ScalarType::Long)
    .Input(Scalar(offset), ScalarType::Long)
    .Output(y)
    .Attr("dtype", y.scalar_type())
    .Run();
  return y;
}

Tensor& bernoulli_npu_(Tensor& self, double p, Generator* gen) {
  auto gen_ = get_generator_or_default<NPUGenerator>(gen, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    bernoulli_npu_nocheck(contiguousSelf, p, seed, offset);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    bernoulli_npu_nocheck(self, p, seed, offset);
  }
  return self;
}

Tensor& bernoulli_npu_(Tensor& self, const Tensor& p, Generator* gen) {
  auto gen_ = get_generator_or_default<NPUGenerator>(gen, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  Tensor pOriFormat = OpPreparation::CastBackToOriFormat(p);
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    bernoulli_npu_nocheck(contiguousSelf, pOriFormat, seed, offset);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    bernoulli_npu_nocheck(self, pOriFormat, seed, offset);
  }
  return self;
}

Tensor bernoulli_npu(const Tensor& self, Generator* gen) {
  Tensor selfCopy = OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options(), ACL_FORMAT_ND);
  selfCopy.copy_(self);
  return bernoulli_npu_(selfCopy, self, gen);
}
} // namespace native
} // namespace at
