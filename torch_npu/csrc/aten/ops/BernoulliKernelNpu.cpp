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

#include <limits.h>
#include <ATen/NamedTensorUtils.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

namespace at_npu {
namespace native {

at::Tensor& bernoulli_npu_nocheck(at::Tensor& y, at::Tensor& shape, double prob, int64_t seed, int64_t offset) {
  shape = NPUNativeFunctions::npu_dtype_cast(shape, at::ScalarType::Long);
  OpCommand cmd;
  cmd.Name("StatelessBernoulli")
    .Input(shape)
    .Input(at::Scalar(prob), at::ScalarType::Float)
    .Input(at::Scalar(seed), at::ScalarType::Long)
    .Input(at::Scalar(offset), at::ScalarType::Long)
    .Output(y)
    .Run();
  return y;
}

at::Tensor& bernoulli_npu_nocheck(at::Tensor& y, at::Tensor& shape, const at::Tensor& prob, int64_t seed, int64_t offset) {
  shape = NPUNativeFunctions::npu_dtype_cast(shape, at::ScalarType::Long);
  OpCommand cmd;
  cmd.Name("StatelessBernoulli")
    .Input(shape)
    .Input(prob)
    .Input(at::Scalar(seed), at::ScalarType::Long)
    .Input(at::Scalar(offset), at::ScalarType::Long)
    .Output(y)
    .Run();
  return y;
}

at::Tensor& NPUNativeFunctions::bernoulli_(at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  auto gen_ = at::get_generator_or_default<NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  at::ScalarType selfType = self.scalar_type();
  at::Tensor selfFp32 = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfFp32 = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    at::Tensor result = bernoulli_npu_nocheck(contiguousSelf, contiguousSelf, p, seed, offset);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bernoulli_npu_nocheck(selfFp32, selfFp32, p, seed, offset);
    self.copy_(selfFp32);
  }

  if(self.scalar_type() != selfType){
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Half);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::bernoulli_(at::Tensor& self, const at::Tensor& p, c10::optional<at::Generator> gen) {
  auto gen_ = at::get_generator_or_default<NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  at::ScalarType selfType = self.scalar_type();
  at::Tensor selfFp32 = self;
  at::Tensor pFp32 = OpPreparation::CastBackToOriFormat(p);
  if (self.scalar_type() == at::ScalarType::Half) {
    selfFp32 = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    pFp32 = NPUNativeFunctions::npu_dtype_cast(p, at::ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    at::Tensor result = bernoulli_npu_nocheck(contiguousSelf, contiguousSelf, pFp32, seed, offset);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bernoulli_npu_nocheck(selfFp32, selfFp32, pFp32, seed, offset);
    self.copy_(selfFp32);
  }

  if(self.scalar_type() != selfType){
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Half);
  }
  return self;
}

at::Tensor NPUNativeFunctions::bernoulli(const at::Tensor& self, c10::optional<at::Generator> gen) {
  at::Tensor selfCopy = OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options(), ACL_FORMAT_ND);
  selfCopy.copy_(self);
  return NPUNativeFunctions::bernoulli_(selfCopy, self, gen);
}

at::Tensor NPUNativeFunctions::bernoulli(const at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT).bernoulli_(p, gen);
}

at::Tensor& NPUNativeFunctions::bernoulli_out(const at::Tensor& self, c10::optional<at::Generator> gen, at::Tensor& result) {
  result.resize_(self.sizes()).bernoulli_(self, gen);
  at::namedinference::propagate_names(result, self);
  return result;
}
} // namespace native
} // namespace at_npu
