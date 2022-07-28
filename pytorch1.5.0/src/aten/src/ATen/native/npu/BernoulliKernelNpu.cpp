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
#include "c10/npu/SecondaryStreamGuard.h"
#include "c10/npu/NPUCachingAllocator.h"
#include "ATen/npu/NPUGenerator.h"
#include "ATen/Utils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bernoulli_npu_nocheck(Tensor& y, const Tensor& x, double p, int64_t seed, int64_t offset) {
  auto original_stream = c10::npu::getCurrentNPUStream();
  {
      auto x_ = at::empty_like(x);
      c10::npu::SecondaryStreamGuard guard(c10::npu::getCurrentSecondaryStream());
      OpCommand cmd;
      cmd.Name("Bernoulli")
        .Input(x_)
        .Input(Scalar(p), ScalarType::Float)
        .Output(y)
        .Attr("seed", seed)
        .Attr("offset", offset)
        .Run();
  }
  c10::npu::NPUCachingAllocator::recordStream(y.storage().data_ptr(), original_stream);
  return y;
}

Tensor& bernoulli_npu_nocheck(Tensor& y, const Tensor& x, const Tensor& p, int64_t seed, int64_t offset) {
  OpCommand cmd;
  cmd.Name("Bernoulli")
    .Input(x)
    .Input(p)
    .Output(y)
    .Attr("seed", seed)
    .Attr("offset", offset)
    .Run();
  return y;
}

Tensor& bernoulli_npu_(Tensor& self, double p, Generator* gen) {
  auto gen_ = get_generator_or_default<NPUGenerator>(gen, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  ScalarType selfType = self.scalar_type();
  Tensor selfFp32 = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfFp32 = self.to(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    Tensor result = bernoulli_npu_nocheck(contiguousSelf, contiguousSelf, p, seed, offset);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bernoulli_npu_nocheck(selfFp32, selfFp32, p, seed, offset);
    self.copy_(selfFp32);
  }

  if(self.scalar_type() != selfType){
    self = self.to(ScalarType::Half);
  }
  return self;
}

Tensor& bernoulli_npu_(Tensor& self, const Tensor& p, Generator* gen) {
  auto gen_ = get_generator_or_default<NPUGenerator>(gen, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  ScalarType selfType = self.scalar_type();
  Tensor selfFp32 = self;
  Tensor pFp32 = OpPreparation::CastBackToOriFormat(p);
  if (self.scalar_type() == ScalarType::Half) {
    selfFp32 = self.to(ScalarType::Float);
    pFp32 = p.to(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    Tensor result = bernoulli_npu_nocheck(contiguousSelf, contiguousSelf, pFp32, seed, offset);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bernoulli_npu_nocheck(selfFp32, selfFp32, pFp32, seed, offset);
    self.copy_(selfFp32);
  }

  if(self.scalar_type() != selfType){
    self = self.to(ScalarType::Half);
  }
  return self;
}

Tensor bernoulli_npu(const Tensor& self, Generator* gen) {
  const Tensor p = self;
  Tensor selfCopy = OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options(), ACL_FORMAT_ND);
  selfCopy.copy_(self);
  return bernoulli_npu_(selfCopy, p, gen);
}
} // namespace native
} // namespace at
