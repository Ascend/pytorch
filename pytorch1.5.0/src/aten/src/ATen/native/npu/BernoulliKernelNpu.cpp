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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bernoulli_out_npu(Tensor& result, const Tensor& self, double p) {
  OpCommand cmd;
  cmd.Name("Bernoulli")
      .Input(self)
      .Input(p, ScalarType::Float)
      .Output(result)
      .Run();

  return result;
}

Tensor& bernoulli_out_npu(Tensor& result, const Tensor& self, const Tensor& p) {
  OpCommand cmd;
  cmd.Name("Bernoulli")
      .Input(self)
      .Input(p)
      .Output(result)
      .Run();

  return result;
}

Tensor& bernoulli_npu_(Tensor& self, double p, Generator* gen) {
  OpPreparation::CheckMemory({self}, {self});
  ScalarType selfType = self.scalar_type();
  Tensor selfFp32 = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfFp32 = self.to(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    Tensor result = bernoulli_out_npu(contiguousSelf, contiguousSelf, p);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bernoulli_out_npu(selfFp32, selfFp32, p);
    self.copy_(selfFp32);
  }

  if(self.scalar_type() != selfType){
    self = self.to(ScalarType::Half);
  }
  return self;
}

Tensor& bernoulli_npu_(Tensor& self, const Tensor& p, Generator* gen) {
  OpPreparation::CheckMemory({self}, {self});
  ScalarType selfType = self.scalar_type();
  Tensor selfFp32 = self;
  Tensor pFp32 = OpPreparation::CastBackToOriFormat(p);;
  if (self.scalar_type() == ScalarType::Half) {
    selfFp32 = self.to(ScalarType::Float);
    pFp32 = p.to(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    Tensor result = bernoulli_out_npu(contiguousSelf, contiguousSelf, pFp32);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bernoulli_out_npu(selfFp32, selfFp32, pFp32);
    self.copy_(selfFp32);
  }

  if(self.scalar_type() != selfType){
    self = self.to(ScalarType::Half);
  }
  return self;
}

Tensor bernoulli_npu(const Tensor& self, Generator* gen) {
  const Tensor p = self;
  Tensor selfCopy = at::empty_with_format(
      self.sizes(), self.options(), ACL_FORMAT_ND);
  selfCopy.copy_(self);
  return bernoulli_npu_(selfCopy, p, gen);
}
} // namespace native
} // namespace at
