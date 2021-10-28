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

Tensor& triu_out_npu_nocheck(const Tensor& self, int64_t k, Tensor& result) {
  OpCommand cmd;
  cmd.Name("Triu")
    .Input(self)
    .Output(result)
    .Attr("diagonal", k)
    .Run();

  return result;
}

Tensor& triu_out_npu(const Tensor& self, int64_t k, Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = self.npu_dtype_cast(ScalarType::Float);
  }
  triu_out_npu_nocheck(selfCopy, k, result);
  if (self.scalar_type() == ScalarType::Half) {
    result = result.npu_dtype_cast(ScalarType::Half);
  }
  return result;
}

Tensor triu_npu(const Tensor& self, int64_t k) {
  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = self.npu_dtype_cast(ScalarType::Float);
  }
  Tensor result = OpPreparation::ApplyTensor(selfCopy);

  triu_out_npu_nocheck(selfCopy, k, result);
  if (self.scalar_type() == ScalarType::Half) {
    result = result.npu_dtype_cast(ScalarType::Half);
  }

  return result;
}

Tensor& triu_npu_(Tensor& self, int64_t k) {
  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = self.npu_dtype_cast(ScalarType::Float);
  }
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    Tensor result = triu_out_npu_nocheck(contiguousSelf, k, contiguousSelf);
    if (self.scalar_type() == ScalarType::Half) {
      result = result.npu_dtype_cast(ScalarType::Half);
    }
    self.copy_(result);
  } else {
    triu_out_npu_nocheck(selfCopy, k, selfCopy);
    if (self.scalar_type() == ScalarType::Half) {
      selfCopy = selfCopy.npu_dtype_cast(ScalarType::Half);
    }
    self.copy_(selfCopy);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("triu", TORCH_FN(triu_npu));
  m.impl("triu_", TORCH_FN(triu_npu_));
  m.impl("triu.out", TORCH_FN(triu_out_npu));
}

} // namespace native
} // namespace at
