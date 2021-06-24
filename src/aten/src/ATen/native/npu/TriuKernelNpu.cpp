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

#include "ATen/native/npu/utils/OpTemplate.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& triu_out_npu(Tensor& result, const Tensor& self, int64_t k) {
  OpCommand cmd;
  cmd.Name("PTTriu")
    .Input(self)
    .Output(result)
    .Attr("diagonal", k)
    .Run();

  return result;
}

Tensor triu_npu(const Tensor& self, int64_t k) {
  Tensor formatCastOfSelf = self;
  if (self.scalar_type() == ScalarType::Half) {
    formatCastOfSelf = self.npu_dtype_cast(ScalarType::Float);
  }
  Tensor result = at::empty_with_format(
      formatCastOfSelf.sizes(), formatCastOfSelf.options(), CalcuOpUtil::get_tensor_npu_format(formatCastOfSelf));

  triu_out_npu(result, formatCastOfSelf, k);
  if (result.scalar_type() != self.scalar_type()) {
    result = result.npu_dtype_cast(ScalarType::Half);
  }

  return result;
}

Tensor& triu_npu_(Tensor& self, int64_t k) {
  SmallVector<Tensor, N> inputs = {self};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  Tensor selfCopy = self;
  if (self.scalar_type() == ScalarType::Half) {
    selfCopy = self.to(ScalarType::Float);
  }
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    Tensor result = triu_out_npu(contiguousSelf, contiguousSelf, k);
    if (result.scalar_type() != self.scalar_type()) {
      result = result.npu_dtype_cast(ScalarType::Half);
    }
    self.copy_(result);
  } else {
    triu_out_npu(selfCopy, selfCopy, k);
    if (selfCopy.scalar_type() != self.scalar_type()) {
      selfCopy = selfCopy.npu_dtype_cast(ScalarType::Half);
    }
    self.copy_(selfCopy);
  }

  return self;
}

} // namespace native
} // namespace at
