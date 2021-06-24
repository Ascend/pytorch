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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& masked_scatter_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask,
    const Tensor& source) {
  Tensor maskBool = mask;
  if (!(mask.dtype() == at::kBool)) {
    maskBool = mask.to(at::kBool);
  }
  
  OpCommand cmd;
  cmd.Name("MaskedScatter")
     .Input(self)
     .Input(maskBool)
     .Input(source)
     .Output(result)
     .Run();

  return result;
}

Tensor& masked_scatter_npu_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& source) {
  SmallVector<Tensor, N> inputs = {self, mask, source};
  SmallVector<Tensor, N> outputs = {self};

  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  Tensor selfFp32 = self;
  Tensor sourceFp32 = source;
  ScalarType selfType = self.scalar_type();
  if (self.scalar_type() == ScalarType::Half) {
    selfFp32 = self.to(ScalarType::Float);
    sourceFp32 = source.to(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    Tensor result = masked_scatter_out_npu(contiguousSelf, contiguousSelf, mask, sourceFp32);
    NpuUtils::format_fresh_view(self, result);
  } else {
    masked_scatter_out_npu(selfFp32, selfFp32, mask, sourceFp32);
    self.copy_(selfFp32);
  }

  return (self.scalar_type() != selfType) ? self = self.to(ScalarType::Half) : self;
}
} // namespace native
} // namespace at