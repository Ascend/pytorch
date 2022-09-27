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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& masked_scatter_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& source) {
  at::Tensor maskBool = mask; 
  if (!(mask.dtype() == at::kBool)) {
    maskBool = NPUNativeFunctions::npu_dtype_cast(maskBool, at::kBool);
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

at::Tensor& NPUNativeFunctions::masked_scatter_(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& source) {
  c10::SmallVector<at::Tensor, N> inputs = {self, mask, source};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  at::Tensor selfFp32 = self;
  at::Tensor sourceFp32 = source;
  at::ScalarType selfType = self.scalar_type();
  if (selfType == at::ScalarType::Half) {
    selfFp32 = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    sourceFp32 = NPUNativeFunctions::npu_dtype_cast(source, at::ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(selfFp32);
    at::Tensor result = masked_scatter_out_npu_nocheck(contiguousSelf, contiguousSelf, mask, sourceFp32);
    NpuUtils::format_fresh_view(self, result);
  } else {
    masked_scatter_out_npu_nocheck(selfFp32, selfFp32, mask, sourceFp32);
    self.copy_(selfFp32);
  }

  return (self.scalar_type() != selfType) ? self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Half) : self;
}
} // namespace native
} // namespace at_npu