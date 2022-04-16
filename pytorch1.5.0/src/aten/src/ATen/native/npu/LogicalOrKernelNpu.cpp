// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& logical_or_out_npu_nocheck(   
    Tensor& result, 
    const Tensor& self, 
    const Tensor& other) {
  Tensor selfTemp = self;
  Tensor otherTemp = other;
  
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfTemp); 
  bool isOtherWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(otherTemp); 
  
  // 't + 2' to work with any type of tensor, not just LongTensor (which is what
  // integersin Python represent).  
  if (isSelfWrapped && (!isOtherWrapped)) {
    selfTemp = selfTemp.npu_dtype_cast(otherTemp.scalar_type());
  } else if (isOtherWrapped && (!isSelfWrapped)) {
    otherTemp = otherTemp.npu_dtype_cast(selfTemp.scalar_type());
  }
  
  OpCommand cmd;
  cmd.Name("LogicalOr")
    .Input(selfTemp)
    .Input(otherTemp)
    .Output(result)
    .Run();    
  
  return result;
}

Tensor& logical_or_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      result.scalar_type(),
      outputSize);

  logical_or_out_npu_nocheck(result, self, other);

  return result;
}

Tensor logical_or_npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  logical_or_out_npu_nocheck(result, self, other);

  return result.toType(kBool);
}

Tensor& logical_or_npu_(Tensor& self, const Tensor& other) {
  SmallVector<Tensor, N> inputs = {self, other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = logical_or_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    logical_or_out_npu_nocheck(self, self, other);
  }

  return self;
}

} // namespace native
} // namespace at