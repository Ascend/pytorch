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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> logical_and_npu_input(
    const Tensor& self,
    const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
  bool isOtherWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(other);
  auto inputs = CalcuOpUtil::create_npu_input_tensor_desc({self, other});

  // 't + 2' to work with any type of tensor, not just LongTensor (which is what
  // integersin Python represent).
  if (isSelfWrapped && (!isOtherWrapped)) {
    inputs[0].scalarType = other.scalar_type();
  } else if (isOtherWrapped && (!isSelfWrapped)) {
    inputs[1].scalarType = self.scalar_type();
  }

  return inputs;
}

SmallVector<NPUTensorDesc, N> logical_and_npu_output(const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> logical_and_npu_attr(const Tensor& self) {
  SmallVector<NPUAttrDesc, N> attrs = {};
  return attrs;
}

Tensor& logical_and_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {

  // constructs the input and output NPUTensorDesc
  auto inputs = logical_and_npu_input(self, other);
  auto outputs = logical_and_npu_output({result});
  // constructs the attr of the NPUAttrDesc
  auto attrs = logical_and_npu_attr(self);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("LogicalAnd", inputs, outputs, attrs);

  return result;
}

Tensor& logical_and_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      result.scalar_type(),
      outputSize);

  logical_and_out_npu_nocheck(result, self, other);

  return result;
}

Tensor logical_and_npu(const Tensor& self, const Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  Tensor result = at::empty_with_format(
      outputSize,
      self.options(), 
      CalcuOpUtil::get_tensor_npu_format(self));

  logical_and_out_npu_nocheck(result, self, other);

  return result.toType(kBool);
}

Tensor& logical_and_npu_(Tensor& self, const Tensor& other) {
  SmallVector<Tensor, N> inputs = {self, other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = logical_and_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    logical_and_out_npu_nocheck(self, self, other);
  }

  return self;
}

} // namespace native
} // namespace at