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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor true_divide_dest_output(const Tensor& self, const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  if (not isSelfWrapped) {
    return self;
  } else {
    return other;
  }
}

SmallVector<NPUTensorDesc, N> true_divide_npu_input(
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

SmallVector<NPUTensorDesc, N> true_divide_npu_input(const Tensor& self, Scalar other) {
  SmallVector<NPUTensorDesc, N> inputs;

  auto inputTensor = CalcuOpUtil::create_npu_input_tensor_desc({self});
  auto inputScalar =
  CalcuOpUtil::create_npu_input_tensor_desc({other}, self.scalar_type());

  inputs.insert(inputs.end(), inputTensor.begin(), inputTensor.end());
  inputs.insert(inputs.end(), inputScalar.begin(), inputScalar.end());
  return inputs;

}

SmallVector<NPUTensorDesc, N> true_divide_npu_output(
const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> true_divide_npu_attr(const Tensor& self) {
  SmallVector<NPUAttrDesc, N> attrs = {};
  return attrs;
}

Tensor& true_divide_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  // constructs the input and output NPUTensorDesc
  Tensor selfTemp=self;
  Tensor otherTemp = other;
  if (self.scalar_type() == ScalarType::Int){
    selfTemp = self.to(ScalarType::Float);
    result = result.to(ScalarType::Float);

  }
  if(other.scalar_type() == ScalarType::Int||other.scalar_type() == ScalarType::Bool){
    otherTemp = other.to(ScalarType::Float);
  }

  auto inputs = true_divide_npu_input(selfTemp, otherTemp);
  auto outputs = true_divide_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = true_divide_npu_attr(selfTemp);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Div", inputs, outputs, attrs);

  return result;
}

Tensor& true_divide_out_npu(Tensor& result, const Tensor& self, const Scalar other) {

  Tensor selfTemp = self;
  if (self.scalar_type() == ScalarType::Int){
    selfTemp = self.to(ScalarType::Float);
    result = result.to(ScalarType::Float);
  }
  // constructs the input and output NPUTensorDesc
  auto inputs = true_divide_npu_input(selfTemp, other);
  auto outputs = true_divide_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = true_divide_npu_attr(selfTemp);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Div", inputs, outputs, attrs);

  return result;
}

Tensor true_divide_npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  Tensor  selftemp=self;
  Tensor  othertemp = other;
  if (self.scalar_type() == ScalarType::Int || self.scalar_type() == ScalarType::Bool){
    selftemp = self.to(ScalarType::Float);
  }
  if (other.scalar_type() == ScalarType::Int || other.scalar_type() == ScalarType::Bool){
    othertemp = other.to(ScalarType::Float);
  }

  Tensor outputTensor = true_divide_dest_output(selftemp, othertemp);
  auto outputSize = broadcast_ops_npu_output_size(selftemp, othertemp);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
  outputSize,
  outputTensor.options(),
  CalcuOpUtil::get_tensor_npu_format(outputTensor));

  // calculate the output result of the NPU
  true_divide_out_npu(result,selftemp, othertemp);

  return result;
}

Tensor true_divide_npu(const Tensor& self, Scalar other) {
  // calculate the output size
  Tensor selftemp =self;
  if (self.scalar_type() == ScalarType::Int || self.scalar_type() == ScalarType::Bool){
    selftemp = self.to(ScalarType::Float);
  }
  auto outputSize = input_same_output_size(selftemp);
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
  outputSize, selftemp.options(), CalcuOpUtil::get_tensor_npu_format(selftemp));

  // calculate the output result of the NPU
  true_divide_out_npu(result,selftemp, other);

  return result;
}

Tensor& true_divide_npu_(Tensor& self, const Tensor& other) {
  SmallVector<Tensor, N> inputs = {self, other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = true_divide_out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    true_divide_out_npu(self, self, other);
  }

  return self;
}

Tensor& true_divide_npu_(Tensor& self, Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = true_divide_out_npu(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    true_divide_out_npu(self, self, other);
  }

  return self;
}

} // namespace native
} // namespace at
