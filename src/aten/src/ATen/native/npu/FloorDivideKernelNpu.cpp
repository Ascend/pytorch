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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& floor_divide_out_npu(Tensor& result, const Tensor& self, Scalar other) {
  OpCommand cmd;
  cmd.Name("FloorDiv")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();
  return result;
}

Tensor& floor_divide_out_npu(Tensor& result, const Tensor& self, const Tensor& other) {
  // executing the NPU operator
  if (other.dim() == 0) {
    floor_divide_out_npu(result, self, other.item());
  } else {
    OpCommand cmd;
    cmd.Name("FloorDiv")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();    
  }

  return result;
}

Tensor floor_divide_npu(const Tensor& self, const Tensor& other) {
    Tensor temp = other;
    if (other.scalar_type() == ScalarType::Double) {
      temp = other.to(ScalarType::Float);
    }
    if (other.scalar_type() == ScalarType::Long) {
      temp = other.to(ScalarType::Int);
    }
    
    // calculate the output size
    bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
    Tensor outputTensor = isSelfWrapped ? temp : self;

    auto outputSize = broadcast_ops_npu_output_size(self, temp);

    // construct the output tensor of the NPU
    Tensor result = at::empty_with_format(
        outputSize,
        outputTensor.options(),
        CalcuOpUtil::get_tensor_npu_format(self));

    // calculate the output result of the NPU
    floor_divide_out_npu(result, self, temp);

    return result;
}

Tensor floor_divide_npu(const Tensor& self, Scalar other) {

    // calculate the output size
    auto outputSize = input_same_output_size(self);

    // construct the output tensor of the NPU
    Tensor result = at::empty_with_format(
        outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

    // calculate the output result of the NPU
    floor_divide_out_npu(result, self, other);

    return result;
}

Tensor& floor_divide_npu_(Tensor& self, const Tensor& other) {
    Tensor temp = other;
    if (other.scalar_type() == ScalarType::Double) {
      temp = other.to(ScalarType::Float);
    }
    if (other.scalar_type() == ScalarType::Long) {
      temp = other.to(ScalarType::Int);
    }
    SmallVector<Tensor, N> inputs = {self, temp};
    SmallVector<Tensor, N> outputs = {self};
    CalcuOpUtil::check_memory_over_laps(inputs, outputs);

    if (!NpuUtils::check_match(&self)) {
      Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      Tensor result = floor_divide_out_npu(contiguousSelf, contiguousSelf, other);
      NpuUtils::format_fresh_view(self, result);
    } else {
      floor_divide_out_npu(self, self, temp);
    }

    return self;
}

Tensor& floor_divide_npu_(Tensor& self, Scalar other) {
    if (!NpuUtils::check_match(&self)) {
      Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      floor_divide_out_npu(contiguousSelf, contiguousSelf, other);
      NpuUtils::format_fresh_view(self, contiguousSelf);
    } else {
      floor_divide_out_npu(self, self, other);
    }
    return self;
}

} // namespace native
} // namespace at
