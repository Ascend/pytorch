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

at::Tensor or___dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  if (isSelfWrapped) {
    return other;
  } else {
    return self;
  }
}

at::Tensor& or___out_scalar_npu(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  string real_op_name =
      (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& or___out_tensor_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    or___out_scalar_npu(result, self, other.item());
  } else if (self.dim() == 0 && !at_npu::key::isDeviceTensor(self)) {
    or___out_scalar_npu(result, other, self.item());
  } else {

    string real_op_name =
        (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
    OpCommand cmd;
    cmd.Name(real_op_name)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

at::Tensor NPUNativeFunctions::__or__(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor outputTensor = or___dest_output(self, other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);
  or___out_tensor_npu(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::__or__(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  or___out_scalar_npu(result, self, other);

  return result;
}

} // namespace native
} // namespace at_npu