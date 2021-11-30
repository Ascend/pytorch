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
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor rsub_dest_output(const Tensor& self, const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  return isSelfWrapped ? other : self;
}

Tensor& rsub_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  // other*alpha
  Tensor otherMulResult;
  if (!CalcuOpUtil::is_scalar_one(alpha)) {
    otherMulResult = at::mul(self, alpha);
  }

  OpCommand cmd;
  if (otherMulResult.defined()) {
    cmd.Name("Sub")
       .Input(other)
       .Input(otherMulResult)
       .Output(result)
       .Run();
  } else {
    cmd.Name("Sub")
       .Input(other)
       .Input(self)
       .Output(result)
       .Run();
  }

  return result;
}

Tensor& rsub_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    Scalar other,
    Scalar alpha) {
  // other*alpha
  Tensor scalarValue(at::mul(self, alpha));

  OpCommand cmd;
  cmd.Name("Sub")
       .Input(other, self.scalar_type())
       .Input(scalarValue)
       .Output(result)
       .Run();

  return result;
}

Tensor rsqrt_tensor_npu(const Tensor& self, const Tensor& other, Scalar alpha) {
  // calculate the output size
  Tensor outputTensor = rsub_dest_output(self, other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);

  // calculate the output result of the NPU
  rsub_out_npu_nocheck(result, self, other, alpha);

  return result;
}

Tensor rsqrt_scalar_npu(const Tensor& self, Scalar other, Scalar alpha) {
  Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  rsub_out_npu_nocheck(result, self, other, alpha);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("rsub.Tensor", TORCH_FN(rsqrt_tensor_npu));
  m.impl("rsub.Scalar", TORCH_FN(rsqrt_scalar_npu));
}

} // namespace native
} // namespace at
