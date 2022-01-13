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

Tensor& bitwise_not_out_npu_nocheck(Tensor& result, const Tensor& self) {
  // executing the NPU operator
  string real_op_name =
      (self.dtype() == ScalarType::Bool) ? "LogicalNot" : "Invert";

  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

Tensor& bitwise_not_out_npu(const Tensor& self, Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  bitwise_not_out_npu_nocheck(result, self);
  return result;
}

Tensor bitwise_not_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  bitwise_not_out_npu_nocheck(result, self);
  return result;
}

Tensor& bitwise_not_npu_(Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = bitwise_not_out_npu_nocheck(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_not_out_npu_nocheck(self, self);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("bitwise_not", TORCH_FN(bitwise_not_npu));
  m.impl("bitwise_not_", TORCH_FN(bitwise_not_npu_));
  m.impl("bitwise_not.out", TORCH_FN(bitwise_not_out_npu));
}
} // namespace native
} // namespace at