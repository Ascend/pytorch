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

Tensor& hardtanh_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar min_val,
    Scalar max_val,
    Tensor& grad_input) {
  OpPreparation::CheckMemory({grad_output, self}, {grad_input});
  OpCommand cmd;
  cmd.Name("HardtanhGrad")
      .Input(self)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("max_val", max_val)
      .Attr("min_val", min_val)
      .Run();
  return grad_input;
}

Tensor hardtanh_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar min_val,
    Scalar max_val) {
  Tensor grad_input = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  hardtanh_backward_out_npu(grad_output, self, min_val, max_val, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("hardtanh_backward", TORCH_FN(hardtanh_backward_npu));
  m.impl("hardtanh_backward.grad_input", TORCH_FN(hardtanh_backward_out_npu));
}
} // namespace native
} // namespace at
