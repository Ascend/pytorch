// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor dropout_backward_npu(
    const Tensor& grad_output,
    const Tensor& mask,
    double scale) {
  TORCH_CHECK(
      at::isFloatingType(grad_output.scalar_type()),
      "dropoutbackward only supports floating-point dtypes");
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype");
  double retain =  1. - scale;
  Tensor result = OpPreparation::ApplyTensor(grad_output);

  OpCommand cmd;
  cmd.Name("DropOutDoMask")
      .Input(grad_output)
      .Input(mask)
      .Input(retain, grad_output.scalar_type(), MemoryType::MEMORY_HOST)
      .Output(result)
      .Run();

  return result;
}

} // namespace native
} // namespace at
