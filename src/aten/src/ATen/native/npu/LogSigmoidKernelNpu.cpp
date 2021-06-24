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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor&, Tensor&> log_sigmoid_forward_out_npu(
    Tensor& output,
    Tensor& buffer,
    const Tensor& self) {
  OpCommand cmd;
  cmd.Name("LogSigmoid")
     .Input(self)
     .Output(output)
     .Run();

  return std::tie(output, buffer);
}

tuple<Tensor, Tensor> log_sigmoid_forward_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  Tensor output = at::empty_with_format(
        self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  Tensor buffer = at::empty({0}, self.options());

  // calculate the output result of the NPU
  log_sigmoid_forward_out_npu(output, buffer, self);
  return tuple<Tensor, Tensor>(output, buffer);
}

Tensor& log_sigmoid_out_npu(Tensor& result, const Tensor& self) {
  Tensor buffer = at::empty({0}, self.options());
  return std::get<0>(at::log_sigmoid_forward_out(result, buffer, self));
}

Tensor log_sigmoid_npu(const Tensor& self) {
  return std::get<0>(at::log_sigmoid_forward(self));
}

} // namespace native
} // namespace at