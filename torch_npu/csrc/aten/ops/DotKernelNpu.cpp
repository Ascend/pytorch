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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::dot_out(const at::Tensor& self, const at::Tensor& tensor, at::Tensor& result) {
  c10::SmallVector<int64_t, N> output_size = {};
  OpPreparation::CheckOut(
      {self, tensor},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(),
      output_size);

  OpCommand cmd;
  cmd.Name("Dot")
      .Input(self)
      .Input(tensor)
      .Output(result)
      .Run();

  return result;
}
at::Tensor NPUNativeFunctions::dot(const at::Tensor& self, const at::Tensor& tensor) {
  c10::SmallVector<int64_t, N> output_size = {};
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  dot_out(self, tensor, result);
  return result;
}
} // namespace native
} // namespace at_npu
