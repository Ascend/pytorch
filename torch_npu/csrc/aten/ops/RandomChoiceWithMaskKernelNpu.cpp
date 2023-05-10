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

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_random_choice_with_mask(
    const at::Tensor& self,
    int64_t count,
    int64_t seed,
    int64_t seed2) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Bool,
      "The input.dtype should be bool, but get",
      self.scalar_type());
  TORCH_CHECK(
      self.dim() <= 5 && self.dim() >= 1,
      "The input.dim should be in [1, 5], but get",
      self.dim());
  TORCH_CHECK(count > 0, "The count must greater than 0, but get", count);

  at::Tensor result = OpPreparation::ApplyTensor({count, self.dim()}, self.options().dtype(at::kInt), self);
  at::Tensor mask = OpPreparation::ApplyTensor(self, {count});
  OpCommand cmd;
  cmd.Name("RandomChoiceWithMask")
      .Input(self)
      .Output(result)
      .Output(mask)
      .Attr("count", count)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();

  return std::tie(result, mask);
}

} // namespace native
} // namespace at