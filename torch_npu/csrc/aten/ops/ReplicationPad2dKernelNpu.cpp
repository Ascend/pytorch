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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& replication_pad2d_out_npu_nocheck(at::Tensor& out, const at::Tensor& self, at::IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  c10::SmallVector<int64_t, N> vectorInt;
  c10::SmallVector<int64_t, N> paddingsVector = array_to_small_vector(padding);
  paddingsVector.resize(2 * self.dim(), 0);
  for (int64_t i = paddingsVector.size(); i > 1; i -= 2) {
    vectorInt.emplace_back(paddingsVector[i - 2]);
    vectorInt.emplace_back(paddingsVector[i - 1]);
  }
  // constructs the attr of the NPUAttrDesc
  c10::SmallVector<int64_t, N> value_tensor = {(int64_t)0};
  OpCommand cmd;
  cmd.Name("PadV3")
    .Input(self)
    .Input(vectorInt, at::kInt)
    .Input(value_tensor, self.scalar_type())
    .Output(out)
    .Attr("mode", (string)"edge")
    .Attr("paddings_contiguous", true)
    .Run();

  return out;
}

at::Tensor& NPUNativeFunctions::replication_pad2d_out(
    const at::Tensor& self,
    at::IntArrayRef padding,
    at::Tensor& out) {
  auto outputSize = replication_pad2d_npu_output_size(self, padding);
  OpPreparation::CheckOut(
      {self},
      out,
      self,
      outputSize);
  return replication_pad2d_out_npu_nocheck(out, self, padding);
}

at::Tensor NPUNativeFunctions::replication_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  auto outputSize = replication_pad2d_npu_output_size(self, padding);
  at::Tensor out = OpPreparation::ApplyTensor(self, outputSize);
  replication_pad2d_out(self, padding, out);

  return out;
}
} // namespace native
} // namespace at_npu