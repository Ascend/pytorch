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

#include "ATen/native/npu/utils/OpAdapter.h"
namespace at {
namespace native {
using namespace at::native::npu;

Tensor& replication_pad2d_out_npu_nocheck(Tensor& out, const Tensor& self, IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  SmallVector<int64_t, N> vectorInt;
  SmallVector<int64_t, N> paddingsVector = array_to_small_vector(padding);
  paddingsVector.resize(2 * self.dim(), 0);
  for (int64_t i = paddingsVector.size(); i > 1; i -= 2) {
    vectorInt.emplace_back(paddingsVector[i - 2]);
    vectorInt.emplace_back(paddingsVector[i - 1]);
  }
  //constructs the attr of the NPUAttrDesc
  SmallVector<int64_t, N> value_tensor = {(int64_t)0};
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

Tensor& replication_pad2d_out_npu(Tensor& out, const Tensor& self, IntArrayRef padding) {
  //calculate the output size
  auto outputSize = replication_pad2d_npu_output_size(self, padding);
  OpPreparation::CheckOut(
  {self},
  out,
  self,
  outputSize);
  return replication_pad2d_out_npu_nocheck(out, self, padding);
}

Tensor replication_pad2d_npu(const Tensor& self, IntArrayRef padding) {
  //calculate the output size
  auto outputSize = replication_pad2d_npu_output_size(self, padding);
  //construct the output tensor of the NPU
  Tensor out = OpPreparation::ApplyTensor(self, outputSize);

  //calculate the output result of the NPU
  replication_pad2d_out_npu(out, self, padding);

  return out;
}
}
} // namespace at::native
