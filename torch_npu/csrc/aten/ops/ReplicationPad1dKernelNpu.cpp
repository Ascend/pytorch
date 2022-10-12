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

c10::SmallVector<int64_t, SIZE> replication_pad1d_npu_output_size(const at::Tensor& self, at::IntArrayRef padding) {
  int64_t N = self.size(0);
  int64_t C = self.size(1);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t padding_l = padding[0];
  int64_t padding_r = padding[1];

  int64_t Wo = W + padding_l + padding_r;

  c10::SmallVector<int64_t, SIZE> outputSize = {N, C, H, Wo};
  return outputSize;
}

at::Tensor& replication_pad1d_out_npu_nocheck(at::Tensor& out, const at::Tensor& self, at::IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  c10::SmallVector<int64_t, N> vectorInt;
  c10::SmallVector<int64_t, N> paddingsVector = array_to_small_vector(padding);
  paddingsVector.resize(2 * self.dim(), 0);
  for (int64_t i = paddingsVector.size(); i > 1; i -= 2) {
    vectorInt.emplace_back(paddingsVector[i - 2]);
    vectorInt.emplace_back(paddingsVector[i - 1]);
  }
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

at::Tensor& NPUNativeFunctions::replication_pad1d_out(
    const at::Tensor& self,
    at::IntArrayRef padding,
    at::Tensor& out) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor selfCopy = self.unsqueeze(0);
  auto outputSize = replication_pad1d_npu_output_size(selfCopy, paddings);
  OpPreparation::CheckOut(
      {selfCopy},
      out,
      selfCopy,
      outputSize);
  replication_pad1d_out_npu_nocheck(out, selfCopy, paddings);
  out = out.squeeze(0);
  return out;
}

at::Tensor NPUNativeFunctions::replication_pad1d(const at::Tensor& self, at::IntArrayRef padding) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor selfCopy = self.unsqueeze(0);

  auto outputSize = replication_pad1d_npu_output_size(selfCopy, paddings);
  at::Tensor out = OpPreparation::ApplyTensor(selfCopy, outputSize);
  replication_pad1d_out_npu_nocheck(out, selfCopy, paddings);
  out = out.squeeze(0);
  return out;
}
} // namespace native
} // namespace at_npu
