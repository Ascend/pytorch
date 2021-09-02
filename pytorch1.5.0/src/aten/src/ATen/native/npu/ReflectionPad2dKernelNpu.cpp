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

SmallVector<int64_t, SIZE> reflection_pad2d_npu_output_size(const Tensor& self, IntArrayRef padding) {
  int64_t N = self.size(0);
  int64_t C = self.size(1);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t padding_l = 0;
  int64_t padding_r = 0;
  int64_t padding_t = 0;
  int64_t padding_b = 0;

  padding_l = padding[0];
  padding_r = padding[1];
  padding_t = padding[2];
  padding_b = padding[3];

  int64_t Ho = H +  padding_t + padding_b;
  int64_t Wo = W +  padding_l + padding_r;

  SmallVector<int64_t, SIZE> outputSize = {N, C, Ho, Wo};
  return outputSize;
}

Tensor& reflection_pad2d_out_npu_nocheck(Tensor& out, const Tensor& self, IntArrayRef padding) {
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
    .Attr("mode", (string)"reflect")
    .Attr("paddings_contiguous", true)
    .Run();

  return out;
}

Tensor& reflection_pad2d_out_npu(Tensor& result, const Tensor& self, IntArrayRef padding){
  //calculate the output size
  auto outputSize = reflection_pad2d_npu_output_size(self, padding);
  //construct the output tensor of the NPU
  OpPreparation::CheckOut(
  {self},
  result,
  self,
  outputSize);
  reflection_pad2d_out_npu_nocheck(result, self, padding);

  return result;
}

Tensor reflection_pad2d_npu(const Tensor& self, IntArrayRef padding) {
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  //calculate the output size
  auto outputSize = reflection_pad2d_npu_output_size(self, padding);
  //construct the output tensor of the NPU
  Tensor out = OpPreparation::ApplyTensor(self, outputSize);
  //calculate the output result of the NPU
  reflection_pad2d_out_npu_nocheck(out, self, padding);

  return out;
}
}
} // namespace at::native
