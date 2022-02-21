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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<int64_t, SIZE> reflection_pad1d_npu_output_size(const at::Tensor& self, at::IntArrayRef padding) {
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

  int64_t Wo = W +  padding_l + padding_r;

  c10::SmallVector<int64_t, SIZE> outputSize = {N, C, H, Wo};
  return outputSize;
}

at::Tensor& reflection_pad1d_out_npu_nocheck(at::Tensor& out, const at::Tensor& self, at::IntArrayRef padding) {
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
  if(self.dtype() == at::kHalf) {
    cmd.Name("PadV3")
    .Input(self)
    .Input(vectorInt, at::kInt)
    .Input(value_tensor, self.scalar_type())
    .Output(out)
    .Attr("mode", (string)"reflect")
    .Attr("paddings_contiguous", true)
    .Run();
  } else {
    cmd.Name("MirrorPad")
    .Input(self)
    .Input(vectorInt, at::kInt)
    .Output(out)
    .Attr("mode", (string)"REFLECT")
    .Run();
  }
  return out;
}

at::Tensor& NPUNativeFunctions::reflection_pad1d_out(
    const at::Tensor& self,
    at::IntArrayRef padding,
    at::Tensor& result){
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor selfCopy = self;
  selfCopy = selfCopy.unsqueeze(0);

  auto outputSize = reflection_pad1d_npu_output_size(selfCopy, paddings);
  OpPreparation::CheckOut(
      {selfCopy},
      result,
      selfCopy,
      outputSize);
  reflection_pad1d_out_npu_nocheck(result, selfCopy, paddings);
  result = result.squeeze(0);
  return result;
}

at::Tensor NPUNativeFunctions::reflection_pad1d(const at::Tensor& self, at::IntArrayRef padding) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor selfCopy = self;
  selfCopy = selfCopy.unsqueeze(0);

  auto outputSize = reflection_pad1d_npu_output_size(selfCopy, paddings);
  at::Tensor out = OpPreparation::ApplyTensor(selfCopy, outputSize);
  reflection_pad1d_out_npu_nocheck(out, selfCopy, paddings);
  out = out.squeeze(0);
  return out;
}

} // namespace native
} // namespace at_npu