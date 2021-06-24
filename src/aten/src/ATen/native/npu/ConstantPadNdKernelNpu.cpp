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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor constant_pad_nd_out_npu_nocheck(Tensor& result, const Tensor& self, IntArrayRef pad, Scalar value){
  SmallVector<int64_t, N> vectorInt;

  SmallVector<int64_t, N> paddingsVector = array_to_small_vector(pad);
  paddingsVector.resize(2 * self.dim(), 0);
  for (int64_t i = paddingsVector.size(); i > 0; i -= 2) {
    vectorInt.emplace_back(paddingsVector[i - 2]);
    vectorInt.emplace_back(paddingsVector[i - 1]);
  }

  float val = CalcuOpUtil::get_scalar_float_value(value);

  SmallVector<int64_t, N> value_tensor = {(int64_t)val};

  OpCommand cmd;
  cmd.Name("PadV3")
    .Input(self)
    .Input(vectorInt, at::kInt)
    .Input(value_tensor, self.scalar_type())
    .Output(result)
    .Attr("mode", (string)"constant")
    .Attr("paddings_contiguous", true)
    .Run();

  return result;
}

bool is_backward(IntArrayRef pad) {
  bool flag = false;
  for (int i=0; i<pad.size(); i++){
    if (pad[i] < 0) {
      flag = true;
      break;
    }
  }
  return flag;
}

Tensor constant_pad_nd_npu(const Tensor& self, IntArrayRef pad, Scalar value){
  TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ",
            pad.size());

  auto input_sizes = self.sizes();
  auto l_inp = self.dim();

  auto l_pad = pad.size() / 2;
  auto l_diff = l_inp - l_pad;
  TORCH_CHECK(l_inp >= (int64_t)l_pad, "Length of pad should be no more than twice the number of "
            "dimensions of the input. Pad length is ", pad.size(), "while the input has ",
            l_inp, "dimensions.");

  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < (size_t)l_diff; i++) {
    new_shape.emplace_back(input_sizes[i]);
  }

  for (size_t i = 0; i < (size_t)l_pad; i++) {
    auto pad_idx = pad.size() - ((i + 1) * 2);
    auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
    TORCH_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
              pad[pad_idx], " and ", pad[pad_idx + 1], "resulted in a negative output size, "
              "which is invalid. Check dimension ", l_diff + i, "of your input.");
    new_shape.emplace_back(new_dim);
  }

  if (is_backward(pad)) {
    TORCH_CHECK(self.dim() == 4, "only support 4D now, but self.dim is",self.dim());
    TORCH_CHECK(pad.size()  == 4, "Length of pad must is 4 now, but pad.size() is", pad.size());

    SmallVector<int64_t, SIZE> begin_list = {0, 0, -pad[2], -pad[0]};
    SmallVector<int64_t, SIZE> end_list = {self.size(0), self.size(1), self.size(-2) + pad[3], self.size(-1) + pad[1]};
    SmallVector<int64_t, SIZE> strides = {1, 1, 1, 1};

    return at::npu_indexing(self, begin_list, end_list, strides);
  }

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
    new_shape, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  constant_pad_nd_out_npu_nocheck(result, self, pad, value);

  return result;
}

} // namespace native
} // namespace at