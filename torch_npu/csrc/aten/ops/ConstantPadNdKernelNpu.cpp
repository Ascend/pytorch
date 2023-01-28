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
at::Tensor constant_pad_nd_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::IntArrayRef pad, at::Scalar value) {
  c10::SmallVector<int64_t, N> vectorInt;
  c10::SmallVector<int64_t, N> paddingsVector = array_to_small_vector(pad);
  paddingsVector.resize(2 * self.dim(), 0);
  for (int64_t i = paddingsVector.size(); i > 0; i -= 2) {
    vectorInt.emplace_back(paddingsVector[i - 2]);
    vectorInt.emplace_back(paddingsVector[i - 1]);
  }

  float val = CalcuOpUtil::GetScalarFloatValue(value);

  at::Tensor value_tensor = at::empty({1}, self.options()).fill_(val);

  OpCommand cmd;
  cmd.Name("PadV3")
      .Input(self)
      .Input(vectorInt, at::kInt)
      .Input(value_tensor)
      .Output(result)
      .Attr("mode", (string)"constant")
      .Attr("paddings_contiguous", true)
      .Run();

  return result;
}

bool is_backward(at::IntArrayRef pad) {
  for (int i = 0; i<pad.size(); i++) {
    if (pad[i] < 0) {
      return true;
    }
  }
  return false;
}

at::Tensor NPUNativeFunctions::constant_pad_nd(const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value) {
  TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size());

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
    TORCH_CHECK(pad.size() % 2 == 0,
        "Length of pad must be even but instead it equals ", pad.size());

    int64_t max_pad_size = 2 * self.dim();
    auto pad_vec = array_to_small_vector(pad);
    if (pad.size() < max_pad_size) {
      for (int64_t i = 0; i < max_pad_size - pad.size(); i++) {
        pad_vec.emplace_back(0);
      }
    }

    c10::SmallVector<int64_t, SIZE> begin_list(self.dim(), 0);
    c10::SmallVector<int64_t, SIZE> end_list;
    for (auto i: self.sizes()) {
      end_list.push_back(i);
    }
    c10::SmallVector<int64_t, SIZE> strides(self.dim(), 1);

    at::Tensor result = self;
    for (int64_t i = 0; i < self.dim(); i++) {
      if (pad_vec[max_pad_size - 2 * (i + 1)] == 0 && pad_vec[max_pad_size - 1 - 2 * i] == 0) {
          continue;
      }
      begin_list[i] = begin_list[i] + (-pad_vec[max_pad_size - 2 * (i + 1)]);
      end_list[i] = end_list[i] + pad_vec[max_pad_size - 1 - 2 * i];
      result = NPUNativeFunctions::npu_indexing(result, begin_list, end_list, strides, 0, 0, 0, 0, 0);
      begin_list[i] = 0;
      end_list[i] = result.size(i);
    }

    return result;
  }

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, new_shape);

  // calculate the output result of the NPU
  constant_pad_nd_out_npu_nocheck(result, self, pad, value);

  return result;
}
} // namespace native
} // namespace at_npu
