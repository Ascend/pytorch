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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"

namespace at_npu {
namespace native {

at::Tensor& npu_indexing_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::IntArrayRef begin,
    c10::IntArrayRef end,
    c10::IntArrayRef strides,
    int64_t begin_mask,
    int64_t end_mask,
    int64_t ellipsis_mask,
    int64_t new_axis_mask,
    int64_t shrink_axis_mask) {
  OpCommand cmd;
  cmd.Name("StridedSlice")
      .Input(self)
      .Input(begin)
      .Input(end)
      .Input(strides)
      .Output(result)
      .Attr("begin_mask", begin_mask)
      .Attr("end_mask", end_mask)
      .Attr("ellipsis_mask", ellipsis_mask)
      .Attr("new_axis_mask", new_axis_mask)
      .Attr("shrink_axis_mask", shrink_axis_mask)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::npu_indexing_out(
    const at::Tensor& self,
    c10::IntArrayRef begin,
    c10::IntArrayRef end,
    c10::IntArrayRef strides,
    int64_t begin_mask,
    int64_t end_mask,
    int64_t ellipsis_mask,
    int64_t new_axis_mask,
    int64_t shrink_axis_mask,
    at::Tensor& result) {
  if (self.is_complex() && self.scalar_type() != c10::ScalarType::ComplexDouble) {
    c10::SmallVector<at::Tensor, N> self_splite = complex_compute_split(self);

    at::Tensor self_left = self_splite[0].squeeze(-1);
    at::Tensor self_right = self_splite[1].squeeze(-1);
    at::Tensor left_result = OpPreparation::ApplyTensor(result, self_left.options());
    at::Tensor right_result = OpPreparation::ApplyTensor(result, self_right.options());

    npu_indexing_out_nocheck(left_result, self_left, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
        new_axis_mask, shrink_axis_mask);
    npu_indexing_out_nocheck(right_result, self_right, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
        new_axis_mask, shrink_axis_mask);

    auto result_coeffs = NPUNativeFunctions::stack({left_result, right_result}, -1);
    result = at::native::view_as_complex(result_coeffs);
  } else {
    npu_indexing_out_nocheck(result, self, begin, end, strides, begin_mask, end_mask, ellipsis_mask,
        new_axis_mask, shrink_axis_mask);
  }
  return result;
}

at::Tensor NPUNativeFunctions::npu_indexing(
    const at::Tensor& self,
    c10::IntArrayRef begin,
    c10::IntArrayRef end,
    c10::IntArrayRef strides,
    int64_t begin_mask,
    int64_t end_mask,
    int64_t ellipsis_mask,
    int64_t new_axis_mask,
    int64_t shrink_axis_mask) {
  c10::SmallVector<int64_t, SIZE> output_size;
  for (int i = 0; i < self.dim(); i++) {
    TORCH_CHECK(strides[i] != 0, "stride should not be 0");
    output_size.emplace_back((end[i] + strides[i] - 1 - begin[i]) / strides[i]);
  }
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  npu_indexing_out(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
      shrink_axis_mask, result);
  return result;
}

} // namespace native
} // namespace at_npu
