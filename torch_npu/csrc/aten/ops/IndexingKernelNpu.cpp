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
#include "torch_npu/csrc/aten/XLANativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& XLANativeFunctions::npu_indexing_out(const at::Tensor& self,
    c10::IntArrayRef begin,
    c10::IntArrayRef end,
    c10::IntArrayRef strides,
    int64_t begin_mask,
    int64_t end_mask,
    int64_t ellipsis_mask,
    int64_t new_axis_mask,
    int64_t shrink_axis_mask,
    at::Tensor& result) {

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

at::Tensor XLANativeFunctions::npu_indexing(
    const at::Tensor& self,
    c10::IntArrayRef begin,
    c10::IntArrayRef end,
    c10::IntArrayRef strides,
    int64_t begin_mask,
    int64_t end_mask,
    int64_t ellipsis_mask,
    int64_t new_axis_mask,
    int64_t shrink_axis_mask) {
  // calculate the output size
  c10::SmallVector<int64_t, SIZE> outputSize;
  for (int i = 0; i < self.dim(); i++) {
    TORCH_CHECK(strides[i] != 0, "stride should not be 0");
    outputSize.emplace_back((end[i] + strides[i] - 1 - begin[i]) / strides[i]);
  }
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  npu_indexing_out(self, begin, end, strides,begin_mask, end_mask,
                   ellipsis_mask, new_axis_mask, shrink_axis_mask, result);
  return result;
}

} // namespace native
} // namespace at
