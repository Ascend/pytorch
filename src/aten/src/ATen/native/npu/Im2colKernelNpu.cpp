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

SmallVector<int64_t, SIZE> image_to_col_npu_output_size(
    const Tensor& self,
    IntArrayRef ksizes,
    IntArrayRef strides,
    IntArrayRef dilations,
    IntArrayRef pads) {

  if (ksizes.size() == 1) {
    SmallVector<int64_t, SIZE> kernel_sizes = {ksizes[0], ksizes[0]};
    ksizes = IntArrayRef(kernel_sizes);
  }

  strides = strides.empty() ? IntArrayRef({1}) : strides;
  if (strides.size() == 1) {
    SmallVector<int64_t, SIZE> stride_sizes = {strides[0], strides[0]};
    strides = IntArrayRef(stride_sizes);
  }

  dilations = dilations.empty() ? IntArrayRef({1}) : dilations;
  if (dilations.size() == 1) {
    SmallVector<int64_t, SIZE> dilation_sizes = {dilations[0], dilations[0]};
    dilations = IntArrayRef(dilation_sizes);
  }

  pads = pads.empty() ? IntArrayRef({0}) : pads;
  if (pads.size() == 1) {
    SmallVector<int64_t, SIZE> pad_sizes = {pads[0], pads[0]};
    pads = IntArrayRef(pad_sizes);
  }

  int64_t out_h = (self.size(2) + 2 * pads[0] - (dilations[0] * (ksizes[0] - 1) + 1)) / strides[0] + 1;
  int64_t out_w = (self.size(3) + 2 * pads[1] - (dilations[1] * (ksizes[1] - 1) + 1)) / strides[1] + 1;
  return {self.size(0), self.size(1) * ksizes[0] * ksizes[1], out_h * out_w};
}

Tensor& im2col_out_npu_nocheck(Tensor& result, const Tensor &self, IntArrayRef kernel_size,
                      IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "im2col: kernel_size must either be a single int, or a tuple of two ints");
  if (kernel_size.size() == 1) {
    SmallVector<int64_t, SIZE> kernel_sizes = {kernel_size[0], kernel_size[0]};
    kernel_size = IntArrayRef(kernel_sizes);
  }

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "im2col: stride must either be omitted, a single int, or a tuple of two ints");
  stride = stride.empty() ? IntArrayRef({1}) : stride;

  TORCH_CHECK(dilation.empty() || dilation.size() == 1 || dilation.size() == 2,
    "im2col: dilation must either be omitted, a single int, or a tuple of two ints");
  dilation = dilation.empty() ? IntArrayRef({1}) : dilation;
  
  TORCH_CHECK(padding.empty() || padding.size() == 1 || padding.size() == 2,
    "im2col: padding must either be omitted, a single int, or a tuple of two ints");
  padding = padding.empty() ? IntArrayRef({0}) : padding;
  if (padding.size() == 1) {
    SmallVector<int64_t, SIZE> pads = {padding[0], padding[0], padding[0], padding[0]};
    padding = IntArrayRef(pads);
  } else if (padding.size() == 2) {
    SmallVector<int64_t, SIZE> pads = {padding[0], padding[0], padding[1], padding[1]};
    padding = IntArrayRef(pads);
  }

  int64_t strideH = 1;
  int64_t strideW = 1;
  if (stride.size() == 1) {
    strideH = stride[0];
    strideW = stride[0];
  } else if (stride.size() == 2) {
    strideH = stride[0];
    strideW = stride[1];
  }

  int64_t dilationH = 1;
  int64_t dilationW = 1;
  if (dilation.size() == 1) {
    dilationH = dilation[0];
    dilationW = dilation[0];
  } else if (dilation.size() == 2) {
    dilationH = dilation[0];
    dilationW = dilation[1];
  }
  SmallVector<int64_t, N> kernelSize = {kernel_size[0], kernel_size[1]};
  SmallVector<int64_t, N> stridesSize = {strideH, strideW};
  SmallVector<int64_t, N> dilationsSize = {dilationH, dilationW};
  SmallVector<int64_t, N> padsSize = {padding[0], padding[1], padding[2], padding[3]};
  string padding_mode = "CALCULATED";

  OpCommand cmd;
  cmd.Name("Im2col")
      .Input(self)
      .Output(result)
      .Attr("ksizes", kernelSize)
      .Attr("strides", stridesSize)
      .Attr("dilations", dilationsSize)
      .Attr("padding_mode", padding_mode)
      .Attr("pads", padsSize)
      .Run();
  return result;
}

Tensor& im2col_out_npu(Tensor& result, const Tensor &self, IntArrayRef kernel_size,
                      IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  OpPreparation::CheckOut(
    {self},
    result,
    self,
    image_to_col_npu_output_size(self, kernel_size, stride, dilation, padding));

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, kernel_size, dilation, padding, stride](Tensor& result){im2col_out_npu_nocheck(result, self, kernel_size, dilation, padding, stride);})
   .Call(result);
}

Tensor im2col_npu(const Tensor &self, IntArrayRef kernel_size, IntArrayRef dilation,
                  IntArrayRef padding, IntArrayRef stride) {
  // calculate the output size
  auto outputSize =
      image_to_col_npu_output_size(self, kernel_size, stride, dilation, padding);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  im2col_out_npu(result, self, kernel_size, dilation, padding, stride);

  return result;
}

} // namespace native
} // namespace at
