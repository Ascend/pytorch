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

Tensor& adaptive_max_pool2d_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    Tensor& grad_input) {
  auto inputsize = self.sizes();
  SmallVector<int64_t, N> input_size;
  if (inputsize.size() == 3) {
      SmallVector<int64_t, N> size = { inputsize[1], inputsize[2] };
      input_size = IntArrayRef(size);
  } else if (inputsize.size() == 4) {
      SmallVector<int64_t, N> size = { inputsize[2], inputsize[3] };
      input_size = IntArrayRef(size);
  }
  SmallVector<int64_t, N> output_size = {
      grad_output.size(grad_output.dim() - 2),
      grad_output.size(grad_output.dim() - 1)};

  if (input_size[0] % output_size[0] == 0 && input_size[1] % output_size[1] == 0) {
      int64_t kernel_size[2];
      int64_t stride[2];
      int64_t padding[2];
      int64_t strideH = input_size[0] / output_size[0];
      int64_t strideW = input_size[1] / output_size[1];
      int64_t kernel_sizeH = input_size[0] - (output_size[0] - 1) * strideH;
      int64_t kernel_sizeW = input_size[1] - (output_size[1] - 1) * strideW;
      stride[0] = strideH;
      stride[1] = strideW;
      kernel_size[0] = kernel_sizeH;
      kernel_size[1] = kernel_sizeW;
      padding[0] = padding[1] = 0;
      SmallVector<int64_t, N> kernelSize = {1, kernel_size[0], kernel_size[1], 1};
      SmallVector<int64_t, N> stridesSize = {1, stride[0], stride[1], 1};
      SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
      SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
      bool ceil_mode = false;
      OpCommand cmd;
      cmd.Name("MaxPoolGradWithArgmaxV1")
          .Input(self)
          .Input(grad_output)
          .Input(indices, "", "uint16")
          .Output(grad_input)
          .Attr("ksize", kernelSize)
          .Attr("strides", stridesSize)
          .Attr("pads", paddings)
          .Attr("dilations", dilations)
          .Attr("ceil_mode", ceil_mode)
          .Run();
    } else {
        // H and W can not be divided, temporarily reported error processing
        AT_ERROR("H and W must be divisible");
    }
    return grad_input;
}

Tensor adaptive_max_pool2d_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices) {
  TORCH_CHECK(
      (self.dim() == 3 || self.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  Tensor grad_input = OpPreparation::ApplyTensorWithFormat(self, ACL_FORMAT_NC1HWC0);
  adaptive_max_pool2d_backward_out_npu(grad_output, self, indices, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("adaptive_max_pool2d_backward", TORCH_FN(adaptive_max_pool2d_backward_npu));
  m.impl("adaptive_max_pool2d_backward.grad_input", TORCH_FN(adaptive_max_pool2d_backward_out_npu));
}
} // namespace native
} // namespace at
