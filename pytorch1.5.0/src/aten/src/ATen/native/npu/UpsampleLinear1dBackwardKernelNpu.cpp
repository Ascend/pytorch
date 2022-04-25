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

static inline void upsample_linear1d_backward_check(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
    
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  TORCH_CHECK(
      (grad_output.size(1) != 0 && grad_output.size(2) != 0) && grad_output.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      grad_output.sizes());
  
  int64_t output_width = grad_output.size(2);
  int64_t input_width = input_size[2];
  
  TORCH_CHECK(
      output_width > 0 && input_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");
}

Tensor& upsample_linear1d_backward_out_npu(
    Tensor& result,
    const Tensor& grad_output,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales) {
  SmallVector<float, N> sc = {};
  if (scales.has_value()) {
    sc.push_back(scales.value());
  } else {
    float temp = float(grad_output.size(3)) / float(input_size[2]);
    sc.push_back(temp);
  }
  string coordinate_transformation_mode =
      align_corners ? "align_corners" : "half_pixel";

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("ResizeGradD")
      .Input(grad_output, "grads", ACL_FORMAT_NCHW)
      .Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("original_size", input_size)
      .Attr("scales", sc)
      .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
      .Attr("mode", (string) "linear")
      .Run();
  return result;
}

Tensor upsample_linear1d_backward_npu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales) {
  upsample_linear1d_backward_check(grad_output, output_size, input_size);
  Tensor _grad_output = grad_output;
  if(grad_output.scalar_type() != ScalarType::Float)
  {
    _grad_output = _grad_output.npu_dtype_cast(ScalarType::Float);
  }

  // calculate the output size
  int64_t N = _grad_output.size(0);
  int64_t C = _grad_output.size(1);
  int64_t W = input_size[2];

  SmallVector<int64_t, SIZE> outputSize = {N, C, W};
  
  // Since only NCHW format input is currently supported, first convert the
  // input grad_output (3 dimensions) to 4 dimensions as the input of npu
  auto grad_output_4dim = _grad_output.unsqueeze(2);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(_grad_output, outputSize);

  // calculate the output result of the NPU
  upsample_linear1d_backward_out_npu(
      result, grad_output_4dim, input_size, align_corners, scales);
    
  if (result.dtype() != grad_output.dtype()) {
    result = result.to(grad_output.dtype());
  }

  return result;
}

} // namespace native
} // namespace at
