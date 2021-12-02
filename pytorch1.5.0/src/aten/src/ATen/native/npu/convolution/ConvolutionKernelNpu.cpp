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
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

constexpr int input_batch_size_dim = 0;
constexpr int output_batch_size_dim = 0;
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

inline SmallVector<int64_t, N> expand_dim_if_needed(
    IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    SmallVector<int64_t, N> expand_dim_param_vec;
    for (int64_t i = 0; i < expected_dim; i++) {
      expand_dim_param_vec.emplace_back(list_param[0]);
    }
    return expand_dim_param_vec;
  } else {
    return CalcuOpUtil::ConvertIntArrayRefToSmallVector(list_param);
  }
}

inline SmallVector<int64_t, N> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation = IntArrayRef()) {
  bool has_dilation = dilation.size() > 0;
  int64_t dim = input_size.size();
  SmallVector<int64_t, N> output_size;
  output_size.resize(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (int64_t d = 2; d < dim; ++d) {
    int64_t dilation_ = has_dilation ? dilation[d - 2] : 1;
    int64_t kernel = dilation_ * (weight_size[d] - 1) + 1;
    output_size[d] =
        (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }

  return output_size;
}

inline SmallVector<int64_t, N> conv_input_size(
    IntArrayRef output_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  int64_t dim = output_size.size();
  SmallVector<int64_t, N> input_size;
  input_size.resize(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (int64_t d = 2; d < dim; ++d) {
    int64_t kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] -
        (2 * padding[d - 2]) + kernel + output_padding[d - 2];
  }

  return input_size;
}

SmallVector<int64_t, SIZE> convolution_transpose3d_npu_output_size(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  int64_t N = input.size(0);
  int64_t D = input.size(2);
  int64_t H = input.size(3);
  int64_t W = input.size(4);
  int64_t Co = weight.size(1) * groups;
  auto kernel_size = weight.sizes().slice(2);

  int64_t Do = (D - 1) * stride[0] - 2 * padding[0] +
      dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
  int64_t Ho = (H - 1) * stride[1] - 2 * padding[1] +
      dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;
  int64_t Wo = (W - 1) * stride[2] - 2 * padding[2] +
      dilation[2] * (kernel_size[2] - 1) + output_padding[2] + 1;

  SmallVector<int64_t, SIZE> outputSize = {N, Co, Do, Ho, Wo};

  return outputSize;
}

Tensor& convolution_transpose3d_out_npu_nocheck(
    Tensor& result,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
  SmallVector<int64_t, N> outputpadding = {0, 0, 0, 0, 0};
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};
  string dataFormat = "NCDHW";

  SmallVector<int64_t, N> sizeVec = array_to_small_vector(result.sizes());
  OpCommand cmd;
  cmd.Name("Conv3DTranspose")
      .Input(sizeVec, at::kInt)
      .Input(input)
      .Input(weight);
  if (bias.defined()){
    cmd.Input(bias);
  }
  cmd.Output(result)
      .Attr("pads", paddings)
      .Attr("output_padding", outputpadding)
      .Attr("strides", stridesSize)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return result;
}

Tensor convolution_transpose3d_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  // calculate the output size
  auto outputSize = convolution_transpose3d_npu_output_size(
      input, weight, bias, padding, output_padding, stride, dilation, groups);

  // construct the output tensor of the NPU
  Tensor result =
      OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_NDC1HWC0);

  // calculate the output result of the NPU
  convolution_transpose3d_out_npu_nocheck(
      result, input, weight, bias, padding, output_padding, stride, dilation, groups);

  return result;
}

void view1d_as_2d(
    SmallVector<int64_t, N>& stride,
    SmallVector<int64_t, N>& padding,
    SmallVector<int64_t, N>& dilation,
    SmallVector<int64_t, N>& output_padding) {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

Tensor view4d(const Tensor& tensor) {
  return tensor.unsqueeze(2);
}

Tensor view3d(const Tensor& tensor) {
  return tensor.squeeze(2);
}

Tensor conv2d_npu_(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  return at::convolution(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

Tensor _conv3d_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(input.ndimension() == 5,
      "The dim of input shoould be 5 but here is ", input.ndimension());
  return at::convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0, 0}},
      groups);
}

Tensor conv_transpose2d_npu_(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  return at::convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      true,
      output_padding,
      groups);
}

Tensor conv_transpose3d_npu_(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  return at::convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      true,
      output_padding,
      groups);
}

Tensor convolution_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  return at::_convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      false,
      false,
      false);
}

Tensor _convolution_npu(
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled) {
  Tensor input = input_;
  Tensor weight = weight_;
  Tensor bias = bias_;

  int64_t k = weight.ndimension();
  int64_t dim = k - 2;

  auto stride = expand_dim_if_needed(stride_, "stride", dim);
  auto padding = expand_dim_if_needed(padding_, "padding", dim);
  auto dilation = expand_dim_if_needed(dilation_, "dilation", dim);
  bool transposed = transposed_;
  auto output_padding =
      expand_dim_if_needed(output_padding_, "output_padding", dim);
  int64_t groups = groups_;

  if (input.size(0) == 0) {
    // don't send empty inputs through backends
    // but need to compute correct output size first and set up history for
    // params
    SmallVector<int64_t, N> o;
    if (!transposed) {
      o = conv_output_size(
          input.sizes(), weight.sizes(), padding, stride, dilation);
    } else {
      o = conv_input_size(
          input.sizes(),
          weight.sizes(),
          padding,
          output_padding,
          stride,
          dilation,
          groups);
    }
    Tensor weight_view = at::_unsafe_view(weight, -1);
    Tensor out = input * weight_view[0];
    if (bias.defined()) {
      out = out + bias[0];
    }
    return out.view(o);
  }

  if (k == 3) {
    view1d_as_2d(stride, padding, dilation, output_padding);
    input = view4d(input);
    weight = view4d(weight);
  }

  Tensor output;
  if (!transposed) {
    output = at::npu_convolution(
        input, weight, bias, stride, padding, dilation, groups);
  } else {
    output = at::npu_convolution_transpose(
        input, weight, bias, padding, output_padding, stride, dilation, groups);
  }

  if (k == 3) {
    output = view3d(output);
  }

  return output;
}

Tensor npu_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  int64_t dim = input.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  Tensor output;
  if (dim == 4) {
    output =
        at::npu_conv2d(input, weight, bias, stride, padding, dilation, groups);
  }

  if (dim == 5) {
    bool is_dilated = false;
    for (unsigned int d : dilation) {
      is_dilated |= (d != 1);
    }
    if (groups == 1 && !is_dilated) {
      output = at::slow_conv3d(
          input, weight, kernel_size, bias, stride, padding);
    } else {
      output = at::npu_conv3d(
          input, weight, bias, stride, padding, dilation, groups);
    }
  }

  return output;
}

Tensor npu_convolution_transpose(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  int64_t dim = input.ndimension();

  Tensor output;
  if (dim == 4) {
    output = at::npu_conv_transpose2d(
        input, weight, bias, padding, output_padding, stride, dilation, groups);
  }

  if (dim == 5) {
    output = convolution_transpose3d_npu(
        input, weight, bias, padding, output_padding, stride, dilation, groups);
  }

  return output;
}

tuple<Tensor, Tensor, Tensor> npu_convolution_backward(
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  int64_t dim = input.ndimension();

  tuple<Tensor, Tensor, Tensor> output;
  if (dim == 4) {
    output = at::npu_conv2d_backward(
        input,
        grad,
        weight,
        stride,
        padding,
        dilation,
        groups,
        grad_input_mask);
  }

  if (dim == 5) {
    output = at::npu_conv3d_backward(
        input,
        grad,
        weight,
        stride,
        padding,
        dilation,
        groups,
        grad_input_mask);
  }
  // Note:weight.grad should be equal weight
  if (std::get<1>(output).defined()) {
    std::get<1>(output) = std::get<1>(output).npu_dtype_cast(weight.scalar_type());
  }
  return output;
}

tuple<Tensor, Tensor, Tensor> npu_convolution_transpose_backward(
    const Tensor& input,
    const Tensor& grad,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  int64_t dim = input.ndimension();

  tuple<Tensor, Tensor, Tensor> output;
  if (dim == 4) {
    output = at::npu_conv_transpose2d_backward(
        input,
        grad,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        grad_input_mask);
  }

  if (dim == 5) {
    output = at::npu_conv_transpose3d_backward(
        input,
        grad,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups,
        grad_input_mask);
  }
  // Note:weight.grad should be equal weight
  if (std::get<1>(output).defined()) {
    std::get<1>(output) = std::get<1>(output).npu_dtype_cast(weight.scalar_type());
  }
  return output;
}

tuple<Tensor, Tensor, Tensor> npu_convolution_double_backward(
    const Tensor& ggI, const Tensor& ggW, const Tensor& ggb,
    const Tensor& input, const Tensor& gO_r, const Tensor& weight_r,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    int64_t groups_, std::array<bool, 3> grad_input_mask){
  int64_t dim = input.ndimension();
  Tensor ggO;
  Tensor gI;
  Tensor gW;
  if (dim == 4) {
    std::tie(ggO, gI, gW) = at::_convolution_double_backward(ggI, ggW, ggb, gO_r, weight_r, input, stride_, padding_,
        {{1, 1}}, false, {{0, 0}}, 1, false, false, false, grad_input_mask);
  }
  if (dim == 5) {
    std::tie(ggO, gI, gW) = at::_convolution_double_backward(ggI, ggW, ggb, gO_r, weight_r, input, stride_, padding_,
        {{1, 1, 1}}, false, {{0, 0, 0}}, 1, false, false, false, grad_input_mask);
  }
  return std::tie(ggO, gI, gW);
}

Tensor _convolution_nogroup_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding) {
  Tensor output;
  if (!transposed) {
    output =
        at::npu_convolution(input, weight, bias, stride, padding, dilation, 1);
  }

  return output;
}

Tensor thnn_conv2d_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  return std::get<0>(at::thnn_conv2d_forward(
      self, weight, kernel_size, bias, stride, padding));
}

Tensor& thnn_conv2d_out_npu(
    Tensor& output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  Tensor finput = at::empty({0}, self.options());
  Tensor fgrad_input = at::empty({0}, self.options());
  return std::get<0>(at::thnn_conv2d_forward_out(
      output,
      finput,
      fgrad_input,
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding));
}

tuple<Tensor, Tensor, Tensor> thnn_conv2d_forward_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  Tensor finput = at::empty({0}, self.options());
  Tensor fgrad_input = at::empty({0}, self.options());
  Tensor output =
      at::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
  return tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}

tuple<Tensor&, Tensor&, Tensor&> thnn_conv2d_forward_out_npu(
    Tensor& output,
    Tensor& finput,
    Tensor& fgrad_input,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  conv2d_out_npu(output, self, weight, bias, stride, padding, {1, 1}, 1);
  return tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
}

tuple<Tensor, Tensor, Tensor> thnn_conv2d_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& finput,
    const Tensor& fgrad_input,
    std::array<bool, 3> output_mask) {
  return at::npu_convolution_backward(
      self, grad_output, weight, stride, padding, {1, 1}, 1, output_mask);
}

Tensor& thnn_conv_depthwise2d_out_npu(
    Tensor& out,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward_out(
      out, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward(
      self, weight, kernel_size, bias, stride, padding, dilation);
}

} // namespace native
} // namespace at
