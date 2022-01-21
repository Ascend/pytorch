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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

using namespace torch::autograd;

constexpr int input_batch_size_dim = 0;
constexpr int output_batch_size_dim = 0;
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

bool is_depthwise(
    const at::Tensor& input,
    const at::Tensor& weight,
    int64_t groups,
    bool transposed) {
  return input.is_npu() && !transposed && input.ndimension() == 4 &&
      input.size(1) == groups &&
      groups > 1 && // no point if there is only a single group
      weight.size(0) % input.size(1) ==
      0; // output channels must be a multiple of input channels
}

inline c10::SmallVector<int64_t, N> expand_dim_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    c10::SmallVector<int64_t, N> expand_dim_param_vec;
    for (int64_t i = 0; i < expected_dim; i++) {
      expand_dim_param_vec.emplace_back(list_param[0]);
    }
    return expand_dim_param_vec;
  } else {
    return CalcuOpUtil::ConvertIntArrayRefToSmallVector(list_param);
  }
}

inline c10::SmallVector<int64_t, N> conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef weight_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation = at::IntArrayRef()) {
  bool has_dilation = dilation.size() > 0;
  int64_t dim = input_size.size();
  c10::SmallVector<int64_t, N> output_size;
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

inline c10::SmallVector<int64_t, N> conv_input_size(
    at::IntArrayRef output_size,
    at::IntArrayRef weight_size,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  int64_t dim = output_size.size();
  c10::SmallVector<int64_t, N> input_size;
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


void view1d_as_2d(
    c10::SmallVector<int64_t, N>& stride,
    c10::SmallVector<int64_t, N>& padding,
    c10::SmallVector<int64_t, N>& dilation,
    c10::SmallVector<int64_t, N>& output_padding) {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

at::Tensor view4d(const at::Tensor& tensor) {
  return tensor.unsqueeze(2);
}

at::Tensor view3d(const at::Tensor& tensor) {
  return tensor.squeeze(2);
}

at::Tensor NPUNativeFunctions::conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  return at::convolution(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor _conv3d_npu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
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

at::Tensor NPUNativeFunctions::conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
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

at::Tensor NPUNativeFunctions::conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
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

at::Tensor NPUNativeFunctions::convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
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

at::Tensor _convolution_npu(
    const at::Tensor& input_,
    const at::Tensor& weight_,
    const c10::optional<at::Tensor>& bias_opt_,
    at::IntArrayRef stride_,
    at::IntArrayRef padding_,
    at::IntArrayRef dilation_,
    bool transposed_,
    at::IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled) {
  at::Tensor input = input_;
  at::Tensor weight = weight_;

  const at::Tensor& bias_ = c10::value_or_else(bias_opt_, [] {return at::Tensor();});
  at::Tensor bias = bias_;

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
    c10::SmallVector<int64_t, N> o;
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
    at::Tensor weight_view = at::_unsafe_view(weight, -1);
    at::Tensor out = input * weight_view[0];
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

  at::Tensor output;
  if (!transposed) {
    output = NPUNativeFunctions::npu_convolution(
        input, weight, bias_opt_, stride, padding, dilation, groups);
  } else {
    output = NPUNativeFunctions::npu_convolution_transpose(
        input, weight, bias_opt_, padding, output_padding, stride, dilation, groups);
  }

  if (k == 3) {
    output = view3d(output);
  }

  return output;
}

at::Tensor convolution_kernel_npu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  int64_t dim = input.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  at::Tensor output;
  if (dim == 4) {
    output =
        NPUNativeFunctions::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
  }

  if (dim == 5) {
    bool is_dilated = false;
    for (int d : dilation) {
      is_dilated |= (d != 1);
    }
    if (groups == 1 && !is_dilated) {
      output = at::slow_conv3d(
          input, weight, kernel_size, bias_opt, stride, padding);
    } else {
      output = NPUNativeFunctions::npu_conv3d(
          input, weight, bias_opt, stride, padding, dilation, groups);
    }
  }

  return output;
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  int64_t dim = input.ndimension();

  tuple<at::Tensor, at::Tensor, at::Tensor> output;
  if (dim == 4) {
    output = NPUNativeFunctions::npu_conv2d_backward(
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
    output = NPUNativeFunctions::npu_conv3d_backward(
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
    std::get<1>(output) = NPUNativeFunctions::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_double_backward(
    const c10::optional<at::Tensor>& ggI, const c10::optional<at::Tensor>& ggW, const c10::optional<at::Tensor>& ggb,
    const at::Tensor& input, const at::Tensor& gO_r, const at::Tensor& weight_r,
    at::IntArrayRef stride_, at::IntArrayRef padding_, at::IntArrayRef dilation_,
    int64_t groups_, std::array<bool, 3> grad_input_mask) {
  int64_t dim = input.ndimension();
  at::Tensor ggO;
  at::Tensor gI;
  at::Tensor gW;
  if (dim == 4) {
      std::tie(ggO, gI, gW) = at::_convolution_double_backward(ggI, ggW, ggb, gO_r, weight_r, input, stride_, padding_,
          {{1, 1}}, false, {{0, 0}}, 1, false, false, false, false, grad_input_mask);
  }
  if (dim == 5) {
      std::tie(ggO, gI, gW) = at::_convolution_double_backward(ggI, ggW, ggb, gO_r, weight_r, input, stride_, padding_,
          {{1, 1, 1}}, false, {{0, 0, 0}}, 1, false, false, false, false, grad_input_mask);
  }
  return std::tie(ggO, gI, gW);
}

at::Tensor NPUNativeFunctions::_convolution_nogroup(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding) {
  at::Tensor output;
  if (!transposed) {
    output =
        NPUNativeFunctions::npu_convolution(input, weight, bias_opt, stride, padding, dilation, 1);
  }

  return output;
}

at::Tensor NPUNativeFunctions::thnn_conv2d(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  return std::get<0>(at::thnn_conv2d_forward(
      self, weight, kernel_size, bias, stride, padding));
}

at::Tensor& NPUNativeFunctions::thnn_conv2d_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
  at::Tensor finput = at::empty({0}, self.options());
  at::Tensor fgrad_input = at::empty({0}, self.options());
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

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::thnn_conv2d_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  at::Tensor finput = at::empty({0}, self.options());
  at::Tensor fgrad_input = at::empty({0}, self.options());
  at::Tensor output =
      NPUNativeFunctions::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
  return tuple<at::Tensor, at::Tensor, at::Tensor>(output, finput, fgrad_input);
}

tuple<at::Tensor&, at::Tensor&, at::Tensor&> NPUNativeFunctions::thnn_conv2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output,
    at::Tensor& finput,
    at::Tensor& fgrad_input) {
  NPUNativeFunctions::npu_conv2d_out(self, weight, bias, stride, padding, {1, 1}, 1, output);
  return tuple<at::Tensor&, at::Tensor&, at::Tensor&>(output, finput, fgrad_input);
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::thnn_conv2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    const at::Tensor& finput,
    const at::Tensor& fgrad_input,
    std::array<bool, 3> output_mask) {
  return NPUNativeFunctions::npu_convolution_backward(
      self, grad_output, weight, stride, padding, {1, 1}, 1, output_mask);
}

at::Tensor& NPUNativeFunctions::thnn_conv_depthwise2d_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::Tensor& out) {
  return at::thnn_conv_depthwise2d_forward_out(
      out, self, weight, kernel_size, bias, stride, padding, dilation);
}

at::Tensor NPUNativeFunctions::thnn_conv_depthwise2d(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward(
      self, weight, kernel_size, bias, stride, padding, dilation);
}

at::Tensor NPUNativeFunctions::_convolution(
    const at::Tensor& input_,
    const at::Tensor& weight_,
    const c10::optional<at::Tensor>& bias_,
    at::IntArrayRef stride_,
    at::IntArrayRef padding_,
    at::IntArrayRef dilation_,
    bool transposed_,
    at::IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {

    return _convolution_npu(input_,
        weight_,
        bias_,
        stride_,
        padding_,
        dilation_,
        transposed_,
        output_padding_,
        groups_,
        benchmark,
        deterministic,
        cudnn_enabled);
}

class NPUConvlutionFunction : public torch::autograd::Function<NPUConvlutionFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {

    ctx->saved_data["padding"] = padding;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["groups"] = groups;
    ctx->saved_data["bias_has_value"] = (bias.has_value() == true) ? bias.value().requires_grad() : false;

    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({input, weight});
    return convolution_kernel_npu(input, weight, bias, stride, padding, dilation, groups);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {

    auto padding = ctx->saved_data["padding"].toIntVector();
    auto stride = ctx->saved_data["stride"].toIntVector();
    auto dilation = ctx->saved_data["dilation"].toIntVector();
    auto groups = ctx->saved_data["groups"].toInt();
    auto bias_has_value = ctx->saved_data["bias_has_value"].toBool();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];

    std::array<bool, 3> grad_input_mask;
    grad_input_mask[0] = input.requires_grad();
    grad_input_mask[1] = weight.requires_grad();
    grad_input_mask[2] = bias_has_value;

    tuple<at::Tensor, at::Tensor, at::Tensor> result = NPUNativeFunctions::npu_convolution_backward(input,
        grad_outputs[0],
        weight,
        stride,
        padding,
        dilation,
        groups,
        grad_input_mask);
    tensor_list output = {std::get<0>(result),
        std::get<1>(result),
        std::get<2>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_convolution(const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
    c10::optional<at::Tensor> bias = c10::nullopt;
    if (bias_opt.has_value()) {
        if (bias_opt.value().defined()) {
            bias = bias_opt;
        }
    }

  return NPUConvlutionFunction::apply(input, weight, bias, stride, padding, dilation, groups);
}

} // namespace native
} // namespace at_npu
