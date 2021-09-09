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

#include <torch/script.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

using namespace torch::autograd;

namespace at {
namespace native {
using namespace at::native::npu;

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
    const c10::optional<Tensor>& bias,
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
    const c10::optional<Tensor>& bias,
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
    const c10::optional<Tensor>& bias,
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
    const c10::optional<Tensor>& bias_opt_,
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

  const Tensor& bias_ = c10::value_or_else(bias_opt_, [] {return Tensor();});
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
  if (is_depthwise(input, weight, groups, transposed)) {
    auto kernel_size = weight.sizes().slice(2);
    output = at::thnn_conv_depthwise2d(
        input.contiguous(),
        weight,
        kernel_size,
        bias_opt_,
        stride,
        padding,
        dilation);
  } else if (!transposed) {
    output = at::npu_convolution(
        input, weight, bias_opt_, stride, padding, dilation, groups);
  } else {
    output = at::npu_convolution_transpose(
        input, weight, bias_opt_, padding, output_padding, stride, dilation, groups);
  }

  if (k == 3) {
    output = view3d(output);
  }

  return output;
}

Tensor npu_convolution(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  int64_t dim = input.ndimension();

  Tensor output;
  if (dim == 4) {
    output =
        at::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
  }

  if (dim == 5) {
    output = at::npu_conv3d(
        input, weight, bias_opt, stride, padding, dilation, groups);
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

  return output;
}

Tensor _convolution_nogroup_npu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding) {
  Tensor output;
  if (!transposed) {
    output =
        at::npu_convolution(input, weight, bias_opt, stride, padding, dilation, 1);
  }

  return output;
}

Tensor thnn_conv2d_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  return std::get<0>(at::thnn_conv2d_forward(
      self, weight, kernel_size, bias, stride, padding));
}

Tensor& thnn_conv2d_out_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
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
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  Tensor finput = at::empty({0}, self.options());
  Tensor fgrad_input = at::empty({0}, self.options());
  Tensor output =
      at::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
  return tuple<Tensor, Tensor, Tensor>(output, finput, fgrad_input);
}

tuple<Tensor&, Tensor&, Tensor&> thnn_conv2d_forward_out_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output,
    Tensor& finput,
    Tensor& fgrad_input) {
  at::npu_conv2d_out(output, self, weight, bias, stride, padding, {1, 1}, 1);
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
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& out) {
  return at::thnn_conv_depthwise2d_forward_out(
      out, self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor thnn_conv_depthwise2d_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  return at::thnn_conv_depthwise2d_forward(
      self, weight, kernel_size, bias, stride, padding, dilation);
}

Tensor _convolution_new_npu(
    const Tensor& input_,
    const Tensor& weight_,
    const c10::optional<Tensor>& bias_,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
    ///TODO:
    return  _convolution_npu (input_,
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
  static Tensor forward(AutogradContext *ctx,
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

    ctx->saved_data["padding"] = padding;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["groups"] = groups;
    ctx->saved_data["bias_has_value"] = (bias.has_value() == true) ? bias.value().requires_grad() : false;

    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({input, weight});
    return at::native::npu_convolution(input, weight, bias, stride, padding, dilation, groups);
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

    tuple<Tensor, Tensor, Tensor> result
        = at::native::npu_convolution_backward(input,
               grad_outputs[0],
               weight,
               stride,
               padding,
               dilation,
               groups,
               grad_input_mask);
    tensor_list output={std::get<0>(result),
        std::get<1>(result),
        std::get<2>(result),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()};
    return output;
  }
};

Tensor npu_convolution_autograd(const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
    optional<Tensor> bias = c10::nullopt;
    if (bias_opt.has_value()) {
      if (bias_opt.value().defined()) {
          bias = bias_opt;
      }
    }
    
  return NPUConvlutionFunction::apply(input, weight, bias, stride, padding, dilation, groups);
}

TORCH_LIBRARY_IMPL(aten, AutogradNPU, m) {
  m.impl("npu_convolution", npu_convolution_autograd);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_convolution", TORCH_FN(npu_convolution));
  m.impl("npu_convolution_backward", TORCH_FN(npu_convolution_backward));
  m.impl("convolution", TORCH_FN(convolution_npu));
  m.impl("_convolution", TORCH_FN(_convolution_new_npu));
  m.impl("_convolution_nogroup", TORCH_FN(_convolution_nogroup_npu));
  m.impl("conv2d", TORCH_FN(conv2d_npu_));
  m.impl("conv_transpose2d.input", TORCH_FN(conv_transpose2d_npu_));

  m.impl("thnn_conv2d.out", TORCH_FN(thnn_conv2d_out_npu));
  m.impl("thnn_conv2d", TORCH_FN(thnn_conv2d_npu));
  m.impl("thnn_conv2d_forward.output", TORCH_FN(thnn_conv2d_forward_out_npu));
  m.impl("thnn_conv2d_forward", TORCH_FN(thnn_conv2d_forward_npu));
  m.impl("thnn_conv_depthwise2d.out", TORCH_FN(thnn_conv_depthwise2d_out_npu));
  m.impl("thnn_conv_depthwise2d", TORCH_FN(thnn_conv_depthwise2d_npu));
}

} // namespace native
} // namespace at
