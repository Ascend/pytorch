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
  return at_npu::key::isDeviceTensor(input) && !transposed && input.ndimension() == 4 &&
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
    //  output = at::slow_conv3d(
       //   input, weight, kernel_size, bias_opt, stride, padding);
    } else {
     // output = NPUNativeFunctions::npu_conv3d(
         // input, weight, bias_opt, stride, padding, dilation, groups);
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
   /* output = NPUNativeFunctions::npu_conv3d_backward(
        input,
        grad,
        weight,
        stride,
        padding,
        dilation,
        groups,
        grad_input_mask);*/
  }
  // Note:weight.grad should be equal weight
  if (std::get<1>(output).defined()) {
    std::get<1>(output) = NPUNativeFunctions::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
  }
  return output;
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

at::Tensor NPUNativeFunctions::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt,
    c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation,
    bool transposed, c10::IntArrayRef output_padding, int64_t groups) {

  return NPUNativeFunctions::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::convolution_backward_overrideable(
        const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight,
        c10::IntArrayRef stride, c10::IntArrayRef padding,
        c10::IntArrayRef dilation, bool transposed, c10::IntArrayRef output_padding,
        int64_t groups, std::array<bool,3> output_mask) {

  return NPUNativeFunctions::npu_convolution_backward(input,
     grad_output,
     weight,
     stride,
     padding,
     dilation,
     groups,
     output_mask);
}

} // namespace native
} // namespace at_npu
