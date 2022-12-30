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
#include <ATen/native/ConvUtils.h>
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

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct ConvParams {
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;
  bool allow_tf32;

  bool is_dilated() const;
  bool is_output_padding_neg() const;
  bool is_padding_neg() const;
  bool is_stride_nonpos() const;
  void view1d_as_2d();
};

auto ConvParams::is_dilated() const -> bool {
  bool is_dilated = false;
  for (auto d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

auto ConvParams::is_output_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (auto p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (auto p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_stride_nonpos() const -> bool {
  bool is_nonpos = false;
  for (auto s : stride) {
    is_nonpos |= (s <= 0);
  }
  return is_nonpos;
}

auto ConvParams::view1d_as_2d() -> void {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

inline std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

static inline std::vector<int64_t> conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef weight_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation = at::IntArrayRef()) {
  bool has_dilation = dilation.size() > 0;
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (const auto d : c10::irange(2, dim)) {
    auto dilation_ = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilation_ * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

static inline std::vector<int64_t> conv_input_size(
    at::IntArrayRef output_size,
    at::IntArrayRef weight_size,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (const auto d : c10::irange(2, dim)) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
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

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 4,
      "expected 4D tensor, got tensor with ", tensor.ndimension(),
      " dimensions instead");
  return tensor.squeeze(2);
}

static std::tuple<at::Tensor, bool> batchify(
    const at::Tensor& input,
    const int64_t num_spatial_dims,
    const std::string& func_name) {
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = (input.dim() == dim_count_batch);
  TORCH_CHECK(input.dim() == dim_count_no_batch || is_batched,
      "Expected ", dim_count_no_batch, "D (unbatched) or ", dim_count_batch,
      "D (batched) input to ", func_name, ", but got input of size: ", input.sizes());
  return std::make_tuple(is_batched ? input : input.unsqueeze(0), is_batched);
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

static void check_shape_forward(const at::Tensor& input,
                                const c10::IntArrayRef& weight_sizes,
                                const at::Tensor& bias,
                                const ConvParams& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight_sizes.size();
  int64_t groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");

  TORCH_CHECK(weight_dim == k,
      "Expected ", weight_dim, "-dimensional input for ", weight_dim,
      "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
      input.sizes(), " instead");
  TORCH_CHECK(weight_sizes[0] >= groups,
      "Given groups=", groups, ", expected weight to be at least ", groups,
      " at dimension 0, but got weight of size ", weight_sizes, " instead");
  TORCH_CHECK(weight_sizes[0] % groups == 0,
      "Given groups=", groups, ", expected weight to be divisible by ",
      groups, " at dimension 0, but got weight of size [", weight_sizes,
      "] instead");

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
      "Given groups=", groups, ", weight of size ", weight_sizes,
      ", expected input", input.sizes(), " to have ",
      (weight_sizes[1] * groups), " channels, but got ", input.size(1),
      " channels instead");

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
        ", but got bias of size ", bias.sizes(), " instead");

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(input.size(i) + 2 * padding[i-2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      AT_ERROR("Calculated padded input size per channel: (", input_ss.str(), "). "
               "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    TORCH_CHECK(input.size(1) == weight_sizes[0],
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected input", input.sizes(), " to have ", weight_sizes[0],
        " channels, but got ", input.size(1), " channels instead");
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=", transposed, ", weight of size ", weight_sizes,
        ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
        ", but got bias of size ", bias.sizes(), " instead");
  }
}

static void check_shape_backward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const ConvParams& params) {
  check_shape_forward(input, weight_sizes, at::Tensor(), params);
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
    const at::Tensor& input_,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  at::Tensor input;
  bool is_batched;
  std::tie(input, is_batched) = batchify(input_, 3, "conv_transpose3d");
  auto output = at::convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  return is_batched ? output : output.squeeze(0);
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
  int64_t dim = input.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  at::Tensor output;
  if (dim == 4) {
    if (transposed) {
      output = 
        NPUNativeFunctions::npu_conv_transpose2d(input, weight, bias_opt, padding, output_padding, stride, dilation, groups);
    }
    else {
      output =
        NPUNativeFunctions::npu_conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
    }
    
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

at::native::ConvBackend select_conv_backend(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::IntArrayRef> bias_sizes_opt,
    const bool need_backward,
    const ConvParams& params) {

  // don't send empty inputs through backends
  if (input.size(0) == 0 || input.size(1) == 0) {
    return at::native::ConvBackend::Empty;
  } else if (input.numel() == 0) {
    TORCH_CHECK(false, "Only zero batch or zero channel inputs are supported, but got input shape: ", input.sizes());
  }

  if (input.device().type() == at_npu::key::NativeDeviceType) {
    // backends without support for groups
    if (params.transposed) {
      if (input.ndimension() == 4) {
        return at::native::ConvBackend::SlowTranspose2d;
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::SlowTranspose3d;
      } else {
        AT_ERROR("unsupport");
      }
    } else {  /* Not transposed */
      if (input.ndimension() == 4) {
        if (params.is_dilated()) {
          return at::native::ConvBackend::SlowDilated2d;
        } else { 
          return at::native::ConvBackend::Slow2d;
        }
      } else if (input.ndimension() == 5) { 
        return at::native::ConvBackend::Slow3d;
      } else {
        AT_ERROR("unsupport");
      }
    }
  } else {
    // Only reach here when input is backend with out-of-source implementation.
    return at::native::ConvBackend::Overrideable;
  }

  // Error out if no suitable backend was found.
  AT_ERROR("unsupported ConvNd parameters");
}

// Selects a backend for convolution based on the inputs and params.
at::native::ConvBackend select_conv_backend(
    const at::Tensor& input_r, const at::Tensor& weight_r, const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride_, at::IntArrayRef padding_, at::IntArrayRef dilation_,
    bool transposed_, at::IntArrayRef output_padding_, int64_t groups_) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  auto& ctx = at::globalContext();
  auto k = weight_r.ndimension();
  int64_t dim = k - 2;
  ConvParams params;
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;

  auto input = input_r;
  auto weight = weight_r;
  check_shape_forward(input, weight.sizes(), bias, params);

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial input.
  if (k == 3 && !input.is_mkldnn()) {
    // avoid accidentally going through NHWC for permuted 3d input.
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  auto bias_sizes_opt = bias.defined() ? c10::optional<at::IntArrayRef>(bias.sizes()) : c10::nullopt;
  bool need_backward = GradMode::is_enabled() &&
      (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
  return select_conv_backend(input, weight, bias_sizes_opt, need_backward, params);
}

tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> grad_input_mask) {
  int64_t dim = input.ndimension();

  tuple<at::Tensor, at::Tensor, at::Tensor> output;
  if (dim == 4) {
    output = NPUNativeFunctions::npu_conv_transpose2d_backward(
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
    output = NPUNativeFunctions::npu_conv_transpose3d_backward(
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
    std::get<1>(output) = NPUNativeFunctions::npu_dtype_cast(std::get<1>(output), weight.scalar_type());
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::convolution_backward(
    const at::Tensor& grad_output_,
    const at::Tensor& input_,
    const at::Tensor& weight_,
    const c10::optional<at::IntArrayRef> bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  auto grad_output = grad_output_;
  auto input = input_;
  auto weight = weight_;

  auto k = weight.ndimension();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  auto& ctx = at::globalContext();
  ConvParams params;
  params.stride = expand_param_if_needed(stride, "stride", dim);
  params.padding = expand_param_if_needed(padding, "padding", dim);
  params.dilation = expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding = expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;

  // Validate inputs.
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(input.dim() == grad_output.dim(),
      "Expected input and grad_output to have the same number of dimensions, but got: ",
      input.dim(), " and ", grad_output.dim());

  // output_padding is only supported for transposed convolutions
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(pad == 0, "output_padding is not supported for non-transposed convolutions; got: ",
          params.output_padding);
    }
  }

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial input.
  if (k == 3) {
    // avoid accidentally going through NHWC for permuted 3d input.
    params.view1d_as_2d();
    grad_output = view4d(grad_output);
    input = view4d(input);
    weight = view4d(weight);
  }

  // Select appropriate backend to use.
  at::native::ConvBackend backend = select_conv_backend(input, weight, bias_sizes_opt, true, params);

  // Call the backend.
  at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
  auto kernel_size = weight.sizes().slice(2);

  switch(backend) {
    case at::native::ConvBackend::Empty:
      if (output_mask[0]) {
        backend_grad_input = at::zeros_like(input);
      }
      if (output_mask[1]) {
        backend_grad_weight = at::zeros_like(weight);
      }
      if (output_mask[2]) {
        backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
      }
      break;
    case at::native::ConvBackend::Overrideable:
      // Only reach here when input is backend with out-of-source implementation.
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        at::convolution_backward_overrideable(grad_output, input, weight, params.stride, params.padding,
            params.dilation, params.transposed, params.output_padding, params.groups, output_mask);
      break;
    case at::native::ConvBackend::Slow3d:
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        NPUNativeFunctions::npu_conv3d_backward(
            input, grad_output, weight, params.stride, 
            params.padding, params.dilation, params.groups, output_mask);
      break;
    // Handle backends that don't natively support groups > 1.
    case at::native::ConvBackend::NnpackSpatial:
    case at::native::ConvBackend::Slow2d:
    case at::native::ConvBackend::SlowDilated2d:
    case at::native::ConvBackend::SlowDilated3d:
    case at::native::ConvBackend::SlowTranspose2d:
    case at::native::ConvBackend::SlowTranspose3d:
    {
      if (!params.transposed) {
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = NPUNativeFunctions::npu_convolution_backward(input,
          grad_output,
          weight,
          params.stride,
          params.padding,
          params.dilation,
          params.groups,
          output_mask);
      } else {
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) = npu_convolution_transpose_backward(input,
          grad_output,
          weight,
          params.padding,
          params.output_padding,
          params.stride,
          params.dilation,
          params.groups,
          output_mask);
      }
      break;
    }
    // Backward is not supported for these backends.
    case at::native::ConvBackend::Winograd3x3Depthwise:
      TORCH_CHECK(false, "Backward is not supported for depthwise 3x3 winograd");
      break;
    case at::native::ConvBackend::Xnnpack2d:
      TORCH_CHECK(false, "Backward is not supported for xnnpack");
      break;
  }

  // Convert 2D inputs back to 1D for backends that don't natively support 1D
  // spatial inputs.
  if (output_mask[0]) {
    if (k == 3) {
      backend_grad_input = view3d(backend_grad_input);
    }
  }
  if (output_mask[1]) {
    if (k == 3) {
      backend_grad_weight = view3d(backend_grad_weight);
    }
  }
  if (output_mask[2]) {
    if (!backend_grad_bias.defined()) {
      // Calculate bias gradients outside of the backend for those that don't support it.
      backend_grad_bias = grad_output.sum((dim == 3) ? at::IntArrayRef{0, 2, 3, 4} : at::IntArrayRef{0, 2, 3});
    }
  }

  return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}

at::Tensor NPUNativeFunctions::_slow_conv2d_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  at::Tensor output =
      NPUNativeFunctions::npu_convolution(self, weight, bias, stride, padding, {1, 1}, 1);
  return output;
}

at::Tensor& NPUNativeFunctions::_slow_conv2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
  NPUNativeFunctions::npu_conv2d_out(self, weight, bias, stride, padding, {1, 1}, 1, output);
  return output;
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::_slow_conv2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  return NPUNativeFunctions::npu_convolution_backward(
      self, grad_output, weight, stride, padding, {1, 1}, 1, output_mask);
}

} // namespace native
} // namespace at_npu
