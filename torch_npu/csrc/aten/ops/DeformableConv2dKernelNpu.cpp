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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_deformable_conv2dbk(
    const at::Tensor& input_ori,
    const at::Tensor& grad_output_ori,
    const at::Tensor& offset_out_ori,
    const at::Tensor& weight_ori,
    const at::Tensor& offset_ori,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated) {
  at::Tensor input = (input_ori.dtype() != at::kFloat) ?
      NPUNativeFunctions::npu_dtype_cast(input_ori, at::kFloat) : input_ori;
  at::Tensor grad_output = (grad_output_ori.dtype() != at::kFloat) ?
      NPUNativeFunctions::npu_dtype_cast(grad_output_ori, at::kFloat) : grad_output_ori;
  at::Tensor offset_out = (offset_out_ori.dtype() != at::kFloat) ?
      NPUNativeFunctions::npu_dtype_cast(offset_out_ori, at::kFloat) : offset_out_ori;
  at::Tensor weight = (weight_ori.dtype() != at::kFloat) ?
      NPUNativeFunctions::npu_dtype_cast(weight_ori, at::kFloat) : weight_ori;
  at::Tensor offset = (offset_ori.dtype() != at::kFloat) ?
      NPUNativeFunctions::npu_dtype_cast(offset_ori, at::kFloat) : offset_ori;
  // deformable_conv2d_backward includes conv2d_backward and DeformableOffsetsGrad
  c10::SmallVector<int64_t, SIZE> conv2dStride = array_to_small_vector(kernel_size);
  c10::SmallVector<int64_t, SIZE> conv2dPadding = {0, 0, 0, 0};
  c10::SmallVector<int64_t, SIZE> conv2dDilation = {1, 1};
  auto conv2dBackwardOutput = NPUNativeFunctions::npu_conv2d_backward(
      offset_out, grad_output, weight, conv2dStride, conv2dPadding, conv2dDilation, groups, {true, true, true});

  // DeformableOffsetsGrad's input 'grad' is the output[0] of conv2d_backward
  at::Tensor deformableOffsetsBackwardInput = std::get<0>(conv2dBackwardOutput);
  at::Tensor grad_weight = std::get<1>(conv2dBackwardOutput);
  at::Tensor grad_bias = std::get<2>(conv2dBackwardOutput);

  c10::SmallVector<int64_t, SHAPE_SIZE> in_perm = {0, 2, 3, 1};
  auto nhwc_grad_input_shape = transpose_npu_output_size(input, in_perm);
  auto nhwc_grad_offset_shape = transpose_npu_output_size(offset, in_perm);
  at::Tensor nhwc_grad_input = OpPreparation::ApplyTensorWithFormat(input, nhwc_grad_input_shape, ACL_FORMAT_NHWC);
  at::Tensor nhwc_grad_offset = OpPreparation::ApplyTensorWithFormat(
      offset, nhwc_grad_offset_shape, ACL_FORMAT_NHWC);

  auto trans_shape = transpose_npu_output_size(deformableOffsetsBackwardInput, in_perm);

  at::Tensor ori_deformableOffsetsBackwardInput = OpPreparation::CastBackToOriFormat(deformableOffsetsBackwardInput);
  at::Tensor ori_input = OpPreparation::CastBackToOriFormat(input);
  at::Tensor ori_offset = OpPreparation::CastBackToOriFormat(offset);

  at::Tensor nhwc_deformableOffsetsBackwardInput = NPUNativeFunctions::npu_transpose(
      ori_deformableOffsetsBackwardInput, in_perm, true);
  at::Tensor nhwc_input = NPUNativeFunctions::npu_transpose(ori_input, in_perm, true);
  at::Tensor nhwc_offset = NPUNativeFunctions::npu_transpose(ori_offset, in_perm, true);

  auto& nhwc_deformableOffsetsBackwardInput_desc = torch_npu::NPUBridge::GetNpuStorageImpl(
      nhwc_deformableOffsetsBackwardInput)->npu_desc_;
  auto& nhwc_input_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_input)->npu_desc_;
  auto& nhwc_offset_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_offset)->npu_desc_;

  nhwc_deformableOffsetsBackwardInput_desc.npu_format_ = ACL_FORMAT_NHWC;
  nhwc_deformableOffsetsBackwardInput_desc.origin_format_ = ACL_FORMAT_NHWC;
  nhwc_input_desc.npu_format_ = ACL_FORMAT_NHWC;
  nhwc_input_desc.origin_format_ = ACL_FORMAT_NHWC;
  nhwc_offset_desc.npu_format_ = ACL_FORMAT_NHWC;
  nhwc_offset_desc.origin_format_ = ACL_FORMAT_NHWC;

  c10::SmallVector<int64_t, SHAPE_SIZE> nhwc_strides = {stride[0], stride[2], stride[3], stride[1]};
  c10::SmallVector<int64_t, SHAPE_SIZE> nhwc_dilations = {dilation[0], dilation[2], dilation[3], dilation[1]};
  string dataFormat = "NHWC";
  OpCommand cmd;
  cmd.Name("DeformableOffsetsGrad")
      .Input(nhwc_deformableOffsetsBackwardInput, "grad")
      .Input(nhwc_input, "X")
      .Input(nhwc_offset, "offsets")
      .Output(nhwc_grad_input, "grad_X")
      .Output(nhwc_grad_offset, "grad_offsets")
      .Attr("strides", nhwc_strides)
      .Attr("pads", padding)
      .Attr("ksize", kernel_size)
      .Attr("dilations", nhwc_dilations)
      .Attr("data_format", dataFormat)
      .Attr("deformable_groups", deformable_groups)
      .Attr("modulated", modulated)
      .Run();
  c10::SmallVector<int64_t, SHAPE_SIZE> out_perm = {0, 3, 1, 2};
  nhwc_deformableOffsetsBackwardInput_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_deformableOffsetsBackwardInput_desc.origin_format_ = ACL_FORMAT_NCHW;
  nhwc_input_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_input_desc.origin_format_ = ACL_FORMAT_NCHW;
  nhwc_offset_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_offset_desc.origin_format_ = ACL_FORMAT_NCHW;
  auto& nhwc_grad_input_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_grad_input)->npu_desc_;
  auto& nhwc_grad_offset_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_grad_offset)->npu_desc_;
  nhwc_grad_input_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_grad_input_desc.origin_format_ = ACL_FORMAT_NCHW;
  nhwc_grad_offset_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_grad_offset_desc.origin_format_ = ACL_FORMAT_NCHW;
  at::Tensor grad_input = NPUNativeFunctions::npu_transpose(nhwc_grad_input, out_perm, true);
  at::Tensor grad_offset = NPUNativeFunctions::npu_transpose(nhwc_grad_offset, out_perm, true);

  return std::tie(grad_input, grad_weight, grad_offset, grad_bias);
}

tuple<at::Tensor, at::Tensor> deformable_conv2d_npu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  at::Tensor bias_fp32 = (bias.defined() && bias.dtype() != at::kFloat) ?
      NPUNativeFunctions::npu_dtype_cast(bias, at::kFloat) : bias;
  auto outputSize = deformable_conv2d_npu_output_size(
      input, weight, offset, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);

  /*
  * DeformableOffsets and DeformableOffsetsGrad only support NHWC and don't support binary.
  * FE will insert Transpose before DeformableOffsets and DeformableOffsetsGrad.
  * In order to allow Transpose into binary,
  * Transpose is called explicitly in adapter.
  */
  c10::SmallVector<int64_t, SIZE> nhwc_deformableOffsetsOutput_shape = {
      outputSize[0], outputSize[2], outputSize[3], outputSize[1]};
  at::Tensor nhwc_deformableOffsetsOutput = OpPreparation::ApplyTensorWithFormat(
      nhwc_deformableOffsetsOutput_shape, input.options(), ACL_FORMAT_NHWC);
  c10::SmallVector<int64_t, SHAPE_SIZE> in_perm = {0, 2, 3, 1};
  at::Tensor ori_input = OpPreparation::CastBackToOriFormat(input);
  at::Tensor ori_offset = OpPreparation::CastBackToOriFormat(offset);
  at::Tensor nhwc_input = NPUNativeFunctions::npu_transpose(ori_input, in_perm, true);
  at::Tensor nhwc_offset = NPUNativeFunctions::npu_transpose(ori_offset, in_perm, true);

  auto& nhwc_input_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_input)->npu_desc_;
  auto& nhwc_offset_desc = torch_npu::NPUBridge::GetNpuStorageImpl(nhwc_offset)->npu_desc_;

  nhwc_input_desc.npu_format_ = ACL_FORMAT_NHWC;
  nhwc_input_desc.origin_format_ = ACL_FORMAT_NHWC;
  nhwc_offset_desc.npu_format_ = ACL_FORMAT_NHWC;
  nhwc_offset_desc.origin_format_ = ACL_FORMAT_NHWC;

  c10::SmallVector<int64_t, SHAPE_SIZE> nhwc_strides = {stride[0], stride[2], stride[3], stride[1]};
  c10::SmallVector<int64_t, SHAPE_SIZE> nhwc_dilations = {dilation[0], dilation[2], dilation[3], dilation[1]};
  string dataFormat = "NHWC";
  OpCommand cmd;
  cmd.Name("DeformableOffsets")
      .Input(nhwc_input, "X")
      .Input(nhwc_offset, "offsets")
      .Output(nhwc_deformableOffsetsOutput, "y")
      .Attr("ksize", kernel_size)
      .Attr("strides", nhwc_strides)
      .Attr("pads", padding)
      .Attr("dilations", nhwc_dilations)
      .Attr("deformable_groups", deformable_groups)
      .Attr("data_format", dataFormat)
      .Attr("modulated", modulated)
      .Run();

  c10::SmallVector<int64_t, SHAPE_SIZE> out_perm = {0, 3, 1, 2};
  nhwc_input_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_input_desc.origin_format_ = ACL_FORMAT_NCHW;
  nhwc_offset_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_offset_desc.origin_format_ = ACL_FORMAT_NCHW;
  auto& nhwc_deformableOffsetsOutput_desc = torch_npu::NPUBridge::GetNpuStorageImpl(
      nhwc_deformableOffsetsOutput)->npu_desc_;
  nhwc_deformableOffsetsOutput_desc.npu_format_ = ACL_FORMAT_NCHW;
  nhwc_deformableOffsetsOutput_desc.origin_format_ = ACL_FORMAT_NCHW;
  at::Tensor deformableOffsetsOutput = NPUNativeFunctions::npu_transpose(nhwc_deformableOffsetsOutput, out_perm, true);

  c10::SmallVector<int64_t, SIZE> conv2dStride = array_to_small_vector(kernel_size);
  c10::SmallVector<int64_t, SIZE> conv2dPadding = {0, 0, 0, 0};
  c10::SmallVector<int64_t, SIZE> conv2dDilation = {1, 1};
  at::Tensor conv2dOutput = NPUNativeFunctions::npu_conv2d(
      deformableOffsetsOutput, weight, bias_fp32, conv2dStride, conv2dPadding, conv2dDilation, groups);

  return std::tie(conv2dOutput, deformableOffsetsOutput);
}

class NPUDeformableConv2dFunction : public torch::autograd::Function<NPUDeformableConv2dFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& input_ori,
    const at::Tensor& weight_ori,
    const at::Tensor& offset_ori,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated) {
    at::Tensor input = (input_ori.dtype() != at::kFloat) ?
        NPUNativeFunctions::npu_dtype_cast(input_ori, at::kFloat) : input_ori;
    at::Tensor weight = (weight_ori.dtype() != at::kFloat) ?
        NPUNativeFunctions::npu_dtype_cast(weight_ori, at::kFloat) : weight_ori;
    at::Tensor offset = (offset_ori.dtype() != at::kFloat) ?
        NPUNativeFunctions::npu_dtype_cast(offset_ori, at::kFloat) : offset_ori;
    ctx->saved_data["kernel_size"] = kernel_size;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["groups"] = groups;
    ctx->saved_data["deformable_groups"] = deformable_groups;
    ctx->saved_data["modulated"] = modulated;
    ctx->saved_data["bias_has_value"] = (bias_opt.has_value() == true) ? bias_opt.value().requires_grad() : false;

    at::AutoNonVariableTypeMode g;
    auto result = deformable_conv2d_npu(
        input, weight, offset, bias_opt, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
    auto result1 = std::get<1>(result);
    ctx->save_for_backward({input, weight, offset, result1});
    tensor_list result_list = {std::get<0>(result), result1};
    return result_list;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
    auto stride = ctx->saved_data["stride"].toIntVector();
    auto padding = ctx->saved_data["padding"].toIntVector();
    auto dilation = ctx->saved_data["dilation"].toIntVector();
    auto groups = ctx->saved_data["groups"].toInt();
    auto deformable_groups = ctx->saved_data["deformable_groups"].toInt();
    auto modulated = ctx->saved_data["modulated"].toBool();
    auto bias_has_value = ctx->saved_data["bias_has_value"].toBool();

    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto offset = saved[2];
    auto offset_out = saved[3];

    tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> result = NPUNativeFunctions::npu_deformable_conv2dbk(
        input, grad_outputs[0], offset_out, weight, offset, kernel_size,
        stride, padding, dilation, groups, deformable_groups, modulated);

    tensor_list output;
    if (bias_has_value) {
      output = {std::get<0>(result),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    } else {
      output = {std::get<0>(result),
        std::get<1>(result),
        std::get<2>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    }
    return output;
  }
};

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_deformable_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated) {
  auto result = NPUDeformableConv2dFunction::apply(
      input, weight, offset, bias_opt, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
  std::tuple<at::Tensor, at::Tensor> output(result[0], result[1]);
  return output;
}

} // namespace native
} // namespace at_npu
