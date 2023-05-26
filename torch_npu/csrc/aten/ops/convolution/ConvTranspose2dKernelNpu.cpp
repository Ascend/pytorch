#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& conv_transpose2d_out_npu(
    at::Tensor& result,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  c10::SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> outputpadding = {0, 0, output_padding[0], output_padding[1]};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  string sizeName = "input_size";

  c10::SmallVector<int64_t, N> sizeVec = array_to_small_vector(result.sizes());
  OpCommand cmd;
  cmd.Name("Conv2DTranspose")
      .Input(sizeVec, at::kInt)
      .Input(input, "x", ACL_FORMAT_NCHW)
      .Input(weight, "filter", ACL_FORMAT_NCHW);
  if (bias.defined()){
    cmd.Input(bias);
  }
  cmd.Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("pads", paddings)
      .Attr("output_padding", outputpadding)
      .Attr("strides", stridesSize)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::npu_conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {

  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});

  // calculate the output size
  auto outputSize = conv_transpose2d_npu_output_size(
      input, weight, bias, padding, output_padding, stride, dilation, groups);

  // construct the output tensor of the NPU
  int64_t result_format = input.dtype() == at::kHalf ? ACL_FORMAT_NC1HWC0 : ACL_FORMAT_ND;
  at::Tensor result =
      OpPreparation::ApplyTensorWithFormat(outputSize, input.options(), result_format);

  // calculate the output result of the NPU
  conv_transpose2d_out_npu(
      result, input, weight, bias, padding, output_padding, stride, dilation, groups);

  return result;
}

} // namespace native
} // namespace at_npu
