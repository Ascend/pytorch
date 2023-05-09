#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor _nnpack_spatial_convolution_output_npu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& result) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[0], padding[0]};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[0]};
  if (padding.size() != 1) {
    paddings[2] = padding[1];
    paddings[3] = padding[1];
  }
  if (stride.size() != 1) {
    stridesSize[3] = stride[1];
  }
  c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
  string dataFormat = "NCHW";
  int64_t groups = 1;

  OpCommand cmd;
  cmd.Name("Conv2D")
      .Input(input)
      .Input(weight)
      .Input(bias)
      .Output(result)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::_nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = nnpack_spatial_convolution_npu_output_size(
      input, weight, padding, stride);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(outputSize, input.options(), ACL_FORMAT_NC1HWC0);
  _nnpack_spatial_convolution_output_npu(
      input, weight, bias, padding, stride, result);
  return result;
}

} // namespace native
} // namespace at_npu