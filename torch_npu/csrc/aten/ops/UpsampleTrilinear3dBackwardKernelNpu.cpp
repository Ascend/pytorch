#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> upsample_trilinear3d_backward_outputsize_npu(
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_depth = input_size[2];
  int64_t input_height = input_size[3];
  int64_t input_width = input_size[4];

  at::SmallVector<int64_t, SIZE> outputSize = 
    {nbatch, channels, input_depth, input_height, input_width};
  return outputSize;
}

at::Tensor& upsample_trilinear3d_backward_npu_nocheck(
    at::Tensor& out,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  OpCommand cmd;
  cmd.Name("UpsampleTrilinear3dGrad")
    .Input(grad_output)
    .Output(out)
    .Attr("input_size", input_size)
    .Attr("output_size", output_size)
    .Attr("align_corners", align_corners)
    .Run();

  return out;
}

at::Tensor& NPUNativeFunctions::upsample_trilinear3d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {
  auto outputsize = upsample_trilinear3d_backward_outputsize_npu(
      output_size, input_size, scales_d, scales_h, scales_w);
  OpPreparation::CheckOut({grad_output}, grad_input, grad_output, outputsize);
  if (!NpuUtils::check_match(&grad_input)) {
    auto contiguous_out = NpuUtils::format_contiguous(grad_input);
    upsample_trilinear3d_backward_npu_nocheck(
        grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    NpuUtils::format_fresh_view(grad_input, contiguous_out);   
  } else {
    upsample_trilinear3d_backward_npu_nocheck(
        grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  }
  return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_trilinear3d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto outputsize = upsample_trilinear3d_backward_outputsize_npu(
      output_size, input_size, scales_d, scales_h, scales_w);
  at::Tensor result = OpPreparation::ApplyTensor(grad_output, outputsize);
  upsample_trilinear3d_backward_npu_nocheck(
      result, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  return result;
}

} // namespace native
} // namespace at_npu
