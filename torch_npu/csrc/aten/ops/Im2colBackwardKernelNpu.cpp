#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& im2col_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  at::Tensor gradOutput = grad_output;
  gradOutput = gradOutput.view({
    grad_output.size(0),
    grad_output.size(1) / (kernel_size[0] * kernel_size[1]),
    kernel_size[0] * kernel_size[1],
    grad_output.size(2)});
  c10::SmallVector<int64_t, N> inputSize = {input_size[0], input_size[1]};
  c10::SmallVector<int64_t, N> kernelSize = {kernel_size[0], kernel_size[1]};
  c10::SmallVector<int64_t, N> dilations = {dilation[0], dilation[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1]};
  c10::SmallVector<int64_t, N> stridesSize = {stride[0], stride[1]};
  OpCommand cmd;
  cmd.Name("Col2im")
      .Input(gradOutput, "x", ACL_FORMAT_NCHW)
      .Input(inputSize, at::kInt)
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("kernel_size", kernelSize)
      .Attr("dilation", dilations)
      .Attr("padding", paddings)
      .Attr("stride", stridesSize)
      .Run();
  return grad_input;
}

at::Tensor& NPUNativeFunctions::col2im_out(
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& grad_input){
  at::Tensor grad_output_cp = grad_output.dim() == 2 ? at::unsqueeze(grad_output, 0) : grad_output;
  c10::SmallVector<int64_t, SIZE> output_size = {
    grad_output_cp.size(0),
    grad_output_cp.size(1) / (kernel_size[0] * kernel_size[1]),
    input_size[0],
    input_size[1]
    };
  OpPreparation::CheckOut(
      {grad_output_cp},
      grad_input,
      grad_output_cp,
      output_size);
  OpPipeWithDefinedOut pipe;
  pipe.CheckMemory({grad_output_cp}, {grad_input})
      .Func([&grad_output_cp, &input_size, &kernel_size, &dilation, &padding, &stride](at::Tensor& grad_input) {
        im2col_backward_out_npu_nocheck(
            grad_input, grad_output_cp, input_size, kernel_size, dilation, padding, stride);})
      .Call(grad_input);
  if (grad_output.dim() == 2) {
    grad_input = at::squeeze(grad_input, 0);
  }
  return grad_input;
}

at::Tensor NPUNativeFunctions::col2im(
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  at::Tensor grad_output_cp = grad_output.dim() == 2 ? at::unsqueeze(grad_output, 0) : grad_output;
  c10::SmallVector<int64_t, SIZE> output_size = {
    grad_output_cp.size(0),
    grad_output_cp.size(1) / (kernel_size[0] * kernel_size[1]),
    input_size[0],
    input_size[1]
    };
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output_cp, output_size);
  im2col_backward_out_npu_nocheck(
      grad_input,
      grad_output_cp,
      input_size,
      kernel_size,
      dilation,
      padding,
      stride);
  if (grad_output.dim() == 2) {
    grad_input = at::squeeze(grad_input, 0);
  }
  return grad_input;
}

} // namespace native
} // namespace at_npu
