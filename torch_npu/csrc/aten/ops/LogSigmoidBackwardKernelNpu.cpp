#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::log_sigmoid_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& buffer,
    at::Tensor& grad_input) {
  OpPreparation::CheckMemory({grad_output, self, buffer}, {grad_input});
  OpCommand cmd;
  cmd.Name("LogSigmoidGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::log_sigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& buffer) {
  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  // calculate the output result of the NPU
  log_sigmoid_backward_out(grad_output, self, buffer, grad_input);

  return grad_input;
}

} // namespace native
} // namespace at_npu