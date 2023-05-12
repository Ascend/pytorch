#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
namespace {

at::Tensor& hardsigmoid_backward_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("HardSigmoidGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Run();

  return grad_input;
}
} // namespace

at::Tensor NPUNativeFunctions::hardsigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  // calculate the output result of the NPU
  hardsigmoid_backward_nocheck(grad_input, grad_output, self);

  return grad_input;
}

} // namespace native
} // namespace at_npu