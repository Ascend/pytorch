#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& hardshrink_backward_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar lambd) {
  OpCommand cmd;
  cmd.Name("HardShrinkGrad")
      .Input(grad_output)
      .Input(self)
      .Attr("lambd", lambd)
      .Output(grad_input)
      .Run();

  return grad_input;
}

at::Tensor NPUNativeFunctions::hardshrink_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& lambd) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  hardshrink_backward_nocheck(grad_input, grad_output, self, lambd);

  return grad_input;
}

} // namespace native
} // namespace at_npu