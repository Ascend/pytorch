#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& hardswish_backward_nocheck(
    at::Tensor& grad_input, 
    const at::Tensor& grad_output, 
    const at::Tensor& self) {
  
  OpCommand cmd;
  cmd.Name("HardSwishGrad")
     .Input(grad_output)
     .Input(self)
     .Output(grad_input)
     .Run();

  return grad_input;
}

at::Tensor NPUNativeFunctions::hardswish_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  
  return hardswish_backward_nocheck(grad_input, grad_output, self);
}

} // namespace native
} // namespace at_npu