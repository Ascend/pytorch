#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& gelu_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad,
    const at::Tensor& self) {
  at::Tensor unused = grad;
  OpCommand cmd;
  cmd.Name("GeluGrad")
     .Input(grad)
     .Input(self)
     .Input(unused)
     .Output(grad_input)
     .Run();

  return grad_input;
}

at::Tensor NPUNativeFunctions::gelu_backward(
    const at::Tensor& grad, 
    const at::Tensor& self,
    c10::string_view approximate) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  gelu_backward_out_npu_nocheck(grad_input, grad, self);
  return grad_input;
}

} // namespace native
} // namespace at_npu