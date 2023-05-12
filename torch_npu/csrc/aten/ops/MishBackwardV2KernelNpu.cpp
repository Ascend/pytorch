#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& mish_backward_out_npu_nocheck(const at::Tensor& grad_output, const at::Tensor& self, at::Tensor& grad_input) {
  OpCommand cmd;
  cmd.Name("MishGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)    
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::mish_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  mish_backward_out_npu_nocheck(grad_output, self, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
