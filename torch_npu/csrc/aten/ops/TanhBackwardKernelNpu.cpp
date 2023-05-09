#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& tanh_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("TanhGrad")
    .Input(self)
    .Input(grad_output)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::tanh_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut({grad_output, self}, result, self);
  tanh_backward_out_npu_nocheck(result, grad_output, self);
  return result;
}

at::Tensor NPUNativeFunctions::tanh_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  tanh_backward_out_npu_nocheck(result, grad_output, self);

  return result;
}

} // namespace native
} // namespace at_npu
