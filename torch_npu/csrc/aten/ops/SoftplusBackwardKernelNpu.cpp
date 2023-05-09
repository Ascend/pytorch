#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& softplus_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold) {
  OpCommand cmd;
  cmd.Name("SoftplusV2Grad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Attr("beta", beta)
      .Attr("threshold", threshold)
      .Run();

    return grad_input;
}

at::Tensor& NPUNativeFunctions::softplus_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& grad_input) {
  OpPreparation::CheckOut(
      {self},
      grad_input,
      self);
  return softplus_backward_out_nocheck(grad_input, grad_output, self, beta, threshold);
}

} // namespace native
} // namespace at_npu