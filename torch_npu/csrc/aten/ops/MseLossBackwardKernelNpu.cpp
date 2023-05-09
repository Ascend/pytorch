#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::mse_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  if (self.numel() == 0 || target.numel() == 0) {
    grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    return grad_input;
  }

  OpPreparation::CheckMemory({grad_output, self, target}, {grad_input});
  string reductionStr(CalcuOpUtil::GetReductionStr(reduction));

  OpCommand cmd;
  cmd.Name("MseLossGrad")
      .Input(self)
      .Input(target)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::mse_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  auto grad_out = grad_output.contiguous();
  if (grad_out.dim() == 0) {
    grad_out.view(1);
  }
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);

  NPUNativeFunctions::mse_loss_backward_out(
      grad_out,
      self,
      target,
      reduction,
      grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu