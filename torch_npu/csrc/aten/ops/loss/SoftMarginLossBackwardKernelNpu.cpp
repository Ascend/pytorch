#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::soft_margin_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  string reductionStr(CalcuOpUtil::GetReductionStr(reduction));

  OpPreparation::CheckMemory({grad_output, input, target}, {grad_input});
  OpCommand cmd;
  cmd.Name("SoftMarginLossGrad")
      .Input(input)
      .Input(target)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::soft_margin_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(input);

  NPUNativeFunctions::soft_margin_loss_backward_out(
      grad_output, input, target, reduction, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu