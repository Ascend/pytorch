#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor& binary_cross_entropy_backward_out_npu_nocheck(
    at::Tensor& gradInput,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction) {
  at::Tensor weightTensor = weight.defined() ? weight :
              at::ones(self.sizes(), self.options());
  std::string reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpCommand cmd;
  cmd.Name("BinaryCrossEntropyGrad")
     .Input(self)
     .Input(target)
     .Input(grad_output)
     .Input(weightTensor)
     .Output(gradInput)
     .Attr("reduction", reductionStr)
     .Run();
  return gradInput;
}

at::Tensor& NPUNativeFunctions::binary_cross_entropy_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    at::Tensor& gradInput) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  binary_cross_entropy_backward_out_npu_nocheck(gradInput, grad_output, self, target, weight, reduction);
  return gradInput;
}

at::Tensor NPUNativeFunctions::binary_cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor gradInput = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  binary_cross_entropy_backward_out_npu_nocheck(gradInput, grad_output, self, target, weight, reduction);
  return gradInput;
}
} // namespace native
} // namespace at_npu