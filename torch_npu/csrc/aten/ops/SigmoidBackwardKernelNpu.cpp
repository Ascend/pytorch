#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& sigmoid_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& output) {
  auto unified_result = OpPreparation::binary_op_check(result, output, grad_output, true);
  OpCommand cmd;
  cmd.Name("SigmoidGrad")
    .Expect(unified_result)
    .Input(output)
    .Input(grad_output)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::sigmoid_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    at::Tensor& result) {
  OpPreparation::CheckOut({grad_output, output}, result, grad_output);
  sigmoid_backward_out_npu_nocheck(result, grad_output, output);
  return result;
}

at::Tensor NPUNativeFunctions::sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& output) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  sigmoid_backward_out_npu_nocheck(grad_input, grad_output, output);

  return grad_input;
}

} // namespace native
} // namespace at_npu