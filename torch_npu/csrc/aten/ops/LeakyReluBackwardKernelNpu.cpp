#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor leaky_relu_backward_out_npu(
    at::Tensor result,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar negval,
    bool is_result) {
  OpCommand cmd;
  cmd.Name("LeakyReluGrad")
      .Input(grad_output)
      .Input(self)
      .Output(result)
      .Attr("negative_slope", negval)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::leaky_relu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negval,
    bool is_result) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  leaky_relu_backward_out_npu(result, grad_output, self, negval, is_result);
  return result;
  
}
} // namespace native
} // namespace at_npu