#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::hardtanh_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val,
    at::Tensor& grad_input) {
  OpPreparation::CheckMemory({grad_output, self}, {grad_input});
  OpCommand cmd;
  cmd.Name("HardtanhGrad")
      .Input(self)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("max_val", max_val)
      .Attr("min_val", min_val)
      .Run();
  return grad_input;
}

at::Tensor NPUNativeFunctions::hardtanh_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(self);
  hardtanh_backward_out(grad_output, self, min_val, max_val, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
