#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& softplus_out_nocheck(
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("SoftplusV2")
      .Input(self)
      .Output(result)
      .Attr("beta", beta)
      .Attr("threshold", threshold)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::softplus_out(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  return softplus_out_nocheck(self, beta, threshold, result);
}

at::Tensor NPUNativeFunctions::softplus(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold) {
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize, self.options(), self);
  softplus_out_nocheck(self, beta, threshold, result);
  return result;
}

} // namespace native
} // namespace at_npu