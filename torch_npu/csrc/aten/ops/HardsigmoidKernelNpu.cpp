#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& hardsigmoid_out_nocheck(
    const at::Tensor& self,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("HardSigmoid")
    .Input(self)
    .Output(result)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::hardsigmoid_out(
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor checkResult = hardsigmoid_out_nocheck(self, contiguousResult);
    NpuUtils::format_fresh_view(result, checkResult);
  } else {
    hardsigmoid_out_nocheck(self, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::hardsigmoid(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  hardsigmoid_out_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::hardsigmoid_(at::Tensor& self) {
  hardsigmoid_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
