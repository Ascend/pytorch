#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& sinh_out_nocheck(const at::Tensor& self, at::Tensor& result) {
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Sinh")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::sinh_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    sinh_out_nocheck(self, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    sinh_out_nocheck(self, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::sinh(const at::Tensor& self) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  sinh_out_nocheck(self, result);

  return result;
}

at::Tensor& NPUNativeFunctions::sinh_(at::Tensor& self) {
  NPUNativeFunctions::sinh_out(self, self);
  
  return self;
}
} // namespace native
} // namespace at_npu
