#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& tan_out_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Tan")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::tan_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    tan_out_nocheck(self, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    tan_out_nocheck(self, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::tan(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  tan_out_nocheck(self, result);

  return result;
}

at::Tensor& NPUNativeFunctions::tan_(at::Tensor& self) {
  NPUNativeFunctions::tan_out(self, self);

  return self;
}
} // namespace native
} // namespace at_npu
