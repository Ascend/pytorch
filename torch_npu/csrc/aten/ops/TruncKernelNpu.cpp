#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& trunc_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Trunc")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::trunc_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor checkResult = trunc_nocheck(self, contiguousResult);
    NpuUtils::format_fresh_view(result, checkResult);
  } else {
    trunc_nocheck(self, result);
  }

  return result;
}

at::Tensor& NPUNativeFunctions::trunc_(at::Tensor& self) {
  return trunc_out(self, self);
}

at::Tensor NPUNativeFunctions::trunc(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  trunc_nocheck(self, result);
  return result;
}

} // namespace native
} // namespace at_npu