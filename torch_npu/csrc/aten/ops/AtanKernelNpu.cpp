#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& atan_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Atan")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::atan_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    atan_out_npu_nocheck(self, contiguous_result);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    atan_out_npu_nocheck(self, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::atan(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  atan_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::atan_(at::Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    at::Tensor result = atan_out_npu_nocheck(contiguous_self, contiguous_self);
    NpuUtils::format_fresh_view(self, result);
  } else {
    atan_out_npu_nocheck(self, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu
