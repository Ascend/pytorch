#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::frac_out(const at::Tensor& self, at::Tensor& out) {
  OpPreparation::CheckOut(
      {self}, 
      out,
      self);
  at::Tensor cast_return_Tensor = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Int);
  at::sub_out(out, self, cast_return_Tensor);
  return out;
}

at::Tensor NPUNativeFunctions::frac(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  at::Tensor cast_return_Tensor = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Int);
  NPUNativeFunctions::frac_out(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::frac_(at::Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = NPUNativeFunctions::frac_out(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::frac_out(self, self);
  }
  return self;
}
} // namespace native
} // namespace at_npu
