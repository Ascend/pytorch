#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::frac_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  at::Tensor cast_return_Tensor = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Int);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    at::sub_out(contiguous_result, self, cast_return_Tensor);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    at::sub_out(result, self, cast_return_Tensor);
  }
  return result;
}

at::Tensor NPUNativeFunctions::frac(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  at::Tensor cast_return_Tensor = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Int);
  NPUNativeFunctions::frac_out(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::frac_(at::Tensor& self) {
  NPUNativeFunctions::frac_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
