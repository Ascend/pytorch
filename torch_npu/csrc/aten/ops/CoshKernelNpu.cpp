#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& cosh_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Cosh")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::cosh_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(result),
      self.scalar_type(),
      self.sizes());

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    cosh_out_npu_nocheck(contiguous_result, self);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    cosh_out_npu_nocheck(result, self);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::cosh_(at::Tensor& self) {
   return NPUNativeFunctions::cosh_out(self, self);
}

at::Tensor NPUNativeFunctions::cosh(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  cosh_out_npu_nocheck(result, self);
  return result;
}

} // namespace native
} // namespace at_npu
