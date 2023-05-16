#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& erfinv_out_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Erfinv")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::erfinv_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(result),
      self.scalar_type(),
      self.sizes());

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    erfinv_out_nocheck(contiguous_result, self);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    erfinv_out_nocheck(result, self);
  }
  return result;
}

at::Tensor NPUNativeFunctions::erfinv(const at::Tensor& self) {
  auto result = OpPreparation::ApplyTensor(self);
  erfinv_out_nocheck(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::erfinv_(at::Tensor& self) {
  return NPUNativeFunctions::erfinv_out(self, self);
}

} // namespace native
} // namespace at_npu
