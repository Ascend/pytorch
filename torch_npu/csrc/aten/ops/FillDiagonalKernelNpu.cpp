#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& fill_diagonal_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& value,
    bool wrap) {
  float fill_value = CalcuOpUtil::GetScalarFloatValue(value);
  OpCommand cmd;
  cmd.Name("FillDiagonal")
      .Input(self)
      .Output(result)
      .Attr("fill_value", fill_value)
      .Attr("wrap", wrap)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::fill_diagonal_(at::Tensor& self, const at::Scalar& value, bool wrap) {
  OpPreparation::CastBackToOriFormat(self);

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result =
        fill_diagonal_out_npu(contiguousSelf, contiguousSelf, value, wrap);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fill_diagonal_out_npu(self, self, value, wrap);
  }

  return self;
}

} // namespace native
} // namespace at_npu