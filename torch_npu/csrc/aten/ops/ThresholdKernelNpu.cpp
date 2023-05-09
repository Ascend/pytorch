#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeFunctions::threshold_out(
    const at::Tensor& self,
    const at::Scalar& threshold,
    const at::Scalar& value,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("ThresholdV2D")
      .Input(self)
      .Output(result)
      .Attr("threshold", threshold)
      .Attr("value", value)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::threshold(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  NPUNativeFunctions::threshold_out(self, threshold, value, result);
  return result;
}

at::Tensor& NPUNativeFunctions::threshold_(at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor selfContiguous = NpuUtils::format_contiguous(self);
    at::Tensor result =
        NPUNativeFunctions::threshold_out(selfContiguous, threshold, value, selfContiguous);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::threshold_out(self, threshold, value, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu