#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::hardtanh_out(
    const at::Tensor& self,
    const at::Scalar& min,
    const at::Scalar& max,
    at::Tensor& result) {
  OpPreparation::CheckMemory({self}, {result});
  OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(min, self.scalar_type())
      .Input(max, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::hardtanh(const at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  hardtanh_out(self, min, max, result);
  return result;
}

at::Tensor& NPUNativeFunctions::hardtanh_(at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = hardtanh_out(contiguousSelf, min, max, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    hardtanh_out(self, min, max, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu
