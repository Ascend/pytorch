#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& mish_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Mish")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::mish_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);
  mish_out_npu_nocheck(self, result);
  return result;
}

at::Tensor NPUNativeFunctions::mish(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  mish_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::mish_(at::Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    at::Tensor result = mish_out_npu_nocheck(contiguous_self, contiguous_self);
    NpuUtils::format_fresh_view(self, result);
  } else {
    mish_out_npu_nocheck(self, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu
