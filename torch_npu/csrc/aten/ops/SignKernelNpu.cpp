#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& sign_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Sign")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}


at::Tensor& NPUNativeFunctions::sign_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  sign_out_npu_nocheck(result, self);

  return result;
}

at::Tensor& NPUNativeFunctions::sgn_out(const at::Tensor& self, at::Tensor& result) {
  return NPUNativeFunctions::sign_out(self, result);
}

at::Tensor NPUNativeFunctions::sign(const at::Tensor& self) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  sign_out_npu_nocheck(result, self);

  return result;
}

at::Tensor& NPUNativeFunctions::sign_(at::Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = sign_out_npu_nocheck(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    sign_out_npu_nocheck(self, self);
  }

  return self;
}
} // namespace native
} // namespace at_npu