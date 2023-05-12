#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& log2_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Log")
      .Input(self)
      .Output(result)
      .Attr("base", (float)2.0)
      .Attr("scale", (float)1.0)
      .Attr("shift", (float)0.0)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::log2_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  log2_out_npu_nocheck(self, result);

  return result;
}

at::Tensor NPUNativeFunctions::log2(const at::Tensor& self) {
  at::Tensor result =  OpPreparation::ApplyTensor(self);
  log2_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::log2_(at::Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = log2_out_npu_nocheck(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    log2_out_npu_nocheck(self, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu