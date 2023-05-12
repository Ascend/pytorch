#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& erf_npu_nocheck(const at::Tensor& self, at::Tensor& out) {
  OpCommand cmd;
  cmd.Name("Erf")
    .Input(self)
    .Output(out)
    .Run();
  return out;
}

at::Tensor& NPUNativeFunctions::erf_out(const at::Tensor& self, at::Tensor& out) {
  OpPreparation::CheckOut(
      {self},
      out,
      self);

  if (!NpuUtils::check_match(&out)) {
      at::Tensor contiguousResult = NpuUtils::format_contiguous(out);
      at::Tensor newResult = erf_npu_nocheck(self, contiguousResult);
      NpuUtils::format_fresh_view(out, newResult);
  } else {
      erf_npu_nocheck(self, out);
  }
  return out;
}

at::Tensor NPUNativeFunctions::erf(const at::Tensor& self) {
  auto outputSize = input_same_output_size(self); 
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  erf_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::erf_(at::Tensor& self) {
  NPUNativeFunctions::erf_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu