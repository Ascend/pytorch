#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& asinh_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Asinh")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::asinh_out(
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    asinh_out_npu_nocheck(contiguousResult, self);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    asinh_out_npu_nocheck(result, self);
  }
  return result;

}

at::Tensor NPUNativeFunctions::asinh(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  asinh_out_npu_nocheck(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::asinh_(at::Tensor& self) {

  return asinh_out(self, self);
}

} // namespace native
} // namespace at_npu
