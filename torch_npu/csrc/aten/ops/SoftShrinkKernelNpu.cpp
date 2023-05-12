#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& softshrink_out_nocheck(
    const at::Tensor& self,
    at::Scalar lambd,
    at::Tensor& result) {

  OpPreparation::CheckMemory({self}, {result});
  OpCommand cmd;
  cmd.Name("SoftShrink")
      .Input(self)
      .Output(result)
      .Attr("lambd", lambd)
      .Run();
      
  return result;
}

at::Tensor& NPUNativeFunctions::softshrink_out(
    const at::Tensor& self,
    const at::Scalar& lambd,
    at::Tensor& result) {
  TORCH_CHECK(lambd.toFloat() > 0, "lambd should be greater than 0");
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    softshrink_out_nocheck(self, lambd, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    softshrink_out_nocheck(self, lambd, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::softshrink(const at::Tensor& self, const at::Scalar& lambd) {
  TORCH_CHECK(lambd.toFloat() > 0, "lambd should be greater than 0");
  at::Tensor result = OpPreparation::ApplyTensor(self);

  softshrink_out_nocheck(self, lambd, result);
  
  return result;
}
} // namespace native
} // namespace at_npu
