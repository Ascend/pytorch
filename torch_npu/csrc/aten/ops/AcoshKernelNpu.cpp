#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& acosh_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Acosh")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::acosh_out(
    const at::Tensor& self,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    acosh_out_npu_nocheck(contiguousResult, self);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    acosh_out_npu_nocheck(result, self);
  }
  return result;

}

at::Tensor NPUNativeFunctions::acosh(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  acosh_out_npu_nocheck(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::acosh_(at::Tensor& self) {

  return acosh_out(self, self);
}

} // namespace native
} // namespace at_npu
