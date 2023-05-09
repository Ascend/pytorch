#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& sin_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Sin")
     .Input(self)
     .Output(result)
     .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::sin_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);

  if (!NpuUtils::check_match(&result)) {
      at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
      at::Tensor newResult = sin_out_npu_nocheck(contiguousResult, self);
      NpuUtils::format_fresh_view(result, newResult);
  } else {
      sin_out_npu_nocheck(result, self);
  }
  
  return result;
}

at::Tensor NPUNativeFunctions::sin(const at::Tensor& self) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      self.sizes(), self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

  // calculate the output result of the NPU
  sin_out_npu_nocheck(result, self);

  return result;
}

at::Tensor& NPUNativeFunctions::sin_(at::Tensor& self) {
  NPUNativeFunctions::sin_out(self, self);
  return self;
}
} // namespace native
} // namespace at_npu