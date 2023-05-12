#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& hardswish_out_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("HardSwish")
     .Input(self)
     .Output(result)
     .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::hardswish_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);

  if (!NpuUtils::check_match(&result)) {
      at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
      hardswish_out_nocheck(self, contiguousResult);
      NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
      hardswish_out_nocheck(self, result);
  }
  
  return result;
}

at::Tensor NPUNativeFunctions::hardswish(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);

  hardswish_out_nocheck(self, result);

  return result;
}

at::Tensor& NPUNativeFunctions::hardswish_(at::Tensor& self) {
  NPUNativeFunctions::hardswish_out(self, self);
  
  return self;
}

} // namespace native
} // namespace at_npu