#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::tanh_out(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Tanh")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::tanh(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  NPUNativeFunctions::tanh_out(self, result);

  return result;
}

at::Tensor& NPUNativeFunctions::tanh_(at::Tensor& self) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = NPUNativeFunctions::tanh_out(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::tanh_out(self, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu
