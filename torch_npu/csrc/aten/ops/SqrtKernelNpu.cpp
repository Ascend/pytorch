#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& sqrt_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Sqrt")
    .Input(self)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::sqrt_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
      .Func([&self](at::Tensor& result){sqrt_out_npu_nocheck(self, result);})
      .Call(result);
}

at::Tensor NPUNativeFunctions::sqrt(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);

  sqrt_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::sqrt_(at::Tensor& self) {
  NPUNativeFunctions::sqrt_out(self, self);

  return self;
}

} // namespace native
} // namespace at_npu
