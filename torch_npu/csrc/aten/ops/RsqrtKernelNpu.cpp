#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& rsqrt_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Rsqrt")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::rsqrt_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
      .Func([&self](at::Tensor& result){rsqrt_out_npu_nocheck(self, result);})
      .Call(result);
}

at::Tensor NPUNativeFunctions::rsqrt(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  rsqrt_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::rsqrt_(at::Tensor& self) {
  rsqrt_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
