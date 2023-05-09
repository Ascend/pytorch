#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& abs_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Abs")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::abs_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){abs_out_npu_nocheck(result, self);})
   .Call(result);
}

at::Tensor NPUNativeFunctions::abs(const at::Tensor& self) {
  OpPipeWithApplyOut pipe;
  return pipe.ApplyOutputSameAs(self)
    .Func([&self](at::Tensor& result) {abs_out_npu_nocheck(result, self);})
    .Call();
}

at::Tensor& NPUNativeFunctions::abs_(at::Tensor& self) {
  abs_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
