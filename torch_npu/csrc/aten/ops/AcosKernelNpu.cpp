#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& acos_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Acos")
     .Input(self)
     .Output(result)
     .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::acos_out(const at::Tensor& self, at::Tensor& result) {
  OpPipeWithDefinedOut pipe;
  OpPreparation::CheckOut(
      {self},
      result,
      self);  
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){acos_out_npu_nocheck(self, result);})
   .Call(result);
}

at::Tensor NPUNativeFunctions::acos(const at::Tensor& self) {
  OpPipeWithApplyOut pipe;
  return pipe.ApplyOutputSameAs(self)
    .Func([&self](at::Tensor& result) {acos_out_npu_nocheck(self, result);})
    .Call();
}

at::Tensor& NPUNativeFunctions::acos_(at::Tensor& self) {
  acos_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu