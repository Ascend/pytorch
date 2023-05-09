#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& exp_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  OpCommand cmd;
  cmd.Name("Exp")
    .Input(self)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::exp_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){exp_out_npu_nocheck(result, self);})
   .Call(result);
}

at::Tensor& NPUNativeFunctions::exp_(at::Tensor& self) {
  exp_out(self, self);

  return self;
}

at::Tensor NPUNativeFunctions::exp(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  exp_out_npu_nocheck(result, self);
  return result;
}

} // namespace native
} // namespace at