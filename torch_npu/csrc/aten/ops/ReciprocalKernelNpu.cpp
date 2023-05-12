#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& reciprocal_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Reciprocal")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::reciprocal_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
      .Func([&self](at::Tensor& result){reciprocal_out_npu_nocheck(self, result);})
      .Call(result);
}

at::Tensor NPUNativeFunctions::reciprocal(const at::Tensor& self) {
  at::Tensor self_cp = isIntegralType(self.scalar_type(), true) ?
      NPUNativeFunctions::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor result = OpPreparation::ApplyTensor(self_cp);
  reciprocal_out_npu_nocheck(self_cp, result);

  return result;
}

at::Tensor& NPUNativeFunctions::reciprocal_(at::Tensor& self) {
  NPUNativeFunctions::reciprocal_out(self, self);

  return self;
}


} // namespace native
} // namespace at_npu
