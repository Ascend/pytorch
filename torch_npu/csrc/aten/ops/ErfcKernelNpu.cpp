#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& erfc_out_npu_no_check(at::Tensor& out, const at::Tensor& self){
  OpCommand cmd;
  cmd.Name("Erfc")
    .Input(self)
    .Output(out)
    .Run();
  return out;
}

at::Tensor& NPUNativeFunctions::erfc_out(const at::Tensor& self, at::Tensor& out) {
  OpPreparation::CheckOut(
      {self},
      out,
      self,
      self.sizes());
  
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {out})
        .Func([&self](at::Tensor& out){erfc_out_npu_no_check(out, self);})
        .Call(out);
}

at::Tensor NPUNativeFunctions::erfc(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  erfc_out_npu_no_check(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::erfc_(at::Tensor& self) {
  NPUNativeFunctions::erfc_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
