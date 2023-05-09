#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& log10_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Log")
      .Input(self)
      .Output(result)
      .Attr("base", (float)10.0)
      .Attr("scale", (float)1.0)
      .Attr("shift", (float)0.0)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::log10_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){log10_out_npu_nocheck(self, result);})
   .Call(result);
}

at::Tensor NPUNativeFunctions::log10(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  log10_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::log10_(at::Tensor& self) {
  log10_out(self, self);
  return self;
}
} // namespace native
} // namespace at_npu
