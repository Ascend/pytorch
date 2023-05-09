#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& floor_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Floor")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::floor_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){floor_out_npu_nocheck(self, result);})
   .Call(result);
}

at::Tensor& NPUNativeFunctions::floor_(at::Tensor& self) {
  NPUNativeFunctions::floor_out(self, self);

  return self;
}

at::Tensor NPUNativeFunctions::floor(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  floor_out_npu_nocheck(self, result);
  return result;
}

} // namespace native
} // namespace at_npu
