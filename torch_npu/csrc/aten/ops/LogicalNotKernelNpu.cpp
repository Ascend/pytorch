#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor& logical_not_out_npu_nocheck(
    const at::Tensor& self, 
    at::Tensor& result) {
  at::ScalarType src_type = self.scalar_type();
  at::Tensor selfCast = self;
  if (src_type != at::ScalarType::Bool) {
    selfCast = NPUNativeFunctions::npu_dtype_cast(self, at::kBool);
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }
  OpCommand cmd;
  cmd.Name("LogicalNot")
      .Input(selfCast)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::logical_not_out(const at::Tensor& self, at::Tensor& result) {
  auto resultDtype = result.scalar_type();
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      resultDtype,
      self.sizes());
  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({self}, {result})
    .Func([&self](at::Tensor& result){logical_not_out_npu_nocheck(self, result);})
    .Call(result);
  result = NPUNativeFunctions::npu_dtype_cast(result, resultDtype);
  return result;
}

at::Tensor NPUNativeFunctions::logical_not(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(
      self.sizes(),
      self.options().dtype(at::kBool),
      self);
  logical_not_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::logical_not_(at::Tensor& self) {
  logical_not_out(self, self);
  return self;
}
} // namespace native
} // namespace at_npu
