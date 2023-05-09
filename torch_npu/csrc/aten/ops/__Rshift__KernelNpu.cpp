#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& rshift_out_npu_nocheck(
    const at::Tensor& self,
    at::Scalar other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("RightShift")
     .Input(self)
     .Input(other, self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

at::Tensor& rshift_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
    OpCommand cmd;
    cmd.Name("RightShift")
       .Input(self)
       .Input(other)
       .Output(result)
       .Run(); 

  return result;
}

at::Tensor NPUNativeFunctions::__rshift__(const at::Tensor& self, const at::Tensor& other) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);
  rshift_out_npu_nocheck(self, other,result);

  return result;
}

at::Tensor NPUNativeFunctions::__rshift__(const at::Tensor& self, const at::Scalar& other) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);

  rshift_out_npu_nocheck(self, other, result);

  return result;
}

} // namespace native
} // namespace at