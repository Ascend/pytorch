#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& lshift_out_npu_nocheck(
    const at::Tensor& self,
    at::Scalar other,
    at::Tensor& result) {
  at::Tensor otherBroadcast = at::empty(self.sizes(), self.options()).fill_(other); 	
  OpCommand cmd;  
  cmd.Name("LeftShift")
     .Input(self)
     .Input(otherBroadcast)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& lshift_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
    at::Tensor otherBroadcast = other.expand(self.sizes());
    OpCommand cmd;
    cmd.Name("LeftShift")
       .Input(self)
       .Input(otherBroadcast)
       .Output(result)
       .Run(); 
  return result;
}

at::Tensor NPUNativeFunctions::__lshift__(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  lshift_out_npu_nocheck(self, other, result);
  return result;
}

at::Tensor NPUNativeFunctions::__lshift__(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  lshift_out_npu_nocheck(self, other, result);
  return result;
}
} // namespace native
} // namespace at_npu
