#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& irshift_out_npu_nocheck(
    at::Tensor& self,
    at::Scalar other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("RightShift")
     .Input(self)
     .Input(other,self.scalar_type())
     .Output(result)
     .Run();
  return result;
}

at::Tensor& irshift_out_npu_nocheck(
    at::Tensor& self,
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

at::Tensor& NPUNativeFunctions::__irshift__(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CheckMemory({self, other}, {self});  
  if(!NpuUtils::check_match(&self)){
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    irshift_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    irshift_out_npu_nocheck(self, other, self);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::__irshift__(at::Tensor& self, const at::Scalar& other) {
  if(!NpuUtils::check_match(&self)){
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    irshift_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    irshift_out_npu_nocheck(self, other, self);
  }
  return self;
}
} // namespace native
} // namespace at_npu
