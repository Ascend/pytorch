#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& ior_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& ior_out_npu_nocheck(const at::Tensor& self, at::Scalar other, at::Tensor& result) {
  string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::__ior__(at::Tensor& self, const at::Tensor& other) { 
  OpPreparation::CheckMemory({self, other}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = ior_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    ior_out_npu_nocheck(self, other, self);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::__ior__(at::Tensor& self, const at::Scalar& other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = ior_out_npu_nocheck(contiguousSelf, other, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    ior_out_npu_nocheck(self, other, self);
  }
  return self;
}
} // namespace native
} // namespace at_npu
