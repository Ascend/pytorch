#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& ge_out_npu_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor selfCast = self;
  at::Tensor otherCast = other;
  if (self.dtype() == at::ScalarType::Int || other.dtype() == at::ScalarType::Int 
      || self.dtype() == at::ScalarType::Bool || other.dtype() == at::ScalarType::Bool) {
    selfCast = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    otherCast = NPUNativeFunctions::npu_dtype_cast(other, at::ScalarType::Float);
  }
  auto unified_result = OpPreparation::comparison_op_check(result, selfCast, otherCast, true);
  OpCommand cmd;
  cmd.Name("GreaterEqual")
     .Expect(unified_result)
     .Input(selfCast)
     .Input(otherCast)
     .Output(result)
     .Run();
  
  return result;
}

at::Tensor& NPUNativeFunctions::ge_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  ge_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
  return result;
}

at::Tensor& ge_out_npu_nocheck(const at::Tensor& self, at::Scalar other, at::Tensor& result) {
  at::Tensor selfCast = self;
  if (self.dtype() == at::ScalarType::Int || self.dtype() == at::ScalarType::Bool) {
    selfCast = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  OpCommand cmd;
  cmd.Name("GreaterEqual")
     .Input(selfCast)
     .Input(other, selfCast.scalar_type())
     .Output(result)
     .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::ge_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes(); 
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      result.scalar_type(),
      outputSize);

  ge_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

at::Tensor NPUNativeFunctions::ge(const at::Tensor& self, const at::Tensor& other) {
  if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
    return NPUNativeFunctions::ge(self, other.item());
  }
  else if (self.dim() == 0 && !torch_npu::utils::is_npu(self)) {
    return NPUNativeFunctions::lt(other, self.item());
  }
  else {
    at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
    at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
    auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);
    at::Tensor result = OpPreparation::ApplyTensorWithFormat(
        outputSize,
        formatCastOfSelf.options().dtype(at::kBool),
        ACL_FORMAT_ND);
    ge_out_npu_nocheck(formatCastOfSelf, formatCastOfOther, result);
    return result;
  }
}

at::Tensor NPUNativeFunctions::ge(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      formatCastOfSelf.sizes(),
      formatCastOfSelf.options().dtype(at::kBool),
      ACL_FORMAT_ND);
  ge_out_npu_nocheck(formatCastOfSelf, other, result);
  return result;
}

at::Tensor& NPUNativeFunctions::ge_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  at::Tensor ori_other = OpPreparation::CastBackToOriFormat(other);
  OpPreparation::CheckMemory({self, ori_other}, {self}); 

  at::Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(at::ScalarType::Byte));

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ge_out_npu_nocheck(contiguousSelf, ori_other, result);
  } else {
    ge_out_npu_nocheck(self, ori_other, result);
  }
  self.copy_(result);
  return self;
}

at::Tensor& NPUNativeFunctions::ge_(at::Tensor& self, const at::Scalar& other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckMemory({self}, {self}); 
  at::Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(at::ScalarType::Byte));
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ge_out_npu_nocheck(contiguousSelf, other, result);
  } else {
    ge_out_npu_nocheck(self, other, result);
  }
  self.copy_(result);
  return self;
}

} // namespace native
} // namespace at_npu
