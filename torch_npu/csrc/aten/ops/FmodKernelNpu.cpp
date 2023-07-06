#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& fmod_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("Mod")
    .Expect(unified_result)
    .Input(self)
    .Input(other)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& fmod_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  OpCommand cmd;
  cmd.Name("Mod")
    .Expect(unified_result)
    .Input(self)
    .Input(other, self.scalar_type())
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::fmod_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self, other}, 
      result,
      self, 
      outputSize);
  
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor& NPUNativeFunctions::fmod_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  OpPreparation::CheckOut({self}, result, self);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor& NPUNativeFunctions::fmod_(at::Tensor& self, const at::Scalar& other) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = fmod_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fmod_out_npu_nocheck(self, self, other);
  }

  return self;
}

at::Tensor& NPUNativeFunctions::fmod_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CheckMemory({self, other}, {self}); 
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = fmod_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fmod_out_npu_nocheck(self, self, other);
  }

  return self;
}

at::Tensor NPUNativeFunctions::fmod(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor NPUNativeFunctions::fmod(const at::Tensor& self, const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  fmod_out_npu_nocheck(result, self, other);
  return result;
}

} // namespace native
} // namespace at_npu