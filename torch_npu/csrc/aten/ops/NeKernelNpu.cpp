#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& ne_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  auto unified_result = OpPreparation::comparison_op_check(result, self, other, true);
  if(self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
      "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  OpCommand cmd;
  cmd.Name("NotEqual")
    .Expect(unified_result)
    .Input(self)
    .Input(other)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& ne_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  if(self.scalar_type() == at::kLong) {
    TORCH_WARN_ONCE("The oprator of ne is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used,"
      "Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
  }
  OpCommand cmd;
  cmd.Name("NotEqual")
    .Input(self)
    .Input(other, self.scalar_type())
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::ne_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self, other},
      result,
      CalcuOpUtil::GetTensorNpuFormat(formatCastOfSelf),
      at::ScalarType::Bool,
      at::IntArrayRef(outputSize));
  ne_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

at::Tensor& NPUNativeFunctions::ne_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(formatCastOfSelf),
      at::ScalarType::Bool,
      formatCastOfSelf.sizes());
  ne_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

at::Tensor NPUNativeFunctions::ne(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);

  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);
  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize,
      formatCastOfSelf.options().dtype(at::kBool),
      formatCastOfSelf);

  ne_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

at::Tensor NPUNativeFunctions::ne(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);

  at::Tensor result = OpPreparation::ApplyTensor(
      formatCastOfSelf,
      formatCastOfSelf.options().dtype(at::kBool));

  ne_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

at::Tensor& NPUNativeFunctions::ne_(at::Tensor& self, const at::Tensor& other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CastBackToOriFormat(other);
  OpPreparation::CheckMemory({self, other}, {self});

  at::Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(at::ScalarType::Byte));

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ne_out_npu_nocheck(result, contiguousSelf, other);
  } else {
    ne_out_npu_nocheck(result, self, other);
  }

  self.copy_(result);

  return self;
}

at::Tensor& NPUNativeFunctions::ne_(at::Tensor& self, const at::Scalar& other) {
  OpPreparation::CastBackToOriFormat(self);
  OpPreparation::CheckMemory({self}, {self});
  at::Tensor result = OpPreparation::ApplyTensor(
      self,
      self.options().dtype(at::ScalarType::Byte));

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    ne_out_npu_nocheck(result, contiguousSelf, other);
  } else {
    ne_out_npu_nocheck(result, self, other);
  }

  self.copy_(result);

  return self;
}

} // namespace native
} // namespace at_npu
