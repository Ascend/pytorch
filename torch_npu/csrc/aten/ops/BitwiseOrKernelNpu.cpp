#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& bitwise_or_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other) {
  // executing the NPU operator
  string real_op_name =
      (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";

  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::bitwise_or_out(
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  bitwise_or_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor& bitwise_or_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  if (OpPreparation::IsCPUScalar(other)) {
    NPUNativeFunctions::bitwise_or_out(self, other.item(), result);
  } else if (OpPreparation::IsCPUScalar(self)) {
    NPUNativeFunctions::bitwise_or_out(other, self.item(), result);
  } else {
    // executing the NPU operator
    string real_op_name =
        (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";

    OpCommand cmd;
    cmd.Name(real_op_name)
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}

at::Tensor& NPUNativeFunctions::bitwise_or_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  at::Tensor outputTensor;
  if (isSelfWrapped) {
    outputTensor = other;
  } else {
    outputTensor = self;
  }
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(outputTensor),
      outputTensor.scalar_type(),
      outputSize);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor newResult = bitwise_or_out_npu_nocheck(contiguousResult, self, other);
    NpuUtils::format_fresh_view(result, newResult);
  } else {
    bitwise_or_out_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor NPUNativeFunctions::bitwise_or(const at::Tensor& self, const at::Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  at::Tensor outputTensor;
  if (isSelfWrapped) {
    outputTensor = other;
  } else {
    outputTensor = self;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output Tensor of the NPUitwiseOrKerne
  at::Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);
  // calculate the output result of the NPU
  bitwise_or_out_npu_nocheck(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::bitwise_or(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  bitwise_or_out_npu_nocheck(result, self, other);
  return result;
}

} // namespace native
} // namespace at_npu
