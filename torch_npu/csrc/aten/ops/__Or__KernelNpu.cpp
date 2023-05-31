#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor or___dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  if (isSelfWrapped) {
    return other;
  } else {
    return self;
  }
}

at::Tensor& or___out_scalar_npu(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
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

at::Tensor& or___out_tensor_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (OpPreparation::IsCPUScalar(other)) {
    or___out_scalar_npu(result, self, other.item());
  } else if (OpPreparation::IsCPUScalar(self)) {
    or___out_scalar_npu(result, other, self.item());
  } else {

    string real_op_name =
        (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
    OpCommand cmd;
    cmd.Name(real_op_name)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

at::Tensor NPUNativeFunctions::__or__(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor outputTensor = or___dest_output(self, other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);
  or___out_tensor_npu(result, self, other);

  return result;
}

at::Tensor NPUNativeFunctions::__or__(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  or___out_scalar_npu(result, self, other);

  return result;
}

} // namespace native
} // namespace at_npu