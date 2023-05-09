#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor rsub_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);

  return isSelfWrapped ? other : self;
}

at::Tensor& rsub_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha) {
  // other*alpha
  at::Tensor otherMulResult;
  if (!CalcuOpUtil::IsScalarOne(alpha)) {
    otherMulResult = at::mul(self, alpha);
  }

  OpCommand cmd;
  if (otherMulResult.defined()) {
    cmd.Name("Sub")
       .Input(other)
       .Input(otherMulResult)
       .Output(result)
       .Run();
  } else {
    cmd.Name("Sub")
       .Input(other)
       .Input(self)
       .Output(result)
       .Run();
  }

  return result;
}

at::Tensor& rsub_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar other,
    at::Scalar alpha) {
  // other*alpha
  at::Tensor scalarValue(at::mul(self, alpha));

  OpCommand cmd;
  cmd.Name("Sub")
       .Input(other, self.scalar_type())
       .Input(scalarValue)
       .Output(result)
       .Run();

  return result;
}

at::Tensor NPUNativeFunctions::rsub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  at::Tensor outputTensor = rsub_dest_output(self, other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  at::Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);

  rsub_out_npu_nocheck(result, self, other, alpha);

  return result;
}

at::Tensor NPUNativeFunctions::rsub(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  at::Tensor result = OpPreparation::ApplyTensor(self);

  rsub_out_npu_nocheck(result, self, other, alpha);

  return result;
}

} // namespace native
} // namespace at_npu
