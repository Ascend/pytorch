#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& addcdiv_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar value,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Addcdiv")
    .Input(self)
    .Input(tensor1)
    .Input(tensor2)
    .Input(value, self.scalar_type())
    .Output(result)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::addcdiv_out(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value,
    at::Tensor& result) {
  auto div_output_size = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto output_size = broadcast_ops_npu_output_size(self.sizes(), div_output_size);
  OpPreparation::CheckOut(
      {self, tensor1, tensor2},
      result,
      self,
      output_size);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    addcdiv_npu_nocheck(self, tensor1, tensor2, value, contiguous_result);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    addcdiv_npu_nocheck(self, tensor1, tensor2, value, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::addcdiv(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  auto divOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), divOutputSize);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addcdiv_npu_nocheck(self, tensor1, tensor2, value, result);
  return result;
}

at::Tensor& NPUNativeFunctions::addcdiv_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  return NPUNativeFunctions::addcdiv_out(self, tensor1, tensor2, value, self);
}

} // namespace native
} // namespace at_npu
