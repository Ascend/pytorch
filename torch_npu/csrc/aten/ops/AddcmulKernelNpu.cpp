#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& addcmul_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  OpCommand cmd;

  cmd.Name("Addcmul")
    .Input(self)
    .Input(tensor1)
    .Input(tensor2)
    .Input(value, self.scalar_type())
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::addcmul_out(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value,
    at::Tensor& result) {
  auto mulOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), mulOutputSize);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self, tensor1, tensor2}, {result})
      .Func([&self, &tensor1, &tensor2, &value](at::Tensor& result)
      {addcmul_out_npu_nocheck(result, self, tensor1, tensor2, value);})
      .Call(result);
}

at::Tensor NPUNativeFunctions::addcmul(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  auto mulOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), mulOutputSize);

  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  addcmul_out_npu_nocheck(result, self, tensor1, tensor2, value);

  return result;
}

at::Tensor& NPUNativeFunctions::addcmul_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = addcmul_out_npu_nocheck(
        contiguousSelf, contiguousSelf, tensor1, tensor2, value);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addcmul_out_npu_nocheck(self, self, tensor1, tensor2, value);
  }

  return self;
}

} // namespace native
} // namespace at_npu
