#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::dot_out(const at::Tensor& self, const at::Tensor& tensor, at::Tensor& result) {
  c10::SmallVector<int64_t, SIZE> outputSize = dot_npu_output_size(self, tensor);
  OpPreparation::CheckOut(
      {self, tensor}, 
      result, 
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(), 
      outputSize);
             
  OpCommand cmd;
  cmd.Name("Dot")
      .Input(self)
      .Input(tensor)
      .Output(result)
      .Run();
  c10::SmallVector<int64_t, N> shape = {};
  result.resize_(shape);
  return result;
}
at::Tensor NPUNativeFunctions::dot(const at::Tensor& self, const at::Tensor& tensor) {
  c10::SmallVector<int64_t, SIZE> outputSize = dot_npu_output_size(self, tensor);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  dot_out(self, tensor, result);
  return result;
}
} // namespace native
} // namespace at_npu