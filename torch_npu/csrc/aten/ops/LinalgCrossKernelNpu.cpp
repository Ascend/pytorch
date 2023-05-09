#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor linalg_cross_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::IsScalarWrappedToTensor(self);
  return isSelfWrapped ? other : self;
}

at::Tensor& linalg_cross_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<int64_t> dim,
    at::Tensor& result) {
  int64_t realDim = dim.has_value() ? dim.value() : -65530;
  OpCommand cmd;
  cmd.Name("Cross")
    .Input(self)
    .Input(other)
    .Output(result)
    .Attr("dim", realDim)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::linalg_cross_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const int64_t dim,
    at::Tensor& result){
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor outputTensor = linalg_cross_dest_output(self, other);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(outputTensor),
      self.scalar_type(),
      outputSize);
  linalg_cross_out_npu_nocheck(self, other, dim, result);
  return result;
}

at::Tensor NPUNativeFunctions::linalg_cross(
    const at::Tensor& self,
    const at::Tensor& other,
    const int64_t dim) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor outputTensor = linalg_cross_dest_output(self, other);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options(), outputTensor);
  linalg_cross_out_npu_nocheck(self, other, dim, result);
  return result;
}

} // namespace native
} // namespace at_npu
