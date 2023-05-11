#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor &, at::Tensor &> batch_norm_reduce_npu_nocheck(
    at::Tensor &sum,
    at::Tensor &square_sum,
    const at::Tensor &self,
    double eps) {
  at::Tensor selfCp = self;
  if (self.scalar_type() != at::kFloat) {
    selfCp = NPUNativeFunctions::npu_dtype_cast(selfCp, at::kFloat);
  }
  OpCommand cmd;
  cmd.Name("BNTrainingReduce")
     .Input(selfCp)
     .Output(sum)
     .Output(square_sum)
     .Run();

  return std::tie(sum, square_sum);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::batch_norm_reduce(
    const at::Tensor &self,
    double eps) {
  at::Tensor sum = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  at::Tensor square_sum = OpPreparation::ApplyTensor({self.size(1)}, self.options().dtype(at::kFloat), self);
  batch_norm_reduce_npu_nocheck(sum, square_sum, self, eps);

  return std::tie(sum, square_sum);
}

} // namespace native
} // namespace at_npu
