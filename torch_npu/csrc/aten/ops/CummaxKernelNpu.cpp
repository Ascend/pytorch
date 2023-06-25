#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
void cummax_out_npu_nocheck(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim) {
  OpCommand cmd;
  cmd.Name("Cummax")
      .Input(self)
      .Output(values)
      .Output(indices)
      .Attr("dim", dim)
      .Run();
}

void NPUNativeFunctions::_cummax_helper(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim) {
  at::Tensor values_temp = OpPreparation::ApplyTensor(self);
  at::Tensor indices_temp = OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options().dtype(at::kLong),
      ACL_FORMAT_ND);
  cummax_out_npu_nocheck(values_temp, indices_temp, self, dim);

  values.copy_(values_temp);
  indices.copy_(indices_temp);
}

} // namespace native
} // namespace at_npu
