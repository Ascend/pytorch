#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& scatter_out_npu(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& updates,
    int64_t dim) {
  OpCommand cmd;
  cmd.Name("ArgMaxGrad")
      .Input(self)
      .Input(indices)
      .Input(updates)
      .Output(output)
      .Attr("dimension", dim)
      .Run();
  
  return output;
}

at::Tensor NPUNativeFunctions::npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim) {
  at::Tensor outputs = OpPreparation::ApplyTensor(self);
  scatter_out_npu(outputs, self, indices, updates, dim);

  return outputs;
}
} // namespace native
} // namespace at_npu