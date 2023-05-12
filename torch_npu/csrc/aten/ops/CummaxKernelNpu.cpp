#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
void cummax_out_npu_nocheck (   
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
  at::Tensor valuesTemp = OpPreparation::ApplyTensor(self);
  at::Tensor indicesTemp = OpPreparation::ApplyTensor(self, self.options().dtype(at::kLong));
    
  cummax_out_npu_nocheck(valuesTemp, indicesTemp, self, dim);

  values.copy_(valuesTemp);
  indices.copy_(indicesTemp);       
}

}}
