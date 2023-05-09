#include<ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor& index_add_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
  at::Tensor indices = index;
  if (index.scalar_type() != at::ScalarType::Int) {
    indices = NPUNativeFunctions::npu_dtype_cast(index, at::kInt);
  }
  if (index.dim() == 0) {
    indices.unsqueeze_(0);
  }

  at::SmallVector<int64_t, N> pad_size = array_to_small_vector(self.sizes());
  pad_size[dim] = indices.sizes()[0];
  at::Tensor source_broadcast = NPUNativeFunctions::npu_broadcast(source, pad_size);
  OpCommand cmd;
  cmd.Name("InplaceIndexAdd")
      .Input(self)
      .Input(indices)
      .Input(source_broadcast)
      .Output(result)
      .Attr("axis", dim)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::index_add_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self, index, source},
      result,
      self);

  index_add_out_npu(result, self, dim, index, source, alpha);

  return result;
}

at::Tensor NPUNativeFunctions::index_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
  return self.clone().index_add_(dim, index, source);
}

at::Tensor NPUNativeFunctions::index_add(
    const at::Tensor& self,
    at::Dimname dim, 
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha)  {
  return NPUNativeFunctions::index_add(self, dimname_to_position(self, dim), index, source, alpha);
}
} // namespace native
} // namespace at_npu