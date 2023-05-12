#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::argmin(
    const at::Tensor& self, 
    c10::optional<int64_t> dim, 
    bool keepdim) {
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function argmin on a "
      "tensor with no elements because the operation does not have an identity");
  at::Tensor input = dim.has_value() ? self : self.reshape({-1});
  int64_t realDim = dim.has_value() ? dim.value() : 0;
  bool realKeepDim = dim.has_value() ? keepdim : false;
  auto outputSize = reduce_ops_npu_output_size(input, realDim, realKeepDim);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kInt),
      ACL_FORMAT_ND);
  c10::Scalar DimScalar = realDim;
  OpCommand cmd;
  cmd.Name("ArgMin")
      .Input(input)
      .Input(DimScalar, at::kInt)
      .Output(result)
      .Attr("keep_dims", realKeepDim)
      .Run();
  result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Long);
  return result;
}

} // namespace native
} // namespace at_npu
