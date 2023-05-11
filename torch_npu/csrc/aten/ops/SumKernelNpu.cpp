#include <ATen/WrapDimUtilsMulti.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu{
namespace native{

at::Tensor &sum_out_npu_nocheck(
    at::Tensor &result,
    const at::Tensor &self,
    at::IntArrayRef dim,
    bool keepdim) {
  at::dim_list_to_bitset(dim, self.dim());
  c10::SmallVector<int64_t, N> dimList = dim.empty() ? CalcuOpUtil::GetDimlistForTensor(self) :
      c10::SmallVector<int64_t, N>(dim);
  OpCommand cmd;
  cmd.Name("ReduceSum")
      .Input(self)
      .Input(dimList, at::kLong)
      .Output(result)
      .Attr("keep_dims", keepdim)
      .Run();
  return result;
}

at::Tensor &NPUNativeFunctions::sum_out(
    const at::Tensor &self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor &result) {
  auto outputSize = sum_npu_output_size(self, dim.value(), keepdim);
  auto res_type = dtype.has_value() ? dtype.value() : result.scalar_type();

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      res_type,
      outputSize);

  auto selfSize = self.sizes();
  for (int64_t i = 0; i < selfSize.size(); i++) {
    if (selfSize[i] == 0) {
      at::Tensor result_cast = at::empty(outputSize);
      result.copy_(result_cast);
      return result;
    }
  }

  at::Tensor self_cp = isIntegralType(self.scalar_type(), true) ?
      NPUNativeFunctions::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor result_cp = result.scalar_type() == self_cp.scalar_type() ? result :
      NPUNativeFunctions::npu_dtype_cast(result, self_cp.scalar_type());

  sum_out_npu_nocheck(result_cp, self_cp, dim.value(), keepdim);
  if (result_cp.scalar_type() != res_type) {
    result_cp = NPUNativeFunctions::npu_dtype_cast(result_cp, res_type);
    result.copy_(result_cp);
  } else {
    result = result_cp;
  }
  return result;
}

at::Tensor &NPUNativeFunctions::sum_out(
    const at::Tensor &self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor &result) {
  return NPUNativeFunctions::sum_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

at::Tensor NPUNativeFunctions::sum(
    const at::Tensor &self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  at::Tensor self_cp = isIntegralType(self.scalar_type(), true) ?
      NPUNativeFunctions::npu_dtype_cast(self, at::kFloat) : self;
  auto outputSize = reduce_ops_npu_output_size(self_cp, dim.value(), keepdim);
  auto selfSize = self_cp.sizes();
  auto out_type = self.scalar_type();

  if (dtype.has_value()) {
    out_type = dtype.value();
  } else if (isIntegralType(out_type, true)) {
    out_type = at::kLong;
  }

  for (int64_t i = 0; i < selfSize.size(); i++) {
    if (selfSize[i] == 0) {
      return at::zeros(outputSize, self_cp.options());
    }
  }

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self_cp.options(), ACL_FORMAT_ND);
  sum_out_npu_nocheck(result, self_cp, dim.value(), keepdim);

  if (result.scalar_type() != out_type) {
    result = NPUNativeFunctions::npu_dtype_cast(result, out_type);
  }
  return result;
}

at::Tensor NPUNativeFunctions::sum(
    const at::Tensor &self,
    at::DimnameList dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  return NPUNativeFunctions::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor NPUNativeFunctions::sum(const at::Tensor &self, c10::optional<c10::ScalarType> dtype) {
  return NPUNativeFunctions::sum(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}

} // namespace native
} // namespace at_npu
