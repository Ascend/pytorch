#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

inline at::Tensor all_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, N> dim_list,
    bool keepdim) {
  OpCommand cmd;
  cmd.Name("ReduceAll")
    .Input(self)
    .Input(dim_list, at::kLong)
    .Output(result) 
    .Attr("keep_dims", keepdim)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::all_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& result) {
  TORCH_CHECK((result.scalar_type() == at::ScalarType::Bool || result.scalar_type() == at::ScalarType::Byte),
      "all only supports bool tensor for result, got: ", result.scalar_type());

  c10::SmallVector<int64_t, N> dim_list = {dim};
  auto output_size = reduce_ops_npu_output_size(self, dim_list, keepdim);
  OpPreparation::CheckOut(
      {self},
      result,
      result,
      output_size);

  at::Tensor self_cast = (self.scalar_type() == at::ScalarType::Bool) ?
      self : NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Bool);
  bool result_is_bool = (result.scalar_type() == at::ScalarType::Bool);
  at::Tensor result_cast = result_is_bool ?
      result : NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Bool);

  if (!NpuUtils::check_match(&result_cast)) {
    at::Tensor contiguous_result_cast = NpuUtils::format_contiguous(result_cast);
    all_out_npu_nocheck(contiguous_result_cast, self_cast, dim_list, keepdim);
    NpuUtils::format_fresh_view(result_cast, contiguous_result_cast);
  } else {
    all_out_npu_nocheck(result_cast, self_cast, dim_list, keepdim);
  }

  if (!result_is_bool) {
    result_cast = NPUNativeFunctions::npu_dtype_cast(result_cast, result.scalar_type());
    result.copy_(result_cast);
  }
  return result;
}

at::Tensor NPUNativeFunctions::all(const at::Tensor& self, int64_t dim, bool keepdim) {
  at::Tensor self_cast = self.scalar_type() == at::ScalarType::Bool ?
      self : NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Bool);

  if (self.dim() != 0) {
    TORCH_CHECK((dim >= -(self.dim()) && dim < self.dim()),
        "The value of dim must be greater than or equal to -self.dim() and less than self.dim()");
  } else {
    TORCH_CHECK_INDEX((self.dim() == dim || dim == -1),
        "Dimension out of range (expected to be in range of [-1, 0], but got ", dim, ")");
  }

  if (self.numel() == 0) {
    c10::SmallVector<int64_t, N> output_size;
    for (int64_t i = 0; i < self.dim(); i++) {
      if (dim != i) {
        output_size.emplace_back(self.size(i));
      }
    }
    at::Tensor result = OpPreparation::ApplyTensor(output_size, self.options().dtype(at::kBool), self).fill_(1);
    return result;
  }

  at::IntArrayRef dims(dim);
  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);
  at::Tensor result = OpPreparation::ApplyTensor(self_cast, output_size);
  all_out_npu_nocheck(result, self_cast, {dim}, keepdim);
  return result;
}

at::Tensor NPUNativeFunctions::all(const at::Tensor& self) {
  at::Tensor self_cast = self.scalar_type() == at::ScalarType::Bool ?
      self : NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Bool);

  if (self.numel() == 0) {
    at::Tensor result = OpPreparation::ApplyTensor({}, self.options().dtype(at::kBool), self).fill_(1);
    return result;
  }

  at::IntArrayRef dims;
  auto output_size = reduce_ops_npu_output_size(self, dims, false);
  at::Tensor result = OpPreparation::ApplyTensor(self_cast, output_size);
  all_out_npu_nocheck(result, self_cast, CalcuOpUtil::GetDimlistForTensor(self), false);
  return result;
}

} // namespace native
} // namespace at_npu
