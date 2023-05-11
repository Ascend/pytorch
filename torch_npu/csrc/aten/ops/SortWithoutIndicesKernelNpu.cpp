#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor& sort_without_indices_no_transpose(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  OpCommand cmd;
  cmd.Name("SortV2")
      .Input(self)
      .Output(result)
      .Attr("axis", dim)
      .Attr("descending", descending)
      .Run();
  
  return result;
}

at::Tensor& NPUNativeFunctions::npu_sort_v2_out(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& result) {
  auto outputSize = input_same_output_size(self);

  OpPreparation::CheckOut(
      {self},
      result,
      self);
  
  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::MakeWrapDim(-1, self.dim());

  if (dim != lastDim) {
    c10::SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);
    at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm, true);

    auto outputSize = transpose_npu_output_size(result, perm);
    at::Tensor transposeResult = OpPreparation::ApplyTensor(result, outputSize);

    sort_without_indices_no_transpose(transposeResult, transposeSelf, lastDim, descending);
    NPUNativeFunctions::npu_transpose_out(transposeResult, perm, true, result);
  } else {
    if (!NpuUtils::check_match(&result)) {
      at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
      sort_without_indices_no_transpose(contiguousResult, self, dim, descending);
      NpuUtils::format_fresh_view(result, contiguousResult);
    } else {
      sort_without_indices_no_transpose(result, self, dim, descending);
    }
  }

  return result;
}

at::Tensor NPUNativeFunctions::npu_sort_v2(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(self);

  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::MakeWrapDim(-1, self.dim());

  if (dim != lastDim) {
    c10::SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);
    at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm, true);

    auto outputSize = transpose_npu_output_size(result, perm);
    at::Tensor transposeResult = OpPreparation::ApplyTensor(result, outputSize);

    sort_without_indices_no_transpose(transposeResult, transposeSelf, lastDim, descending);
    NPUNativeFunctions::npu_transpose_out(transposeResult, perm, true, result);
  } else {
    sort_without_indices_no_transpose(result, self, dim, descending);
  }

  return result;
}
} // namespace native
} // namespace at_npu