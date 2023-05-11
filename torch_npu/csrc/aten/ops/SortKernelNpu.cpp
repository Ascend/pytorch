#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> sort_out_npu_no_transpose(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices) {
  OpCommand cmd;
  cmd.Name("Sort")
     .Input(self)
     .Output(values)
     .Output(indices)
     .Attr("axis", dim)
     .Attr("descending", descending)
     .Run();

  return std::tie(values, indices);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::sort_out(
    const at::Tensor& self,
    int64_t dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices) {
  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  int64_t lastDim = CalcuOpUtil::MakeWrapDim(-1, self.dim());

  if (dim != lastDim) {
    at::SmallVector<int64_t, SHAPE_SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
      perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[lastDim]);

    at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm, true);
    auto outputSize = transpose_npu_output_size(values, perm);
    at::Tensor transposeValues = OpPreparation::ApplyTensor(values, outputSize);
    at::Tensor transposeIndices =OpPreparation::ApplyTensor(indices, outputSize);

    sort_out_npu_no_transpose(
        transposeSelf, lastDim, descending, transposeValues, transposeIndices);
    
    NPUNativeFunctions::npu_transpose_out(transposeValues, perm, true, values);
    NPUNativeFunctions::npu_transpose_out(transposeIndices, perm, true, indices);
  } else {
    sort_out_npu_no_transpose(
        self, lastDim, descending, values, indices);
  }
  
  // indices dtype transform Int64
  indices = NPUNativeFunctions::npu_dtype_cast(indices, at::kLong);
  
  return std::tie(values, indices);
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::sort_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool descending,
    at::Tensor& values,
    at::Tensor& indices) {
  return NPUNativeFunctions::sort_out(self, dimname_to_position(self, dim), descending, values, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  auto outputSize = input_same_output_size(self);

  at::Tensor values = OpPreparation::ApplyTensor(self);
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options().dtype(at::kInt), ACL_FORMAT_NCHW);

  NPUNativeFunctions::sort_out(self, dim, descending, values, indices);

  return std::tie(values, indices);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::sort(
    const at::Tensor& self,
    at::Dimname dim,
    bool descending) {
  return NPUNativeFunctions::sort(self, dimname_to_position(self, dim), descending);
}

} // namespace native
} // namespace at_npu
