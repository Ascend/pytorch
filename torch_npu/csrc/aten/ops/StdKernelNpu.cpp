#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> std_mean_out_npu_nocheck(
    at::Tensor& resultStd, 
    at::Tensor& resultMean, 
    const at::Tensor& self, 
    at::IntArrayRef dim, 
    bool unbiased, 
    bool keepdim) {
  OpCommand cmd1;
  cmd1.Name("ReduceMeanD")
      .Input(self)
      .Output(resultMean)
      .Attr("axes", dim)
      .Attr("keep_dims", keepdim)
      .Run();

  at::Tensor resultMeanCopy = resultMean;
  if (resultMean.dim() != 0 && keepdim == false) {
    auto dimVector = array_to_small_vector(dim);
    std::sort(dimVector.begin(), dimVector.end());
    for (int64_t i = 0; i < dimVector.size(); i++) {
      resultMeanCopy = resultMeanCopy.unsqueeze(dimVector[i]);
    }
  }
  resultMeanCopy = resultMeanCopy.expand(self.sizes());
  OpCommand cmd2;
  cmd2.Name("ReduceStdWithMean")
      .Input(self)
      .Input(resultMeanCopy)
      .Output(resultStd)
      .Attr("dim", dim)
      .Attr("unbiased", unbiased)
      .Attr("keepdim", keepdim)
      .Run();

  return std::tie(resultStd, resultMean);
}

at::Tensor& NPUNativeFunctions::std_out(
    const at::Tensor& self, 
    at::OptionalIntArrayRef dim,
    bool unbiased, 
    bool keepdim,
    at::Tensor& result) {
  auto outputSize = reduce_ops_npu_output_size(self, dim.value(), keepdim);
  at::Tensor meanResult = OpPreparation::ApplyTensor(self, outputSize);

  OpPreparation::CheckOut(
      {self}, 
      result, 
      ACL_FORMAT_ND,
      self.scalar_type(),
      outputSize);

  std_mean_out_npu_nocheck(result, meanResult, self, dim.value(), unbiased, keepdim);

  return result;
}

at::Tensor& NPUNativeFunctions::std_out(
    const at::Tensor& self, 
    at::DimnameList dim, 
    bool unbiased, 
    bool keepdim,
    at::Tensor& result) {
  return NPUNativeFunctions::std_out(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

at::Tensor NPUNativeFunctions::std(
    const at::Tensor & self, 
    at::OptionalIntArrayRef dim,
    bool unbiased, 
    bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dim.value(), keepdim);

  at::Tensor result1 = OpPreparation::ApplyTensor(self, outputSize);
  at::Tensor result2 = OpPreparation::ApplyTensor(self, outputSize);

  std_mean_out_npu_nocheck(result1, result2, self, dim.value(), unbiased, keepdim);
  return result1;
}

at::Tensor NPUNativeFunctions::std(const at::Tensor& self, at::OptionalIntArrayRef dim,
           const c10::optional<at::Scalar>& correction, bool keepdim) {
           const auto correction_double = correction.value_or(1).toDouble();
    return NPUNativeFunctions::std(self, dim, correction_double > 0, keepdim);
}

at::Tensor NPUNativeFunctions::std(
    const at::Tensor & self, 
    bool unbiased) {
  c10::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  return NPUNativeFunctions::std(self, dims, unbiased, false);
}

tuple <at::Tensor, at::Tensor> NPUNativeFunctions::std_mean(
    const at::Tensor & self, 
    at::OptionalIntArrayRef dim,
    bool unbiased, 
    bool keepdim) {
  auto outputSize = reduce_ops_npu_output_size(self, dim.value(), keepdim);

  at::Tensor result1 = OpPreparation::ApplyTensor(self, outputSize);
  at::Tensor result2 = OpPreparation::ApplyTensor(self, outputSize);

  std_mean_out_npu_nocheck(result1, result2, self, dim.value(), unbiased, keepdim);

  return std::tie(result1, result2);
}

tuple <at::Tensor, at::Tensor> NPUNativeFunctions::std_mean(
    const at::Tensor & self, 
    bool unbiased) {
  c10::SmallVector<int64_t, SIZE> dims = CalcuOpUtil::GetDimlistForTensor(self);
  return NPUNativeFunctions::std_mean(self, dims, unbiased, false);
}

tuple <at::Tensor, at::Tensor> NPUNativeFunctions::std_mean(
    const at::Tensor & self, 
    at::DimnameList dim, 
    bool unbiased, 
    bool keepdim) {
  return NPUNativeFunctions::std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

at::Tensor NPUNativeFunctions::std(
    const at::Tensor & self, 
    at::DimnameList dim, 
    bool unbiased, 
    bool keepdim) {
  return NPUNativeFunctions::std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}
} // namespace native
} // namespace at_npu
