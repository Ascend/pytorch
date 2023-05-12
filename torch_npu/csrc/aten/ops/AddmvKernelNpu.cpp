#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::addmv_out(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  NpuUtils::check_1d(vec, "vec", "addmv");

  at::Tensor mat1 = vec.unsqueeze(1);

  // matmul mat*alpha
  at::Tensor mat_alpha = at::mul(mat, alpha);

  // matmul*alpha
  at::Tensor mmMulResult = at::mm(mat_alpha, mat1);

  at::Tensor mmMulResult1 = mmMulResult.squeeze();

  // calculate the output size
  auto outputSize = addmv_npu_output_size(self, mat, vec, beta, alpha);

  if (!result.sizes().equals(outputSize)) {
    result.resize_(outputSize);
  }
  // matmul*alpha+self*beta
  at::add_out(result, mmMulResult1, self, beta);

  return result;
}

at::Tensor NPUNativeFunctions::addmv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto outputSize = addmv_npu_output_size(self, mat, vec, beta, alpha);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addmv_out(self, mat, vec, beta, alpha, result);

  return result;
}

at::Tensor& NPUNativeFunctions::addmv_(
    at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  OpPreparation::CheckMemory({self, mat, vec}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result =
        addmv_out(contiguousSelf, mat, vec, beta, alpha, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    addmv_out(self, mat, vec, beta, alpha, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu


