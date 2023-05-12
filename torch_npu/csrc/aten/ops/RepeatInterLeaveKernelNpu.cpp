#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& repeat_interleave_out_npu(at::Tensor& result, const at::Tensor& self, int64_t repeats) {
  at::Scalar repeat = repeats;
  OpCommand cmd;
  cmd.Name("RepeatInterleave")
    .Input(self)
    .Input(repeat, at::kLong)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();

  return result;
}

at::Tensor& repeat_interleave_out_npu(at::Tensor& result, const at::Tensor& self, const at::Tensor& repeats) {
  OpCommand cmd;
  cmd.Name("RepeatInterleave")
    .Input(self)
    .Input(repeats)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();

  return result;
}

at::Tensor NPUNativeFunctions::repeat_interleave_symint(
    const at::Tensor& self,
    c10::SymInt repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t realDim = dim.value_or(0);
  int64_t self_dim = self.dim();
  int64_t repeats_ = repeats.guard_int(__FILE__, __LINE__);
  TORCH_CHECK(
      (realDim >= -self_dim) && (realDim <= self_dim - 1),
      "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");
  TORCH_CHECK(
      repeats_ >= 1,
      "repeats can not be negative.");
  at::Tensor selfTensor = self;
  if (!dim.has_value()) {
    selfTensor = at::flatten(selfTensor);
  }
  if (repeats_ == 1) {
    return selfTensor;
  }

  if (self_dim > 1 && realDim != 0) {
    selfTensor = selfTensor.transpose(0, realDim);
  }

  auto outputSize = repeat_interleave_npu_output_size(selfTensor, repeats_, 0);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(selfTensor, outputSize, ACL_FORMAT_ND);
  repeat_interleave_out_npu(result, selfTensor, repeats_);
  if (self_dim > 1 && realDim != 0) {
    result = result.transpose(0, realDim);
  }
  return result;
}

at::Tensor NPUNativeFunctions::repeat_interleave(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t realDim = dim.value_or(0);
  int64_t self_dim = self.dim();
  TORCH_CHECK(
      (realDim >= -self_dim) && (realDim <= self_dim - 1),
      "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");

  at::Tensor selfTensor = self;
  at::Tensor repeatsTensor = repeats;
  if (!dim.has_value()) {
    selfTensor = at::flatten(selfTensor);
  }
  if (repeats.dim() == 1 && repeats.size(0) == 1) {
    return selfTensor;
  }

  TORCH_CHECK(
      repeats.size(0) == selfTensor.size(realDim),
      "repeats must have the same size as input along dim.");

  if (self_dim > 1 && realDim != 0) {
    selfTensor = selfTensor.transpose(0, realDim);
  }

  repeatsTensor = NPUNativeFunctions::npu_dtype_cast(repeatsTensor, at::ScalarType::Int);
  repeatsTensor = NPUNativeFunctions::npu_dtype_cast(repeatsTensor, at::ScalarType::Float);
  auto outputSize = repeat_interleave_npu_output_size(selfTensor, repeatsTensor, 0);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(selfTensor, outputSize, ACL_FORMAT_ND);
  repeat_interleave_out_npu(result, selfTensor, repeats);
  if (self_dim > 1 && realDim != 0) {
    result = result.transpose(0, realDim);
  }
  return result;
}

} // namespace native
} // namespace at_npu