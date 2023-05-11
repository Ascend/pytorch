#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

constexpr float_t EPSILON = 1e-5;

namespace at_npu {
namespace native {

static inline void normalize_batch_check(
    const at::Tensor& self,
    const at::Tensor& seq_len,
    int64_t normalize_type){
  TORCH_CHECK(
      seq_len.dim() == 1,
      "Non-empty 1D seq_len tensor expected but got a tensor with sizes ",
      seq_len.sizes());
  TORCH_CHECK(
      seq_len.size(0) == self.size(0),
      "seq_len's length should be equal self' num, but got seq_len length ",
      seq_len.size(0),
      "self num ",
      self.size(0));
  TORCH_CHECK(
      normalize_type >= 0 && normalize_type <= 1,
      "normalize_type expected to be in range [0, 1], but got ",
      normalize_type);
}

at::Tensor NPUNativeFunctions::npu_normalize_batch(
    const at::Tensor& self,
    const at::Tensor& seq_len,
    int64_t normalize_type){
  normalize_batch_check(self, seq_len, normalize_type);
  // apply output tensor
  at::Tensor result = OpPreparation::ApplyTensor(self);
  string normalizeType = normalize_type == 0 ? "per_feature" : "all_features";

  OpCommand cmd;
  cmd.Name("NormalizeBatch")
      .Input(self)
      .Input(seq_len)
      .Output(result)
      .Attr("normalize_type", normalizeType)
      .Attr("epsilon", EPSILON)
      .Run();
  return result;
}

} // namespace native
} // namespace at