#include "torch_npu/csrc/aten/ops/QuantizedFlipKernelNpu.h"

#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/flip.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor quantized_flip(const at::Tensor& self, at::IntArrayRef dims) {
  TORCH_CHECK(
      self.scalar_type() != at::kQUInt4x2 &&
          self.scalar_type() != at::kQUInt2x4,
      "flip is not supported for tensor with data type ",
      self.scalar_type());

  // Match the native flip validation order before rejecting per-channel
  // quantization.
  (void)at::dim_list_to_bitset(dims, self.dim());
  TORCH_CHECK(
      self.qscheme() == at::kPerTensorAffine,
      "Setting strides is possible only on uniformly quantized tensor");

  at::Tensor repr = self.int_repr();
  at::Tensor flipped = at::flip(repr, dims);
  at::Tensor result = at::_empty_affine_quantized(
      self.sizes(),
      self.options(),
      self.q_scale(),
      self.q_zero_point(),
      self.suggest_memory_format());
  at_npu::native::NPUNativeFunctions::set_(result, flipped);
  return result;
}

} // namespace native
} // namespace at_npu
