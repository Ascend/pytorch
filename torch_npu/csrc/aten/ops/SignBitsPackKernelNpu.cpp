#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& sign_bits_pack_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t size) {
  OpCommand cmd;
  cmd.Name("SignBitsPack")
    .Input(self)
    .Output(result) 
    .Attr("size", size)
    .Run();
  return result;
}

at::Tensor NPUNativeFunctions::npu_sign_bits_pack(const at::Tensor& self, int64_t size) {
  TORCH_CHECK(self.dim() == 1, "input must be one-dimensional");
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::Float,
      "all only supports torch.float16 and torch.float32 dtypes");
  auto ysize = (self.numel() + 7) / 8;
  TORCH_CHECK(size != 0 && ysize % size == 0, "all must be divisible by size");
  at::Tensor result = OpPreparation::ApplyTensor({size, ysize / size}, self.options().dtype(at::kByte), self);
  
  // calculate the output result of the NPU
  sign_bits_pack_npu_nocheck(result, self, size);

  return result;
}

} // namespace native
} // namespace at_npu