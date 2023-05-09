#include "torch_npu/csrc/framework/utils/OpAdapter.h"

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& fill_out_npu(at::Tensor& result, at::Tensor& self, const at::Tensor& other) {
  OpCommand cmd;
  cmd.Name("Fill");
  if (self.dim() == 0) {
    c10::SmallVector<int64_t, N> dims = {1};
    cmd.Input(dims, at::kLong);
  } else {
    cmd.Input(self.sizes(), at::kLong);
  }
  cmd.Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& fill_out_npu(at::Tensor& result, at::Tensor& self, at::Scalar value) {
  OpCommand cmd;
  cmd.Name("Fill");
  if (self.dim() == 0) {
    c10::SmallVector<int64_t, N> dims = {1};
    cmd.Input(dims, at::kLong);
  } else {
    cmd.Input(self.sizes(), at::kLong);
  }
  cmd.Input(value, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::fill_(at::Tensor& self, const at::Tensor& other) {
  auto other_dim = other.dim();
  TORCH_CHECK(other_dim <= 1, "fill_ only supports 0 or 1 dimension value tensor but got tensor with ",
      other_dim, " dimension.");
  if (other_dim == 0 && !torch_npu::utils::is_npu(other)) {
    fill_out_npu(self, self, other.item());
  } else {
    fill_out_npu(self, self, other);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::fill_(at::Tensor& self, const at::Scalar& value) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = fill_out_npu(contiguousSelf, contiguousSelf, value);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fill_out_npu(self, self, value);
  }

  return self;
}

} // namespace native
} // namespace at_npu
