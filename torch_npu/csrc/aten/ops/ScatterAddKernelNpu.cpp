#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& scatter_add_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  OpCommand cmd;
  cmd.Name("ScatterAddWithAxis")
      .Input(self)
      .Input(index)
      .Input(src)
      .Output(result)
      .Attr("axis", dim)
      .Run();
  return result;
}

at::Tensor NPUNativeFunctions::scatter_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  return self.clone(at::MemoryFormat::Contiguous).scatter_add_(dim, index, src);
}

at::Tensor& NPUNativeFunctions::scatter_add_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  OpPreparation::CheckMemory({self, index, src}, {self});

  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    scatter_add_out_npu_nocheck(contiguousSelf, self, dim, index, src);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    scatter_add_out_npu_nocheck(self, self, dim, index, src);
  }

  return self;
}

at::Tensor NPUNativeFunctions::scatter_add(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  return scatter_add(self, dimname_to_position(self, dim), index, src);
}
} // namespace native
} // namespace at_npu
