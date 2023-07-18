#ifndef PLUGIN_NATIVE_UTILS_CUSTOM_FUNCTIONS
#define PLUGIN_NATIVE_UTILS_CUSTOM_FUNCTIONS

#include <ATen/ATen.h>

namespace at_npu {
namespace native {
namespace custom_ops {
at::Tensor npu_dtype_cast(const at::Tensor &self, at::ScalarType dtype);
at::Tensor &npu_dtype_cast_(at::Tensor &self, const at::Tensor &src);
at::Tensor npu_bmmV2(const at::Tensor &self, const at::Tensor &mat2, at::IntArrayRef output_sizes);
at::Tensor npu_broadcast(const at::Tensor &self, at::IntArrayRef size);
at::Tensor &npu_broadcast_out(const at::Tensor &self, at::IntArrayRef size, at::Tensor &result);
at::Tensor &npu_indexing_out(const at::Tensor &self, c10::IntArrayRef begin, c10::IntArrayRef end,
                             c10::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                             int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor &result);
at::Tensor &npu_reshape_out(const at::Tensor &src, at::IntArrayRef shape, bool can_refresh, at::Tensor &result);
at::Tensor &npu_slice_out(const at::Tensor &self, c10::IntArrayRef offsets, c10::IntArrayRef size, at::Tensor &result);
}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu

#endif
