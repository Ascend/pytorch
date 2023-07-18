#include "torch_npu/csrc/framework/utils/CustomFunctions.h"

#include <ATen/core/dispatch/Dispatcher.h>

#define GET_OP_HANDLE(BASE_NAME, OVERLOAD_NAME, FUNC) \
  c10::Dispatcher::singleton().findSchemaOrThrow(BASE_NAME, OVERLOAD_NAME).typed<decltype(FUNC)>()

namespace at_npu {
namespace native {
namespace custom_ops {
at::Tensor npu_dtype_cast(const at::Tensor &self, at::ScalarType dtype) {
  static auto op = GET_OP_HANDLE("npu::npu_dtype_cast", "", npu_dtype_cast);
  return op.call(self, dtype);
}

at::Tensor &npu_dtype_cast_(at::Tensor &self, const at::Tensor &src) {
  static auto op = GET_OP_HANDLE("npu::npu_dtype_cast_", "", npu_dtype_cast_);
  return op.call(self, src);
}

at::Tensor npu_bmmV2(const at::Tensor &self, const at::Tensor &mat2, at::IntArrayRef output_sizes) {
  static auto op = GET_OP_HANDLE("npu::npu_bmmV2", "", npu_bmmV2);
  return op.call(self, mat2, output_sizes);
}

at::Tensor npu_broadcast(const at::Tensor &self, at::IntArrayRef size) {
  static auto op = GET_OP_HANDLE("npu::npu_broadcast", "", npu_broadcast);
  return op.call(self, size);
}

at::Tensor &npu_broadcast_out(const at::Tensor &self, at::IntArrayRef size, at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_broadcast", "out", npu_broadcast_out);
  return op.call(self, size, result);
}

at::Tensor &npu_indexing_out(const at::Tensor &self, c10::IntArrayRef begin, c10::IntArrayRef end,
                             c10::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                             int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_indexing", "out", npu_indexing_out);
  return op.call(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                 result);
}

at::Tensor &npu_reshape_out(const at::Tensor &src, at::IntArrayRef shape, bool can_refresh, at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_reshape", "out", npu_reshape_out);
  return op.call(src, shape, can_refresh, result);
}

at::Tensor &npu_slice_out(const at::Tensor &self, c10::IntArrayRef offsets, c10::IntArrayRef size,
                          at::Tensor &result) {
  static auto op = GET_OP_HANDLE("npu::npu_slice", "out", npu_slice_out);
  return op.call(self, offsets, size, result);
}
}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu
