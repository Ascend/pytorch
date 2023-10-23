#ifndef __PLUGIN_NATIVE_NPU_COMMON_INNER_NATIVE_FUNCTION__
#define __PLUGIN_NATIVE_NPU_COMMON_INNER_NATIVE_FUNCTION__

#include <ATen/ATen.h>

namespace at_npu {
namespace native {

bool can_use_memcpy(at::Tensor& dst, const at::Tensor& src);
void copy_d2d_by_memcpy(at::Tensor& dst, const at::Tensor& src, int64_t exceptSize = 0);
void copy_d2d_dtype(at::Tensor& self, const at::Tensor& src, bool non_blocking);
void copy_d2d_dtype_baseformat(at::Tensor& self, const at::Tensor& src, bool non_blocking);
bool try_to_optimize_copy_with_any_format(at::Tensor& self, const at::Tensor& src);
at::Tensor matmul_by_bmmV2(const at::Tensor& tensor1, const at::Tensor& tensor2);

/**
  Refresh base tensor's metadata of an unmatch tensor to obtain matched tensor
  */
void npu_fast_reshape_(at::Tensor& tensor);
} // namespace native
} // namespace at_npu

#endif