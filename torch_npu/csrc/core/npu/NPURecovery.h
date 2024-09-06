#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {

bool get_npu_data_unsafe_flag();
void set_npu_data_unsafe_flag(bool flag);
void check_npu_tensor_is_safe(const at::Tensor& self);
void check_npu_tensor_is_safe(const c10::optional<at::Tensor>& self);
void check_npu_tensor_is_safe(const at::TensorList& self);
void check_npu_tensor_is_safe(const at::ITensorListRef& self);
void check_npu_tensor_is_safe(const c10::List<c10::optional<at::Tensor>>& self);
void update_npu_tensor_is_safe(const at::Tensor& self);
void update_npu_tensor_is_safe(const at::TensorList& self);
void check_and_update_npu_tensor_for_copy(const at::Tensor& dst, const at::Tensor& src);
void check_and_update_npu_tensor_for_copy(const at::TensorList& dsts, const at::TensorList& srcs);
#ifndef BUILD_LIBTORCH
TORCH_NPU_API void bind_npu_recovery_functions(PyObject* module);
#endif
}
