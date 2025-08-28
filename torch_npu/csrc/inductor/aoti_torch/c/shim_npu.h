#ifndef AOTI_TORCH_SHIM_NPU
#define AOTI_TORCH_SHIM_NPU

#include <torch_npu/csrc/inductor/aoti_torch/c/shim.h>

#ifdef USE_NPU
#ifdef __cplusplus
extern "C" {
#endif

struct NPUGuardOpaque;
using NPUGuardHandle = NPUGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_npu_guard(
    int32_t device_index,
    NPUGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_npu_guard(NPUGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_npu_guard_set_index(NPUGuardHandle guard, int32_t device_index);

struct NPUStreamGuardOpaque;
using NPUStreamGuardHandle = NPUStreamGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_npu_stream_guard(
    void* stream,
    int32_t device_index,
    NPUStreamGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_npu_stream_guard(NPUStreamGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_npu_stream(int32_t device_index, void** ret_stream);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_npu_device(int32_t* device_index);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_set_current_npu_device(const int32_t& device_index);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_current_sycl_queue(void** ret);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // USE_NPU
#endif // AOTI_TORCH_SHIM_NPU