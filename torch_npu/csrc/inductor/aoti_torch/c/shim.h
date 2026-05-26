#ifndef AOTI_TORCH_NPU_SHIM
#define AOTI_TORCH_NPU_SHIM

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTI_TORCH_EXPORT int32_t aoti_torch_device_type_npu();

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob_npu(void* data, int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset, int32_t dtype,
    int32_t device_type, int32_t device_index,
    AtenTensorHandle* ret // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob_npu_v2(
    void* data, int64_t ndim, const int64_t* sizes_ptr, const int64_t* strides_ptr, int64_t storage_offset,
    int32_t dtype, int32_t device_type, int32_t device_index,
    AtenTensorHandle* ret, // returns new reference
    int32_t layout, const uint8_t* opaque_metadata, int64_t opaque_metadata_size);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // AOTI_TORCH_NPU_SHIM
