#include <cstdint>
#include <ATen/TensorUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/core/DeviceType.h>
#include <ATen/ops/empty_strided.h>

#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/inductor/aoti_torch/c/shim_npu.h>
#include <torch_npu/csrc/inductor/aoti_torch/utils.h>
#include <torch_npu/csrc/inductor/inductor_ops.h>

#ifdef __cplusplus
extern "C" {
#endif
int32_t aoti_torch_device_type_npu() {
    return (int32_t)c10::DeviceType::PrivateUse1;
}

#ifdef __cplusplus
} // extern "C"
#endif

namespace {
    static c10::Device c10_device(int32_t device_type, int32_t device_index) {
        if (device_type == aoti_torch_device_type_cpu()) {
            return c10::Device(static_cast<c10::DeviceType>(device_type));
        } else {
            return c10::Device(
                static_cast<c10::DeviceType>(device_type),
                static_cast<c10::DeviceIndex>(device_index));
        }
    }
} // namespace

#ifdef USE_NPU
AOTITorchError aoti_torch_create_npu_guard(int32_t device_index, NPUGuardHandle* ret_guard)
{
    // todo: implement create npu guard logic
    return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_delete_npu_guard(NPUGuardHandle guard)
{
    // todo: implement delete npu guard logic
    return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_npu_guard_set_index(NPUGuardHandle guard, int32_t device_index)
{
    // todo: implement npu guard set index logic
    return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_create_npu_stream_guard(
    void* stream,
    int32_t device_index,
    NPUStreamGuardHandle* ret_guard)
{
    // todo: implement create npu stream guard logic
    return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_delete_npu_stream_guard(NPUStreamGuardHandle guard)
{
    // todo: implement delete npu stream guard logic
    return AOTI_TORCH_SUCCESS;
}
#endif // USE_NPU

AOTITorchError aoti_torch_create_tensor_from_blob_npu(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    c10::Device device = c10_device(device_type, device_index);
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(dtype));
    *ret_new_tensor = torch::aot_inductor::new_tensor_handle(
        // data == nullptr can happen for a 0-size tensor
        (data != nullptr) ? at_npu::native::from_blob(data, sizes, strides, storage_offset, options, device)
                          : at::empty_strided(sizes, strides, options));
  });
}

AOTITorchError aoti_torch_create_tensor_from_blob_npu_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        if (layout == static_cast<int32_t>(at::kMkldnn)) {
            throw std::runtime_error("do not support mkldnn on npu.");
        } else {
            aoti_torch_create_tensor_from_blob_npu(
                data,
                ndim,
                sizes_ptr,
                strides_ptr,
                storage_offset,
                dtype,
                device_type,
                device_index,
                ret_new_tensor);
        }
    });
}