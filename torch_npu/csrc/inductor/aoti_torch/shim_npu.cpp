#include <ATen/TensorUtils.h>
#include <ATen/ops/empty_strided.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/inductor/aoti_torch/c/shim.h>
#include <torch_npu/csrc/inductor/aoti_torch/utils.h>
#include <torch_npu/csrc/inductor/inductor_ops.h>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif
int32_t aoti_torch_device_type_npu() { return (int32_t)c10::DeviceType::PrivateUse1; }

#ifdef __cplusplus
} // extern "C"
#endif

namespace c10_npu {
NPUStream getNPUStreamFromManagedAclrtStream(aclrtStream stream, c10::DeviceIndex device_index);
}

namespace {
static c10::Device c10_device(int32_t device_type, int32_t device_index)
{
    if (device_type == aoti_torch_device_type_cpu()) {
        return c10::Device(static_cast<c10::DeviceType>(device_type));
    } else {
        return c10::Device(static_cast<c10::DeviceType>(device_type), static_cast<c10::DeviceIndex>(device_index));
    }
}
} // namespace

AOTITorchError aoti_torch_create_tensor_from_blob_npu(void* data, int64_t ndim, const int64_t* sizes_ptr,
                                                      const int64_t* strides_ptr, int64_t storage_offset, int32_t dtype,
                                                      int32_t device_type, int32_t device_index,
                                                      AtenTensorHandle* ret_new_tensor)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        TORCH_CHECK(ret_new_tensor != nullptr, "ret_new_tensor is nullptr");
        *ret_new_tensor = nullptr;
        c10::IntArrayRef sizes(sizes_ptr, ndim);
        c10::IntArrayRef strides(strides_ptr, ndim);
        c10::Device device = c10_device(device_type, device_index);
        c10::TensorOptions options = c10::TensorOptions().device(device).dtype(static_cast<c10::ScalarType>(dtype));
        *ret_new_tensor = torch::aot_inductor::new_tensor_handle(
            // data == nullptr can happen for a 0-size tensor
            (data != nullptr) ? at_npu::native::from_blob(data, sizes, strides, storage_offset, options, device)
                              : at::empty_strided(sizes, strides, options));
    });
}

AOTITorchError aoti_torch_create_tensor_from_blob_npu_v2(void* data, int64_t ndim, const int64_t* sizes_ptr,
                                                         const int64_t* strides_ptr, int64_t storage_offset,
                                                         int32_t dtype, int32_t device_type, int32_t device_index,
                                                         AtenTensorHandle* ret_new_tensor, int32_t layout,
                                                         const uint8_t* opaque_metadata, int64_t opaque_metadata_size)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        TORCH_CHECK(ret_new_tensor != nullptr, "ret_new_tensor is nullptr");
        *ret_new_tensor = nullptr;
        if (layout == static_cast<int32_t>(at::kMkldnn)) {
            TORCH_CHECK(false, "do not support mkldnn on npu.");
        } else {
            auto err = aoti_torch_create_tensor_from_blob_npu(data, ndim, sizes_ptr, strides_ptr, storage_offset,
                                                              dtype, device_type, device_index, ret_new_tensor);
            if (err != AOTI_TORCH_SUCCESS) {
                return err;
            }
        }
    });
}

AOTITorchError aoti_torch_create_npu_guard(int32_t device_index, NPUGuardHandle* ret_guard)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        TORCH_CHECK(ret_guard != nullptr, "ret_guard is nullptr");
        *ret_guard = nullptr;
        c10_npu::NPUGuard* guard = new c10_npu::NPUGuard(static_cast<c10::DeviceIndex>(device_index));
        *ret_guard = reinterpret_cast<NPUGuardHandle>(guard);
    });
}

AOTITorchError aoti_torch_delete_npu_guard(NPUGuardHandle guard)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ delete reinterpret_cast<c10_npu::NPUGuard*>(guard); });
}

AOTITorchError aoti_torch_npu_guard_set_index(NPUGuardHandle guard, int32_t device_index)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        reinterpret_cast<c10_npu::NPUGuard*>(guard)->set_index(static_cast<c10::DeviceIndex>(device_index));
    });
}

AOTITorchError aoti_torch_create_npu_stream_guard(void* stream, int32_t device_index, NPUStreamGuardHandle* ret_guard)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        TORCH_CHECK(ret_guard != nullptr, "ret_guard is nullptr");
        *ret_guard = nullptr;

        auto raw_stream = static_cast<aclrtStream>(stream);
        auto managed_stream = c10_npu::getNPUStreamFromManagedAclrtStream(
            raw_stream, static_cast<c10::DeviceIndex>(device_index));
        auto* guard = new c10_npu::NPUStreamGuard(static_cast<c10::Stream>(managed_stream));
        *ret_guard = reinterpret_cast<NPUStreamGuardHandle>(guard);
    });
}

AOTITorchError aoti_torch_delete_npu_stream_guard(NPUStreamGuardHandle guard)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ delete reinterpret_cast<c10_npu::NPUStreamGuard*>(guard); });
}

AOTITorchError aoti_torch_get_current_npu_stream(int32_t device_index, void** ret_stream)
{
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        TORCH_CHECK(ret_stream != nullptr, "ret_stream is nullptr");
        *ret_stream = reinterpret_cast<void*>(
            c10_npu::getCurrentNPUStream(static_cast<c10::DeviceIndex>(device_index)).stream());
    });
}
