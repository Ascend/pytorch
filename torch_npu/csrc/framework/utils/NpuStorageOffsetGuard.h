#ifndef __NPU_STORAGE_GUARD__
#define __NPU_STORAGE_GUARD__
#include <ATen/ATen.h>
#include <stdint.h>

#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu {
namespace native {
class NpuStorageOffsetGuard
{
public:
    NpuStorageOffsetGuard() = delete;
    NpuStorageOffsetGuard(const NpuStorageOffsetGuard &guard) = delete;
    NpuStorageOffsetGuard &operator= (const NpuStorageOffsetGuard &guard) = delete;

    NpuStorageOffsetGuard(NpuStorageOffsetGuard &&guard) = delete;
    NpuStorageOffsetGuard &operator= (NpuStorageOffsetGuard &&guard) = delete;

    explicit NpuStorageOffsetGuard(at::Tensor &tensor) noexcept : guard_(tensor) {
        SetTensorStorageOffset();
    }
    ~NpuStorageOffsetGuard() noexcept {
        RecoverTensorStorageOffset();
    }

private:
    void SetTensorStorageOffset() {
        origin_allow_tensor_metadata_change_ = guard_.unsafeGetTensorImpl()->allow_tensor_metadata_change();
        origin_storage_offset_ = guard_.storage_offset();

        guard_.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
        guard_.unsafeGetTensorImpl()->set_storage_offset(0);
    }
    void RecoverTensorStorageOffset() {
        guard_.unsafeGetTensorImpl()->set_storage_offset(origin_storage_offset_);
        guard_.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(origin_allow_tensor_metadata_change_);
    }
    int64_t origin_storage_offset_ = 0;
    bool origin_allow_tensor_metadata_change_ = true;
    at::Tensor guard_;
};
}  // namespace native
}  // namespace at_npu

#endif // __NPU_STORAGE_GUARD__
