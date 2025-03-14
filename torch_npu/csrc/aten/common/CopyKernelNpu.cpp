#include <c10/core/Scalar.h>
#include <ATen/record_function.h>

#include "op_plugin/OpInterface.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

// the dst and src are same dtype
// the dst and src have same elemsize
// if exceptCopySize is not defined, we will copy dst storage size
// so: caller should make sure that the storage size of src and dst are reasonable.
void copy_d2d_by_memcpy(at::Tensor& dst, const at::Tensor& src, int64_t exceptSize)
{
    c10_npu::NPUGuard guard(src.device());
    int64_t size = exceptSize;
    auto dst_mem_size = StorageDescHelper::GetMemorySize(dst);
    if (exceptSize == 0) {
        size = dst_mem_size;
    }

    if (!dst.data_ptr()) {
        TORCH_NPU_WARN("copy_d2d_by_memcpy, dst.data_ptr() is null.");
        return;
    }

    if (!src.data_ptr()) {
        TORCH_NPU_WARN("copy_d2d_by_memcpy, src.data_ptr() is null.");
        return;
    }

    if (dst.data_ptr() == src.data_ptr() && dst.element_size() == src.element_size()) {
        return;
    }

    // The current logic is only used in single op mode.
    NPU_CHECK_ERROR(c10_npu::queue::LaunchAsyncCopyTask(
        dst.data_ptr(),
        size * dst.element_size(),
        src.data_ptr(),
        size * dst.element_size(),
        ACL_MEMCPY_DEVICE_TO_DEVICE));
}

} // namespace native
} // namespace at_npu
