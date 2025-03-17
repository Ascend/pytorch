#include <ATen/ATen.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "third_party/acl/inc/acl/acl.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::copy_memory_(at::Tensor& self, const at::Tensor& src, bool non_blocking)
{
    c10_npu::NPUGuard guard(src.device());
    AT_ASSERT(torch_npu::utils::is_npu(src), "copy_memory_ only support npu tensor", OPS_ERROR(ErrCode::PARAM));
    AT_ASSERT(
        src.dtype() == self.dtype(),
        "input tensors of copy_memory_ should have same dtype", OPS_ERROR(ErrCode::PARAM));
    AT_ASSERT(
        src.device().index() == self.device().index(),
        "input tensors of copy_memory_ should have same device index", OPS_ERROR(ErrCode::PARAM));
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;

    int dst_size = 0;
    int src_size = 0;

    if (FormatHelper::IsPadded(&self)) {
        AT_ASSERT(self.storage_offset() == 0, OPS_ERROR(ErrCode::VALUE));
        dst_size = c10::multiply_integers(dst_desc.storage_sizes_);
    } else {
        auto dst_element = c10::multiply_integers(self.sizes());
        auto dst_storage = c10::multiply_integers(dst_desc.storage_sizes_);
        dst_size = (dst_element > dst_storage) ? dst_storage : dst_element;
    }

    if (FormatHelper::IsPadded(&src)) {
        AT_ASSERT(src.storage_offset() == 0, OPS_ERROR(ErrCode::VALUE));
        src_size = c10::multiply_integers(src_desc.storage_sizes_);
    } else {
        auto src_element = c10::multiply_integers(src.sizes());
        auto src_storage = c10::multiply_integers(src_desc.storage_sizes_);
        src_size = (src_element > src_storage) ? src_storage : src_element;
    }

    // Designed for the gather of tensors, ignoring npu_format_ and
    // copying continuous memory between npu tensors.
    auto ret = CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
        self,
        dst_size * self.itemsize(),
        src,
        dst_size * self.itemsize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
    NPU_CHECK_ERROR(ret);

    if (!non_blocking) {
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
    return self;
}

} // namespace native
} // namespace at_npu
