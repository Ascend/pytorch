#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::Scalar NPUNativeFunctions::_local_scalar_dense(const at::Tensor& self)
{
    c10::Scalar r;
    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "_local_scalar_dense_npu",
        [&] {
            scalar_t value = 0;
            c10_npu::NPUStream copy_stream = c10_npu::getCurrentNPUStream();
            // Synchronous copy after stream synchronization
            NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream));

            NPU_CHECK_ERROR(CalcuOpUtil::AclrtMemcpyWithModeSwitch(
                &value,
                sizeof(scalar_t),
                std::make_pair(self.storage().unsafeGetStorageImpl(), self.storage_offset() * self.itemsize()),
                sizeof(scalar_t),
                ACL_MEMCPY_DEVICE_TO_HOST));
            r = c10::Scalar(value);
        });
    return r;
}

} // namespace native
} // namespace at_npu
