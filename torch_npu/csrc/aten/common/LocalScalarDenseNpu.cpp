#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

#define AT_DISPATCH_CASE_ALL_TYPES_AND5(        \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
    AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)       \
    AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)    \
    AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)    \
    AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)    \
    AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)    \
    AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)


#define AT_DISPATCH_ALL_TYPES_AND5(                         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(                                       \
        TYPE,                                                 \
        NAME,                                                 \
        AT_DISPATCH_CASE_ALL_TYPES_AND5(                      \
            SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, __VA_ARGS__))


c10::Scalar NPUNativeFunctions::_local_scalar_dense(const at::Tensor& self)
{
    c10::Scalar r;
    AT_DISPATCH_ALL_TYPES_AND5(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fn,
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
