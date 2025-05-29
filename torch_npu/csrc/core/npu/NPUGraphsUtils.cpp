#include "NPUGraphsUtils.h"

namespace c10_npu {
CaptureStatus currentStreamCaptureStatusMayInitCtx()
{
    if (!c10_npu::acl::IsCaptureSupported()) {
        return CaptureStatus::None;
    }

    aclmdlRICaptureStatus is_capturing{ACL_MODEL_RI_CAPTURE_STATUS_NONE};
    aclmdlRI model_ri;
    auto s = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(
        c10_npu::acl::AclmdlRICaptureGetInfo(s.stream(false), &is_capturing, &model_ri));
    return CaptureStatus(is_capturing);
}

} // namespace c10_npu
