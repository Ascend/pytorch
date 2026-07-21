#include "NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUStreamUtils.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

namespace c10_npu {
CaptureId_t captureIdFromModelRI(aclmdlRI modelRI)
{
    uint32_t modelRIId = 0;
    if (c10_npu::acl::TryAclmdlRIGetId(modelRI, &modelRIId)) {
        return static_cast<CaptureId_t>(modelRIId);
    }
    return reinterpret_cast<CaptureId_t>(modelRI);
}

CaptureStatus captureStatusMayInitCtx(aclrtStream stream)
{
    if (!c10_npu::acl::IsCaptureSupported() || !c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        return CaptureStatus::None;
    }

    aclmdlRICaptureStatus is_capturing{ACL_MODEL_RI_CAPTURE_STATUS_NONE};
    aclmdlRI model_ri;
    NPU_CHECK_ERROR(
        c10_npu::acl::AclmdlRICaptureGetInfo(stream, &is_capturing, &model_ri));
    return CaptureStatus(is_capturing);
}

bool isStreamCapturingMayInitCtx(aclrtStream stream)
{
    return captureStatusMayInitCtx(stream) == CaptureStatus::Active;
}

std::optional<CaptureId_t> captureIdMayInitCtx(aclrtStream stream)
{
    if (!c10_npu::acl::IsCaptureSupported() || !c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        return std::nullopt;
    }

    aclmdlRICaptureStatus is_capturing{ACL_MODEL_RI_CAPTURE_STATUS_NONE};
    aclmdlRI model_ri;
    NPU_CHECK_ERROR(
        c10_npu::acl::AclmdlRICaptureGetInfo(stream, &is_capturing, &model_ri));
    if (CaptureStatus(is_capturing) == CaptureStatus::Active) {
        return captureIdFromModelRI(model_ri);
    }
    return std::nullopt;
}

CaptureStatus currentStreamCaptureStatusMayInitCtx()
{
    if (c10_npu::detail::isCurrentStreamExternal()) {
        return CaptureStatus::None;
    }

    auto s = c10_npu::getCurrentNPUStream();
    return captureStatusMayInitCtx(s.stream(false));
}

std::optional<CaptureId_t> currentStreamCaptureIdMayInitCtx()
{
    if (c10_npu::detail::isCurrentStreamExternal()) {
        return std::nullopt;
    }

    auto s = c10_npu::getCurrentNPUStream();
    return captureIdMayInitCtx(s.stream(false));
}

} // namespace c10_npu
