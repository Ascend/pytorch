#pragma once

#include <iostream>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace c10_npu {

static std::atomic<bool> is_stream_capturing(false);

using CaptureId_t = unsigned long long;

// first is set if the instance is created by NPUGraph::capture_begin.
// second is set if the instance is created by at::cuda::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

// RAII guard for "aclmdlCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct C10_NPU_API NPUStreamCaptureModeGuard{
    NPUStreamCaptureModeGuard(aclmdlCaptureMode desired)
    : strictness_(desired) {}
    ~NPUStreamCaptureModeGuard() {}

    private:
    aclmdlCaptureMode strictness_;
};

// Protects against enum aclmdlCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(
    int(aclmdlCaptureStatus::ACL_MODEL_CAPTURE_STATUS_NONE) == 0,
    "unexpected int(ACL_MODEL_CAPTURE_STATUS_NONE) value");
static_assert(
    int(aclmdlCaptureStatus::ACL_MODEL_CAPTURE_STATUS_ACTIVE) == 1,
    "unexpected int(ACL_MODEL_CAPTURE_STATUS_ACTIVE) value");
static_assert(
    int(aclmdlCaptureStatus::ACL_MODEL_CAPTURE_STATUS_INVALIDATED) == 2,
    "unexpected int(ACL_MODEL_CAPTURE_STATUS_INVALIDATED) value");

enum class CaptureStatus : int {
    None = int(aclmdlCaptureStatus::ACL_MODEL_CAPTURE_STATUS_NONE),
    Active = int(aclmdlCaptureStatus::ACL_MODEL_CAPTURE_STATUS_ACTIVE),
    Invalidated = int(aclmdlCaptureStatus::ACL_MODEL_CAPTURE_STATUS_INVALIDATED)
};

inline std::ostream &operator<<(std::ostream &os, CaptureStatus status)
{
    switch (status) {
        case CaptureStatus::None:
            os << "npuStreamCaptureStatusNone";
            break;
        case CaptureStatus::Active:
            os << "npuStreamCaptureStatusActive";
            break;
        case CaptureStatus::Invalidated:
            os << "npuStreamCaptureStatusInvalidated";
            break;
        default:
            TORCH_INTERNAL_ASSERT(
                false, "Unknown NPU graph CaptureStatus", int(status));
    }
    return os;
}

// Use this version where you're sure a CUDA context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx()
{
    if (!c10_npu::acl::IsCaptureSupported()) {
        return CaptureStatus::None;
    }

    aclmdlCaptureStatus is_capturing{ACL_MODEL_CAPTURE_STATUS_NONE};
    uint32_t modelId;
    NPU_CHECK_ERROR(
        c10_npu::acl::AclmdlCaptureGetInfo(c10_npu::getCurrentNPUStream(), &is_capturing, &modelId));
    return CaptureStatus(is_capturing);
}

// Use this version where you don't want to create a CUDA context if none exists.
inline CaptureStatus currentStreamCaptureStatus()
{
    // don't create a context if we don't have to
    if (c10_npu::IsContextInitialized()) {
        return currentStreamCaptureStatusMayInitCtx();
    } else {
        return CaptureStatus::None;
    }
}

inline void assertNotCapturing(const std::string &attempt)
{
    auto status = currentStreamCaptureStatus();
    TORCH_CHECK(status == CaptureStatus::None,
                attempt,
                " during NPU graph capture. If you need this call to be captured, "
                "please file an issue. "
                "Current npuStreamCaptureStatus: ",
                status);
}

} // namespace c10_npu
